"""The view objects of Rok's editing layer, on top of Rok's camera.

`Region` is the view's size in logical pixels. `ViewState` reads and writes
the camera. Region pixels have their origin at the bottom left; the host's
pixels have it at the top left, and `to_host_y` converts. The functions at
the bottom turn region pixels into rays and points and back.
"""

from __future__ import annotations

import math

import numpy as np

from .camera import Camera


class Region:
    """The window's drawing area. In the quad view it holds all four views;
    each view's ViewState knows its rectangle inside it."""

    type = "WINDOW"
    x = 0
    y = 0
    alignment = "NONE"

    def __init__(self, camera: Camera, app=None) -> None:
        self._camera = camera
        self._app = app

    @property
    def width(self) -> int:
        if self._app is not None and getattr(self._app, "width", None):
            return int(round(self._app.width))
        return int(round(self._camera.width))

    @property
    def height(self) -> int:
        if self._app is not None and getattr(self._app, "height", None):
            return int(round(self._app.height))
        return int(round(self._camera.height))

    def as_pointer(self) -> int:
        return 1  # Rok has one view

    def tag_redraw(self) -> None:
        if self._app is not None:
            self._app.request_redraw()


class ViewState:
    """The camera as the editing layer sees it: in perspective, or orthographic
    when the camera is (a side view of the quad view)."""

    is_orthographic_side_view = False

    def __init__(self, camera: Camera, app=None, name: str = "Perspective") -> None:
        self._camera = camera
        self._app = app
        self.name = name
        # The view's rectangle in region pixels (origin bottom left); None: the whole region.
        self.rect: tuple[float, float, float, float] | None = None

    @property
    def is_perspective(self) -> bool:
        return not self._camera.ortho

    @property
    def view_perspective(self) -> str:
        return "ORTHO" if self._camera.ortho else "PERSP"

    def area(self, region) -> tuple[float, float, float, float]:
        """(x, y, w, h) of this view in region pixels, origin bottom left."""
        return self.rect if self.rect is not None else (0.0, 0.0, float(region.width), float(region.height))

    def to_camera(self, region, x: float, y: float) -> tuple[float, float]:
        """Region pixels to this camera's pixels (origin top left of the view)."""
        vx, vy, _vw, vh = self.area(region)
        return float(x) - vx, (vy + vh) - float(y)

    def from_camera(self, region, x: float, y: float) -> tuple[float, float]:
        vx, vy, _vw, vh = self.area(region)
        return vx + float(x), (vy + vh) - float(y)

    def contains(self, region, x: float, y: float) -> bool:
        vx, vy, vw, vh = self.area(region)
        return vx <= x < vx + vw and vy <= y < vy + vh

    @property
    def camera(self) -> Camera:
        return self._camera

    def _moved(self) -> None:
        if self._app is not None:
            self._app.camera_moved()

    @property
    def orbit_centre(self) -> np.ndarray:
        return self._camera.target.copy()

    @orbit_centre.setter
    def orbit_centre(self, value) -> None:
        self._camera.target = np.asarray(value, dtype=np.float64).copy()
        self._moved()

    @property
    def orbit_distance(self) -> float:
        return float(self._camera.distance)

    @orbit_distance.setter
    def orbit_distance(self, value: float) -> None:
        self._camera.distance = float(value)
        self._moved()

    @property
    def view_matrix(self) -> np.ndarray:
        return self._camera.view()

    @property
    def screen_matrix(self) -> np.ndarray:
        """Projection times view, mapped into this view's part of the region:
        code that projects with it and the region's size lands on the right
        pixels in every view of the quad view."""
        vp = self._camera.view_proj()
        if self.rect is None or self._app is None:
            return vp
        W, H = float(self._app.width), float(self._app.height)
        x, y, w, h = self.rect
        fit = np.eye(4)
        fit[0, 0], fit[0, 3] = w / W, (2.0 * x + w) / W - 1.0
        fit[1, 1], fit[1, 3] = h / H, (2.0 * y + h) / H - 1.0
        return fit @ vp

    @property
    def window_matrix(self) -> np.ndarray:
        return self._camera.projection()

    def axes_matrix(self) -> np.ndarray:
        """3 x 3, columns = the view's right, up and backward axes in world space
        (the camera looks down its -Z)."""
        v = self._camera.view()
        return v[:3, :3].T.copy()

    def view_direction(self) -> np.ndarray:
        """Unit vector the camera looks along."""
        d = self._camera.target - self._camera.eye()
        return d / np.linalg.norm(d)

    def set_view_direction(self, forward) -> None:
        """Turn the camera to look along `forward` (keeps target and distance)."""
        f = np.asarray(forward, dtype=np.float64)
        f = f / np.linalg.norm(f)
        back = -f
        self._camera.pitch = math.asin(max(-1.0, min(1.0, back[2])))
        self._camera.yaw = math.atan2(back[1], back[0])
        self._moved()


def to_host_y(region: Region, y: float) -> float:
    return region.height - float(y)


def ray_origin(region: Region, view_state: ViewState, coord) -> np.ndarray:
    o, _d = view_state.camera.ray(*view_state.to_camera(region, coord[0], coord[1]))
    return o


def ray_direction(region: Region, view_state: ViewState, coord) -> np.ndarray:
    _o, d = view_state.camera.ray(*view_state.to_camera(region, coord[0], coord[1]))
    return d


def world_to_region(region: Region, view_state: ViewState, coord, default=None):
    x, y = view_state.camera.project([coord])[0]
    if not np.isfinite(x) or not np.isfinite(y):
        return default
    return np.array(view_state.from_camera(region, x, y))
