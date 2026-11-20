"""The orbit camera. Python owns it; Rust only draws with its matrices.

World space has Z up and one unit per voxel. The camera circles `target` at
`distance`, turned by `yaw` about Z and raised by `pitch`. The matrices follow
the wgpu convention the ray march expects: right-handed, depth 0..1. Screen
positions are logical pixels with the origin at the top left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rok.core import view_keys

FOV_Y_DEG = 35.0
VIEW_LIFT = 60.0  # the main view's lift (Camera.lift), in logical pixels
NEAR = 0.5
ORBIT_RAD_PER_PX = 0.005  # two-finger scroll and Ctrl-drag
ZOOM_PER_PX = 0.01  # Option + two-finger scroll: the distance changes by e^(0.01 * px)
MAX_PITCH = math.radians(view_keys.MAX_ELEVATION_DEG)


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    if abs(float(np.dot(f, up))) > 0.9999:
        # Straight down or up: the screen's up is the horizontal way the camera
        # came from, turned round (a Top view has +Y up when it was looking from -Y).
        up = (0.0, 1.0, 0.0) if abs(f[1]) < 0.9 else (1.0, 0.0, 0.0)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


def orthographic(half_h: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed, depth 0 at `near` and 1 at `far`; `half_h` world units
    from the middle to the top edge."""
    m = np.eye(4)
    m[0, 0] = 1.0 / (half_h * aspect)
    m[1, 1] = 1.0 / half_h
    m[2, 2] = 1.0 / (near - far)
    m[2, 3] = near / (near - far)
    return m


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed, depth 0 at `near` and 1 at `far`."""
    h = 1.0 / math.tan(fov_y / 2.0)
    r = far / (near - far)
    m = np.zeros((4, 4))
    m[0, 0] = h / aspect
    m[1, 1] = h
    m[2, 2] = r
    m[2, 3] = r * near
    m[3, 2] = -1.0
    return m


@dataclass
class Camera:
    target: np.ndarray
    yaw: float = -0.9
    pitch: float = 0.55
    distance: float = 100.0
    width: float = 1440.0  # the view in logical pixels
    height: float = 900.0
    ortho: bool = False  # an orthographic view (the quad view's Top, Front and Right)
    locked: bool = False  # orbit does nothing (the quad view's side views)
    up_hint: tuple | None = None  # the screen's up for a view straight down or up
    near_clip: float | None = None  # a near plane this far from the eye cuts away what is in front (close-ups)
    fov_deg: float | None = None  # the vertical field of view in degrees; None: FOV_Y_DEG (the editor's)
    # How far above the view's middle the target shows, in logical pixels: the
    # main view lifts its model clear of the palette card and the hint line at
    # the bottom (a lens shift, so orbit still turns about the target).
    lift: float = 0.0

    @classmethod
    def default(cls) -> Camera:
        return cls(target=np.zeros(3))

    @property
    def fov_y(self) -> float:
        return math.radians(FOV_Y_DEG if self.fov_deg is None else self.fov_deg)

    @property
    def aspect(self) -> float:
        return max(self.width, 1.0) / max(self.height, 1.0)

    def eye(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return self.target + self.distance * np.array([cp * cy, cp * sy, sp])

    def _key(self) -> tuple:
        return (float(self.target[0]), float(self.target[1]), float(self.target[2]), self.yaw, self.pitch, self.distance,
                self.width, self.height, self.ortho, self.up_hint, self.near_clip, self.fov_deg, self.lift)

    def _cached(self, name: str, make):
        """A matrix kept until one of the camera's parameters changes: a frame asks
        for the view and the projection many times (every ray, every projected point)."""
        key = self._key()
        cache = self.__dict__.get("_matrices")
        if cache is None or cache[0] != key:
            cache = (key, {})
            self.__dict__["_matrices"] = cache
        found = cache[1].get(name)
        if found is None:
            found = cache[1][name] = make()
        return found

    def view(self) -> np.ndarray:
        return self._cached("view", self._view)

    def _view(self) -> np.ndarray:
        eye = self.eye()
        f = self.target - eye
        if self.up_hint is not None and abs(float(f[2])) > 0.9999 * float(np.linalg.norm(f)):
            return look_at(eye, self.target, self.up_hint)
        return look_at(eye, self.target)

    @property
    def half_height(self) -> float:
        """An orthographic view's half height in world units: as much as the
        perspective view shows at the target."""
        return self.distance * math.tan(self.fov_y / 2.0)

    @property
    def pixel(self) -> float:
        """World units per logical pixel of an orthographic view; 0 in perspective."""
        return 2.0 * self.half_height / max(self.height, 1.0) if self.ortho else 0.0

    def projection(self) -> np.ndarray:
        return self._cached("projection", self._projection)

    def _projection(self) -> np.ndarray:
        far = max(self.distance * 40.0, 4000.0)
        if self.ortho:
            # The near plane far behind the eye: the box is never cut (unless a close-up asks).
            near = -far if self.near_clip is None else self.near_clip
            m = orthographic(self.half_height, self.aspect, near, far)
        else:
            m = perspective(self.fov_y, self.aspect, max(NEAR, self.near_clip or NEAR), far)
        if self.lift:
            m = m.copy()
            m[1] += (2.0 * self.lift / max(self.height, 1.0)) * m[3]  # every point up by `lift` pixels
        return m

    def view_proj(self) -> np.ndarray:
        return self._cached("view_proj", lambda: self.projection() @ self.view())

    def inverse_view_proj(self) -> np.ndarray:
        return self._cached("inverse", lambda: np.linalg.inv(self.view_proj()))

    # --- screen and world -----------------------------------------------------

    def ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """World origin and unit direction of the ray through a screen point."""
        inv = self.inverse_view_proj()
        nx, ny = x / self.width * 2.0 - 1.0, 1.0 - y / self.height * 2.0
        far = inv @ np.array([nx, ny, 1.0, 1.0])
        far = far[:3] / far[3]
        if self.ortho:
            near = inv @ np.array([nx, ny, 0.0, 1.0])
            near = near[:3] / near[3]
            d = far - near
            return near, d / np.linalg.norm(d)
        eye = self.eye()
        d = far - eye
        return eye, d / np.linalg.norm(d)

    def project(self, points) -> np.ndarray:
        """(N, 3) world points -> (N, 2) screen points. Points behind the camera
        come out as NaN."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        h = np.c_[p, np.ones(len(p))] @ self.view_proj().T
        w = h[:, 3:4]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndc = np.where(w > 1e-9, h[:, :2] / w, np.nan)
        return np.c_[(ndc[:, 0] + 1.0) * 0.5 * self.width, (1.0 - ndc[:, 1]) * 0.5 * self.height]

    # --- moves ----------------------------------------------------------------

    def orbit(self, dx_px: float, dy_px: float) -> None:
        if self.locked:
            return
        self.yaw -= dx_px * ORBIT_RAD_PER_PX
        self.pitch = min(MAX_PITCH, max(-MAX_PITCH, self.pitch + dy_px * ORBIT_RAD_PER_PX))

    def turn_degrees(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
        if self.locked:
            return
        self.yaw += math.radians(yaw_deg)
        pitch = math.degrees(self.pitch)
        self.pitch = math.radians(pitch + view_keys.clamp_tilt(pitch, pitch_deg))

    def zoom(self, factor: float) -> None:
        """factor > 1 moves closer."""
        self.distance = view_keys.clamp_distance(self.distance / factor)

    def pan(self, dx_px: float, dy_px: float) -> None:
        """Move the target so the scene follows the fingers."""
        v = self.view()
        right, up = v[0, :3], v[1, :3]
        per_px = 2.0 * self.distance * math.tan(self.fov_y / 2.0) / max(self.height, 1.0)
        self.target = self.target - right * dx_px * per_px + up * dy_px * per_px

    def fit(self, lo, hi, margin: float = 1.12) -> None:
        """Look at the box lo..hi from the current angle, just far enough back that
        all eight corners show, with `margin` to spare."""
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        self.target = (lo + hi) * 0.5
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        if self.ortho:
            v = self.view()
            centred = corners - self.target
            half_w = float(np.max(np.abs(centred @ v[0, :3])))
            half_h = float(np.max(np.abs(centred @ v[1, :3])))
            room = max(1.0 - 2.0 * abs(self.lift) / max(self.height, 1.0), 0.1)  # the smaller half, lifted
            half = max(half_h / room, half_w / self.aspect, 0.5) * margin
            self.distance = view_keys.clamp_distance(half / math.tan(self.fov_y / 2.0))
            return
        self.frame(lo, hi)  # the box's sphere fits: every corner is in front of the eye
        # The target shows `lift` pixels above the middle, so the room below it is the smaller half.
        cy, half_h = self.height / 2.0 - self.lift, max(self.height / 2.0 - abs(self.lift), 1.0)
        for _ in range(4):
            p = self.project(corners)
            fx = np.max(np.abs(p[:, 0] - self.width / 2.0)) / (self.width / 2.0)
            fy = np.max(np.abs(p[:, 1] - cy)) / half_h
            if not np.isfinite(fx) or not np.isfinite(fy):
                break
            self.distance = view_keys.clamp_distance(self.distance * max(fx, fy) * margin)

    def frame(self, lo, hi) -> None:
        """Look at the box lo..hi so all of it shows."""
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        self.target = (lo + hi) * 0.5
        radius = max(float(np.linalg.norm(hi - lo)) * 0.5, 1.0)
        half = min(self.fov_y, 2.0 * math.atan(math.tan(self.fov_y / 2.0) * self.aspect)) / 2.0
        self.distance = view_keys.clamp_distance(radius / math.sin(half) * 1.05)
