"""Rok without a window: a Session for scripts, batch jobs and the MCP server.

    from rok.headless import Session
    s = Session("ship.vox")
    s.run("rok.mirror_half", axis="X", from_positive=True)
    png = s.render_png("quad", size=(1000, 1000))
    s.save("ship.vox")

A Session is the whole app (`RokApp`) on a host with no window
(`Host.start_headless`). The GPU renders off-screen, so pictures look as they
do in the editor. Python drives: after each call the session hands the queued
commands to the host and delivers the events they cause.

One session per process: the app keeps its document in module state.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

# A session keeps its recent files and settings apart from the editor's, and
# shows no welcome screen. Both are read when the app modules load.
os.environ.setdefault("ROK_CONFIG", tempfile.mkdtemp(prefix="rok-headless-"))
os.environ.setdefault("ROK_NO_WELCOME", "1")

from rok.core import png as png_core  # noqa: E402

# The named views: (yaw, pitch, orthographic). Yaw and pitch as in
# rok.app.camera.Camera: the eye sits at yaw around Z, pitch above the floor.
VIEWS: dict[str, tuple[float, float, bool]] = {
    "top": (-math.pi / 2, math.pi / 2, True),
    "bottom": (-math.pi / 2, -math.pi / 2, True),
    "front": (-math.pi / 2, 0.0, True),
    "back": (math.pi / 2, 0.0, True),
    "right": (0.0, 0.0, True),
    "left": (math.pi, 0.0, True),
    "iso": (-math.pi / 4, math.radians(35.0), False),
}
# The quad image: the editor's quad view order, top left first.
QUAD = ("top", "iso", "front", "right")
MARGIN = 1.08  # room around the model in a framed view


class SessionError(RuntimeError):
    """An operator that does not exist, or a call the app refused."""


@dataclass
class Result:
    """What one `run` did: the operator's result set and what it reported."""

    status: set
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return "FINISHED" in self.status


_session: Session | None = None


class Session:
    def __init__(self, path: str | os.PathLike | None = None, width: float = 1000.0, height: float = 1000.0) -> None:
        global _session
        if _session is not None:
            raise SessionError("one Session per process: the app keeps its document in module state")
        import rok_native

        from rok.app.app import RokApp

        _session = self
        self.host = rok_native.Host("Rok (headless)", None)
        self.app = RokApp(self.host, str(path) if path else None)
        self.size = (float(width), float(height))
        self._shots: dict[str, tuple[int, int, bytes]] = {}
        self._token = 0
        self._messages: list[str] = []
        self._agent = None
        say = self.app.say

        def capture(message: str, seconds: float = 3.0) -> None:
            self._messages.append(str(message))
            say(message, seconds)

        self.app.say = capture
        self.host.start_headless(self._on_event, self.size[0], self.size[1], 1.0)
        self.process()

    def close(self) -> None:
        """End the session, so this process can start another."""
        global _session
        if _session is self:
            _session = None

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- the loop ---------------------------------------------------------------

    def _on_event(self, kind: str, data: dict) -> None:
        if kind == "screenshot" and not str(data["token"]).startswith(("picture:", "anim:")):
            self._shots[data["token"]] = (int(data["width"]), int(data["height"]), data["rgba"])
            return
        self.app.handle(kind, data)

    def process(self) -> None:
        """Hand the queued commands to the host and deliver what they cause."""
        self.host.process()

    # --- commands ---------------------------------------------------------------

    def run(self, idname: str, **props) -> Result:
        """Run an operator by id (`rok.mirror_half`), as a menu would. Every
        operator that changes the document is one undo step."""
        from rok.app import opkit

        if idname not in opkit.REGISTRY:
            raise SessionError(f"no operator {idname!r}; Session.commands() lists them")
        self._messages = []
        try:
            status = opkit.call(idname, "EXEC", **props)
        finally:
            self.app.after_event()
            self.process()
        return Result(set(status or ()), list(self._messages))

    @staticmethod
    def commands() -> dict[str, str]:
        """Every operator id with its description."""
        from rok.app import opkit

        return {k: (getattr(cls, "tooltip", "") or getattr(cls, "title", "")) for k, cls in sorted(opkit.REGISTRY.items())}

    def open(self, path: str | os.PathLike) -> Result:
        return self.run("rok.open", filepath=str(path))

    def save(self, path: str | os.PathLike) -> Result:
        return self.run("rok.save_as", filepath=str(path))

    @property
    def document(self):
        from rok.app.object import document

        return document

    # --- pictures ---------------------------------------------------------------

    def bounds(self, fit: str = "grid") -> tuple[np.ndarray, np.ndarray]:
        """The world box around every shown model's grid (fit="grid") or around
        its filled voxels (fit="voxels"): (low, high)."""
        from rok.app.object import model_origin

        scene = self.document.scene
        boxes = []
        for model in scene.models if scene is not None else ():
            if scene.is_hidden(model):
                continue
            lo = np.asarray(model_origin(model), dtype=np.float64)
            if fit == "voxels":
                filled = np.argwhere(model.grid.voxels != 0)
                if len(filled) == 0:
                    continue
                boxes.append((lo + filled.min(axis=0), lo + filled.max(axis=0) + 1))
            else:
                boxes.append((lo, lo + np.asarray(model.grid.size, dtype=np.float64)))
        if not boxes:
            return np.zeros(3), np.ones(3)
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)

    def camera(self, view: str, width: float, height: float, lo=None, hi=None):
        """A camera for a named view, framed on the box lo..hi (default: every model)."""
        if lo is None or hi is None:
            lo, hi = self.bounds()
        return camera_for(view, width, height, lo, hi)

    def render(self, views="iso", size=(1000, 1000), ui: bool = False, look: str | None = None,
               fit: str = "grid") -> np.ndarray:
        """An (h, w, 4) uint8 RGBA picture. `views` is a view name, "quad" (top,
        iso, front and right in a 2 x 2 grid) or a list of up to four names in
        a 2 x 2 grid. `ui` keeps the editor's panels; without it only the scene
        shows."""
        if views == "quad":
            views = QUAD
        names = [views] if isinstance(views, str) else list(views)
        if not 1 <= len(names) <= 4:
            raise SessionError("render takes one to four views")
        self._token += 1
        token = f"render-{self._token}"
        if ui:
            self.host.screenshot(token)
        else:
            w, h = int(size[0]), int(size[1])
            rows = []
            lo, hi = self.bounds(fit)
            for name, (x, y, tw, th) in zip(names, tiles(len(names), w, h)):
                cam = self.camera(name, tw, th, lo, hi)
                rows.append((x, y, tw, th, cam.view_proj(), tuple(float(c) for c in cam.eye()), float(cam.pixel), 0.0, 0.0))
            if look is not None:
                from rok.app.looks import LOOKS

                self.host.set_look(**LOOKS[look].host_args())
            self.host.render_views(token, w, h, rows, 0.0, look is not None)
        self.process()
        if look is not None:
            from rok.app.looks import DEFAULT, LOOKS

            self.host.set_look(**LOOKS[DEFAULT].host_args())
            self.process()
        shot = self._shots.pop(token, None)
        if shot is None:
            raise SessionError("the host sent no picture")
        sw, sh, rgba = shot
        return np.frombuffer(rgba, dtype=np.uint8).reshape(sh, sw, 4).copy()

    def render_png(self, views="iso", size=(1000, 1000), ui: bool = False, look: str | None = None,
                   fit: str = "grid") -> bytes:
        return png_core.encode(self.render(views, size, ui, look, fit))

    # --- the agent tools -----------------------------------------------------------

    @property
    def agent(self):
        if self._agent is None:
            from rok.agent import Agent

            self._agent = Agent(self.app, self.host)
        return self._agent

    def tool(self, name: str, args: dict | None = None):
        """Run one agent tool (rok/agent.py). Returns (Reply, PNG bytes or None)."""
        reply = self.agent.call(name, args or {})
        self._finish_exports()
        png = reply.image  # flat views come drawn
        if reply.picture is not None:
            self._token += 1
            token = f"agent-{self._token}"
            tokens = self.agent.picture_tokens(reply.picture, token)
            self.agent.queue_picture(reply.picture, token)
            self.process()
            shots = [self._shots.pop(t, None) for t in tokens]
            if all(shot is not None for shot in shots):
                png = png_core.encode(self.agent.assemble(reply.picture, shots))
                self.agent.finish_picture(reply.picture, png)
                if not reply.picture.send:
                    png = None
        return reply, png

    def _finish_exports(self, seconds: float = 120.0) -> None:
        """Deliver the pictures of a running sprite sheet, GIF or MP4 export
        until it has written its files: a window does it frame by frame."""
        import time

        from rok.app import anim_pictures

        end = time.monotonic() + seconds
        while anim_pictures.running() and time.monotonic() < end:
            self.process()


def tiles(n: int, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """Rectangles (logical pixels, origin top left) for n views: one to three
    side by side, four in a 2 x 2 grid. No part of the picture stays empty."""
    if n <= 3:
        cw = w / n
        return [(k * cw, 0.0, cw, h) for k in range(n)]
    hw, hh = w / 2.0, h / 2.0
    return [(0.0, 0.0, hw, hh), (hw, 0.0, hw, hh), (0.0, hh, hw, hh), (hw, hh, hw, hh)]


def depth_range(cam, lo, hi) -> tuple[float, float]:
    """How far the box lo..hi lies along the camera's view: (near, length)."""
    eye = cam.eye()
    fwd = cam.target - eye
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=np.float64)
    d = (corners - eye) @ fwd
    return float(d.min()), float(max(d.max() - d.min(), 1e-6))


def camera_for(view: str, width: float, height: float, lo, hi, clip: bool = False):
    """A camera for a named view, framed on the world box lo..hi. With `clip`
    the near plane sits at the box's near side: nothing in front of the box
    hides it (a close-up)."""
    if view not in VIEWS:
        raise SessionError(f"no view {view!r}; the views are {', '.join(VIEWS)}")
    yaw, pitch, ortho = VIEWS[view]
    return aimed_camera(yaw, pitch, ortho, width, height, lo, hi, clip=clip)


def aimed_camera(yaw: float, pitch: float, ortho: bool, width: float, height: float, lo, hi,
                 clip: bool = False, fov: float | None = None):
    """A camera at yaw and pitch (radians, as Camera has them) that frames the
    world box lo..hi; `fov` the vertical field of view in degrees (None: the
    editor's)."""
    from rok.app.camera import Camera

    cam = Camera(target=(np.asarray(lo, float) + np.asarray(hi, float)) / 2.0, yaw=yaw, pitch=pitch,
                 width=float(width), height=float(height), ortho=ortho,
                 up_hint=(0.0, 1.0, 0.0) if abs(pitch) > 1.5 else None, fov_deg=fov)
    if ortho:
        # Fit the box's extent across the view; the distance sets the half height.
        v = cam.view()
        right, up = v[0, :3], v[1, :3]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        centred = corners - cam.target
        half_w = float(np.max(np.abs(centred @ right)))
        half_h = float(np.max(np.abs(centred @ up)))
        half = max(half_h, half_w / cam.aspect, 0.5) * MARGIN
        cam.distance = half / math.tan(cam.fov_y / 2.0)
    else:
        cam.frame(lo, hi)
        # frame() fits the box's sphere; pull in until the corners fill the view.
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        for _ in range(3):
            p = cam.project(corners)
            fx = np.max(np.abs(p[:, 0] - cam.width / 2.0)) / (cam.width / 2.0)
            fy = np.max(np.abs(p[:, 1] - cam.height / 2.0)) / (cam.height / 2.0)
            cam.distance *= max(fx, fy) * MARGIN
    if clip:
        near, _depth = depth_range(cam, lo, hi)
        cam.near_clip = max(near, 0.0)
    return cam


def free_camera(width: float, height: float, lo, hi, *, azimuth: float | None = None, elevation: float | None = None,
                eye=None, target=None, distance: float | None = None, fov: float | None = None,
                ortho: bool = False):
    """A camera placed by hand, in world units (rok_render and rok_look).

    With `eye` it stands there and looks at `target` (default: the middle of
    lo..hi): a low eye at the edge of a landscape looks along the ground. Else
    it stands at `azimuth` (degrees about z, counted from +x toward +y: 0 is
    the +x side, 270 the -y (front) side) and `elevation` (degrees above the
    ground) round `target`, at `distance`, or far enough that lo..hi fills
    the picture. `fov` is the vertical field of view in degrees."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if eye is not None:
        if azimuth is not None or elevation is not None or distance is not None:
            raise SessionError("give eye (and target), or azimuth, elevation and distance; not both")
        eye = np.asarray(eye, float)
        aim = (lo + hi) / 2.0 if target is None else np.asarray(target, float)
        d = eye - aim
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            raise SessionError("eye and target are the same point; the camera needs a direction")
        yaw, pitch = math.atan2(d[1], d[0]), math.asin(max(-1.0, min(1.0, d[2] / dist)))
        if ortho:
            cam = aimed_camera(yaw, pitch, True, width, height, lo, hi, fov=fov)  # a flat view frames the box
            cam.target = aim
            return cam
        from rok.app.camera import Camera

        return Camera(target=aim, yaw=yaw, pitch=pitch, distance=dist, width=float(width), height=float(height),
                      up_hint=(0.0, 1.0, 0.0) if abs(pitch) > 1.5 else None, fov_deg=fov)
    yaw = math.radians(-45.0 if azimuth is None else float(azimuth))
    pitch = math.radians(35.0 if elevation is None else max(-90.0, min(90.0, float(elevation))))
    if target is not None:
        # Frame the box as if it were centred on the target.
        aim = np.asarray(target, float)
        half = np.maximum(np.abs(hi - aim), np.abs(lo - aim))
        lo, hi = aim - half, aim + half
    cam = aimed_camera(yaw, pitch, bool(ortho), width, height, lo, hi, fov=fov)
    if distance is not None:
        if float(distance) <= 0:
            raise SessionError("distance must be more than 0")
        cam.distance = float(distance)
    return cam



def main(argv: list[str]) -> int:
    """`python -m rok run SCRIPT.py [FILE.vox]`: run SCRIPT with a headless
    `session` in its globals."""
    import runpy

    if not argv:
        print("usage: python -m rok run SCRIPT.py [FILE.vox]", file=sys.stderr)
        return 2
    script, path = argv[0], (argv[1] if len(argv) > 1 else None)
    with Session(path) as session:
        runpy.run_path(script, init_globals={"session": session}, run_name="__main__")
    return 0
