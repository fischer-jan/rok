"""File > Export Animation > Sprite Sheet, GIF and MP4: pictures of the
active clip.

The renderer draws them as it draws the view (Host.render_views, the
current look), so they show what the view shows. A job steps through the
clip: it sends the parts of one point in time (anim_mode.shown_at), asks for
a picture, and sends the next when the picture comes back as a
"screenshot" event whose token starts with "anim:". At the end the view
gets its own parts again.

- Sprite sheet: an orthographic camera at the angle of the view or of a
  saved view, turned about the up axis once per direction, framed the same
  for every frame and direction. Two pictures per cell, on black and on
  white, give the transparent background (core.anim_pictures.matte).
- GIF and MP4: the view as it is, at the clip's frame rate, with the
  editor's background.

The files go next to the document as <name>_<clip>.png and .json, .gif or
.mp4, or into Pictures/Rok for a file that is not saved yet.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rok.core import anim, gif, grid_ops, png, video
from rok.core import anim_pictures as pics

from . import anim_mode
from . import object as voxel_object
from .camera import Camera
from .object import document, style_of
from .opkit import EnumProp, IntProp, Action, register

TOKEN = "anim:"
SPRITES, GIF, MP4 = "SPRITES", "GIF", "MP4"
LABELS = {SPRITES: "Sprite Sheet", GIF: "GIF", MP4: "MP4"}
BLACK, WHITE = -1.0, -2.0  # Host.render_views backgrounds: flat black and flat white (shader.wgsl)
MARGIN = 1.08  # room round the sprite in its cell
GIF_WIDTH = 640  # the widest GIF, in pixels; the view's shape stays
MP4_WIDTH = 1280
STALL = 30.0  # seconds without a picture: the job ends, so the view never stays frozen


def output_path(clip: anim.Clip, suffix: str) -> Path:
    """<file folder>/<file name>_<clip>.<suffix>, or Pictures/Rok/untitled_<clip>.<suffix>."""
    clip_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip.name).strip("_") or "clip"
    if document.filepath:
        p = Path(document.filepath)
        return p.with_name(f"{p.stem}_{clip_name}{suffix}")
    return Path.home() / "Pictures" / "Rok" / f"untitled_{clip_name}{suffix}"


# --- framing ---------------------------------------------------------------------------


def shown_box(shown: anim_mode.Shown) -> tuple[np.ndarray, np.ndarray] | None:
    """The world box round every part that shows at one point in time."""
    scene = document.scene
    los, his = [], []
    for model in scene.models:
        if scene.is_hidden(model) or model.uid in shown.hidden:
            continue
        entry = shown.poses.get(model.uid)
        voxels, place = model.grid.voxels, np.asarray(model.position, dtype=np.float64)
        if entry is not None and entry[1] is not None:
            voxels, place = entry[1], np.asarray(entry[2], dtype=np.float64)
        elif entry is not None:
            place = np.asarray(entry[2], dtype=np.float64)
        bounds = grid_ops.fit_bounds(np.asarray(voxels))
        if bounds is None:
            continue
        lo = np.asarray(bounds[0], dtype=np.float64)
        hi = np.asarray(bounds[1], dtype=np.float64) + 1.0
        if model.uid in shown.turns:  # turned smoothly: the turned corners
            m = shown.turns[model.uid]
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            corners = corners @ m[:3, :3].T + m[:3, 3]
            lo, hi = corners.min(axis=0), corners.max(axis=0)
        los.append(place + lo)
        his.append(place + hi)
    if not los:
        return None
    return np.min(los, axis=0), np.max(his, axis=0)


def _corners(lo, hi) -> np.ndarray:
    return np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _ndc(cam: Camera, points: np.ndarray) -> np.ndarray:
    clip = points @ cam.view_proj().T
    return clip[:, :2] / clip[:, 3:4]


def sprite_cameras(yaw: float, pitch: float, angles: list[float], lo, hi, size: int) -> list[Camera]:
    """One orthographic camera per direction, all with the same size and
    centre, so the sprite stands on the same pixel in every cell."""
    centre = (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) / 2.0
    corners = _corners(lo, hi)
    limit = math.radians(89.0)  # straight down has no up to turn about
    pitch = min(max(pitch, -limit), limit)
    cams = [Camera(target=centre.copy(), yaw=yaw + math.radians(a), pitch=pitch, distance=100.0, ortho=True,
                   width=float(size), height=float(size)) for a in angles]
    reach = max(float(np.abs(_ndc(c, corners)).max()) for c in cams)  # in half heights at distance 100
    half = cams[0].half_height * reach * MARGIN
    for cam in cams:
        cam.distance = half / math.tan(cam.fov_y / 2.0)
    return cams


def pivot_pixel(cam: Camera, lo, hi) -> tuple[float, float]:
    """Where the middle of the bottom of the box shows in the cell, in pixels
    from the top left: the point the sprite stands on."""
    foot = np.array([[(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2], 1.0]])
    x, y = _ndc(cam, foot)[0]
    return (x + 1.0) / 2.0 * cam.width, (1.0 - y) / 2.0 * cam.height


def _row(cam: Camera, x: float = 0.0) -> tuple:
    return (x, 0.0, cam.width, cam.height, cam.view_proj(), tuple(float(c) for c in cam.eye()), float(cam.pixel),
            0.0, 0.0)


# --- the job ---------------------------------------------------------------------------


@dataclass
class Job:
    """One export: what to render at which time, and what came back."""

    kind: str
    clip: anim.Clip
    times: list[float]
    cameras: list[Camera]
    path: Path
    fps: int
    backgrounds: tuple = (0.0,)
    pivot: tuple | None = None
    angles: list = field(default_factory=lambda: [0.0])
    step: int = 0
    pictures: dict = field(default_factory=dict)  # (time index, background) -> RGBA
    sent: float = 0.0  # time.monotonic() of the last request

    @property
    def steps(self) -> list[tuple[int, float]]:
        return [(i, bg) for i in range(len(self.times)) for bg in self.backgrounds]

    @property
    def size(self) -> tuple[int, int]:
        cam = self.cameras[0]
        return int(cam.width) * len(self.cameras), int(cam.height)

    def send(self, app) -> None:
        i, bg = self.steps[self.step]
        shown = anim_mode.shown_at(self.clip, self.times[i])
        voxel_object.anim_hidden = shown.hidden
        app.sync.flush(document.scene, style_of=style_of, turns=shown.turns or None, poses=shown.poses,
                       palette=shown.palette)
        w, h = self.size
        rows = [_row(cam, k * cam.width) for k, cam in enumerate(self.cameras)]
        app.host.render_views(f"{TOKEN}{self.step}", w, h, rows, bg, True)
        self.sent = time.monotonic()

    def receive(self, app, data: dict) -> bool:
        """Keep a picture and ask for the next. True when the job is done."""
        w, h = int(data["width"]), int(data["height"])
        i, bg = self.steps[self.step]
        self.pictures[(i, bg)] = np.frombuffer(data["rgba"], dtype=np.uint8).reshape(h, w, 4).copy()
        self.step += 1
        if self.step < len(self.steps):
            self.send(app)
            return False
        return True

    def write(self) -> list[Path]:
        """Write the files; returns them."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == SPRITES:
            cw = int(self.cameras[0].width)
            rows = [[] for _ in self.cameras]
            for i in range(len(self.times)):
                both = pics.matte(self.pictures[(i, BLACK)], self.pictures[(i, WHITE)])
                for d in range(len(self.cameras)):
                    rows[d].append(both[:, d * cw:(d + 1) * cw])
            sheet = pics.assemble(rows, self.clip, self.fps, self.angles, self.path.name, self.pivot)
            self.path.write_bytes(png.encode(sheet.image))
            data = self.path.with_suffix(".json")
            data.write_text(json.dumps(sheet.json, indent=1), encoding="utf-8")
            return [self.path, data]
        frames = [self.pictures[(i, 0.0)] for i in range(len(self.times))]
        if self.kind == GIF:
            self.path.write_bytes(gif.encode(frames, self.fps, loop=self.clip.loop_mode != anim.ONCE))
        else:
            video.write_mp4(self.path, frames, self.fps)
        return [self.path]


_job: Job | None = None


def running() -> bool:
    """True while a job waits for its pictures. A job that has waited longer
    than STALL seconds ends here, so the view gets its own parts again."""
    global _job
    if _job is not None and time.monotonic() - _job.sent > STALL:
        _job = None
        voxel_object.anim_hidden = frozenset()
    return _job is not None


def start(app, job: Job) -> None:
    global _job
    _job = job
    job.send(app)


def on_picture(app, data: dict) -> list[Path] | None:
    """A picture of the running job came back; at the end, write the files
    and give the view its own parts again. Returns the files when done."""
    global _job
    job = _job
    if job is None or str(data.get("token", "")) != f"{TOKEN}{job.step}":
        return None  # a picture of a job that ended
    try:
        if not job.receive(app, data):
            return None
        files = job.write()
    except (OSError, ValueError) as exc:
        _job = None
        _restore(app)
        app.say(f"Export: {exc}", 8.0)
        return None
    _job = None
    _restore(app)
    app.say(f"Saved {', '.join(p.name for p in files)}", 8.0)
    return files


def _restore(app) -> None:
    voxel_object.anim_hidden = frozenset()
    app.flush()
    app.redraw()


def cancel() -> None:
    global _job
    _job = None


# --- building a job ------------------------------------------------------------------


def sprite_job(app, directions: int = 8, size: int = 128, fps: int = 0, view: int = 0, angle=None,
               path: Path | None = None) -> Job:
    """The sprite sheet of the active clip. `view` 0 takes the angle of the
    view, 1..4 that of a saved view; `angle` (yaw, pitch) in radians, when
    given, is the camera's. `path` overrides where it goes."""
    clip = anim_mode.clip()
    angles = pics.direction_angles(int(directions))
    times = pics.sheet_times(clip, fps)
    yaw, pitch = app.main_camera.yaw, app.main_camera.pitch
    saved = document.views.get(int(view)) if view else None
    if view and saved is None:
        raise ValueError(f"View {view} is not saved")
    if saved is not None:
        _location, (yaw, pitch), _distance, _persp = saved
    if angle is not None:
        yaw, pitch = angle
    boxes = [b for b in (shown_box(anim_mode.shown_at(clip, t)) for t in times) if b is not None]
    if not boxes:
        raise ValueError("there are no voxels to draw")
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    cams = sprite_cameras(float(yaw), float(pitch), angles, lo, hi, int(size))
    path = Path(path).with_suffix(".png") if path else output_path(clip, ".png")
    return Job(SPRITES, clip, times, cams, path, int(fps) or clip.fps, backgrounds=(BLACK, WHITE),
               pivot=pivot_pixel(cams[0], lo, hi), angles=angles)


def video_job(app, kind: str, camera: Camera | None = None, path: Path | None = None) -> Job:
    """A GIF or an MP4 of the active clip from the view as it is, or from
    `camera`. `path` overrides where it goes."""
    clip = anim_mode.clip()
    view = camera or app.main_camera
    widest = GIF_WIDTH if kind == GIF else MP4_WIDTH
    scale = min(1.0, widest / max(view.width, 1.0)) if kind == GIF else min(2.0, widest / max(view.width, 1.0))
    cam = Camera(target=np.asarray(view.target, dtype=np.float64).copy(), yaw=view.yaw, pitch=view.pitch,
                 distance=view.distance, ortho=view.ortho, width=float(round(view.width * scale)),
                 height=float(round(view.height * scale)), up_hint=view.up_hint)
    suffix = ".gif" if kind == GIF else ".mp4"
    return Job(kind, clip, pics.video_times(clip), [cam], Path(path).with_suffix(suffix) if path else output_path(clip, suffix),
               clip.fps)


@register
class ExportPicturesAction(Action):
    command_id = "rok.export_pictures"
    title = "Export Pictures"
    tooltip = ("Pictures of the active clip as the view draws them: a sprite sheet with its JSON, "
                      "a GIF or an MP4. Saved next to the file")

    kind = EnumProp(name="Kind", items=[(k, v, "") for k, v in LABELS.items()], default=SPRITES)
    directions = EnumProp(name="Directions", items=[(str(n), str(n), "") for n in pics.DIRECTIONS],
                              default="8")
    size = IntProp(name="Cell size", description="Each cell's width and height in pixels", default=128,
                       min=16, max=1024)
    fps = IntProp(name="Frames per second", description="The sheet's frame rate; 0: the clip's", default=0,
                      min=0, max=120)
    view = IntProp(name="Saved view", description="The camera's angle: 0 the view's, 1 to 4 a saved view's",
                       default=0, min=0, max=4)

    @classmethod
    def poll(cls, context):
        scene = document.scene
        return scene is not None and bool(scene.models) and anim_mode.clip() is not None and not running()

    def invoke(self, context, event):
        if self.kind == SPRITES:
            self.fps = 0
            return context.prefs.open_dialog(self, title="Sprite Sheet", confirm_text="Export")
        return self.execute(context)

    def draw(self, context):
        for name in ("directions", "size", "fps", "view"):
            self.layout.prop(self, name)

    def execute(self, context):
        if running():
            self.report({"WARNING"}, "An export is still running")
            return {"CANCELLED"}
        if self.kind == MP4 and video.find_ffmpeg() is None:
            self.report({"ERROR"}, video.MISSING)
            return {"CANCELLED"}
        try:
            if self.kind == SPRITES:
                job = sprite_job(context.app, int(self.directions), int(self.size), int(self.fps), int(self.view))
            else:
                job = video_job(context.app, self.kind)
        except ValueError as exc:
            self.report({"ERROR"}, f"Export: {exc}")
            return {"CANCELLED"}
        context.app.say(f"Rendering the {LABELS[self.kind]} of {job.clip.name}…", 30.0)
        start(context.app, job)
        return {"FINISHED"}
