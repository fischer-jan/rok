"""File > Render Picture: a finished picture with the file's light (Cmd+R).

The picture uses the file's light (View > Render Settings) and the
camera, size and samples of the preset that light started from. Isometric
and Diorama have a camera of their own, a true isometric one turned to the
side nearest the view; the others look as the view does. Rust traces it
off-screen (Host.render_picture); the PNG goes next to the .vox as
<name>_<preset>.png, or into Pictures/Rok for a file that is not saved yet.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from rok.core import png
from rok.core import light as light_core
from rok.core import render_presets as rp

from . import render_view
from .camera import Camera
from .object import document
from .opkit import Action, StringProp, register

TOKEN = "picture:"


def output_path(preset: rp.Preset) -> Path:
    """<vox folder>/<vox name>_<preset>.png, or Pictures/Rok/untitled_<preset>.png."""
    name = f"{preset.id.lower()}.png"
    if document.filepath:
        p = Path(document.filepath)
        return p.with_name(f"{p.stem}_{name}")
    return Path.home() / "Pictures" / "Rok" / f"untitled_{name}"


def picture_camera(view: Camera, preset: rp.Preset, lo, hi) -> Camera:
    """The camera for a preset's picture of the box [lo, hi], its size the
    preset's. A fixed-angle preset turns to the side nearest `view`."""
    w, h = preset.resolution
    if not preset.ortho:
        return Camera(target=np.asarray(view.target, dtype=np.float64).copy(), yaw=view.yaw, pitch=view.pitch,
                      distance=view.distance, width=float(w), height=float(h))
    corners = rp.box_corners(lo, hi)
    eye = np.asarray(view.eye(), dtype=np.float64) - np.asarray(view.target, dtype=np.float64)
    az = preset.camera_azimuth + rp.iso_turn(preset.camera_azimuth, rp.view_azimuth(eye))
    direction = rp.direction(az, preset.camera_elevation)
    _loc, width = rp.frame_ortho(corners, direction, h / w, preset.margin)
    half_h = width * (h / w) / 2.0
    cam = Camera(target=corners.mean(axis=0), yaw=math.radians(az), pitch=math.radians(preset.camera_elevation),
                 distance=1.0, ortho=True, width=float(w), height=float(h))
    cam.distance = half_h / math.tan(cam.fov_y / 2.0)  # an orthographic view's half height follows the distance
    return cam


def send(app, preset: rp.Preset, own: bool = True) -> Path | None:
    """Ask the host for the picture: the preset's camera and size, the file's
    light (`own`) or the preset's. Returns where it will be saved."""
    scene = document.scene
    if scene is None or not scene.models:
        return None
    lo, hi = render_view._scene_box(scene)
    setup = render_view.scene_light() if own else light_core.from_preset(preset)
    settings = render_view.settings_for(setup, lo, hi)
    settings["max_samples"] = render_view.picture_samples(preset)
    cam = picture_camera(app.main_camera, preset, lo, hi)
    path = output_path(preset)
    w, h = preset.resolution
    app.host.render_picture(f"{TOKEN}{path}", int(w), int(h), cam.view_proj(),
                            tuple(float(v) for v in cam.eye()), **settings)
    return path


def on_picture(app, data: dict) -> Path:
    """The host's picture: save it as a PNG and say so."""
    path = Path(str(data["token"])[len(TOKEN):])
    w, h = int(data["width"]), int(data["height"])
    rgba = np.frombuffer(data["rgba"], dtype=np.uint8).reshape(h, w, 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png.encode(rgba))
    app.say(f"Saved {path.name}", 6.0)
    return path


@register
class RenderPictureAction(Action):
    command_id = "rok.render_picture"
    title = "Render Picture"
    tooltip = ("Render a finished picture with the file's light and its preset's camera and size, "
                      "and save it as a PNG next to the file")

    preset = StringProp(name="Preset", description="A render preset id, its own light; empty: the file's light",
                            default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return document.scene is not None and bool(document.scene.models)

    def _preset(self) -> rp.Preset:
        pid = self.preset or render_view.preset_id()
        try:
            return rp.preset_by_id(pid)
        except KeyError:
            return rp.preset_by_id(rp.DEFAULT)

    def execute(self, context):
        path = send(context.app, self._preset(), own=not self.preset)
        return {"FINISHED"} if path is not None else {"CANCELLED"}

    def invoke(self, context, event):
        # Say so first: the picture takes a few seconds and the window waits.
        preset = self._preset()
        scene = document.scene
        bricks = scene is not None and any(m.grid.bricks is not None for m in scene.models)
        note = " The rendered picture shows bricks without seams." if bricks else ""
        context.app.say(f"Rendering the {preset.label} picture…{note}", 30.0)
        context.app.redraw()
        self._timer = context.prefs.add_timer(0.05, window=context.window)
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS"}
        context.prefs.remove_timer(self._timer)
        send(context.app, self._preset(), own=not self.preset)
        return {"FINISHED"}
