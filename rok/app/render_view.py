"""View > Rendered View: the path traced picture in the viewport.

Rust traces it (native/src/trace.rs): glass you see through, voxels that give
light, metal, soft shadows, a sky and bloom, adding samples while the view
stands still. This module holds the switch (the nav cube's sun button, the
View menu item, the action) and the light that lights it: the file's own
light (rok.core.light, `VoxScene.light`), or the default light for a file
without one. A change to the light is one undo step and marks the file
changed. The samples are yours, not the file's (settings.txt). The rendered
view shows one view, so it ends the quad view, and the quad view ends it.
"""

from __future__ import annotations

from rok.core import light as light_core
from rok.core import render_presets
from rok.core.undo import LightRecord

from .opkit import Action, StringProp, register

# The samples (settings.txt): name -> (key, default). Picture 0: the preset's.
QUALITY = {
    "samples": ("render_samples", 512),
    "picture_samples": ("picture_samples", 0),
}


def quality() -> dict:
    """Samples of the view, and of a picture (0: the preset's)."""
    from . import settings

    out = {}
    for name, (key, default) in QUALITY.items():
        try:
            out[name] = int(float(settings.get(key) or default))
        except ValueError:
            out[name] = default
    return out


def save_quality(app, values: dict) -> None:
    from . import settings

    for name, (key, _default) in QUALITY.items():
        if name in values:
            settings.put(key, str(int(values[name])))
    app.request_redraw()


def picture_samples(preset: render_presets.Preset) -> int:
    return int(quality()["picture_samples"] or preset.samples)


def scene_light(scene=None) -> light_core.LightSetup:
    """The light of `scene` (the open one by default): its own, or the default."""
    from .object import document

    scene = document.scene if scene is None else scene
    own = getattr(scene, "light", None) if scene is not None else None
    return own if own is not None else light_core.default()


def has_own_light(scene=None) -> bool:
    from .object import document

    scene = document.scene if scene is None else scene
    return scene is not None and scene.light is not None


def set_light(app, setup: light_core.LightSetup | None, label: str = "Light") -> bool:
    """Give the open file `setup` as its light (None: no light of its own):
    one undo step, the file changed. False when nothing changed."""
    from rok.core import undo

    from .object import document

    scene = document.scene
    if scene is None:
        return False
    before = scene.light
    if before == setup:
        return False
    scene.light = setup
    with undo.naming(label):
        document.undo.push(LightRecord(before, setup))
    document.modified = True
    if app is not None:
        app.request_redraw()
    return True


def preview_light(app, setup: light_core.LightSetup) -> None:
    """Show `setup` without an undo step (a drag in progress); `set_light`
    at its end makes the step."""
    from .object import document

    if document.scene is None:
        return
    document.scene.light = setup
    if app is not None:
        app.request_redraw()


def preset_id() -> str:
    """The preset the open file's light started from."""
    pid = scene_light().preset
    return pid if pid in {p.id for p in render_presets.PRESETS} else render_presets.DEFAULT


def picture_preset() -> render_presets.Preset:
    """The preset whose camera and size a picture of the open file takes."""
    return render_presets.preset_by_id(preset_id())


def settings_for(setup: light_core.LightSetup, lo, hi) -> dict:
    """Host.set_render's arguments for the light and the scene box."""
    s = light_core.trace_settings(setup, lo, hi)
    s["max_samples"] = max(1, int(quality()["samples"]))
    return s


def rendered(app) -> bool:
    return bool(getattr(app, "rendered", False))


def _scene_box(scene) -> tuple:
    shown = [m for m in scene.models if not scene.is_hidden(m)] or list(scene.models)
    los = [tuple(int(p) for p in m.position) for m in shown]
    his = [tuple(int(p) + int(s) - 1 for p, s in zip(m.position, m.grid.size)) for m in shown]
    return tuple(min(c[k] for c in los) for k in range(3)), tuple(max(c[k] for c in his) for k in range(3))


def send(app) -> None:
    """Send the light for the scene as it is now (the lamps' distance and
    power follow the scene's size). Only when something changed."""
    from .object import document

    if not rendered(app) or document.scene is None or not document.scene.models:
        return
    lo, hi = _scene_box(document.scene)
    setup = scene_light()
    key = (setup, lo, hi, tuple(sorted(quality().items())))
    if getattr(app, "_render_sent", None) == key:
        return
    app._render_sent = key
    app.host.set_render(True, **settings_for(setup, lo, hi))


def set_rendered(app, on: bool) -> None:
    """Switch the rendered view on or off."""
    on = bool(on)
    if on == rendered(app):
        return
    if on:
        from . import cad_mode

        cad_mode.set_on(app, False)  # the rendered view replaces the CAD view
    app.rendered = on
    app._render_sent = None
    if on and getattr(app, "quad", False):
        app.set_quad(False)
    if on:
        send(app)
    else:
        app.host.set_render(False)
    app.set_menu()
    app.request_redraw()


@register
class RenderedViewAction(Action):
    command_id = "rok.rendered_view"
    title = "Rendered View"
    tooltip = "Show the model path traced: glass, lights, metal and soft shadows; again for the editing view"

    def execute(self, context):
        set_rendered(context.app, not rendered(context.app))
        return {"FINISHED"}


@register
class RenderPresetAction(Action):
    command_id = "rok.render_preset"
    title = "Render Preset"
    tooltip = "Light the file with a preset: its lamps, sky, ground, colours and bloom"

    preset = StringProp(name="Preset", default=render_presets.DEFAULT)

    def execute(self, context):
        if self.preset not in {p.id for p in render_presets.PRESETS}:
            return {"CANCELLED"}
        app = context.app
        set_light(app, light_core.from_preset(self.preset), label=f"Light: {self.preset.title()}")
        if not rendered(app):
            set_rendered(app, True)  # picking a preset shows it
        app.set_menu()
        app.request_redraw()
        return {"FINISHED"}
