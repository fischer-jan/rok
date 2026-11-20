"""View > View Settings and View > Render Settings: a panel at the right edge
of the view.

Every change shows in the view at once; nothing waits for an Apply. The panel
shows one of two pages:

- View Settings: the editing view's look, ambient occlusion and smooth edges.
  These are yours, in every file (settings.txt).
- Render Settings: the file's light (rok.core.light): a preset to start
  from, the lights, the sky and ground, the finish, and the samples (yours).

The page shows its view: Render Settings turns the rendered view on, View
Settings shows the editing view, and Close puts back the view the panel opened
with. The other command while the panel is open turns to its page; the same
command again closes the panel. A light
change is one undo step when the gesture ends (a slider on release, a click);
during a drag the view shows the light without a step. Revert puts back the
look and the light as they were when the panel opened.

Beside the panel the view still orbits, pans and zooms (the hold keys,
the trackpad, the wheel); a plain click in the view closes the panel.
Esc closes the open list or picker first, then the panel.

The layout is rok/core/view_panel.py.
"""

from __future__ import annotations

import time
from dataclasses import replace

from rok.core import chrome as kit
from rok.core import light as light_core
from rok.core import render_presets
from rok.core import shader_panel as sp
from rok.core import view_panel as vp

from . import colour_picker, looks, render_view
from . import object as voxel_object
from .gfx import pen
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, register
from .paint import clip_text, round_outline, round_rect, text, text_width
from .state import state

PANEL_BG = (0.060, 0.056, 0.053, 0.97)
PANEL_EDGE = (0.228, 0.212, 0.196, 1.0)
TRACK_BG = (1.0, 1.0, 1.0, 0.10)
BUTTON_BG = (1.0, 1.0, 1.0, 0.07)
BUTTON_HOVER = (0.208, 0.192, 0.180, 1.0)
SECTION_BG = (1.0, 1.0, 1.0, 0.035)
EMBER = (0.910, 0.365, 0.227, 1.0)
EMBER_DARK = (0.102, 0.055, 0.031, 1.0)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)
TEXT_FAINT = (0.450, 0.430, 0.410, 1.0)

_UNSET = object()  # no gesture runs
_session: "_Session | None" = None


class _Session:
    def __init__(self, region_ptr: int, tab: str) -> None:
        self.region_ptr = region_ptr
        self.tab = tab
        self.section: str | None = vp.LIGHTS
        self.picked = 0
        self.menu = False
        self.hover: vp.Item | None = None
        self.hover_since = 0.0
        self.drag: tuple[str, float, float, bool] | None = None  # slider key, start value, start x, fine
        self.before = _UNSET  # the file's light when the gesture started
        self.picker: str | None = None  # the swatch whose picker is open
        self.pick_hsv = (0.0, 0.0, 0.0)
        self.pick_drag: str | None = None
        self.pick_hover: str | None = None
        self.opened: dict = {}
        self.dome_drag: tuple | None = None  # a drag on the light dome (light_dome.py)
        self.mouse: tuple[float, float] | None = None


def active() -> bool:
    return _session is not None


# --- what the panel shows ---------------------------------------------------------------


def _setup() -> light_core.LightSetup:
    return render_view.scene_light()


def _picked(session: _Session, setup: light_core.LightSetup) -> int:
    return min(max(session.picked, 0), max(0, len(setup.lights) - 1))


def _field(key: str) -> str:
    """A slider's field of a Light or the LightSetup."""
    name = key[len("light."):] if key.startswith("light.") else key
    return "softness" if name.startswith("softness") else name


def value(session: _Session, key: str) -> float:
    setup = _setup()
    if key.startswith("light."):
        if not setup.lights:
            return 0.0
        return float(getattr(setup.lights[_picked(session, setup)], _field(key)))
    return float(getattr(setup, key))


def with_value(session: _Session, setup: light_core.LightSetup, key: str, v) -> light_core.LightSetup:
    if key.startswith("light."):
        if not setup.lights:
            return setup
        return light_core.with_light(setup, _picked(session, setup), **{_field(key): v})
    return replace(setup, **{key: v})


def _colour(session: _Session, setup: light_core.LightSetup, key: str):
    """A swatch's linear colour, or None."""
    if key == "light.colour":
        return setup.lights[_picked(session, setup)].colour if setup.lights else None
    if key == "backdrop_colour":
        return setup.backdrop
    if key == "ground_colour":
        return setup.ground
    return getattr(setup, key)


def _with_colour(session: _Session, setup, key: str, rgb) -> light_core.LightSetup:
    if key == "light.colour":
        return light_core.with_light(setup, _picked(session, setup), colour=rgb) if setup.lights else setup
    if key == "backdrop_colour":
        return replace(setup, backdrop=rgb)
    if key == "ground_colour":
        return replace(setup, ground=rgb)
    return replace(setup, **{key: rgb})


def _state(session: _Session, app) -> vp.State:
    setup = _setup()
    return vp.State(
        tab=session.tab,
        section=session.section,
        looks=tuple((k, v.label) for k, v in looks.LOOKS.items()),
        lights=tuple(light.kind for light in setup.lights),
        picked=_picked(session, setup),
        backdrop="CLEAR" if setup.transparent else ("COLOUR" if setup.backdrop is not None else "SKY"),
        ground=setup.ground is not None,
        changed=not light_core.same_as_preset(setup),
        menu=session.menu,
        presets=tuple((p.id, p.label) for p in render_presets.PRESETS),
        dome=True,
    )


def _layout(session: _Session, region, app=None) -> vp.PanelLayout:
    return vp.panel_layout(1.0, region.width, region.height, _state(session, app))


def _chip_on(session: _Session, app, item: vp.Item) -> bool:
    """Whether a chip or toggle is on."""
    setup = _setup()
    key = item.key
    if item.kind == "TOGGLE":
        if key == "box":
            return bool(state.model_box)
        return bool(getattr(app, "occlusion", True)) if key == "occlusion" else bool(getattr(app, "smooth", True))
    if item.kind == "LIGHT":
        return item.value == _picked(session, setup)
    if key == "look":
        return looks.current(app) == item.value
    if key == "kind":
        return bool(setup.lights) and setup.lights[_picked(session, setup)].kind == item.value
    if key == "backdrop":
        return _state(session, app).backdrop == item.value
    if key == "ground":
        return (setup.ground is not None) == (item.value == "ON")
    if key == "tone":
        return setup.tone == item.value
    if key in ("samples", "picture_samples"):
        return render_view.quality()[key] == item.value
    return False


# --- changes ------------------------------------------------------------------------------


def _begin(session: _Session) -> None:
    if session.before is _UNSET and document.scene is not None:
        session.before = document.scene.light


def _show(app, setup: light_core.LightSetup) -> None:
    render_view.preview_light(app, setup)


def _commit(session: _Session, app, label: str = "Light") -> None:
    """End the gesture: one undo step from the light at its start to now."""
    if session.before is _UNSET or document.scene is None:
        session.before = _UNSET
        return
    before, after = session.before, document.scene.light
    session.before = _UNSET
    document.scene.light = before
    if after is not None and after != render_view.scene_light():
        render_view.set_light(app, after, label)
    app.request_redraw()


def _apply(session: _Session, app, setup: light_core.LightSetup, label: str = "Light") -> None:
    """A change in one click: shown and one undo step."""
    _begin(session)
    _show(app, setup)
    _commit(session, app, label)


def _set_tab(session: _Session, app, tab: str) -> None:
    session.tab = tab
    session.menu = False
    session.picker = None
    render_view.set_rendered(app, tab == vp.RENDER)


def _press(session: _Session, app, item: vp.Item, layout: vp.PanelLayout, mx: float, my: float, shift: bool):
    """A click on the panel. Returns "CLOSE" when the panel closes."""
    setup = _setup()
    kind, key = item.kind, item.key
    if kind not in ("PRESET", "MENU") and session.menu:
        _close_menu(session, app)
    if kind == "CLOSE" or (kind == "BUTTON" and key == "done"):
        return "CLOSE"
    if kind == "TOGGLE":
        from . import settings

        if key == "occlusion":
            app.set_occlusion(not bool(getattr(app, "occlusion", True)))
            settings.put("ambient_occlusion", "on" if app.occlusion else "off")
        elif key == "box":
            _set_box(app, not state.model_box)
        else:
            app.smooth = not bool(getattr(app, "smooth", True))
            app.host.set_smooth_edges(app.smooth)
    elif kind == "CHIP":
        _chip(session, app, setup, key, item.value)
    elif kind == "SECTION":
        session.section = None if session.section == key else key
        session.picker = None
    elif kind == "LIGHT":
        session.picked = int(item.value)
    elif kind == "ADD":
        new = light_core.added(setup)
        _apply(session, app, new, "Add Light")
        session.picked = len(new.lights) - 1
    elif kind == "PRESET":
        if session.menu:
            _close_menu(session, app)
        else:
            _begin(session)  # the hover shows each preset; the click makes the step
            session.menu = True
    elif kind == "MENU":
        _show(app, light_core.from_preset(key))
        _commit(session, app, f"Light: {item.text}")
        session.menu = False
        session.picked = 0
    elif kind == "SWATCH":
        colour = _colour(session, setup, key)
        if colour is None:
            return None
        if session.picker == key:
            session.picker = None
        else:
            session.picker = key
            session.pick_hsv = colour_picker.hsv_of(light_core.srgb255(colour))
            session.pick_drag = None
    elif kind == "SLIDER":
        control = item.control
        start = value(session, key)
        _begin(session)
        if not shift:
            start = sp.value_at(control, mx)
            _show(app, with_value(session, setup, key, start))
        session.drag = (key, start, mx, bool(shift))
    elif kind == "DOME":
        from . import light_dome

        light_dome.press(session, app, item, mx, my)
    elif kind == "BUTTON":
        if key == "reset":
            _apply(session, app, replace(light_core.from_preset(setup.preset)), "Reset Light")
        elif key == "revert":
            _revert(session, app)
        elif key == "mute" and setup.lights:
            k = _picked(session, setup)
            _apply(session, app, light_core.with_light(setup, k, on=not setup.lights[k].on), "Light")
        elif key == "remove" and setup.lights:
            _apply(session, app, light_core.removed(setup, _picked(session, setup)), "Remove Light")
            session.picked = max(0, session.picked - 1)
    return None


def _chip(session: _Session, app, setup, key: str, choice) -> None:
    if key == "look":
        looks.apply(app, choice)
    elif key == "kind" and setup.lights:
        k = _picked(session, setup)
        light = setup.lights[k]
        if light.kind != choice:
            soft = 2.0 if choice == light_core.SUN else 0.3
            _apply(session, app, light_core.with_light(setup, k, kind=choice, softness=soft,
                                                       distance=light_core.LAMP_DISTANCE), "Light")
    elif key == "backdrop":
        if choice == "SKY":
            new = replace(setup, backdrop=None, transparent=False)
        elif choice == "COLOUR":
            new = replace(setup, backdrop=setup.backdrop or setup.sky_horizon, transparent=False)
        else:
            new = replace(setup, transparent=True)
        _apply(session, app, new, "Background")
    elif key == "ground":
        try:
            preset = render_presets.preset_by_id(setup.preset)
        except KeyError:
            preset = None
        _apply(session, app, light_core.with_ground(setup, choice == "ON", preset), "Ground")
    elif key == "tone":
        _apply(session, app, replace(setup, tone=choice), "Curve")
    elif key in ("samples", "picture_samples"):
        render_view.save_quality(app, {key: int(choice)})


def _close_menu(session: _Session, app) -> None:
    """Put the preset list away: the light it showed on hover goes back."""
    session.menu = False
    if session.before is not _UNSET and document.scene is not None:
        document.scene.light = session.before
        session.before = _UNSET
        app.request_redraw()


def _set_box(app, on: bool) -> None:
    """The faint box around the models, on or off, kept for the next start."""
    from . import settings

    state.model_box = bool(on)
    settings.put("model_box", "on" if on else "off")
    app.request_redraw()


def _revert(session: _Session, app) -> None:
    """Everything as it was when the panel opened: the look and the light."""
    o = session.opened
    if looks.current(app) != o["look"]:
        looks.apply(app, o["look"])
    if bool(app.occlusion) != o["occlusion"]:
        from . import settings

        app.set_occlusion(o["occlusion"])
        settings.put("ambient_occlusion", "on" if app.occlusion else "off")
    if bool(app.smooth) != o["smooth"]:
        app.smooth = o["smooth"]
        app.host.set_smooth_edges(app.smooth)
    if state.model_box != o["box"]:
        _set_box(app, o["box"])
    if document.scene is not None and document.scene.light != o["light"]:
        render_view.set_light(app, o["light"], "Revert Light")
    session.picked = 0
    app.request_redraw()


def _picker_parts(session: _Session, region, layout: vp.PanelLayout) -> dict | None:
    if session.picker is None:
        return None
    item = layout.find("SWATCH", session.picker)
    if item is None:
        return None
    return colour_picker.layout(item.rect, region, layout.scale, len(colour_picker.quick_colours()))


def _picker_event(session: _Session, app, parts: dict, kind: str, value_: str, mx: float, my: float,
                  layout: vp.PanelLayout) -> bool:
    """An event while a picker is open. True when the picker took it."""
    setup = _setup()
    key = session.picker
    if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
        if session.pick_drag is not None:
            session.pick_hsv = colour_picker.dragged(session.pick_hsv, session.pick_drag, parts[session.pick_drag],
                                                     mx, my)
            _show(app, _with_colour(session, setup, key, light_core.linear_of(colour_picker.rgb_of(session.pick_hsv))))
            return True
        session.pick_hover = colour_picker.part_at(parts, mx, my)
        return colour_picker._inside(parts["PANEL"], mx, my)
    if kind == "LEFT_MOUSE" and value_ == "RELEASE" and session.pick_drag is not None:
        session.pick_drag = None
        _commit(session, app, "Colour")
        return True
    if kind == "LEFT_MOUSE" and value_ == "PRESS":
        if not colour_picker._inside(parts["PANEL"], mx, my):
            hit = vp.hit(layout, mx, my)
            if hit is not None and hit.kind == "SWATCH":
                return False  # another swatch: the panel opens that one
            session.picker = None  # a click outside the picker closes it
            return True
        part = colour_picker.part_at(parts, mx, my)
        if part in ("SV", "HUE"):
            _begin(session)
            session.pick_drag = part
            session.pick_hsv = colour_picker.dragged(session.pick_hsv, part, parts[part], mx, my)
            _show(app, _with_colour(session, setup, key, light_core.linear_of(colour_picker.rgb_of(session.pick_hsv))))
        elif part is not None and part.startswith("Q:"):
            quick = colour_picker.quick_colours()
            q = int(part[2:])
            if q < len(quick):
                session.pick_hsv = colour_picker.hsv_of(quick[q])
                _apply(session, app, _with_colour(session, setup, key, light_core.linear_of(quick[q])), "Colour")
        elif part == "RESET":
            own = _preset_colour(session, setup, key)
            if own is not None:
                session.pick_hsv = colour_picker.hsv_of(light_core.srgb255(own))
                _apply(session, app, _with_colour(session, setup, key, own), "Colour")
        return True
    if value_ == "PRESS" and kind in {"ESCAPE", "ENTER", "NUMPAD_ENTER"}:
        session.picker = None
        return True
    return False


def _preset_colour(session: _Session, setup, key: str):
    """The colour the preset gives this swatch."""
    try:
        preset = light_core.from_preset(setup.preset)
    except KeyError:
        return None
    if key == "light.colour":
        k = _picked(session, setup)
        return preset.lights[k].colour if k < len(preset.lights) else (1.0, 1.0, 1.0)
    if key == "backdrop_colour":
        return preset.backdrop or preset.sky_horizon
    if key == "ground_colour":
        return preset.ground or (0.3, 0.3, 0.3)
    return getattr(preset, key)


def _is_navigation(event) -> bool:
    """A press with a hold key down: Ctrl, Cmd or Option move the view (view_ops.HoldMoveAction)."""
    return bool(getattr(event, "ctrl", False) or getattr(event, "cmd_key", False) or getattr(event, "alt", False))


# --- the action ------------------------------------------------------------------------------


@register
class ViewSettingsAction(Action):
    command_id = "rok.view_settings"
    title = "View Settings"
    tooltip = "How the editing view looks, in a panel beside the model: every change shows at once"
    page = vp.LOOK

    def invoke(self, context, event):
        global _session
        app = context.app
        region = context.region
        if region is None:
            return {"CANCELLED"}
        if _session is not None:
            if _session.tab == self.page:
                _close(app, keep_view=True)  # again: the panel closes
            else:
                _set_tab(_session, app, self.page)  # the other command turns the page
                app.request_redraw()
            return {"FINISHED"}
        rendered = render_view.rendered(app)
        _session = _Session(region.as_pointer(), self.page)
        _session.opened = {
            "look": looks.current(app),
            "occlusion": bool(getattr(app, "occlusion", True)),
            "smooth": bool(getattr(app, "smooth", True)),
            "box": bool(state.model_box),
            "rendered": rendered,
            "light": document.scene.light if document.scene is not None else None,
        }
        render_view.set_rendered(app, self.page == vp.RENDER)
        context.prefs.add_modal(self)
        app.request_redraw()
        return {"RUNNING"}

    def modal(self, context, event):
        session = _session
        app = context.app
        if session is None:
            return {"CANCELLED"}
        region = context.region
        if region is None:
            return {"PASS"}
        kind, val = event.type, event.value
        mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
        layout = _layout(session, region, app)
        inside = _inside(layout.rect, mx, my) or (layout.menu is not None and _inside(layout.menu, mx, my))
        parts = _picker_parts(session, region, layout)
        if parts is not None and _picker_event(session, app, parts, kind, val, mx, my, layout):
            app.request_redraw()
            return {"RUNNING"}
        session.mouse = (mx, my)
        if session.dome_drag is not None:
            from . import light_dome

            light_dome.drag_event(session, app, layout, event, mx, my)
            app.request_redraw()
            return {"RUNNING"}
        if session.drag is not None:
            key, start, start_x, _fine = session.drag
            control = layout.control(key)
            if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"} and control is not None:
                v = sp.fine(control, start, start_x, mx) if event.shift else sp.value_at(control, mx)
                if not event.shift:
                    session.drag = (key, v, mx, False)
                _show(app, with_value(session, _setup(), key, v))
            elif kind == "LEFT_MOUSE" and val == "RELEASE":
                session.drag = None
                _commit(session, app, (control.label if control is not None else "Light"))
            app.request_redraw()
            return {"RUNNING"}
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            found = vp.hit(layout, mx, my)
            if _key(found) != _key(session.hover):
                session.hover = found
                session.hover_since = time.monotonic()
                if found is not None and found.kind == "SLIDER_LABEL":
                    app.host.tick(kit.TOOLTIP_DELAY_S + 0.02)  # a mouse that rests sends no event
                if session.menu and found is not None and found.kind == "MENU":
                    _show(app, light_core.from_preset(found.key))  # the hovered preset, live
                elif session.menu and session.before is not _UNSET and document.scene is not None:
                    document.scene.light = session.before  # off the list: the light it had
                app.request_redraw()
            if found is not None and found.kind == "DOME":
                app.request_redraw()
            return {"RUNNING"} if inside else {"PASS"}
        if kind == "LEFT_MOUSE" and val == "PRESS" and inside:
            found = vp.hit(layout, mx, my)
            if found is not None and found.kind not in ("PANEL", "MENU_BG", "SLIDER_LABEL"):
                if _press(session, app, found, layout, mx, my, bool(event.shift)) == "CLOSE":
                    _close(app)
                    return {"FINISHED"}
            app.request_redraw()
            return {"RUNNING"}
        if kind in {"WHEEL_UP", "WHEEL_DOWN"} and inside:
            found = vp.hit(layout, mx, my)
            if found is not None and found.kind == "SLIDER":
                v = sp.nudged(found.control, value(session, found.key), 1 if kind == "WHEEL_UP" else -1)
                _apply(session, app, with_value(session, _setup(), found.key, v), found.text)
                app.request_redraw()
            return {"RUNNING"}
        if kind in {"TRACKPAD_PAN", "TRACKPAD_ZOOM"} and inside:
            return {"RUNNING"}
        if val == "PRESS" and kind == "ESCAPE":
            if session.menu:
                _close_menu(session, app)
                app.request_redraw()
                return {"RUNNING"}
            _close(app)
            return {"FINISHED"}
        if val == "PRESS" and kind in {"LEFT_MOUSE", "RIGHT_MOUSE"} and not inside:
            if _is_navigation(event):
                return {"PASS"}
            if session.menu:
                _close_menu(session, app)
                app.request_redraw()
                return {"RUNNING"}
            _close(app)  # a plain click in the view: the panel goes, the click does nothing else
            return {"FINISHED"}
        if kind in {"LEFT_MOUSE", "RIGHT_MOUSE"} and inside:
            return {"RUNNING"}
        return {"PASS"}

    def cancel(self, context):
        _close(context.app)


@register
class RenderSettingsAction(ViewSettingsAction):
    command_id = "rok.render_settings"
    title = "Render Settings"
    tooltip = "The file's light and the rendered view's finish, in a panel beside the model: every change shows at once"
    page = vp.RENDER


def _key(item: vp.Item | None):
    return None if item is None else (item.kind, item.key, item.value)


def _inside(rect, x: float, y: float) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _close(app, keep_view: bool = False) -> None:
    """Close the panel: an unfinished gesture ends, and the view goes back to
    the one the panel opened with."""
    global _session
    session = _session
    if session is None:
        return
    if session.menu:
        _close_menu(session, app)
    if session.before is not _UNSET:
        _commit(session, app)
    _session = None
    if not keep_view and app is not None:
        render_view.set_rendered(app, session.opened.get("rendered", False))
    voxel_object.redraw_3d_views()


# --- drawing -----------------------------------------------------------------------------------


def draw(region, app) -> None:
    session = _session
    if session is None or region.as_pointer() != session.region_ptr:
        return
    layout = _layout(session, region, app)
    s = layout.scale
    hover = session.hover
    round_rect(layout.rect, 14 * s, PANEL_BG)
    round_outline(layout.rect, 14 * s, PANEL_EDGE, 1.0 * s)
    setup = _setup()
    dragging = session.drag[0] if session.drag is not None else None
    for item in layout.items:
        lit = hover is not None and _key(hover) == _key(item)
        if item.kind == "MENU":
            continue  # the list is its own layer
        if item.kind == "TITLE":
            text(item.text, item.rect[0], item.rect[1] + item.rect[3] / 2, 14 * s, TEXT)
        elif item.kind in ("CHIP", "LIGHT"):
            on = _chip_on(session, app, item)
            round_rect(item.rect, item.rect[3] / 2, EMBER if on else (BUTTON_HOVER if lit else BUTTON_BG))
            label = item.text
            if item.kind == "LIGHT":
                lt = setup.lights[int(item.value)]
                dot = item.rect[0] + 12 * s
                cy = item.rect[1] + item.rect[3] / 2
                _disc(dot, cy, 5 * s, tuple(c / 255.0 for c in light_core.srgb255(lt.colour)) + (1.0 if lt.on else 0.35,))
                text(clip_text(label, item.rect[2] - 24 * s, 11 * s), dot + 9 * s, cy, 11 * s,
                     EMBER_DARK if on else (TEXT if lt.on else TEXT_FAINT))
                continue
            text(clip_text(label, item.rect[2] - 8 * s, 12 * s), item.rect[0] + item.rect[2] / 2,
                 item.rect[1] + item.rect[3] / 2, 12 * s, EMBER_DARK if on else (TEXT if lit else TEXT_DIM), "centre")
        elif item.kind == "CLOSE" or item.kind == "ADD":
            round_rect(item.rect, item.rect[3] / 2, BUTTON_HOVER if lit else BUTTON_BG)
            text(item.text, item.rect[0] + item.rect[2] / 2, item.rect[1] + item.rect[3] / 2, 14 * s,
                 TEXT if lit else TEXT_DIM, "centre")
        elif item.kind == "LABEL":
            if item.text:
                text(clip_text(item.text, item.rect[2] - 6 * s, 12 * s), item.rect[0], item.rect[1] + item.rect[3] / 2,
                     12 * s, TEXT_DIM)
        elif item.kind == "NOTE":
            text(clip_text(item.text, item.rect[2], 11 * s), item.rect[0], item.rect[1] + item.rect[3] / 2,
                 11 * s, TEXT_FAINT)
        elif item.kind == "TOGGLE":
            on = _chip_on(session, app, item)
            box = (item.rect[0], item.rect[1] + (item.rect[3] - 16 * s) / 2, 16 * s, 16 * s)
            round_rect(box, 4 * s, EMBER if on else BUTTON_BG)
            round_outline(box, 4 * s, TEXT if lit else PANEL_EDGE, 1.0 * s)
            if on:
                pts = [(box[0] + 4 * s, box[1] + 8 * s), (box[0] + 7 * s, box[1] + 4.5 * s),
                       (box[0] + 12.5 * s, box[1] + 11.5 * s)]
                pen.lines(pen.to_3d([pts[0], pts[1], pts[1], pts[2]]), EMBER_DARK, 2.0 * s)
            text(item.text, box[0] + 26 * s, item.rect[1] + item.rect[3] / 2, 12 * s, TEXT if lit else TEXT_DIM)
        elif item.kind == "SECTION":
            round_rect(item.rect, 7 * s, BUTTON_HOVER if lit else SECTION_BG)
            cx, cy = item.rect[0] + 12 * s, item.rect[1] + item.rect[3] / 2
            _chevron(cx, cy, s, down=bool(item.value), colour=TEXT if lit else TEXT_DIM)
            text(item.text, item.rect[0] + 24 * s, cy, 12 * s, TEXT if item.value or lit else TEXT_DIM)
        elif item.kind == "PRESET":
            round_rect(item.rect, item.rect[3] / 2, BUTTON_HOVER if lit or session.menu else BUTTON_BG)
            try:
                label = render_presets.preset_by_id(setup.preset).label
            except KeyError:
                label = setup.preset.title()
            if not light_core.same_as_preset(setup):
                label += ", changed"
            text(clip_text(label, item.rect[2] - 30 * s, 12 * s), item.rect[0] + 12 * s,
                 item.rect[1] + item.rect[3] / 2, 12 * s, TEXT)
            _chevron(item.rect[0] + item.rect[2] - 14 * s, item.rect[1] + item.rect[3] / 2, s, down=True,
                     colour=TEXT_DIM)
        elif item.kind == "BUTTON":
            primary = item.key == "done"
            muted = item.key == "mute" and setup.lights and not setup.lights[_picked(session, setup)].on
            word = ("off" if muted else "on") if item.key == "mute" else item.text
            round_rect(item.rect, item.rect[3] / 2, EMBER if primary else (BUTTON_HOVER if lit else BUTTON_BG))
            if primary and lit:
                round_outline(item.rect, item.rect[3] / 2, TEXT, 1.5 * s)
            text(word, item.rect[0] + item.rect[2] / 2, item.rect[1] + item.rect[3] / 2, 12 * s,
                 EMBER_DARK if primary else (TEXT_FAINT if muted else (TEXT if lit else TEXT_DIM)), "centre")
        elif item.kind == "SWATCH":
            colour = _colour(session, setup, item.key)
            if colour is None:
                continue
            round_rect(item.rect, 5 * s, tuple(c / 255.0 for c in light_core.srgb255(colour)) + (1.0,))
            ring = EMBER if session.picker == item.key else (TEXT if lit else PANEL_EDGE)
            round_outline(item.rect, 5 * s, ring, (2.0 if session.picker == item.key or lit else 1.0) * s)
            if item.text:
                text(item.text, item.rect[0] + item.rect[2] + 6 * s, item.rect[1] + item.rect[3] / 2, 11 * s,
                     TEXT_DIM)
        elif item.kind == "SLIDER":
            _draw_slider(session, item, lit or dragging == item.key, s)
        elif item.kind == "DOME":
            from . import light_dome

            light_dome.draw(session, app, item.rect, s, session.mouse)


def _disc(cx: float, cy: float, r: float, colour) -> None:
    """A filled circle among the rounded rects (a layer draws its triangles
    under them)."""
    round_rect((cx - r, cy - r, 2 * r, 2 * r), r, colour)


def _chevron(cx: float, cy: float, s: float, down: bool, colour) -> None:
    """A small open arrow: down for an open section and a list, right for a shut one."""
    if down:
        pts = [(cx - 4 * s, cy + 2 * s), (cx, cy - 2 * s), (cx, cy - 2 * s), (cx + 4 * s, cy + 2 * s)]
    else:
        pts = [(cx - 2 * s, cy + 4 * s), (cx + 2 * s, cy), (cx + 2 * s, cy), (cx - 2 * s, cy - 4 * s)]
    pen.lines(pen.to_3d(pts), colour, 1.5 * s)


def _draw_slider(session: _Session, item: vp.Item, lit: bool, s: float) -> None:
    c = item.control
    v = value(session, item.key)
    lx, ly, lw, lh = c.label_rect
    text(clip_text(c.label, lw - 8 * s, 12 * s), lx, ly + lh / 2, 12 * s, TEXT if lit else TEXT_DIM)
    bx, by, bw, bh = c.track
    round_rect(c.track, bh / 2, TRACK_BG)
    knob_x = sp.x_of(c, v)
    zero = sp.x_of(c, 0.0) if c.low < 0 < c.high else bx  # a range around 0 fills from 0
    lo_x, hi_x = sorted((zero, knob_x))
    if hi_x > lo_x:
        round_rect((lo_x, by, hi_x - lo_x, bh), bh / 2, EMBER)
    r = (8.0 if lit else 7.0) * s
    _disc(knob_x, by + bh / 2, r, TEXT)
    _disc(knob_x, by + bh / 2, r - 2.5 * s, EMBER if lit else PANEL_BG)
    vx, vy, vw, vh = c.value_rect
    text(sp.text_of(v, c.precision), vx + vw, vy + vh / 2, 12 * s, TEXT if lit else TEXT_DIM, "right")


def _draw_top(context) -> None:
    """The preset list, the colour picker and a slider's tooltip: a layer over the panel."""
    session = _session
    region = context.region
    if session is None or region is None or region.as_pointer() != session.region_ptr:
        return
    app = CONTEXT.app
    layout = _layout(session, region, app)
    s = layout.scale
    if layout.menu is not None:
        round_rect(layout.menu, 10 * s, PANEL_BG)
        round_outline(layout.menu, 10 * s, PANEL_EDGE, 1.0 * s)
        current = _setup().preset
        for item in layout.items:
            if item.kind != "MENU":
                continue
            lit = session.hover is not None and _key(session.hover) == _key(item)
            if lit:
                round_rect(item.rect, 6 * s, BUTTON_HOVER)
            text(item.text, item.rect[0] + 10 * s, item.rect[1] + item.rect[3] / 2, 12 * s,
                 EMBER if item.key == current else (TEXT if lit else TEXT_DIM))
    parts = _picker_parts(session, region, layout)
    if parts is not None:
        setup = _setup()
        current = light_core.srgb255(_colour(session, setup, session.picker) or (0.0, 0.0, 0.0))
        colour_picker.draw(parts, session.pick_hsv, colour_picker.quick_colours(), current, session.pick_hover,
                           "Back to the preset's", s)
        return
    hover = session.hover
    if hover is not None and hover.kind == "SLIDER_LABEL" and hover.tip \
            and kit.tooltip_due(session.hover_since, time.monotonic()):
        lines = sp.wrap(hover.tip, lambda t: text_width(t, 11 * s), 240 * s, lines=4)
        w = 256 * s
        h = (16 + 16 * len(lines)) * s
        x = layout.rect[0] - 8 * s - w
        top = hover.rect[1] + hover.rect[3]
        box = (max(4 * s, x), max(4 * s, top - h), w, h)
        round_rect(box, 8 * s, PANEL_BG)
        round_outline(box, 8 * s, PANEL_EDGE, 1.0 * s)
        for k, line in enumerate(lines):
            text(line, box[0] + 8 * s, box[1] + box[3] - (16 + 16 * k) * s, 11 * s, TEXT)


def _draw_handler(context) -> None:
    if _session is not None and context.region is not None:
        draw(context.region, CONTEXT.app)


registry.add_draw("view_panel", 765, "PIXEL", _draw_handler)
registry.add_draw("view_panel_top", 766, "PIXEL", _draw_top)
