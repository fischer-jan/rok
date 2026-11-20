"""View operators: saved views, the WASD + QE view keys and the hold keys
(Ctrl orbits, Cmd pans, Option zooms on a plain mouse move).

Save 1..4 and Go to 1..4, plus `rok.view_key` and `rok.hold_move`.

A saved view is the camera's target, turn (yaw and pitch) and distance. It
lives in `document.views` for the running session only (not in the file):
New and Open keep it, `document.reset` clears it. Rok has one view, so the
actions act on it alone. The camera keeps yaw and pitch, and the saved
view keeps those.
"""

from __future__ import annotations

import time

from rok.core import view_keys

from .object import document
from .opkit import EnumProp, IntProp, Action, register
from .state import state

ORBIT_RAD_PER_PX = 0.0038  # Ctrl + move: radians per pixel of mouse move
ROK_ORBIT_RAD_PER_PX = 0.005  # what camera.orbit turns per pixel


def _rehover() -> None:
    from . import mode  # late: mode imports modules that import this one

    mode.rehover()


@register
class ViewSaveAction(Action):
    command_id = "rok.view_save"
    title = "Save View"
    tooltip = "Save the current view to a slot, for View > Saved Views > Go to View"
    slot = IntProp(name="Slot", description="Which of the four saved views", default=1, min=1, max=4)

    @classmethod
    def poll(cls, context):
        return document.grid is not None

    def execute(self, context):
        view_state = context.view_state
        if view_state is None:
            self.report({"WARNING"}, "No 3D viewport")
            return {"CANCELLED"}
        cam = view_state.camera
        document.views[self.slot] = (
            tuple(float(v) for v in view_state.orbit_centre),
            (float(cam.yaw), float(cam.pitch)),
            float(view_state.orbit_distance),
            "PERSP",
        )
        self.report({"INFO"}, f"View {self.slot} saved")
        return {"FINISHED"}


@register
class ViewGoAction(Action):
    command_id = "rok.view_go"
    title = "Go to View"
    tooltip = "Turn the view to a saved slot"
    slot = IntProp(name="Slot", description="Which of the four saved views", default=1, min=1, max=4)

    @classmethod
    def poll(cls, context):
        return document.grid is not None

    def execute(self, context):
        saved = document.views.get(self.slot)
        if saved is None:
            self.report({"WARNING"}, f"View {self.slot} is not saved")
            return {"CANCELLED"}
        view_state = context.view_state
        if view_state is None:
            self.report({"WARNING"}, "No 3D viewport")
            return {"CANCELLED"}
        location, (yaw, pitch), distance, _perspective = saved
        cam = view_state.camera
        cam.yaw, cam.pitch = float(yaw), float(pitch)
        view_state.orbit_centre = location
        view_state.orbit_distance = distance
        context.area.tag_redraw()
        _rehover()
        return {"FINISHED"}


def _apply_orbit(context, last, now) -> None:
    """Turntable orbit around the view centre at ORBIT_RAD_PER_PX."""
    dx = float(now[0] - last[0])
    dy = float(now[1] - last[1])
    k = ORBIT_RAD_PER_PX / ROK_ORBIT_RAD_PER_PX
    context.view_state.camera.orbit(dx * k, -dy * k)  # mouse up tilts the view down
    context.app.camera_moved()


def _apply_pan(context, last, now) -> None:
    context.view_state.camera.pan(float(now[0] - last[0]), float(now[1] - last[1]))
    context.app.camera_moved()


def _apply_zoom(context, last, now) -> None:
    import math

    from . import navigation

    context.view_state.camera.zoom(math.exp(float(now[1] - last[1]) * navigation.MOVE_ZOOM_PER_PX))  # up: closer
    context.app.camera_moved()


# The held key, the modifier flag on the event, and what a move does.
HOLD = {
    "ORBIT": (("LEFT_CTRL", "RIGHT_CTRL"), "ctrl", _apply_orbit),
    "PAN": (("CMD",), "cmd_key", _apply_pan),
    "ZOOM": (("LEFT_ALT", "RIGHT_ALT"), "alt", _apply_zoom),
}


_FLAGS = ("shift", "ctrl", "alt", "cmd_key")


def _over_card(x: float, y: float) -> bool:
    from . import overlay

    dock = getattr(overlay, "_dock", None)
    if dock is None:
        return False
    cx, cy, w, h = dock.card
    return cx <= x < cx + w and cy <= y < cy + h


def _ctrl_click_has_a_use(context, x: float, y: float) -> bool:
    """A Ctrl+left press that belongs to something: the rail or a strip under
    the mouse, a swatch on the palette card (its swap drag), a gizmo handle,
    or a model in the scene (the carve drag)."""
    if state.hover_chrome is not None or state.hover_strip is not None or _over_card(x, y):
        return True
    if context.view_state is not None:
        from .tools import StrokeAction

        if StrokeAction.handle_at(context.region, context.view_state, x, y) is not None:
            return True
    if state.level == "SCENE" and context.view_state is not None:
        from . import mode

        under, _solid = mode.models_hit(context.region, context.view_state, x, y)
        return bool(under)
    return False


@register
class HoldMoveAction(Action):
    """Hold a key and move the mouse, no button: Ctrl orbits, Cmd pans,
    Option zooms. With a second modifier held the view stays put.

    The keys are also the keys of shortcuts and clicks, so this stays out of
    their way: every other key goes through (Cmd+Z still undoes), a move with
    a button held belongs to that drag, and a click goes through to its own
    binding (Option-click erases, Cmd-drag moves a model freely). A Ctrl+left
    press with nothing of its own under it keeps orbiting. Over the rail or
    the palette card a press of the key does not start it.
    """

    command_id = "rok.hold_move"
    title = "Move View"
    tooltip = "Hold Ctrl, Cmd or Option and move the mouse: orbit, pan or zoom"
    how = EnumProp(items=tuple((k, k.title(), "") for k in HOLD), default="ORBIT", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.view_state is not None

    def invoke(self, context, event):
        if state.hover_chrome is not None or state.hover_strip is not None:
            return {"PASS"}
        from . import stroke_panel

        if stroke_panel.contains(context.region, event.mouse_region_x, event.mouse_region_y):
            return {"PASS"}
        if getattr(context.app, "buttons_down", 0):
            return {"PASS"}  # a drag runs: the key changes that drag (Option: up and down, Cmd: no snapping)
        self._keys, self._flag, self._apply = HOLD[self.how]
        self._last = (event.mouse_region_x, event.mouse_region_y)
        self._clicked = False  # a Ctrl+left press this action took
        context.prefs.add_modal(self)
        return {"RUNNING", "PASS"}

    def modal(self, context, event):
        view_state = context.view_state
        if view_state is None:
            return {"CANCELLED", "PASS"}
        if event.type == "MOUSE_MOVE":
            state.last_mouse = (event.mouse_x, event.mouse_y)
            now = (event.mouse_region_x, event.mouse_region_y)
            if not getattr(event, self._flag):
                return {"FINISHED", "PASS"}  # the release was missed
            if any(getattr(event, flag) for flag in _FLAGS if flag != self._flag):
                self._last = now  # each key moves the view only when it is the only one held
                return {"RUNNING", "PASS"}
            from . import palette_ui

            if palette_ui._drag is not None or _over_card(*now):
                self._last = now  # over the palette card the view stays put (Ctrl drags swatches there)
                return {"RUNNING", "PASS"}
            if getattr(context.app, "buttons_down", 0) and not self._clicked:
                self._last = now  # a drag of something else
                return {"RUNNING", "PASS"}
            self._apply(context, self._last, now)
            self._last = now
            return {"RUNNING"}
        if event.type in self._keys and event.value == "RELEASE":
            _rehover()
            return {"FINISHED", "PASS"}
        if event.type == "LEFT_MOUSE" and self._clicked:
            if event.value == "RELEASE":
                self._clicked = False
            return {"RUNNING"}
        if event.type in {"LEFT_MOUSE", "RIGHT_MOUSE", "MIDDLE_MOUSE"} and event.value == "PRESS":
            x, y = event.mouse_region_x, event.mouse_region_y
            if (self.how == "ORBIT" and event.type == "LEFT_MOUSE" and not event.shift and not event.alt
                    and not _ctrl_click_has_a_use(context, x, y)):
                self._clicked = True  # nothing of its own under it: the orbit goes on
                return {"RUNNING"}
            return {"FINISHED", "PASS"}  # the click belongs to its own binding
        if event.type == "FOCUS_LOST":
            return {"FINISHED", "PASS"}
        return {"RUNNING", "PASS"}  # every key goes through: Cmd+Z, Ctrl+C, ...


@register
class ViewKeyAction(Action):
    """WASD + QE (rok/core/view_keys.py): A/D orbit, W/S tilt, Q/E zoom.
    Holding the key moves the view smoothly until it comes up; a tap moves
    one step. Every other event goes through, so a click still works while
    the view turns."""

    command_id = "rok.view_key"
    title = "Move View"
    tooltip = "A/D orbit, W/S tilt, Q/E zoom; hold to keep moving"
    key = EnumProp(items=tuple((k, k, "") for k in view_keys.VIEW_KEYS), options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        from . import cad_mode
        from .ops_common import _late

        command_active = _late("command", "active")()
        return (
            context.view_state is not None
            and not state.stroke_active
            and state.size_edit is None
            and not cad_mode.editing()
            and not command_active
        )

    def invoke(self, context, event):
        self._kind, self._sign = view_keys.VIEW_KEYS[self.key]
        self._start = time.monotonic()
        self._held = 0.0
        self._view_state = context.view_state
        self._timer = context.prefs.add_timer(1.0 / 60.0, window=context.window)
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def _move(self, amount: float) -> None:
        """Turn, tilt or zoom by `amount`: degrees, or a zoom factor above 1."""
        view_state = self._view_state
        if self._kind == view_keys.ZOOM:
            factor = amount if self._sign < 0 else 1.0 / amount
            view_state.orbit_distance = view_keys.clamp_distance(view_state.orbit_distance * factor)
        elif self._kind == view_keys.ORBIT:
            # The camera goes round world Z: right is a turn like a mouse drag to the left.
            view_state.camera.turn_degrees(yaw_deg=self._sign * amount)
            view_state._moved()
        else:
            # Up: the camera rises. turn_degrees keeps it short of straight above.
            view_state.camera.turn_degrees(pitch_deg=self._sign * amount)
            view_state._moved()

    def _step(self, held: float) -> None:
        if self._kind == view_keys.ZOOM:
            self._move(view_keys.zoom_between(self._held, held))
        else:
            self._move(view_keys.degrees_between(self._kind, self._held, held))
        self._held = held

    def _end(self, context) -> set:
        context.prefs.remove_timer(self._timer)
        _rehover()
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type == "TIMER":
            self._step(time.monotonic() - self._start)
            return {"RUNNING"}
        if event.type == self.key:
            if event.value == "RELEASE":
                held = time.monotonic() - self._start
                self._step(held)
                if held < view_keys.TAP_S:
                    rest = view_keys.tap_rest(self._kind, held)
                    if (self._kind == view_keys.ZOOM and rest > 1.0) or (self._kind != view_keys.ZOOM and rest > 0):
                        self._move(rest)
                return self._end(context)
            return {"RUNNING"}  # the key's own repeats
        if event.type == "FOCUS_LOST":
            return self._end(context)
        return {"PASS"}
