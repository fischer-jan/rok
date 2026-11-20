"""How the mouse moves the view, and Rok > Settings.

With a mouse, a held key and a plain move (no button) move the view: Ctrl
orbits, Cmd pans, Option zooms (view_ops.HoldMoveAction); the wheel zooms.
The trackpad does not change: two fingers orbit, Ctrl + two fingers pan,
Option + two fingers and a pinch zoom (`RokApp.navigate`). The middle
button is never used.

Rok > Settings holds the wheel's direction, Steady Hand and the questions
Rok asks.
"""

from __future__ import annotations

import math

from rok.core import steady_hand

from . import steady_ops
from .handlers import registry
from .opkit import BoolProp, IntProp, Action, register

WHEEL_ZOOM_PER_PX = 0.01
MOVE_ZOOM_PER_PX = 0.005  # Option + move: up moves closer; 150 px about doubles
MODS = ("shift", "ctrl", "alt", "cmd_key")
HOLD_KEYS = "Ctrl + move: orbit · Cmd + move: pan · Option + move: zoom · the wheel zooms"

_invert_wheel = False


def load() -> None:
    from . import settings

    global _invert_wheel
    _invert_wheel = settings.get("nav_invert_wheel") == "on"


def wheel_zoom(ev) -> float | None:
    """The zoom factor of a wheel event with no key held, else None."""
    if any(getattr(ev, m) for m in MODS):
        return None
    sign = -1.0 if _invert_wheel else 1.0
    return math.exp(sign * float(ev.dy) * WHEEL_ZOOM_PER_PX)


def _start(context) -> None:
    load()


registry.startup.append(_start)


# --- the dialog ----------------------------------------------------------------------


@register
class PreferencesAction(Action):
    command_id = "rok.preferences"
    title = "Settings"
    tooltip = "The wheel, Steady Hand and the questions Rok asks"

    invert_wheel = BoolProp(name="Invert Wheel", description="The wheel zooms the other way", default=False)
    steady_on = BoolProp(name="Steady Hand", description="Voxel strokes: the brush trails the mouse on a string",
                         default=False)
    steady_length = IntProp(
        name="String Length",
        description="How far the mouse can move before it pulls the brush, in pixels",
        default=steady_hand.DEFAULT_LENGTH, min=steady_hand.MIN_LENGTH, max=steady_hand.MAX_LENGTH,
    )
    ask_unique = BoolProp(name="Ask before Make Unique",
                          description="A click on a linked copy's chain asks before the copy stops sharing its shape",
                          default=True)

    def invoke(self, context, event):
        from . import scene_ops, settings

        self.invert_wheel = _invert_wheel
        self.steady_on = bool(context.prefs.rok_steady_hand)
        self.steady_length = int(context.prefs.rok_steady_length)
        self.ask_unique = settings.get(scene_ops.ASK_MAKE_UNIQUE, "on") != "off"
        return context.prefs.open_dialog(self, width=560, title="Settings", confirm_text="Apply")

    def draw(self, context):
        layout = self.layout
        mouse = layout.card("Mouse")
        mouse.label(text=HOLD_KEYS)
        mouse.prop(self, "invert_wheel", toggle=True, text="Invert the wheel")
        mouse.label(text="The trackpad stays: two fingers orbit, Ctrl pans, Option or a pinch zooms.", small=True)
        drawing = layout.card("Drawing")
        drawing.prop(self, "steady_on", toggle=True, text="Steady Hand  (key H)")
        drawing.prop(self, "steady_length", slider=True, suffix=" px")
        drawing.label(text="The brush trails the mouse on a string. A longer string smooths more.", small=True)
        questions = layout.card("Questions")
        questions.prop(self, "ask_unique", toggle=True, text="Ask before Make Unique")
        questions.label(text="A linked copy that turns unique stops sharing its shape with its copies.", small=True)

    def execute(self, context):
        from . import scene_ops, settings

        global _invert_wheel
        settings.put("nav_invert_wheel", "on" if self.invert_wheel else "off")
        _invert_wheel = bool(self.invert_wheel)
        context.prefs.rok_steady_hand = bool(self.steady_on)
        context.prefs.rok_steady_length = int(self.steady_length)
        settings.put(scene_ops.ASK_MAKE_UNIQUE, "on" if self.ask_unique else "off")
        steady_ops.save(context.prefs)
        return {"FINISHED"}
