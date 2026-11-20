"""The history strip: the undo steps as dots along the bottom of the view.

View > Show History (Cmd+L) turns it on and off; the choice is kept. A click
on a dot undoes or redoes to that step, a drag along the strip scrubs, and
the view shows the real voxels at each step. The name at the right end is
the current step's, or the one under the mouse. The geometry is
rok/core/history_strip.py.

The strip sits right of the palette card. In Animate mode, where the palette
card hides and the timeline dock fills the bottom, it sits on top of the dock.
"""

from __future__ import annotations

import time

from rok.core import history_strip as kit
from rok.core import hud

from . import object as voxel_object
from . import settings
from .gfx import pen
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, register
from .paint import clip_text, round_outline, round_rect, text
from .state import state

SETTING = "history_strip"
CARD_BG = (0.11, 0.10, 0.10, 0.86)  # the palette card's plate
CARD_EDGE = (1.0, 1.0, 1.0, 0.07)
TRACK = (1.0, 1.0, 1.0, 0.16)
DONE = (0.905, 0.885, 0.855, 1.0)
UNDONE = (0.620, 0.600, 0.575, 1.0)
EMBER = (0.910, 0.365, 0.227, 1.0)
HOVER = (1.0, 1.0, 1.0, 0.10)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)
TEXT_FAINT = (0.450, 0.430, 0.410, 1.0)
GAP = 12  # between the palette card and the strip, and above the dock

_hover: int | None = None  # the place under the mouse


def shown() -> bool:
    app = CONTEXT.app
    return bool(getattr(app, "history_strip", False)) and document.scene is not None


def toggle(app) -> None:
    """View > Show History."""
    app.history_strip = not getattr(app, "history_strip", False)
    settings.put(SETTING, "on" if app.history_strip else "off")  # it stays
    app.set_menu()
    app.request_redraw()


def _left_and_bottom(region) -> tuple[float, float]:
    from . import chrome, dock, overlay

    scale = overlay.ui_scale()
    layout = dock.layout_for(region)
    if layout is not None:  # Animate mode: on top of the dock, from its left edge
        return layout.rect[0], layout.rect[1] + layout.rect[3] + GAP * scale
    if state.animating:
        rail = chrome.layout_for(region)
        return rail.rect[0] + rail.rect[2] + GAP * scale, GAP * scale
    strip = overlay.strip_layout(region)
    pad = overlay.STRIP_PLATE_PAD * scale
    # The card's widest row of colours, so the strip stays put while a
    # scrub changes the colours the model uses.
    width = max(strip.width, hud.STRIP_USED_COLUMNS * strip.swatch)
    return strip.x0 + width + pad + GAP * scale, strip.y0 - pad


def layout_for(region) -> kit.StripLayout | None:
    if region is None or not shown():
        return None
    from . import overlay

    left, bottom = _left_and_bottom(region)
    right = float(region.width) - GAP * overlay.ui_scale()
    stack = document.undo
    return kit.strip_layout(overlay.ui_scale(), left, right, bottom, len(stack.steps()) + 1, stack.position)


def top(region) -> float | None:
    """The top of the card, for what sits above it (the hint line); None when hidden."""
    layout = layout_for(region)
    return None if layout is None else layout.rect[1] + layout.rect[3]


def hover(region, x: float, y: float) -> bool:
    """Track the mouse over the strip. True when it is inside."""
    global _hover
    layout = layout_for(region)
    inside = layout is not None and layout.contains(x, y)
    found = layout.place_at(x) if inside and layout.on_track(x, y) else None
    if found != _hover:
        _hover = found
        voxel_object.redraw_3d_views()
    return inside


def press(context, region, x: float, y: float, left: bool, shift: bool) -> bool:
    """A press on the strip: a left press on the track scrubs. Any press
    inside is eaten, so nothing behind the card is edited."""
    layout = layout_for(region)
    if layout is None or not layout.contains(x, y):
        return False
    if left and layout.on_track(x, y):
        from .opkit import ops

        ops.rok.history_scrub("INVOKE")
    return True


def go_to(position: int) -> int:
    """Undo or redo until `position` steps are done, each step as Undo and
    Redo do it. Returns the number of steps applied."""
    from .undo_ops import _after_step

    if document.scene is None or state.stroke_active:
        return 0
    return document.undo.go_to(position, document.scene, _after_step)


@register
class HistoryScrubAction(Action):
    command_id = "rok.history_scrub"
    title = "Go to Step"
    tooltip = "Click or drag along the history strip: the model goes back or forward to that step"

    @classmethod
    def poll(cls, context):
        return shown() and not state.stroke_active

    def invoke(self, context, event):
        self._go(context, event)
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def _go(self, context, event) -> None:
        layout = layout_for(context.region)
        if layout is not None:
            go_to(layout.place_at(event.mouse_region_x))

    def modal(self, context, event):
        if event.type in ("MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"):
            self._go(context, event)
            return {"RUNNING"}
        if (event.type == "LEFT_MOUSE" and event.value == "RELEASE") or event.type == "ESCAPE":
            return {"FINISHED"}
        return {"RUNNING"}


# --- drawing ------------------------------------------------------------------------


def _when(t: float | None) -> str:
    return "" if not t else time.strftime("%H:%M", time.localtime(t))


def _name(place: int, steps) -> tuple[str, str]:
    """The name and the time of a place: the start, or step `place`."""
    if place == 0:
        return ("Older steps dropped" if document.undo.dropped else "Start"), ""
    record = steps[place - 1]
    return getattr(record, "label", "") or "Change", _when(getattr(record, "when", None))


def _draw_bg(context) -> None:
    layout = layout_for(context.region)
    if layout is None:
        return
    s = layout.scale
    round_rect(layout.rect, 10 * s, CARD_BG)
    round_outline(layout.rect, 10 * s, CARD_EDGE, 1.0 * s)
    y = layout.centre_y
    x0, x1 = layout.track
    if layout.count > 1:
        pen.lines([(x0, y), (x1, y)], TRACK, 2.0 * s)
    if _hover is not None and _hover < layout.count and not layout.as_line:
        r = (kit.DOT_R + 4) * s
        hx = layout.x_of(_hover)
        round_rect((hx - r, y - r, 2 * r, 2 * r), r, HOVER)


def _draw(context) -> None:
    layout = layout_for(context.region)
    if layout is None:
        return
    s = layout.scale
    y = layout.centre_y
    steps = document.undo.steps()
    here = layout.position
    if layout.as_line:
        # Too many steps for dots: the done part bright, a tick at each step.
        xh = layout.x_of(here)
        pen.lines([(layout.track[0], y), (xh, y)], DONE, 2.0 * s)
        ticks = []
        for k in range(layout.count):
            x = layout.x_of(k)
            ticks += [(x, y - 3 * s), (x, y + 3 * s)]
        pen.lines(ticks, TRACK, 1.0 * s)
    else:
        r = kit.DOT_R * s
        for k in range(layout.count):
            if k == here:
                continue
            x = layout.x_of(k)
            if k < here:
                round_rect((x - r, y - r, 2 * r, 2 * r), r, DONE)
            else:
                round_rect((x - r, y - r, 2 * r, 2 * r), r, CARD_BG)
                round_outline((x - r, y - r, 2 * r, 2 * r), r, UNDONE, 1.2 * s)
    # The current place: an ember diamond.
    x = layout.x_of(here)
    d = (kit.DOT_R + 2.5) * s
    pen.tris([(x, y + d), (x - d, y), (x, y - d), (x, y - d), (x + d, y), (x, y + d)], EMBER)
    # The name at the right end: the step under the mouse, else the current one.
    # The hover can be stale: an edit or a key may change the steps under a resting mouse.
    place = _hover if _hover is not None and _hover < layout.count else here
    name, when = _name(place, steps)
    lx, ly, lw, lh = layout.label
    colour = TEXT if place <= here else TEXT_DIM
    when_w = 40 * s if when else 0.0
    name = clip_text(name + ("  (undone)" if place > here else ""), lw - when_w, 12 * s)
    text(name, lx + lw - when_w, ly + lh / 2, 12 * s, colour, "right")
    if when:
        text(when, lx + lw, ly + lh / 2, 11 * s, TEXT_FAINT, "right")


def _start(context) -> None:
    app = context.app
    if app is not None:
        app.history_strip = settings.get(SETTING) == "on"


registry.add_draw("history_strip_bg", 692, "PIXEL", _draw_bg)
registry.add_draw("history_strip", 693, "PIXEL", _draw)
registry.startup.append(_start)
