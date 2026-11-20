"""Paint's Gradient mode: the settings, the ramp's palette slots, and the end colour's picker.

A drag paints a ramp from the current colour to the end colour across the
model (rok/core/gradient.py, GradientGesture). The rail shows three chips in
this mode: how many colours (click for one more, Shift+click for one less),
the end colour (a click opens a picker beside it) and Dither. The three are
yours, in every file (settings.txt).

The ramp's colours go into the palette when the drag starts: a colour the
palette has already is used again, a new one gets a free slot. The palette
change and the voxels are one undo step.
"""

from __future__ import annotations

import numpy as np

from rok.core import chrome as kit
from rok.core import gradient

from . import colour_picker, settings
from . import object as voxel_object
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, register
from .paint import round_outline, round_rect

STEPS_KEY, END_KEY, DITHER_KEY = "gradient_steps", "gradient_end", "gradient_dither"
HINT = "Gradient: drag across the model from the current colour to the end colour"
FULL = "Palette full: no room for the gradient's colours"


def steps() -> int:
    try:
        n = int(settings.get(STEPS_KEY) or gradient.DEFAULT_STEPS)
    except ValueError:
        n = gradient.DEFAULT_STEPS
    return max(gradient.MIN_STEPS, min(gradient.MAX_STEPS, n))


def step(by: int) -> None:
    """One more (or fewer) colour; past the end it wraps round."""
    n = steps() + int(by)
    span = gradient.MAX_STEPS - gradient.MIN_STEPS + 1
    n = gradient.MIN_STEPS + (n - gradient.MIN_STEPS) % span
    settings.put(STEPS_KEY, str(n))
    voxel_object.redraw_3d_views()


def dither() -> bool:
    return settings.get(DITHER_KEY) == "on"


def toggle_dither() -> None:
    settings.put(DITHER_KEY, "off" if dither() else "on")
    voxel_object.redraw_3d_views()


def current_rgb() -> tuple[int, int, int]:
    scene = document.scene
    if scene is None:
        return (255, 255, 255)
    return tuple(int(c) for c in scene.palette[int(document.colour), :3])


def end_rgb() -> tuple[int, int, int]:
    """The end colour; until you pick one, a darker shade of the current colour."""
    text = (settings.get(END_KEY) or "").lstrip("#")
    if len(text) == 6:
        try:
            return tuple(int(text[k:k + 2], 16) for k in (0, 2, 4))
        except ValueError:
            pass
    return gradient.darker(current_rgb(), 0.5)


def set_end(rgb) -> None:
    settings.put(END_KEY, "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb)))
    voxel_object.redraw_3d_views()


def slots_for(colours) -> list[int] | None:
    """A palette slot for every colour of the ramp, or None when the palette is full."""
    store = document.scene.colours
    used = document.used_counts()
    out = []
    after = store.id_of(int(document.colour))
    for rgb in colours:
        rgba = (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)
        ident = store.find(rgba)
        if ident is None:
            ident = store.add(rgba, after=after)
        after = ident
        slot = store.lease(ident, used, pinned=document.pinned())
        if slot is None:
            return None
        out.append(int(slot))
    return out


def gesture(voxels, set_many):
    """(GradientGesture, palette snapshot) for a drag, or (None, None) when
    nothing can be painted or the palette is full (it says so)."""
    from rok.core.gesture import GradientGesture

    if document.scene is None:
        return None, None
    cells = np.argwhere(voxels != 0)
    selection = document.selection
    if selection is not None and not selection.is_empty():
        cells = cells[selection.mask[cells[:, 0], cells[:, 1], cells[:, 2]]]
    if len(cells) == 0:
        return None, None
    snapshot = document.scene.colours.snapshot()
    slots = slots_for(gradient.ramp(current_rgb(), end_rgb(), steps()))
    if slots is None:
        document.scene.colours.restore(snapshot)
        if CONTEXT.app is not None:
            CONTEXT.app.say(FULL, 4.0)
        return None, None
    return GradientGesture(voxels, cells, slots, set_many, dither()), snapshot


# --- the end colour's picker --------------------------------------------------------------

_picker: dict | None = None  # {"rect": the chip, "hsv", "drag", "hover"} while the picker is open


def picker_open() -> bool:
    return _picker is not None


def _parts(region) -> dict | None:
    if _picker is None or region is None:
        return None
    return colour_picker.layout(_picker["rect"], region, 1.0, len(colour_picker.quick_colours()))


@register
class GradientEndAction(Action):
    command_id = "rok.gradient_end"
    title = "Gradient End Colour"
    tooltip = "Pick the colour Paint's gradient ends in"

    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def invoke(self, context, event):
        global _picker
        region = context.region
        if region is None:
            return {"CANCELLED"}
        widget = kit_layout_widget(region)
        rect = widget.rect if widget is not None else (60.0, region.height / 2, 40.0, 40.0)
        # Beside the chip, past the rail: the picker opens to the right of it.
        rect = (rect[0] + rect[2] + 130.0, rect[1], rect[2], rect[3])
        _picker = {"rect": rect, "hsv": colour_picker.hsv_of(end_rgb()), "drag": None, "hover": None}
        context.prefs.add_modal(self)
        voxel_object.redraw_3d_views()
        return {"RUNNING"}

    def modal(self, context, event):
        global _picker
        if _picker is None:
            return {"CANCELLED"}
        parts = _parts(context.region)
        if parts is None:
            _picker = None
            return {"CANCELLED"}
        mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
        kind, value = event.type, event.value
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            if _picker["drag"] is not None:
                _picker["hsv"] = colour_picker.dragged(_picker["hsv"], _picker["drag"], parts[_picker["drag"]], mx, my)
                set_end(colour_picker.rgb_of(_picker["hsv"]))
            else:
                _picker["hover"] = colour_picker.part_at(parts, mx, my)
            voxel_object.redraw_3d_views()
            return {"RUNNING"}
        if kind == "LEFT_MOUSE" and value == "RELEASE":
            _picker["drag"] = None
            return {"RUNNING"}
        if kind == "LEFT_MOUSE" and value == "PRESS":
            if not colour_picker._inside(parts["PANEL"], mx, my):
                return self._close()
            part = colour_picker.part_at(parts, mx, my)
            if part in ("SV", "HUE"):
                _picker["drag"] = part
                _picker["hsv"] = colour_picker.dragged(_picker["hsv"], part, parts[part], mx, my)
                set_end(colour_picker.rgb_of(_picker["hsv"]))
            elif part is not None and part.startswith("Q:"):
                quick = colour_picker.quick_colours()
                k = int(part[2:])
                if k < len(quick):
                    set_end(quick[k])
                    _picker["hsv"] = colour_picker.hsv_of(quick[k])
            elif part == "RESET":
                shade = gradient.darker(current_rgb(), 0.5)
                set_end(shade)
                _picker["hsv"] = colour_picker.hsv_of(shade)
            voxel_object.redraw_3d_views()
            return {"RUNNING"}
        if value == "PRESS" and kind in {"ESCAPE", "ENTER", "NUMPAD_ENTER", "RIGHT_MOUSE"}:
            return self._close()
        return {"RUNNING"}

    def cancel(self, context):
        self._close()

    @staticmethod
    def _close():
        global _picker
        _picker = None
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


def kit_layout_widget(region):
    from . import chrome

    return chrome.layout_for(region).by_id(kit.GRADIENT_END_ID)


def draw_end_swatch(rect, scale: float, hovered: bool) -> None:
    """The end colour's chip: a swatch of the colour, ringed under the mouse."""
    x, y, w, h = rect
    inset = 7.0 * scale
    swatch = (x + inset, y + inset, w - 2 * inset, h - 2 * inset)
    round_rect(swatch, 6.0 * scale, tuple(c / 255.0 for c in end_rgb()) + (1.0,))
    round_outline(swatch, 6.0 * scale, (1.0, 1.0, 1.0, 0.8 if hovered else 0.3), 1.2 * scale)


def _draw_picker(context) -> None:
    parts = _parts(context.region)
    if parts is None:
        return
    colour_picker.draw(parts, _picker["hsv"], colour_picker.quick_colours(), end_rgb(), _picker["hover"],
                       "A darker shade of the current colour", 1.0)


registry.add_draw("gradient_end_picker", 770, "PIXEL", _draw_picker)
