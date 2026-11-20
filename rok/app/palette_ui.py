"""The colour popup, its window-manager settings, `+`, the eyedropper and the
palette drag (swap and gradient).

The picker edits the current palette slot in place. Every voxel of that index
in every model takes the new colour. One popup session (one
`rok.colour_popup` call) is one undo step: `after_palette_edit` pushes a
PaletteRecord with merge=True, so edits of the same slot and session fold
into the top record. The material type and its sliders write the slot's
Material through the same path, so colour and material edits share that step.

Rok draws the popup itself: a
saturation/value square with a hue bar, the colour's hex (typed as printable
ASCII), how many voxels use it, the material type and the type's sliders. All
of them read and write window-manager settings
(`rok_colour`, `rok_hex`, `rok_material_type`, `rok_metal`, ...).

`rok.palette_drag` is started by `tools._strip_press` when Ctrl or Shift is
down on a swatch of the strip's used row or the 256 grid. It rides the hover
operator for the swatch under the mouse (`state.hover_strip`) instead of
tracking the mouse itself, so it stays a plain pass-through modal: Ctrl drags a
swap, Shift a gradient, Esc or a release off any swatch cancels.

Animation is parked: there is no "Record" path that keys a colour track.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

from rok.core import hud
from rok.core.edit import ACTIVATE
from rok.core.material import DEFAULT, Material

from . import dialog as look
from . import object as voxel_object
from .app import RokPrefs
from .gfx import fonts, pen
from .handlers import registry
from .object import document
from .opkit import (
    CONTEXT,
    BoolProp,
    EnumProp,
    FloatProp,
    FloatVectorProp,
    IntProp,
    Action,
    StringProp,
    register,
)
from .paint import round_outline, round_rect
from .state import state

PANEL_NAME = "ROK_PT_colour"
PALETTE_FULL = "Palette full: all 255 colours are in use"
NEW_RGB = (255, 255, 255)  # the + button's new colour: white, Plain
EYEDROP_HINT = "Click a voxel to pick its colour   Esc: cancel"

# The palette before the edit that after_palette_edit is about to record.
_before: np.ndarray | None = None
# The material table before that edit; None when the edit left the table alone.
_materials_before = None
# True while rok.eyedrop runs modal; the hint line shows EYEDROP_HINT then.
_eyedrop_active = False

# Events the eyedropper lets through to the hover operator and the navigation.
_PASS_ON = {
    "MOUSE_MOVE", "MOUSE_MOVE_BETWEEN", "MIDDLE_MOUSE", "WHEEL_UP", "WHEEL_DOWN",
    "WHEELINMOUSE", "WHEELOUTMOUSE", "TRACKPAD_PAN", "TRACKPAD_ZOOM", "MOUSEPAN", "MOUSEZOOM",
    "MOUSEROTATE", "LEFT_CTRL", "RIGHT_CTRL", "LEFT_SHIFT", "RIGHT_SHIFT", "LEFT_ALT", "RIGHT_ALT", "CMD",
}

# The material types.
_TYPE_ITEMS = (
    ("DIFFUSE", "Plain", "A matte surface, no reflection"),
    ("METAL", "Metal", "Reflects light and the surroundings"),
    ("GLASS", "Glass", "Lets light through"),
    ("EMIT", "Light", "Glows and lights its surroundings"),
    ("PLASTIC", "Plastic", "Matte with a shiny coat"),
)
_OTHER_ITEM = ("OTHER", "Other", "A material of the .vox file that Rok does not edit. It is kept as it is")
_TYPE_ITEMS_OTHER = _TYPE_ITEMS + (_OTHER_ITEM,)
_TYPE_IDS = tuple(item[0] for item in _TYPE_ITEMS_OTHER)

TYPE_DESCRIPTION = (
    "How this colour looks in a render. Plain is a matte surface. Metal reflects. "
    "Glass lets light through. Light glows and lights its surroundings. Plastic is "
    "matte with a shiny coat. The view shows only the colour"
)

# The sliders each type shows. Plain and Other show none.
SLIDERS = {
    "METAL": ("rok_metal", "rok_roughness"),
    "GLASS": ("rok_transparency", "rok_roughness"),
    "EMIT": ("rok_glow", "rok_power"),
    "PLASTIC": ("rok_gloss", "rok_roughness"),
}


def status_hint() -> str | None:
    """The hint line while the eyedropper is modal, else None."""
    return EYEDROP_HINT if _eyedrop_active else None


# --- the palette edit ---------------------------------------------------------------


def after_palette_edit(index: int | None) -> None:
    """Record and show a change to the colours: upload, undo, modified, redraw.
    The edits of one popup session fold into one undo step."""
    if document.scene is None:
        return
    voxel_object.update_palette()
    document.undo.push_colours(session=document.palette_session)
    document.modified = True
    if index is not None:
        document.made.add(int(index))  # a colour you made: the strip shows it until it is used
    voxel_object.redraw_3d_views()


def current_entry() -> int | None:
    """The colour store entry of the colour you paint with. A slot that holds
    no entry yet (a default colour) becomes one."""
    scene = document.scene
    if scene is None or document.colour < 1:
        return None
    store = scene.colours
    ident = store.id_of(document.colour)
    if ident is None:
        slot = document.colour
        ident = store.claim(slot, scene.palette[slot], scene.materials.get(slot))
    return ident


def _write_rgb(rgb) -> None:
    """Give the current colour new (r, g, b) bytes when they differ: every voxel of it changes."""
    ident = current_entry()
    if ident is None:
        return
    store = document.scene.colours
    entry = store.entries[ident]
    rgb = tuple(int(c) for c in rgb)
    if entry.rgba[:3] == rgb:
        return
    store.edit(ident, (*rgb, entry.rgba[3]))
    after_palette_edit(document.colour)


def _get_colour(self):
    scene = document.scene
    if scene is None:
        return (0.0, 0.0, 0.0)
    r, g, b = scene.palette[document.colour, :3]
    return (r / 255.0, g / 255.0, b / 255.0)


def _set_colour(self, value):
    rgb = np.clip(np.round(np.asarray(value[:3], dtype=np.float64) * 255.0), 0, 255)
    _write_rgb(rgb.astype(np.uint8))


def parse_hex(text: str) -> tuple[int, int, int] | None:
    """'RRGGBB' or '#RRGGBB', any case, to bytes. None for anything else."""
    text = text.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        value = int(text, 16)
    except ValueError:
        return None
    return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


def _get_hex(self):
    scene = document.scene
    if scene is None:
        return ""
    r, g, b = (int(c) for c in scene.palette[document.colour, :3])
    return f"#{r:02X}{g:02X}{b:02X}"


def _set_hex(self, text):
    rgb = parse_hex(text)
    if rgb is not None:
        _write_rgb(rgb)


def _on_grid_toggle(self, context):
    voxel_object.redraw_3d_views()


# --- the material edit --------------------------------------------------------------


def _current_material() -> Material:
    scene = document.scene
    if scene is None:
        return DEFAULT
    return scene.materials.get(document.colour)


def _write_material(material: Material) -> None:
    """Give the current colour `material` when it differs. One path for every setter."""
    ident = current_entry()
    if ident is None:
        return
    store = document.scene.colours
    if store.entries[ident].material == material:
        return
    store.edit(ident, material=material)
    after_palette_edit(document.colour)


def _type_items(self, context):
    """Other is on offer only while the slot is Other: it can be left, never chosen."""
    return _TYPE_ITEMS_OTHER if _current_material().type == "OTHER" else _TYPE_ITEMS


def _get_type(self):
    return _current_material().type


def _set_type(self, value):
    if value not in _TYPE_IDS[:len(_TYPE_ITEMS)]:
        return
    # Re-selecting the slot's own type is a no-op. with_type("EMIT") would give a
    # Light at Power 0 the default power, so the check comes before it.
    current = _current_material()
    if value != current.type:
        _write_material(current.with_type(value))


def _get_weight(self):
    return _current_material().weight


def _set_weight(self, value):
    # Two decimals: the text a .vox file stores, and what the getter reads back.
    _write_material(_current_material().with_weight(round(float(value), 2)))


def _get_rough(self):
    return _current_material().rough


def _set_rough(self, value):
    _write_material(_current_material().with_value(b"_rough", round(float(value), 2)))


def _get_flux(self):
    return _current_material().flux


def _set_flux(self, value):
    _write_material(_current_material().with_value(b"_flux", int(value)))


# --- the popup ------------------------------------------------------------------------

POPUP_W = 240  # logical pixels
PAD = 10
SV_H = 150
HUE_W = 16
GAP = 8
ROW = 24
LABEL_H = 18
TYPE_COLUMNS = 3
FONT = 12
SV_STEPS = 16
HUE_STEPS = 24
MARKER_DARK = (0.0, 0.0, 0.0, 0.8)
MARKER_LIGHT = (1.0, 1.0, 1.0, 1.0)
SLIDER_FILL = (0.91, 0.36, 0.23, 0.55)
TEXT_MARK = (0.91, 0.36, 0.23, 0.45)  # typed-over text in a field
TIP_BG = (0.06, 0.055, 0.05, 0.96)
FAV_MAX = 64  # favourite colours kept, newest first; the palette card shows them all
FAV_POPUP = 16  # the colour popup shows the newest ones in two rows
FAV_PER_ROW = 8
STAR_ON = (0.96, 0.70, 0.24, 1.0)  # amber, as the Paint family
STAR_OFF = (0.62, 0.60, 0.575, 1.0)


# --- favourite colours ------------------------------------------------
#
# The star in the popup marks the current colour as a favourite. A favourite is
# an RGB colour, not a palette slot: it stays between models, files and
# sessions (settings.txt, "favourite_colours"). The popup shows them in a row;
# a click uses the slot that holds that colour already, or puts it into a free
# slot.


# --- the last colours you drew with (the star shows three) ------

_recent: list[int] = []  # palette slots, newest first


def note_used(index: int) -> None:
    """A stroke or Paint Selection drew with colour `index`."""
    index = int(index)
    if index < 1:
        return
    if index in _recent:
        _recent.remove(index)
    _recent.insert(0, index)
    del _recent[8:]  # the palette dock's collapsed row shows 8


def recent_colours(count: int = 3) -> list[int]:
    """The last `count` colours drawn with (3, at most 8), newest first, still in the palette."""
    scene = document.scene
    if scene is None:
        return []
    return [i for i in _recent if int(scene.palette[i, 3]) > 0][:count]


def _forget_recent(_context) -> None:
    _recent.clear()  # New and Open: the slots are another file's


registry.load_post.append(_forget_recent)


def favourites() -> list[tuple[int, int, int]]:
    from . import settings

    out = []
    for h in settings.get("favourite_colours").split(","):
        rgb = parse_hex(h)
        if rgb is not None and rgb not in out:
            out.append(rgb)
    return out[:FAV_MAX]


def _save_favourites(favs) -> None:
    from . import settings

    settings.put("favourite_colours", ",".join(f"{r:02x}{g:02x}{b:02x}" for r, g, b in favs[:FAV_MAX]))


def _hex(rgb) -> str:
    return "{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def favourite_names() -> dict[str, str]:
    """hex -> the name you gave the favourite (settings.txt, "favourite_names")."""
    import json

    from . import settings

    try:
        names = json.loads(settings.get("favourite_names") or "{}")
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {}


def favourite_name(rgb) -> str:
    return favourite_names().get(_hex(rgb), "")


def set_favourite_name(rgb, name: str) -> None:
    import json

    from . import settings

    names = favourite_names()
    name = " ".join(str(name).split())[:40]  # one line, not too long
    if name:
        names[_hex(rgb)] = name
    else:
        names.pop(_hex(rgb), None)
    settings.put("favourite_names", json.dumps(names, ensure_ascii=False))
    voxel_object.redraw_3d_views()


def add_favourites(pairs) -> int:
    """Add (rgb, name) pairs to the favourites: the new colours go to the
    front in the order given, a colour that is a favourite already keeps its
    place, and a colour gets its name when it has none. Returns how many were
    new."""
    favs = favourites()
    names = favourite_names()
    fresh = []
    for rgb, _name in pairs:
        rgb = tuple(int(c) for c in rgb[:3])
        if rgb not in favs and rgb not in fresh:
            fresh.append(rgb)
    new = len(fresh)
    favs = fresh + favs
    _save_favourites(favs)
    for rgb, name in pairs:
        rgb = tuple(int(c) for c in rgb[:3])
        if name and not names.get(_hex(rgb)):
            set_favourite_name(rgb, str(name))
    return new


def toggle_favourite() -> bool:
    """Mark the current colour as a favourite, or unmark it. True when it is one now."""
    rgb = _rgb_bytes()
    favs = favourites()
    if rgb in favs:
        favs.remove(rgb)
        set_favourite_name(rgb, "")  # its name goes with it
        on = False
    else:
        favs.insert(0, rgb)
        on = True
    _save_favourites(favs)
    voxel_object.redraw_3d_views()
    return on


def pick_entry(ident: int | None) -> int | None:
    """Make colour `ident` (an unused one, maybe without a slot) the colour you
    paint with. Returns its slot; None and a note when every slot holds voxels."""
    scene = document.scene
    if scene is None or ident is None or ident not in scene.colours.entries:
        return None
    slot = scene.colours.lease(ident, document.used_counts(), pinned=document.pinned())
    if slot is None:
        if CONTEXT.app is not None:
            CONTEXT.app.say(PALETTE_FULL, 3.0)
        return None
    voxel_object.update_palette()
    return slot


def use_favourite(rgb) -> int | None:
    """Make `rgb` the current colour: the colour that is exactly it (opaque,
    plain), else a new one. None when every slot holds voxels."""
    scene = document.scene
    if scene is None:
        return None
    store = scene.colours
    want = (*(int(c) for c in rgb), 255)
    ident = store.find(want)
    document.palette_session += 1
    if ident is None:
        ident = store.add(want, after=store.id_of(document.colour))
    index = store.lease(ident, document.used_counts(), pinned=document.pinned())
    if index is None:
        store.remove([ident], document.used_counts())
        return None
    after_palette_edit(index)
    document.colour = index
    voxel_object.redraw_3d_views()
    return index


@dataclass
class _Popup:
    """The open popup: where it is and what the mouse and the keys do in it."""

    x: float
    top: float
    hsv: tuple[float, float, float]
    seen: tuple[int, int, int]  # the colour bytes `hsv` was taken from
    drag: str | None = None
    hover: str | None = None
    editing: bool = False  # the hex field takes the keys
    text: str = ""
    fresh: bool = False  # the whole typed text is selected: the first key replaces it
    docked: bool = False  # inline on the palette card: its bottom sits on the card's top
    base: float = 0.0  # the card header's top edge: the editor's bottom
    width: float = POPUP_W


_popup: _Popup | None = None


def popup_open() -> bool:
    return _popup is not None


def _rgb_bytes() -> tuple[int, int, int]:
    scene = document.scene
    if scene is None:
        return (0, 0, 0)
    r, g, b = (int(c) for c in scene.palette[document.colour, :3])
    return (r, g, b)


def _hsv_bytes(hsv) -> tuple[int, int, int]:
    rgb = colorsys.hsv_to_rgb(*hsv)
    return tuple(int(v) for v in np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255))


def _sync_hsv(popup: _Popup) -> None:
    """Take the hue, saturation and value from the colour when it changed
    elsewhere (the hex field, an undo). A grey keeps its hue."""
    rgb = _rgb_bytes()
    if rgb == popup.seen or rgb == _hsv_bytes(popup.hsv):
        popup.seen = rgb
        return
    h, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
    if s == 0.0 or v == 0.0:
        h = popup.hsv[0]
    popup.hsv = (h, s, v)
    popup.seen = rgb


def popup_layout(x: float, top: float, kind: str, other: bool = False, favs: int = 0,
                 width: float = POPUP_W, hex_rect=None) -> dict:
    """The parts of the popup in region pixels (origin bottom left): "PANEL",
    "SV", "HUE", "SWATCH", "HEX", "STAR", "INFO", "FAV_LABEL", "FAV:<i>",
    "TYPE_LABEL", "TYPE:<id>" and "SLIDER:<setting>", each an (x, y, w, h)
    rect. `kind` is the slot's material type; `other` adds the Other button;
    `favs` favourite colours get a row of swatches."""
    parts: dict[str, tuple] = {}
    inner = width - 2 * PAD
    left = x + PAD
    y = top - PAD
    sv_w = inner - HUE_W - GAP
    parts["SV"] = (left, y - SV_H, sv_w, SV_H)
    parts["HUE"] = (left + sv_w + GAP, y - SV_H, HUE_W, SV_H)
    y -= SV_H + GAP
    if hex_rect is not None:
        # Inside the palette card: the card's header is the chip, the hex
        # (`hex_rect`, its first line, edits in place), the star and the count.
        parts["HEX"] = tuple(hex_rect)
    else:
        parts["SWATCH"] = (left, y - ROW, 44, ROW)
        parts["HEX"] = (left + 44 + GAP, y - ROW, inner - 44 - 2 * GAP - ROW, ROW)
        parts["STAR"] = (left + inner - ROW, y - ROW, ROW, ROW)
        y -= ROW + GAP
        parts["INFO"] = (left, y - LABEL_H, inner, LABEL_H)
        y -= LABEL_H + GAP
    favs = max(0, min(int(favs), FAV_POPUP))
    if favs:
        parts["FAV_LABEL"] = (left, y - LABEL_H, inner, LABEL_H)
        y -= LABEL_H + 2
        side = (inner - 4 * (FAV_PER_ROW - 1)) / FAV_PER_ROW
        for i in range(favs):
            r, c = divmod(i, FAV_PER_ROW)
            parts[f"FAV:{i}"] = (left + c * (side + 4), y - (r + 1) * side - r * 4, side, side)
        rows = -(-favs // FAV_PER_ROW)
        y -= rows * side + (rows - 1) * 4 + GAP
    parts["TYPE_LABEL"] = (left, y - LABEL_H, inner, LABEL_H)
    y -= LABEL_H + 2
    items = _TYPE_ITEMS_OTHER if other else _TYPE_ITEMS
    bw = (inner - 4 * (TYPE_COLUMNS - 1)) / TYPE_COLUMNS
    for i, (ident, _label, _desc) in enumerate(items):
        r, c = divmod(i, TYPE_COLUMNS)
        parts[f"TYPE:{ident}"] = (left + c * (bw + 4), y - (r + 1) * ROW - r * 4, bw, ROW)
    rows = -(-len(items) // TYPE_COLUMNS)
    y -= rows * ROW + (rows - 1) * 4  # y is the bottom of the last part from here on
    for k, name in enumerate(SLIDERS.get(kind, ())):
        y -= GAP if k == 0 else 4
        parts[f"SLIDER:{name}"] = (left, y - ROW, inner, ROW)
        y -= ROW
    bottom = y - PAD
    parts["PANEL"] = (x, bottom, width, top - bottom)
    return parts


def _current_layout(popup: _Popup) -> dict:
    kind = _current_material().type
    other, favs = kind == "OTHER", len(favourites())
    if popup.docked:
        # Inside the card: the editor unfolds upward from the header at the
        # card's width (its own PAD makes the card's inner width), so its
        # bottom stays put when a type adds sliders.
        height = popup_layout(0.0, 0.0, kind, other=other, favs=favs, width=popup.width, hex_rect=(0, 0, 1, 1))["PANEL"][3]
        from . import overlay

        # The header moves when the sections give up rows to the editor: follow it.
        dock = overlay._dock
        if dock is not None and dock.editor is not None:
            popup.base = float(dock.editor[1])
        popup.top = popup.base + height
        return popup_layout(popup.x, popup.top, kind, other=other, favs=favs, width=popup.width,
                            hex_rect=dock_hex_rect())
    return popup_layout(popup.x, popup.top, kind, other=other, favs=favs)


def docked_editor_height() -> int:
    """Height of the open editor inside the palette card, 0 when it is closed or floats."""
    popup = _popup
    if popup is None or not popup.docked or document.scene is None:
        return 0
    kind = _current_material().type
    return int(round(popup_layout(0.0, 0.0, kind, other=kind == "OTHER", favs=len(favourites()),
                                  width=popup.width, hex_rect=(0, 0, 1, 1))["PANEL"][3]))


def _in_dock(px: float, py: float) -> bool:
    from . import overlay

    dock = getattr(overlay, "_dock", None)
    return dock is not None and _inside(dock.card, px, py)


def dock_hex_rect():
    """The palette card header's first line, where the hex is edited while the editor is open."""
    from . import overlay

    dock = getattr(overlay, "_dock", None)
    if dock is None:
        return (0, 0, 1, 1)
    x, y, w, h = dock.label
    return (x - 4, y, w + 4, h)  # line 1 of the header, its full width


def _inside(rect, px: float, py: float) -> bool:
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def _part_at(popup: _Popup, px: float, py: float) -> str | None:
    for name, rect in _current_layout(popup).items():
        if name in ("PANEL", "INFO", "TYPE_LABEL", "FAV_LABEL"):
            continue
        if _inside(rect, px, py):
            return name
    return None


def dock_editor_anchor(height: float, region_height: float):
    """(x, top) of the colour editor inline on the palette card: left-aligned
    with it, unfolding upward from the card's top edge. None without a card."""
    from . import overlay

    dock = getattr(overlay, "_dock", None)
    if dock is None:
        return None
    x, y, _w, h = dock.card
    top = min(float(y + h) + height, float(region_height) - 4.0)
    return float(x), top


def _open_popup(context) -> None:
    """Open the colour popup above the mouse."""
    global _popup
    if document.scene is None or context.region is None:
        return
    region = context.region
    event = context.app.last_event if context.app is not None else None
    mx = float(event.mouse_region_x) if event is not None else region.width / 2.0
    my = float(event.mouse_region_y) if event is not None else region.height / 2.0
    rgb = _rgb_bytes()
    hsv = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
    height = popup_layout(0.0, 0.0, "METAL", favs=FAV_POPUP)["PANEL"][3]  # the tallest: two sliders, two favourite rows
    x = min(max(4.0, mx - POPUP_W / 2.0), max(4.0, region.width - POPUP_W - 4.0))
    top = min(max(my + 12.0 + height, height + 4.0), region.height - 4.0)
    anchor = dock_editor_anchor(height, region.height)
    if anchor is not None:
        x, top = anchor
    was_open = _popup is not None
    _popup = _Popup(x, top, hsv, rgb, docked=anchor is not None)
    if anchor is not None:
        from . import overlay

        dock = overlay._dock
        _popup.base = float(dock.card[1] + dock.card[3] - (dock.editor[3] if dock.editor else 0))
        _popup.width = float(dock.card[2])
        _popup.x = float(dock.card[0])
        overlay._TW.jump("fade:editor", 0.0)  # fades in as it slides up
        overlay.invalidate_palette()
    if not was_open:
        context.prefs.add_modal(_PopupModal())
    voxel_object.redraw_3d_views()


def close_popup() -> None:
    global _popup
    if _popup is not None and _popup.editing:
        _commit_hex(_popup)
    _popup = None
    from . import overlay

    overlay._TW.jump("fade:editor", 0.0)  # the next open fades in again
    voxel_object.redraw_3d_views()


def _commit_hex(popup: _Popup) -> None:
    popup.editing = False
    if parse_hex(popup.text) is not None:
        CONTEXT.prefs.rok_hex = popup.text


def _slider_value(name: str, frac: float):
    frac = min(1.0, max(0.0, frac))
    if name == "rok_power":
        return int(round(frac * 4))
    return round(frac, 2)


def _slider_frac(name: str, value) -> float:
    if name == "rok_power":
        return float(value) / 4.0
    return float(value)


def _drag_to(popup: _Popup, part: str, px: float, py: float) -> None:
    """Act on a press or a drag on `part` at (px, py)."""
    wm = CONTEXT.prefs
    rect = _current_layout(popup).get(part)
    if rect is None:
        return
    x, y, w, h = rect
    fx = min(1.0, max(0.0, (px - x) / w))
    fy = min(1.0, max(0.0, (py - y) / h))
    if part == "SV":
        popup.hsv = (popup.hsv[0], fx, fy)
    elif part == "HUE":
        popup.hsv = (min(fy, 0.999), popup.hsv[1], popup.hsv[2])
    elif part.startswith("SLIDER:"):
        name = part.split(":", 1)[1]
        setattr(wm, name, _slider_value(name, fx))
        return
    else:
        return
    wm.rok_colour = colorsys.hsv_to_rgb(*popup.hsv)
    popup.seen = _rgb_bytes()


class _PopupModal(Action):
    """The popup's event handler. It lives while the popup is open."""

    command_id = "rok.colour_popup_modal"
    title = "Colour Popup"

    def modal(self, context, event):
        popup = _popup
        if popup is None:
            return {"CANCELLED", "PASS"}
        if document.scene is None:
            close_popup()
            return {"CANCELLED", "PASS"}
        region = context.region
        px, py = float(event.mouse_region_x), float(event.mouse_region_y)
        panel = _current_layout(popup)["PANEL"]
        kind, value = event.type, event.value
        if kind == "MOUSE_MOVE":
            if popup.drag is not None:
                _drag_to(popup, popup.drag, px, py)
                region.tag_redraw()
                return {"RUNNING"}
            hover = _part_at(popup, px, py)
            if hover != popup.hover:
                popup.hover = hover
                region.tag_redraw()
            return {"RUNNING"} if _inside(panel, px, py) else {"PASS"}
        if kind == "LEFT_MOUSE" and value == "PRESS":
            part = _part_at(popup, px, py)
            if popup.editing and part != "HEX":
                _commit_hex(popup)
            if part is None:
                if popup.docked and _in_dock(px, py):
                    return {"PASS"}  # the card's own buttons and swatches work with the editor open
                if not _inside(panel, px, py):
                    close_popup()  # a click outside closes the popup and does nothing else
                    return {"FINISHED"}
                region.tag_redraw()
                return {"RUNNING"}
            if part == "HEX":
                if not popup.editing:
                    popup.editing = True
                    popup.text = CONTEXT.prefs.rok_hex
                    popup.fresh = True
            elif part == "STAR":
                on = toggle_favourite()
                context.app.say("Favourite colour: it stays in every file" if on else "No longer a favourite", 2.0)
            elif part.startswith("FAV:"):
                favs = favourites()
                k = int(part.split(":", 1)[1])
                if k < len(favs) and use_favourite(favs[k]) is None:
                    context.app.say(PALETTE_FULL, 3.0)
            elif part.startswith("TYPE:"):
                context.prefs.rok_material_type = part.split(":", 1)[1]
            elif part in ("SV", "HUE") or part.startswith("SLIDER:"):
                popup.drag = part
                _drag_to(popup, part, px, py)
            region.tag_redraw()
            return {"RUNNING"}
        if kind == "RIGHT_MOUSE" and value == "PRESS":
            part = _part_at(popup, px, py)
            if part is not None and part.startswith("FAV:"):
                favs = favourites()
                k = int(part.split(":", 1)[1])
                if k < len(favs):
                    from . import opkit

                    opkit.call("rok.favourite_name", "INVOKE", hex=_hex(favs[k]))
                return {"RUNNING"}
            return {"RUNNING"} if _inside(panel, px, py) else {"PASS"}
        if kind == "LEFT_MOUSE" and value == "RELEASE":
            if popup.drag is not None:
                popup.drag = None
                region.tag_redraw()
                return {"RUNNING"}
            return {"PASS"}
        if kind == "FOCUS_LOST":
            popup.drag = None
            return {"PASS"}
        if value != "PRESS":
            return {"PASS"}
        if popup.editing:
            if kind in ("ENTER", "NUMPAD_ENTER"):
                _commit_hex(popup)
            elif kind == "ESCAPE":
                popup.editing = False
            elif kind == "BACKSPACE":
                popup.text = "" if popup.fresh else popup.text[:-1]
                popup.fresh = False
            elif event.unicode and event.unicode.isprintable() and event.unicode.isascii() and not event.cmd_key:
                popup.text = (event.unicode if popup.fresh else popup.text + event.unicode)[:7]
                popup.fresh = False
            region.tag_redraw()
            return {"RUNNING"}
        if kind in ("ESCAPE", "ENTER", "NUMPAD_ENTER"):
            close_popup()
            return {"FINISHED"}
        return {"PASS"}  # Cmd+Z and the other keys still work


def _draw_text(value: str, x: float, y: float, h: float, size: float, colour, align: str = "left", w: float = 0.0):
    """`value` in the band y..y+h, left from x, centred in w, or right to x + w."""
    fonts.size(0, size)
    tw, _th = fonts.dimensions(0, value)
    _cw, cap = fonts.dimensions(0, "H")
    if align == "centre":
        x = x + (w - tw) / 2.0
    elif align == "right":
        x = x + w - tw
    fonts.color(0, *colour)
    fonts.position(0, float(round(x)), float(round(y + (h - cap) / 2.0)), 0)
    fonts.draw(0, value)
    return tw


def _sv_square(rect, hue: float) -> None:
    x, y, w, h = rect
    n = SV_STEPS
    s = np.linspace(0.0, 1.0, n + 1)
    grid_s, grid_v = np.meshgrid(s, s, indexing="ij")  # [i, j]: saturation i, value j
    rgb = np.array([colorsys.hsv_to_rgb(hue, a, b) for a, b in zip(grid_s.ravel(), grid_v.ravel())])
    rgb = rgb.reshape(n + 1, n + 1, 3)
    px = x + grid_s * w
    py = y + grid_v * h
    quads = []
    colours = []
    for i in range(n):
        for j in range(n):
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j), (i + 1, j + 1), (i, j + 1))
            for a, b in corners:
                quads.append((px[a, b], py[a, b]))
                colours.append((*rgb[a, b], 1.0))
    pen.flat(np.asarray(quads, dtype=np.float32), np.asarray(colours, dtype=np.float32))


def _hue_bar(rect) -> None:
    x, y, w, h = rect
    n = HUE_STEPS
    quads = []
    colours = []
    for k in range(n):
        h0, h1 = k / n, (k + 1) / n
        c0 = (*colorsys.hsv_to_rgb(h0, 1.0, 1.0), 1.0)
        c1 = (*colorsys.hsv_to_rgb(h1 % 1.0, 1.0, 1.0), 1.0)
        y0, y1 = y + h0 * h, y + h1 * h
        quads += [(x, y0), (x + w, y0), (x + w, y1), (x, y0), (x + w, y1), (x, y1)]
        colours += [c0, c0, c1, c0, c1, c1]
    pen.flat(np.asarray(quads, dtype=np.float32), np.asarray(colours, dtype=np.float32))


def draw_popup(popup: _Popup) -> None:
    """The popup and everything on it."""
    _sync_hsv(popup)
    wm = CONTEXT.prefs
    material = _current_material()
    parts = _current_layout(popup)
    # The plate as triangles: the host draws a layer's triangles before its
    # rounded rects, and the gradients below are triangles too.
    if not popup.docked:  # inside the palette card the card's plate is the panel
        pen.fan(hud.rounded_rect(*parts["PANEL"], 8.0), look.PANEL)
        round_outline(parts["PANEL"], 8, look.EDGE, 1.0)
    # Saturation and value, and the hue bar, with their markers.
    h, s, v = popup.hsv
    sv = parts["SV"]
    _sv_square(sv, h)
    mx, my = sv[0] + s * sv[2], sv[1] + v * sv[3]
    round_outline((mx - 6, my - 6, 12, 12), 6, MARKER_DARK, 3.0)
    round_outline((mx - 5, my - 5, 10, 10), 5, MARKER_LIGHT, 1.5)
    hue = parts["HUE"]
    _hue_bar(hue)
    hy = hue[1] + h * hue[3]
    round_outline((hue[0] - 2, hy - 3, hue[2] + 4, 6), 2, MARKER_DARK, 2.5)
    round_outline((hue[0] - 1, hy - 2, hue[2] + 2, 4), 2, MARKER_LIGHT, 1.0)
    # The swatch and the hex field.
    r, g, b = _rgb_bytes()
    if "SWATCH" in parts:
        round_rect(parts["SWATCH"], 4, (r / 255.0, g / 255.0, b / 255.0, 1.0))
        round_outline(parts["SWATCH"], 4, look.EDGE, 1.0)
    field = parts["HEX"]
    lit = popup.hover == "HEX" or popup.editing
    round_rect(field, 5, look.FIELD_HOVER if lit else look.FIELD)
    if popup.editing:
        round_outline(field, 5, look.PICKED, 1.5)
        shown = popup.text
        if popup.fresh and shown:
            fonts.size(0, FONT)
            tw, _th = fonts.dimensions(0, shown)
            round_rect((field[0] + 7, field[1] + 4, tw + 2, field[3] - 8), 2, TEXT_MARK)
        tw = _draw_text(shown, field[0] + 8, field[1], field[3], FONT, look.TEXT)
        if not popup.fresh:
            cx = field[0] + 8 + tw + 1
            pen.lines([(cx, field[1] + 5), (cx, field[1] + field[3] - 5)], look.TEXT, 1.0)
    else:
        _draw_text(wm.rok_hex, field[0] + 8, field[1], field[3], FONT, look.TEXT)
    # The star: the current colour is a favourite, or a click makes it one.
    favs = favourites()
    if "STAR" in parts:
        star = parts["STAR"]
        on = _rgb_bytes() in favs
        if popup.hover == "STAR":
            round_rect(star, 5, look.FIELD_HOVER)
        _draw_star(star, STAR_ON if on else STAR_OFF, filled=on)
    if favs:
        label = parts["FAV_LABEL"]
        _draw_text("Favourites", label[0], label[1], label[3], 11, look.DIM)
        for k, rgb in enumerate(favs):
            rect = parts.get(f"FAV:{k}")
            if rect is None:
                break
            round_rect(rect, 4, (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0))
            here = rgb == _rgb_bytes()
            round_outline(rect, 4, look.PICKED if here else (look.TEXT if popup.hover == f"FAV:{k}" else look.EDGE),
                          2.0 if here or popup.hover == f"FAV:{k}" else 1.0)
    # What the colour is used by.
    if "INFO" in parts:  # inside the card its header shows the count
        index = document.colour
        count = int(document.used_counts()[index])
        info = parts["INFO"]
        name = favourite_name(_rgb_bytes()) if _rgb_bytes() in favourites() else ""
        used = f"Used by {count} voxels"
        _draw_text(f"{name} · {used}" if name else used, info[0], info[1], info[3], 11, look.DIM)
    label = parts["TYPE_LABEL"]
    _draw_text("Type", label[0], label[1], label[3], 11, look.DIM)
    for name, rect in parts.items():
        if name.startswith("TYPE:"):
            ident = name.split(":", 1)[1]
            picked = ident == material.type
            hovered = popup.hover == name
            round_rect(rect, 5, look.PICKED if picked else (look.FIELD_HOVER if hovered else look.BUTTON))
            text = dict((i, lab) for i, lab, _d in _TYPE_ITEMS_OTHER)[ident]
            _draw_text(text, rect[0], rect[1], rect[3], 11, look.TEXT, "centre", rect[2])
        elif name.startswith("SLIDER:"):
            setting = name.split(":", 1)[1]
            value = getattr(wm, setting)
            prop = type(wm)._props()[setting]
            hovered = popup.hover == name or popup.drag == name
            round_rect(rect, 5, look.FIELD_HOVER if hovered else look.FIELD)
            frac = min(1.0, max(0.0, _slider_frac(setting, value)))
            if frac > 0:
                round_rect((rect[0], rect[1], max(10.0, rect[2] * frac), rect[3]), 5, SLIDER_FILL)
            _draw_text(prop.name, rect[0] + 8, rect[1], rect[3], 11, look.TEXT)
            shown = str(int(value)) if setting == "rok_power" else f"{float(value):.2f}"
            _draw_text(shown, rect[0], rect[1], rect[3], 11, look.TEXT, "right", rect[2] - 8)


def _star_points(rect, inner: float = 0.45) -> np.ndarray:
    """The ten corners of a five-pointed star in `rect`, point up."""
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0 - h * 0.03
    r = min(w, h) * 0.42
    angles = np.pi / 2 + np.arange(10) * np.pi / 5
    radii = np.where(np.arange(10) % 2 == 0, r, r * inner)
    return np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)


def _draw_star(rect, colour, filled: bool) -> None:
    pts = _star_points(rect)
    if filled:
        centre = pts.mean(axis=0)
        tris = []
        for k in range(10):
            tris += [centre, pts[k], pts[(k + 1) % 10]]
        pen.flat(np.asarray(tris, dtype=np.float32), np.tile(np.asarray(colour, np.float32), (30, 1)))
    loop = np.concatenate([pts, pts[:1]])
    pen.lines(np.repeat(loop, 2, axis=0)[1:-1], colour, 1.5)


def _draw_tip(context) -> None:
    """The name of the favourite under the mouse, at once (a
    quickly appearing tooltip), in a layer above the popup."""
    popup = _popup
    if popup is None or document.scene is None or not (popup.hover or "").startswith("FAV:"):
        return
    favs = favourites()
    k = int(popup.hover.split(":", 1)[1])
    rect = _current_layout(popup).get(popup.hover)
    if rect is None or k >= len(favs):
        return
    name = favourite_name(favs[k])
    text = name if name else f"#{_hex(favs[k])} · right-click to name"
    fonts.size(0, 11)
    tw, _th = fonts.dimensions(0, text)
    w, h = tw + 16, 22
    x = min(max(4.0, rect[0] + rect[2] / 2.0 - w / 2.0), CONTEXT.region.width - w - 4.0)
    y = rect[1] + rect[3] + 6
    round_rect((x, y, w, h), 6, TIP_BG)
    round_outline((x, y, w, h), 6, look.EDGE, 1.0)
    _draw_text(text, x + 8, y, h, 11, look.TEXT)


def _draw(context) -> None:
    if _popup is not None and document.scene is not None:
        if _popup.docked:
            from . import gfx, overlay

            # Slides up out of the card's header while the card grows (overlay._dock_dy < 0).
            # Slides up out of the header with the card, cut to the card as it is
            # drawn, fading in over CONTENT_FADE.
            dock = overlay._dock
            dy = float(overlay._dock_dy)
            fade = overlay._TW.value("fade:editor", 1.0, overlay.palette_anim.CONTENT_FADE, eased=False)
            card = dock.card if dock is not None else (0, 0, 1e6, 1e6)
            shown = (card[0], card[1], card[2], max(0.0, card[3] + dy))
            with gfx.clipped(shown), gfx.shifted(0.0, dy), gfx.faded(fade):
                draw_popup(_popup)
        else:
            draw_popup(_popup)


def _pick(index: int) -> None:
    """Make `index` the current colour."""
    document.colour = index
    voxel_object.redraw_3d_views()


@register
class ColourPopupAction(Action):
    command_id = "rok.colour_popup"
    title = "Edit Colour"
    tooltip = "Open the colour picker for the current colour"
    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def execute(self, context):
        document.palette_session += 1
        _open_popup(context)
        return {"FINISHED"}


@register
class PaletteGridAction(Action):
    command_id = "rok.palette_grid"
    title = "Show All 256 Colours"
    tooltip = "Show or hide the full 256-colour grid above the colour strip"
    def execute(self, context):
        wm = context.prefs
        wm.rok_show_palette_grid = not wm.rok_show_palette_grid
        return {"FINISHED"}


# --- palette v2 dock: collapse and per-section folds, kept in settings.txt ---------


def dock_collapsed() -> bool:
    """Whether the palette dock card is collapsed to the chip and recent colours."""
    from . import settings

    return settings.get("palette_dock_collapsed") == "1"


def set_dock_collapsed(value: bool) -> None:
    from . import settings

    settings.put("palette_dock_collapsed", "1" if value else "0")


def dock_folds() -> dict:
    """Your fold state per dock section ("USED", "UNUSED", "FAV")."""
    from . import settings
    from ..core import palette_dock

    return palette_dock.fold_state(settings.get("palette_dock_folds"))


def toggle_dock_fold(section: str) -> None:
    """Fold or unfold one dock section (a click on its heading)."""
    from . import settings
    from ..core import palette_dock

    folds = dock_folds()
    folds[section] = not folds.get(section, False)
    settings.put("palette_dock_folds", palette_dock.fold_text(folds))


@register
class TogglePaletteAction(Action):
    command_id = "rok.toggle_palette"
    title = "Toggle Palette"
    tooltip = "Collapse or expand the palette card (the chevron in its header does the same)"

    def execute(self, context):
        set_dock_collapsed(not dock_collapsed())
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class ColourNewAction(Action):
    command_id = "rok.colour_new"
    title = "New Colour"
    tooltip = "Make a new white Plain colour in a free slot and edit it"
    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def execute(self, context):
        scene = document.scene
        store = scene.colours
        # A new white Plain colour right after the current one, on a slot no
        # voxel uses; never a colour you have.
        ident = store.add(NEW_RGB, after=store.id_of(document.colour))
        index = store.lease(ident, document.used_counts(), pinned=document.pinned())
        if index is None:
            store.remove([ident], document.used_counts())
            # No popup: it would edit the current colour, not a new one.
            self.report({"WARNING"}, PALETTE_FULL)
            return {"CANCELLED"}
        # The new colour is recorded in the new session, so the popup's edits merge
        # into it and one undo takes it away.
        document.palette_session += 1
        document.colour = index
        after_palette_edit(index)
        _open_popup(context)
        return {"FINISHED"}


# --- the eyedropper ------------------------------------------------------------------


def hovered_index() -> int | None:
    """Palette index of the voxel under the mouse in any model, or None.

    An ACTIVATE hover sits on another model: its hit is in that model's space.
    A wall hit and slot 0 give None.
    """
    hover = state.hover
    scene = document.scene
    if scene is None or hover is None or hover.hit is None or hover.hit.wall:
        return None
    if hover.tool == ACTIVATE:
        if state.hover_model is None:
            return None
        try:
            model = scene.by_uid(state.hover_model)
        except ValueError:
            return None
    else:
        model = document.model
        if model is None:
            return None
    voxel = hover.hit.voxel
    if any(c < 0 or c >= n for c, n in zip(voxel, model.grid.size)):
        return None
    index = int(model.grid.voxels[voxel])
    return index if index > 0 else None


def _strip_index() -> int | None:
    """The swatch of the strip or the 256 grid under the mouse, or None."""
    item = state.hover_strip
    if item is None:
        return None
    kind, index = item
    return index if kind == "SWATCH" and index > 0 else None


@register
class EyedropAction(Action):
    command_id = "rok.eyedrop"
    title = "Pick Colour"
    tooltip = "Pick the colour of the voxel under the mouse"
    @classmethod
    def poll(cls, context):
        return document.scene is not None and not state.stroke_active

    def execute(self, context):
        index = hovered_index()
        if index is None:
            return {"CANCELLED"}
        _pick(index)
        return {"FINISHED"}

    def invoke(self, context, event):
        index = hovered_index()
        if index is not None:
            _pick(index)
            return {"FINISHED"}
        if _eyedrop_active:
            return {"CANCELLED"}
        self._start(context)
        return {"RUNNING"}

    def modal(self, context, event):
        if document.scene is None or event.type == "FOCUS_LOST":
            return self._end(context, cancel=True)
        if event.type in _PASS_ON or event.type.startswith(("NDOF_", "TIMER")):
            return {"PASS"}
        if event.type in ("ESCAPE", "RIGHT_MOUSE"):
            if event.value == "PRESS":
                return self._end(context, cancel=True)
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE":
            if event.value != "PRESS":
                return {"RUNNING"}
            index = hovered_index()
            if index is None:
                index = _strip_index()
            if index is None:
                return {"RUNNING"}  # nothing under the mouse: keep picking
            _pick(index)
            return self._end(context, cancel=False)
        return {"RUNNING"}

    def _start(self, context) -> None:
        global _eyedrop_active
        _eyedrop_active = True
        context.window.cursor_modal_set("EYEDROPPER")
        context.prefs.add_modal(self)
        voxel_object.redraw_statusbar()

    def _end(self, context, cancel: bool):
        global _eyedrop_active
        _eyedrop_active = False
        if context.window is not None:
            context.window.cursor_modal_restore()
        voxel_object.redraw_statusbar()
        return {"CANCELLED"} if cancel else {"FINISHED"}


# --- the palette drag: Ctrl swaps two slots, Shift blends a gradient between them ----

# (kind, start_index) while rok.palette_drag runs, else None. Read by the overlay's
# drag_label to draw the small label at the mouse.
_drag: tuple[str, int] | None = None


def drag_label(hover: tuple[str, int] | None) -> str | None:
    """The label for the running palette drag, or None when no drag runs.

    `hover` is state.hover_strip: the swatch under the mouse right now, the same
    value the strip highlight uses. A mouse that is not over a swatch shows '?'.
    """
    if _drag is None:
        return None
    kind, _start = _drag
    return None if kind == "SWAP" else "blend to here"  # a move shows an ember insertion line instead


def _entries_between(a: int, b: int) -> tuple[int, int, list[int]] | None:
    """The colour ids of strip keys a and b, and every id from one to the
    other in your order (both ends in it), or None."""
    from . import overlay

    scene = document.scene
    if scene is None or a == b:
        return None
    ia, ib = overlay.entry_of_key(a), overlay.entry_of_key(b)
    order = scene.colours.order
    if ia is None or ib is None or ia not in order or ib not in order:
        return None
    lo, hi = sorted((order.index(ia), order.index(ib)))
    return ia, ib, order[lo:hi + 1]


def _apply_move(a: int, b: int) -> None:
    """Move the colour of strip key a to the place of key b, in your order.
    No voxel and no slot changes. One undo step."""
    found = _entries_between(a, b)
    if found is None:
        return
    ia, ib, _span = found
    store = document.scene.colours
    store.move(ia, store.order.index(ib))
    document.palette_session += 1
    after_palette_edit(None)


def _apply_gradient(a: int, b: int) -> None:
    """Blend the colours between strip keys a and b, in your order, from a's
    colour to b's: every colour between becomes a step of the gradient,
    plain. Voxels of those colours change with them. One undo step."""
    found = _entries_between(a, b)
    if found is None:
        return
    _ia, _ib, span = found
    if len(span) < 3:
        return  # nothing between them to blend
    store = document.scene.colours
    first, last = store.entries[span[0]].rgba, store.entries[span[-1]].rgba
    steps = len(span) - 1
    for k, ident in enumerate(span[1:-1], start=1):
        t = k / steps
        colour = tuple(int(round(first[c] + (last[c] - first[c]) * t)) for c in range(4))
        store.edit(ident, colour, DEFAULT)
    document.palette_session += 1
    after_palette_edit(None)


@register
class PaletteDragAction(Action):
    command_id = "rok.palette_drag"
    title = "Move or Blend Colours"
    tooltip = (
        "Drag onto another colour. Ctrl moves the colour to that place in the strip. "
        "Shift blends the colours between them into a gradient. Esc cancels"
    )
    start_index = IntProp(name="From Colour", min=1, max=4096, default=1)
    kind = EnumProp(
        name="Kind",
        items=(
            ("SWAP", "Move", "Move the colour to another place in the strip"),
            ("GRADIENT", "Gradient", "Blend the colours between them"),
        ),
        default="SWAP",
    )

    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def invoke(self, context, event):
        global _drag
        _drag = (self.kind, self.start_index)
        context.prefs.add_modal(self)
        voxel_object.redraw_3d_views()
        return {"RUNNING"}

    def modal(self, context, event):
        if document.scene is None or event.type == "FOCUS_LOST":
            return self._end(cancel=True)
        if event.type in _PASS_ON or event.type.startswith(("NDOF_", "TIMER")):
            return {"PASS"}
        if event.type in ("ESCAPE", "RIGHT_MOUSE"):
            if event.value == "PRESS":
                return self._end(cancel=True)
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE":
            if event.value != "RELEASE":
                return {"RUNNING"}
            return self._end(cancel=False)
        return {"RUNNING"}

    def _end(self, cancel: bool):
        global _drag
        target = None
        if not cancel:
            item = state.hover_strip
            if item is not None and item[0] == "SWATCH":
                target = item[1]
        _drag = None
        voxel_object.redraw_3d_views()
        from . import overlay

        if target is None or target == self.start_index:
            region = CONTEXT.region
            if self.kind == "SWAP" and region is not None:
                # Released off any swatch: cancelled, the ghost flies back.
                overlay.start_fly_back(self.start_index, (state.last_mouse[0] - region.x,
                                                          state.last_mouse[1] - region.y))
            return {"CANCELLED"}
        if self.kind == "SWAP":
            overlay.arm_swap_motion()
            _apply_move(self.start_index, target)
        else:
            _apply_gradient(self.start_index, target)
        return {"FINISHED"}


# --- settings -----------------------------------------------------------------------


def _factor_property(name: str, description: str, get, set):
    """A 0..1 slider with two decimals, the precision of the file's number text."""
    return FloatProp(
        name=name, description=description, min=0.0, max=1.0, precision=2, step=1,
        subtype="FACTOR", get=get, set=set,
    )


RokPrefs.add_property("rok_colour", FloatVectorProp(
    name="Colour",
    description="The current colour, shared by every voxel that uses it",
    subtype="COLOR_GAMMA",  # the palette bytes / 255, no colour transform
    size=3, min=0.0, max=1.0, get=_get_colour, set=_set_colour,
))
RokPrefs.add_property("rok_hex", StringProp(
    name="Hex", description="The current colour as #RRGGBB", get=_get_hex, set=_set_hex,
))
RokPrefs.add_property("rok_show_palette_grid", BoolProp(
    name="256", description="Show all 256 palette slots", default=False, update=_on_grid_toggle,
))
RokPrefs.add_property("rok_material_type", EnumProp(
    name="Type", description=TYPE_DESCRIPTION, items=_type_items, get=_get_type, set=_set_type,
))
RokPrefs.add_property("rok_metal", _factor_property(
    "Metal", "How metallic the surface is. 1 is pure metal, 0 looks like Plain", _get_weight, _set_weight,
))
RokPrefs.add_property("rok_transparency", _factor_property(
    "Transparency", "How much light passes through. 1 is clear glass, 0 is solid", _get_weight, _set_weight,
))
RokPrefs.add_property("rok_glow", _factor_property(
    "Glow", "How bright the colour glows, 0 to 1. Power sets the range", _get_weight, _set_weight,
))
RokPrefs.add_property("rok_power", IntProp(
    name="Power",
    description="Range of the light, 0 to 4. Each step is about three times brighter. 0 is a lit surface that lights nothing",
    min=0, max=4, get=_get_flux, set=_set_flux,
))
RokPrefs.add_property("rok_gloss", _factor_property(
    "Gloss", "Strength of the shiny coat on the plastic. 0 is matte", _get_weight, _set_weight,
))
RokPrefs.add_property("rok_roughness", _factor_property(
    "Roughness", "0 is a mirror finish, 1 is dull. Blurs reflections and refractions", _get_rough, _set_rough,
))

MATERIAL_PROPERTIES = (
    "rok_material_type", "rok_metal", "rok_transparency", "rok_glow", "rok_power", "rok_gloss", "rok_roughness",
)


def _on_load(context) -> None:
    global _popup, _drag, _before, _materials_before
    _popup = None
    _drag = None
    _before = None
    _materials_before = None


registry.add_draw("colour_popup", 650, "PIXEL", _draw)
registry.add_draw("colour_popup_tip", 651, "PIXEL", _draw_tip)


def cleanup_slots() -> list[int]:
    """The palette slots Clean Up Palette empties: no voxel of any model uses
    them, they hold no favourite colour, and they are not empty already."""
    scene = document.scene
    if scene is None:
        return []
    used = document.used_counts()
    favs = set(favourites())
    out = []
    for i in range(1, 256):
        if int(used[i]):
            continue
        rgba = tuple(int(c) for c in scene.palette[i])
        if rgba[:3] in favs:
            continue
        if rgba == (0, 0, 0, 0) and scene.materials.get(i) == DEFAULT:
            continue  # empty already
        out.append(i)
    return out


def cleanup_ids() -> list[int]:
    """The unused colours without a slot that Clean Up Palette removes too:
    not a favourite, not the colour you paint with."""
    scene = document.scene
    if scene is None:
        return []
    store = scene.colours
    favs = set(favourites())
    keep = store.id_of(document.colour)
    return [i for i in store.unused_ids(document.used_counts())
            if store.slot_of(i) is None and i != keep and store.entries[i].rgba[:3] not in favs]


@register
class PaletteCleanupAction(Action):
    command_id = "rok.palette_cleanup"
    title = "Clean Up Palette"
    tooltip = "Remove every colour no voxel uses; favourite colours stay"

    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def invoke(self, context, event):
        n = len(cleanup_slots()) + len(cleanup_ids())
        if n == 0:
            self.report({"INFO"}, "The palette is clean: every colour is used or a favourite")
            return {"CANCELLED"}
        return context.prefs.ask_confirm(
            self, event, title="Clean up the palette?",
            message=f"Remove the {n} colour{'s' if n != 1 else ''} that no voxel uses. "
                    "Favourite colours stay. Undo brings them back.",
            confirm_text="Remove",
        )

    def execute(self, context):
        """Remove the unused colours and empty their slots: transparent black,
        plain material. One undo step."""
        scene = document.scene
        slots, ids = cleanup_slots(), cleanup_ids()
        if not slots and not ids:
            return {"CANCELLED"}
        store = scene.colours
        store.remove(ids + [i for i in (store.id_of(s) for s in slots) if i is not None], document.used_counts())
        store.blank(slots)
        document.palette_session += 1
        after_palette_edit(None)
        document.made.difference_update(slots)  # nothing made is left to show
        if document.colour in slots:
            used = np.flatnonzero(document.used_counts()[1:]) + 1
            document.colour = int(used[0]) if len(used) else 1
        n = len(slots) + len(ids)
        self.report({"INFO"}, f"Removed {n} unused colour{'s' if n != 1 else ''}")
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class FavouriteNameAction(Action):
    command_id = "rok.favourite_name"
    title = "Name Favourite"
    tooltip = "Give a favourite colour a name; it shows when the mouse rests on it"
    hex = StringProp(name="Colour", options={"HIDDEN", "SKIP_SAVE"})
    name = StringProp(name="Name", description="Empty takes the name away", options={"SKIP_SAVE"})

    def invoke(self, context, event):
        rgb = parse_hex(self.hex)
        if rgb is None:
            return {"CANCELLED"}
        self.name = favourite_name(rgb)
        return context.prefs.open_dialog(self, title="Name Favourite", confirm_text="Save")

    def draw(self, context):
        self.layout.prop(self, "name", text="Name")

    def execute(self, context):
        rgb = parse_hex(self.hex)
        if rgb is None:
            return {"CANCELLED"}
        set_favourite_name(rgb, self.name)
        return {"FINISHED"}
registry.load_post.append(_on_load)
