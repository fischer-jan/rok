"""Rok's own UI: the left rail, drawn in the view.

`rok.core.chrome` decides where everything sits; this module draws it, acts
on a press, and keeps the hover and tooltip state. The rail holds the level
switch, the four tools with their flyouts, the primitives grid, the Model
tools, the Layers list, the Size slider and the Visible chip.

Rok has only this UI: there is no classic UI, no switch and no header swap.
So `new_ui_on()` is always True; other modules still call it.

The mouse reaches the rail through `hover(region, rx, ry)`,
`set_hover(...)` (mode.py) and `press(...)` (tools.py).

Animation is parked: the Animate cell of the level switch is drawn, dim and
disabled, and the Animate rail and its gallery are not built.

The colours are sRGB hex values.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rok.core import chrome as kit
from rok.core import hud
from rok.core.chrome import ChromeLayout, ChromeState, Widget, loop_lines
from rok.core.edit import ATTACH, BOX, CENTER, VOXEL
from rok.core.geom2d import rect_tris
from rok.core.shape import BOX_SOLID

from . import object as voxel_object
from . import opkit
from . import gfx
from . import live_shaders
from .gfx import ICONS, fonts, pen
from .handlers import registry
from .object import document
from .opkit import CONTEXT, IntProp, Action, StringProp, register
from .ops_common import _late
from .state import state

NEW = "NEW"  # the only UI Rok has


RAIL_TILE_RADIUS = 8  # the rail's active and hovered tiles are rounded, as the strip's
# An active tile: its accent per glyph (a glyph not named here takes EMBER), the
# fill's alpha, and the glyph's ink (None: the accent itself).
TILE_ACCENTS: dict[str, tuple] = {}
ACTIVE_FILL_ALPHA = 0.14
ACTIVE_INK = None


def _srgb(hex_rgb: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = hex_rgb.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, a)


# Rail plate and widgets. The plate is translucent so a little of the view
# shows through. No drop shadows.
RAIL_BG = _srgb("2A2724", 0.90)
RAIL_EDGE = _srgb("3A3632")  # hairline on the right of the rail, and the divider
PANEL_BG = (0.060, 0.056, 0.053, 1.0)  # behind a flyout and a tooltip
WIDGET_HOVER = _srgb("35312E")
EMBER = _srgb("E85D3A")
TICK_OFF = _srgb("8A847C")
GLYPH = (0.780, 0.760, 0.735, 1.0)
GLYPH_HOVER = (1.0, 0.985, 0.960, 1.0)
GLYPH_ON = (0.070, 0.060, 0.055, 1.0)
OUTLINE = (0.300, 0.285, 0.272, 1.0)
FOCUS = _srgb("E85D3A")
TEXT = (0.905, 0.885, 0.855, 1.0)
LINE_PX = 1.0  # one-pixel glyphs, as the proposal asks
TICK_FONT = 12
CHIP_FONT = 14
HEADER_FONT = 10  # "Model", "Layers" and "Size" over their widgets
# The scene's rail: the same one-ink design in a strong blue, so the colour alone
# says the rail holds the model tools. Ember stays the colour of "active" on both.
_INSIDE_INKS = {
    "RAIL_BG": RAIL_BG, "RAIL_EDGE": RAIL_EDGE, "WIDGET_HOVER": WIDGET_HOVER,
    "TICK_OFF": TICK_OFF, "GLYPH": GLYPH,
}
_SCENE_INKS = {
    "RAIL_BG": _srgb("1B4F8F", 0.97), "RAIL_EDGE": _srgb("5B9BE6"), "WIDGET_HOVER": _srgb("2B66AE"),
    "TICK_OFF": _srgb("B8D3F4"), "GLYPH": _srgb("E8F1FC"),
}
# Animate mode's rail: violet, the dock's colour, so you always see you are posing.
_ANIM_INKS = {
    "RAIL_BG": _srgb("3B2C78", 0.97), "RAIL_EDGE": _srgb("8B6CF2"), "WIDGET_HOVER": _srgb("4E3C96"),
    "TICK_OFF": _srgb("D6CCFA"), "GLYPH": _srgb("F1EDFF"),
}
SLIDER_GROOVE = _srgb("4A4540")  # the Size slider's track above the knob
SLIDER_GHOST = (0.780, 0.760, 0.735, 0.55)  # the ring on the stop a click would pick
PARKED_DIM = 0.22  # a cell that takes no press

# The Size slider: the stop under the mouse (None off the slider) and whether a
# drag is running. Kept here because draw_chrome reads no context.
hover_stop: int | None = None
slider_dragging = False

# The layout of the last draw, per region pointer: the hit test uses what was
# drawn, as the gizmo does.
_layout_cache: dict[int, tuple[tuple, ChromeLayout]] = {}

# The timer that redraws while the rail changes colour between the levels.
_fade_timer = None


def ui_mode() -> str:
    """Rok has only the NEW UI mode."""
    return NEW


def new_ui_on() -> bool:
    """True: Rok always draws the new UI."""
    return True


def chrome_state(context=None) -> ChromeState:
    """What the rail shows, read from the window manager and the document."""
    context = context or CONTEXT
    wm = context.prefs
    return ChromeState(
        tool=getattr(wm, "rok_tool", ATTACH),  # header tool; Option-erase does not move the ember
        mode=getattr(wm, "rok_mode", VOXEL),
        mirror=tuple(bool(v) for v in getattr(wm, "rok_mirror", (False, False, False))),
        brush_size=int(getattr(wm, "rok_brush_size", 1)),
        colour_tolerance=int(getattr(wm, "rok_colour_tolerance", 0)),
        solid=getattr(wm, "rok_solid", "BOX"),
        select_visible=bool(getattr(wm, "rok_select_visible", False)),
        pattern_space=str(getattr(wm, "rok_pattern_space", "ADD")),
        gradient_steps=_late("gradient_ops", "steps")() or 5,
        gradient_dither=bool(_late("gradient_ops", "dither")()),
        masked=_late("select_ops", "mask_of")() is not None,
        level=state.level,
        back_name=_back_name(),
        scene_tool=state.scene_tool,
        animating=state.animating,
        anim_pick=False,
        anim_paint=state.anim_paint if state.animating else None,
        anchor_ok=False,
        model_items=_model_items(),
        anim_menu=bool(getattr(context.app, "animation_menu", False)),
        readout_top=_readout_top(context),
        stroke_on=bool(getattr(wm, "rok_stroke_design", False)),
        brick_tool=state.brick_tool,
        brick_hand=state.brick_hand,
        brick_model=document.grid is not None and document.grid.bricks is not None,
        **_layer_state(),
    )


def _readout_top(context) -> int:
    """Where the tool readout ends at the top, so the Model flyout does not cover it."""
    region = context.region
    if region is None or state.animating:
        return 0
    from . import overlay

    return overlay.readout_top(region)


def _model_items() -> tuple[str, ...]:
    from . import model_menu

    return model_menu.items()


def active_track_index() -> int | None:
    """The animation track under edit. Animation is parked: always None."""
    return None


def _back_name() -> str:
    """In the scene: the name of the model the Model cell steps into, the
    picked one (`level_ops.step_into_picked`)."""
    scene = document.scene
    model = document.model
    if state.level != kit.LEVEL_SCENE or scene is None or model is None or scene.is_hidden(model):
        return ""
    return model.display_name(scene.models.index(model))


def _layer_state() -> dict:
    """The Layers list's part of the rail state: every layer with its models,
    and the model being edited."""
    from rok.core import layers as layers_core

    scene = document.scene
    if scene is None:
        return {}
    rows = []
    for info in layers_core.infos(scene.layers):
        models = tuple(
            (m.uid, m.display_name(k)) for k, m in enumerate(scene.models) if int(m.layer) == info.id
        )
        rows.append((info.id, info.name, info.hidden, models))
    model = document.model
    groups: dict[int, list] = {}
    for k, m in enumerate(scene.models):
        if m.link:
            groups.setdefault(m.link, []).append((m.uid, m.display_name(k)))
    linked = tuple((m.uid, tuple(n for u, n in groups[m.link] if u != m.uid))
                   for m in scene.models if m.link and len(groups[m.link]) > 1)
    live = tuple((uid, sh.id, live_shaders.label(sh), sh.on, sh.error)
                 for uid, shaders in scene.live.items() for sh in shaders)
    return {
        "layers": tuple(rows),
        "linked": linked,
        "live": live,
        "model_uid": model.uid if model is not None else -1,
        "model_name": model.display_name(scene.models.index(model)) if model is not None else "",
    }


def layout_for(region, context=None) -> ChromeLayout:
    """The rail of this region, rebuilt only when something in it changed."""
    region = region if region is not None else CONTEXT.region
    st = chrome_state(context)
    scale = 1.0  # Rok works in logical pixels
    key = (st, scale, int(region.width), int(region.height))
    pointer = region.as_pointer()
    cached = _layout_cache.get(pointer)
    if cached is None or cached[0] != key:
        layout = kit.chrome_layout(scale, st, region.width, region.height)
        _layout_cache[pointer] = (key, layout)
        return layout
    return cached[1]


def invalidate() -> None:
    """Drop the layout cache. A file load calls it."""
    _layout_cache.clear()


_icon_paths: dict[tuple[str, bool], Path | None] = {}


def icon_png_path(name: str, scale: float, *, wheel: bool = False) -> Path | None:
    """The PNG for `name`, or None if the file is missing. Looked up once: a
    frame draws dozens of icons, and two stats each added up."""
    if not name:
        return None
    key = (name, wheel)
    if key in _icon_paths:
        return _icon_paths[key]
    folder = ICONS / "chrome" / ("wheel" if wheel else "1x")
    found = None
    for candidate in (folder / f"{name}@2x.png", folder / f"{name}.png"):
        if candidate.is_file():
            found = candidate
            break
    _icon_paths[key] = found
    return found


def active_ink(glyph: str) -> tuple:
    """The ink of an active glyph: ACTIVE_INK, or else the glyph's accent."""
    return ACTIVE_INK or TILE_ACCENTS.get(glyph, EMBER)


def draw_icon(
    name: str,
    rect,
    scale: float,
    *,
    active: bool = False,
    hovered: bool = False,
    wheel: bool = False,
    icon_side: float | None = None,
    ring: bool = True,
    ink=None,
) -> bool:
    """Draw the PNG mask `name` in `rect`, tinted. True when the file was there.

    The ink is the accent when active (ember, or the glyph's TILE_ACCENTS
    colour), near white when hovered, grey otherwise, unless the caller passes
    one. Active also gets a tile and a ring in the accent.
    The hover plate is drawn by the caller. `icon_side` is the glyph's box in
    pixels (32 for the action tools); otherwise a fraction of `rect`.
    """
    app = CONTEXT.app
    if app is None or icon_png_path(name, scale, wheel=wheel) is None:
        return False
    accent = TILE_ACCENTS.get(name, EMBER)
    if ink is None:
        ink = active_ink(name) if active else (GLYPH_HOVER if hovered else GLYPH)
    x, y, w, h = rect
    cx = x + w / 2.0
    cy = y + h / 2.0
    if icon_side is not None:
        side = float(icon_side)
    else:
        factor = 0.92 if active else (0.88 if hovered else 0.82)
        side = min(w, h) * factor
    if active and ring:
        # Rok's look: a soft tile in the accent with a rounded ring.
        gfx.rrect(rect, accent[:3] + (ACTIVE_FILL_ALPHA,), RAIL_TILE_RADIUS * scale)
        gfx.rrect(rect, accent, RAIL_TILE_RADIUS * scale, stroke=1.5 * scale)
    folder = "chrome/wheel" if wheel else "chrome/1x"
    return app.icons.draw(name, (cx - side / 2.0, cy - side / 2.0, side, side), tuple(ink), folder=folder)


# --- drawing ------------------------------------------------------------------


def draw_chrome(
    layout: ChromeLayout,
    *,
    hover_id: str | None = None,
    open_id: str | None = None,
    focus_id: str | None = None,
    tooltip_for: str | None = None,
    region_width: int | None = None,
    size_hover: int | None = None,
    size_dragging: bool = False,
    tooltips: bool = True,
) -> None:
    """The rail, the open flyout and (with `tooltips`) the tooltip.

    `size_hover` is the Size slider stop under the mouse; `size_dragging` is
    True while the slider's knob is dragged."""
    scale = layout.scale
    now = time.monotonic()
    mix = kit.level_mix(layout.state.level, state.level_since, now, state.level_prev)
    amix = kit.anim_mix(layout.state.animating, state.anim_since, now)
    _check_fade(now)
    _set_inks(mix, amix)
    try:
        pen.tris(pen.to_3d(rect_tris(layout.rect)), RAIL_BG)
        pen.lines(pen.to_3d(kit.right_edge_line(layout.rect)), RAIL_EDGE, LINE_PX * scale)
        if layout.divider[2] > 0 and layout.divider[3] > 0:
            pen.tris(pen.to_3d(rect_tris(layout.divider)), RAIL_EDGE)
        for text, rect in layout.headers:
            _draw_centred_text(text, rect, HEADER_FONT * scale, TICK_OFF)
        _draw_level_switch(layout, scale, hover_id, focus_id, mix, amix)
        for widget in layout.widgets:
            if widget.kind == kit.SEGMENT_KIND:
                continue  # the level switch draws as one piece, above
            elif widget.kind == kit.SLIDER_KIND and widget.id in (kit.SIZE_ID, kit.TOLERANCE_ID):
                _draw_size_box(widget, scale, widget.id == hover_id, size_dragging, focus_id)
            else:
                _draw_widget(widget, scale, hover_id, focus_id)
        flyout = kit.flyout_widgets(layout, open_id)
        if flyout and open_id == kit.MOTION_ID:
            flyout = ()  # the motion gallery has layers of its own (_draw_gallery_*)
        if flyout:
            # The open primitives (and a Model tool's choices) get their own
            # panel under their row.
            in_grid = kit.GRID_SOLID_OF_ID.keys() | kit.CHOICE_OF_ID.keys()
            row = tuple(w for w in flyout if w.id not in in_grid)
            grid = tuple(w for w in flyout if w.id in in_grid)
            for group in (row, grid):
                if group:
                    _draw_panel(group, scale)
            for widget in flyout:
                _draw_widget(widget, scale, hover_id, focus_id)
        if tooltips:
            draw_tooltips(layout, flyout, tooltip_for, open_id, region_width)
    finally:
        _set_inks(0.0)  # other modules read GLYPH and TICK_OFF: they keep the inside inks


def _set_inks(mix: float, amix: float = 0.0) -> None:
    """Point the rail's ink names at the inside colours (mix 0), the scene's
    (mix 1), or a blend while the level changes; `amix` blends on toward
    Animate's violet."""
    g = globals()
    for name, inside in _INSIDE_INKS.items():
        scene = _SCENE_INKS[name]
        level = tuple(a + (b - a) * mix for a, b in zip(inside, scene))
        g[name] = tuple(a + (b - a) * amix for a, b in zip(level, _ANIM_INKS[name]))


def start_level_fade() -> None:
    """Redraw at 60 Hz while the rail's colour changes; a still mouse sends no event."""
    global _fade_timer
    wm = CONTEXT.prefs
    if wm is None or CONTEXT.app is None:
        return
    if _fade_timer is not None:
        wm.remove_timer(_fade_timer)
    _fade_timer = wm.add_timer(1.0 / 60.0)
    voxel_object.redraw_3d_views()


def _check_fade(now: float) -> None:
    """Stop the fade's timer once the colour change is over."""
    global _fade_timer
    if _fade_timer is None:
        return
    if now - max(state.level_since, state.anim_since) > kit.LEVEL_FADE_S:
        wm = CONTEXT.prefs
        if wm is not None:
            wm.remove_timer(_fade_timer)
        _fade_timer = None


# The level switch: a sunken track, the same on both rail colours, and a
# lighter thumb that slides to the current level with the rail's colour change.
SWITCH_TRACK = (0.035, 0.035, 0.04, 0.92)
SWITCH_TRACK_EDGE = (0.16, 0.16, 0.18, 1.0)
THUMB_SCENE = (0.30, 0.36, 0.46, 1.0)  # a cool lift over the blue rail
THUMB_MODEL = (0.33, 0.31, 0.29, 1.0)  # a warm lift over the charcoal rail
THUMB_ANIM = (0.42, 0.34, 0.72, 1.0)  # a violet lift over Animate's rail
THUMB_EDGE = (0.46, 0.47, 0.5, 1.0)
SWITCH_HOVER = (1.0, 1.0, 1.0, 0.07)
SWITCH_DIM = 0.4  # the other cell's icon: grey at this much
HAIRLINE = (1.0, 1.0, 1.0, 0.18)


def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def _outline(shape):
    pairs = np.empty((len(shape) * 2, 2), dtype=np.float32)
    pairs[0::2] = shape
    pairs[1::2] = np.roll(shape, -1, axis=0)
    return pen.to_3d(pairs)


def _draw_level_switch(layout: ChromeLayout, scale: float, hover_id, focus_id, mix: float,
                       amix: float = 0.0) -> None:
    """The level switch: stacked cells in a sunken track, Model, Scene and
    Animate from the top. The thumb sits under the current one and slides
    (`mix` from the model toward the scene, `amix` on toward Animate) as the
    rail changes colour. The current icon is near white, the others dim; the
    mouse lifts another."""
    order = [w for ident in kit.LEVEL_IDS for w in layout.widgets if w.id == ident]
    if len(order) != len(kit.LEVEL_IDS):
        return
    cells = {w.id: w for w in order}
    scene_cell, model_cell = cells[kit.LEVEL_SCENE_ID], cells[kit.LEVEL_MODEL_ID]
    top, bottom = order[0], order[-1]
    pad = round(kit.LEVEL_PAD * scale)
    x0 = top.rect[0] - pad
    y1 = top.rect[1] + top.rect[3] + pad
    y0 = bottom.rect[1] - pad
    w = scene_cell.rect[2] + 2 * pad
    track = hud.rounded_rect(x0, y0, w, y1 - y0, 9.0 * scale)
    pen.fan(pen.to_3d(track), SWITCH_TRACK)
    pen.lines(_outline(track), SWITCH_TRACK_EDGE, LINE_PX * scale)
    anim_cell = cells[kit.LEVEL_ANIM_ID]
    ty = model_cell.rect[1] + (scene_cell.rect[1] - model_cell.rect[1]) * mix
    ty = ty + (anim_cell.rect[1] - ty) * amix
    thumb = hud.rounded_rect(scene_cell.rect[0], ty, scene_cell.rect[2], scene_cell.rect[3], 7.0 * scale)
    pen.fan(pen.to_3d(thumb), _lerp(_lerp(THUMB_MODEL, THUMB_SCENE, mix), THUMB_ANIM, amix))
    pen.lines(_outline(thumb), THUMB_EDGE, LINE_PX * scale)
    for widget in order:
        x, y, cw, ch = widget.rect
        hovered = widget.id == hover_id and not widget.active and widget.enabled
        if hovered:
            lift = hud.rounded_rect(x, y, cw, ch, 7.0 * scale)
            pen.fan(pen.to_3d(lift), SWITCH_HOVER)
        if widget.active or hovered:
            ink = GLYPH_HOVER
        else:
            # The host tints with straight alpha, so only the alpha goes down.
            ink = tuple(GLYPH[:3]) + (SWITCH_DIM if widget.enabled else PARKED_DIM,)
        side = round(ch * 1.05)  # the glyphs carry their own margin
        if not draw_icon(widget.glyph, widget.rect, scale, icon_side=side, ring=False, ink=ink):
            _draw_centred_text(widget.glyph[6:7], widget.rect, CHIP_FONT * scale, ink)
        if widget.id == focus_id:
            pen.lines(pen.to_3d(kit.rect_lines(widget.rect, inset=-1.0)), FOCUS, LINE_PX * scale)
    # A hairline under the track sets the switch apart from the tools.
    hy = y0 - round(kit.LEVEL_GAP * scale / 2)
    pen.tris(
        pen.to_3d(rect_tris((round(8 * scale), hy, layout.rect[2] - round(16 * scale), max(1, round(scale))))),
        HAIRLINE,
    )


def _hover_plate(rect, scale: float) -> None:
    """Hover: a light rounded tile."""
    gfx.rrect(rect, WIDGET_HOVER, RAIL_TILE_RADIUS * scale)


def _draw_widget(widget: Widget, scale: float, hover_id, focus_id) -> None:
    if widget.kind == kit.LIST_KIND:
        _draw_list_widget(widget, scale, hover_id)
        return
    hovered = widget.id == hover_id
    has_png = bool(widget.glyph) and not widget.label and icon_png_path(widget.glyph, scale) is not None
    if widget.kind == kit.CHIP_KIND:
        if hovered:
            _hover_plate(widget.rect, scale)
        ink = EMBER if widget.active else (GLYPH_HOVER if hovered else GLYPH)
        if not widget.enabled:
            ink = tuple(GLYPH[:3]) + (PARKED_DIM,)
        if widget.id == kit.GRADIENT_END_ID:
            _late("gradient_ops", "draw_end_swatch")(widget.rect, scale, hovered)
        else:
            _draw_chip_contents(widget, scale, ink)
    elif has_png:
        # One ink: ember when active (with a ring), near white under the mouse,
        # grey at rest. Never a fill over the glyph.
        if hovered and not widget.active:
            _hover_plate(widget.rect, scale)
        icon_side = round(kit.TOOL_ICON * scale) if widget.id in kit.TOOL_IDS else None
        ink = active_ink(widget.glyph) if widget.active else (GLYPH_HOVER if hovered else GLYPH)
        if not widget.enabled:
            ink = tuple(GLYPH[:3]) + (PARKED_DIM,)  # it does not run here
        if not draw_icon(widget.glyph, widget.rect, scale, active=widget.active, hovered=hovered,
                         icon_side=icon_side, ink=ink):
            lines = kit.glyph_lines(widget.glyph, widget.rect, round(3 * scale))
            pen.lines(pen.to_3d(lines), ink, LINE_PX * scale)
    else:
        # Ticks and text buttons: on = ember letter, off = #8A847C. No gray box.
        if widget.kind == kit.ICON_KIND and widget.active:
            # A lit icon button: a rounded ember plate, no square frame over it.
            gfx.rrect(widget.rect, EMBER[:3] + (0.30,), 8 * scale)
            ink = EMBER
        elif widget.active:
            pen.lines(pen.to_3d(kit.rect_lines(widget.rect)), EMBER, LINE_PX * scale)
            ink = EMBER
        elif hovered:
            _hover_plate(widget.rect, scale)
            ink = GLYPH_HOVER
        else:
            ink = TICK_OFF
        inset = round(4 * scale)
        if widget.glyph:
            lines = kit.glyph_lines(widget.glyph, widget.rect, inset)
            pen.lines(pen.to_3d(lines), ink, LINE_PX * scale)
        if widget.label:
            _draw_centred_text(widget.label, widget.rect, TICK_FONT * scale, ink)
    if widget.id == focus_id:
        pen.lines(pen.to_3d(kit.rect_lines(widget.rect, inset=-1.0)), FOCUS, LINE_PX * scale)


def _draw_chip_contents(widget: Widget, scale: float, ink) -> None:
    """A 40 x 40 chip: its glyph, or its label, centred."""
    if widget.glyph:
        icon_side = round(kit.TOOL_ICON * scale)
        if not draw_icon(widget.glyph, widget.rect, scale, active=widget.active, hovered=False,
                         icon_side=icon_side, ring=False, ink=ink):
            pen.lines(pen.to_3d(kit.glyph_lines(widget.glyph, widget.rect, round(4 * scale))),
                           ink, LINE_PX * scale)
        return
    # "Part: Recon" shows its name alone; any other label shows whole (it
    # shrinks to fit the chip when it must).
    text = widget.label.split(":", 1)[1].strip() if ":" in widget.label else widget.label
    _draw_centred_text(text, widget.rect, CHIP_FONT * scale, ink)


def _draw_size_box(widget: Widget, scale: float, hovered: bool, dragging: bool, focus_id) -> None:
    """The Size field: a plate with the brush size as a number you drag left or
    right; a small chevron on each side says so."""
    x, y, w, h = widget.rect
    value = widget.label if widget.label.isdigit() else str(kit.BRUSH_SIZES[0])
    inset = 3.0 * scale
    plate = hud.rounded_rect(x + inset, y + 2.0 * scale, w - 2 * inset, h - 4.0 * scale, 6.0 * scale)
    pen.fan(pen.to_3d(plate), SLIDER_GROOVE)
    if hovered or dragging:
        pen.lines(pen.to_3d(loop_lines(plate)), EMBER if dragging else TICK_OFF, LINE_PX * scale)
    font = 0
    fonts.size(font, (CHIP_FONT + 2) * scale)
    tw, th = fonts.dimensions(font, value)
    fonts.color(font, *(EMBER if dragging else (GLYPH_HOVER if hovered else TEXT)))
    fonts.position(font, float(x + (w - tw) / 2.0), float(y + (h - th) / 2.0), 0)
    fonts.draw(font, value)
    fonts.size(font, CHIP_FONT * scale)
    fonts.color(font, *TICK_OFF)
    for mark, cx in (("‹", x + inset + 5.0 * scale), ("›", x + w - inset - 5.0 * scale)):
        mw, mh = fonts.dimensions(font, mark)
        fonts.position(font, float(cx - mw / 2.0), float(y + (h - mh) / 2.0), 0)
        fonts.draw(font, mark)
    if widget.id == focus_id:
        pen.lines(pen.to_3d(kit.rect_lines(widget.rect, inset=-1.0)), FOCUS, LINE_PX * scale)


LIST_FONT = 13
LIST_DIM = (0.78, 0.76, 0.735, 0.45)  # an empty or hidden layer's name: the grey, fainter
LIST_SELECTED = _srgb("4A4540")  # the model being edited: a lighter plate, no ember


def _draw_list_widget(widget: Widget, scale: float, hover_id) -> None:
    """A part of the Layers list, in the rail's one ink, laid out as a
    layer list: a layer's row (the eye, a small open triangle
    when it holds models, the name), its models indented under it, the model
    being edited on a lighter plate. The move arrow shows only while its
    layer's row is under the mouse."""
    x, y, w, h = widget.rect
    part = kit.layer_part(widget.id)
    hover = kit.layer_part(hover_id)
    layer_parts = (kit.LAYER_EYE, kit.LAYER_ROW, kit.LAYER_MOVE)  # ids that name a layer, not a model
    same_row = part is not None and hover is not None and hover[0] in layer_parts and hover[1] == part[1]
    hovered = widget.id == hover_id
    ink = GLYPH_HOVER if hovered else (LIST_DIM if widget.dim else GLYPH)
    if part[0] == kit.LAYER_MODEL:
        if widget.active:
            pen.tris(pen.to_3d(rect_tris(widget.rect)), LIST_SELECTED)
        elif hovered:
            pen.tris(pen.to_3d(rect_tris(widget.rect)), WIDGET_HOVER)
        name_ink = GLYPH_HOVER if (widget.active or hovered) else ink
        indent = round(kit.LIST_INDENT * scale)
        box = (x + indent - round(12 * scale), y + h / 2.0 - 3 * scale, round(6 * scale), round(6 * scale))
        pen.lines(pen.to_3d(kit.rect_lines(tuple(int(v) for v in box))), name_ink, LINE_PX * scale)
        _draw_left_text(widget.label, (x + indent, y, w, h), LIST_FONT * scale, name_ink)
        return
    if part[0] == kit.LAYER_ROW:
        if same_row:
            pen.tris(pen.to_3d(rect_tris(widget.rect)), WIDGET_HOVER)
        if widget.glyph:
            # A small open triangle: this layer holds models, listed below.
            cx, cy, r = x + 9 * scale, y + h / 2.0, 3.5 * scale
            tri = np.array([(cx - r, cy + r * 0.6), (cx + r, cy + r * 0.6), (cx + r, cy + r * 0.6),
                            (cx, cy - r * 0.8), (cx, cy - r * 0.8), (cx - r, cy + r * 0.6)], dtype=np.float32)
            pen.lines(pen.to_3d(tri), ink, LINE_PX * scale)
        _draw_left_text(widget.label, (x + round(20 * scale), y, w, h), LIST_FONT * scale, ink)
        return
    if part[0] == kit.LIVE_ROW:
        # A live shader, under its model: a small diamond, then its name.
        if hovered:
            pen.tris(pen.to_3d(rect_tris(widget.rect)), WIDGET_HOVER)
        cx, cy, r = x + 8 * scale, y + h / 2.0, 3.5 * scale
        diamond = np.array([(cx, cy - r), (cx + r, cy), (cx + r, cy), (cx, cy + r), (cx, cy + r), (cx - r, cy),
                            (cx - r, cy), (cx, cy - r)], dtype=np.float32)
        pen.lines(pen.to_3d(diamond), ink, LINE_PX * scale)
        _draw_left_text(widget.label, (x + round(18 * scale), y, w, h), LIST_FONT * scale, ink)
        return
    if part[0] in (kit.LAYER_EYE, kit.LIVE_EYE):
        if hovered:
            pen.tris(pen.to_3d(rect_tris(widget.rect)), WIDGET_HOVER)
        eye_ink = GLYPH_HOVER if hovered else (GLYPH if widget.active else LIST_DIM)
        side = round(18 * scale)
        box = (x + (w - side) // 2, y + (h - side) // 2, side, side)
        if not draw_icon("SEEN", box, scale, icon_side=side, ring=False, ink=eye_ink):
            pen.lines(pen.to_3d(kit.circle_lines(x + w / 2, y + h / 2, 5 * scale)), eye_ink, LINE_PX * scale)
        if not widget.active:  # hidden: a stroke through the eye
            a = (x + w * 0.25, y + h * 0.25)
            b = (x + w * 0.75, y + h * 0.75)
            pen.lines(pen.to_3d(np.array([a, b], dtype=np.float32)), eye_ink, 1.5 * scale)
        return
    if part[0] == kit.LAYER_LINK:
        # The chain of a linked copy: always shown; a click makes it unique.
        if hovered:
            pen.tris(pen.to_3d(rect_tris(widget.rect)), WIDGET_HOVER)
        side = round(16 * scale)
        box = (x + (w - side) // 2, y + (h - side) // 2, side, side)
        draw_icon("LINK", box, scale, icon_side=side, ring=False, ink=ink)
        return
    # The move arrow, on the layer row under the mouse.
    if same_row and widget.enabled:
        if hovered:
            pen.tris(pen.to_3d(rect_tris(widget.rect)), WIDGET_HOVER)
        mid = y + h / 2.0
        pts = [(x + w * 0.25, mid), (x + w * 0.72, mid), (x + w * 0.55, mid + h * 0.16), (x + w * 0.72, mid),
               (x + w * 0.55, mid - h * 0.16), (x + w * 0.72, mid)]
        pen.lines(pen.to_3d(np.array(pts, dtype=np.float32)), GLYPH_HOVER if hovered else GLYPH, 1.5 * scale)


def _draw_left_text(text: str, rect, size: float, colour) -> None:
    x, y, w, h = rect
    font = 0
    fonts.size(font, size)
    _tw, th = fonts.dimensions(font, "Hg")
    fonts.color(font, *colour)
    fonts.position(font, float(x), float(y + (h - th) / 2.0 + th * 0.2), 0)
    fonts.draw(font, text)


def _draw_centred_text(text: str, rect, size: float, colour) -> None:
    """Text centred in `rect`. A label wider than the rect shrinks until it fits."""
    x, y, w, h = rect
    font = 0
    fonts.size(font, size)
    tw, _th = fonts.dimensions(font, text)
    while tw > w - 4 and size > 6:
        size -= 1
        fonts.size(font, size)
        tw, _th = fonts.dimensions(font, text)
    # Centre the capitals, not the text's own box: text with and without
    # descenders then sits on one baseline (with the box centred, rows jump
    # by a pixel or two).
    _cw, cap = fonts.dimensions(font, "H")
    fonts.color(font, *colour)
    fonts.position(font, float(x + (w - tw) / 2.0), float(y + (h - cap) / 2.0), 0)
    fonts.draw(font, text)


def _draw_panel(widgets: tuple[Widget, ...], scale: float) -> None:
    """The backdrop of a flyout: the bounding box of its buttons, padded."""
    pad = round(3 * scale)
    x0 = min(w.rect[0] for w in widgets) - pad
    y0 = min(w.rect[1] for w in widgets) - pad
    x1 = max(w.rect[0] + w.rect[2] for w in widgets) + pad
    y1 = max(w.rect[1] + w.rect[3] for w in widgets) + pad
    rect = (x0, y0, x1 - x0, y1 - y0)
    pen.tris(pen.to_3d(rect_tris(rect)), PANEL_BG)
    pen.lines(pen.to_3d(kit.rect_lines(rect)), OUTLINE, LINE_PX * scale)


def draw_tooltips(layout: ChromeLayout, flyout, tooltip_for: str | None, open_id: str | None,
                  region_width: int | None) -> None:
    """The tooltip of the rail widget under the mouse."""
    scale = layout.scale
    if kit.layer_part(tooltip_for) is not None and flyout:
        # The Layers list covers the lower-left readout: its hints go beside it.
        item = next((w for w in flyout if w.id == tooltip_for), None)
        if item is not None and item.tooltip:
            right = max(w.rect[0] + w.rect[2] for w in flyout) + round(kit.FLYOUT_GAP * 2 * scale)
            beside = Widget(id=item.id, kind=item.kind, rect=(right, item.rect[1], 1, item.rect[3]),
                            tooltip=item.tooltip)
            _draw_tooltip(beside, layout, region_width if region_width is not None else 4096, scale)
    elif tooltip_for is not None and not kit.is_flyout_item(tooltip_for) and not (flyout and tooltip_for == open_id):
        # A flyout item is named in the lower-left readout instead, and an
        # open row hides its opener's tooltip (it would cover the row).
        widget = layout.by_id(tooltip_for)
        if widget is not None and widget.tooltip:
            width = region_width if region_width is not None else 4096
            _draw_tooltip(widget, layout, width, scale)


def _draw_tooltip(widget: Widget, layout: ChromeLayout, region_width: int, scale: float) -> None:
    rect = kit.tooltip_rect(widget, layout, region_width)
    pen.tris(pen.to_3d(rect_tris(rect)), PANEL_BG)
    pen.lines(pen.to_3d(kit.rect_lines(rect)), OUTLINE, LINE_PX * scale)
    _draw_centred_text(widget.tooltip, rect, CHIP_FONT * scale, TEXT)


def _draw(context) -> None:
    """The draw handler: the rail over the view, once a model is open."""
    if document.grid is None:
        return
    region = context.region
    draw_chrome(
        layout_for(region, context),
        hover_id=state.hover_chrome,
        open_id=state.chrome_open,
        focus_id=state.chrome_focus,
        tooltip_for=tooltip_id(),
        region_width=region.width,
        size_hover=hover_stop,
        size_dragging=slider_dragging,
        tooltips=False,  # their own layer (_draw_tips), over the history strip
    )


def _draw_tips(context) -> None:
    """The rail's tooltip, in a layer above the history strip and the other cards."""
    if document.grid is None:
        return
    region = context.region
    layout = layout_for(region, context)
    open_id = state.chrome_open
    flyout = kit.flyout_widgets(layout, open_id)
    if flyout and open_id == kit.MOTION_ID:
        flyout = ()
    draw_tooltips(layout, flyout, tooltip_id(), open_id, region.width)


# --- hover and press ----------------------------------------------------------


def hover(region, rx: float, ry: float) -> object:
    """The widget id under the pixel, None inside the rail with no widget, or
    False outside it. The shape `mode.update_hover` expects from the rail."""
    global hover_stop
    region = region if region is not None else CONTEXT.region
    layout = layout_for(region)
    widget = kit.hit_test(layout, rx, ry, state.chrome_open)
    if hover_stop is not None:
        hover_stop = None
        region.tag_redraw()
    if widget is not None:
        return widget.id
    if kit.contains(layout, rx, ry, state.chrome_open):
        return None
    return False


def set_hover(widget_id: str | None, inside: bool) -> bool:
    """Store the hovered widget and the flyout it opens. True on a change.

    An opener (a tool button, the Model button) keeps its flyout open while the
    mouse is on it, on the flyout itself or on the bare plate between them;
    another widget or a mouse outside the rail closes it.
    """
    now = time.monotonic()
    open_id = state.chrome_open
    if widget_id in kit.FLYOUT_OPENERS:
        open_id = widget_id
    elif widget_id is not None and kit.is_flyout_item(widget_id):
        pass  # the mouse is inside the open flyout
    elif widget_id is not None or not inside:
        open_id = None
    changed = widget_id != state.hover_chrome or open_id != state.chrome_open
    if changed:
        state.chrome_since = now if widget_id is not None else None
        if widget_id is not None:
            _schedule_tooltip()
    moved = widget_id != state.hover_chrome
    state.hover_chrome = widget_id
    state.chrome_open = open_id
    if moved:
        from . import mirror_preview  # late: it imports opkit's operators

        mirror_preview.hover(widget_id)
    return changed


def _schedule_tooltip() -> None:
    """Ask for the redraw that shows the tooltip; a mouse that rests sends no event.
    The app redraws after every tick, and the tooltip is due by then."""
    app = CONTEXT.app
    if app is not None:
        app.host.tick(kit.TOOLTIP_DELAY_S + 0.02)


def tooltip_id(now: float | None = None) -> str | None:
    """The widget whose tooltip is due, or None."""
    if state.hover_chrome is None:
        return None
    from . import marking  # late: marking imports this module

    if marking.active():
        return None  # the star is open over the rail: its own words say what each chip is
    when = time.monotonic() if now is None else now
    return state.hover_chrome if kit.tooltip_due(state.chrome_since, when) else None


def press(context, region, mx: float, my: float, left: bool, shift: bool) -> bool:
    """Act on a press in the rail. True when the press was inside it.

    A press anywhere in the rail is eaten, so a click beside a button never
    reaches the voxels behind it.
    """
    region = region if region is not None else context.region
    if document.grid is None:
        return False
    layout = layout_for(region, context)
    widget = kit.hit_test(layout, mx, my, state.chrome_open)
    if widget is not None:
        if left:
            _run(context, widget.id, shift, layout, region, mx, my)
            region.tag_redraw()
        return True
    return kit.contains(layout, mx, my, state.chrome_open)


def _op(ident: str):
    """`opkit.ops.rok.<ident>` when it exists and its poll passes, else None.
    Actions of other modules may not be registered yet."""
    op = getattr(opkit.ops.rok, ident)
    return op if op.poll() else None


def _run_command_later(ident: str) -> None:
    """Run a whole-model command from the rail. It runs at once: a dialog that
    opens here does not break Cmd+Z."""
    op = _op(ident)
    if op is not None:
        op("INVOKE")


def _run(context, widget_id: str, shift: bool, layout: ChromeLayout, region,
         mx: float | None = None, my: float | None = None) -> None:
    """What one widget does. Every action is a property write or an operator
    the app already has: the rail adds no new behaviour."""
    wm = context.prefs
    if widget_id in kit.TOOL_IDS and state.animating:
        op = _op("anim_paint")  # Animate mode: the tool paints the picked part's frame
        if op is not None:
            op(tool=kit.TOOL_IDS[widget_id])
    elif widget_id in kit.TOOL_IDS:
        wm.rok_tool = kit.TOOL_IDS[widget_id]
    elif widget_id in kit.MODEL_CHOICES:
        # A click opens the tool's choices under the item, a second closes them.
        opened = state.chrome_open == widget_id
        state.chrome_open = kit.CHOICE_PARENT[widget_id] if opened else widget_id
    elif widget_id in kit.CHOICE_OF_ID:
        choice = kit.CHOICE_OF_ID[widget_id]
        op = _op(choice.op)
        if op is not None:
            op(**dict(choice.props))
    elif widget_id in kit.COMMAND_OF_MODEL_ID:
        _run_command_later(kit.COMMAND_OF_MODEL_ID[widget_id])
    elif widget_id in kit.COMMAND_OF_ANIM_ID:
        _run_command_later(kit.COMMAND_OF_ANIM_ID[widget_id])
    elif widget_id == kit.STROKE_ID:
        op = _op("stroke_design")
        if op is not None:
            op()
    elif widget_id in (kit.SIZE_ID, kit.TOLERANCE_ID):
        # A press on the number starts a drag: right grows it, left shrinks it.
        widget = layout.by_id(widget_id)
        if widget is not None:
            op = _op("size_slider")
            if op is not None:
                tolerance = widget_id == kit.TOLERANCE_ID
                prop = "rok_colour_tolerance" if tolerance else "rok_brush_size"
                op("INVOKE", before=int(getattr(wm, prop, 0 if tolerance else 1)), field="TOLERANCE" if tolerance else "SIZE")
    elif widget_id == kit.LAYERS_ID:
        pass  # its list opens on hover, like the Model button's
    elif widget_id == kit.LEVEL_SCENE_ID:
        _late("anim_mode", "go")(kit.LEVEL_SCENE)
    elif widget_id == kit.LEVEL_MODEL_ID:
        _late("anim_mode", "go")(kit.LEVEL_MODEL)
    elif widget_id == kit.LEVEL_ANIM_ID:
        _late("anim_mode", "go")("ANIMATE")
    elif widget_id in kit.BRICK_TOOL_IDS:
        _late("brick_mode", "set_tool")(kit.BRICK_TOOL_IDS[widget_id])
    elif widget_id == kit.BRICK_CONVERT_ID:
        _run_command_later("convert_to_bricks")
    elif widget_id == kit.BRICK_TO_VOXELS_ID:
        _run_command_later("convert_to_voxels")
    elif widget_id in kit.PRESET_OF_ID:
        state.chrome_open = None
        op = _op("anim_preset")
        if op is not None:
            op(name=kit.PRESET_OF_ID[widget_id])
    elif widget_id == kit.SCENE_TOOLS_ID:
        pass  # its row opens on hover
    elif widget_id == kit.SCENE_DROP_ID:
        state.chrome_open = None
        op = _op("scene_drop")
        if op is not None:
            op()
    elif widget_id == kit.SCENE_DROP_ONTO_ID:
        state.chrome_open = None
        op = _op("scene_drop_onto")
        if op is not None:
            op()
    elif widget_id == kit.SCENE_FIT_ID:
        op = _op("scene_fit")
        if op is not None:
            op()
    elif widget_id == kit.SCENE_DELETE_ID:
        op = _op("scene_delete")
        if op is not None:
            op()
    elif widget_id in (kit.SCENE_PICK_LINKED_ID, kit.SCENE_ALIGN_ID, kit.SCENE_ARRAY_ID, kit.SCENE_CARVE_ID):
        state.chrome_open = None
        op = _op({kit.SCENE_PICK_LINKED_ID: "scene_pick_linked", kit.SCENE_ALIGN_ID: "scene_align",
                  kit.SCENE_ARRAY_ID: "scene_array", kit.SCENE_CARVE_ID: "scene_carve"}[widget_id])
        if op is not None:
            op("INVOKE")
    elif widget_id in (kit.SCENE_PICK_ID, kit.SCENE_CUT_ID):
        state.scene_tool = "PICK" if widget_id == kit.SCENE_PICK_ID else "CUT"
        state.cut_aim = None
        _late("gizmo_ops", "refresh")()  # the Cut tool has no gizmo
    elif widget_id in kit.NEW_SOLID_OF_ID:
        # A click makes the shape next to the scene; a drag places it (scene_tools).
        state.chrome_open = None
        op = _op("scene_place")
        if op is not None:
            op("INVOKE", solid=kit.NEW_SOLID_OF_ID[widget_id])
            _late("last_command", "note_widget")(widget_id)
    elif widget_id == "NEW_EMPTY":
        op = _op("model_add")
        if op is not None:
            op()
        state.chrome_open = None
    elif widget_id == "NEW_FILE":
        state.chrome_open = None
        _run_command_later("model_add_file")
    elif widget_id == "NEW_TERRAIN":
        state.chrome_open = None
        _run_command_later("terrain_new")
    elif widget_id == "NEW_HEIGHTMAP":
        state.chrome_open = None
        _run_command_later("heightmap_new")
    elif kit.layer_part(widget_id) is not None:
        part, layer_id = kit.layer_part(widget_id)
        if part == kit.LAYER_EYE:
            op = _op("layer_toggle")
            if op is not None:
                op(layer=layer_id)
        elif part == kit.LAYER_MODEL:
            # In the scene a click picks the model and a double-click steps in;
            # inside a model a click moves to that one.
            now = time.monotonic()
            twice = kit.is_double_click(state.last_pick, layer_id, now)
            picked_before = layer_id == document.active_uid
            op = _op("model_edit_from_list")
            if op is not None:
                op(uid=layer_id)
            # Only a click on the picked model's row can start a double-click.
            state.last_pick = (layer_id, now) if picked_before else None
            if twice and layout.state.level == kit.LEVEL_SCENE:
                _late("level_ops", "step_into")(layer_id)
        elif part in (kit.LIVE_EYE, kit.LIVE_ROW):
            uid = live_shaders.owner(layer_id)
            if uid is None:
                pass
            elif part == kit.LIVE_EYE:
                op = _op("live_toggle")
                if op is not None:
                    op(uid=uid, live=layer_id)
            else:
                if uid != document.active_uid:
                    op = _op("model_edit_from_list")
                    if op is not None:
                        op(uid=uid)
                state.chrome_open = None
                if uid == document.active_uid:
                    opkit.ops.rok.shader_panel("INVOKE", live=layer_id)
        elif part == kit.LAYER_LINK:
            op = _op("model_make_unique")
            if op is not None:
                op("INVOKE", uid=layer_id)  # asks first, unless that was turned off
        elif part == kit.LAYER_MOVE:
            op = _op("model_to_layer")
            if op is not None:
                op(layer=layer_id)
        elif part == kit.LAYER_ROW:
            # A click on a layer's own row frames every shown model on it; the eye and the arrow act on their own.
            scene = document.scene
            models = [m for m in scene.models if int(m.layer) == layer_id and not scene.is_hidden(m)]
            if models:
                voxel_object.frame_models(context, models)
                voxel_object.redraw_3d_views()
    elif widget_id == kit.MASK_ID:
        op = getattr(opkit.ops.rok, "mask_clear")
        if op.poll():
            op()
    elif widget_id == kit.STAMP_ID:
        op = getattr(opkit.ops.rok, "pattern_space")
        if op.poll():
            op(step="NEXT")
    elif widget_id == kit.GRADIENT_STEPS_ID:
        _late("gradient_ops", "step")(-1 if shift else 1)
    elif widget_id == kit.GRADIENT_DITHER_ID:
        _late("gradient_ops", "toggle_dither")()
    elif widget_id == kit.GRADIENT_END_ID:
        op = getattr(opkit.ops.rok, "gradient_end")
        if op.poll():
            op("INVOKE")
    elif widget_id == kit.VISIBLE_ID:
        wm.rok_select_visible = not bool(getattr(wm, "rok_select_visible", False))
    elif widget_id == kit.PRIMITIVES_ID:
        # A click opens the grid of primitives under the item, a second closes it.
        opened = state.chrome_open == kit.PRIMITIVES_ID
        state.chrome_open = "TOOL_ATTACH" if opened else kit.PRIMITIVES_ID
    elif widget_id in kit.SOLID_OF_ID:
        # A primitive: Attach builds it in Box mode, or from the centre in Center mode.
        centre = getattr(wm, "rok_tool", ATTACH) == ATTACH and getattr(wm, "rok_mode", VOXEL) == CENTER
        wm.rok_tool = ATTACH
        wm.rok_mode = CENTER if centre else BOX
        wm.rok_solid = kit.SOLID_OF_ID[widget_id]
        state.chrome_open = None
        _late("last_command", "note_widget")(widget_id)
    elif widget_id in kit.TOOL_MODE_OF_ID:
        wm.rok_tool, wm.rok_mode = kit.TOOL_MODE_OF_ID[widget_id]
        if widget_id == "TOOL_ATTACH_BOX":
            wm.rok_solid = BOX_SOLID  # Box is the plain box; the primitives have their own item
    # The Animate rail's widgets (motion gallery, colour pick, anchor) are parked.


# --- the Size slider drag -------------------------------------------------------


@register
class SizeSliderAction(Action):
    """Drag the Size slider's knob: the mouse height along the track picks the
    stop until the button comes up. Esc or the right button put the old size
    back."""

    command_id = "rok.size_slider"
    title = "Brush Size"
    tooltip = "Drag to set the brush size"
    before = IntProp(name="Before", default=0, options={"SKIP_SAVE", "HIDDEN"})
    field = StringProp(name="Field", default="SIZE", options={"SKIP_SAVE", "HIDDEN"})  # SIZE or TOLERANCE

    def _prop(self) -> str:
        return "rok_colour_tolerance" if self.field == "TOLERANCE" else "rok_brush_size"

    def _value(self, wm) -> int:
        return int(getattr(wm, self._prop(), 0 if self.field == "TOLERANCE" else 1))

    @classmethod
    def poll(cls, context):
        return context.region is not None

    def invoke(self, context, event):
        global slider_dragging
        region = context.region
        layout = layout_for(region, context)
        widget = layout.by_id(kit.TOLERANCE_ID if self.field == "TOLERANCE" else kit.SIZE_ID)
        if widget is None:
            return {"CANCELLED"}
        wm = context.prefs
        self._widget = widget
        self._scale = layout.scale
        self._before = int(self.before) if self.field == "TOLERANCE" else (int(self.before) or self._value(wm))
        self._x0 = float(event.mouse_region_x)  # the drag counts from the press
        # The field sits at the window's left edge: the pointer would stop
        # there and the size could only shrink by one. The
        # pointer hides and holds still; the drag adds up the raw moves.
        self._dx = 0.0
        self._relative = context.window.relative_set(True) if context.window is not None else False
        slider_dragging = True
        wm.add_modal(self)
        region.tag_redraw()
        return {"RUNNING"}

    def _end(self) -> None:
        global slider_dragging
        slider_dragging = False
        if self._relative:
            CONTEXT.window.relative_set(False)
        voxel_object.redraw_3d_views()

    def modal(self, context, event):
        wm = context.prefs
        if event.type in ("MOUSE_MOVE", "MOUSE_MOVE_BETWEEN", "MOUSE_DELTA"):
            if event.type == "MOUSE_DELTA":
                self._dx += event.dx
            elif not self._relative:
                self._dx = float(event.mouse_region_x) - self._x0
            if self.field == "TOLERANCE":
                value = kit.tolerance_from_drag(self._before, self._dx, self._scale)
            else:
                value = kit.size_from_drag(self._before, self._dx, self._scale)
            if value != self._value(wm):
                setattr(wm, self._prop(), value)
                voxel_object.redraw_3d_views()
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            self._end()
            return {"FINISHED"}
        if event.value == "PRESS" and event.type in ("ESCAPE", "RIGHT_MOUSE"):
            setattr(wm, self._prop(), self._before)
            self._end()
            return {"CANCELLED"}
        if event.type == "FOCUS_LOST":
            self._end()
            return {"FINISHED"}
        return {"RUNNING"}


def _on_load(context) -> None:
    invalidate()
    state.hover_chrome = None
    state.chrome_open = None
    state.chrome_since = None


registry.add_draw("chrome", 500, "PIXEL", _draw)
registry.add_draw("chrome_tips", 695, "PIXEL", _draw_tips)  # over the history strip (692, 693)


# --- the Motion gallery (Animate mode) --------------------------------------------------
#
# A dark card beside the rail with a card per motion; the card under the mouse
# plays its little picture (rok/core/motion_preview.py). Two layers, so the
# plates sit under the pictures (a layer draws its triangles first).

GALLERY_BG = (0.090, 0.075, 0.133, 0.97)
GALLERY_EDGE = (0.180, 0.153, 0.282, 1.0)
CARD_BG = (1.0, 1.0, 1.0, 0.05)
CARD_HOVER = (0.545, 0.424, 0.949, 0.24)
CARD_STAGE = (0.0, 0.0, 0.0, 0.26)
GALLERY_TEXT = (0.94, 0.92, 0.98, 1.0)
GALLERY_DIM = (0.64, 0.61, 0.74, 1.0)
GALLERY_FAINT = (0.45, 0.42, 0.54, 1.0)
_card_hover: tuple[str, float] | None = None


def _gallery_widgets(context):
    if not state.animating or state.chrome_open != kit.MOTION_ID or context.region is None:
        return None, ()
    layout = layout_for(context.region)
    return layout, kit.flyout_widgets(layout, kit.MOTION_ID)


def _card_parts(rect, scale: float):
    x, y, w, h = rect
    label_h = 22 * scale
    pad = 6 * scale
    return (x + pad, y + label_h, w - 2 * pad, h - label_h - pad), label_h


def _draw_gallery_bg(context) -> None:
    from .paint import round_outline, round_rect

    layout, widgets = _gallery_widgets(context)
    if not widgets:
        return
    scale = layout.scale
    pad = 10 * scale
    x0 = min(w.rect[0] for w in widgets) - pad
    y0 = min(w.rect[1] for w in widgets) - pad
    x1 = max(w.rect[0] + w.rect[2] for w in widgets) + pad
    y1 = max(w.rect[1] + w.rect[3] for w in widgets) + pad
    back = (x0, y0, x1 - x0, y1 - y0)
    round_rect(back, 12 * scale, GALLERY_BG)
    round_outline(back, 12 * scale, GALLERY_EDGE, 1.0 * scale)
    for w in widgets:
        if w.kind == kit.CARD_KIND:
            hovered = w.id == state.hover_chrome
            round_rect(w.rect, 8 * scale, CARD_HOVER if hovered else CARD_BG)
            stage, _label_h = _card_parts(w.rect, scale)
            round_rect(stage, 6 * scale, CARD_STAGE)


def _card_time(name: str, hovered: bool) -> float:
    global _card_hover
    from rok.core import motion_preview

    rest = motion_preview.REST_T.get(name, 0.0)
    if not hovered:
        return rest
    now = time.monotonic()
    if _card_hover is None or _card_hover[0] != name:
        _card_hover = (name, now)
    app = CONTEXT.app
    if app is not None:
        app.host.tick(1.0 / 30.0)  # the next picture
    return rest + (now - _card_hover[1])


def _draw_gallery(context) -> None:
    from rok.core import motion_preview

    from .paint import text

    layout, widgets = _gallery_widgets(context)
    if not widgets:
        return
    scale = layout.scale
    for w in widgets:
        x, y, ww, h = w.rect
        if w.kind == kit.CAPTION_KIND:
            text(w.label.upper(), x + 2 * scale, y + h / 2, 10 * scale, GALLERY_FAINT)
            continue
        if w.kind != kit.CARD_KIND:
            continue
        name = kit.PRESET_OF_ID[w.id]
        hovered = w.id == state.hover_chrome
        stage, label_h = _card_parts(w.rect, scale)
        sx, sy = stage[0] + stage[2] / 2, stage[1] + stage[3] / 2
        k = min(stage[2], stage[3]) / 2.25
        for shape in motion_preview.frame(name, _card_time(name, hovered)):
            pen.fan([(sx + px * k, sy + py * k) for px, py in shape.points], shape.rgba)
        text(name.title(), x + ww / 2, y + label_h / 2 + 1 * scale, 12 * scale,
             GALLERY_TEXT if hovered else GALLERY_DIM, "centre")


registry.add_draw("motion_gallery_bg", 505, "PIXEL", _draw_gallery_bg)
registry.add_draw("motion_gallery", 506, "PIXEL", _draw_gallery)
registry.load_post.append(_on_load)
