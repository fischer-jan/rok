"""The Stroke Designer: a floating panel of three curves.

A button above the Size field opens it. Each row is a curve from the start of
a stroke to its end, with a switch and a reset. Drag a point or a handle to
shape it. The colour row picks a second palette slot. The curves are kept
between starts (rok.app.stroke_ops).

The view still orbits and draws beside the panel. A press on the panel never
starts a stroke.
"""

from __future__ import annotations

from rok.core import stroke_design as sd
from rok.core import stroke_panel as kit
from rok.core.geom2d import in_rect

from . import opkit, stroke_ops
from .gfx import pen
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, register
from .paint import clip_text, round_outline, round_rect, text

PANEL_BG = (0.060, 0.056, 0.053, 1.0)  # opaque: the tool hint under it must not show
PANEL_EDGE = (0.228, 0.212, 0.196, 1.0)
WELL_BG = (0.034, 0.031, 0.028, 1.0)
WELL_EDGE = (0.180, 0.166, 0.152, 1.0)
GRID = (1.0, 1.0, 1.0, 0.07)
CURVE = (0.910, 0.365, 0.227, 1.0)
CURVE_FILL = (0.910, 0.365, 0.227, 0.18)
CURVE_OFF = (0.450, 0.430, 0.410, 0.70)
POINT = (0.965, 0.945, 0.915, 1.0)
HANDLE = (0.780, 0.735, 0.680, 1.0)
HANDLE_LINE = (1.0, 1.0, 1.0, 0.32)
BUTTON_BG = (1.0, 1.0, 1.0, 0.07)
BUTTON_HOVER = (0.220, 0.200, 0.185, 1.0)
EMBER = (0.910, 0.365, 0.227, 1.0)
EMBER_DARK = (0.102, 0.055, 0.031, 1.0)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)
TEXT_FAINT = (0.450, 0.430, 0.410, 1.0)
SWITCH_OFF = (1.0, 1.0, 1.0, 0.14)
SWITCH_THUMB = (0.965, 0.945, 0.915, 1.0)
HAIRLINE = (1.0, 1.0, 1.0, 0.08)

_session: "_Session | None" = None


class _Session:
    def __init__(self) -> None:
        self.x: float | None = None
        self.y: float | None = None
        self.hover: kit.Item | None = None
        self.drag: tuple | None = None  # ("point", row, which) or ("move", dx, dy)
        self.before: str | None = None  # the curves at a point drag's press: Esc goes back to them
        self.cached: tuple | None = None  # (key, layout): one layout per panel state, not per mouse move
        self.palette = False


def active() -> bool:
    return _session is not None and _shown()


def _shown() -> bool:
    """The panel shows with its button: in Voxel mode, for the tools a designed
    stroke shapes. Another tool hides it; the designer stays on for later."""
    prefs = CONTEXT.prefs
    return (getattr(prefs, "rok_mode", "") == "VOXEL"
            and getattr(prefs, "rok_tool", "") in ("ATTACH", "ERASE", "PAINT"))


def open_panel() -> None:
    global _session
    if _session is None:
        _session = _Session()


def close_panel() -> None:
    global _session
    _session = None


def contains(region, x: float, y: float) -> bool:
    """True when the pixel is on the panel or its colour grid."""
    if _session is None or region is None or not _shown():
        return False
    layout = _layout(_session, region)
    return in_rect(layout.rect, x, y) or (layout.popover is not None and in_rect(layout.popover, x, y))


def hover(region, x: float, y: float) -> bool:
    """Remember the part under the mouse. True when it changed."""
    session = _session
    if session is None or region is None or session.drag is not None:
        return False
    item = kit.hit(_layout(session, region), x, y)
    if _key(item) == _key(session.hover):
        return False
    session.hover = item
    return True


def press(context, region, x: float, y: float, left: bool = True) -> bool:
    """Act on a press. True when the panel took it, so it does not start a stroke."""
    session = _session
    if session is None or region is None or not _shown():
        return False
    if not contains(region, x, y):
        if session.palette and left:
            session.palette = False
            context.app.request_redraw()
            return True
        return False
    if not left:
        return True
    layout = _layout(session, region)
    item = kit.hit(layout, x, y)
    if item is None:
        if session.palette:
            session.palette = False
            context.app.request_redraw()
            return True  # the click only closes the colour grid
        return False
    if item.kind == "PLATE":
        return True
    if item.kind == "CLOSE":
        _turn_off(context)
        return True
    if item.kind == "TITLE":
        session.drag = ("move", x - (session.x or 0.0), y - (session.y or 0.0))
        _begin_drag(context)
        return True
    if item.kind == "PRESET":
        _apply(context, sd.PRESETS[item.key].with_second(stroke_ops.stored(context.prefs).second))
        return True
    if item.kind == "SWITCH":
        design = stroke_ops.stored(context.prefs)
        _apply(context, design.with_row(item.key, not design.row_on(item.key)))
        return True
    if item.kind == "RESET":
        design = stroke_ops.stored(context.prefs)
        _apply(context, design.with_curve(item.key, sd.reset_curve(item.key)))
        return True
    if item.kind == "SWATCH":
        session.palette = not session.palette
        context.app.request_redraw()
        return True
    if item.kind == "SLOT":
        if int(item.value) >= 1:
            _apply(context, stroke_ops.stored(context.prefs).with_second(int(item.value)))
            session.palette = False
        return True
    if item.kind == "POINT":
        session.drag = ("point", item.key, item.value)
        session.before = str(context.prefs.rok_stroke_curves)
        _move_point(context, layout, item.key, item.value, x, y, write=False)
        _begin_drag(context)
        return True
    return True


def _begin_drag(context) -> None:
    op = getattr(opkit.ops.rok, "stroke_drag", None)
    if op is not None and op.poll():
        op("INVOKE")


def _turn_off(context) -> None:
    context.prefs.rok_stroke_design = False
    stroke_ops.save(context.prefs)
    close_panel()
    if context.app is not None:
        context.app.request_redraw()


def _apply(context, design) -> None:
    stroke_ops.store(context.prefs, design)
    if context.app is not None:
        context.app.request_redraw()


def _move_point(context, layout, row: str, which: str, x: float, y: float, *, write: bool) -> None:
    px, py = kit.point_at(layout, row, which, x, y)
    design = stroke_ops.stored(context.prefs)
    stroke_ops.store(context.prefs, design.with_curve(row, design.curve(row).with_point(which, px, py)), write=write)


def _layout(session: _Session, region) -> kit.PanelLayout:
    """The panel's layout, built again only when the panel changed: the hover
    asks for it on every mouse move."""
    _place(session, region)
    packed = str(getattr(CONTEXT.prefs, "rok_stroke_curves", ""))
    key = (session.x, session.y, region.width, region.height, packed, session.palette)
    if session.cached is None or session.cached[0] != key:
        layout = kit.panel_layout(
            session.x or 0.0, session.y or 0.0, 1.0, stroke_ops.stored(CONTEXT.prefs),
            palette_open=session.palette, region_width=region.width,
        )
        session.cached = (key, layout)
    return session.cached[1]


def _place(session: _Session, region) -> None:
    height = kit.panel_height(1.0)
    if session.x is None:
        from rok.core.chrome import RAIL_WIDTH

        session.x = RAIL_WIDTH + 16.0
        session.y = max(64.0, region.height * 0.22)
    session.x = min(max(8.0, session.x), max(8.0, region.width - kit.WIDTH - 8.0))
    session.y = min(max(8.0, session.y), max(8.0, region.height - height - 8.0))


def _key(item: kit.Item | None):
    if item is None:
        return None
    return (item.kind, item.key, item.value)


# --- drawing ------------------------------------------------------------------------------------


def _draw(region) -> None:
    session = _session
    if session is None or region is None:
        return
    if not getattr(CONTEXT.prefs, "rok_stroke_design", False) or not _shown():
        return
    layout = _layout(session, region)
    s = layout.scale
    round_rect(layout.rect, 16 * s, PANEL_BG)
    round_outline(layout.rect, 16 * s, PANEL_EDGE, 1.0)
    hover = session.hover
    for item in layout.items:
        if item.kind == "POPOVER":
            continue
        lit = hover is not None and _key(hover) == _key(item)
        _draw_item(item, layout, lit, s)
    if layout.popover is not None:
        _draw_palette(layout, s, hover)


def _draw_item(item: kit.Item, layout: kit.PanelLayout, lit: bool, s: float) -> None:
    if item.kind == "TITLE":
        text(item.text, item.rect[0], item.rect[1] + item.rect[3] / 2, 15 * s, TEXT)
        pen.lines(pen.to_3d(_hair(layout.rect, item.rect[1], s)), HAIRLINE, 1.0)
    elif item.kind == "CLOSE":
        round_rect(item.rect, item.rect[3] / 2, BUTTON_HOVER if lit else BUTTON_BG)
        _cross(item.rect, TEXT if lit else TEXT_DIM, s)
    elif item.kind == "PRESET":
        round_rect(item.rect, item.rect[3] / 2, EMBER if item.on else (BUTTON_HOVER if lit else BUTTON_BG))
        text(item.text, item.rect[0] + item.rect[2] / 2, item.rect[1] + item.rect[3] / 2, 11 * s,
             EMBER_DARK if item.on else (TEXT if lit else TEXT_DIM), "centre")
    elif item.kind == "LABEL" and item.text and item.rect[2] > 60 * s:
        text(item.text, item.rect[0], item.rect[1] + item.rect[3] / 2, 12 * s, TEXT)
    elif item.kind == "LABEL" and item.text:
        text(item.text, item.rect[0], item.rect[1] + item.rect[3] / 2, 9 * s, TEXT_FAINT)
    elif item.kind == "SWITCH":
        _switch(item.rect, item.on, lit, s)
    elif item.kind == "RESET":
        ink = TEXT if lit else TEXT_FAINT
        text(item.text, item.rect[0] + item.rect[2], item.rect[1] + item.rect[3] / 2, 11 * s, ink, "right")
    elif item.kind == "SWATCH":
        _swatch(item.rect, s, lit or (_session is not None and _session.palette))
    elif item.kind == "WELL":
        _well(item, layout, s)
    elif item.kind == "NOTE":
        size = 10.5 * s
        shown = clip_text(item.text, item.rect[2], size)
        text(shown, item.rect[0], item.rect[1] + item.rect[3] / 2, size, TEXT_FAINT)
    elif item.kind == "POINT":
        _point(item, layout, lit, s)


def _switch(rect, on: bool, lit: bool, s: float) -> None:
    round_rect(rect, rect[3] / 2, EMBER if on else SWITCH_OFF)
    if lit and not on:
        round_outline(rect, rect[3] / 2, TEXT_DIM, 1.0)
    pad = 2.0 * s
    thumb = rect[3] - 2 * pad
    tx = rect[0] + rect[2] - pad - thumb if on else rect[0] + pad
    round_rect((tx, rect[1] + pad, thumb, thumb), thumb / 2, EMBER_DARK if on else SWITCH_THUMB)


def _swatch(rect, s: float, lit: bool) -> None:
    colour = _slot_rgba(stroke_ops.stored(CONTEXT.prefs).second)
    round_rect(rect, 4 * s, colour)
    round_outline(rect, 4 * s, EMBER if lit else PANEL_EDGE, 1.5 if lit else 1.0)


def _slot_rgba(index: int):
    scene = document.scene
    if scene is None:
        return (0.4, 0.4, 0.4, 1.0)
    rgb = scene.palette[int(index) % 256, :3]
    return (int(rgb[0]) / 255.0, int(rgb[1]) / 255.0, int(rgb[2]) / 255.0, 1.0)


def _well(item: kit.Item, layout: kit.PanelLayout, s: float) -> None:
    round_rect(item.rect, 8 * s, WELL_BG)
    round_outline(item.rect, 8 * s, WELL_EDGE, 1.0)
    inner = (
        item.rect[0] + kit.WELL_INSET * s, item.rect[1] + kit.WELL_INSET * s,
        item.rect[2] - 2 * kit.WELL_INSET * s, item.rect[3] - 2 * kit.WELL_INSET * s,
    )
    for frac in (0.5,):
        y = inner[1] + frac * inner[3]
        pen.lines(pen.to_3d([(inner[0], y), (inner[0] + inner[2], y)]), GRID, 1.0)
    design = stroke_ops.stored(CONTEXT.prefs)
    curve = design.curve(item.key)
    on = design.row_on(item.key)
    pts = [(_xy(inner, x, y)) for x, y in curve.samples(28)]
    if on:
        _fill_under(pts, inner[1], CURVE_FILL)
    pen.lines(pen.to_3d(_pairs(pts)), CURVE if on else CURVE_OFF, 2.0 * s)
    # Handles: a thin line from each end to its handle.
    ends = {p.value: p for p in layout.points(item.key)}
    for a, b in (("start", "h1"), ("end", "h2")):
        if a in ends and b in ends:
            pen.lines(pen.to_3d([_centre(ends[a].rect), _centre(ends[b].rect)]), HANDLE_LINE, 1.0)


def _point(item: kit.Item, layout: kit.PanelLayout, lit: bool, s: float) -> None:
    cx, cy = _centre(item.rect)
    handle = item.value in ("h1", "h2")
    r = (3.2 if handle else 4.6) * s
    if handle:
        round_rect((cx - r, cy - r, 2 * r, 2 * r), r, HANDLE if not lit else POINT)
        round_outline((cx - r, cy - r, 2 * r, 2 * r), r, WELL_BG, 1.2)
    else:
        round_rect((cx - r, cy - r, 2 * r, 2 * r), r, EMBER if lit else POINT)
        if lit:
            round_outline((cx - r - 2, cy - r - 2, 2 * r + 4, 2 * r + 4), r + 2, EMBER, 1.0)


def _draw_palette(layout: kit.PanelLayout, s: float, hover: kit.Item | None) -> None:
    pop = layout.popover
    if pop is None:
        return
    round_rect(pop, 12 * s, PANEL_BG)
    round_outline(pop, 12 * s, PANEL_EDGE, 1.0)
    text("Second colour", pop[0] + 12 * s, pop[1] + pop[3] - 16 * s, 11 * s, TEXT_DIM)
    current = int(document.colour)
    second = stroke_ops.stored(CONTEXT.prefs).second
    for item in layout.items:
        if item.kind != "SLOT":
            continue
        index = int(item.value)
        colour = (0.12, 0.11, 0.10, 1.0) if index == 0 else _slot_rgba(index)
        round_rect(item.rect, 2 * s, colour)
        if index == second:
            round_outline(item.rect, 2 * s, EMBER, 1.5)
        elif index == current:
            round_outline(item.rect, 2 * s, POINT, 1.0)
        elif hover is not None and hover.kind == "SLOT" and hover.value == index:
            round_outline(item.rect, 2 * s, TEXT, 1.0)


def _fill_under(pts, bottom: float, colour) -> None:
    tris = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        tris.extend([(x0, bottom), (x1, bottom), (x1, y1), (x0, bottom), (x1, y1), (x0, y0)])
    if tris:
        pen.tris(pen.to_3d(tris), colour)


def _pairs(pts):
    out = []
    for a, b in zip(pts, pts[1:]):
        out.extend((a, b))
    return out


def _xy(inner, px: float, py: float) -> tuple[float, float]:
    return (inner[0] + px * inner[2], inner[1] + py * inner[3])


def _centre(rect) -> tuple[float, float]:
    return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)


def _hair(plate, y: float, s: float):
    return [(plate[0] + 14 * s, y), (plate[0] + plate[2] - 14 * s, y)]


# --- the actions --------------------------------------------------------------------------------


@register
class StrokeDesignAction(Action):
    command_id = "rok.stroke_design"
    title = "Stroke"
    tooltip = "Shape thickness, roundness and colour along a stroke"

    def execute(self, context):
        prefs = context.prefs
        prefs.rok_stroke_design = not prefs.rok_stroke_design
        stroke_ops.save(prefs)
        if prefs.rok_stroke_design:
            open_panel()
        else:
            close_panel()
        if context.app is not None:
            context.app.request_redraw()
        return {"FINISHED"}


@register
class StrokeDragAction(Action):
    """A drag of a curve point, or of the panel itself."""

    command_id = "rok.stroke_drag"
    title = "Stroke Drag"
    tooltip = "Drag a point of a stroke curve, or drag the panel"

    @classmethod
    def poll(cls, context):
        return _session is not None and _session.drag is not None

    def invoke(self, context, event):
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        session = _session
        if session is None or session.drag is None:
            return {"CANCELLED"}
        region = context.region
        if region is None:
            return {"PASS"}
        mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
        kind, val = event.type, event.value
        if session.drag[0] == "move" and kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            _dx, dy = session.drag[1], session.drag[2]
            session.x = mx - session.drag[1]
            session.y = my - dy
            context.app.request_redraw()
            return {"RUNNING"}
        if session.drag[0] == "point" and kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            _row, which = session.drag[1], session.drag[2]
            _move_point(context, _layout(session, region), session.drag[1], which, mx, my, write=False)
            context.app.request_redraw()
            return {"RUNNING"}
        if kind == "FOCUS_LOST" or (kind == "LEFT_MOUSE" and val == "RELEASE"):
            _end_drag(context, session)
            return {"FINISHED"}
        if val == "PRESS" and kind == "ESCAPE":
            if session.drag[0] == "point" and session.before is not None:
                context.prefs.rok_stroke_curves = session.before  # the point goes back
            _end_drag(context, session)
            return {"CANCELLED"}
        return {"RUNNING"}


def _end_drag(context, session: _Session) -> None:
    if session.drag is not None and session.drag[0] == "point":
        stroke_ops.save(context.prefs)
    session.drag = None
    session.before = None
    if context.app is not None:
        context.app.request_redraw()


def _cross(rect, ink, s: float) -> None:
    cx, cy = _centre(rect)
    d = 4.0 * s
    pen.lines(pen.to_3d([(cx - d, cy - d), (cx + d, cy + d), (cx + d, cy - d), (cx - d, cy + d)]), ink, 1.4 * s)


def _draw_handler(context) -> None:
    if _session is not None and context.region is not None:
        _draw(context.region)


registry.add_draw("stroke_panel", 762, "PIXEL", _draw_handler)


def _open_if_kept(context) -> None:
    """A designer that was on at the last start opens again."""
    if getattr(context.prefs, "rok_stroke_design", False):
        open_panel()


registry.startup.append(_open_if_kept)
