"""The Stroke Designer panel's layout: where every part stands. Pure.

The panel floats over the view, to the right of the rail. Three rows, each a
curve from the stroke's start (left) to its end (right). Points and handles
can be dragged. A switch turns a row off, which leaves that part of the stroke
as it is. The colour row picks a second palette slot; the grid of slots opens
beside the panel.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geom2d import Rect, in_rect
from .stroke_design import PRESET_LABELS, PRESET_ORDER, ROWS, Design, same_preset

WIDTH = 356
PAD = 14
TITLE_H = 36
PRESET_H = 26
LABEL_H = 22
WELL_H = 78
FOOT_H = 26
GAP = 8
GUTTER = 46
WELL_INSET = 12
POINT_HIT = 16
SWITCH_W = 34
SWITCH_H = 16
SWATCH = 18
SLOT = 14
SLOT_GAP = 2
SLOT_COLUMNS = 16
ROW_NAMES = {"thickness": "Thickness", "roundness": "Roundness", "colour": "Colour"}
# Words at the top and bottom of a row's curve, so the axes read without a key.
ENDS = {
    "thickness": ("Full", "1"),
    "roundness": ("Cube", "Diamond"),
    "colour": ("To", "Now"),
}


@dataclass(frozen=True)
class Item:
    kind: str  # TITLE CLOSE PRESET LABEL SWITCH RESET WELL POINT SWATCH NOTE SLOT POPOVER
    key: str
    rect: Rect
    text: str = ""
    value: object = None
    on: bool = False


@dataclass(frozen=True)
class PanelLayout:
    rect: Rect
    scale: float
    items: tuple[Item, ...]
    popover: Rect | None = None

    def find(self, kind: str, key: str = "", value=None) -> Item | None:
        return next(
            (i for i in self.items if i.kind == kind and (not key or i.key == key)
             and (value is None or i.value == value)),
            None,
        )

    def points(self, row: str) -> tuple[Item, ...]:
        return tuple(i for i in self.items if i.kind == "POINT" and i.key == row)


def panel_height(scale: float = 1.0) -> float:
    s = float(scale)
    rows = len(ROWS) * (LABEL_H + WELL_H + GAP)
    return (PAD + TITLE_H + GAP + PRESET_H + GAP + rows + FOOT_H + PAD) * s


def panel_layout(x: float, y: float, scale: float, design: Design, *, palette_open: bool = False,
                  region_width: float = 10_000) -> PanelLayout:
    """The panel with its bottom-left corner at (x, y), in region pixels."""
    s = float(scale)
    width = WIDTH * s
    height = panel_height(s)
    top = y + height
    items: list[Item] = []
    cursor = top  # the top of the next row, moving down

    def take(h: float) -> Rect:
        nonlocal cursor
        h *= s
        cursor -= h
        return (x, cursor, width, h)

    def gap(g: float = GAP) -> None:
        nonlocal cursor
        cursor -= g * s

    take(PAD)
    title = take(TITLE_H)
    items.append(Item("TITLE", "", (title[0] + PAD * s, title[1], title[2] - 52 * s, title[3]), "Stroke"))
    close = (title[0] + title[2] - (PAD + 22) * s, title[1] + (TITLE_H - 22) * s / 2, 22 * s, 22 * s)
    items.append(Item("CLOSE", "", close, "×"))
    gap()
    presets = take(PRESET_H)
    _preset_row(items, presets, s, design)
    gap()
    for row in ROWS:
        label = take(LABEL_H)
        _label_row(items, label, s, row, design)
        well = take(WELL_H)
        _well(items, well, s, row, design)
        gap()
    foot = take(FOOT_H)
    items.append(Item(
        "NOTE", "", (foot[0] + PAD * s, foot[1], foot[2] - 2 * PAD * s, foot[3]),
        "Drawing uses each curve's start. Letting go applies the whole curve.",
    ))
    take(PAD)
    plate = (x, y, width, height)
    popover = None
    if palette_open:
        popover = _palette(items, plate, s, design, region_width)
    return PanelLayout(plate, s, tuple(items), popover)


def hit(layout: PanelLayout, px: float, py: float) -> Item | None:
    """The part under the pixel, or a stand-in when the pixel is on the plate
    but on no part, or None outside the panel."""
    inside = in_rect(layout.rect, px, py) or (layout.popover is not None and in_rect(layout.popover, px, py))
    if not inside:
        return None
    found = [i for i in layout.items if i.kind not in ("LABEL", "NOTE", "WELL", "POPOVER") and in_rect(i.rect, px, py)]
    if not found:
        return Item("PLATE", "", layout.rect)
    rank = {"POINT": 0, "SLOT": 1, "SWITCH": 2, "RESET": 2, "SWATCH": 2, "CLOSE": 2, "PRESET": 3, "TITLE": 4}
    return min(found, key=lambda i: (rank.get(i.kind, 5), _dist(i, px, py)))


def point_at(layout: PanelLayout, row: str, which: str, mx: float, my: float) -> tuple[float, float]:
    """Curve coordinates (x along the stroke, y the value) for a drag of `which`."""
    well = layout.find("WELL", row)
    if well is None:
        return (0.0, 0.0)
    inner = _inner(well.rect, layout.scale)
    x = (mx - inner[0]) / max(inner[2], 1.0)
    y = (my - inner[1]) / max(inner[3], 1.0)
    if which == "start":
        x = 0.0
    elif which == "end":
        x = 1.0
    return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))


def _preset_row(items: list[Item], row: Rect, s: float, design: Design) -> None:
    gap = 6 * s
    inner = row[2] - 2 * PAD * s
    w = (inner - gap * (len(PRESET_ORDER) - 1)) / len(PRESET_ORDER)
    x = row[0] + PAD * s
    for name in PRESET_ORDER:
        items.append(Item(
            "PRESET", name, (x, row[1], w, row[3]), PRESET_LABELS[name], on=same_preset(design, name),
        ))
        x += w + gap


def _label_row(items: list[Item], row: Rect, s: float, name: str, design: Design) -> None:
    x = row[0] + PAD * s
    items.append(Item("LABEL", name, (x, row[1], 92 * s, row[3]), ROW_NAMES[name]))
    switch = (x + 96 * s, row[1] + (row[3] - SWITCH_H * s) / 2, SWITCH_W * s, SWITCH_H * s)
    items.append(Item("SWITCH", name, switch, on=design.row_on(name)))
    if name == "colour":
        sw = (switch[0] + switch[2] + 10 * s, row[1] + (row[3] - SWATCH * s) / 2, SWATCH * s, SWATCH * s)
        items.append(Item("SWATCH", name, sw, "To"))
    reset_w = 48 * s
    items.append(Item("RESET", name, (row[0] + row[2] - PAD * s - reset_w, row[1], reset_w, row[3]), "Reset"))


def _well(items: list[Item], row: Rect, s: float, name: str, design: Design) -> None:
    well = (row[0] + (PAD + GUTTER) * s, row[1], row[2] - (2 * PAD + GUTTER) * s, row[3])
    items.append(Item("WELL", name, well))
    top, bottom = ENDS[name]
    items.append(Item("LABEL", name, (row[0] + PAD * s, well[1] + well[3] - 14 * s, GUTTER * s, 14 * s), top))
    items.append(Item("LABEL", name, (row[0] + PAD * s, well[1], GUTTER * s, 14 * s), bottom))
    if name == "roundness":
        items.append(Item(
            "LABEL", name,
            (row[0] + PAD * s, well[1] + well[3] / 2 - 7 * s, GUTTER * s, 14 * s), "Sphere",
        ))
    inner = _inner(well, s)
    curve = design.curve(name)
    for which, px, py in (
        ("start", 0.0, curve.y0), ("end", 1.0, curve.y3),
        ("h1", curve.x1, curve.y1), ("h2", curve.x2, curve.y2),
    ):
        cx = inner[0] + px * inner[2]
        cy = inner[1] + py * inner[3]
        hit = POINT_HIT * s
        items.append(Item("POINT", name, (cx - hit / 2, cy - hit / 2, hit, hit), value=which))


def _palette(items: list[Item], plate: Rect, s: float, design: Design, region_width: float) -> Rect:
    cols = SLOT_COLUMNS
    rows = 256 // cols
    gap = SLOT_GAP * s
    cell = SLOT * s
    inner_w = cols * cell + (cols - 1) * gap
    inner_h = rows * cell + (rows - 1) * gap
    pad = 10 * s
    w, h = inner_w + 2 * pad, inner_h + 2 * pad + 22 * s
    right = plate[0] + plate[2] + 8 * s
    if right + w > region_width - 8:
        right = plate[0] - 8 * s - w
    bottom = max(8.0, plate[1] + plate[3] - h)
    rect = (right, bottom, w, h)
    items.append(Item("POPOVER", "", rect, "Second colour"))
    origin_x = right + pad
    origin_y = bottom + pad
    for index in range(256):
        col, row = index % cols, index // cols
        # Row 0 of the palette at the top of the grid.
        sx = origin_x + col * (cell + gap)
        sy = origin_y + (rows - 1 - row) * (cell + gap)
        items.append(Item("SLOT", "", (sx, sy, cell, cell), value=index, on=index == design.second))
    return rect


def _inner(well: Rect, s: float) -> Rect:
    inset = WELL_INSET * s
    return (well[0] + inset, well[1] + inset, max(1.0, well[2] - 2 * inset), max(1.0, well[3] - 2 * inset))


def _dist(item: Item, px: float, py: float) -> float:
    cx = item.rect[0] + item.rect[2] / 2
    cy = item.rect[1] + item.rect[3] / 2
    return (cx - px) ** 2 + (cy - py) ** 2
