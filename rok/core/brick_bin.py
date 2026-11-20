"""The brick bin's layout: where every brick stands. Pure.

The bin is a panel on the right edge of the view while Brick mode is on. It
stands under the nav cube, so the cube and its Mirror buttons stay in reach,
and has the View Settings panel's width and paddings. Its rows hold the
bricks, each drawn as a top-down footprint: a w × d block with a line
between its cells. The keys 1 to 9 take the first nine bricks.

`bin_layout` returns `Item`s: what a part is (`kind`), the brick it holds
(`value`, (w, d)), and its rect in region pixels (origin bottom left). The
app draws them by kind and `hit` finds the brick under the mouse. When the
rows do not fit the region, the cells get smaller until they do.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geom2d import Rect, in_rect

WIDTH = 380  # as the View Settings panel
PAD = 14
MARGIN = 16
HEADER = 40
FOOT = 46
ROW_TITLE = 22
GAP = 8  # between two bricks of a row
TILE_PAD = 6  # around a footprint inside its tile
CELL = 9  # one cell of a footprint, at full size
CELL_MIN = 4
RANDOM_W = 140  # the Random colours switch beside the title
CUBE_CLEAR = 300  # the nav cube and the round view buttons under it stay clear, in region pixels

ROWS = (
    ("1 wide", ((1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 8))),
    ("2 wide", ((2, 2), (2, 3), (2, 4), (2, 6), (2, 8))),
    ("Wide", ((4, 4), (4, 6), (6, 8), (8, 8))),
)
FOOT_TEXT = "1 to 9 take a brick · R turns it · Esc empties the hand"


def bricks() -> tuple[tuple[int, int], ...]:
    """Every brick of the bin in its order: the keys 1 to 9 take the first nine."""
    return tuple(size for _title, row in ROWS for size in row)


def by_key(number: int) -> tuple[int, int] | None:
    """The brick the number key `number` (1 to 9) takes, or None."""
    every = bricks()
    return every[number - 1] if 1 <= number <= min(9, len(every)) else None


@dataclass(frozen=True)
class Item:
    kind: str  # TITLE, RANDOM (the Random colours switch), ROW, BRICK, FOOT
    rect: Rect
    text: str = ""
    value: tuple[int, int] | None = None  # BRICK: (w, d)
    key: int = 0  # BRICK: its number key, 0 past the ninth


@dataclass(frozen=True)
class BinLayout:
    rect: Rect
    items: tuple[Item, ...]
    scale: float
    cell: float  # one cell of a footprint, in region pixels

    def brick_rect(self, size) -> Rect | None:
        for item in self.items:
            if item.kind == "BRICK" and item.value == tuple(size):
                return item.rect
        return None


def tile_size(size, cell: float) -> tuple[float, float]:
    """A footprint's tile: the long side across."""
    w, d = max(size), min(size)
    return w * cell + 2 * TILE_PAD, d * cell + 2 * TILE_PAD


def _rows(scale: float, inner_w: float, cell: float):
    """[(title, [(size, x offset, y offset from the row's top)], row height)]."""
    out = []
    for title, row in ROWS:
        x = 0.0
        y = 0.0
        line_h = 0.0
        placed = []
        for size in row:
            tw, th = tile_size(size, cell)
            tw, th = tw * scale, th * scale
            if x > 0 and x + tw > inner_w:
                x = 0.0
                y += line_h + GAP * scale
                line_h = 0.0
            placed.append((size, x, y, tw, th))
            x += tw + GAP * scale
            line_h = max(line_h, th)
        out.append((title, placed, y + line_h))
    return out


def _content_h(scale: float, inner_w: float, cell: float) -> float:
    rows = _rows(scale, inner_w, cell)
    return (HEADER + FOOT) * scale + sum(ROW_TITLE * scale + h + GAP * scale for _t, _p, h in rows)


def bin_layout(scale: float, region_width: float, region_height: float) -> BinLayout:
    """The bin for a region of this size."""
    s = float(scale)
    width = min(WIDTH * s, region_width - 2 * MARGIN * s)
    x = max(MARGIN * s, region_width - width - MARGIN * s)
    top = region_height - CUBE_CLEAR
    room = max(0.0, top - MARGIN * s)
    inner_w = width - 2 * PAD * s
    cell = float(CELL)
    while cell > CELL_MIN and _content_h(s, inner_w, cell) > room:
        cell -= 0.5
    height = min(room, _content_h(s, inner_w, cell))
    rect = (x, top - height, width, height)
    items: list[Item] = []
    left = x + PAD * s
    y = top - HEADER * s
    switch_w = RANDOM_W * s
    items.append(Item("TITLE", (left, y + 6 * s, inner_w - switch_w, HEADER * s - 12 * s), "Bricks"))
    items.append(Item("RANDOM", (left + inner_w - switch_w, y + 8 * s, switch_w, HEADER * s - 16 * s),
                      "Random colours"))
    every = bricks()
    for title, placed, row_h in _rows(s, inner_w, cell):
        y -= ROW_TITLE * s
        items.append(Item("ROW", (left, y, inner_w, ROW_TITLE * s), title))
        for size, ox, oy, tw, th in placed:
            number = every.index(size) + 1
            items.append(Item("BRICK", (left + ox, y - oy - th, tw, th), f"{size[0]}×{size[1]}", size,
                              number if number <= 9 else 0))
        y -= row_h + GAP * s
    items.append(Item("FOOT", (left, rect[1] + 8 * s, inner_w, FOOT * s - 16 * s), FOOT_TEXT))
    return BinLayout(rect, tuple(items), s, cell * s)


def hit(layout: BinLayout, x: float, y: float) -> Item | None:
    """The brick under (x, y), or None; a press elsewhere in the bin is None too."""
    for item in layout.items:
        if item.kind == "BRICK" and in_rect(item.rect, x, y):
            return item
    return None


def hit_switch(layout: BinLayout, x: float, y: float) -> bool:
    """True when (x, y) is on the Random colours switch."""
    return any(item.kind == "RANDOM" and in_rect(item.rect, x, y) for item in layout.items)


def contains(layout: BinLayout, x: float, y: float) -> bool:
    return in_rect(layout.rect, x, y)


def divisions(item: Item, cell: float) -> list[tuple[float, float, float, float]]:
    """The lines between a brick tile's cells, each (x0, y0, x1, y1), the long side across."""
    w, d = max(item.value), min(item.value)
    x0 = item.rect[0] + (item.rect[2] - w * cell) / 2
    y0 = item.rect[1] + (item.rect[3] - d * cell) / 2
    lines = [(x0 + i * cell, y0, x0 + i * cell, y0 + d * cell) for i in range(1, w)]
    lines += [(x0, y0 + j * cell, x0 + w * cell, y0 + j * cell) for j in range(1, d)]
    return lines
