"""Palette layout, overlay geometry and the new-UI tool HUD. Pure maths."""

from __future__ import annotations

import math

from dataclasses import dataclass, replace

import numpy as np

from .edit import ATTACH, ERASE, MODE_ITEMS, PAINT, SELECT, effective_tool
from .geom2d import QUAD_TRIS, Rect, in_rect, rect_tris
from .mesh import _CORE, _FACE_CORNERS, _NEIGHBOR_SLICES

PALETTE_COLUMNS = 16
PALETTE_ROWS = 16

# Colour of slot 0. It is never pickable, so it is drawn dark grey.
SLOT0_COLOUR = (0.15, 0.15, 0.15, 1.0)

# Row of _FACE_CORNERS for each face normal.
_NORMAL_ROWS = {
    (1, 0, 0): 0,
    (-1, 0, 0): 1,
    (0, 1, 0): 2,
    (0, -1, 0): 3,
    (0, 0, 1): 4,
    (0, 0, -1): 5,
}

# Outward normal of each _FACE_CORNERS row.
_ROW_NORMALS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.float32,
)

# Index patterns that turn one quad into two triangles and into four LINES pairs.
_QUAD_LINES = np.array([0, 1, 1, 2, 2, 3, 3, 0])

# The 12 edges of the unit cube as pairs of corner indices.
_CUBE_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float32,
)
_CUBE_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class PaletteLayout:
    """Where the palette sits in the region, in pixels."""

    x0: int  # bottom-left corner of the whole palette
    y0: int
    cell: int  # swatch size

    @property
    def width(self) -> int:
        return PALETTE_COLUMNS * self.cell

    @property
    def height(self) -> int:
        return PALETTE_ROWS * self.cell


STRIP_MARGIN = 8  # classic left/bottom inset; new UI x0 is rail + this


def palette_x0(scale: float, rail_width: float = 0) -> int:
    """Left edge of the colour strip and the 256 grid, pixel-snapped.

    Classic has no rail (`rail_width=0`): STRIP_MARGIN from the region edge.
    New UI passes the rail width so the strip sits that far plus the same margin.
    """
    return round((float(rail_width) + STRIP_MARGIN) * scale)


def make_layout(
    scale: float, y0: int | None = None, *, rail_width: float = 0
) -> PaletteLayout:
    """Layout for a UI scale. scale = dpi / 72, so 2.0 on a Retina Mac.

    y0 is the bottom of the grid; it defaults to the margin. With the strip
    shown, pass StripLayout.grid_y0 so the grid sits above the strip and text.

    rail_width is the left rail in points (0 in classic). New UI passes
    HUD_RAIL_PX so the grid's left edge is past the plate.
    """
    cell = round(16 * scale)
    margin = round(STRIP_MARGIN * scale)
    return PaletteLayout(
        x0=palette_x0(scale, rail_width),
        y0=margin if y0 is None else int(y0),
        cell=cell,
    )


# --- Phase 3: the palette strip -----------------------------------------------

STRIP_SWATCH = 16
STRIP_CURRENT = 40
STRIP_BUTTON = 20
STRIP_GAP = 4
STRIP_TEXT_BAND = 24  # room for the status text between the strip and the grid
STRIP_CAPTION = 14  # the column left of the rows that holds each section's name, turned upright
STRIP_CAPTION_SIZE = 8  # its text size, small capitals
STRIP_CAPTIONS = ("USED", "UNUSED", "FAV")
STRIP_USED_COLUMNS = 16
STRIP_USED_MAX = 48
# No 256 button: the full grid is not a daily tool; Cmd+K
# "Show All 256 Colours" still opens it.
STRIP_BUTTONS = ("NEW", "EYEDROP")


@dataclass(frozen=True)
class StripLayout:
    """Where the palette strip sits in the region, in pixels.

    Rows from the bottom up: the current swatch with the buttons, then the
    colours the model uses, 16 per row, then (a gap above them) the colours
    you made and have not used yet.
    """

    x0: int  # bottom-left corner of the strip
    y0: int
    swatch: int  # 16 px * scale
    current: Rect  # the current swatch, 40 px * scale square
    buttons: tuple[tuple[str, Rect], ...]  # ('NEW', rect), ('EYEDROP', rect), ('GRID', rect)
    used: tuple[tuple[int, Rect], ...]  # (key, rect), in your order
    width: int  # bounding width; strip_contains uses it
    height: int  # bounding height; the status text sits above it
    grid_y0: int  # y of the 256 grid when it is shown, above the text
    made: tuple[tuple[int, Rect], ...] = ()  # unused colours: (key, rect), in your order
    divider: int | None = None  # y of the line between the two sections, None without made colours
    # Favourite colours, on top: (k, rect), k their place in
    # the favourites list; RGB colours, not palette slots, so they stay with
    # the grid shown too.
    favourites: tuple[tuple[int, Rect], ...] = ()
    fav_divider: int | None = None  # y of the line under the favourites
    captions: tuple[tuple[str, Rect], ...] = ()  # each section's name, in the column left of its rows
    # Hit cells ("SWATCH" or "FAV", key, cell) when a swatch's target is larger than
    # its drawn rect (the dock card: swatch + gap, centred); empty uses the rects.
    cells: tuple[tuple[str, int, Rect], ...] = ()
    @property
    def swatches(self) -> tuple[tuple[int, Rect], ...]:
        """Every swatch of both sections."""
        return self.used + self.made

    def section_of(self, index: int) -> str | None:
        """"USED", "MADE" or None for a colour not on the strip."""
        if any(i == index for i, _r in self.used):
            return "USED"
        if any(i == index for i, _r in self.made):
            return "MADE"
        return None


def strip_layout(
    scale: float, used: tuple[int, ...], show_grid: bool, *, rail_width: float = 0, made: tuple[int, ...] = (),
    favourites: int = 0, caption_length=None,
) -> StripLayout:
    """Strip layout for a UI scale.

    When the 256 grid is shown, the used rows are left out: the grid lists
    every colour, so the strip is only the current swatch and buttons.

    rail_width is the left rail in points (0 in classic). New UI passes
    HUD_RAIL_PX so the strip's left edge is past the plate; hit tests use
    the same x0, so a click under the rail never hits a swatch.

    Each section's name reads upright in a column left of its rows; a section
    is at least as tall as its name. caption_length(text) gives a name's
    length in points at STRIP_CAPTION_SIZE (the host measures it; without
    one, an estimate).
    """
    if caption_length is None:
        caption_length = _caption_estimate
    margin = round(STRIP_MARGIN * scale)
    swatch = round(STRIP_SWATCH * scale)
    cur = round(STRIP_CURRENT * scale)
    btn = round(STRIP_BUTTON * scale)
    gap = round(STRIP_GAP * scale)
    x0 = palette_x0(scale, rail_width)
    y0 = margin

    current = (x0, y0, cur, cur)
    buttons = []
    bx = x0 + cur + gap
    by = y0 + (cur - btn) // 2
    for name in STRIP_BUTTONS:
        buttons.append((name, (bx, by, btn, btn)))
        bx += btn + gap
    width = bx - gap - x0
    top = y0 + cur

    used_rects: list[tuple[int, Rect]] = []
    made_rects: list[tuple[int, Rect]] = []
    captions: list[tuple[str, Rect]] = []
    caption_w = round(STRIP_CAPTION * scale)
    sx = x0 + caption_w  # the rows start right of the caption column

    def section_height(name: str, rows: int) -> int:
        return max(rows * swatch, math.ceil(caption_length(name) * scale))

    divider = None
    if not show_grid:
        # In the order given (your order); a key past 255 is a colour that holds no slot.
        shown_used = list(dict.fromkeys(int(i) for i in used if 0 < int(i) < 256))[:STRIP_USED_MAX]
        used_set = set(shown_used)
        shown_made = [i for i in dict.fromkeys(int(i) for i in made if 0 < int(i)) if i not in used_set]
        shown_made = shown_made[:STRIP_USED_MAX]
        for indices, out, section_gap in ((shown_used, used_rects, gap), (shown_made, made_rects, 3 * gap)):
            if not indices:
                continue
            if out is made_rects and used_rects:
                divider = top + int(round(1.5 * gap))
            y = top + section_gap
            for k, index in enumerate(indices):
                col = k % STRIP_USED_COLUMNS
                row = k // STRIP_USED_COLUMNS
                out.append((index, (sx + col * swatch, y + row * swatch, swatch, swatch)))
            rows = (len(indices) + STRIP_USED_COLUMNS - 1) // STRIP_USED_COLUMNS
            width = max(width, caption_w + min(len(indices), STRIP_USED_COLUMNS) * swatch)
            name = STRIP_CAPTIONS[0] if out is used_rects else STRIP_CAPTIONS[1]
            h = section_height(name, rows)
            captions.append((name, (x0, y, caption_w, h)))
            top = y + h

    fav_rects: list[tuple[int, Rect]] = []
    fav_divider = None
    if favourites:
        if top > y0 + cur:  # colour rows below: a line between them and the favourites
            fav_divider = top + int(round(1.5 * gap))
        y = top + 3 * gap
        for k in range(int(favourites)):
            col, row = k % STRIP_USED_COLUMNS, k // STRIP_USED_COLUMNS
            fav_rects.append((k, (sx + col * swatch, y + row * swatch, swatch, swatch)))
        rows = (int(favourites) + STRIP_USED_COLUMNS - 1) // STRIP_USED_COLUMNS
        width = max(width, caption_w + min(int(favourites), STRIP_USED_COLUMNS) * swatch)
        h = section_height(STRIP_CAPTIONS[2], rows)
        captions.append((STRIP_CAPTIONS[2], (x0, y, caption_w, h)))
        top = y + h

    height = top - y0
    return StripLayout(
        x0=x0,
        y0=y0,
        swatch=swatch,
        current=current,
        buttons=tuple(buttons),
        used=tuple(used_rects),
        width=width,
        height=height,
        grid_y0=top + round(STRIP_TEXT_BAND * scale),
        made=tuple(made_rects),
        divider=divider,
        favourites=tuple(fav_rects),
        fav_divider=fav_divider,
        captions=tuple(captions),
    )


def _caption_estimate(text: str) -> float:
    """A caption's length in points without a host to measure it."""
    return len(text) * STRIP_CAPTION_SIZE * 0.56


def strip_hit_test(layout: StripLayout, px: float, py: float) -> tuple[str, int] | None:
    """What sits under the region pixel.

    ('CURRENT', 0), ('NEW', 0), ('EYEDROP', 0), ('GRID', 0), ('SWATCH', index),
    ('FAV', k) for the k-th favourite colour, or None. The caller knows the current index, so CURRENT carries 0.
    """
    if in_rect(layout.current, px, py):
        return ("CURRENT", 0)
    for name, rect in layout.buttons:
        if in_rect(rect, px, py):
            return (name, 0)
    if layout.cells:
        # This block returns None once the cells are checked: a target added after
        # the swatches must go above it.
        for kind, key, rect in layout.cells:
            if in_rect(rect, px, py):
                return (kind, key)
        return None
    for index, rect in layout.swatches:
        if in_rect(rect, px, py):
            return ("SWATCH", index)
    for k, rect in layout.favourites:
        if in_rect(rect, px, py):
            return ("FAV", k)
    return None


def strip_contains(layout: StripLayout, px: float, py: float) -> bool:
    """True inside the strip's bounding rectangle."""
    return in_rect((layout.x0, layout.y0, layout.width, layout.height), px, py)


SWATCH_FLY_S = 0.6  # a swatch's flight from one section to the other


def swatch_moves(previous: dict[int, tuple[str, Rect]], layout: StripLayout) -> dict[int, Rect]:
    """The swatches that changed section since `previous` ({index: (section,
    rect)} as last drawn): {index: the rect it flies from}. A colour that
    was not on the strip before, or that left it, does not fly."""
    out = {}
    for section, items in (("USED", layout.used), ("MADE", layout.made)):
        for index, _rect in items:
            before = previous.get(index)
            if before is not None and before[0] != section:
                out[index] = before[1]
    return out


def flying_rect(start: Rect, end: Rect, t: float) -> Rect:
    """Where a flying swatch is at `t` (0 to 1 of its flight): it rises in an
    arc, grows a little on the way, and pops as it lands."""
    t = min(max(float(t), 0.0), 1.0)
    e = t * t * (3.0 - 2.0 * t)
    sx, sy, sw, sh = start
    ex, ey, ew, eh = end
    cx = (sx + sw / 2) + ((ex + ew / 2) - (sx + sw / 2)) * e
    cy = (sy + sh / 2) + ((ey + eh / 2) - (sy + sh / 2)) * e + 2.5 * eh * math.sin(math.pi * e)
    grow = 1.0 + 0.35 * math.sin(math.pi * e)
    if t > 0.8:
        grow += 0.25 * math.sin((t - 0.8) / 0.2 * math.pi)  # the pop as it lands
    w = (sw + (ew - sw) * e) * grow
    h = (sh + (eh - sh) * e) * grow
    return (cx - w / 2, cy - h / 2, w, h)


def strip_quads(
    layout: StripLayout, palette_linear: np.ndarray, rects: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Used swatches as a TRIS batch: positions (Q * 6, 2), colours (Q * 6, 4).

    The current swatch is not in the batch; its colour changes without the
    layout changing, so the overlay draws it on its own.
    """
    items = layout.swatches
    if not items:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    rects = rects or {}
    # Flying swatches last, so they pass over the others.
    items = sorted(items, key=lambda item: item[0] in rects)
    positions = rect_tris(np.array([rects.get(index, rect) for index, rect in items], dtype=np.float32))
    indices = np.array([index for index, _ in items], dtype=np.int64)
    colours = np.asarray(palette_linear, dtype=np.float32)[indices]
    colours = np.repeat(colours, 6, axis=0).astype(np.float32)
    return positions, colours


# --- the material marker: a dot on every swatch whose slot is not Plain ---------

MARK_PX = 3  # the dot's side on a 16 px swatch, times scale; doubled on the current swatch
MARK_INSET = 2  # from the swatch's top-right corner, times scale
MARK_OUTLINE = 1  # the dark ring, times scale


def _marker_rect(rect: Rect, scale: float, big: bool = False) -> Rect:
    """The dot rect in the top-right corner of `rect`."""
    x, y, w, h = rect
    side = max(1, round(MARK_PX * scale)) * (2 if big else 1)
    inset = round(MARK_INSET * scale)
    return (x + w - inset - side, y + h - inset - side, side, side)


def strip_marker_rects(
    layout: StripLayout, type_codes: np.ndarray, colour: int
) -> tuple[Rect, ...]:
    """The dot rect of every used swatch whose slot is not Plain, plus the current
    swatch's doubled dot when its slot is not Plain."""
    codes = np.asarray(type_codes)
    scale = layout.swatch / STRIP_SWATCH
    rects = [
        _marker_rect(rect, scale) for index, rect in layout.swatches if codes[index] != 0
    ]
    if 0 <= int(colour) < codes.shape[0] and codes[int(colour)] != 0:
        rects.append(_marker_rect(layout.current, scale, big=True))
    return tuple(rects)


def palette_marker_rects(layout: PaletteLayout, type_codes: np.ndarray) -> tuple[Rect, ...]:
    """The dot rect of every cell 1..255 whose slot is not Plain."""
    codes = np.asarray(type_codes)
    scale = layout.cell / 16
    return tuple(
        _marker_rect(cell_rect(layout, index), scale)
        for index in range(1, min(256, codes.shape[0]))
        if codes[index] != 0
    )


def marker_quads(rects: tuple[Rect, ...], outline: int) -> tuple[np.ndarray, np.ndarray]:
    """(dark, white) TRIS batches, each (M * 6, 2) float32.

    The white quads are the rects; the dark quads are the rects grown by
    `outline` on every side. Both are empty for no rects.
    """
    if not rects:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty.copy()
    white = np.array(rects, dtype=np.float32).reshape(-1, 4)
    dark = white.copy()
    dark[:, 0:2] -= outline
    dark[:, 2:4] += 2 * outline
    return rect_tris(dark), rect_tris(white)


def cell_rect(layout: PaletteLayout, index: int) -> tuple[int, int, int, int]:
    """Rectangle (x, y, w, h) of one swatch. Index 0 is top-left, 255 bottom-right."""
    col = index % PALETTE_COLUMNS
    row = index // PALETTE_COLUMNS
    x = layout.x0 + col * layout.cell
    y = layout.y0 + (PALETTE_ROWS - 1 - row) * layout.cell
    return x, y, layout.cell, layout.cell


def contains(layout: PaletteLayout, px: float, py: float) -> bool:
    """True when the region pixel lies inside the whole palette rectangle."""
    inside_x = layout.x0 <= px < layout.x0 + layout.width
    inside_y = layout.y0 <= py < layout.y0 + layout.height
    return bool(inside_x and inside_y)


def hit_test(layout: PaletteLayout, px: float, py: float) -> int | None:
    """Palette index under the region pixel, or None outside and on slot 0."""
    if not contains(layout, px, py):
        return None
    col = int((px - layout.x0) // layout.cell)
    row_from_bottom = int((py - layout.y0) // layout.cell)
    row = PALETTE_ROWS - 1 - row_from_bottom
    index = row * PALETTE_COLUMNS + col
    if index == 0:
        return None
    return index


def palette_quads(
    layout: PaletteLayout, palette_linear: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Positions (1536, 2) and colours (1536, 4) for a TRIS batch of 256 swatches."""
    count = PALETTE_COLUMNS * PALETTE_ROWS
    index = np.arange(count)
    col = index % PALETTE_COLUMNS
    row = index // PALETTE_COLUMNS
    x = layout.x0 + col * layout.cell
    y = layout.y0 + (PALETTE_ROWS - 1 - row) * layout.cell
    c = layout.cell
    # Corners in order: bottom-left, bottom-right, top-right, top-left.
    corners = np.stack(
        [
            np.stack([x, y], axis=1),
            np.stack([x + c, y], axis=1),
            np.stack([x + c, y + c], axis=1),
            np.stack([x, y + c], axis=1),
        ],
        axis=1,
    ).astype(np.float32)  # (256, 4, 2)
    # Two triangles per swatch.
    tri = np.array([0, 1, 2, 0, 2, 3])
    positions = corners[:, tri, :].reshape(count * 6, 2)

    colours = np.asarray(palette_linear, dtype=np.float32).copy()
    colours[0] = SLOT0_COLOUR
    colours = np.repeat(colours, 6, axis=0)
    return positions, colours


def rounded_rect(
    x: float, y: float, w: float, h: float, radius: float, segments: int = 6
) -> np.ndarray:
    """A rounded rectangle as a closed polygon, (4 * (segments + 1), 2) float32,
    counter-clockwise from the bottom-right corner arc. A TRI_FAN draws it."""
    r = max(0.0, min(float(radius), w / 2.0, h / 2.0))
    n = max(1, int(segments))
    corners = ((x + w - r, y + r, -0.5), (x + w - r, y + h - r, 0.0),
               (x + r, y + h - r, 0.5), (x + r, y + r, 1.0))
    points = []
    for cx, cy, start in corners:
        for k in range(n + 1):
            a = (start + 0.5 * k / n) * np.pi
            points.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(points, dtype=np.float32)


def rect_outline(
    x: float, y: float, w: float, h: float, inset: float = 0.0
) -> np.ndarray:
    """Four edges of a rectangle as LINES pairs, (8, 2) float32."""
    x0 = x + inset
    y0 = y + inset
    x1 = x + w - inset
    y1 = y + h - inset
    return np.array(
        [
            [x0, y0], [x1, y0],
            [x1, y0], [x1, y1],
            [x1, y1], [x0, y1],
            [x0, y1], [x0, y0],
        ],
        dtype=np.float32,
    )


def srgb_to_linear(rgba_uint8: np.ndarray) -> np.ndarray:
    """Standard piecewise sRGB curve on RGB. Alpha is only divided by 255."""
    c = np.asarray(rgba_uint8, dtype=np.float32) / 255.0
    out = c.copy()
    rgb = c[..., :3]
    low = rgb / 12.92
    high = ((rgb + 0.055) / 1.055) ** 2.4
    out[..., :3] = np.where(rgb <= 0.04045, low, high)
    return out.astype(np.float32)


def face_quad(cell, normal, offset: float = 0.003) -> np.ndarray:
    """Corners (4, 3) of one face of the cell, pushed `offset` along the normal."""
    n = tuple(int(v) for v in normal)
    row = _NORMAL_ROWS[n]
    base = np.asarray(cell, dtype=np.float32)
    push = np.asarray(n, dtype=np.float32) * np.float32(offset)
    return (base + _FACE_CORNERS[row] + push).astype(np.float32)


def cube_edges(cell) -> np.ndarray:
    """The 12 edges of the unit cube at cell as LINES pairs, (24, 3) float32."""
    base = np.asarray(cell, dtype=np.float32)
    corners = _CUBE_CORNERS + base
    return corners[_CUBE_EDGES.reshape(-1)].astype(np.float32)


def box_edges(size) -> np.ndarray:
    """The 12 edges of the box [0, sx] x [0, sy] x [0, sz], (24, 3) float32."""
    scale = np.asarray(size, dtype=np.float32)
    corners = _CUBE_CORNERS * scale
    return corners[_CUBE_EDGES.reshape(-1)].astype(np.float32)


# --- Phase 2: many cells at once, no Python loop over cells ------------------


def _cells_array(cells) -> np.ndarray:
    """Cells as an (N, 3) float32 array; an empty input gives (0, 3)."""
    arr = np.asarray(cells, dtype=np.float32)
    return arr.reshape(-1, 3)


def cells_face_quads(cells, normal, offset: float = 0.003) -> np.ndarray:
    """One face per cell on the normal side, (N * 6, 3) float32 TRIS."""
    n = tuple(int(v) for v in normal)
    row = _NORMAL_ROWS[n]
    base = _cells_array(cells)
    push = np.asarray(n, dtype=np.float32) * np.float32(offset)
    corners = base[:, None, :] + _FACE_CORNERS[row][None, :, :] + push  # (N, 4, 3)
    return corners[:, QUAD_TRIS, :].reshape(-1, 3).astype(np.float32)


def cells_edges(cells) -> np.ndarray:
    """cube_edges for every cell, (N * 24, 3) float32 LINES."""
    base = _cells_array(cells)
    unit = _CUBE_CORNERS[_CUBE_EDGES.reshape(-1)]  # (24, 3)
    edges = base[:, None, :] + unit[None, :, :]
    return edges.reshape(-1, 3).astype(np.float32)


_NATIVE: list = []  # [module or None], once looked up


def _native():
    """rok_native's surface builder, or None (no native module, or ROK_PURE_HUD set)."""
    if not _NATIVE:
        import os

        mod = None
        if not os.environ.get("ROK_PURE_HUD"):
            try:
                import rok_native

                mod = rok_native if hasattr(rok_native, "mask_surface") else None
            except ImportError:
                mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


def mask_surface(
    mask: np.ndarray, lo, hi, offset: float = 0.003, neighbours: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Exposed faces of the bool mask inside the inclusive box [lo, hi]:
    (tris (Q * 6, 3), lines (Q * 8, 3)) float32 in grid space. The native
    module builds them when it is there (a big box's preview
    and a big selection's skin spent 10 to 15 ms a step here); the numpy
    path below builds the same quads in the same order."""
    native = _native()
    if native is not None:
        lo3 = tuple(int(v) for v in np.asarray(lo).reshape(3))
        hi3 = tuple(int(v) for v in np.asarray(hi).reshape(3))
        tris, lines = native.mask_surface(np.ascontiguousarray(mask, dtype=bool), lo3, hi3, float(offset),
                                          bool(neighbours))
        return tris, lines
    return _mask_surface_numpy(mask, lo, hi, offset, neighbours)


def mask_surface_blocks(mask: np.ndarray, boxes, offset: float = 0.003, neighbours: bool = False) -> list:
    """mask_surface for many inclusive boxes [lo, hi] of one mask: a list of
    (tris, lines) in the boxes' order. One native call for them all, in
    parallel, when the module has it."""
    boxes = [(tuple(int(v) for v in np.asarray(lo).reshape(3)), tuple(int(v) for v in np.asarray(hi).reshape(3)))
             for lo, hi in boxes]
    native = _native()
    if native is not None and hasattr(native, "mask_surface_blocks"):
        return native.mask_surface_blocks(np.ascontiguousarray(mask, dtype=bool), boxes, float(offset), bool(neighbours))
    return [mask_surface(mask, lo, hi, offset, neighbours) for lo, hi in boxes]


def _mask_surface_numpy(
    mask: np.ndarray, lo, hi, offset: float = 0.003, neighbours: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Exposed faces of the bool mask inside the inclusive box [lo, hi].

    Returns (tris (Q * 6, 3), lines (Q * 8, 3)) float32 in grid space. Every
    quad is pushed outward by `offset`. Vectorised like mesh._build. With
    `neighbours`, a cell at the box's edge looks at the mask just outside it
    (one block of a skin built in blocks: no faces at the seams); without, the
    outside counts as empty.
    """
    shape = np.asarray(mask.shape, dtype=np.int64)
    lo = np.maximum(np.asarray(lo, dtype=np.int64), 0)
    hi = np.minimum(np.asarray(hi, dtype=np.int64), shape - 1)
    if np.any(hi < lo):
        return _empty_points(), _empty_points()
    if neighbours:
        plo = np.maximum(lo - 1, 0)
        phi = np.minimum(hi + 1, shape - 1)
        sub = np.zeros(tuple(int(v) for v in hi - lo + 3), dtype=bool)
        at = plo - (lo - 1)
        sub[tuple(slice(int(a), int(a) + int(b - c) + 1) for a, b, c in zip(at, phi, plo))] = mask[
            tuple(slice(int(a), int(b) + 1) for a, b in zip(plo, phi))
        ]
    else:
        window = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
        sub = np.pad(mask[window].astype(bool), 1, constant_values=False)
    core = sub[_CORE]
    origin = lo.astype(np.float32)

    tri_parts: list[np.ndarray] = []
    line_parts: list[np.ndarray] = []
    for row, neighbour in enumerate(_NEIGHBOR_SLICES):
        face = core & ~sub[neighbour]
        xs, ys, zs = np.nonzero(face)
        if xs.size == 0:
            continue
        cells = np.stack((xs, ys, zs), axis=1).astype(np.float32) + origin
        push = _ROW_NORMALS[row] * np.float32(offset)
        quads = cells[:, None, :] + _FACE_CORNERS[row][None, :, :] + push  # (Q, 4, 3)
        tri_parts.append(quads[:, QUAD_TRIS, :].reshape(-1, 3))
        line_parts.append(quads[:, _QUAD_LINES, :].reshape(-1, 3))

    if not tri_parts:
        return _empty_points(), _empty_points()
    tris = np.concatenate(tri_parts, axis=0).astype(np.float32)
    lines = np.concatenate(line_parts, axis=0).astype(np.float32)
    return tris, lines


def cells_faces(cells, offset: float = 0.003) -> np.ndarray:
    """Exposed faces of a cell set, (Q * 6, 3) float32 TRIS."""
    arr = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if arr.shape[0] == 0:
        return _empty_points()
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    local = arr - lo
    mask = np.zeros(tuple(int(v) for v in hi - lo + 1), dtype=bool)
    mask[local[:, 0], local[:, 1], local[:, 2]] = True
    tris, _ = mask_surface(mask, (0, 0, 0), hi - lo, offset=offset)
    return (tris + lo.astype(np.float32)).astype(np.float32)


def box_faces(lo, hi, offset: float = 0.003) -> np.ndarray:
    """The six faces of the box from lo to hi + 1 as (36, 3) float32 TRIS, each
    pushed out by `offset`: the exposed faces of a filled box, as cells_faces
    gives them cell by cell, in six quads instead of one per cell."""
    start = np.asarray(lo, dtype=np.float32)
    extent = np.asarray(hi, dtype=np.float32) + 1.0 - start
    out = []
    for row in range(6):
        corners = start + _FACE_CORNERS[row] * extent + _ROW_NORMALS[row] * np.float32(offset)
        out.append(corners[QUAD_TRIS])
    return np.concatenate(out, axis=0).astype(np.float32)


def box_face(lo, hi, normal, offset: float = 0.003) -> np.ndarray:
    """The face of the box from lo to hi + 1 on the normal side, (6, 3) float32
    TRIS: what cells_face_quads gives for a full one-cell-thick rectangle of
    cells, in one quad instead of one per cell."""
    start = np.asarray(lo, dtype=np.float32)
    extent = np.asarray(hi, dtype=np.float32) + 1.0 - start
    row = _NORMAL_ROWS[tuple(int(v) for v in normal)]
    corners = start + _FACE_CORNERS[row] * extent + _ROW_NORMALS[row] * np.float32(offset)
    return corners[QUAD_TRIS].astype(np.float32)


def slab_edges(lo, hi) -> np.ndarray:
    """The 12 edges of the box from lo to hi + 1, (24, 3) float32 LINES."""
    start = np.asarray(lo, dtype=np.float32)
    extent = np.asarray(hi, dtype=np.float32) + 1.0 - start
    corners = start + _CUBE_CORNERS * extent
    return corners[_CUBE_EDGES.reshape(-1)].astype(np.float32)


def mirror_plane_quad(size, axis: int) -> np.ndarray:
    """Quad (4, 3) float32 at size[axis] / 2, spanning the other two extents."""
    extent = np.asarray(size, dtype=np.float32)
    a, b = [i for i in range(3) if i != axis]
    quad = np.zeros((4, 3), dtype=np.float32)
    quad[:, axis] = extent[axis] / 2.0
    quad[:, a] = [0.0, extent[a], extent[a], 0.0]
    quad[:, b] = [0.0, 0.0, extent[b], extent[b]]
    return quad


def _empty_points() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


# --- new UI: the tool HUD (draw-only, bottom-left of the 3D view) --------------


def _rgba(hex_rgb: str) -> tuple[float, float, float, float]:
    h = hex_rgb.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, 1.0)


HUD_TITLE_PX = 16  # the tool, one line at about the status bar's weight
HUD_TIP_PX = 12  # the mode and the hint under it
HUD_LEFT_PX = 16  # extra inset past the left rail
HUD_RAIL_PX = 56  # matches rok.core.chrome.RAIL_WIDTH; HUD is new-UI only
HUD_ABOVE_HINT_PX = 12
HUD_TITLE_TIP_GAP_PX = 4
HUD_ICON_PX = 28  # the build mode's icon, left of the readout
HUD_ICON_GAP_PX = 8  # between the icon and the text
HUD_TITLE_RGBA = _rgba("F2EDE6")
HUD_ERASE_RGBA = _rgba("FF8A9A")  # Option-held erase
HUD_TIP_RGBA = _rgba("B0A99F")
HUD_HALO_RGBA = (0.055, 0.050, 0.048, 1.0)  # 1px charcoal
OPTION_ERASE_FADE_AFTER = 3  # idle tip hides after this many Option-erases

HUD_TITLES = {
    ATTACH: "Attach",
    ERASE: "Erase",
    PAINT: "Paint",
    SELECT: "Select",
}
HUD_TIPS = {
    ATTACH: "⌥ click or drag to erase",
    ERASE: "Click or drag to erase",
    PAINT: "Click or drag to paint",
    SELECT: "Click or drag to select · ⌥ or right-click to deselect",
}
HUD_DESELECT_TIP = "Click or drag to take voxels out of the selection"
HUD_ERASING_TIP = "Erasing…"
HUD_MODE_NAMES = {ident: name for ident, name, _desc in MODE_ITEMS}


@dataclass(frozen=True)
class HudReadout:
    """What the bottom-left tool HUD draws. No hit rect: it steals no clicks."""

    title: str
    title_rgba: tuple[float, float, float, float]
    tip: str | None
    tip_rgba: tuple[float, float, float, float]
    erase_held: bool


def hud_erase_held(base: str, alt: bool, shift: bool) -> bool:
    """True when Option is held and the press would erase, not select-subtract."""
    return bool(alt) and effective_tool(base, alt, shift) == ERASE


def hud_readout(
    base: str, alt: bool, shift: bool, *, tip_faded: bool = False, mode: str | None = None
) -> HudReadout:
    """Title and tip for the current tool and modifiers.

    The strip ember stays on `base`. Option-held erase flips only this readout:
    title `Erase` in coral, tip `Erasing…`. After the first three Option-erases
    of a session the idle tip is omitted until the header tool changes. With a
    `mode`, the tip line starts with the mode's name ("Box · ⌥ click or drag
    to erase"), and the name alone stays when the tip has faded.
    """
    tool = effective_tool(base, alt, shift)
    erase_held = hud_erase_held(base, alt, shift)
    title = HUD_TITLES.get(tool, HUD_TITLES[ATTACH])
    if tool == SELECT and alt:
        # Option in Select: deselect, in the erase colour like the hover.
        return HudReadout(
            title="Deselect",
            title_rgba=HUD_ERASE_RGBA,
            tip=HUD_DESELECT_TIP,
            tip_rgba=HUD_ERASE_RGBA,
            erase_held=False,
        )
    if erase_held:
        return HudReadout(
            title="Erase",
            title_rgba=HUD_ERASE_RGBA,
            tip=HUD_ERASING_TIP,
            tip_rgba=HUD_ERASE_RGBA,
            erase_held=True,
        )
    tip = None if tip_faded else HUD_TIPS.get(tool, HUD_TIPS[ATTACH])
    if mode is not None:
        name = HUD_MODE_NAMES.get(mode, mode.title())
        tip = name if tip is None else f"{name} · {tip}"
    return HudReadout(
        title=title,
        title_rgba=HUD_TITLE_RGBA,
        tip=tip,
        tip_rgba=HUD_TIP_RGBA,
        erase_held=False,
    )


SCENE_TIP = "Drag to move · Click again: the one behind · Double-click to step in"


def linked_note(copies: int) -> str:
    """The words for a linked copy: `copies` models share its shape."""
    return f"Linked: edits change all {copies}" if copies > 1 else ""


def scene_readout(model_name: str, copies: int = 0) -> HudReadout:
    """The readout in the scene: which model is picked, and how to step in.
    `copies` above 1: it is a linked copy, one of that many."""
    note = linked_note(copies)
    return HudReadout(
        title=f"{model_name} picked" if model_name else "Scene",
        title_rgba=HUD_TITLE_RGBA,
        tip=f"{note} · {SCENE_TIP}" if note else f"Scene · {SCENE_TIP}",
        tip_rgba=HUD_TIP_RGBA,
        erase_held=False,
    )


def inside_readout(readout: HudReadout, model_name: str, copies: int = 0) -> HudReadout:
    """`readout` with the model you are inside named first in its tip, and
    whether it is a linked copy (`copies` above 1): then an edit shows in all."""
    if not model_name:
        return readout
    note = linked_note(copies)
    where = f"Inside {model_name}" + (f" · {note}" if note else "")
    tip = where if readout.tip is None else f"{where} · {readout.tip}"
    return replace(readout, tip=tip)


def hud_anchor(scale: float, hint_top: float, rail_width: float | None = None) -> tuple[int, int]:
    """Pixel-snapped bottom-left of the HUD: rail + 16px from left, 12px above the hint strip."""
    rail = HUD_RAIL_PX if rail_width is None else float(rail_width)
    return (
        round((rail + HUD_LEFT_PX) * scale),
        round(float(hint_top) + HUD_ABOVE_HINT_PX * scale),
    )


def hud_title_origin(scale: float, hint_top: float) -> tuple[int, int]:
    """Pixel-snapped baseline of the title. Stays put when the idle tip is hidden."""
    x, y0 = hud_anchor(scale, hint_top)
    tip_h = round(HUD_TIP_PX * scale)
    gap = round(HUD_TITLE_TIP_GAP_PX * scale)
    return (x, y0 + tip_h + gap)


def hud_icon_rect(scale: float, hint_top: float) -> tuple[int, int, int, int]:
    """The build mode's icon at the anchor, spanning the two text lines."""
    x, y0 = hud_anchor(scale, hint_top)
    side = round(HUD_ICON_PX * scale)
    block = round(HUD_TIP_PX * scale) + round(HUD_TITLE_TIP_GAP_PX * scale) + round(HUD_TITLE_PX * scale)
    return (x, y0 + max(0, (block - side) // 2), side, side)


def hud_text_x(scale: float, hint_top: float) -> int:
    """Where the title and the tip start: right of the icon."""
    x, _y, side, _h = hud_icon_rect(scale, hint_top)
    return x + side + round(HUD_ICON_GAP_PX * scale)


def hud_after_option_erase(count: int, faded: bool) -> tuple[int, bool]:
    """Count one Option-erase. After OPTION_ERASE_FADE_AFTER the idle tip fades."""
    count = int(count) + 1
    if count >= OPTION_ERASE_FADE_AFTER:
        faded = True
    return count, faded


def hud_on_tool_change(previous: str | None, current: str, faded: bool) -> tuple[str, bool]:
    """The idle tip returns when the header tool changes. Returns (current, faded)."""
    if previous is not None and previous != current:
        return current, False
    return current, faded
