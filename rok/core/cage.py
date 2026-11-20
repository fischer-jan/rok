"""The selection cage: the AABB outline and the selection bar.

A thin ember box around the selection, and a bar above its projection.
The bar is four groups: paint and mask, grow shrink and invert, bevel and
smooth, and split. Bevel and Smooth run at once on the selection. The box
has no handles (the selection moves by dragging a selected block). Pure
numpy: the overlay draws what `geometry` and `strip_layout` return. Imports
.gizmo only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .geom2d import fit_glyph, loop_lines
from .gizmo import project

# ---------------------------------------------------------------- names

FACES = tuple(f"FACE_{axis}{side}" for axis in "XYZ" for side in "01")
CORNERS = tuple(f"CORNER_{x}{y}{z}" for x in "01" for y in "01" for z in "01")
# The bar's buttons in groups, left to right.
GROUPS = (
    ("PAINT", "MASK"),
    ("SEL_GROW", "SEL_SHRINK", "INVERT"),
    ("BEVEL", "SMOOTH"),
    ("SPLIT",),
)
GLYPHS = tuple(name for group in GROUPS for name in group)

AXIS_NAMES = "XYZ"

# The bar: 36 px buttons, 8 px off the projected AABB.
PILL_SIZE = 36.0
PILL_GAP = 2.0
GROUP_GAP = 10.0
BAR_PAD = 5.0
ABOVE_GAP = 8.0
PILL_HOVER_S = 0.2  # HUD tip after the mouse has rested this long


def _edges() -> tuple[str, ...]:
    names = []
    for along in range(3):
        b, c = (i for i in range(3) if i != along)
        for bs in (0, 1):
            for cs in (0, 1):
                names.append(
                    f"EDGE_{AXIS_NAMES[along]}_{AXIS_NAMES[b]}{bs}_{AXIS_NAMES[c]}{cs}"
                )
    return tuple(names)


EDGES = _edges()

# HUD title, HUD tip line, status-bar hover line.
PILL_TITLES = {
    "PAINT": "Paint",
    "MASK": "Mask",
    "SEL_GROW": "Grow",
    "SEL_SHRINK": "Shrink",
    "INVERT": "Invert",
    "BEVEL": "Bevel",
    "SMOOTH": "Smooth",
    "SPLIT": "Split off",
}
PILL_TIPS = {
    "PAINT": "Give the selection the current colour (Shift+P)",
    "MASK": "Protect the selection: Attach, Erase and Paint leave it alone (Shift+M)",
    "SEL_GROW": "Select one more layer of voxels around the selection (Ctrl++)",
    "SEL_SHRINK": "Select one layer less around the edge of the selection (Ctrl+-)",
    "INVERT": "Select what is not selected, and drop what is (Ctrl+I)",
    "BEVEL": "Take one voxel off every convex edge of the selection. Again for a rounder edge",
    "SMOOTH": "Corners and bumps go, dents fill. Again for a smoother surface",
    "SPLIT": "Move the selection into a new model",
}
PILL_STATUS = {
    "PAINT": "Paint the selection in the current colour",
    "MASK": "Mask the selection",
    "SEL_GROW": "Grow the selection by a layer",
    "SEL_SHRINK": "Shrink the selection by a layer",
    "INVERT": "Invert the selection",
    "BEVEL": "Bevel the selection",
    "SMOOTH": "Smooth the selection",
    "SPLIT": "Split off",
}
CAGE_IDLE_HINT = "Drag a selected block to move it · the bar edits the selection"


def is_glyph(handle: str) -> bool:
    """True for a pill id (a button of the bar)."""
    return handle in GLYPHS


def pill_hint(name: str) -> str:
    """Status-bar text for the pill under the mouse."""
    return PILL_STATUS.get(name, "")


def pill_hud(name: str) -> tuple[str, str]:
    """(title, tip) for the HUD while the mouse rests on this pill."""
    return PILL_TITLES.get(name, ""), PILL_TIPS.get(name, "")


def pill_tip_due(
    started: float | None, now: float | None = None, delay: float = PILL_HOVER_S
) -> bool:
    """True once the mouse has rested on a pill for `delay` seconds."""
    if started is None:
        return False
    if now is None:
        now = time.monotonic()
    return (now - started) >= delay


def face_axis_side(handle: str) -> tuple[int, int]:
    """FACE_X0 -> (0, 0): axis index and lo/hi side."""
    return "XYZ".index(handle[5]), int(handle[6])


def corner_sides(handle: str) -> tuple[int, int, int]:
    """CORNER_101 -> (1, 0, 1)."""
    return int(handle[7]), int(handle[8]), int(handle[9])


def parse_edge(handle: str) -> tuple[int, int, int, int, int]:
    """EDGE_X_Y0_Z1 -> (along=0, b=1, bs=0, c=2, cs=1)."""
    along = "XYZ".index(handle[5])
    b = "XYZ".index(handle[7])
    bs = int(handle[8])
    c = "XYZ".index(handle[10])
    cs = int(handle[11])
    return along, b, bs, c, cs


def aabb(bounds) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive cell (lo, hi) to the float box from lo to hi+1."""
    lo_c, hi_c = bounds
    lo = np.asarray(lo_c, dtype=np.float64).reshape(3)
    hi = np.asarray(hi_c, dtype=np.float64).reshape(3) + 1.0
    return lo, hi


def _corner_pos(lo: np.ndarray, hi: np.ndarray, sx: int, sy: int, sz: int) -> np.ndarray:
    return np.array(
        [
            lo[0] if sx == 0 else hi[0],
            lo[1] if sy == 0 else hi[1],
            lo[2] if sz == 0 else hi[2],
        ],
        dtype=np.float64,
    )


def _face_quad(lo: np.ndarray, hi: np.ndarray, axis: int, side: int) -> np.ndarray:
    b, c = (i for i in range(3) if i != axis)
    pts = []
    for sb, sc in ((0, 0), (1, 0), (1, 1), (0, 1)):
        bits = [0, 0, 0]
        bits[axis] = side
        bits[b] = sb
        bits[c] = sc
        pts.append(_corner_pos(lo, hi, *bits))
    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------------- geometry


@dataclass(frozen=True)
class CageGeometry:
    """The AABB in the model's grid space, ready to draw and to project."""

    lo: np.ndarray  # (3,) float64, min corner
    hi: np.ndarray  # (3,) float64, max corner (hi cell + 1)
    lines: np.ndarray  # (24, 3) float32, 12 edges as LINES pairs
    faces: dict[str, np.ndarray]  # (4, 3) float32
    edges: dict[str, np.ndarray]  # (2, 3) float32
    corners: dict[str, np.ndarray]  # (3,) float32


def geometry(bounds) -> CageGeometry:
    """Build the cage for inclusive cell bounds (lo, hi)."""
    lo, hi = aabb(bounds)
    faces = {name: _face_quad(lo, hi, *face_axis_side(name)) for name in FACES}
    corners = {
        name: _corner_pos(lo, hi, *corner_sides(name)).astype(np.float32) for name in CORNERS
    }
    edges: dict[str, np.ndarray] = {}
    line_pairs = []
    for name in EDGES:
        along, b, bs, c, cs = parse_edge(name)
        bits_a = [0, 0, 0]
        bits_a[b] = bs
        bits_a[c] = cs
        bits_b = list(bits_a)
        bits_a[along] = 0
        bits_b[along] = 1
        a = _corner_pos(lo, hi, *bits_a)
        d = _corner_pos(lo, hi, *bits_b)
        pair = np.array([a, d], dtype=np.float32)
        edges[name] = pair
        line_pairs.append(pair)
    lines = np.concatenate(line_pairs, axis=0)
    return CageGeometry(lo, hi, lines, faces, edges, corners)


# ---------------------------------------------------------------- screen / hit


@dataclass(frozen=True)
class StripLayout:
    """The selection bar in region pixels, from the last drawn frame."""

    pills: tuple[tuple[str, tuple[int, int, int, int]], ...]  # (name, x, y, w, h)
    rect: tuple[int, int, int, int]  # the bar's plate: the groups and a little around them
    above: bool  # True when the bar sits above the projected AABB
    wells: tuple[tuple[int, int, int, int], ...] = ()  # one rounded group behind its buttons


@dataclass(frozen=True)
class CageScreen:
    """Projected handles of the last drawn frame, in region pixels."""

    faces: dict[str, np.ndarray]  # (4, 2)
    edges: dict[str, np.ndarray]  # (2, 2)
    corners: dict[str, np.ndarray]  # (2,)
    face_centers: dict[str, np.ndarray]  # (2,)
    strip: StripLayout | None = None


def screen_aabb(screen: CageScreen | None) -> tuple[float, float, float, float] | None:
    """Axis-aligned bounds of the finite projected corners, or None."""
    if screen is None:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for p in screen.corners.values():
        if np.all(np.isfinite(p)):
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def strip_layout(
    aabb: tuple[float, float, float, float] | None,
    region_width: float,
    region_height: float,
    scale: float = 1.0,
) -> StripLayout | None:
    """The bar above `aabb` (below when that would clip the top), centred on it and kept inside the view."""
    if aabb is None:
        return None
    minx, miny, maxx, maxy = (float(v) for v in aabb)
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        return None
    size = int(round(PILL_SIZE * float(scale)))
    gap = int(round(PILL_GAP * float(scale)))
    group_gap = int(round(GROUP_GAP * float(scale)))
    pad = int(round(BAR_PAD * float(scale)))
    away = int(round(ABOVE_GAP * float(scale)))
    margin = away
    count = len(GLYPHS)
    width = 2 * pad + count * size + (count - len(GROUPS)) * gap + (len(GROUPS) - 1) * group_gap
    height = 2 * pad + size
    rw = float(region_width)
    rh = float(region_height)
    if size <= 0:
        return None

    cx = 0.5 * (minx + maxx)
    x0 = int(round(cx - width / 2.0))
    max_x0 = int(rw) - margin - width
    if max_x0 < margin:
        x0 = max(0, int(round((rw - width) / 2.0)))
    else:
        x0 = max(margin, min(x0, max_x0))

    above_y = int(round(maxy + away))
    below_y = int(round(miny - away - height))
    if above_y + height + margin <= rh:
        y0, above = above_y, True
    else:
        y0, above = below_y, False
        if y0 < margin:
            room_above = rh - maxy
            room_below = miny
            if room_above >= room_below:
                y0, above = above_y, True
            y0 = max(0, min(y0, int(rh) - height))

    button_y = y0 + pad
    pills = []
    wells = []
    x = x0 + pad
    for g, group in enumerate(GROUPS):
        n = len(group)
        well_w = n * size + (n - 1) * gap
        wells.append((x, button_y, well_w, size))
        for k, name in enumerate(group):
            pills.append((name, (x, button_y, size, size)))
            x += size + (gap if k < n - 1 else 0)
        if g < len(GROUPS) - 1:
            x += group_gap
    return StripLayout(tuple(pills), (x0, y0, width, height), above, tuple(wells))


def strip_hit(layout: StripLayout | None, mx: float, my: float) -> str | None:
    """The pill under a region pixel, or None (gaps between pills miss)."""
    if layout is None:
        return None
    mx, my = float(mx), float(my)
    for name, (x, y, w, h) in layout.pills:
        if x <= mx < x + w and y <= my < y + h:
            return name
    return None


def strip_contains(layout: StripLayout | None, mx: float, my: float) -> bool:
    """True when the pixel is on the bar's plate, buttons and gaps."""
    if layout is None:
        return False
    x, y, w, h = layout.rect
    return x <= float(mx) < x + w and y <= float(my) < y + h


def screen_geometry(
    geom: CageGeometry, matrix, width: int, height: int, scale: float = 1.0
) -> CageScreen:
    """Project the cage and lay the selection bar out above it."""
    faces = {k: project(matrix, v, width, height) for k, v in geom.faces.items()}
    edges = {k: project(matrix, v, width, height) for k, v in geom.edges.items()}
    corners = {k: project(matrix, v.reshape(1, 3), width, height)[0] for k, v in geom.corners.items()}
    face_centers = {}
    for name, pts in faces.items():
        finite = pts[np.isfinite(pts).all(axis=1)]
        face_centers[name] = (
            finite.mean(axis=0) if len(finite) else np.array([np.nan, np.nan], dtype=np.float64)
        )
    screen = CageScreen(faces, edges, corners, face_centers)
    strip = strip_layout(screen_aabb(screen), width, height, scale)
    return CageScreen(faces, edges, corners, face_centers, strip)


# ---------------------------------------------------------------- glyphs (unit square line lists)


def _pairs2(*points: tuple[float, float]) -> np.ndarray:
    return np.array(points, dtype=np.float32).reshape(-1, 2)


_GLYPH_LINES: dict[str, np.ndarray] = {
    "HOLLOW": np.concatenate(
        [
            loop_lines(((0.12, 0.12), (0.88, 0.12), (0.88, 0.88), (0.12, 0.88))),
            loop_lines(((0.34, 0.34), (0.66, 0.34), (0.66, 0.66), (0.34, 0.66))),
        ]
    ),
    "SPLIT": np.concatenate(
        [
            loop_lines(((0.10, 0.18), (0.42, 0.18), (0.42, 0.82), (0.10, 0.82))),
            loop_lines(((0.58, 0.18), (0.90, 0.18), (0.90, 0.82), (0.58, 0.82))),
        ]
    ),
    "BRIDGE": np.concatenate(
        [
            loop_lines(((0.08, 0.35), (0.32, 0.35), (0.32, 0.65), (0.08, 0.65))),
            loop_lines(((0.68, 0.35), (0.92, 0.35), (0.92, 0.65), (0.68, 0.65))),
            _pairs2((0.32, 0.5), (0.68, 0.5)),
        ]
    ),
    # A square with its upper right corner cut off.
    "BEVEL": loop_lines(((0.16, 0.16), (0.84, 0.16), (0.84, 0.50), (0.50, 0.84), (0.16, 0.84))),
    # A square with its lower right half hatched: what was out is in.
    "INVERT": np.concatenate(
        [
            loop_lines(((0.12, 0.12), (0.88, 0.12), (0.88, 0.88), (0.12, 0.88))),
            _pairs2((0.88, 0.40), (0.40, 0.12), (0.88, 0.64), (0.64, 0.12)),
        ]
    ),
    # A domino mask: the band and two eye holes.
    "MASK": np.concatenate(
        [
            loop_lines(((0.10, 0.40), (0.90, 0.40), (0.90, 0.66), (0.10, 0.66))),
            loop_lines(((0.24, 0.48), (0.42, 0.48), (0.42, 0.58), (0.24, 0.58))),
            loop_lines(((0.58, 0.48), (0.76, 0.48), (0.76, 0.58), (0.58, 0.58))),
        ]
    ),
    # A brush: the handle and the bristles, slanted.
    "PAINT": np.concatenate(
        [
            _pairs2((0.62, 0.62), (0.88, 0.88)),
            loop_lines(((0.30, 0.46), (0.46, 0.30), (0.62, 0.46), (0.46, 0.62))),
            _pairs2((0.30, 0.46), (0.14, 0.14), (0.14, 0.14), (0.46, 0.30)),
        ]
    ),
}


def glyph_lines(name: str, rect) -> np.ndarray:
    """The glyph as LINES pairs in region pixels, inside `rect`.

    An unknown name draws nothing. Coordinates land on half pixels.
    """
    unit = _GLYPH_LINES.get(name)
    if unit is None:
        return np.zeros((0, 2), dtype=np.float32)
    return fit_glyph(unit, rect)


def glyph_rect(cx: float, cy: float, size: float) -> tuple[float, float, float, float]:
    """(x, y, w, h) centred on a glyph."""
    h = float(size) / 2.0
    return (float(cx) - h, float(cy) - h, float(size), float(size))


# ---------------------------------------------------------------- the drag
