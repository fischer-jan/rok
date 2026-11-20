"""The cage's chip row: Points, the axis locks, Soft, Reset and Done.

A row of chips above the cage's projected box, laid out as the selection's pill
strip is (rok/core/cage.py `strip_layout`): "Points" (opens a small dialog for
the number of points on each axis), "Lock X", "Lock Y", "Lock Z" (lit while on:
a point cannot move along a locked axis), "Soft" (lit when the full-degree
Bernstein basis is on: every point pulls the whole volume, which differs from
the default only at 5 and 6 points), "Reset" (undoes the whole cage) and
"Done". The row sits above the box,
or below it when the top would clip; with `avoid`, the selection's own pill
strip, it stacks beyond that strip on the same side so the two never overlap.
Pure numpy: the overlay draws what `chip_layout` returns, and `tools.py` asks
`chip_hit` where it asks the pill strip.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

POINTS = "CAGE_POINTS"
LOCK_PREFIX = "CAGE_LOCK_"
SOFT = "CAGE_SOFT"
RESET = "CAGE_RESET"
DONE = "CAGE_DONE"

# Points, before the UI scale.
CHIP_HEIGHT = 30.0
POINTS_WIDTH = 62.0
LOCK_WIDTH = 66.0
SOFT_WIDTH = 54.0
WORD_WIDTH = (RESET, 62.0), (DONE, 56.0)
CHIP_GAP = 8.0

AXIS_NAMES = "XYZ"

CHIP_TITLES = {POINTS: "Points", RESET: "Reset", DONE: "Done", SOFT: "Soft"} | {
    f"{LOCK_PREFIX}{a}": f"Lock {AXIS_NAMES[a]}" for a in range(3)}
CHIP_TIPS = {
    POINTS: "Points: type how many points the cage has along X, Y and Z, 2 to 6 each",
    SOFT: "Soft: every point pulls the whole cage, not just its own part (full-degree Bernstein). Only differs at 5 and 6 points",
    RESET: "Undo the whole cage: the voxels go back to how they were before the first drag, and the points to their default",
    DONE: "Hide the cage; the voxels stay as they are (Return)",
} | {f"{LOCK_PREFIX}{a}": f"Lock {AXIS_NAMES[a]}: a dragged point cannot move along {AXIS_NAMES[a]}. Light two to move along the third only"
     for a in range(3)}


def lock_id(axis: int) -> str:
    return f"{LOCK_PREFIX}{int(axis)}"


def parse_lock(name) -> int | None:
    """1 for "CAGE_LOCK_1", else None."""
    if isinstance(name, str) and name.startswith(LOCK_PREFIX) and name[len(LOCK_PREFIX):] in ("0", "1", "2"):
        return int(name[len(LOCK_PREFIX):])
    return None


def is_chip(name) -> bool:
    return name in (POINTS, RESET, DONE, SOFT) or parse_lock(name) is not None


@dataclass(frozen=True)
class ChipLayout:
    """The chips in region pixels, from the last drawn frame."""

    chips: tuple[tuple[str, tuple[int, int, int, int]], ...]  # (id, (x, y, w, h)), left to right
    rect: tuple[int, int, int, int]  # the whole row, including gaps
    above: bool  # True when the row sits above the box


def chip_layout(
    aabb: tuple[float, float, float, float] | None,
    region_width: float,
    region_height: float,
    scale: float = 1.0,
    avoid: tuple[float, float, float, float] | None = None,
) -> ChipLayout | None:
    """The row, centred over `aabb` (minx, miny, maxx, maxy; y grows upward) and
    kept inside the region. With `avoid` (x, y, w, h) it stands beyond that rect
    on the same side the box gives it."""
    if aabb is None:
        return None
    minx, miny, maxx, maxy = (float(v) for v in aabb)
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        return None
    s = float(scale)
    height = int(round(CHIP_HEIGHT * s))
    gap = int(round(CHIP_GAP * s))
    points_w = int(round(POINTS_WIDTH * s))
    words = [(name, int(round(w * s))) for name, w in WORD_WIDTH]
    lock_w = int(round(LOCK_WIDTH * s))
    soft_w = int(round(SOFT_WIDTH * s))
    width = points_w + 3 * lock_w + soft_w + sum(w for _n, w in words) + (len(words) + 4) * gap
    if height <= 0 or points_w <= 0:
        return None
    margin = gap
    rw, rh = float(region_width), float(region_height)

    x0 = int(round(0.5 * (minx + maxx) - width / 2.0))
    max_x0 = int(rw) - margin - width
    if max_x0 < margin:
        x0 = max(0, int(round((rw - width) / 2.0)))
    else:
        x0 = max(margin, min(x0, max_x0))

    top = maxy
    bottom = miny
    if avoid is not None:
        ax, ay, aw, ah = (float(v) for v in avoid)
        top = max(top, ay + ah)
        bottom = min(bottom, ay)
    above_y = int(round(top + gap))
    below_y = int(round(bottom - gap - height))
    if above_y + height + margin <= rh:
        y0, above = above_y, True
    else:
        y0, above = below_y, False
        if y0 < margin:
            if rh - top >= bottom:
                y0, above = above_y, True
            y0 = max(0, min(y0, int(rh) - height))

    chips = []
    x = x0
    chips.append((POINTS, (x, y0, points_w, height)))
    x += points_w
    for a in range(3):
        x += gap
        chips.append((lock_id(a), (x, y0, lock_w, height)))
        x += lock_w
    x += gap
    chips.append((SOFT, (x, y0, soft_w, height)))
    x += soft_w
    for name, w in words:
        x += gap
        chips.append((name, (x, y0, w, height)))
        x += w
    return ChipLayout(tuple(chips), (x0, y0, width, height), above)


def chip_hit(layout: ChipLayout | None, mx: float, my: float) -> str | None:
    """The chip under a region pixel, or None (gaps between the words miss)."""
    if layout is None:
        return None
    mx, my = float(mx), float(my)
    for name, (x, y, w, h) in layout.chips:
        if x <= mx < x + w and y <= my < y + h:
            return name
    return None


def chip_contains(layout: ChipLayout | None, mx: float, my: float) -> bool:
    """True when the pixel is in the row's bounding box, including gaps."""
    if layout is None:
        return False
    x, y, w, h = layout.rect
    return x <= float(mx) < x + w and y <= float(my) < y + h
