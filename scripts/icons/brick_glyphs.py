"""Brick mode's glyphs, drawn here rather than generated, so that every
tool shows the same brick.

    uv run python scripts/icons/make_icons.py BRICK_PLACE BRICK_PULL BRICK_PAINT BRICK_PICK
    uv run python scripts/icons/make_icons.py BRICK_CONVERT BRICK_TO_VOXELS

A brick is a low box, one cell high, with a line on its top between its
cells, seen from the usual isometric angle. Outlines at alpha 1 and top
faces at FACE, as in the generated set.
"""

from __future__ import annotations

import math

import numpy as np

from anim_glyphs import FACE, LINE, SIDE, Canvas
from tool_glyphs import _extrude, _plane_mask

C30 = math.cos(math.radians(30))
SEAM = 0.045  # half the width of the line between two cells, in cells


def _brick(cv: Canvas, w: int, d: int, pitch: float, origin) -> None:
    """A w × d brick, one cell high, its near bottom corner at `origin`
    (canvas units); `pitch` is one cell on the canvas."""
    def box(x, y):
        return (x >= 0) & (x <= w * pitch) & (y >= 0) & (y <= d * pitch)

    _extrude(cv, box, 0.0, pitch, origin)
    half = SEAM * pitch
    for i in range(1, w):
        cv.a[_plane_mask(cv, lambda x, y, c=i * pitch: box(x, y) & (np.abs(x - c) <= half), pitch, origin)] = 1.0
    for j in range(1, d):
        cv.a[_plane_mask(cv, lambda x, y, c=j * pitch: box(x, y) & (np.abs(y - c) <= half), pitch, origin)] = 1.0


# The tools' brick: a 2 × 1, at one place in every tool glyph so they line up.
TOOL_PITCH = 0.25
TOOL_ORIGIN = (0.4, 0.1)


def _tool_brick(cv: Canvas) -> None:
    _brick(cv, 2, 1, TOOL_PITCH, TOOL_ORIGIN)


def _arrow(cv: Canvas, x: float, y0: float, y1: float) -> None:
    """A vertical arrow from y0 to y1, its head at y1."""
    cv.line((x, y0), (x, y1), half=LINE)
    head = 0.06 if y1 > y0 else -0.06
    cv.fill([(x - 0.06, y1 - head), (x + 0.06, y1 - head), (x, y1 + head * 0.4)], alpha=1.0)


def brick_place(side: int = SIDE) -> np.ndarray:
    """Place: the brick, an arrow coming down onto it."""
    cv = Canvas(side)
    _tool_brick(cv)
    _arrow(cv, 0.86, 0.97, 0.7)
    return cv.a


def brick_pull(side: int = SIDE) -> np.ndarray:
    """Pull: the brick, an arrow going up from it."""
    cv = Canvas(side)
    _tool_brick(cv)
    _arrow(cv, 0.86, 0.66, 0.95)
    return cv.a


def brick_paint(side: int = SIDE) -> np.ndarray:
    """Paint: the brick, a drop of paint falling onto it."""
    cv = Canvas(side)
    _tool_brick(cv)
    c, r = (0.86, 0.74), 0.075
    cv.disc(c, r, alpha=FACE)
    cv.fill([(c[0] - r * 0.9, c[1] + r * 0.4), (c[0] + r * 0.9, c[1] + r * 0.4), (c[0], c[1] + r * 2.6)], alpha=FACE)
    cv.arc(c, r, math.radians(-215), math.radians(35))
    cv.line((c[0] - r * 0.82, c[1] + r * 0.57), (c[0], c[1] + r * 2.6))
    cv.line((c[0] + r * 0.82, c[1] + r * 0.57), (c[0], c[1] + r * 2.6))
    return cv.a


def brick_pick(side: int = SIDE) -> np.ndarray:
    """Pick: the brick, a pipette over it that takes its shape and colour."""
    cv = Canvas(side)
    _tool_brick(cv)
    tip, top = (0.74, 0.7), (0.92, 0.92)
    cv.line(tip, top, half=LINE)
    d = (top[0] - tip[0], top[1] - tip[1])
    n = math.hypot(*d)
    u = (d[0] / n, d[1] / n)
    bulb = (top[0] + u[0] * 0.03, top[1] + u[1] * 0.03)
    cv.disc(bulb, 0.055)
    return cv.a


def _small_cube(cv: Canvas, origin, s: float) -> None:
    _extrude(cv, lambda x, y: (x >= 0) & (x <= s) & (y >= 0) & (y <= s), 0.0, s, origin)


def _right_arrow(cv: Canvas, y: float) -> None:
    cv.line((0.4, y), (0.56, y))
    cv.fill([(0.56, y - 0.05), (0.56, y + 0.05), (0.62, y)], alpha=1.0)


def brick_convert(side: int = SIDE) -> np.ndarray:
    """Convert to Bricks: a cube, an arrow, a brick."""
    cv = Canvas(side)
    _small_cube(cv, (0.18, 0.3), 0.24)
    _right_arrow(cv, 0.5)
    _brick(cv, 2, 1, 0.14, (0.745, 0.325))
    return cv.a


def brick_to_voxels(side: int = SIDE) -> np.ndarray:
    """Convert to Voxels: a brick, an arrow, a cube."""
    cv = Canvas(side)
    _brick(cv, 2, 1, 0.14, (0.135, 0.325))
    _right_arrow(cv, 0.5)
    _small_cube(cv, (0.82, 0.3), 0.24)
    return cv.a


GLYPHS = {
    "BRICK_PLACE": brick_place,
    "BRICK_PULL": brick_pull,
    "BRICK_PAINT": brick_paint,
    "BRICK_PICK": brick_pick,
    "BRICK_CONVERT": brick_convert,
    "BRICK_TO_VOXELS": brick_to_voxels,
}
