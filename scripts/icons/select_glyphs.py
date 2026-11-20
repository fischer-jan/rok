"""The Connected select mode glyphs, drawn here rather than generated.

    uv run python scripts/icons/make_icons.py CONNECTED CONNECTED_CORNERS

Isometric blocks in the style of the generated set: outlines at alpha 1, top
faces at FACE. A nearer block hides what is behind it.
"""

from __future__ import annotations

import math

import numpy as np

from anim_glyphs import FACE, SIDE, Canvas

C30 = math.cos(math.radians(30))


def _block(cv: Canvas, origin, s: float, cell, faces: str = "top") -> None:
    """The unit block at `cell` (x, y, z), seen from the usual iso angle:
    whatever it covers is cleared, then its outline and its top face drawn."""
    ox, oy = origin

    def p(x, y, z):
        x, y, z = x + cell[0], y + cell[1], z + cell[2]
        return (ox + (x - y) * C30 * s, oy + (x + y) * 0.5 * s + z * s)

    hexagon = [p(1, 0, 0), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1), p(0, 1, 0), p(0, 0, 0)]
    # Clear what the block hides (its silhouette, a little wider for the line).
    cx = sum(q[0] for q in hexagon) / 6.0
    cy = sum(q[1] for q in hexagon) / 6.0
    grown = [(cx + (q[0] - cx) * 1.08, cy + (q[1] - cy) * 1.08) for q in hexagon]
    inside = np.ones_like(cv.a, dtype=bool)
    for a, b in zip(grown, grown[1:] + grown[:1]):
        inside &= (b[0] - a[0]) * (cv.y - a[1]) - (b[1] - a[1]) * (cv.x - a[0]) >= 0
    cv.a[inside] = 0.0
    if faces in ("top", "all"):
        cv.fill([p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1)], alpha=FACE)
    if faces == "all":  # a block of the picked colour: every face filled
        cv.fill([p(0, 0, 0), p(1, 0, 0), p(1, 0, 1), p(0, 0, 1)], alpha=FACE)
        cv.fill([p(0, 0, 0), p(0, 0, 1), p(0, 1, 1), p(0, 1, 0)], alpha=FACE)
    cv.path(hexagon, closed=True)
    near = p(0, 0, 1)
    for far in (p(1, 0, 1), p(0, 1, 1), p(0, 0, 0)):
        cv.line(near, far)


def connected(side: int = SIDE) -> np.ndarray:
    """Connected: three blocks joined face to face, an L."""
    cv = Canvas(side)
    s = 0.24
    origin = (0.44, 0.14)
    for cell in ((1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1)):  # far to near
        _block(cv, origin, s, cell)
    return cv.a


def connected_corners(side: int = SIDE) -> np.ndarray:
    """Connected by corners: two blocks that meet at one corner, a dot there."""
    cv = Canvas(side)
    s = 0.26
    origin = (0.22, 0.12)
    _block(cv, origin, s, (0, 0, 0))
    _block(cv, origin, s, (1, -1, 1))
    x, y, z = 1, 0, 1  # the corner both blocks share
    corner = (origin[0] + (x - y) * C30 * s, origin[1] + (x + y) * 0.5 * s + z * s)
    cv.disc(corner, 0.055)
    return cv.a


def by_colour(side: int = SIDE) -> np.ndarray:
    """By colour: three loose blocks, the two of one colour filled."""
    cv = Canvas(side)
    s = 0.2
    origin = (0.5, 0.1)
    _block(cv, origin, s, (2, 0, 0), faces="none")
    _block(cv, origin, s, (0, 2, 0), faces="all")
    _block(cv, origin, s, (0, 0, 1), faces="all")
    return cv.a


def colour_connected(side: int = SIDE) -> np.ndarray:
    """By colour, connected by faces: two filled blocks face to face, a loose
    empty one beside them."""
    cv = Canvas(side)
    s = 0.2
    origin = (0.5, 0.1)
    _block(cv, origin, s, (2, 0, 0), faces="none")
    _block(cv, origin, s, (0, 1, 0), faces="all")
    _block(cv, origin, s, (0, 0, 0), faces="all")
    return cv.a


def colour_connected_corners(side: int = SIDE) -> np.ndarray:
    """By colour, connected by corners: two filled blocks meeting at a corner."""
    cv = Canvas(side)
    s = 0.26
    origin = (0.22, 0.12)
    _block(cv, origin, s, (0, 0, 0), faces="all")
    _block(cv, origin, s, (1, -1, 1), faces="all")
    return cv.a


def drop_floor(side: int = SIDE) -> np.ndarray:
    """Drop to Floor: a block above a floor plate, an arrow down beside it."""
    cv = Canvas(side)
    s = 0.24
    # The floor: an isometric plate, its top filled, a thin front edge.
    top = [(0.06, 0.24), (0.46, 0.03), (0.86, 0.24), (0.46, 0.45)]
    cv.fill(top, alpha=FACE)
    cv.path(top, closed=True)
    cv.path([(0.06, 0.24), (0.06, 0.19), (0.46, -0.02 + 0.0), (0.86, 0.19), (0.86, 0.24)])
    _block(cv, (0.46, 0.5), s, (0, 0, 0))
    cv.line((0.9, 0.92), (0.9, 0.5))
    cv.fill([(0.82, 0.56), (0.98, 0.56), (0.9, 0.42)], alpha=1.0)
    return cv.a


def drop_onto(side: int = SIDE) -> np.ndarray:
    """Drop: a block above a filled block, an arrow down beside them. It lands
    on the first voxel below, not on the floor."""
    cv = Canvas(side)
    s = 0.2
    _block(cv, (0.42, 0.04), s, (0, 0, 0), faces="all")
    _block(cv, (0.42, 0.54), s, (0, 0, 0))
    cv.line((0.86, 0.92), (0.86, 0.5))
    cv.fill([(0.78, 0.56), (0.94, 0.56), (0.86, 0.42)], alpha=1.0)
    return cv.a


def terrain(side: int = SIDE) -> np.ndarray:
    """Terrain: a 2 x 2 patch of block columns rising to a peak at the back."""
    cv = Canvas(side)
    s = 0.18
    origin = (0.5, 0.06)
    heights = {(1, 1): 3, (1, 0): 2, (0, 1): 2, (0, 0): 1}
    for (x, y), h in sorted(heights.items(), key=lambda item: -(item[0][0] + item[0][1])):  # far to near
        for z in range(h):
            _block(cv, origin, s, (x, y, z), faces="top" if z == h - 1 else "none")
    return cv.a


GLYPHS = {
    "TERRAIN": terrain,
    "DROP_FLOOR": drop_floor,
    "DROP_ONTO": drop_onto,
    "CONNECTED": connected,
    "CONNECTED_CORNERS": connected_corners,
    "BY_COLOUR": by_colour,
    "COLOUR_CONNECTED": colour_connected,
    "COLOUR_CONNECTED_CORNERS": colour_connected_corners,
}
