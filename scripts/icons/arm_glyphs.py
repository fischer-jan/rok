"""The glyphs of the four tool arms (Attach, Erase, Paint, Select), drawn here
rather than generated, so that all four show the same cube.

    uv run python scripts/icons/make_icons.py CUBE_PLUS CUBE_MINUS BRUSH MARQUEE

These glyphs are drawn in their final frame: make_icons.py does not fit them
to the square, because a fit would shrink the cube of a glyph with a wide
extra (the brush, the selection frame).
"""

from __future__ import annotations

import math

import numpy as np

from anim_glyphs import FACE, LINE, SIDE, Canvas

EDGE = 0.36  # the cube's edge, as a fraction of the side
CX, BOTTOM = 0.5, 0.14  # where every glyph puts the cube, so the four line up
C30 = math.cos(math.radians(30))


def _cube(cv: Canvas, right_face: bool = True) -> dict:
    """The tool cube, the same in every glyph: its near vertical edge at x = CX,
    its lowest point at BOTTOM. Top face filled, the right face too if
    `right_face`."""
    s, w, cx, bottom = EDGE, C30 * EDGE, CX, BOTTOM
    near_b, near_t = (cx, bottom), (cx, bottom + s)
    left_b, left_t = (cx - w, bottom + s / 2), (cx - w, bottom + 1.5 * s)
    right_b, right_t = (cx + w, bottom + s / 2), (cx + w, bottom + 1.5 * s)
    back = (cx, bottom + 2 * s)
    cv.fill([near_t, right_t, back, left_t])
    if right_face:
        cv.fill([near_b, right_b, right_t, near_t])
    cv.path([near_t, left_t, back, right_t], closed=True)
    cv.path([left_t, left_b, near_b, right_b, right_t])
    cv.line(near_t, near_b)
    return {"near_t": near_t, "back": back, "left_t": left_t, "right_t": right_t}


def _sign(cv: Canvas, plus: bool) -> None:
    """A plus or minus in the upper right corner, clear of the cube."""
    c, r = (0.86, 0.88), 0.09
    cv.line((c[0] - r, c[1]), (c[0] + r, c[1]))
    if plus:
        cv.line((c[0], c[1] - r), (c[0], c[1] + r))


def cube_plus(side: int = SIDE) -> np.ndarray:
    """Attach: the cube with a plus."""
    cv = Canvas(side)
    _cube(cv)
    _sign(cv, plus=True)
    return cv.a


def cube_minus(side: int = SIDE) -> np.ndarray:
    """Erase: the cube with a minus."""
    cv = Canvas(side)
    _cube(cv)
    _sign(cv, plus=False)
    return cv.a


def brush(side: int = SIDE) -> np.ndarray:
    """Paint: the cube, top face filled, a slim brush whose tip touches it."""
    cv = Canvas(side)
    k = _cube(cv, right_face=False)
    top = ((k["near_t"][0] + k["back"][0]) / 2, (k["near_t"][1] + k["back"][1]) / 2)
    tip = (top[0] + 0.03, top[1] + 0.01)
    d = (math.cos(math.radians(55)), math.sin(math.radians(55)))  # along the brush, up and right
    n = (-d[1], d[0])

    def at(t, u):
        return (tip[0] + d[0] * t + n[0] * u, tip[1] + d[1] * t + n[1] * u)

    half = LINE * 0.75
    # The bristles: a teardrop from the tip to the ferrule, cleared first so the cube's lines stop at it.
    bristles = [at(0.0, 0.0), at(0.04, -0.04), at(0.12, -0.05), at(0.12, 0.05), at(0.04, 0.04)]
    cleared = Canvas(side)
    cleared.fill(bristles, alpha=1.0)
    cv.a = np.minimum(cv.a, 1.0 - cleared.a)
    cv.path(bristles, closed=True, half=half)
    # The handle: two lines from the ferrule to a round end.
    cv.line(at(0.12, -0.03), at(0.26, -0.024), half=half)
    cv.line(at(0.12, 0.03), at(0.26, 0.024), half=half)
    cv.arc(at(0.26, 0.0), 0.024, math.atan2(n[1], n[0]) - math.pi, math.atan2(n[1], n[0]), half=half)
    return cv.a


def marquee(side: int = SIDE) -> np.ndarray:
    """Select: the cube inside a dashed square frame."""
    cv = Canvas(side)
    _cube(cv)
    lo, hi = 0.04, 0.96
    dashes, fill = 5, 0.55  # per side; the part of each step that is drawn
    step = (hi - lo) / dashes
    half = LINE * 0.75
    for i in range(dashes):
        a = lo + i * step + step * (1 - fill) / 2
        b = a + step * fill
        for p, q in (((a, lo), (b, lo)), ((a, hi), (b, hi)), ((lo, a), (lo, b)), ((hi, a), (hi, b))):
            cv.line(p, q, half=half)
    return cv.a


GLYPHS = {"CUBE_PLUS": cube_plus, "CUBE_MINUS": cube_minus, "BRUSH": brush, "MARQUEE": marquee}
