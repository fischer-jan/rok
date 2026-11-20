"""The Stroke Designer's glyph, drawn here rather than generated.

    uv run python scripts/icons/make_icons.py STROKE

A stroke seen from above: a round, thick start that bends in a wave and thins
to a point, as a designed taper draws it. The body is filled at FACE, the
outline at alpha 1, like the other glyphs.
"""

from __future__ import annotations

import math

import numpy as np

from anim_glyphs import FACE, SIDE, Canvas

START_HALF = 0.13  # the half width at the round start, as a fraction of the side


def _bezier(p0, p1, p2, p3, steps: int = 40) -> list[tuple[float, float]]:
    pts = np.array([p0, p1, p2, p3], np.float32)
    out = []
    for k in range(steps + 1):
        t = k / steps
        u = 1.0 - t
        p = u ** 3 * pts[0] + 3 * u * u * t * pts[1] + 3 * u * t * t * pts[2] + t ** 3 * pts[3]
        out.append((float(p[0]), float(p[1])))
    return out


def _sides(centre: list[tuple[float, float]], half0: float) -> tuple[list, list]:
    """The two edges of the stroke: the half width shrinks from `half0` to 0."""
    pts = np.array(centre, np.float32)
    n = len(pts)
    left, right = [], []
    for i in range(n):
        t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        normal = np.array([-t[1], t[0]])
        half = half0 * (1.0 - (i / (n - 1)) ** 0.8)
        left.append(tuple(pts[i] + normal * half))
        right.append(tuple(pts[i] - normal * half))
    return left, right


def stroke(side: int = SIDE) -> np.ndarray:
    """Stroke: a filled wave, thick and round at the start, a point at the end."""
    cv = Canvas(side)
    centre = _bezier((0.18, 0.50), (0.45, 0.85), (0.60, 0.15), (0.90, 0.50))
    left, right = _sides(centre, START_HALF)
    for i in range(len(left) - 1):
        cv.fill([left[i], left[i + 1], right[i + 1], right[i]])
    start = centre[0]
    cv.disc(start, START_HALF, alpha=FACE)
    cv.path(left)
    cv.path(right)
    a0 = math.atan2(left[0][1] - start[1], left[0][0] - start[0])
    cv.arc(start, START_HALF, a0, a0 + math.pi)  # the round cap behind the start
    return cv.a


GLYPHS = {"STROKE": stroke}
