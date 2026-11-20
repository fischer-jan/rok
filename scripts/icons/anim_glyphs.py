"""The Animate mode glyphs, drawn here rather than generated.

    uv run python scripts/icons/make_icons.py LEVEL_ANIM MOTION COLOUR_PICK ANCHOR

Each function returns a 1024 px alpha mask in the style of the generated set:
isometric shapes, the outline at alpha 1 and the faces at FACE (0.5), so the
drawing's one-ink tint shows them like the others. Coordinates are in the
unit square, y up.
"""

from __future__ import annotations

import math

import numpy as np

SIDE = 1024
LINE = 0.021  # half the stroke width, as a fraction of the side
FACE = 0.5


class Canvas:
    def __init__(self, side: int = SIDE) -> None:
        self.side = side
        yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
        self.x = (xx + 0.5) / side
        self.y = 1.0 - (yy + 0.5) / side
        self.a = np.zeros((side, side), np.float32)
        self.aa = 1.5 / side

    def _cover(self, d: np.ndarray) -> np.ndarray:
        return np.clip(0.5 - d / self.aa, 0.0, 1.0)

    def line(self, p, q, half: float = LINE) -> None:
        px, py = self.x - p[0], self.y - p[1]
        dx, dy = q[0] - p[0], q[1] - p[1]
        t = np.clip((px * dx + py * dy) / max(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
        d = np.hypot(px - t * dx, py - t * dy) - half
        self.a = np.maximum(self.a, self._cover(d))

    def path(self, points, closed: bool = False, half: float = LINE) -> None:
        pts = list(points)
        pairs = zip(pts, pts[1:] + (pts[:1] if closed else []))
        for p, q in pairs:
            self.line(p, q, half)

    def fill(self, points, alpha: float = FACE) -> None:
        """A convex polygon, counter-clockwise or not."""
        pts = list(points)
        inside = np.ones_like(self.a, dtype=bool)
        sign = None
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
        sign = 1.0 if area > 0 else -1.0
        for a, b in zip(pts, pts[1:] + pts[:1]):
            cross = (b[0] - a[0]) * (self.y - a[1]) - (b[1] - a[1]) * (self.x - a[0])
            inside &= cross * sign >= 0
        self.a = np.maximum(self.a, inside.astype(np.float32) * alpha)

    def ring(self, c, r: float, half: float = LINE) -> None:
        d = np.abs(np.hypot(self.x - c[0], self.y - c[1]) - r) - half
        self.a = np.maximum(self.a, self._cover(d))

    def disc(self, c, r: float, alpha: float = 1.0) -> None:
        d = np.hypot(self.x - c[0], self.y - c[1]) - r
        self.a = np.maximum(self.a, self._cover(d) * alpha)

    def arc(self, c, r: float, a0: float, a1: float, steps: int = 48, half: float = LINE) -> None:
        pts = [(c[0] + r * math.cos(a0 + (a1 - a0) * k / steps), c[1] + r * math.sin(a0 + (a1 - a0) * k / steps))
               for k in range(steps + 1)]
        self.path(pts, half=half)


def _iso_cube(cv: Canvas, cx: float, cy: float, s: float, face: bool = True) -> dict:
    """An isometric cube: its centre of the top-front corner at (cx, cy), edge
    `s`. Returns its named corners."""
    c30, s30 = math.cos(math.radians(30)) * s, 0.5 * s
    front = (cx, cy)  # the near vertical edge's top
    bottom = (cx, cy - s)
    left_t, right_t = (cx - c30, cy + s30), (cx + c30, cy + s30)
    left_b, right_b = (cx - c30, cy + s30 - s), (cx + c30, cy + s30 - s)
    back = (cx, cy + 2 * s30)
    if face:
        cv.fill([front, right_t, back, left_t])  # the top face
    cv.path([front, left_t, back, right_t], closed=True)
    cv.path([left_t, left_b, bottom, right_b, right_t])
    cv.line(front, bottom)
    return {"front": front, "bottom": bottom, "left_t": left_t, "right_t": right_t,
            "left_b": left_b, "right_b": right_b, "back": back}


def level_anim(side: int = SIDE) -> np.ndarray:
    """The Animate cell: a cube in flight, three streaks behind it."""
    cv = Canvas(side)
    corners = _iso_cube(cv, 0.62, 0.56, 0.3)
    lx = corners["left_t"][0]
    for k, (y, length) in enumerate(((0.62, 0.26), (0.47, 0.34), (0.32, 0.22))):
        x1 = lx - 0.06 - 0.02 * k
        cv.line((x1 - length, y), (x1, y))
    return cv.a


def motion(side: int = SIDE) -> np.ndarray:
    """Motion (the gallery): a card with a play mark, a second card behind it."""
    cv = Canvas(side)

    def card(x0, y0, x1, y1):
        cv.path([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True)

    card(0.30, 0.44, 0.84, 0.86)
    # The front card hides the back one's lower left.
    cv.a[(cv.x > 0.12) & (cv.x < 0.74) & (cv.y > 0.10) & (cv.y < 0.62)] = 0.0
    cv.fill([(0.16, 0.14), (0.70, 0.14), (0.70, 0.58), (0.16, 0.58)])
    card(0.16, 0.14, 0.70, 0.58)
    cv.fill([(0.35, 0.24), (0.35, 0.48), (0.55, 0.36)], alpha=1.0)
    return cv.a


def colour_pick(side: int = SIDE) -> np.ndarray:
    """Colour: an eyedropper over a drop."""
    cv = Canvas(side)
    a, b = (0.40, 0.46), (0.74, 0.80)
    # The glass tube: two rails and a tip.
    nx, ny = -(b[1] - a[1]), b[0] - a[0]
    n = math.hypot(nx, ny)
    nx, ny = nx / n * 0.055, ny / n * 0.055
    cv.fill([(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)])
    cv.path([(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny)])
    cv.path([(a[0] - nx, a[1] - ny), (b[0] - nx, b[1] - ny)])
    cv.line((a[0] + nx, a[1] + ny), (0.32, 0.38))
    cv.line((a[0] - nx, a[1] - ny), (0.32, 0.38))
    # The bulb and its collar.
    cv.disc((0.80, 0.86), 0.075)
    cv.line((0.66, 0.84), (0.84, 0.66), half=LINE * 1.2)
    # The drop.
    cv.disc((0.22, 0.20), 0.085, alpha=FACE)
    cv.ring((0.22, 0.20), 0.085)
    cv.fill([(0.22, 0.36), (0.155, 0.245), (0.285, 0.245)], alpha=FACE)
    cv.path([(0.147, 0.24), (0.22, 0.36), (0.293, 0.24)])
    return cv.a


def anchor(side: int = SIDE) -> np.ndarray:
    """Anchor: a cube with a pivot mark at its centre and the axis through it."""
    cv = Canvas(side)
    c = _iso_cube(cv, 0.5, 0.60, 0.40, face=False)
    centre = (0.5, 0.60 - 0.20 + 0.10)
    cv.line((centre[0], 0.06), (centre[0], 0.94), half=LINE * 0.7)
    cv.disc(centre, 0.1, alpha=1.0)
    cv.ring(centre, 0.17)
    return cv.a


GLYPHS = {"LEVEL_ANIM": level_anim, "MOTION": motion, "COLOUR_PICK": colour_pick, "ANCHOR": anchor}
