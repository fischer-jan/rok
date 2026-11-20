"""2D geometry the UI code shares: rectangles, line lists and polyline distance.

Every function works in region pixels."""

from __future__ import annotations

import math

import numpy as np

Rect = tuple[float, float, float, float]  # x, y, width, height; y grows upward

QUAD_TRIS = np.array([0, 1, 2, 0, 2, 3])  # a quad's four corners as two triangles

_EPS = 1e-9


def in_rect(rect: Rect | None, px: float, py: float) -> bool:
    """Whether the pixel lies in `rect`: left and bottom edges in, right and top out.
    None is no rectangle, so nothing lies in it."""
    if rect is None:
        return False
    x, y, w, h = rect
    return bool(x <= px < x + w and y <= py < y + h)


def rect_tris(rects) -> np.ndarray:
    """One rectangle (x, y, w, h) or a (Q, 4) batch as TRIS positions, (6 Q, 2) float32."""
    rects = np.asarray(rects, dtype=np.float32).reshape(-1, 4)
    x = rects[:, 0]
    y = rects[:, 1]
    x1 = x + rects[:, 2]
    y1 = y + rects[:, 3]
    corners = np.stack(
        [
            np.stack([x, y], axis=1),
            np.stack([x1, y], axis=1),
            np.stack([x1, y1], axis=1),
            np.stack([x, y1], axis=1),
        ],
        axis=1,
    )  # (Q, 4, 2)
    return corners[:, QUAD_TRIS, :].reshape(-1, 2).astype(np.float32)


def loop_lines(points) -> np.ndarray:
    """A closed polygon (N, 2) as LINES pairs, (2 N, 2) float32."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1).reshape(-1, 2)


def fit_glyph(unit, rect: Rect, inset: float = 0.0) -> np.ndarray:
    """Place a glyph drawn in the unit square as LINES pairs in the centred square of
    `rect` minus `inset`. Nothing when that square is empty. Coordinates land on
    half pixels, which is where a one-pixel line is sharp."""
    x, y, w, h = (float(v) for v in rect)
    side = min(w, h) - 2.0 * inset
    if side <= 0.0:
        return np.zeros((0, 2), dtype=np.float32)
    x0 = x + (w - side) / 2.0
    y0 = y + (h - side) / 2.0
    out = np.asarray(unit, dtype=np.float32) * side + np.array([x0, y0], dtype=np.float32)
    return (np.floor(out) + 0.5).astype(np.float32)


def polyline_distance(pts: np.ndarray, mx: float, my: float) -> float:
    """The nearest distance from (mx, my) to the closed polyline `pts`, or inf when
    it has fewer than two points or any point is not finite (an edge-on or
    off-screen projection)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2 or not np.all(np.isfinite(pts)):
        return math.inf
    a = pts
    b = np.roll(pts, -1, axis=0)
    ab = b - a
    ap = np.array([mx, my], dtype=np.float64) - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > _EPS, np.einsum("ij,ij->i", ap, ab) / np.maximum(denom, _EPS), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + ab * t[:, None]
    d = np.hypot(closest[:, 0] - mx, closest[:, 1] - my)
    return float(d.min())
