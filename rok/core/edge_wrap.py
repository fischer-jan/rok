"""Endless drags: a move that reaches the edge of the view keeps going.

When the mouse comes within EDGE_PX of a region edge during a move or a
stretch, the cursor jumps to the opposite side and the jump is added to an
offset, so the drag sees one unbroken mouse path.
"""

from __future__ import annotations

EDGE_PX = 2  # this close to an edge, the cursor wraps
INSET_PX = 8  # it lands this far inside the opposite edge


def wrap_target(mx: float, my: float, width: int, height: int) -> tuple[int, int] | None:
    """Where the cursor at region pixel (mx, my) jumps to, or None away from the edges.

    A region too small to wrap in never wraps.
    """
    if width <= 2 * INSET_PX + 2 or height <= 2 * INSET_PX + 2:
        return None
    x, y = int(mx), int(my)
    tx, ty = x, y
    if x <= EDGE_PX:
        tx = width - INSET_PX
    elif x >= width - 1 - EDGE_PX:
        tx = INSET_PX
    if y <= EDGE_PX:
        ty = height - INSET_PX
    elif y >= height - 1 - EDGE_PX:
        ty = INSET_PX
    if (tx, ty) == (x, y):
        return None
    return tx, ty
