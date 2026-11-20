"""The cursor side of an endless drag (rok.core.edge_wrap).

After a jump, macOS can still deliver mouse events from before it, with the
cursor still at the old edge. Taken as new, each one would jump again and
add the window's width to the path once more: the drag ran away to the far
end (seen on the nav cube's section slide). So after a jump the
wrap waits: an event far from where the cursor landed is an old one and
leaves the path as it was, until an event comes near the landing point or
SETTLE_S has passed.
"""

from __future__ import annotations

import time

from rok.core.edge_wrap import wrap_target

SETTLE_PX = 48  # an event this close to the landing point is a new one
SETTLE_S = 0.25  # after this long every event counts again


class MouseWrap:
    """One drag's mouse path across cursor jumps. `pos` gives the region
    pixel the drag should use; near an edge it moves the cursor to the other
    side and adds the jump to the offset."""

    def __init__(self, clock=time.monotonic) -> None:
        self.dx = 0
        self.dy = 0
        self._clock = clock
        self._landing: tuple[int, int, float] | None = None  # (x, y, time) of the last jump
        self._last: tuple[int, int] | None = None

    def pos(self, context, event) -> tuple[int, int]:
        region = context.region
        mx, my = event.mouse_region_x, event.mouse_region_y
        if self._landing is not None:
            lx, ly, at = self._landing
            fresh = abs(mx - lx) <= SETTLE_PX and abs(my - ly) <= SETTLE_PX
            if not fresh and self._clock() - at < SETTLE_S and self._last is not None:
                return self._last  # from before the jump: the path stays
            self._landing = None
        path = (mx + self.dx, my + self.dy)
        self._last = path
        if region is None or context.window is None or event.type != "MOUSE_MOVE":
            return path
        target = wrap_target(mx, my, region.width, region.height)
        if target is not None:
            self.dx += mx - target[0]
            self.dy += my - target[1]
            self._landing = (int(target[0]), int(target[1]), self._clock())
            context.window.cursor_warp(region.x + target[0], region.y + target[1])
        return path
