"""The light dome's maths: a disc that shows the lights seen from above. Pure.

The middle of the disc is straight above the model, the rim is the horizon.
The disc turns with the view: the camera always stands at the bottom, so a
light on the left of the disc lights the model from the left in the view.
The height runs straight from the rim (0 degrees) to the middle (90). A
light below the horizon sits on the rim. Outside the rim, a ring turns
every light together.

Points are region pixels (y up); angles are degrees: a light's azimuth is the
world's (0 = +X, counter-clockwise seen from above), `camera` is the view's
azimuth (the side the camera is on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

RING = 12.0  # the turn ring's width, outside the disc
RING_GAP = 4.0
MARGIN = 6.0
HANDLE = 7.0  # a light's handle radius
GRAB = 11.0  # how near a press must be to take a handle


@dataclass(frozen=True)
class Dome:
    cx: float
    cy: float
    radius: float  # the horizon
    scale: float

    @property
    def ring_inner(self) -> float:
        return self.radius + RING_GAP * self.scale

    @property
    def ring_outer(self) -> float:
        return self.ring_inner + RING * self.scale


def dome_in(rect, scale: float = 1.0) -> Dome:
    """The biggest dome with its ring in `rect`, in the middle."""
    x, y, w, h = rect
    outer = min(w, h) / 2 - MARGIN * scale
    radius = outer - (RING + RING_GAP) * scale
    return Dome(x + w / 2, y + h / 2, max(10.0, radius), scale)


def screen_angle(azimuth: float, camera: float) -> float:
    """Degrees on the disc (0 = right, counter-clockwise) of a world azimuth:
    the camera's side at the bottom (-90)."""
    return -90.0 + (azimuth - camera)


def point(dome: Dome, azimuth: float, elevation: float, camera: float) -> tuple[float, float]:
    """Where a light stands on the disc."""
    r = dome.radius * (1.0 - min(90.0, max(0.0, elevation)) / 90.0)
    a = math.radians(screen_angle(azimuth, camera))
    return dome.cx + r * math.cos(a), dome.cy + r * math.sin(a)


def light_at(dome: Dome, x: float, y: float, camera: float) -> tuple[float, float]:
    """(world azimuth, elevation) of a point on the disc; a point outside the
    rim gives the horizon."""
    dx, dy = x - dome.cx, y - dome.cy
    r = math.hypot(dx, dy)
    elevation = 90.0 * (1.0 - min(1.0, r / dome.radius))
    if r < 1e-6:
        return camera + 180.0, 90.0
    a = math.degrees(math.atan2(dy, dx))
    return a + 90.0 + camera, elevation


def angle_at(dome: Dome, x: float, y: float) -> float:
    """The disc angle of a point (for a drag on the ring)."""
    return math.degrees(math.atan2(y - dome.cy, x - dome.cx))


def on_ring(dome: Dome, x: float, y: float) -> bool:
    r = math.hypot(x - dome.cx, y - dome.cy)
    return dome.ring_inner - 2 * dome.scale <= r <= dome.ring_outer + 3 * dome.scale


def on_disc(dome: Dome, x: float, y: float) -> bool:
    return math.hypot(x - dome.cx, y - dome.cy) <= dome.radius + HANDLE * dome.scale


def nearest(points, x: float, y: float, reach: float) -> int | None:
    """The index of the point nearest (x, y) within `reach`, the last one
    first when two lie on each other (it is drawn on top)."""
    best, best_d = None, reach
    for k in range(len(points) - 1, -1, -1):
        px, py = points[k]
        d = math.hypot(px - x, py - y)
        if d < best_d:
            best, best_d = k, d
    return best


def circle(cx: float, cy: float, r: float, steps: int = 64) -> list[tuple[float, float]]:
    """A circle as line segments: pairs of points."""
    pts = [(cx + r * math.cos(2 * math.pi * k / steps), cy + r * math.sin(2 * math.pi * k / steps))
           for k in range(steps + 1)]
    out = []
    for a, b in zip(pts, pts[1:]):
        out += [a, b]
    return out
