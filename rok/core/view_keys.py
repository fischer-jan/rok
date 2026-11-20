"""WASD + QE move the view.

A and D orbit the camera left and right around the model, W and S tilt it
up and down, Q and E zoom out and in. Holding a key moves the view smoothly;
a quick tap moves it one step. The camera's height stays short of straight
above or below, so the view never flips over.
"""

from __future__ import annotations

import math

ORBIT, TILT, ZOOM = "ORBIT", "TILT", "ZOOM"
# key -> (what it does, which way): +1 is right for ORBIT, up for TILT, in for ZOOM.
VIEW_KEYS = {
    "A": (ORBIT, -1),
    "D": (ORBIT, 1),
    "W": (TILT, 1),
    "S": (TILT, -1),
    "Q": (ZOOM, -1),
    "E": (ZOOM, 1),
}

ORBIT_DEG_PER_S = 120.0  # at full speed
TILT_DEG_PER_S = 90.0
ZOOM_PER_S = 2.4  # the view distance shrinks or grows by this factor per second
RAMP_S = 0.25  # from rest to full speed
TAP_S = 0.2  # a release sooner than this is a tap
TAP_DEG = 15.0  # a tap orbits or tilts this far
TAP_ZOOM = 1.25  # a tap zooms by this factor
MAX_ELEVATION_DEG = 85.0
MIN_DISTANCE = 0.5
MAX_DISTANCE = 5000.0


def travelled(held_s: float) -> float:
    """Seconds of full speed travelled after holding `held_s`: the speed
    rises linearly over RAMP_S, then stays at full."""
    t = max(0.0, float(held_s))
    if t <= RAMP_S:
        return t * t / (2.0 * RAMP_S)
    return RAMP_S / 2.0 + (t - RAMP_S)


def degrees_between(kind: str, t0: float, t1: float) -> float:
    """Degrees an ORBIT or TILT key turns from held time t0 to t1."""
    rate = ORBIT_DEG_PER_S if kind == ORBIT else TILT_DEG_PER_S
    return rate * (travelled(t1) - travelled(t0))


def zoom_between(t0: float, t1: float) -> float:
    """The factor a ZOOM key scales by from held time t0 to t1 (above 1)."""
    return math.exp(math.log(ZOOM_PER_S) * (travelled(t1) - travelled(t0)))


def tap_rest(kind: str, held_s: float) -> float:
    """What a tap still owes after `held_s` of holding: degrees for ORBIT
    and TILT, a zoom factor for ZOOM (1.0: nothing), so a tap always moves
    one whole step."""
    if kind == ZOOM:
        done = zoom_between(0.0, held_s)
        return max(1.0, TAP_ZOOM / done)
    done = degrees_between(kind, 0.0, held_s)
    return max(0.0, TAP_DEG - done)


def clamp_tilt(elevation_deg: float, delta_deg: float) -> float:
    """The tilt that keeps the camera's elevation inside +-MAX_ELEVATION_DEG:
    `delta_deg` itself, or as much of it as fits."""
    target = max(-MAX_ELEVATION_DEG, min(MAX_ELEVATION_DEG, elevation_deg + delta_deg))
    return target - elevation_deg


def clamp_distance(distance: float) -> float:
    return max(MIN_DISTANCE, min(MAX_DISTANCE, float(distance)))
