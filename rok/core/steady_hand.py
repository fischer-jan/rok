"""Steady Hand: the brush hangs on a string behind the mouse.

The brush stays where it is while the mouse moves inside the string's length.
Past that length the mouse pulls the brush along, so the brush trails the
mouse by exactly the length. Small shakes of the hand never reach the stroke,
and a long, slow pull gives a smooth line. A length of 0 turns it off.
"""

from __future__ import annotations

import math

MIN_LENGTH = 4  # pixels
MAX_LENGTH = 200
DEFAULT_LENGTH = 40


def length_of(prefs) -> int:
    """The string's length in pixels, 0 while Steady Hand is off."""
    return int(prefs.rok_steady_length) if getattr(prefs, "rok_steady_hand", False) else 0


def follow(brush: tuple[float, float], mouse: tuple[float, float], length: float) -> tuple[float, float]:
    """The brush's new place when the mouse is at `mouse` and the string is
    `length` pixels long."""
    if length <= 0:
        return mouse
    dx, dy = brush[0] - mouse[0], brush[1] - mouse[1]
    dist = math.hypot(dx, dy)
    if dist <= length:
        return brush
    k = length / dist
    return (mouse[0] + dx * k, mouse[1] + dy * k)
