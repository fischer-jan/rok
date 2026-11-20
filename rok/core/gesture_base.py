"""The base of every gesture the stroke operator drives."""

from __future__ import annotations


class Gesture:
    """What `rok.stroke` asks of a gesture.

    Every gesture writes `begin`, `extend`, `preview` and `finish` and sets `tool`,
    `hit` and `last`. The parts below are optional: a gesture overrides the ones it
    uses, and the defaults do nothing, so the operator never asks what a gesture
    can do.
    """

    hit = None  # the Hit under the press, for the hover; None when there is none
    last = None  # the last cell the drag reached
    visited = ()  # the cells a brush drag has touched, for the profile summary
    viewer = None  # Select: take only what this camera sees; a gesture that ignores it leaves it
    screen_drag = False  # True: the drag feeds pixels to `extend_screen`, not rays to `extend`
    locks_axis = False  # True: X, Y and Z during the drag call `toggle_axis_lock`

    def set_shift(self, shift: bool) -> None:
        """Called on every mouse move with the real Shift state."""

    def set_mouse(self, mx: float, my: float) -> None:
        """Called before every `extend` with the mouse in region pixels."""

    def toggle_axis_lock(self, axis: int) -> bool:
        """Lock the drag to one axis. True when something changed."""
        return False

    def begin_height(self) -> bool:
        """Start a height stage after the footprint. False when there is none."""
        return False
