"""The key cursor: one cell of the active model that the arrow keys step
around, so a click can land exactly where you want without the mouse. Pure:
the steps and the clamp.

The steps follow the view. They are named after the camera keys they match:
W (Up) away from you and S (Down) toward you, along the floor axis nearest
the way you look; D (Right) to the right and A (Left) to the left; E
(Shift+Up) a layer up and Q (Shift+Down) a layer down.
"""

from __future__ import annotations

Cell = tuple[int, int, int]
KEYS = ("W", "A", "S", "D", "Q", "E")


def floor_axes(view_dir) -> tuple[Cell, Cell]:
    """(forward, right): the floor's grid axes nearest the way the view
    looks and to its right. `view_dir` is the direction the camera looks."""
    fx, fy = float(view_dir[0]), float(view_dir[1])
    if abs(fx) >= abs(fy):
        forward = (1 if fx >= 0 else -1, 0, 0)
    else:
        forward = (0, 1 if fy >= 0 else -1, 0)
    right = (forward[1], -forward[0], 0)  # forward × up
    return forward, right


def step(key: str, view_dir) -> Cell:
    """The one-cell step of a key, for a view looking along `view_dir`."""
    forward, right = floor_axes(view_dir)
    key = key.upper()
    if key == "W":
        return forward
    if key == "S":
        return (-forward[0], -forward[1], 0)
    if key == "D":
        return right
    if key == "A":
        return (-right[0], -right[1], 0)
    if key == "E":
        return (0, 0, 1)
    if key == "Q":
        return (0, 0, -1)
    raise ValueError(f"not a cursor key: {key!r}")


def moved(cell, key: str, view_dir, size) -> Cell:
    """`cell` one step on, kept inside a grid of `size`."""
    d = step(key, view_dir)
    return tuple(max(0, min(int(size[k]) - 1, int(cell[k]) + d[k])) for k in range(3))  # type: ignore[return-value]
