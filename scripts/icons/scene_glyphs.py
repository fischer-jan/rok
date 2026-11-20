"""The glyphs of the scene's Tools that act on several models, drawn here
rather than generated.

    uv run python scripts/icons/make_icons.py PICK_LINKED ALIGN ARRAY CARVE

Isometric blocks as in select_glyphs.py: outlines at alpha 1, faces at FACE.
"""

from __future__ import annotations

import numpy as np

from anim_glyphs import SIDE, Canvas
from select_glyphs import C30, _block


def _at(origin, s: float, x: float, y: float, z: float):
    return (origin[0] + (x - y) * C30 * s, origin[1] + (x + y) * 0.5 * s + z * s)


def pick_linked(side: int = SIDE) -> np.ndarray:
    """Pick Linked Copies: two blocks side by side, a chain of two links between them."""
    cv = Canvas(side)
    s, origin = 0.22, (0.24, 0.3)
    _block(cv, origin, s, (1.45, -1.45, 0))
    _block(cv, origin, s, (0, 0, 0))
    mid = _at(origin, s, 1.225, -0.225, 0.5)
    for dx in (-0.06, 0.06):
        cv.ring((mid[0] + dx, mid[1]), 0.07)
    return cv.a


def align(side: int = SIDE) -> np.ndarray:
    """Align: three rows of blocks of different lengths, their left ends on one line."""
    cv = Canvas(side)
    s, origin = 0.125, (0.2, 0.1)
    for z, length in ((4.5, 2), (2.25, 3), (0.0, 1)):  # high rows first: the lower ones are nearer
        for k in range(length):
            _block(cv, origin, s, (k * 1.15, -k * 1.15, z))
    left = origin[0] - C30 * s - 0.04
    cv.line((left, origin[1] - 0.02), (left, origin[1] + 6.6 * s))
    return cv.a


def array(side: int = SIDE) -> np.ndarray:
    """Array: a row of blocks; the first is the model, the others its copies."""
    cv = Canvas(side)
    s, origin = 0.22, (0.24, 0.17)
    for x in (3, 2, 1):
        _block(cv, origin, s, (x, 0, 0), faces="none")
    _block(cv, origin, s, (0, 0, 0), faces="all")
    return cv.a


def carve(side: int = SIDE) -> np.ndarray:
    """Carve with Model: a slab with a block cut out of its near corner, the
    cutter block above the hole."""
    cv = Canvas(side)
    s, origin = 0.175, (0.5, 0.06)
    for cell in ((1, 1, 0), (1, 0, 0), (0, 1, 0)):
        _block(cv, origin, s, cell)
    _block(cv, origin, s, (0, 0, 2.7), faces="all")
    cv.line(_at(origin, s, 0.5, 0.5, 2.55), _at(origin, s, 0.5, 0.5, 1.35))  # the cut goes down

    return cv.a


GLYPHS = {"PICK_LINKED": pick_linked, "ALIGN": align, "ARRAY": array, "CARVE": carve}
