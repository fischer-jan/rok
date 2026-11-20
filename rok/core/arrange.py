"""Where models go when the scene lines them up, spreads them, repeats them or
carves one with another. Pure: boxes in, positions and masks out.

A box is (position, size): the model's low corner in scene cells and its grid
size, both (x, y, z).
"""

from __future__ import annotations

import numpy as np

Cell = tuple[int, int, int]
Box = tuple[Cell, Cell]

MIN, CENTRE, MAX, SPREAD = "MIN", "CENTRE", "MAX", "SPREAD"
ARRAY_MAX = 64  # copies of one array, the original counted


def _moved(position: Cell, axis: int, value: int) -> Cell:
    out = [int(p) for p in position]
    out[axis] = int(value)
    return (out[0], out[1], out[2])


def aligned(boxes: list[Box], lead: int, axis: int, how: str) -> list[Cell]:
    """The positions that line every box up on box `lead` along `axis`: their
    low sides (MIN), centres (CENTRE) or high sides (MAX) meet the lead's. A
    centre that falls between two cells rounds down."""
    (lp, ls) = boxes[lead]
    lo, hi = int(lp[axis]), int(lp[axis]) + int(ls[axis])
    out = []
    for position, size in boxes:
        n = int(size[axis])
        if how == MIN:
            at = lo
        elif how == MAX:
            at = hi - n
        else:
            at = (lo + hi - n) // 2
        out.append(_moved(position, axis, at))
    return out


def spread(boxes: list[Box], axis: int) -> list[Cell]:
    """The positions that leave equal gaps between the boxes along `axis`, in
    the order of their centres. The first and the last box stay; a gap that
    does not divide evenly gives its spare cells to the first gaps."""
    if len(boxes) < 3:
        return [tuple(int(v) for v in p) for p, _s in boxes]
    order = sorted(range(len(boxes)), key=lambda k: (2 * int(boxes[k][0][axis]) + int(boxes[k][1][axis]), k))
    first, last = boxes[order[0]], boxes[order[-1]]
    start = int(first[0][axis]) + int(first[1][axis])
    end = int(last[0][axis])
    inner = sum(int(boxes[k][1][axis]) for k in order[1:-1])
    room = end - start - inner
    gaps = len(order) - 1
    step, spare = divmod(room, gaps)
    out = [tuple(int(v) for v in p) for p, _s in boxes]
    at = start
    for g, k in enumerate(order[1:-1]):
        at += step + (1 if g < spare else 0)
        out[k] = _moved(boxes[k][0], axis, at)
        at += int(boxes[k][1][axis])
    return out


def array_offsets(size: Cell, count: int, axis: int, direction: int, gap: int) -> list[Cell]:
    """The offsets of an array's copies from the original (the original is
    not in the list): copy k moves k * (size + gap) cells along `axis`, in
    `direction` (+1 or -1). The gap may be negative down to 1 - size, so the
    copies overlap but never sit on the original."""
    count = max(1, min(int(count), ARRAY_MAX))
    n = int(size[axis])
    step = max(1, n + int(gap)) * (1 if direction >= 0 else -1)
    out = []
    for k in range(1, count):
        d = [0, 0, 0]
        d[axis] = k * step
        out.append((d[0], d[1], d[2]))
    return out


def carve_mask(target_voxels: np.ndarray, target_position: Cell, cutter_voxels: np.ndarray,
               cutter_position: Cell, protected: np.ndarray | None = None) -> np.ndarray | None:
    """The target's cells that the cutter empties: filled in both where the
    two grids overlap in the scene, and not protected (the target's mask).
    None when the cutter takes nothing."""
    tshape, cshape = target_voxels.shape, cutter_voxels.shape
    lo = [max(int(t), int(c)) for t, c in zip(target_position, cutter_position)]
    hi = [min(int(t) + ts, int(c) + cs) for t, c, ts, cs in zip(target_position, cutter_position, tshape, cshape)]
    if any(h <= low for low, h in zip(lo, hi)):
        return None
    tbox = tuple(slice(low - int(t), h - int(t)) for low, h, t in zip(lo, hi, target_position))
    cbox = tuple(slice(low - int(c), h - int(c)) for low, h, c in zip(lo, hi, cutter_position))
    mask = np.zeros(tshape, dtype=bool)
    mask[tbox] = (target_voxels[tbox] != 0) & (cutter_voxels[cbox] != 0)
    if protected is not None:
        mask &= ~protected
    return mask if mask.any() else None
