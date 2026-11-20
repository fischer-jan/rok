"""Grow, shrink and hollow: binary morphology on bool masks and uint8 voxels.

numpy only. Every step is a handful of in-place slice ORs on the padded
bounding box of the mask, so a small selection costs little on a big grid.
6-connectivity (the default) keeps the stair-step voxel look; `round_corners`
is 26-connectivity, a 3 x 3 x 3 box, done separably along the three axes.
The grid border counts as empty.
"""

from __future__ import annotations

import numpy as np

# Neighbour order for grown colours: -x, +x, -y, +y, -z, +z.
_STEPS = ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))


def _shift_slices(axis: int, step: int, k: int = 1) -> tuple[tuple, tuple]:
    """(dst, src) slice tuples so that dst reads the cell `step * k` away along axis."""
    dst = [slice(None)] * 3
    src = [slice(None)] * 3
    if step < 0:
        dst[axis] = slice(k, None)
        src[axis] = slice(None, -k)
    else:
        dst[axis] = slice(None, -k)
        src[axis] = slice(k, None)
    return tuple(dst), tuple(src)


def bbox_slices(mask: np.ndarray, pad: int) -> tuple[slice, slice, slice] | None:
    """Bounding box of the True cells padded by `pad`, clipped to the grid. None when empty."""
    out = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        hit = np.flatnonzero(mask.any(axis=other))
        if hit.size == 0:
            return None
        lo = max(int(hit[0]) - pad, 0)
        hi = min(int(hit[-1]) + 1 + pad, mask.shape[axis])
        out.append(slice(lo, hi))
    return tuple(out)


def _dilate_cross(m: np.ndarray) -> np.ndarray:
    """One 6-connected step: six slice ORs, no roll, no pad."""
    out = m.copy()
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    out[:, :, 1:] |= m[:, :, :-1]
    out[:, :, :-1] |= m[:, :, 1:]
    return out


def _dilate_axis(m: np.ndarray, axis: int, n: int) -> np.ndarray:
    """1D dilation of radius n along one axis."""
    out = m.copy()
    for k in range(1, n + 1):
        for step in (-1, 1):
            dst, src = _shift_slices(axis, step, k)
            out[dst] |= m[src]
    return out


def _dilate_box(m: np.ndarray, n: int) -> np.ndarray:
    """26-connected dilation by n, separable: exact for the Chebyshev ball in one pass."""
    for axis in range(3):
        m = _dilate_axis(m, axis, n)
    return m


def _dilate_local(m: np.ndarray, n: int, round_corners: bool) -> np.ndarray:
    if round_corners:
        return _dilate_box(m, n)
    for _ in range(n):
        m = _dilate_cross(m)
    return m


def dilate(mask: np.ndarray, n: int = 1, round_corners: bool = False) -> np.ndarray:
    """Grow a bool mask by n cells. Returns a new mask of the same shape."""
    mask = np.asarray(mask, dtype=bool)
    if n <= 0:
        return mask.copy()
    out = np.zeros(mask.shape, dtype=bool)
    box = bbox_slices(mask, n)
    if box is None:
        return out
    out[box] = _dilate_local(mask[box], n, round_corners)
    return out


def erode(mask: np.ndarray, n: int = 1, round_corners: bool = False) -> np.ndarray:
    """Shrink a bool mask by n cells. A cell on the grid wall counts as exposed."""
    mask = np.asarray(mask, dtype=bool)
    if n <= 0:
        return mask.copy()
    out = np.zeros(mask.shape, dtype=bool)
    box = bbox_slices(mask, 0)
    if box is None:
        return out
    # ~dilate(~mask) on the box with one False layer around it, so the border spreads in.
    local = mask[box]
    padded = np.zeros(tuple(d + 2 for d in local.shape), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = local
    grown = _dilate_local(~padded, n, round_corners)
    out[box] = ~grown[1:-1, 1:-1, 1:-1]
    return out


def _colour_from(out: np.ndarray, todo: np.ndarray, source: np.ndarray, axis: int, step: int) -> None:
    """Give the uncoloured `todo` cells the colour of their `source` neighbour along axis."""
    dst, src = _shift_slices(axis, step)
    sel = todo[dst] & source[src]
    if not sel.any():
        return
    out[dst][sel] = out[src][sel]
    todo[dst][sel] = False


def _grow_step_cross(out: np.ndarray, filled: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """One 6-connected grow step from `seed`. Colours in the fixed neighbour order."""
    new = _dilate_cross(seed) & ~filled
    todo = new.copy()
    for axis, step in _STEPS:
        _colour_from(out, todo, seed, axis, step)
    return new


def _grow_step_box(out: np.ndarray, filled: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """One 26-connected grow step, separable: x, then y, then z, colours along the chain."""
    new = np.zeros(seed.shape, dtype=bool)
    spread = seed
    for axis in range(3):
        wider = _dilate_axis(spread, axis, 1)
        added = wider & ~filled & ~new
        todo = added.copy()
        for step in (-1, 1):
            _colour_from(out, todo, spread, axis, step)
        new |= added
        spread = wider
    return new


def grow(
    voxels: np.ndarray, n: int, *, limit: np.ndarray | None = None, round_corners: bool = False
) -> np.ndarray:
    """Add n layers of voxels. A new cell takes the colour of the neighbour it grew from.

    With `limit` (a bool mask) only cells next to the mask grow, and nothing
    outside the mask is overwritten. Returns a uint8 copy.
    """
    out = np.array(voxels, dtype=np.uint8, copy=True)
    filled = out != 0
    seed = filled if limit is None else filled & np.asarray(limit, dtype=bool)
    step = _grow_step_box if round_corners else _grow_step_cross
    for _ in range(n):
        box = bbox_slices(seed, 1)
        if box is None:
            break
        new = np.zeros(seed.shape, dtype=bool)
        new[box] = step(out[box], filled[box], seed[box])
        if not new.any():
            break
        filled |= new
        seed = new
    return out


def shrink(
    voxels: np.ndarray, n: int, *, limit: np.ndarray | None = None, round_corners: bool = False
) -> np.ndarray:
    """Remove n layers: per step the filled cells with an empty neighbour.

    With `limit` only the selected ones among them go. A selected cell whose
    neighbours are all filled is interior and stays. Returns a uint8 copy.
    """
    out = np.array(voxels, dtype=np.uint8, copy=True)
    filled = out != 0
    for _ in range(n):
        remove = filled & ~erode(filled, 1, round_corners)
        if limit is not None:
            remove &= limit
        if not remove.any():
            break
        out[remove] = 0
        filled &= ~remove
    return out


def hollow(voxels: np.ndarray, wall: int, *, limit: np.ndarray | None = None) -> np.ndarray:
    """Empty the inside and keep a wall of `wall` voxels. With `limit` only inside the mask."""
    out = np.array(voxels, dtype=np.uint8, copy=True)
    interior = erode(out != 0, wall)
    if limit is not None:
        interior &= limit
    out[interior] = 0
    return out


def keep_masked(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`after` with the mask's rules, as the tools keep them: a masked voxel
    stays as it was, and no new voxel touches a masked one by a face."""
    mask = np.asarray(mask, dtype=bool)
    out = np.where(mask, before, after).astype(np.uint8)
    added = (np.asarray(before) == 0) & (out != 0)
    out[added & dilate(mask, 1)] = 0
    return out


LOOSE_DROP = 4  # a new part this small that cannot be joined is left out


def join_loose(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """`after` with every new part joined face to face to the rest.

    A part is loose when no face-to-face path leads from it to a voxel that was
    there before or to the grid's floor (z = 0). A loose part that touches the
    rest edge to edge gets one bridge voxel, one that touches it corner to
    corner gets two, in the part's own colour. A loose part of at most
    LOOSE_DROP voxels that cannot be joined is left out; a bigger one stays.
    Voxels that were there before never change. (Tree shaders made leaves that
    touched their crown only at an edge: rok_check counted them as separate
    parts.)"""
    from .region import components, connected_blobs

    before = np.asarray(before)
    out = np.array(after, dtype=np.uint8, copy=True)
    filled = out != 0
    if not filled.any():
        return out
    seeds = np.concatenate([np.argwhere((before != 0) & filled), np.argwhere(filled[:, :, :1])])
    anchored = connected_blobs(out, seeds)
    loose = filled & ~anchored
    if not loose.any():
        return out
    todo = components(loose)
    while todo:
        left = []
        for part in todo:
            joined = _bridge(out, part, anchored)
            if joined:
                anchored[tuple(part.T)] = True
            else:
                left.append(part)
        if len(left) == len(todo):
            break  # no part joined in this round: the rest touch nothing that holds
        todo = left
    for part in todo:
        if len(part) <= LOOSE_DROP and not np.any(before[tuple(part.T)]):
            out[tuple(part.T)] = 0
    return out


def _bridge(out: np.ndarray, part: np.ndarray, anchored: np.ndarray) -> bool:
    """Join `part` (N, 3 cells) to `anchored` with one or two new voxels in
    empty cells of `out`. Writes into `out` and `anchored`; True when joined."""
    lo = np.maximum(part.min(axis=0) - 3, 0)
    hi = np.minimum(part.max(axis=0) + 4, np.asarray(out.shape))
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    c = np.zeros(tuple(int(v) for v in hi - lo), dtype=bool)
    c[tuple((part - lo).T)] = True
    a = anchored[box]
    empty = out[box] == 0
    near_c = _dilate_cross(c) & empty
    near_a = _dilate_cross(a) & empty
    values, counts = np.unique(out[tuple(part.T)], return_counts=True)
    colour = values[np.argmax(counts)]
    both = near_c & near_a
    cells = []
    if both.any():
        cells = [np.argwhere(both)[0]]
    else:
        step = near_c & _dilate_cross(near_a)
        if step.any():
            first = np.argwhere(step)[0]
            one = np.zeros_like(c)
            one[tuple(first)] = True
            second = near_a & _dilate_cross(one)
            cells = [first, np.argwhere(second)[0]]
    if not cells:
        return False
    sub = out[box]
    held = anchored[box]
    for cell in cells:
        sub[tuple(cell)] = colour
        held[tuple(cell)] = True
    return True


def changed_cells(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) int64 cells where before != after, and after's values there as (N,) uint8."""
    diff = before != after
    cells = np.argwhere(diff).astype(np.int64)
    values = np.asarray(after, dtype=np.uint8)[diff]
    return cells, values


def bevel_mask(filled: np.ndarray, limit: np.ndarray | None = None) -> np.ndarray:
    """The cells one bevel takes off: every convex edge, one voxel deep.

    A cell goes when it is exposed on two sides at right angles, the block
    behind it is at least two thick both ways (the cells opposite the two open
    sides and the one diagonally behind are filled), and it has a filled
    neighbour along the third axis (so a one-voxel plate keeps its outline).
    So a one-voxel wing, rim or plate keeps its edge, and a 3-cube keeps its
    centre and six face middles. The grid wall counts as filled, so a model
    standing on the floor keeps its base edge. `limit` keeps the bevel to a
    mask (the selection).
    """
    f = np.asarray(filled, dtype=bool)
    p = np.pad(f, 1, constant_values=True)  # the wall is not an open side
    n = p.shape

    def at(dx: int, dy: int, dz: int) -> np.ndarray:
        """The neighbour at an offset, for every cell of the unpadded grid."""
        return p[1 + dx : n[0] - 1 + dx, 1 + dy : n[1] - 1 + dy, 1 + dz : n[2] - 1 + dz]

    remove = np.zeros(f.shape, dtype=bool)
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (-1, 1):
                for sj in (-1, 1):
                    open_i = [0, 0, 0]
                    open_j = [0, 0, 0]
                    open_i[i] = si
                    open_j[j] = sj
                    back_i = [-v for v in open_i]
                    back_j = [-v for v in open_j]
                    diag = [back_i[k] + back_j[k] for k in range(3)]
                    third = 3 - i - j
                    side_a = [0, 0, 0]
                    side_b = [0, 0, 0]
                    side_a[third] = 1
                    side_b[third] = -1
                    remove |= (
                        f
                        & ~at(*open_i)
                        & ~at(*open_j)
                        & at(*back_i)
                        & at(*back_j)
                        & at(*diag)
                        & (at(*side_a) | at(*side_b))
                    )
    if limit is not None:
        remove &= np.asarray(limit, dtype=bool)
    return remove



SMOOTH_MAJORITY = 14  # of the 27 cells of a 3 x 3 x 3 block: more than half


def _box_count(filled: np.ndarray) -> np.ndarray:
    """How many cells of the 3 x 3 x 3 block around each cell are filled, the
    cell too. Outside the grid the border cell repeats, so a model standing on
    the grid wall is neither eaten nor grown there."""
    p = np.pad(filled.astype(np.uint8), 1, mode="edge")
    sx, sy, sz = filled.shape
    count = np.zeros(filled.shape, dtype=np.uint8)
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                count += p[dx : dx + sx, dy : dy + sy, dz : dz + sz]
    return count


def _plate_cells(filled: np.ndarray) -> np.ndarray:
    """Filled cells in a wall or plate at most two thick, along exactly one axis.
    A bump or a line is thin along two axes and is not a plate."""
    p = np.pad(filled, 2, constant_values=False)
    n = p.shape

    def at(axis: int, k: int) -> np.ndarray:
        s = [slice(2, n[a] - 2) for a in range(3)]
        s[axis] = slice(2 + k, n[axis] - 2 + k)
        return p[tuple(s)]

    thin = []
    for axis in range(3):
        one = ~at(axis, -1) & ~at(axis, 1)
        two = (~at(axis, -1) & at(axis, 1) & ~at(axis, 2)) | (~at(axis, 1) & at(axis, -1) & ~at(axis, -2))
        thin.append(one | two)
    axes = thin[0].astype(np.uint8) + thin[1] + thin[2]
    return filled & (axes == 1)


def _mode_colours(voxels: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """For each cell (an (n, 3) array), the most common colour among its 26
    neighbours; ties go to the lower colour index."""
    p = np.pad(voxels, 1)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    near = np.stack([p[cells[:, 0] + 1 + dx, cells[:, 1] + 1 + dy, cells[:, 2] + 1 + dz] for dx, dy, dz in offsets], axis=1)
    votes = np.zeros((len(cells), 256), dtype=np.int32)
    rows = np.repeat(np.arange(len(cells)), near.shape[1])
    np.add.at(votes, (rows, near.ravel()), 1)
    votes[:, 0] = 0  # empty is no colour
    return votes.argmax(axis=1).astype(voxels.dtype)


def smooth(voxels: np.ndarray, *, limit: np.ndarray | None = None) -> np.ndarray:
    """One smoothing pass: every cell takes the majority of its 3 x 3 x 3 block.

    A filled cell with fewer than SMOOTH_MAJORITY filled cells around it (a
    corner, a bump, a lone voxel) goes; an empty cell with at least that many
    (a dent, a hole) is filled with the colour most of its neighbours have.
    Walls and plates up to two voxels thick keep their shape: the vote would
    eat them. With `limit` (the selection) only selected voxels go, and only
    empty cells touching the selection fill. Run it again for more.
    """
    filled = voxels != 0
    count = _box_count(filled)
    remove = filled & (count < SMOOTH_MAJORITY) & ~_plate_cells(filled)
    add = ~filled & (count >= SMOOTH_MAJORITY)
    if limit is not None:
        mask = np.asarray(limit, dtype=bool)
        remove &= mask
        add &= _dilate_box(mask, 1)
    out = voxels.copy()
    out[remove] = 0
    cells = np.argwhere(add)
    if len(cells):
        out[add] = _mode_colours(voxels, cells)
    return out
