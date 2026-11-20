"""Whole-model operations: flip, rotate, resize, rotation bytes, mirror, fill, loop,
scale, repeat, diagonal. Pure numpy."""

from __future__ import annotations

import numpy as np

MIN_SIZE = 1
MAX_SIZE = 256

# Plane that turns for a rotation about each axis. np.rot90 with these axes is a
# counter-clockwise turn seen from the positive end of the axis.
_ROTATE_AXES = {2: (0, 1), 0: (1, 2), 1: (2, 0)}


def flip(voxels: np.ndarray, axis: int) -> np.ndarray:
    """Mirror the model along one axis. Contiguous uint8 copy, same shape."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    return np.array(np.flip(voxels, axis=axis), dtype=np.uint8, order="C")


def rotate90(voxels: np.ndarray, axis: int) -> np.ndarray:
    """Turn the model by 90 degrees counter-clockwise seen from the positive axis.

    A voxel at (2, 0, 0) of a (3, 2, 1) grid lands at (1, 2, 0) for axis 2.
    The shape changes when the turning plane is not square.
    """
    if axis not in _ROTATE_AXES:
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    turned = np.rot90(voxels, k=1, axes=_ROTATE_AXES[axis])
    return np.array(turned, dtype=np.uint8, order="C")


def resize(voxels: np.ndarray, new_size) -> np.ndarray:
    """New grid of new_size with the overlap copied. Anchored at the origin corner."""
    size = tuple(int(v) for v in new_size)
    if len(size) != 3:
        raise ValueError(f"new_size must have 3 dimensions, got {new_size!r}")
    for value in size:
        if not MIN_SIZE <= value <= MAX_SIZE:
            raise ValueError(f"size must be within {MIN_SIZE}..{MAX_SIZE}, got {size}")
    out = np.zeros(size, dtype=np.uint8)
    keep = tuple(slice(0, min(old, new)) for old, new in zip(voxels.shape, size))
    out[keep] = voxels[keep]
    return out


# --- .vox rotation bytes ---------------------------------------------

# The _r byte: bits 0-1 the column of row 0, bits 2-3 the column of row 1,
# bits 4, 5, 6 the sign of rows 0, 1, 2 of the row-major matrix M. Rok applies
# R = M^T to local points.
IDENTITY_BYTE = 4


def rotation_from_byte(b: int) -> np.ndarray:
    """The (3, 3) int64 apply matrix R for a .vox _r byte."""
    b = int(b)
    if not 0 <= b < 128:
        raise ValueError(f"invalid rotation byte {b}")
    c0 = b & 3
    c1 = (b >> 2) & 3
    if c0 == 3 or c1 == 3 or c0 == c1:
        raise ValueError(f"invalid rotation byte {b}")
    c2 = 3 - c0 - c1
    m = np.zeros((3, 3), dtype=np.int64)
    for row, col in enumerate((c0, c1, c2)):
        m[row, col] = -1 if b & (1 << (4 + row)) else 1
    return m.T.copy()


def rotation_to_byte(R: np.ndarray) -> int:
    """The _r byte for an apply matrix R. ValueError when R is not a signed permutation."""
    m = np.asarray(R).T
    if m.shape != (3, 3) or not np.array_equal(np.abs(m).sum(axis=0), [1, 1, 1]):
        raise ValueError("R is not a signed permutation matrix")
    b = 0
    for row in range(3):
        cols = np.nonzero(m[row])[0]
        if cols.size != 1 or abs(int(m[row, cols[0]])) != 1:
            raise ValueError("R is not a signed permutation matrix")
        col = int(cols[0])
        if row < 2:
            b |= col << (2 * row)
        if m[row, col] < 0:
            b |= 1 << (4 + row)
    return b


def apply_rotation(voxels: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bake a signed permutation R into the array.

    A voxel at local cell c lands at R c + s. Returns (contiguous uint8 copy, s).
    """
    R = np.asarray(R, dtype=np.int64)
    perm = [int(np.nonzero(R[i])[0][0]) for i in range(3)]
    out = np.transpose(voxels, perm)
    s = np.zeros(3, dtype=np.int64)
    for i in range(3):
        if R[i, perm[i]] < 0:
            out = np.flip(out, i)
            s[i] = out.shape[i] - 1
    return np.array(out, dtype=np.uint8, order="C"), s


# One quarter turn about each axis, counter-clockwise seen from the positive end.
_QUARTER = {
    0: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.int64),
    1: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.int64),
    2: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64),
}


def rotate_model(voxels, position, axis: int, k: int, pivot2) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Turn a whole model k quarter turns about `pivot2` (doubled world cells).

    Every world cell centre maps by the turn about pivot2 / 2 when the pivot has
    the right parity (see gizmo.snap_pivot).
    """
    if axis not in _QUARTER:
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    k = int(k) % 4
    out = np.array(voxels, dtype=np.uint8, order="C")
    for _ in range(k):
        out = rotate90(out, axis)
    R = np.linalg.matrix_power(_QUARTER[axis], k)
    pos = np.asarray(position, dtype=np.int64)
    size = np.asarray(voxels.shape, dtype=np.int64)
    piv = np.asarray(pivot2, dtype=np.int64)
    lo2 = 2 * pos + 1
    hi2 = 2 * (pos + size - 1) + 1
    lo2 = R @ (lo2 - piv) + piv
    hi2 = R @ (hi2 - piv) + piv
    new_pos = (np.minimum(lo2, hi2) - 1) // 2
    return out, tuple(int(v) for v in new_pos)


def flip_model(voxels, position, axis: int, pivot2) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Mirror a whole model across the plane pivot2[axis] / 2."""
    out = flip(voxels, axis)
    pos = [int(v) for v in position]
    pos[axis] = int(pivot2[axis]) - pos[axis] - int(voxels.shape[axis])
    return out, (pos[0], pos[1], pos[2])


def fill_all(voxels: np.ndarray, colour: int) -> np.ndarray:
    """Every filled voxel takes `colour`; empty cells stay empty. Returns a uint8 copy."""
    out = np.array(voxels, dtype=np.uint8, copy=True)
    out[voxels != 0] = np.uint8(colour)
    return out


def fill_volume(shape, colour: int) -> np.ndarray:
    """A new grid of `shape` where every cell is `colour`."""
    size = tuple(int(v) for v in shape)
    if len(size) != 3:
        raise ValueError(f"shape must have 3 dimensions, got {shape!r}")
    return np.full(size, np.uint8(colour), dtype=np.uint8)


def inverse(voxels: np.ndarray, colour: int) -> np.ndarray:
    """Swap filled and empty: empty cells take `colour`, filled cells become empty."""
    out = np.zeros(voxels.shape, dtype=np.uint8)
    out[voxels == 0] = np.uint8(colour)
    return out


def loop(voxels: np.ndarray, axis: int, offset: int) -> np.ndarray:
    """Shift the voxels along `axis` by `offset`, wrapping around at the edge."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    return np.array(np.roll(voxels, int(offset), axis=axis), dtype=np.uint8, order="C")


def fit_bounds(voxels: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """The (lo, hi) bounding box of the filled voxels, both inclusive. None when the grid is empty."""
    filled = voxels != 0
    lo, hi = [], []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        hit = np.flatnonzero(filled.any(axis=other))
        if hit.size == 0:
            return None
        lo.append(int(hit[0]))
        hi.append(int(hit[-1]))
    return tuple(lo), tuple(hi)


def scale_up(voxels: np.ndarray) -> np.ndarray:
    """Repeat every voxel into a 2 x 2 x 2 block, doubling the size on every axis.

    Raises ValueError when the doubled size would go past MAX_SIZE.
    """
    size = tuple(2 * d for d in voxels.shape)
    if any(d > MAX_SIZE for d in size):
        raise ValueError(f"the doubled size {size} would exceed {MAX_SIZE}")
    out = np.repeat(voxels, 2, axis=0)
    out = np.repeat(out, 2, axis=1)
    out = np.repeat(out, 2, axis=2)
    return np.array(out, dtype=np.uint8, order="C")


def _resample_axis(voxels: np.ndarray, axis: int, new: int) -> np.ndarray:
    """One axis resampled to `new` cells. Longer: each new cell takes the old
    cell under its middle. Shorter: each new cell covers a run of old cells;
    it is filled when any of them is, with the colour of the filled one
    nearest its middle. So a thin shell squeezed at a slant keeps no holes."""
    old = voxels.shape[axis]
    middle = np.minimum(((np.arange(new) + 0.5) * old / new).astype(np.int64), old - 1)
    out = np.take(voxels, middle, axis=axis)
    if new >= old:
        return out
    start = (np.arange(new) * old) // new
    stop = np.maximum(start + 1, ((np.arange(new) + 1) * old + new - 1) // new)  # ceil
    reach = int((stop - start).max())
    for step in range(1, reach):
        for offset in (-step, step):
            index = middle + offset
            inside = (index >= start) & (index < stop)
            if not inside.any():
                continue
            near = np.take(voxels, np.clip(index, 0, old - 1), axis=axis)
            shape = [1, 1, 1]
            shape[axis] = new
            take = (out == 0) & (near != 0) & inside.reshape(shape)
            out[take] = near[take]
    return out


def resample(voxels: np.ndarray, target) -> np.ndarray:
    """The voxels resampled to `target` cells per axis (see _resample_axis):
    nearest neighbour when growing, no lost cells when shrinking."""
    out = voxels
    for axis in range(3):
        new = int(target[axis])
        if out.shape[axis] != new:
            out = _resample_axis(out, axis, new)
    return np.ascontiguousarray(out)


def scale_down(voxels: np.ndarray) -> np.ndarray:
    """Halve the size on every axis (rounded up). Each new cell covers a 2 x 2 x 2
    block and is filled when any of it is, so a thin shell keeps no holes; a
    model scaled up by two comes back exactly."""
    return np.array(resample(voxels, [(d + 1) // 2 for d in voxels.shape]), dtype=np.uint8, order="C")


def repeat(voxels: np.ndarray, counts) -> np.ndarray:
    """Tile the model `counts[i]` times along each axis.

    Each axis is capped so the result never exceeds MAX_SIZE: a count that would go
    past it is brought down to the largest whole number of copies that still fits.
    """
    counts = tuple(int(c) for c in counts)
    if len(counts) != 3:
        raise ValueError(f"counts must have 3 dimensions, got {counts!r}")
    if any(c < 1 for c in counts):
        raise ValueError(f"counts must be at least 1, got {counts}")
    reps = tuple(max(1, min(counts[i], MAX_SIZE // voxels.shape[i])) for i in range(3))
    out = np.tile(voxels, reps)
    return np.array(out, dtype=np.uint8, order="C")


def drop_to_floor(voxels: np.ndarray) -> tuple[np.ndarray, int]:
    """The voxels moved straight down, all together, until the lowest filled
    one sits on the grid's floor (z = 0). Returns (voxels, how far they fell);
    an empty model falls 0."""
    filled_z = np.flatnonzero(np.asarray(voxels).any(axis=(0, 1)))
    if filled_z.size == 0 or int(filled_z[0]) == 0:
        return np.array(voxels, dtype=np.uint8, order="C"), 0
    fall = int(filled_z[0])
    out = np.zeros_like(voxels, dtype=np.uint8)
    out[:, :, : voxels.shape[2] - fall] = voxels[:, :, fall:]
    return np.ascontiguousarray(out), fall


def fall_distance(voxels: np.ndarray, mask: np.ndarray) -> int:
    """How far the cells of `mask` can fall straight down as one piece: until
    one of them would enter a filled cell outside the mask or go below the
    grid's floor (z = 0). 0 when one already rests; 0 for an empty mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    blocker = (np.asarray(voxels) != 0) & ~mask
    sz = mask.shape[2]
    z = np.arange(sz, dtype=np.int64)
    # For each cell, the z of the highest blocker strictly below it in its column (-1: the floor).
    marks = np.where(blocker, z[None, None, :], -1)
    below = np.maximum.accumulate(marks, axis=2)
    below = np.concatenate([np.full(mask.shape[:2] + (1,), -1, dtype=np.int64), below[:, :, :-1]], axis=2)
    gaps = (z[None, None, :] - below - 1)[mask]
    return int(gaps.min())


def settle_moves(voxels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where every filled cell of `mask` ends up when each falls straight down on
    its own: until it rests on the floor (z = 0), on a filled cell outside the
    mask, or on a mask cell that has already settled. Returns (x, y, z_from,
    z_to), one entry per voxel. Each column is worked from the bottom up, so a
    pile of selected voxels in one column lands as a pile in the same order: the
    lowest first, the others on top of it. Pure numpy, one pass per layer of z."""
    voxels = np.asarray(voxels)
    mask = np.asarray(mask, dtype=bool) & (voxels != 0)
    fixed = (voxels != 0) & ~mask
    # The next free height in each column: above the highest fixed cell or settled cell so far.
    height = np.zeros(mask.shape[:2], dtype=np.int64)
    xs_all, ys_all, z0_all, z1_all = [], [], [], []
    for z in range(mask.shape[2]):
        height = np.where(fixed[:, :, z], z + 1, height)  # a fixed cell here: the next one rests on it
        xs, ys = np.nonzero(mask[:, :, z])
        if not len(xs):
            continue
        xs_all.append(xs)
        ys_all.append(ys)
        z0_all.append(np.full(len(xs), z, dtype=np.int64))
        z1_all.append(height[xs, ys].copy())
        height[xs, ys] += 1
    if not xs_all:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    return (np.concatenate(xs_all), np.concatenate(ys_all), np.concatenate(z0_all), np.concatenate(z1_all))


def settle_cells(voxels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The voxels after `settle_moves` (the cells keep their colours), and the mask
    where the moved cells came to rest."""
    voxels = np.asarray(voxels)
    xs, ys, z0, z1 = settle_moves(voxels, mask)
    out = np.array(voxels, dtype=np.uint8, order="C")
    colours = voxels[xs, ys, z0]
    out[xs, ys, z0] = 0
    out[xs, ys, z1] = colours
    new_mask = np.zeros(voxels.shape, dtype=bool)
    new_mask[xs, ys, z1] = True
    return out, new_mask


def diagonal(voxels: np.ndarray) -> np.ndarray:
    """Mirror across the x = y diagonal: swap the x and y axes. The grid size swaps too."""
    return np.array(np.transpose(voxels, (1, 0, 2)), dtype=np.uint8, order="C")


def mirror_half(voxels: np.ndarray, axis: int, from_positive: bool) -> np.ndarray:
    """Copy one half of the model onto the other, mirrored. The middle layer of an odd size stays."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    out = np.array(voxels, dtype=np.uint8, order="C")
    n = out.shape[axis]
    k = n // 2
    low = [slice(None)] * 3
    high = [slice(None)] * 3
    low[axis] = slice(0, k)
    high[axis] = slice(n - k, n)
    low, high = tuple(low), tuple(high)
    if from_positive:
        out[low] = np.flip(out[high], axis)
    else:
        out[high] = np.flip(out[low], axis)
    return out
