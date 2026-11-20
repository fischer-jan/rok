"""Turning a model by any angle, in tenths of a degree.

A voxel model has no angle of its own, so a turn other than a multiple of 90
degrees is baked into the voxels: every cell of the new grid takes the colour
of the nearest cell of the old one, looked up through the inverse turn about the
model's centre. The new grid is as big as the turned box needs. Because the
lookup runs backwards there are no holes, and a colour is never invented. Each
turn resamples, so a caller that lets the angle change (the modal tool) always
turns from the original voxels and not from the last preview.

A multiple of 90 degrees is not resampled: it goes through the exact quarter
turn of grid_ops, so 90 degrees loses nothing.
"""

from __future__ import annotations

import math

import numpy as np

from . import grid_ops

STEP = 0.1  # degrees: the finest angle the tool sets
MAX_SIZE = grid_ops.MAX_SIZE


def snap(degrees: float) -> float:
    """`degrees` rounded to tenths and wrapped into (-180, 180]."""
    d = round(float(degrees) / STEP) * STEP
    d = (d + 180.0) % 360.0 - 180.0
    if d <= -180.0:
        d += 360.0
    return round(d, 1)


def quarter_turns(degrees: float) -> int | None:
    """The number of quarter turns when `degrees` is an exact multiple of 90 (to a tenth), else None."""
    d = snap(degrees)
    q = d / 90.0
    return int(round(q)) % 4 if abs(q - round(q)) < 1e-9 else None


def matrix(axis: int, degrees: float) -> np.ndarray:
    """The right-handed 3x3 turn about `axis` (0 X, 1 Y, 2 Z), counter-clockwise
    seen from the positive end, as grid_ops' quarter turns are."""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 2:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"axis must be 0, 1 or 2, got {axis}")


def turned_size(size, axis: int, degrees: float) -> tuple[int, int, int]:
    """The size of the grid that holds `size` turned by `degrees` about `axis`."""
    q = quarter_turns(degrees)
    out = [int(v) for v in size]
    a, b = (axis + 1) % 3, (axis + 2) % 3
    if q is not None:
        if q % 2:
            out[a], out[b] = out[b], out[a]
        return out[0], out[1], out[2]
    c, s = abs(math.cos(math.radians(degrees))), abs(math.sin(math.radians(degrees)))
    A, B = size[a], size[b]
    out[a] = max(1, math.ceil(c * A + s * B - 1e-6))
    out[b] = max(1, math.ceil(s * A + c * B - 1e-6))
    return out[0], out[1], out[2]


def rotate_voxels(voxels: np.ndarray, axis: int, degrees: float, limit: int | None = MAX_SIZE) -> np.ndarray:
    """`voxels` turned by `degrees` about `axis` through the grid's centre.
    Raises ValueError when the turned box is larger than `limit` per side
    (MAX_SIZE, a grid's most; None for no limit: a selection's turn is
    clipped to its grid afterwards, so its box may be larger)."""
    voxels = np.asarray(voxels, dtype=np.uint8)
    q = quarter_turns(degrees)
    if q is not None:
        out = voxels
        for _ in range(q):
            out = grid_ops.rotate90(out, axis)
        return np.array(out, dtype=np.uint8, order="C")
    new = turned_size(voxels.shape, axis, degrees)
    if limit is not None and max(new) > limit:
        raise ValueError(f"turned, the model would be {new[0]} x {new[1]} x {new[2]}: at most {limit} per side")
    native = _native()
    if native is not None:
        # The same nearest-cell lookup, in Rust and in parallel (bake.rs; its
        # round-ties-even is numpy's rint).
        return np.ascontiguousarray(native.bake_rotate(np.ascontiguousarray(voxels), int(axis), float(snap(degrees))))
    return _rotate_voxels_numpy(voxels, axis, degrees, new)


def _native():
    """rok_native's resampler, or None (no native module, or ROK_PURE_ROTATE set)."""
    if not _NATIVE:
        import os

        mod = None
        if not os.environ.get("ROK_PURE_ROTATE"):
            try:
                import rok_native

                mod = rok_native if hasattr(rok_native, "bake_rotate") else None
            except ImportError:
                mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


_NATIVE: list = []


def _rotate_voxels_numpy(voxels: np.ndarray, axis: int, degrees: float, new) -> np.ndarray:
    """The free turn in numpy: every new cell takes the nearest old cell."""
    a, b = (axis + 1) % 3, (axis + 2) % 3
    plane = np.moveaxis(voxels, (a, b), (0, 1))
    A, B = plane.shape[0], plane.shape[1]
    A2, B2 = new[a], new[b]
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    ii, jj = np.meshgrid(np.arange(A2) - (A2 - 1) / 2.0, np.arange(B2) - (B2 - 1) / 2.0, indexing="ij")
    # The inverse turn takes a new cell's centre back to the old grid.
    si = np.rint(c * ii + s * jj + (A - 1) / 2.0).astype(np.int64)
    sj = np.rint(-s * ii + c * jj + (B - 1) / 2.0).astype(np.int64)
    ok = (si >= 0) & (si < A) & (sj >= 0) & (sj < B)
    out = np.zeros((A2, B2) + plane.shape[2:], dtype=np.uint8)
    out[ok] = plane[si[ok], sj[ok]]
    return np.ascontiguousarray(np.moveaxis(out, (0, 1), (a, b)))


def group_pivot(items, pivot2=None) -> np.ndarray:
    """The point the models of `items` turn about, in grid cells with cell
    (i, j, k) centred at (i, j, k): the middle of the box around them all, or
    `pivot2` (doubled world cells)."""
    if pivot2 is not None:
        return np.asarray(pivot2, dtype=np.float64) / 2.0 - 0.5
    pos = [np.asarray(p, dtype=np.float64) for _v, p in items]
    lo = np.min(pos, axis=0)
    hi = np.max([p + np.asarray(v.shape) for (v, _p), p in zip(items, pos)], axis=0)
    return (lo + hi - 1) / 2.0


def preview_matrices(items, axis: int, degrees: float, pivot2=None) -> list[np.ndarray]:
    """The live preview of `rotate_models`, with no voxel touched: one 4x4 per
    model, in mesh units relative to the model's rest place (cell (0, 0, 0) at
    the origin, cell (i, j, k) filling [i, i + 1)), that turns the model's
    meshes as `rotate_models` would turn its voxels. A model Empty at the
    model position takes `Translation(position) @ matrix`."""
    if not items:
        return []
    pivot = group_pivot(items, pivot2) + 0.5  # cell centres are at + 0.5 in mesh units
    R = matrix(axis, snap(degrees))
    about_pivot = len(items) > 1 or pivot2 is not None
    out = []
    for voxels, position in items:
        centre = pivot - np.asarray(position, dtype=np.float64) if about_pivot else np.asarray(voxels.shape) / 2.0
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = centre - R @ centre
        out.append(m)
    return out


def turned_place(shape, position, axis: int, degrees: float, pivot2) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Where one box of `shape` at `position` lands when it turns about
    `pivot2` (doubled world cells): (the turned box's shape, its position),
    as rotate_models places it."""
    size = np.asarray(shape, dtype=np.float64)
    p = np.asarray(position, dtype=np.float64)
    pivot = group_pivot([], pivot2)
    R = matrix(axis, snap(degrees))
    centre = p + (size - 1) / 2.0
    centre = R @ (centre - pivot) + pivot
    new = turned_size(shape, axis, snap(degrees))
    new_pos = np.floor(centre - (np.asarray(new) - 1) / 2.0 + 0.5).astype(np.int64)
    return tuple(int(v) for v in new), tuple(int(v) for v in new_pos)


def rotate_models(items, axis: int, degrees: float, pivot2=None, limit: int | None = MAX_SIZE):
    """Turn several models as one piece.

    `items` is a list of (voxels, position) with the position of grid cell
    (0, 0, 0) in world cells. Each model's voxels turn about its own centre; its
    centre then moves about the group's centre (`pivot2`, doubled world cells;
    default: the middle of the box around them all). One model turns in place,
    unless `pivot2` is given (the gizmo's pivot). Returns a list of
    (voxels, position) in the same order. Raises ValueError when a turned model
    would be larger than `limit` per side (MAX_SIZE; None for no limit).
    """
    if not items:
        return []
    pos = [np.asarray(p, dtype=np.float64) for _v, p in items]
    pivot = group_pivot(items, pivot2)
    R = matrix(axis, snap(degrees))
    about_pivot = len(items) > 1 or pivot2 is not None  # one model with no pivot turns in place
    out = []
    for (voxels, _p), p in zip(items, pos):
        size = np.asarray(voxels.shape, dtype=np.float64)
        centre = p + (size - 1) / 2.0
        centre = R @ (centre - pivot) + pivot if about_pivot else centre
        turned = rotate_voxels(voxels, axis, snap(degrees), limit)
        new_pos = np.floor(centre - (np.asarray(turned.shape) - 1) / 2.0 + 0.5).astype(np.int64)
        out.append((turned, tuple(int(v) for v in new_pos)))
    return out
