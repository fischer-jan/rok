"""Voxel look: a posed part as real voxels.

Each frame, a part with a turn or a size change is turned from its rest
voxels into a new grid: every cell of the new grid takes the colour of the
nearest rest cell, found through the inverse pose. It is never turned from
the previous frame, so nothing wears down. Moves are whole cells, so edges
do not shimmer. A quarter turn without a size change goes through the exact
quarter turn and loses nothing.

`posed` returns the new voxels and where their cell (0, 0, 0) sits,
relative to the part's rest place. numpy only.
"""

from __future__ import annotations

import numpy as np

from . import grid_ops

MAX_SIZE = grid_ops.MAX_SIZE


def _quarter_matrix(linear: np.ndarray) -> np.ndarray | None:
    """The signed permutation `linear` is, or None."""
    r = np.rint(linear)
    if not np.allclose(linear, r, atol=1e-6):
        return None
    r = r.astype(np.int64)
    if sorted(np.abs(r).sum(axis=0)) != [1, 1, 1] or sorted(np.abs(r).sum(axis=1)) != [1, 1, 1]:
        return None
    return r if int(round(np.linalg.det(r))) == 1 else None


def posed(voxels: np.ndarray, linear: np.ndarray, pivot, offset) -> tuple[np.ndarray, tuple[int, int, int]]:
    """`voxels` (the rest grid) turned and sized by `linear` (3 x 3) about
    `pivot` (grid units: cell (i, j, k) fills [i, i + 1)), then moved by
    `offset` cells, rounded to whole cells. Returns (new voxels, the grid
    cell where the new cell (0, 0, 0) sits, relative to the rest place).
    Raises ValueError when the result is larger than MAX_SIZE on a side."""
    voxels = np.asarray(voxels, dtype=np.uint8)
    linear = np.asarray(linear, dtype=np.float64)
    pivot = np.asarray(pivot, dtype=np.float64)
    move = np.rint(np.asarray(offset, dtype=np.float64)).astype(np.int64)
    if np.allclose(linear, np.eye(3)):
        return voxels, tuple(int(v) for v in move)
    # Only the filled box turns: the empty rest of the grid costs nothing.
    bounds = grid_ops.fit_bounds(voxels)
    if bounds is None:
        return np.zeros((1, 1, 1), np.uint8), tuple(int(v) for v in move)
    (x0, y0, z0), (x1, y1, z1) = bounds
    crop = np.array((x0, y0, z0), dtype=np.int64)
    if tuple(crop) != (0, 0, 0) or (x1 + 1, y1 + 1, z1 + 1) != voxels.shape:
        # The crop, posed about the same point; its places count from the crop's corner.
        inner, origin = posed(voxels[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1], linear, pivot - crop, offset)
        return inner, tuple(int(o + c) for o, c in zip(origin, crop))
    size = np.array(voxels.shape, dtype=np.float64)
    # The box the posed grid needs: the turned corners, out to whole cells.
    corners = np.array([[x, y, z] for x in (0, size[0]) for y in (0, size[1]) for z in (0, size[2])])
    turned = (corners - pivot) @ linear.T + pivot
    lo = np.floor(turned.min(axis=0) + 1e-6).astype(np.int64)
    hi = np.ceil(turned.max(axis=0) - 1e-6).astype(np.int64)
    shape = np.maximum(hi - lo, 1)
    if shape.max() > MAX_SIZE:
        raise ValueError(f"posed, the part would be {shape[0]} x {shape[1]} x {shape[2]}: at most {MAX_SIZE}")
    quarter = _quarter_matrix(linear)
    if quarter is not None:
        # Exact: the turned grid itself, placed on whole cells. A pivot on a half
        # cell (a part's middle, the .vox file's own frames) puts the turned box
        # half a cell off; it snaps rather than resample, which doubled voxels.
        out, _shift = grid_ops.apply_rotation(voxels, quarter)
        snapped = np.rint(turned.min(axis=0) + 1e-6).astype(np.int64)
        return np.ascontiguousarray(out, dtype=np.uint8), tuple(int(v) for v in snapped + move)
    # Each new cell's centre, back through the inverse pose to a rest cell.
    inv = np.linalg.inv(linear)
    ii, jj, kk = np.meshgrid(*(np.arange(n) + 0.5 for n in shape), indexing="ij")
    centres = np.stack([ii, jj, kk], axis=-1) + lo - pivot
    src = centres @ inv.T + pivot
    idx = np.floor(src).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array(voxels.shape)), axis=-1)
    out = np.zeros(tuple(int(v) for v in shape), dtype=np.uint8)
    out[ok] = voxels[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out, tuple(int(v) for v in lo + move)
