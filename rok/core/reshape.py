"""Whole-model reshaping tools: Lathe, Twist and Remove Slice.

Each takes the model's (X, Y, Z) uint8 array and returns a new one. They do
work that would otherwise need a shader (milling, screw, the knives and
slices).
"""

from __future__ import annotations

import numpy as np

AXES = "XYZ"


def _others(axis: int) -> tuple[int, int]:
    """The two axes across `axis`, in order."""
    return tuple(a for a in range(3) if a != axis)  # type: ignore[return-value]


def lathe(voxels: np.ndarray, axis: int = 2) -> np.ndarray:
    """Spin what is drawn around the grid's centre line along `axis` into a
    round solid, like a lathe. Each voxel stands for a ring: its distance from
    the centre line and its height along the axis. A cell of the result takes
    the colour of the ring it lies on, so a profile drawn on one side of the
    centre line becomes a vase, a column or a wheel. Where two voxels give the
    same ring, the one further from the start of the grid wins."""
    voxels = np.asarray(voxels, dtype=np.uint8)
    u, v = _others(axis)
    shape = voxels.shape
    cu, cv = shape[u] / 2.0, shape[v] / 2.0
    grid = np.indices(shape, dtype=np.float64)
    radius = np.floor(np.hypot(grid[u] + 0.5 - cu, grid[v] + 0.5 - cv)).astype(np.int64)
    height = np.indices(shape)[axis]
    rings = np.zeros((int(radius.max()) + 1, shape[axis]), dtype=np.uint8)
    filled = voxels > 0
    rings[radius[filled], height[filled]] = voxels[filled]
    return rings[radius, height]


def twist(voxels: np.ndarray, degrees: float, axis: int = 2) -> np.ndarray:
    """Turn each layer along `axis` a little more than the one before, about
    the grid's centre line: the first layer stays, the last turns by
    `degrees`. Each cell takes the voxel that turns onto it (the nearest
    one), so the grid keeps its size; voxels that turn out of it are lost."""
    voxels = np.asarray(voxels, dtype=np.uint8)
    u, v = _others(axis)
    shape = voxels.shape
    n = shape[axis]
    out = np.zeros_like(voxels)
    cu, cv = shape[u] / 2.0, shape[v] / 2.0
    gu, gv = np.meshgrid(np.arange(shape[u]) + 0.5 - cu, np.arange(shape[v]) + 0.5 - cv, indexing="ij")
    for k in range(n):
        angle = np.radians(degrees) * (k / (n - 1) if n > 1 else 0.0)
        c, s = np.cos(angle), np.sin(angle)
        # Where the cell comes from: turn it back.
        su = np.floor(c * gu + s * gv + cu).astype(np.int64)
        sv = np.floor(-s * gu + c * gv + cv).astype(np.int64)
        inside = (su >= 0) & (su < shape[u]) & (sv >= 0) & (sv < shape[v])
        layer_in = np.take(voxels, k, axis=axis)
        layer_out = np.zeros_like(layer_in)
        layer_out[inside] = layer_in[su[inside], sv[inside]]
        index = [slice(None)] * 3
        index[axis] = k
        out[tuple(index)] = layer_out
    return out


def slice_axis(mask: np.ndarray) -> tuple[int, int, int] | None:
    """(axis, first, last) of the layers a selection covers, along the axis
    where its box is thinnest (Z first when two are as thin): the layers to
    remove. None for an empty selection."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    spans = []
    for axis in (2, 0, 1):
        others = tuple(a for a in range(3) if a != axis)
        filled = np.flatnonzero(mask.any(axis=others))
        spans.append((int(filled[-1] - filled[0] + 1), axis, int(filled[0]), int(filled[-1])))
    _span, axis, first, last = min(spans, key=lambda s: s[0])
    return axis, first, last


def remove_slices(voxels: np.ndarray, axis: int, first: int, last: int) -> np.ndarray:
    """The model without the layers first..last along `axis`: the rest moves
    in to close the gap and the grid gets that much shorter. Raises
    ValueError when nothing would be left."""
    voxels = np.asarray(voxels, dtype=np.uint8)
    n = voxels.shape[axis]
    first, last = max(0, int(first)), min(n - 1, int(last))
    if first == 0 and last == n - 1:
        raise ValueError("that would remove every layer")
    return np.delete(voxels, np.arange(first, last + 1), axis=axis)
