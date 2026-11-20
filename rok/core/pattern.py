"""The pattern brush: a box of voxels (the clipboard) stamped on a surface.

Orientation, placement on a hit face, the cells of a placed box and the stamp
rules per tool. numpy and stdlib only.
"""

from __future__ import annotations

import numpy as np

from .edit import ATTACH, ERASE, PAINT, SELECT, Cell, SetMany, in_bounds_mask
from .grid_ops import _QUARTER, apply_rotation
from .raycast import Hit

# A rotation as a flat tuple of nine ints, row by row: hashable for caches.
Rotation = tuple[int, int, int, int, int, int, int, int, int]
IDENTITY: Rotation = (1, 0, 0, 0, 1, 0, 0, 0, 1)


def turned(rotation: Rotation, axis: int) -> Rotation:
    """One more quarter turn about `axis`, counter-clockwise from the positive end."""
    R = _QUARTER[int(axis)] @ np.asarray(rotation, dtype=np.int64).reshape(3, 3)
    return tuple(int(v) for v in R.ravel())


def oriented(box: np.ndarray, rotation: Rotation) -> np.ndarray:
    """The box turned by `rotation`, as a contiguous uint8 copy."""
    R = np.asarray(rotation, dtype=np.int64).reshape(3, 3)
    return apply_rotation(box, R)[0]


def anchor_cell(box: np.ndarray, axis: int, near_index: int) -> Cell:
    """The filled cell of the box's layer `near_index` along `axis` that lies
    closest to the middle of that layer. The bounding box is tight, so every
    outermost layer has one."""
    layer = np.take(box, near_index, axis=axis)
    filled = np.argwhere(layer != 0)
    if len(filled) == 0:  # not a tight box: fall back to the layer's middle
        u, v = layer.shape
        pick = ((u - 1) // 2, (v - 1) // 2)
    else:
        middle = (np.asarray(layer.shape, dtype=np.float64) - 1.0) / 2.0
        pick = filled[np.argmin(((filled - middle) ** 2).sum(axis=1))]
    cell = [int(pick[0]), int(pick[1])]
    cell.insert(axis, int(near_index))
    return (cell[0], cell[1], cell[2])


def placement(box: np.ndarray, hit: Hit, tool: str) -> Cell | None:
    """Where `box` goes for a hit: a filled cell of its layer nearest the face
    sits in the cell under the mouse, so the stamp is flush with the surface.

    Attach puts the box outside the face, its near layer in the target cell.
    Erase, Paint and Select put it inside, its near layer in the hit voxel,
    reaching away from the face. None for a wall hit with any tool but Attach.
    """
    if hit.wall and tool != ATTACH:
        return None
    axis = 0
    for i in range(3):
        if hit.normal[i] != 0:
            axis = i
            break
    outward = hit.normal[axis] > 0
    size = box.shape
    anchor = hit.target if tool == ATTACH else hit.voxel
    # The box lies along the normal from the anchor for Attach, against it
    # otherwise; the layer that touches the face is index 0 or the last one.
    along = outward if tool == ATTACH else not outward
    near = 0 if along else int(size[axis]) - 1
    ref = anchor_cell(box, axis, near)
    return (int(anchor[0]) - ref[0], int(anchor[1]) - ref[1], int(anchor[2]) - ref[2])


def box_cells(
    box: np.ndarray,
    origin: Cell,
    grid_size,
    remap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(cells (N, 3), values (N,)) of the filled cells of `box` placed at `origin`.

    Cells outside the grid are dropped. `remap` maps the box's palette indices
    to the grid's.
    """
    cells = np.argwhere(box != 0)
    values = box[cells[:, 0], cells[:, 1], cells[:, 2]]
    cells = cells + np.asarray(origin, dtype=np.int64)
    keep = in_bounds_mask(cells, grid_size)
    cells = cells[keep]
    values = values[keep]
    if remap is not None:
        values = np.asarray(remap, dtype=np.uint8)[values]
    return cells, values.astype(np.uint8)


def mirror_pairs(cells: np.ndarray, values: np.ndarray, size, axes) -> tuple[np.ndarray, np.ndarray]:
    """The cells and their mirror images, values carried along. A cell that is
    its own image, or the image of another, keeps the first value."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    for i in range(3):
        if not axes[i]:
            continue
        image = cells.copy()
        image[:, i] = int(size[i]) - 1 - cells[:, i]
        cells = np.concatenate([cells, image])
        values = np.concatenate([values, values])
    if len(cells) == 0:
        return cells, values
    _, first = np.unique(cells, axis=0, return_index=True)
    first.sort()
    return cells[first], values[first]


def empty_cells(box: np.ndarray, origin: Cell, grid_size) -> np.ndarray:
    """(N, 3) the empty cells of `box` placed at `origin`, inside the grid: the
    pattern's negative space, which a stamp with holes clears."""
    cells = np.argwhere(np.asarray(box) == 0) + np.asarray(origin, dtype=np.int64)
    return cells[in_bounds_mask(cells, grid_size)]


def stamp(
    voxels: np.ndarray,
    tool: str,
    cells: np.ndarray,
    values: np.ndarray,
    set_many: SetMany,
    mirror=(False, False, False),
) -> np.ndarray:
    """Apply one stamp. Returns the (M, 3) cells that were written.

    Attach writes the pattern into empty cells only. Erase clears the filled
    cells under the pattern. Paint gives the filled cells the pattern's colours.
    Select writes nothing (the gesture selects).
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if any(mirror):
        cells, values = mirror_pairs(cells, values, voxels.shape, mirror)
    if tool == SELECT or len(cells) == 0:
        return cells[:0]
    cur = voxels[cells[:, 0], cells[:, 1], cells[:, 2]]
    if tool == ATTACH:
        m = cur == 0
    elif tool == ERASE:
        m = cur != 0
        values = np.zeros(len(cells), dtype=np.uint8)
    elif tool == PAINT:
        m = (cur != 0) & (cur != values)
    else:
        return cells[:0]
    if not m.any():
        return cells[:0]
    set_many(cells[m], values[m])
    return cells[m]
