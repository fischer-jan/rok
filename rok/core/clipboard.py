"""The voxel clipboard: copy a selection out of a grid, paste it into any grid.

A Clip carries its own palette and materials, so a paste into another file can
fit its colours in through `palette.merge_claimed`. numpy and stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edit import Cell, SetMany
from .material import MaterialTable
from .pattern import box_cells
from .select import Selection


@dataclass(frozen=True)
class Clip:
    """The copied voxels. `voxels` is the selection's bounding box; 0 = not copied."""

    voxels: np.ndarray  # uint8 (sx, sy, sz)
    origin: Cell  # where the box sat in the source grid
    palette: np.ndarray  # (256, 4) uint8, a copy taken at copy time
    materials: MaterialTable

    @property
    def size(self) -> Cell:
        return (int(self.voxels.shape[0]), int(self.voxels.shape[1]), int(self.voxels.shape[2]))

    def count(self) -> int:
        return int(np.count_nonzero(self.voxels))

    def used_counts(self) -> np.ndarray:
        """(256,) int64 voxel count per palette index, index 0 zero."""
        counts = np.bincount(self.voxels.ravel(), minlength=256).astype(np.int64)
        counts[0] = 0
        return counts


def copy_selection(
    voxels: np.ndarray, selection: Selection, palette: np.ndarray, materials: MaterialTable
) -> Clip | None:
    """The selected voxels of `voxels` as a Clip, or None when none is filled."""
    bounds = selection.bounds()
    if bounds is None:
        return None
    lo, hi = bounds
    box = tuple(slice(lo[i], hi[i] + 1) for i in range(3))
    part = np.where(selection.mask[box], voxels[box], 0).astype(np.uint8)
    if not part.any():
        return None
    return Clip(
        part,
        (int(lo[0]), int(lo[1]), int(lo[2])),
        np.array(palette, dtype=np.uint8, copy=True).reshape(256, 4),
        materials.copy(),
    )


def paste_origin(clip_size: Cell, origin: Cell, grid_size: Cell) -> Cell:
    """Where the clip box goes: at `origin` when it fits there, else moved inside.

    A box larger than the grid on an axis starts at 0 there and is cut off.
    """
    out = []
    for i in range(3):
        room = int(grid_size[i]) - int(clip_size[i])
        out.append(min(max(int(origin[i]), 0), max(room, 0)))
    return (out[0], out[1], out[2])


def paste_cells(
    clip: Clip,
    origin: Cell,
    grid_size: Cell,
    remap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(cells (N, 3), values (N,)) of the clip placed at `origin`: `pattern.box_cells`."""
    return box_cells(clip.voxels, origin, grid_size, remap)


def paste(
    voxels: np.ndarray,
    clip: Clip,
    origin: Cell,
    set_many: SetMany,
    remap: np.ndarray | None = None,
) -> np.ndarray:
    """Write the clip at `origin`. Returns the bool mask of the pasted cells.

    Empty cells of the clip leave the grid alone; filled ones overwrite.
    """
    cells, values = paste_cells(clip, origin, voxels.shape, remap)
    mask = np.zeros(voxels.shape, dtype=bool)
    if len(cells) == 0:
        return mask
    set_many(cells, values)
    mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return mask
