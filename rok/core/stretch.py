"""Stretching whole models in the scene: the handles on the picked model's
faces, and what a drag on one does.

A handle sits in the middle of a face. Dragging it moves that face along its
axis; the opposite face stays where it is. By default the voxels stretch or
squeeze to fill the new length (nearest neighbour,
but a squeeze keeps every filled cell it covers, so thin shells stay closed);
with `crop` the box grows with empty cells or loses the cells past the face.
With `uniform` all three axes change by the same factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_ops import MAX_SIZE, MIN_SIZE, resample

HANDLE_RADIUS_PT = 7.0
HANDLE_HIT_PT = 12.0


@dataclass(frozen=True)
class Handle:
    axis: int  # 0, 1, 2
    side: int  # +1: the face at the box's high end, -1: at its low end

    @property
    def id(self) -> str:
        return f"SH_{'XYZ'[self.axis]}{'+' if self.side > 0 else '-'}"


HANDLES = tuple(Handle(a, s) for a in range(3) for s in (1, -1))


def handle_from_id(ident: str | None) -> Handle | None:
    for handle in HANDLES:
        if handle.id == ident:
            return handle
    return None


def face_centre(position, size, handle: Handle) -> tuple[float, float, float]:
    """The middle of the handle's face, in the scene's cells (a box spans
    [position, position + size))."""
    centre = [position[i] + size[i] / 2.0 for i in range(3)]
    centre[handle.axis] = position[handle.axis] + (size[handle.axis] if handle.side > 0 else 0)
    return (centre[0], centre[1], centre[2])


def facing_handles(view_dir) -> tuple[Handle, ...]:
    """The handles on the faces turned toward the view: their outward normal
    points against the view direction."""
    return tuple(h for h in HANDLES if h.side * view_dir[h.axis] < -1e-6)


def cells_along(start_px, now_px, axis_px) -> int:
    """How many cells a drag from `start_px` to `now_px` moves a face whose
    outward axis is `axis_px` on screen (pixels per cell, with direction)."""
    ax = np.asarray(axis_px, dtype=np.float64)
    length2 = float(ax @ ax)
    if length2 < 1e-6:
        return 0
    d = np.asarray(now_px, dtype=np.float64) - np.asarray(start_px, dtype=np.float64)
    return int(round(float(d @ ax) / length2))


def new_size(size, handle: Handle, cells: int, uniform: bool, limit: int = MAX_SIZE) -> tuple[int, int, int]:
    """The box after moving the handle's face out by `cells` (in: negative),
    kept inside MIN_SIZE..limit. With `uniform` every axis scales by the
    same factor as the dragged one. A group's box may pass MAX_SIZE: its
    models are held to it one by one (group_parts)."""
    size = [int(v) for v in size]
    length = min(limit, max(MIN_SIZE, size[handle.axis] + int(cells)))
    if not uniform:
        size[handle.axis] = length
        return (size[0], size[1], size[2])
    factor = length / size[handle.axis]
    return tuple(min(limit, max(MIN_SIZE, int(round(v * factor)))) for v in size)


def group_parts(parts, box_lo, box_size, target, new_lo) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Where each model of a stretched group goes and how big it gets.

    `parts` are (position, size) boxes inside the group's box (low corner
    `box_lo`, `box_size` cells), which becomes `target` cells from `new_lo`.
    Each face maps by the same rounding, so models that touched still touch
    and models that lined up still line up. No model gets thinner than one
    cell or bigger than MAX_SIZE."""
    out = []
    for position, size in parts:
        pos, new = [0, 0, 0], [0, 0, 0]
        for i in range(3):
            factor = int(target[i]) / int(box_size[i])
            start = int(round((int(position[i]) - int(box_lo[i])) * factor))
            end = int(round((int(position[i]) + int(size[i]) - int(box_lo[i])) * factor))
            new[i] = min(MAX_SIZE, max(MIN_SIZE, end - start))
            pos[i] = int(new_lo[i]) + start
        out.append(((pos[0], pos[1], pos[2]), (new[0], new[1], new[2])))
    return out


def new_position(position, size, target, handle: Handle, uniform: bool) -> tuple[int, int, int]:
    """Where the box starts after the drag, so the face opposite the handle stays
    put. With `uniform` the other two axes keep their centre."""
    out = [int(v) for v in position]
    for axis in range(3):
        grow = int(target[axis]) - int(size[axis])
        if axis == handle.axis:
            if handle.side < 0:
                out[axis] -= grow
        elif uniform:
            out[axis] -= grow // 2
    return (out[0], out[1], out[2])


def stretch(voxels: np.ndarray, target) -> np.ndarray:
    """The voxels resampled to `target` cells per axis (grid_ops.resample):
    nearest neighbour when growing, no lost cells when shrinking."""
    return resample(voxels, target)


def crop_or_grow(voxels: np.ndarray, target, handle: Handle, uniform: bool) -> np.ndarray:
    """The voxels in a box of `target` cells: cells past a moved face are cut
    off, new cells are empty. The dragged face moves; with `uniform` the other
    axes grow or shrink evenly on both sides."""
    out = voxels
    for axis in range(3):
        old, new = out.shape[axis], int(target[axis])
        if old == new:
            continue
        if axis == handle.axis:
            lead = (new - old) if handle.side < 0 else 0
        else:
            lead = (new - old) // 2
        # Copy the overlap of [0, old) shifted by `lead` into [0, new).
        result_shape = list(out.shape)
        result_shape[axis] = new
        result = np.zeros(result_shape, dtype=out.dtype)
        src0, dst0 = max(0, -lead), max(0, lead)
        count = min(old - src0, new - dst0)
        if count > 0:
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            src[axis] = slice(src0, src0 + count)
            dst[axis] = slice(dst0, dst0 + count)
            result[tuple(dst)] = out[tuple(src)]
        out = result
    return out
