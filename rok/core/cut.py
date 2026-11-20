"""Cutting whole models in two in the scene.

The Cut tool aims a plane where the mouse meets a model. The plane stands
across the face under the mouse: its axis is one of the two axes that lie in
that face. By default it is the one that runs more across the screen, so the
plane stands upright on screen, like a knife slicing bread; Tab picks the
other. The plane sits on the voxel boundary nearest the mouse.
"""

from __future__ import annotations

import numpy as np


def face_axes(normal) -> tuple[int, int]:
    """The two axes that lie in a face with this axis-aligned normal."""
    n = next(i for i in range(3) if normal[i] != 0)
    return tuple(i for i in range(3) if i != n)


def pick_axis(normal, screen_dirs, other: bool = False) -> int:
    """The axis of the cutting plane for a face: of the face's two axes, the
    one whose screen direction is more horizontal (the plane then stands
    upright on screen). `screen_dirs[axis]` is that axis's direction on screen;
    `other` picks the second axis instead."""
    a, b = face_axes(normal)

    def across(axis) -> float:
        dx, dy = screen_dirs[axis]
        length = float(np.hypot(dx, dy))
        return abs(dx) / length if length > 1e-9 else 0.0

    first, second = (a, b) if across(a) >= across(b) else (b, a)
    return second if other else first


def boundary(point, axis: int, size) -> int | None:
    """The voxel boundary along `axis` nearest a point in the model's cells, or
    None when it is at the model's edge (a cut there would leave one side
    empty)."""
    k = int(round(float(point[axis])))
    if k <= 0 or k >= int(size[axis]):
        return None
    return k


def split_at(voxels: np.ndarray, axis: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The voxels before and from boundary `k` along `axis`, each in a box of
    its own: the first keeps the model's place, the second starts `k` cells on."""
    index = [slice(None)] * 3
    index[axis] = slice(0, k)
    low = np.ascontiguousarray(voxels[tuple(index)])
    index[axis] = slice(k, None)
    high = np.ascontiguousarray(voxels[tuple(index)])
    return low, high


def fitted(voxels: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The voxels in the smallest box that holds them, and that box's offset
    in the old one. An empty grid comes back as it is."""
    filled = np.argwhere(voxels != 0)
    if len(filled) == 0:
        return voxels, (0, 0, 0)
    lo = filled.min(axis=0)
    hi = filled.max(axis=0) + 1
    box = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    return np.ascontiguousarray(voxels[box]), (int(lo[0]), int(lo[1]), int(lo[2]))


def plane_corners(position, size, axis: int, k: int) -> tuple[tuple[float, float, float], ...]:
    """The four corners of the cutting plane across the model's box, in the
    scene's cells, in order around the rectangle."""
    a, b = [i for i in range(3) if i != axis]
    corners = []
    for ua, ub in ((0, 0), (1, 0), (1, 1), (0, 1)):
        c = [0.0, 0.0, 0.0]
        c[axis] = position[axis] + k
        c[a] = position[a] + ua * size[a]
        c[b] = position[b] + ub * size[b]
        corners.append((c[0], c[1], c[2]))
    return tuple(corners)
