"""Ray picking on the dense grid. Amanatides-Woo DDA with a slab pre-clip.

Grid space: voxel (x, y, z) fills [x, x+1) x [y, y+1) x [z, z+1). The DDA box is
[0, sx] x [0, sy] x [0, sz]. When the ray leaves the box without a hit, the exit
is reported as a wall hit on the cell just outside, so all six walls take voxels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

Cell = tuple[int, int, int]

_INF = math.inf
_EPS = 1e-6


@dataclass(frozen=True)
class Hit:
    """One ray hit on a solid cell or on a grid wall."""

    voxel: Cell  # the solid cell hit; for a wall hit the cell just outside the grid
    normal: Cell  # one of the six unit axis vectors, toward the ray origin
    t: float  # ray parameter of the face crossing: origin + t * direction
    wall: bool = False  # True when the ray left the box without a solid hit

    @property
    def target(self) -> Cell:
        """The cell in front of the hit face. For a wall hit the last in-grid cell."""
        return (
            self.voxel[0] + self.normal[0],
            self.voxel[1] + self.normal[1],
            self.voxel[2] + self.normal[2],
        )


def raycast(
    voxels: np.ndarray,
    origin,
    direction,
    *,
    walls: bool = True,
    max_steps: int = 1024,
    blocked=None,
    ignore=None,
) -> Hit | None:
    """First solid cell along the ray, else the exit wall, else None.

    `direction` need not be normalised. `blocked` is an optional container of
    cells that count as solid; `ignore` one of filled cells that count as empty
    (the voxels an Attach drag placed, so the ray keeps hitting the surface
    under them). Pure Python floats in the loop; numpy is used only for one
    array index per step.
    """
    sx, sy, sz = (int(d) for d in voxels.shape)
    lo = (0.0, 0.0, 0.0)
    hi = (float(sx), float(sy), float(sz))
    o = (float(origin[0]), float(origin[1]), float(origin[2]))
    d = (float(direction[0]), float(direction[1]), float(direction[2]))

    clip = _slab_clip(o, d, lo, hi)
    if clip is None:
        return None
    t_enter, enter_axis = clip

    cell = [0, 0, 0]
    if t_enter > 0.0:
        t_start = t_enter + _EPS
        for i in range(3):
            p = o[i] + t_start * d[i]
            cell[i] = _clamp(math.floor(p), int(lo[i]), int(hi[i]) - 1)
        if _solid(voxels, cell, blocked, ignore):
            normal = [0, 0, 0]
            normal[enter_axis] = -1 if d[enter_axis] > 0.0 else 1
            return Hit(tuple(cell), tuple(normal), t_enter)
    else:
        for i in range(3):
            cell[i] = _clamp(math.floor(o[i]), int(lo[i]), int(hi[i]) - 1)
        if _solid(voxels, cell, blocked, ignore):
            return None

    step = [0, 0, 0]
    t_delta = [_INF, _INF, _INF]
    t_max = [_INF, _INF, _INF]
    for i in range(3):
        if d[i] > 0.0:
            step[i] = 1
            t_delta[i] = 1.0 / d[i]
            t_max[i] = (cell[i] + 1 - o[i]) / d[i]
        elif d[i] < 0.0:
            step[i] = -1
            t_delta[i] = -1.0 / d[i]
            t_max[i] = (cell[i] - o[i]) / d[i]

    lo_cell = (0, 0, 0)
    hi_cell = (sx - 1, sy - 1, sz - 1)
    for _ in range(max_steps):
        axis = 0
        if t_max[1] < t_max[axis]:
            axis = 1
        if t_max[2] < t_max[axis]:
            axis = 2
        t = t_max[axis]
        if t == _INF:
            return None
        cell[axis] += step[axis]
        t_max[axis] += t_delta[axis]
        c = cell[axis]
        if c < lo_cell[axis] or c > hi_cell[axis]:
            if not walls:
                return None
            normal = [0, 0, 0]
            normal[axis] = -step[axis]
            return Hit(tuple(cell), tuple(normal), t, wall=True)
        if _solid(voxels, cell, blocked, ignore):
            normal = [0, 0, 0]
            normal[axis] = -step[axis]
            return Hit(tuple(cell), tuple(normal), t)
    return None


def ray_to_rest(
    origin, direction, matrix4
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """A ray cast at a posed part, mapped into the part's rest space.

    `matrix4` is the part's 4x4 pose matrix: it maps a point of the rest grid
    to where that point sits once posed. `origin` is a point, so it takes the
    full inverse; `direction` is a vector, so only the inverse's linear part
    applies (a translation must not move it). `direction` need not be
    normalised, and the result keeps whatever scale the caller passed.
    """
    inverse = np.linalg.inv(np.asarray(matrix4, dtype=np.float64))
    o = inverse @ np.array([origin[0], origin[1], origin[2], 1.0])
    d = inverse[:3, :3] @ np.array([direction[0], direction[1], direction[2]])
    return (float(o[0]), float(o[1]), float(o[2])), (float(d[0]), float(d[1]), float(d[2]))


def box_enter(origin, direction, size) -> float | None:
    """Where a ray first meets the box from 0 to `size` (a model's grid, in its
    own space): the ray parameter t, 0 when the ray starts inside, None when
    it misses. The scene picks an empty model by its box."""
    o = [float(v) for v in origin]
    d = [float(v) for v in direction]
    hit = _slab_clip(o, d, (0.0, 0.0, 0.0), tuple(float(v) for v in size))
    if hit is None:
        return None
    return max(0.0, hit[0])


def _slab_clip(o, d, lo, hi) -> tuple[float, int] | None:
    """Clip the ray to the box. Returns (t_enter, enter_axis) or None on a miss."""
    t_enter = -_INF
    t_exit = _INF
    enter_axis = 0
    for i in range(3):
        if d[i] == 0.0:
            if o[i] < lo[i] or o[i] > hi[i]:
                return None
            continue
        inv = 1.0 / d[i]
        t0 = (lo[i] - o[i]) * inv
        t1 = (hi[i] - o[i]) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 > t_enter:
            t_enter = t0
            enter_axis = i
        if t1 < t_exit:
            t_exit = t1
    if t_enter > t_exit or t_exit < 0.0:
        return None
    return t_enter, enter_axis


def _solid(voxels: np.ndarray, cell: list[int], blocked, ignore=None) -> bool:
    """True for a filled voxel not in `ignore`, or an empty cell listed in `blocked`."""
    if voxels[cell[0], cell[1], cell[2]] != 0:
        return ignore is None or (cell[0], cell[1], cell[2]) not in ignore
    if blocked is None:
        return False
    return (cell[0], cell[1], cell[2]) in blocked


def _clamp(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value
