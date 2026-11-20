"""Moving whole models in the scene: the drag, the arrow-key nudge and the drop.

 Everything is in world space, where a model's cells are a translation
of the scene's cells, so an offset in world units is an offset in cells.
"""

from __future__ import annotations

import numpy as np

DRAG_START_PX = 4.0  # the mouse moves this far before a press becomes a drag
NUDGE_BIG = 10  # Shift+arrow


def ray_plane_z(origin, direction, z: float):
    """Where the ray meets the horizontal plane at height `z`, or None when it
    runs parallel to it or points away."""
    oz, dz = float(origin[2]), float(direction[2])
    if abs(dz) < 1e-9:
        return None
    t = (z - oz) / dz
    if t <= 0:
        return None
    return np.asarray(origin, dtype=np.float64) + t * np.asarray(direction, dtype=np.float64)


def ray_plane_facing(origin, direction, point, view_dir):
    """Where the ray meets the upright plane through `point` that faces the
    view (its normal is the view direction flattened onto the floor). None
    when the view looks straight down."""
    normal = np.array([view_dir[0], view_dir[1], 0.0], dtype=np.float64)
    length = np.linalg.norm(normal)
    if length < 1e-6:
        return None
    normal /= length
    d = np.asarray(direction, dtype=np.float64)
    denom = float(normal @ d)
    if abs(denom) < 1e-9:
        return None
    o = np.asarray(origin, dtype=np.float64)
    t = float(normal @ (np.asarray(point, dtype=np.float64) - o)) / denom
    if t <= 0:
        return None
    return o + t * d


def drag_offset(start, now, vertical: bool) -> tuple[int, int, int]:
    """Whole cells between two points on the drag plane: x and y on the floor,
    z alone when the drag is vertical."""
    delta = np.asarray(now, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    if vertical:
        return (0, 0, int(round(delta[2])))
    return (int(round(delta[0])), int(round(delta[1])), 0)


def nudge_axes(right, forward) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """The world axes the arrow keys move along: Right is the floor axis closest
    to the screen's right, Up the floor axis closest to the view's forward
    direction. `right` and `forward` are world vectors; their z is ignored."""

    def closest(v, exclude=None):
        best, best_dot = None, -1.0
        for axis in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
            if exclude is not None and abs(axis[0]) == abs(exclude[0]) and abs(axis[1]) == abs(exclude[1]):
                continue
            dot = axis[0] * v[0] + axis[1] * v[1]
            if dot > best_dot:
                best, best_dot = axis, dot
        return best

    r = closest(right)
    return r, closest(forward, exclude=r)


def floor_offset(models, floor_z: int = 0) -> int:
    """How far to move models, together, so the lowest of them sits on the
    floor grid (Drop to Floor). `models` lists (occupied, position): a bool (X, Y, Z) grid and its
    world position. A model's lowest cell is its lowest voxel, or its box's
    bottom when empty. Negative: down; positive: up out of the floor."""
    lows = []
    for occupied, position in models:
        filled = np.flatnonzero(np.asarray(occupied).any(axis=(0, 1)))
        lows.append(int(position[2]) + (int(filled[0]) if filled.size else 0))
    return int(floor_z) - min(lows) if lows else 0


def drop_distance(occupied: np.ndarray, position, others=(), floor_z: int = 0) -> int:
    """How far a model can fall straight down before it rests on the floor or
    on another model's voxels. 0 when it already rests or nothing is below.

    `occupied` is the model's (X, Y, Z) bool mask at `position` (its lowest
    cell). `others` lists (mask, position) of the models it may land on. An
    empty model falls by its box, as the scene picks it by its box.
    """
    px, py, pz = (int(v) for v in position)
    columns = occupied.any(axis=2)
    if not columns.any():
        occupied = np.ones(occupied.shape, dtype=bool)  # empty: the whole box falls
        columns = occupied.any(axis=2)
        if not columns.any():
            return 0
    # The lowest filled cell of each column, in world z.
    lowest = np.argmax(occupied, axis=2) + pz
    best = int(lowest[columns].min()) - int(floor_z)
    for mask, (ox, oy, oz) in others:
        ox, oy, oz = int(ox), int(oy), int(oz)
        # The overlap of the two footprints, in world x and y.
        x0, x1 = max(px, ox), min(px + occupied.shape[0], ox + mask.shape[0])
        y0, y1 = max(py, oy), min(py + occupied.shape[1], oy + mask.shape[1])
        if x0 >= x1 or y0 >= y1:
            continue
        mine = columns[x0 - px:x1 - px, y0 - py:y1 - py]
        low = lowest[x0 - px:x1 - px, y0 - py:y1 - py]
        theirs = mask[x0 - ox:x1 - ox, y0 - oy:y1 - oy, :]
        z = np.arange(theirs.shape[2]) + oz
        # Their highest filled cell below my lowest cell, per column.
        below = theirs & (z[None, None, :] < low[:, :, None])
        has = below.any(axis=2) & mine
        if not has.any():
            continue
        top = theirs.shape[2] - 1 - np.argmax(below[:, :, ::-1], axis=2) + oz
        gaps = low[has] - top[has] - 1
        best = min(best, int(gaps.min()))
    return max(0, best)


def drop_onto_offset(members, others=(), floor_z: int = 0) -> int:
    """How far to move models, together, so they fall straight down until
    one of them rests on another model's voxel or on the floor (Drop: it
    stops at any voxel). `members` and `others` list
    (occupied, position). 0 or negative: 0 when they already rest."""
    fall = min((drop_distance(occupied, position, others, floor_z) for occupied, position in members), default=0)
    return -fall


def pick_order(voxel_hits, box_hits) -> list[int]:
    """The models under the mouse in click order: those with a voxel there,
    nearest voxel first, then those whose box alone is there (an empty model,
    or a gap in a sparse one), nearest box first. Each hit is (t, uid)."""
    first = [uid for _t, uid in sorted(voxel_hits)]
    seen = set(first)
    rest = [uid for _t, uid in sorted(box_hits) if uid not in seen]
    return first + list(dict.fromkeys(rest))


def choose_grab(under: list[int], solid, members) -> int:
    """The model a scene press grabs. `under` is the click order
    (`pick_order`), `solid` the uids with a voxel under the mouse, `members`
    the picked group. A picked model grabs the press when one of its voxels is
    there, even behind another model's (so it can still be dragged); by its
    box alone it does not, or a click on another model beside it could never
    pick that one. Else the first in `under`."""
    solid = set(solid)
    members = set(members)
    return next((uid for uid in under if uid in members and uid in solid), under[0])


def next_in_cycle(under: list[int], picked: int) -> int | None:
    """The model a click on the picked one goes to: the next one along the line
    of sight, back to the nearest after the last. None when nothing else is
    under the mouse."""
    if len(under) < 2:
        return None
    if picked not in under:
        return under[0]
    return under[(under.index(picked) + 1) % len(under)]


SNAP_CELLS = 2  # a face this close to a neighbour's snaps to it
SNAP_NEAR = 8  # a neighbour counts when its box is this close across the other axes


def _ranges_near(a_lo, a_hi, b_lo, b_hi, near: int) -> bool:
    """True when [a_lo, a_hi) and [b_lo, b_hi) overlap or lie within `near`."""
    return a_lo < b_hi + near and b_lo < a_hi + near


def snap_offset(box, offset, others, axes, radius: int = SNAP_CELLS, near: int = SNAP_NEAR):
    """The drag offset after snapping, and the indices of the boxes it snapped to.

    `box` is the moving (lo, hi) in cells, hi exclusive, before the offset.
    `others` lists the (lo, hi) of the boxes it may snap to. On each axis in
    `axes`, a face of the moved box within `radius` of a neighbour's face
    snaps to it: flush (the two boxes touch) or aligned (the faces line up).
    A neighbour counts when it is near across the other two axes. The nearest
    snap wins per axis; an exact fit stays.
    """
    out = [int(v) for v in offset]
    snapped: set[int] = set()
    lo = [int(box[0][i]) + out[i] for i in range(3)]
    hi = [int(box[1][i]) + out[i] for i in range(3)]
    for axis in axes:
        rest = [i for i in range(3) if i != axis]
        best, best_k = None, []
        for k, (olo, ohi) in enumerate(others):
            if not all(_ranges_near(lo[i], hi[i], olo[i], ohi[i], near) for i in rest):
                continue
            for delta, flush in ((ohi[axis] - lo[axis], True), (olo[axis] - hi[axis], True),
                                 (olo[axis] - lo[axis], False), (ohi[axis] - hi[axis], False)):
                if abs(delta) > radius:
                    continue
                # A neighbour lights up when the snap pulls, or when the boxes touch;
                # faces that merely already line up (two ships in a row) do not.
                shows = delta != 0 or flush
                if best is None or abs(delta) < abs(best):
                    best, best_k = delta, ([k] if shows else [])
                elif delta == best and shows:
                    best_k.append(k)
        if best is not None:
            out[axis] += best
            lo[axis] += best
            hi[axis] += best
            snapped.update(best_k)
    return (out[0], out[1], out[2]), snapped
