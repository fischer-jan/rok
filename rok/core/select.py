"""Selection mask, column pick, brush, move, copy, turn, flip, delete, the
screen-space pick and the Select gestures.

Imports numpy, .edit and .raycast only. The gesture module imports this one,
so the move preview is a small dataclass of its own with the same field names
as `gesture.Preview`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .edit import SELECT, Cell, PlaneLock, SetMany, brush_sphere, in_bounds, in_bounds_mask
from .gesture_base import Gesture
from .raycast import Hit, raycast

# (cells (N, 3) float centres) -> (N, 2) region pixels, nan rows behind the eye.
Project = Callable[[np.ndarray], np.ndarray]

# Quarter turn about each axis, counter-clockwise seen from the positive end,
# as a matrix on column vectors: about z (x, y) -> (-y, x); about x
# (y, z) -> (-z, y); about y (z, x) -> (-x, z). Same turn as grid_ops.rotate90.
_TURN = {
    0: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.int64),
    1: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.int64),
    2: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64),
}

_POLY_CHUNK = 32  # edges per vectorised crossing test


def _empty_cells() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.int64)


@dataclass(frozen=True, eq=False)
class MovePreview:
    """What the overlay draws during a move drag: the selection shifted by offset."""

    offset: Cell = (0, 0, 0)
    kind: str = "MOVE"
    tool: str = SELECT
    cells: np.ndarray = field(default_factory=_empty_cells)
    normal: Cell = (0, 0, 1)
    box: tuple[Cell, Cell] | None = None
    label: str = ""


class Selection:
    """A bool mask of the grid shape plus a version counter for the overlay cache."""

    # Class counter: a fresh Selection never repeats a version.
    _next_version = 1

    def __init__(self, size) -> None:
        self.size = (int(size[0]), int(size[1]), int(size[2]))
        self.mask = np.zeros(self.size, dtype=bool)
        self.version = 0
        self._count = 0
        self._count_version = 0
        self._bounds: tuple[Cell, Cell] | None = None
        self._bounds_version = 0
        self._bump()

    def _bump(self) -> None:
        self.version = Selection._next_version
        Selection._next_version += 1

    def is_empty(self) -> bool:
        return self.count() == 0

    def count(self) -> int:
        if self._count_version != self.version:
            self._count = int(np.count_nonzero(self.mask))
            self._count_version = self.version
        return self._count

    def bounds(self) -> tuple[Cell, Cell] | None:
        """Inclusive (lo, hi) of the selected cells, or None when empty."""
        if self._bounds_version != self.version:
            self._bounds = self._compute_bounds()
            self._bounds_version = self.version
        return self._bounds

    def _compute_bounds(self) -> tuple[Cell, Cell] | None:
        lo = [0, 0, 0]
        hi = [0, 0, 0]
        for axis in range(3):
            other = tuple(i for i in range(3) if i != axis)
            used = np.flatnonzero(self.mask.any(axis=other))
            if len(used) == 0:
                return None
            lo[axis] = int(used[0])
            hi[axis] = int(used[-1])
        return ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))

    def cells(self) -> np.ndarray:
        """(N, 3) array of the selected cells in C order."""
        return np.argwhere(self.mask)

    def contains(self, cell) -> bool:
        if not in_bounds(cell, self.size):
            return False
        return bool(self.mask[int(cell[0]), int(cell[1]), int(cell[2])])

    def add_cells(self, cells) -> None:
        """Add the in-bounds rows of an (N, 3) array. Bumps only when something is new."""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        if len(cells) == 0:
            return
        cells = cells[in_bounds_mask(cells, self.size)]
        if len(cells) == 0:
            return
        xs, ys, zs = cells[:, 0], cells[:, 1], cells[:, 2]
        if self.mask[xs, ys, zs].all():
            return
        self.mask[xs, ys, zs] = True
        self._bump()

    def _full(self, mask: np.ndarray) -> np.ndarray:
        """A mask of the grid shape. A smaller one (from a grid cut at the section
        plane) is padded with False."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == self.mask.shape:
            return mask
        if mask.ndim != 3 or any(mask.shape[i] > self.size[i] for i in range(3)):
            raise ValueError(f"mask shape {mask.shape} does not fit {self.mask.shape}")
        out = np.zeros(self.size, dtype=bool)
        out[: mask.shape[0], : mask.shape[1], : mask.shape[2]] = mask
        return out

    def add_mask(self, mask: np.ndarray) -> None:
        new = self._full(mask) & ~self.mask
        if not new.any():
            return
        self.mask |= new
        self._bump()

    def remove_cells(self, cells) -> None:
        """Remove the in-bounds rows of an (N, 3) array. Bumps only when something
        selected was cleared."""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        if len(cells) == 0:
            return
        cells = cells[in_bounds_mask(cells, self.size)]
        if len(cells) == 0:
            return
        xs, ys, zs = cells[:, 0], cells[:, 1], cells[:, 2]
        if not self.mask[xs, ys, zs].any():
            return
        self.mask[xs, ys, zs] = False
        self._bump()

    def remove_mask(self, mask: np.ndarray) -> None:
        """Clear every selected cell mask covers. A smaller mask (a grid cut at the
        section plane) is padded with False, as add_mask pads it."""
        gone = self._full(mask) & self.mask
        if not gone.any():
            return
        self.mask &= ~gone
        self._bump()

    def replace(self, mask: np.ndarray) -> None:
        mask = self._full(mask)
        if np.array_equal(mask, self.mask):
            return
        self.mask = mask.copy()
        self._bump()

    def clear(self) -> None:
        if self.is_empty():
            return
        self.mask[...] = False
        self._bump()

    def prune(self, voxels: np.ndarray) -> None:
        """Drop selected cells that are empty now. Works inside bounds() only."""
        b = self.bounds()
        if b is None:
            return
        lo, hi = b
        sl = (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1))
        sub = self.mask[sl]
        gone = sub & (voxels[sl] == 0)
        if not gone.any():
            return
        sub[gone] = False
        self._bump()


# ---------------------------------------------------------------- masks


def column_mask(size, axis: int, cells) -> np.ndarray:
    """Bool grid with the whole line along axis through each cell.

    Cells whose two free coordinates lie outside the grid are ignored.
    """
    size = (int(size[0]), int(size[1]), int(size[2]))
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    u, v = (i for i in range(3) if i != axis)
    flat = np.zeros((size[u], size[v]), dtype=bool)
    if len(cells) > 0:
        cu, cv = cells[:, u], cells[:, v]
        keep = (cu >= 0) & (cu < size[u]) & (cv >= 0) & (cv < size[v])
        flat[cu[keep], cv[keep]] = True
    return np.array(np.broadcast_to(np.expand_dims(flat, axis), size))


def footprint_mask(voxels: np.ndarray, axis: int, cells) -> np.ndarray:
    """The filled cells of the columns along axis through each cell."""
    return column_mask(voxels.shape, axis, cells) & (voxels != 0)


def brush_cells(voxels: np.ndarray, centre, radius: int) -> np.ndarray:
    """(N, 3): the filled cells of the brush sphere of size `radius` around centre.

    Clipped to the grid. Size 1 is the one cell at centre (when filled); the
    sphere is `edit.sphere_offsets`, the same brush Attach, Erase and Paint use.
    """
    cells = brush_sphere(centre, radius, voxels.shape)
    if len(cells) == 0:
        return _empty_cells()
    return cells[voxels[cells[:, 0], cells[:, 1], cells[:, 2]] != 0]


# ---------------------------------------------------------------- move, delete


def move_cells(
    voxels: np.ndarray,
    selection: Selection,
    offset: Cell,
    copy: bool,
    set_many: SetMany,
) -> np.ndarray:
    """Move (or copy) the selected voxels by offset. Returns the new selection mask.

    Cells pushed outside the grid are dropped. Writes at the destination win.
    """
    src = selection.cells()
    dst = src + np.asarray(offset, dtype=src.dtype)
    return _relocate(voxels, selection, dst, set_many, copy=copy)


def turned_cells(cells, axis: int, turns: int, pivot2) -> np.ndarray:
    """Cells after `turns` quarter turns about `axis` around a doubled pivot.

    Doubled coordinates keep the turn exact: c2 = 2 cells + 1 is the centre,
    c2' = R^k (c2 - pivot2) + pivot2, cells' = (c2' - 1) // 2. No bounds test.
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    k = int(turns) % 4
    p = np.asarray(pivot2, dtype=np.int64).reshape(3)
    if k == 0 or len(cells) == 0:
        return cells.copy()
    r = np.linalg.matrix_power(_TURN[int(axis)], k)
    c2 = 2 * cells + 1 - p
    c2 = c2 @ r.T + p
    return (c2 - 1) // 2


def flipped_cells(cells, axis: int, pivot2) -> np.ndarray:
    """Cells mirrored across the plane coordinate[axis] == pivot2[axis] / 2."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    out = cells.copy()
    if len(cells) == 0:
        return out
    c2 = 2 * cells[:, axis] + 1
    c2 = 2 * int(pivot2[axis]) - c2
    out[:, axis] = (c2 - 1) // 2
    return out


def _relocate(
    voxels: np.ndarray,
    selection: Selection,
    dst: np.ndarray,
    set_many: SetMany,
    *,
    copy: bool = False,
) -> np.ndarray:
    """Write the selected colours at dst; without `copy`, clear the sources. New mask.

    Destinations outside the grid are dropped: the voxel is gone. Writes at the
    destination win, as in a move.
    """
    size = voxels.shape
    src = selection.cells()
    new = np.zeros(size, dtype=bool)
    if len(src) == 0:
        return new
    vals = voxels[src[:, 0], src[:, 1], src[:, 2]].copy()
    inside = in_bounds_mask(dst, size)
    if not copy:
        set_many(src, np.zeros(len(src), dtype=np.uint8))
    dst = dst[inside]
    if len(dst) > 0:
        set_many(dst, vals[inside])
        new[dst[:, 0], dst[:, 1], dst[:, 2]] = True
    return new


def rotate_cells(
    voxels: np.ndarray,
    selection: Selection,
    axis: int,
    turns: int,
    pivot2,
    set_many: SetMany,
) -> np.ndarray:
    """Turn the selected voxels by quarter turns about the doubled pivot.

    The pivot must pass `gizmo.snap_pivot` for cells to land on cells. Returns
    the new selection mask; zero turns write nothing.
    """
    if int(turns) % 4 == 0:
        return selection.mask.copy()
    dst = turned_cells(selection.cells(), axis, turns, pivot2)
    return _relocate(voxels, selection, dst, set_many)


def flip_cells(
    voxels: np.ndarray,
    selection: Selection,
    axis: int,
    pivot2,
    set_many: SetMany,
) -> np.ndarray:
    """Mirror the selected voxels across the pivot plane. Returns the new mask."""
    dst = flipped_cells(selection.cells(), axis, pivot2)
    return _relocate(voxels, selection, dst, set_many)


def delete_cells(voxels: np.ndarray, selection: Selection, set_many: SetMany) -> int:
    """Clear the selected voxels. Returns the changed count."""
    cells = selection.cells()
    if len(cells) == 0:
        return 0
    return int(set_many(cells, np.zeros(len(cells), dtype=np.uint8)))


# ---------------------------------------------------------------- screen space


def _is_box(polygon: np.ndarray) -> bool:
    """True for four vertices whose every edge is horizontal or vertical."""
    if len(polygon) != 4:
        return False
    nxt = polygon[[1, 2, 3, 0]]
    return bool(np.all((polygon[:, 0] == nxt[:, 0]) | (polygon[:, 1] == nxt[:, 1])))


def point_in_polygon(points, polygon) -> np.ndarray:
    """(N,) bool: which points lie inside the polygon, even-odd crossing rule.

    Vectorised over the edges in chunks of 32. A four-vertex axis-aligned
    polygon is a plain box test (bounds included).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    n = len(points)
    if n == 0 or len(polygon) < 3:
        return np.zeros(n, dtype=bool)
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    if _is_box(polygon):
        return np.all((points >= lo) & (points <= hi), axis=1)
    px = points[:, 0:1]
    py = points[:, 1:2]
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    inside = np.zeros(n, dtype=bool)
    for start in range(0, len(a), _POLY_CHUNK):
        ax = a[start : start + _POLY_CHUNK, 0]
        ay = a[start : start + _POLY_CHUNK, 1]
        bx = b[start : start + _POLY_CHUNK, 0]
        by = b[start : start + _POLY_CHUNK, 1]
        dy = by - ay
        dy = np.where(dy == 0.0, 1.0, dy)  # horizontal edges never straddle
        straddle = (ay > py) != (by > py)
        x_at = ax + (py - ay) * (bx - ax) / dy
        crossings = straddle & (px < x_at)
        inside ^= (np.count_nonzero(crossings, axis=1) & 1).astype(bool)
    return inside


def exposed_cells(voxels: np.ndarray) -> np.ndarray:
    """(N, 3): the filled cells with an open face (a neighbour empty, or the
    grid's edge): the only cells a camera can see (seen_cells wants an open
    face), by mask arithmetic."""
    filled = voxels != 0
    covered = np.ones(filled.shape, dtype=bool)
    for axis in range(3):
        lo = tuple(slice(1, None) if i == axis else slice(None) for i in range(3))
        hi = tuple(slice(None, -1) if i == axis else slice(None) for i in range(3))
        edge_lo = tuple(slice(0, 1) if i == axis else slice(None) for i in range(3))
        edge_hi = tuple(slice(-1, None) if i == axis else slice(None) for i in range(3))
        covered[lo] &= filled[hi]
        covered[hi] &= filled[lo]
        covered[edge_lo] = False
        covered[edge_hi] = False
    return np.argwhere(filled & ~covered)


def screen_cells(voxels: np.ndarray, project: Project, polygon, cells=None) -> np.ndarray:
    """(N, 3): every filled cell whose projected centre lies inside the polygon.

    Interior cells count too. `project` maps (N, 3) centres to region pixels
    with nan rows behind the eye. A bounding-box test runs before the polygon.
    `cells` narrows the candidates (the exposed cells, for a pick of what the
    camera sees); None: every filled cell.
    """
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return _empty_cells()
    pts = np.argwhere(voxels != 0) if cells is None else np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if len(pts) == 0:
        return _empty_cells()
    px = np.asarray(project(pts.astype(np.float64) + 0.5), dtype=np.float64)
    px = px.reshape(-1, 2)
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    ok = np.all(np.isfinite(px) & (px >= lo) & (px <= hi), axis=1)
    if not ok.any():
        return _empty_cells()
    pts = pts[ok]
    px = px[ok]
    return pts[point_in_polygon(px, polygon)]


_SIGHT_STEP = 0.25  # voxel lengths between two samples of a line of sight
_SIGHT_CHUNK = 4096


_NATIVE: list = []


def _native():
    """rok_native's seen_cells, or None (no native module, or ROK_PURE_SEEN set)."""
    if not _NATIVE:
        import os

        mod = None
        if not os.environ.get("ROK_PURE_SEEN"):
            try:
                import rok_native

                mod = rok_native if hasattr(rok_native, "seen_cells") else None
            except ImportError:
                mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


def seen_cells(voxels: np.ndarray, cells: np.ndarray, eye=None, toward=None) -> np.ndarray:
    """(N,) bool: which filled cells the camera sees. The native module
    answers in parallel when it is there (a Marquee pick with
    "Visible only" took seconds); the numpy path below samples the same lines."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    native = _native()
    if native is not None and len(cells):
        e = None if eye is None else tuple(float(v) for v in np.asarray(eye, dtype=np.float64).reshape(3))
        t = None if toward is None else tuple(float(v) for v in np.asarray(toward, dtype=np.float64).reshape(3))
        return np.asarray(native.seen_cells(np.ascontiguousarray(voxels, dtype=np.uint8),
                                            np.ascontiguousarray(cells), e, t), dtype=bool)
    return _seen_cells_numpy(voxels, cells, eye, toward)


def _seen_cells_numpy(voxels: np.ndarray, cells: np.ndarray, eye=None, toward=None) -> np.ndarray:
    """(N,) bool: which filled cells the camera sees.

    A cell is seen when one of its open faces that points toward the camera
    has a clear line from its centre to the camera: no other filled cell on
    it. `eye` is the camera position in grid coordinates (a perspective
    view); `toward` is the direction to the camera (an orthographic view, the
    camera infinitely far). A line is sampled every _SIGHT_STEP voxels until
    it reaches the camera or leaves the grid.
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    out = np.zeros(len(cells), dtype=bool)
    if len(cells) == 0:
        return out
    size = np.array(voxels.shape, dtype=np.int64)
    reach = float(np.linalg.norm(size)) + 2.0  # longer than any line inside the grid
    for start in range(0, len(cells), _SIGHT_CHUNK):
        part = cells[start : start + _SIGHT_CHUNK]
        centre = part.astype(np.float64) + 0.5
        if eye is not None:
            to_cam = np.asarray(eye, dtype=np.float64).reshape(1, 3) - centre
        else:
            to_cam = np.broadcast_to(np.asarray(toward, dtype=np.float64).reshape(1, 3), centre.shape)
        # The faces that point toward the camera, one per axis, if open.
        owners, points = [], []
        for axis in range(3):
            sign = np.sign(to_cam[:, axis]).astype(np.int64)
            facing = sign != 0
            nb = part.copy()
            nb[:, axis] += sign
            in_grid = np.all((nb >= 0) & (nb < size), axis=1)
            open_face = facing & ~(in_grid & _filled_at(voxels, nb, in_grid))
            idx = np.flatnonzero(open_face)
            point = centre[idx].copy()
            point[:, axis] += 0.5 * sign[idx] * (1.0 + 1e-6)  # just outside the face
            owners.append(idx)
            points.append(point)
        owner = np.concatenate(owners)
        point = np.concatenate(points)
        if len(owner) == 0:
            continue
        if eye is not None:
            line = np.asarray(eye, dtype=np.float64).reshape(1, 3) - point
            length = np.linalg.norm(line, axis=1)
            direction = line / np.maximum(length, 1e-9)[:, None]
        else:
            d = np.asarray(toward, dtype=np.float64).reshape(3)
            direction = np.broadcast_to(d / max(float(np.linalg.norm(d)), 1e-9), point.shape)
            length = np.full(len(point), np.inf)
        blocked = np.zeros(len(point), dtype=bool)
        for t in np.arange(0.0, reach, _SIGHT_STEP):
            live = ~blocked & (t < length)
            if not live.any():
                break
            at = np.floor(point + direction * t).astype(np.int64)
            inside = live & np.all((at >= 0) & (at < size), axis=1)
            if not inside.any():
                break  # every live line has left the grid: nothing more can block it
            blocked |= inside & _filled_at(voxels, at, inside)
        clear = np.zeros(len(part), dtype=bool)
        np.logical_or.at(clear, owner, ~blocked)
        out[start : start + len(part)] = clear
    return out


def _filled_at(voxels: np.ndarray, at: np.ndarray, where: np.ndarray) -> np.ndarray:
    """(N,) bool: the cell at each row is filled, only read where `where` holds."""
    hit = np.zeros(len(at), dtype=bool)
    idx = np.flatnonzero(where)
    hit[idx] = voxels[at[idx, 0], at[idx, 1], at[idx, 2]] != 0
    return hit


def invert_mask(visible: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The filled cells that are not in mask."""
    return (visible != 0) & ~np.asarray(mask, dtype=bool)


# ---------------------------------------------------------------- gestures


class PickGesture(Gesture):
    """Voxel mode with Select: the brush footprint around every hovered voxel
    joins the selection. Radius 1 is the hovered voxel alone.
    """

    tool = SELECT

    def __init__(
        self,
        voxels: np.ndarray,
        selection: Selection,
        radius: int = 1,
        subtract: bool = False,
    ) -> None:
        self.voxels = voxels
        self.selection = selection
        self.radius = max(1, int(radius))
        self.subtract = bool(subtract)
        self.hit: Hit | None = None
        self.last: Cell | None = None

    def _add(self, cell: Cell) -> bool:
        """Add or, with subtract, remove the brush cells around cell. True on a change."""
        cells = brush_cells(self.voxels, cell, self.radius)
        version = self.selection.version
        if self.subtract:
            self.selection.remove_cells(cells)
        else:
            self.selection.add_cells(cells)
        return self.selection.version != version

    def begin(self, hit: Hit) -> bool:
        if hit.wall:
            return False
        self.hit = hit
        self._add(hit.voxel)
        self.last = hit.voxel
        return True

    def extend(self, origin, direction) -> bool:
        """True when a voxel under the brush was new to the selection."""
        hit = raycast(self.voxels, origin, direction, walls=False)
        if hit is None:
            return False
        new = self._add(hit.voxel)
        self.last = hit.voxel
        return new

    def finish(self) -> int:
        return 0

    def preview(self) -> None:
        return None


class ConnectedGesture(Gesture):
    """Select's Connected modes: a click takes every voxel joined to the
    clicked one, face to face or, with `corners`, also by an edge or a
    corner. A drag takes each further region it crosses; subtract removes
    them instead."""

    tool = SELECT

    def __init__(
        self, voxels: np.ndarray, selection: Selection, corners: bool = False, subtract: bool = False
    ) -> None:
        self.voxels = voxels
        self.selection = selection
        self.corners = bool(corners)
        self.subtract = bool(subtract)
        self.hit: Hit | None = None
        self.last: Cell | None = None
        self._done = np.zeros(voxels.shape, dtype=bool)  # the regions this gesture took already

    def region(self, cell: Cell) -> np.ndarray:
        """The bool mask this gesture takes for a click on `cell`."""
        from .region import connected_blobs  # region imports numpy only

        return connected_blobs(self.voxels, np.asarray([cell], dtype=np.int64), self.corners)

    def _take(self, cell: Cell) -> bool:
        """Add (or remove) the region through `cell`. True on a change."""
        if self._done[tuple(cell)]:
            return False
        mask = self.region(cell)
        self._done |= mask
        version = self.selection.version
        if self.subtract:
            self.selection.remove_cells(np.argwhere(mask))
        else:
            self.selection.add_mask(mask)
        return self.selection.version != version

    def begin(self, hit: Hit) -> bool:
        if hit.wall:
            return False
        self.hit = hit
        self._take(hit.voxel)
        self.last = hit.voxel
        return True

    def extend(self, origin, direction) -> bool:
        hit = raycast(self.voxels, origin, direction, walls=False)
        if hit is None:
            return False
        self.last = hit.voxel
        return self._take(hit.voxel)

    def finish(self) -> int:
        return 0

    def preview(self) -> None:
        return None


TOLERANCE_MAX = 100  # Select's By colour tolerance: 0 the same colour, 100 colours half the black-white way apart
TOLERANCE_REACH = 0.5  # the OKLab distance at tolerance 100 (black to white is 1)


def similar_colours(palette: np.ndarray, index: int, tolerance: int = 0) -> np.ndarray:
    """(256,) bool: the palette slots whose colour is within `tolerance` of
    slot `index`'s (OKLab distance, TOLERANCE_REACH at 100). Slot 0 (empty)
    never; at 0 only `index` itself."""
    out = np.zeros(256, dtype=bool)
    index = int(index)
    if not 0 < index < 256:
        return out
    out[index] = True
    if tolerance <= 0:
        return out
    from .gradient import oklab

    lab = np.asarray(oklab(np.asarray(palette, dtype=np.float64)[:, :3].T)).T
    near = np.linalg.norm(lab - lab[index], axis=1) <= TOLERANCE_REACH * min(int(tolerance), TOLERANCE_MAX) / TOLERANCE_MAX
    near[0] = False
    return out | near


def colour_mask(voxels: np.ndarray, palette, index: int, tolerance: int = 0) -> np.ndarray:
    """3D bool: the voxels of slot `index`'s colour, or of a colour within `tolerance` of it."""
    if not index:
        return np.zeros(voxels.shape, dtype=bool)
    if tolerance <= 0 or palette is None:
        return voxels == int(index)
    return similar_colours(palette, index, tolerance)[voxels]


class ColourGesture(ConnectedGesture):
    """Select's By colour modes: a click takes the clicked voxel's colour, all
    over the model (`joined` False), or only where it joins the clicked voxel,
    face to face or with `corners` also by an edge or a corner. A drag takes
    each further region it crosses. With a `tolerance` above 0 colours close
    to the clicked one count as the same (similar_colours)."""

    def __init__(self, voxels, selection, joined: bool = False, corners: bool = False, subtract: bool = False,
                 palette=None, tolerance: int = 0):
        super().__init__(voxels, selection, corners, subtract)
        self.joined = bool(joined)
        self.palette = palette
        self.tolerance = int(tolerance)

    def region(self, cell: Cell) -> np.ndarray:
        index = int(self.voxels[tuple(cell)])
        if self.joined:
            from .region import connected_colour, connected_where  # region imports numpy only

            if self.tolerance <= 0 or self.palette is None:
                return connected_colour(self.voxels, cell, self.corners)
            return connected_where(self.voxels, cell, colour_mask(self.voxels, self.palette, index, self.tolerance),
                                   self.corners)
        return colour_mask(self.voxels, self.palette, index, self.tolerance)


class MoveGesture(Gesture):
    """A drag on a selected voxel slides the selection in the pressed face plane.

    Nothing is written during the drag; `finish` writes once.
    """

    tool = SELECT
    locks_axis = True

    def __init__(
        self,
        voxels: np.ndarray,
        selection: Selection,
        copy: bool,
        set_many: SetMany,
    ) -> None:
        self.voxels = voxels
        self.selection = selection
        self.copy = bool(copy)
        self.set_many = set_many
        self.hit: Hit | None = None
        self.last: Cell | None = None
        self.lock = PlaneLock(2, 0.0, 0)
        self.anchor: Cell = (0, 0, 0)
        self.offset: Cell = (0, 0, 0)
        self.lock_axis: int | None = None  # X, Y or Z pressed during the drag
        self._last_ray = None

    def toggle_axis_lock(self, axis: int) -> bool:
        """Lock the move to one axis; the same key again unlocks. The offset is read again from the last ray. True on a change."""
        self.lock_axis = None if self.lock_axis == axis else int(axis)
        if self._last_ray is None:
            return False
        return self.extend(*self._last_ray, force=True)

    def begin(self, hit: Hit) -> bool:
        if hit.wall or not self.selection.contains(hit.voxel):
            return False
        self.hit = hit
        self.lock = PlaneLock.from_hit(hit, hit.voxel)
        self.anchor = hit.voxel
        self.offset = (0, 0, 0)
        self.last = hit.voxel
        return True

    def extend(self, origin, direction, force: bool = False) -> bool:
        """True when the offset changed."""
        self._last_ray = (origin, direction)
        cell = self.lock.cell_at(origin, direction)
        if cell is None:
            return False
        offset = [int(cell[i]) - int(self.anchor[i]) for i in range(3)]
        offset[self.lock.axis] = 0
        if self.lock_axis is not None:
            offset = [offset[i] if i == self.lock_axis else 0 for i in range(3)]
        offset_cell = (offset[0], offset[1], offset[2])
        if offset_cell == self.offset and not force:
            return False
        self.offset = offset_cell
        self.last = (
            self.anchor[0] + offset[0],
            self.anchor[1] + offset[1],
            self.anchor[2] + offset[2],
        )
        return True

    def finish(self) -> int:
        """Write the move and let the selection travel. Returns the changed count."""
        if self.offset == (0, 0, 0):
            return 0
        total = 0

        def counted(cells: np.ndarray, values: np.ndarray) -> int:
            nonlocal total
            n = int(self.set_many(cells, values))
            total += n
            return n

        new = move_cells(self.voxels, self.selection, self.offset, self.copy, counted)
        self.selection.replace(new)
        return total

    def preview(self) -> MovePreview:
        label = "" if self.lock_axis is None else f"lock {'XYZ'[self.lock_axis]}"
        return MovePreview(offset=self.offset, label=label)
