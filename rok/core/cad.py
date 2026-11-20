"""The CAD view's facts: fitted-face normals, corner bits, straight edges
and voxel length.

The picture stays the real voxels. The light is one grey on each voxel
face, fixed in the world, so a step keeps its own tone and the form stays
readable. A corner bit is set only where two fitted faces meet at a sharp
edge. The shader can draw the four edges of a voxel face, so a bit names
one of the six face directions: the other face's outward axis, which is
the in-face direction of that corner on this cell.

A straight edge is one outer or inner corner run along a single axis. Its
length is what a click measures. The part a solid cell hides from the
camera is a hidden run. The middle of a flat face is not an edge.

Lengths are one number for the whole scene. They are stored in metres.
Until a length is set, text stays in cells.
"""

from __future__ import annotations

import numpy as np

from .facets import Faces

FOOT_METRES = 0.3048  # one foot, exactly the international foot
CORNER_DEG = 20.0  # the same turn Crease calls a sharp edge
OUTSET = 0.8  # cells: how far a side label sits outside the box

def field(voxels: np.ndarray) -> np.ndarray:
    """(x, y, z, 4) uint8. rgb is the fitted face normal, 1..255 around 128,
    and 0 where the cell has no fitted face. The picture does not tint by
    it. a is the corner mask. Fewer than two faces (one voxel) leaves the
    array at zero, so the view draws no corner line."""
    voxels = np.asarray(voxels)
    if voxels.ndim != 3:
        raise ValueError(f"voxels must be a 3D grid, got {voxels.ndim}D")
    shape = tuple(int(d) for d in voxels.shape)
    out = np.zeros(shape + (4,), dtype=np.uint8)
    # The fitter needs two cells on every axis. A thinner grid keeps the
    # geometric normal and draws no corner line.
    if min(shape) < 2:
        return out
    found = Faces.of(voxels)
    known = found.label >= 0
    if not np.any(known) or len(found.normals) < 2:
        return out
    cells = found.cells[known]
    labels_of = found.label[known]
    labels = np.full(shape, -1, np.int32)
    labels[cells[:, 0], cells[:, 1], cells[:, 2]] = labels_of
    normals = found.normals[labels_of]
    rgb = np.clip(np.rint(normals * 127.0 + 128.0), 1, 255).astype(np.uint8)
    out[cells[:, 0], cells[:, 1], cells[:, 2], :3] = rgb

    edge_cells, _edge_normals = found.edges(CORNER_DEG, width=2)
    if len(edge_cells) == 0:
        return out
    edge = np.zeros(shape, dtype=bool)
    edge[edge_cells[:, 0], edge_cells[:, 1], edge_cells[:, 2]] = True
    grid_n = np.zeros(shape + (3,), dtype=np.float64)
    grid_n[cells[:, 0], cells[:, 1], cells[:, 2]] = normals
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        lo_s, hi_s = tuple(lo), tuple(hi)
        la, lb = labels[lo_s], labels[hi_s]
        na, nb = grid_n[lo_s], grid_n[hi_s]
        # Two opposite sides of a thin solid touch, and that meeting is not a
        # corner: both normals run along the join and point away from each other.
        dot = (na * nb).sum(axis=-1)
        along = (np.abs(na[..., axis]) > 0.7) & (np.abs(nb[..., axis]) > 0.7)
        hit = (la >= 0) & (lb >= 0) & (la != lb) & (edge[lo_s] | edge[hi_s])
        hit &= ~((dot < -0.5) & along)
        if not np.any(hit):
            continue
        # The corner on this cell runs toward the other face's outward axis.
        out[lo_s + (3,)] |= _mask(grid_n[hi_s], hit)
        out[hi_s + (3,)] |= _mask(grid_n[lo_s], hit)
    return out


def straight_edges(voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The straight sharp corners of a voxel grid.

    Returns `(segments, keys)`. `segments` is `(n, 2, 3)` float, the two
    corners in cell coordinates. `keys` is `(n, 5)` int: the axis, the two
    fixed corner coordinates, and the start and end along the axis. The
    length in cells is end minus start. An empty grid has none."""
    voxels = np.asarray(voxels)
    if voxels.ndim != 3:
        raise ValueError(f"voxels must be a 3D grid, got {voxels.ndim}D")
    empty_s = np.zeros((0, 2, 3), np.float64)
    empty_k = np.zeros((0, 5), np.int32)
    solid = voxels != 0
    if solid.size == 0 or not np.any(solid):
        return empty_s, empty_k
    nx, ny, nz = (int(d) for d in solid.shape)
    pad = np.pad(solid, 1)
    parts_s, parts_k = [], []
    # Along X. The four cells around a unit edge meet at its corner.
    parts = _edge_parts(0, (1, 2), _sharp(
        pad[1:nx + 1, 0:ny + 1, 0:nz + 1],
        pad[1:nx + 1, 0:ny + 1, 1:nz + 2],
        pad[1:nx + 1, 1:ny + 2, 0:nz + 1],
        pad[1:nx + 1, 1:ny + 2, 1:nz + 2],
    ))
    parts_s.append(parts[0])
    parts_k.append(parts[1])
    # Along Y. sharp is stored with the run axis first.
    sharp = _sharp(
        pad[0:nx + 1, 1:ny + 1, 0:nz + 1],
        pad[0:nx + 1, 1:ny + 1, 1:nz + 2],
        pad[1:nx + 2, 1:ny + 1, 0:nz + 1],
        pad[1:nx + 2, 1:ny + 1, 1:nz + 2],
    )
    parts = _edge_parts(1, (0, 2), sharp.transpose(1, 0, 2))
    parts_s.append(parts[0])
    parts_k.append(parts[1])
    # Along Z.
    sharp = _sharp(
        pad[0:nx + 1, 0:ny + 1, 1:nz + 1],
        pad[0:nx + 1, 1:ny + 2, 1:nz + 1],
        pad[1:nx + 2, 0:ny + 1, 1:nz + 1],
        pad[1:nx + 2, 1:ny + 2, 1:nz + 1],
    )
    parts = _edge_parts(2, (0, 1), sharp.transpose(2, 0, 1))
    parts_s.append(parts[0])
    parts_k.append(parts[1])
    return np.concatenate(parts_s), np.concatenate(parts_k)


def _sharp(a, b, c, d) -> np.ndarray:
    """True where the four cells around an edge make a corner.

    One filled cell is an outer corner. Three is an inner corner. Two
    filled cells are the middle of a flat face, so that place is not an edge."""
    count = a.astype(np.uint8)
    count += b
    count += c
    count += d
    return (count == 1) | (count == 3)


def _edge_parts(axis: int, fixed: tuple[int, int], sharp: np.ndarray):
    """Segments and keys for the runs in `sharp`. Axis 0 of `sharp` is the run."""
    starts, ends = _runs(sharp)
    n = len(starts)
    segments = np.zeros((n, 2, 3), np.float64)
    keys = np.zeros((n, 5), np.int32)
    if n == 0:
        return segments, keys
    u_axis, v_axis = fixed
    segments[:, 0, axis] = starts[:, 0]
    segments[:, 1, axis] = ends[:, 0]
    segments[:, :, u_axis] = starts[:, 1:2]
    segments[:, :, v_axis] = starts[:, 2:3]
    keys[:, 0] = axis
    keys[:, 1] = starts[:, 1]
    keys[:, 2] = starts[:, 2]
    keys[:, 3] = starts[:, 0]
    keys[:, 4] = ends[:, 0]
    return segments, keys


def _runs(sharp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and exclusive end of each True run along axis 0, as `(n, 3)`."""
    empty = np.zeros((0, 3), np.int32)
    if sharp.size == 0 or not np.any(sharp):
        return empty, empty
    padded = np.pad(sharp, ((1, 1), (0, 0), (0, 0)))
    change = np.diff(padded.astype(np.int8), axis=0)
    starts = np.argwhere(change == 1)
    ends = np.argwhere(change == -1)
    if len(starts) == 0 or len(starts) != len(ends):
        return empty, empty
    stride = max(int(sharp.shape[2]), 1)
    line_s = starts[:, 1] * stride + starts[:, 2]
    line_e = ends[:, 1] * stride + ends[:, 2]
    starts = starts[np.lexsort((starts[:, 0], line_s))]
    ends = ends[np.lexsort((ends[:, 0], line_e))]
    same = np.all(starts[:, 1:] == ends[:, 1:], axis=1)
    return starts[same].astype(np.int32), ends[same].astype(np.int32)


def hidden_segments(voxels: np.ndarray, segments: np.ndarray, toward, covers=()) -> np.ndarray:
    """The parts of `segments` that a solid cell hides.

    `toward` points at the camera, in the same coordinates as `segments`.
    An orthographic view uses one direction for every edge. `covers` is
    other grids in front, each `(shift, voxels)`: a point `p` on a segment
    is `p + shift` in that grid. Returns `(m, 2, 3)` float. A fully visible
    edge adds nothing. A zero direction adds nothing."""
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    empty = np.zeros((0, 2, 3), np.float64)
    if len(segments) == 0:
        return empty
    direction = np.asarray(toward, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-8:
        return empty
    direction = direction / norm
    voxels = np.asarray(voxels)
    if voxels.ndim != 3:
        raise ValueError(f"voxels must be a 3D grid, got {voxels.ndim}D")
    grids = [(np.zeros(3), voxels != 0)]
    for shift, block in covers:
        block = np.asarray(block)
        if block.ndim != 3:
            raise ValueError(f"a cover must be a 3D grid, got {block.ndim}D")
        grids.append((np.asarray(shift, dtype=np.float64).reshape(3), block != 0))

    delta = segments[:, 1] - segments[:, 0]
    length = np.max(np.abs(delta), axis=1)
    # One sample in the middle of each cell of the edge. A corner sits on the
    # grid boundary, and a step toward the camera can leave the grid there.
    count = np.maximum(np.rint(length).astype(np.int32), 1)
    steps = np.arange(int(count.max()))
    valid = steps[None, :] < count[:, None]
    alpha = (steps[None, :] + 0.5) / count[:, None].astype(np.float64)
    points = segments[:, 0][:, None, :] + delta[:, None, :] * alpha[..., None]
    flags = _occluded_by(grids, points[valid], direction)
    hidden = np.zeros(valid.shape, dtype=bool)
    hidden[valid] = flags

    parts = []
    for index in range(len(segments)):
        row = hidden[index, : int(count[index])]
        if not row.any():
            continue
        change = np.diff(np.pad(row.astype(np.int8), 1))
        starts = np.flatnonzero(change == 1)
        ends = np.flatnonzero(change == -1)
        span = float(count[index])
        for start, end in zip(starts, ends):
            parts.append((
                segments[index, 0] + delta[index] * (float(start) / span),
                segments[index, 0] + delta[index] * (float(end) / span),
            ))
    if not parts:
        return empty
    out = np.empty((len(parts), 2, 3), np.float64)
    for index, (start, end) in enumerate(parts):
        out[index, 0] = start
        out[index, 1] = end
    return out


def _occluded_by(grids, points, direction) -> np.ndarray:
    """True where any grid has a solid cell between the point and the camera."""
    hidden = np.zeros(len(points), dtype=bool)
    for shift, solid in grids:
        todo = np.flatnonzero(~hidden)
        if len(todo) == 0:
            return hidden
        hidden[todo] = _occluded(solid, points[todo] + shift, direction)
    return hidden


def _occluded(solid: np.ndarray, points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """True where `solid` blocks the view of a point.

    The ray starts a short step toward the camera, so a corner is hidden
    only when the step lands in a solid cell or a later cell is solid."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    hidden = np.zeros(len(points), dtype=bool)
    if len(points) == 0 or solid.size == 0:
        return hidden
    origin = points + direction * 0.05
    shape = np.asarray(solid.shape, dtype=np.int32)
    # A point outside the grid's box (a grid in front of the point's own model): its ray
    # starts where it enters the box. A ray that never enters is not hidden by this grid.
    near = np.full((len(points), 3), -np.inf)
    far = np.full((len(points), 3), np.inf)
    for axis in range(3):
        if abs(direction[axis]) > 1e-8:
            a = (0.0 - origin[:, axis]) / direction[axis]
            b = (float(shape[axis]) - origin[:, axis]) / direction[axis]
            near[:, axis] = np.minimum(a, b)
            far[:, axis] = np.maximum(a, b)
        else:
            beside = (origin[:, axis] < 0.0) | (origin[:, axis] >= shape[axis])
            near[beside, axis] = np.inf
    t_in = near.max(axis=1)
    enters = (t_in > 0.0) & np.isfinite(t_in) & (t_in <= far.min(axis=1))
    origin[enters] += direction * (t_in[enters, None] + 1e-6)
    cell = np.floor(origin).astype(np.int32)
    cell[enters] = np.clip(cell[enters], 0, shape - 1)
    inside = np.all((cell >= 0) & (cell < shape), axis=1)
    if inside.any():
        sel = np.flatnonzero(inside)
        hidden[sel] = solid[cell[sel, 0], cell[sel, 1], cell[sel, 2]]
    alive = np.flatnonzero(inside & ~hidden)
    if len(alive) == 0:
        return hidden

    step = np.zeros(3, np.int32)
    step[direction > 1e-8] = 1
    step[direction < -1e-8] = -1
    t_delta = np.full(3, np.inf)
    moving = step != 0
    t_delta[moving] = np.abs(1.0 / direction[moving])
    cell = cell[alive]
    origin = origin[alive]
    t_max = np.full((len(alive), 3), np.inf)
    for axis in range(3):
        if step[axis] > 0:
            t_max[:, axis] = (cell[:, axis] + 1.0 - origin[:, axis]) / direction[axis]
        elif step[axis] < 0:
            t_max[:, axis] = (cell[:, axis] - origin[:, axis]) / direction[axis]
    np.maximum(t_max, 0.0, out=t_max)
    ids = alive
    limit = int(shape.sum()) + 2
    for _ in range(limit):
        if len(ids) == 0:
            break
        axis = np.argmin(t_max, axis=1)
        rows = np.arange(len(ids))
        finite = np.isfinite(t_max[rows, axis])
        if not finite.all():
            ids = ids[finite]
            cell = cell[finite]
            t_max = t_max[finite]
            axis = axis[finite]
            rows = np.arange(len(ids))
            if len(ids) == 0:
                break
        t_max[rows, axis] += t_delta[axis]
        cell[rows, axis] += step[axis]
        outside = np.any((cell < 0) | (cell >= shape), axis=1)
        stay = ~outside
        if stay.any():
            sel = np.flatnonzero(stay)
            hit = solid[cell[sel, 0], cell[sel, 1], cell[sel, 2]]
            hidden[ids[sel[hit]]] = True
            stay[sel[hit]] = False
        ids = ids[stay]
        cell = cell[stay]
        t_max = t_max[stay]
    return hidden


def nearest_segment(starts, ends, point, radius: float, depth=None) -> int:
    """Index of the segment closest to a screen `point`, or -1.

    `starts` and `ends` are `(n, 2)`. A non-finite endpoint is ignored.
    Inside `radius`, the closest segment wins. A nearer `depth` breaks a
    tie of one pixel, then the longer segment wins."""
    starts = np.asarray(starts, dtype=np.float64).reshape(-1, 2)
    ends = np.asarray(ends, dtype=np.float64).reshape(-1, 2)
    n = len(starts)
    if n == 0 or len(ends) != n:
        return -1
    point = np.asarray(point, dtype=np.float64).reshape(2)
    ab = ends - starts
    ab2 = np.sum(ab * ab, axis=1)
    ap = point - starts
    t = np.zeros(n, dtype=np.float64)
    np.divide(np.sum(ap * ab, axis=1), ab2, out=t, where=ab2 > 1e-12)
    np.clip(t, 0.0, 1.0, out=t)
    closest = starts + ab * t[:, None]
    dist = np.linalg.norm(closest - point, axis=1)
    dist[~np.isfinite(dist)] = np.inf
    inside = np.flatnonzero(dist <= float(radius))
    if len(inside) == 0:
        return -1
    best = float(dist[inside].min())
    close = inside[dist[inside] <= best + 1.0]
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        if len(depth) == n:
            scored = depth[close].copy()
            scored[~np.isfinite(scored)] = np.inf
            near = float(scored.min())
            close = close[scored <= near + 1e-6]
    length = np.linalg.norm(ends[close] - starts[close], axis=1)
    return int(close[int(np.argmax(length))])


def _mask(normals: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """uint8 corner bit of each normal's strongest axis, or 0 where `hit` is false."""
    axis = np.argmax(np.abs(normals), axis=-1)
    component = np.take_along_axis(normals, axis[..., None], axis=-1)[..., 0]
    bit = axis.astype(np.uint8) * np.uint8(2) + np.where(component >= 0.0, np.uint8(0), np.uint8(1))
    return np.where(hit, np.left_shift(np.uint8(1), bit), np.uint8(0))


def decode_normal(texel: np.ndarray) -> np.ndarray:
    """The unit-ish normal stored in a field texel. (0, 0, 0) means no face."""
    rgb = np.asarray(texel, dtype=np.float64)[:3]
    if not np.any(rgb):
        return np.zeros(3)
    return (rgb - 128.0) / 127.0


def measurement(cells: int, metres: float | None, unit: str) -> str:
    """`cells` in the scene's unit, or in cells when no length is set."""
    n = int(cells)
    if metres is None or metres <= 0:
        return f"{n} cell" if n == 1 else f"{n} cells"
    length = metres / FOOT_METRES if unit == "ft" else metres
    suffix = "ft" if unit == "ft" else "m"
    return f"{trim(n * length)} {suffix}"


def side_text(axis: int, cells: int, metres: float | None, unit: str) -> str:
    """A side label: the axis, then the length."""
    return f"{'XYZ'[int(axis)]} {measurement(cells, metres, unit)}"


def trim(value: float) -> str:
    """Up to four decimal places, without trailing zeros."""
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def display_number(metres: float, unit: str) -> str:
    """The stored metres as a number in `unit`."""
    value = metres / FOOT_METRES if unit == "ft" else metres
    return trim(value)


def to_metres(value: float, unit: str) -> float:
    """`value`, typed in `unit`, as metres."""
    return float(value) * FOOT_METRES if unit == "ft" else float(value)


def parse_length(text: str) -> float | None:
    """A positive number below one million, or None when `text` is empty.

    Raises ValueError when `text` is not such a number."""
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError("length") from None
    if not (value > 0) or not (value < 1_000_000):
        raise ValueError("length")
    return value


def dimension_ends(lo, hi, toward_eye) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three segments in world cells, one per axis, just outside the box.

    `lo` is the inclusive corner and `hi` the exclusive one. A component of
    `toward_eye` at or above zero means that high side faces the eye, so the
    line sits there. Each line is `OUTSET` cells further out."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    eye = np.asarray(toward_eye, dtype=np.float64)
    out = []
    for axis in range(3):
        start = lo.copy()
        end = lo.copy()
        end[axis] = hi[axis]
        for other in range(3):
            if other == axis:
                continue
            high = float(eye[other]) >= 0.0
            side = hi[other] if high else lo[other]
            push = OUTSET if high else -OUTSET
            start[other] = side + push
            end[other] = side + push
        out.append((start, end))
    return out
