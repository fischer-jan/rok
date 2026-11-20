"""Drawing tools for placing voxels by hand: a silhouette pushed along an axis,
clean lines and curves, mirrored placement and shading from a ramp. Numpy only.

A silhouette is drawn as the flat views of rok.core.voxel_views show a model:
pushed along y it is a front view (columns x, rows z high to low), along x a
right view (columns y, rows z high to low), along z a top view (columns x, rows
y high to low). So a view copied as text and edited extrudes back as it reads.
"""

from __future__ import annotations

import numpy as np

EMPTY = ". "


def extrude(rows: list[str], legend: dict[str, int], axis: int, corner, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """(cells, colours) of a silhouette pushed `depth` cells along `axis` from `corner`
    (the low corner of the box it fills). '.' and ' ' leave a cell alone."""
    if depth < 1:
        raise ValueError("depth must be 1 or more")
    height = len(rows)
    cells, values = [], []
    cx, cy, cz = (int(v) for v in corner)
    for r, row in enumerate(rows):
        for c, ch in enumerate(str(row)):
            if ch in EMPTY:
                continue
            if ch not in legend:
                raise KeyError(ch)
            up = height - 1 - r  # rows run from high to low
            for d in range(depth):
                if axis == 1:
                    cell = (cx + c, cy + d, cz + up)
                elif axis == 0:
                    cell = (cx + d, cy + c, cz + up)
                else:
                    cell = (cx + c, cy + up, cz + d)
                cells.append(cell)
                values.append(int(legend[ch]))
    return np.asarray(cells, dtype=np.int64).reshape(-1, 3), np.asarray(values, dtype=np.int64)


def clean(cells: np.ndarray) -> np.ndarray:
    """A path of cells without L-shaped corners: where three cells in a row make an
    L and the first and the last touch diagonally, the middle one goes. What is
    left steps evenly, as a hand-drawn voxel line does."""
    pts = [tuple(int(v) for v in c) for c in np.asarray(cells)]
    out: list[tuple] = []
    for p in pts:
        if out and p == out[-1]:
            continue
        out.append(p)
        while len(out) >= 3:
            a, b, c = (np.asarray(v) for v in out[-3:])
            ab, bc, ac = np.abs(b - a).sum(), np.abs(c - b).sum(), np.abs(c - a)
            if ab == 1 and bc == 1 and int(ac.max()) == 1 and int(ac.sum()) == 2:
                del out[-2]  # an L: the diagonal is enough
            else:
                break
    return np.asarray(out, dtype=np.int64).reshape(-1, 3)


def _line(a, b) -> list[tuple]:
    """Cells from a to b, both included: one step per cell along the longest axis."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(np.abs(b - a).max())
    if n == 0:
        return [tuple(int(round(v)) for v in a)]
    return [tuple(int(round(v)) for v in a + (b - a) * k / n) for k in range(n + 1)]


def _spline(points, closed: bool = False) -> list[tuple[tuple, int, float]]:
    """The cells of a Catmull-Rom spline through `points`, in order, each with the
    index of the point its stretch starts from and how far along that stretch it is
    (0 to 1)."""
    p = [np.asarray(v, float) for v in points]
    if len(p) < 2:
        raise ValueError("a curve needs at least two points")
    if closed:
        p = p + [p[0]]
    ext = [p[0] - (p[1] - p[0])] + p + [p[-1] + (p[-1] - p[-2])]
    path: list[tuple[tuple, int, float]] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        steps = max(2, int(np.abs(p2 - p1).max() * 3))
        prev = None
        for k in range(steps + 1):
            t = k / steps
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
            cell = tuple(int(round(v)) for v in q)
            if prev is not None and max(abs(u - w) for u, w in zip(cell, prev)) > 1:
                path.extend((c, i - 1, t) for c in _line(prev, cell)[1:-1])  # no gaps where the curve runs fast
            path.append((cell, i - 1, t))
            prev = cell
    return path


def curve(points, closed: bool = False) -> np.ndarray:
    """A smooth curve through `points` (cells), drawn one cell wide with clean steps:
    a Catmull-Rom spline through every point, laid down cell by cell."""
    return clean(np.asarray([c for c, _i, _t in _spline(points, closed)], dtype=np.int64))


def brush_curve(points, sizes, closed: bool = False) -> np.ndarray:
    """The Voxel-mode brush swept along the curve through `points`: at each cell of
    the curve a brush sphere (size 1 is one voxel, 2 is 3 across, as the Size field).
    `sizes` holds one size per point; between two points the size changes evenly,
    so a tentacle thins toward its tip."""
    from rok.core.edit import sphere_offsets

    sizes = [int(v) for v in sizes]
    if closed:
        sizes = sizes + [sizes[0]]
    out = []
    for cell, i, t in _spline(points, closed):
        size = max(1, int(round(sizes[i] + (sizes[i + 1] - sizes[i]) * t)))
        out.append(sphere_offsets(size) + np.asarray(cell, dtype=np.int64))
    return np.unique(np.concatenate(out), axis=0)


def mirrored(cells: np.ndarray, values: np.ndarray, axis: int, size) -> tuple[np.ndarray, np.ndarray]:
    """The cells and their mirror image across the middle of the grid on `axis`
    (cell i and cell size - 1 - i), without doubles."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    other = cells.copy()
    other[:, axis] = int(size[axis]) - 1 - other[:, axis]
    both = np.concatenate([cells, other])
    vals = np.concatenate([np.asarray(values), np.asarray(values)])
    _u, keep = np.unique(both, axis=0, return_index=True)
    keep = np.sort(keep)
    return both[keep], vals[keep]


def shade(voxels: np.ndarray, target: np.ndarray, ramp: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(cells, colours) that shade the `target` voxels from a ramp, lightest first,
    the way light from above falls on blocks: a voxel open to the top takes the
    lightest step, one open only to the bottom the darkest, one open only at a side
    a middle step. Voxels hidden inside keep their colour."""
    v = np.asarray(voxels) != 0
    pad = np.pad(v, 1)
    open_up = ~pad[1:-1, 1:-1, 2:]
    open_down = ~pad[1:-1, 1:-1, :-2]
    open_side = ~pad[2:, 1:-1, 1:-1] | ~pad[:-2, 1:-1, 1:-1] | ~pad[1:-1, 2:, 1:-1] | ~pad[1:-1, :-2, 1:-1]
    t = np.asarray(target, dtype=bool) & v
    n = len(ramp)
    step = np.full(v.shape, -1, dtype=np.int64)
    step[t & open_side] = n // 2 if n > 2 else min(1, n - 1)
    step[t & open_down & ~open_up] = n - 1
    step[t & open_up] = 0
    cells = np.argwhere(step >= 0)
    return cells, np.asarray(ramp, dtype=np.int64)[step[tuple(cells.T)]]
