"""Connected regions: 2D scanline floods on one layer and the 3D colour blob.

Imports numpy only. The floods walk runs, not cells, with an explicit stack,
so the cost is proportional to the number of runs.
"""

from __future__ import annotations

import numpy as np

Cell = tuple[int, int, int]


def _run_around(line: np.ndarray, i: int) -> tuple[int, int]:
    """Bounds [i0, i1) of the True run of a 1D bool line that contains index i."""
    holes = ~line
    before = np.flatnonzero(holes[:i])
    i0 = int(before[-1]) + 1 if before.size else 0
    after = np.flatnonzero(holes[i:])
    i1 = i + int(after[0]) if after.size else int(line.shape[0])
    return i0, i1


def _run_starts(seg: np.ndarray) -> np.ndarray:
    """Indices where a True run starts in a 1D bool segment."""
    return np.flatnonzero(seg & ~np.concatenate(([False], seg[:-1])))


def _axis_sign(normal) -> tuple[int, int]:
    for i in range(3):
        n = int(normal[i])
        if n != 0:
            return i, (1 if n > 0 else -1)
    raise ValueError("normal has no non-zero component")


def _free_coords(cell, axis: int) -> tuple[int, int]:
    """The two free coordinates of cell in ascending axis order."""
    u, v = (int(cell[i]) for i in range(3) if i != axis)
    return u, v


def flood_2d(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """4-connected component of seed in a 2D bool mask, as a mask of the same shape.

    A seed outside the mask, or on a False cell, gives an all-False mask.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    n_u, n_v = mask.shape
    u, v = int(seed[0]), int(seed[1])
    if not (0 <= u < n_u and 0 <= v < n_v) or not mask[u, v]:
        return out
    todo = mask.copy()
    stack = [(u, v)]
    while stack:
        u, v = stack.pop()
        col = todo[:, v]
        if not col[u]:
            continue
        u0, u1 = _run_around(col, u)
        out[u0:u1, v] = True
        col[u0:u1] = False
        for vn in (v - 1, v + 1):
            if 0 <= vn < n_v:
                for s in _run_starts(todo[u0:u1, vn]):
                    stack.append((u0 + int(s), vn))
    return out


def _layer_cells(region: np.ndarray, axis: int, k: int) -> np.ndarray:
    """Lift a 2D region mask back to (N, 3) cells with coordinate[axis] == k."""
    return np.insert(np.argwhere(region).astype(np.int64), axis, int(k), axis=1)


def face_region(voxels: np.ndarray, voxel: Cell, normal) -> np.ndarray:
    """(N, 3): the connected, coplanar, same-colour, exposed face region.

    The region contains voxel on its face with the given normal. An empty seed
    cell gives shape (0, 3).
    """
    a, s = _axis_sign(normal)
    k = int(voxel[a])
    layer = np.take(voxels, k, axis=a)
    seed = _free_coords(voxel, a)
    colour = int(layer[seed])
    if colour == 0:
        return np.zeros((0, 3), dtype=np.int64)
    kn = k + s
    if 0 <= kn < voxels.shape[a]:
        exposed = np.take(voxels, kn, axis=a) == 0
    else:
        exposed = np.ones(layer.shape, dtype=bool)
    mask = (layer == colour) & exposed
    return _layer_cells(flood_2d(mask, seed), a, k)


def wall_region(voxels: np.ndarray, target: Cell, normal) -> np.ndarray:
    """(N, 3): the 4-connected empty area of the layer target[a] that contains target."""
    a, _ = _axis_sign(normal)
    k = int(target[a])
    layer = np.take(voxels, k, axis=a)
    seed = _free_coords(target, a)
    return _layer_cells(flood_2d(layer == 0, seed), a, k)


def connected_colour(voxels: np.ndarray, seed: Cell, corners: bool = False) -> np.ndarray:
    """3D bool mask of the same-colour blob that contains seed: joined face to
    face (6-connected), or with `corners` also by an edge or a corner.

    Runs along x; the neighbour lines get one seed per run that touches the
    marked run. A seed on an empty cell gives an all-False mask.
    """
    out = np.zeros(voxels.shape, dtype=bool)
    if not _filled_seed(voxels, seed):
        return out
    x, y, z = (int(seed[i]) for i in range(3))
    return _walk_runs(out, voxels == int(voxels[x, y, z]), (x, y, z), corners)


def connected_where(voxels: np.ndarray, seed: Cell, allowed: np.ndarray, corners: bool = False) -> np.ndarray:
    """3D bool mask of the blob of `allowed` cells (a 3D bool mask) that
    contains seed, joined as connected_colour joins. A seed on an empty cell
    gives an all-False mask."""
    out = np.zeros(voxels.shape, dtype=bool)
    if not _filled_seed(voxels, seed):
        return out
    x, y, z = (int(seed[i]) for i in range(3))
    todo = np.array(allowed, dtype=bool, copy=True)
    todo[x, y, z] = True
    return _walk_runs(out, todo, (x, y, z), corners)


def connected_filled(voxels: np.ndarray, seed: Cell) -> np.ndarray:
    """3D bool mask of the 6-connected filled blob of any colour that contains seed."""
    out = np.zeros(voxels.shape, dtype=bool)
    if not _filled_seed(voxels, seed):
        return out
    x, y, z = (int(seed[i]) for i in range(3))
    return _walk_runs(out, voxels != 0, (x, y, z))


def connected_blobs(voxels: np.ndarray, seeds: np.ndarray, corners: bool = False) -> np.ndarray:
    """The union of the filled blobs through the seed cells: voxels joined
    face to face (6-connected), or with `corners` also edge to edge and
    corner to corner (26-connected).

    A seed inside a blob found already costs nothing, so a whole selected
    blob is one flood. Empty and out-of-grid seeds are skipped.
    """
    out = np.zeros(voxels.shape, dtype=bool)
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1, 3)
    if len(seeds) == 0:
        return out
    shape = np.asarray(voxels.shape)
    seeds = seeds[((seeds >= 0) & (seeds < shape)).all(axis=1)]
    if len(seeds):
        seeds = seeds[voxels[seeds[:, 0], seeds[:, 1], seeds[:, 2]] != 0]
    todo = voxels != 0
    # One flood per blob: the seeds a flood took drop out together
    # (Select Connected from a 700 000-cell selection looped over
    # every cell in Python).
    while len(seeds):
        seeds = seeds[~out[seeds[:, 0], seeds[:, 1], seeds[:, 2]]]
        if len(seeds) == 0:
            break
        x, y, z = (int(v) for v in seeds[0])
        _walk_runs(out, todo, (x, y, z), corners)
    return out


def _filled_seed(voxels: np.ndarray, seed: Cell) -> bool:
    """True when seed is inside the grid and on a filled cell."""
    nx, ny, nz = voxels.shape
    x, y, z = (int(seed[i]) for i in range(3))
    if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
        return False
    return int(voxels[x, y, z]) != 0


_FACE_LINES = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CORNER_LINES = tuple((dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dy, dz) != (0, 0))


def _walk_runs(out: np.ndarray, todo: np.ndarray, seed: Cell, corners: bool = False) -> np.ndarray:
    """Flood `todo` from seed along x runs into `out`. `todo` is consumed.
    Face to face, a run reaches the four lines beside it over its own length;
    with `corners` it reaches all eight lines around it, one cell further at
    both ends (an edge or a corner touches there)."""
    nx, ny, nz = todo.shape
    lines = _CORNER_LINES if corners else _FACE_LINES
    reach = 1 if corners else 0
    stack = [seed]
    while stack:
        x, y, z = stack.pop()
        line = todo[:, y, z]
        if not line[x]:
            continue
        x0, x1 = _run_around(line, x)
        out[x0:x1, y, z] = True
        line[x0:x1] = False
        a, b = max(0, x0 - reach), min(nx, x1 + reach)
        for dy, dz in lines:
            yn, zn = y + dy, z + dz
            if 0 <= yn < ny and 0 <= zn < nz:
                for s in _run_starts(todo[a:b, yn, zn]):
                    stack.append((a + int(s), yn, zn))
    return out


def inset_region(cells: np.ndarray, normal, width: int) -> np.ndarray:
    """(M, 3): the face region shrunk by a rim `width` cells wide, in its plane.

    A cell stays while its four in-plane neighbours are in the region, `width`
    times over. The region's bounding box edge counts as outside.
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if len(cells) == 0 or width <= 0:
        return cells.copy()
    a, _s = _axis_sign(normal)
    u, v = (i for i in range(3) if i != a)
    lo = cells.min(axis=0)
    hi = cells.max(axis=0)
    mask = np.zeros((hi[u] - lo[u] + 3, hi[v] - lo[v] + 3), dtype=bool)  # one cell of margin
    mask[cells[:, u] - lo[u] + 1, cells[:, v] - lo[v] + 1] = True
    for _ in range(int(width)):
        inner = mask.copy()
        inner[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        inner[0, :] = inner[-1, :] = False
        inner[:, 0] = inner[:, -1] = False
        mask = inner
        if not mask.any():
            break
    uv = np.argwhere(mask)
    out = np.zeros((len(uv), 3), dtype=np.int64)
    out[:, u] = uv[:, 0] + lo[u] - 1
    out[:, v] = uv[:, 1] + lo[v] - 1
    out[:, a] = int(cells[0, a])
    return out


def components(mask: np.ndarray) -> list[np.ndarray]:
    """The 6-connected parts of a bool mask, each as (N, 3) cells, largest first."""
    todo = np.array(mask, dtype=bool)  # a copy: the caller's mask stays as it was
    solid = todo.astype(np.uint8)
    parts = []
    while todo.any():
        seed = np.argwhere(todo)[0]
        part = connected_blobs(solid, seed.reshape(1, 3))
        parts.append(np.argwhere(part))
        todo &= ~part
    parts.sort(key=len, reverse=True)
    return parts

