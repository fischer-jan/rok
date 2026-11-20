"""Flat faces and their edges: the buckyball's planes, and Crease, which finds
the sharp edges of any voxel shape and raises bars on them.

The buckyball is the truncated icosahedron: 12 five-sided and 20 six-sided
faces. It stands with the coordinate planes as mirror planes, so it is
symmetric in x, y and z, and the top is a ridge along y where two six-sided
faces meet (the primitive turns it to run along the longer side). Its cells are those inside every face plane, so the faces are exactly
flat, and its frame bars come from the planes too: a cell near two planes at
once is on an edge.

Crease works on voxels alone. It splits the surface into flat faces by
fitting planes: every voxel of a flat face lies within a cell of one plane,
however the steps of the voxels run. Then it marks the voxels where two faces
meet at a sharp enough angle, and raises bars on them. Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product

import numpy as np

PHI = (1.0 + 5.0**0.5) / 2.0
FRAME_MIN = 6  # cells: a smaller box has no room for bars round a ball
FRAME_ANGLE = 10.0  # every edge of the ball gets a bar (its faces meet at 37 or 42 degrees)
FRAME_HEIGHT = 1.0  # the bars stand this many cells over the faces


def _even_permutations(v) -> list[tuple[float, float, float]]:
    x, y, z = v
    return [(x, y, z), (y, z, x), (z, x, y)]


@lru_cache(maxsize=1)
def bucky_vertices() -> np.ndarray:
    """(60, 3) the corners, scaled so the shape spans -1 .. 1 on every axis."""
    base = [(0.0, 1.0, 3.0 * PHI), (1.0, 2.0 + PHI, 2.0 * PHI), (PHI, 2.0, 2.0 * PHI + 1.0)]
    out = set()
    for v in base:
        for p in _even_permutations(v):
            for signs in product((-1.0, 1.0), repeat=3):
                out.add(tuple(round(s * c, 9) for s, c in zip(signs, p)))
    verts = np.array(sorted(out), dtype=np.float64)
    return verts / np.abs(verts).max()


@lru_cache(maxsize=1)
def bucky_planes() -> tuple[np.ndarray, np.ndarray]:
    """(normals (32, 3), offsets (32,)): a point p is inside when n . p <= d for
    every face. The five-sided faces face the icosahedron's corners, the
    six-sided ones the dodecahedron's."""
    dirs = set()
    for v in [(0.0, 1.0, PHI)]:
        for p in _even_permutations(v):
            for signs in product((-1.0, 1.0), repeat=3):
                dirs.add(tuple(round(s * c, 9) for s, c in zip(signs, p)))
    for v in [(1.0, 1.0, 1.0)] + _even_permutations((0.0, PHI, 1.0 / PHI)):
        for signs in product((-1.0, 1.0), repeat=3):
            dirs.add(tuple(round(s * c, 9) for s, c in zip(signs, v)))
    normals = np.array(sorted(dirs), dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = (bucky_vertices() @ normals.T).max(axis=0)
    return normals, offsets


def bucky_distances(q: np.ndarray, extent) -> np.ndarray:
    """(N, 32) each point's signed distance in cells to each face plane, for
    points `q` (N, 3) in the shape's own units (-1 .. 1 per axis) when the shape
    spans `extent` cells from its centre to its side along each axis."""
    normals, offsets = bucky_planes()
    e = np.asarray(extent, dtype=np.float64)
    scale = np.linalg.norm(normals / e, axis=1)  # one unit of n . q, in cells
    return (q @ normals.T - offsets) / scale


def bucky_grid(size, framed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(filled, frame) bool grids of `size`: the buckyball filling the box.
    With `framed`, the ball is one cell smaller on every side and bars one
    voxel high stand on its edges (`frame`), raised as Crease raises them, but
    from the exact faces, so the lines stay clean on a small ball. A box under
    FRAME_MIN on a side gets the plain ball."""
    size = tuple(int(v) for v in size)
    framed = framed and min(size) >= FRAME_MIN  # too small for bars round a ball: the plain ball
    e = np.array(size, dtype=np.float64) / 2.0
    inner = np.maximum(e - FRAME_HEIGHT, 0.5) if framed else e
    axes = [(np.arange(n) + 0.5 - n / 2.0) / inner[i] for i, n in enumerate(size)]
    q = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    dist = bucky_distances(q, inner)
    filled = (dist.max(axis=1) <= 1e-9).reshape(size)
    frame = np.zeros(size, dtype=bool)
    if framed and filled.any():
        cells = np.argwhere(surface_mask(filled))
        rows = np.ravel_multi_index(cells.T, size)
        normals, _offsets = bucky_planes()
        found = Faces(filled, cells, dist[rows].argmax(axis=1), normals)
        edge, edge_normals = found.edges(FRAME_ANGLE, 1)
        up = raise_cells(filled, edge, edge_normals)
        frame[up[:, 0], up[:, 1], up[:, 2]] = True
    return filled | frame, frame


# --- Crease ------------------------------------------------------------------------------

PLANE_REACH = 3  # cells: the neighbourhood a surface voxel's own plane is fitted to
PLANE_TOLERANCE = 0.75  # cells: how far a voxel's centre may lie off its face's plane
FACE_TURN = 20.0  # degrees: a voxel whose own plane turns more than this from the face's is not on it
MIN_FACE = 4  # a face of fewer voxels is no face; its voxels join the faces round them
NORMAL_SIGMA = 1.5  # the blur that tells outside from inside
RAISE_LEAN = 0.45  # a bar also grows along an axis the face normal leans out on this much
_N26 = np.array([d for d in product((-1, 0, 1), repeat=3) if d != (0, 0, 0)], dtype=np.int64)
_HALF26 = _N26[[i for i, d in enumerate(_N26) if tuple(d) > (0, 0, 0)]]  # one of each pair: 13
_N6 = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int64)


def surface_mask(filled: np.ndarray) -> np.ndarray:
    """Filled cells with an empty face neighbour (the grid's outside is empty)."""
    p = np.pad(filled, 1)
    inner = np.ones(filled.shape, dtype=bool)
    for d in _N6:
        sl = tuple(slice(1 + int(o), 1 + int(o) + n) for o, n in zip(d, filled.shape))
        inner &= p[sl]
    return filled & ~inner


def _gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur, one axis at a time; the grid's outside counts as empty."""
    k = int(np.ceil(3.0 * sigma))
    x = np.arange(-k, k + 1)
    w = np.exp(-x * x / (2.0 * sigma * sigma))
    w /= w.sum()
    out = a.astype(np.float64)
    for axis in range(3):
        pad = [(0, 0)] * 3
        pad[axis] = (k, k)
        p = np.pad(out, pad)
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, wi in enumerate(w):
            acc += wi * np.take(p, np.arange(i, i + n), axis=axis)
        out = acc
    return out


def _outward(filled: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """(N, 3) roughly outward directions at `cells`, from the blurred shape."""
    g = np.stack(np.gradient(_gauss(filled, NORMAL_SIGMA)), axis=-1)[cells[:, 0], cells[:, 1], cells[:, 2]]
    return -g


def local_planes(surf: np.ndarray, cells: np.ndarray, outward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(normals (N, 3), flatness (N,)): the plane through the surface voxels
    within PLANE_REACH of each cell, its normal turned outward; flatness is
    the spread off that plane (0 for a flat patch)."""
    r = PLANE_REACH
    offs = np.array([o for o in product(range(-r, r + 1), repeat=3) if sum(v * v for v in o) <= r * r], dtype=np.int64)
    count = np.zeros(len(cells))
    first = np.zeros((len(cells), 3))
    second = np.zeros((len(cells), 3, 3))
    sp = np.pad(surf, r)
    for o in offs:
        nb = cells + o + r
        m = sp[nb[:, 0], nb[:, 1], nb[:, 2]].astype(np.float64)
        count += m
        first += m[:, None] * o
        second += m[:, None, None] * np.outer(o, o)
    mean = first / count[:, None]
    cov = second / count[:, None, None] - mean[:, :, None] * mean[:, None, :]
    w, vec = np.linalg.eigh(cov)
    normal = vec[:, :, 0]
    normal *= np.where((normal * outward).sum(axis=1) < 0.0, -1.0, 1.0)[:, None]
    return normal, w[:, 0]


def _fit(points: np.ndarray, like: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(normal, centre) of the plane through `points`, the normal on the side of `like`."""
    c = points.mean(axis=0)
    _w, vec = np.linalg.eigh(np.cov((points - c).T))
    n = vec[:, 0]
    return (n if n @ like >= 0.0 else -n), c


def faces(filled: np.ndarray, limit: np.ndarray | None = None):
    """(cells (N, 3), face of each (N,), face normals (F, 3)): the surface
    voxels split into flat faces. The flattest voxel not yet taken starts a
    face: its plane takes the touching voxels that lie on it and turn with it,
    the plane is fitted again to them, and once more. A voxel that lies on no
    face of MIN_FACE voxels joins a face it touches (-1 if none)."""
    filled = np.asarray(filled, dtype=bool)
    surf = surface_mask(filled)
    if limit is not None:
        surf &= limit
    cells = np.argwhere(surf)
    if len(cells) == 0:
        return cells, np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    index = np.full(filled.shape, -1, dtype=np.int64)
    index[cells[:, 0], cells[:, 1], cells[:, 2]] = np.arange(len(cells))
    own, flatness = local_planes(surf, cells, _outward(filled, cells))
    points = cells.astype(np.float64)
    label = np.full(len(cells), -1, dtype=np.int64)
    normals: list[np.ndarray] = []
    turn = np.cos(np.radians(FACE_TURN))
    shape = np.asarray(filled.shape)
    for seed in np.argsort(flatness, kind="stable"):
        if label[seed] >= 0:
            continue
        n, c = own[seed], points[seed]
        taken = np.array([seed])
        for _round in range(3):
            # Grow from the seed through touching voxels on the plane.
            reach = np.zeros(len(cells), dtype=bool)
            reach[seed] = True
            front = np.array([seed])
            while len(front):
                nb = (cells[front][:, None, :] + _N26[None]).reshape(-1, 3)
                nb = nb[np.all((nb >= 0) & (nb < shape), axis=1)]
                j = index[nb[:, 0], nb[:, 1], nb[:, 2]]
                j = np.unique(j[j >= 0])
                j = j[~reach[j] & (label[j] < 0)]
                j = j[(np.abs((points[j] - c) @ n) <= PLANE_TOLERANCE) & (own[j] @ n >= turn)]
                reach[j] = True
                front = j
            taken = np.nonzero(reach)[0]
            if len(taken) < MIN_FACE:
                break
            n, c = _fit(points[taken], n)
        if len(taken) < MIN_FACE:
            continue
        label[taken] = len(normals)
        normals.append(n)
    # Strays join a face they touch, a few rounds.
    a, b = _pairs(index, cells)
    for _round in range(4):
        stray = label < 0
        if not stray.any():
            break
        give = stray[a] & ~stray[b]
        label[a[give]] = label[b[give]]
        give = stray[b] & ~stray[a] & (label[b] < 0)
        label[b[give]] = label[a[give]]
    label, kept = _dissolve_strips(points, label, np.array(normals).reshape(-1, 3), a, b)
    return cells, label, kept


STRIP_SHARE = 0.9  # a face is a strip when this share of its voxels lies on a bigger neighbour's plane
STRIP_EDGE = 0.45  # ... and this share on two such planes at once: along the edge where they meet


def _dissolve_strips(points: np.ndarray, label: np.ndarray, normals: np.ndarray,
                     a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(label, normals) with the strips gone. The voxels along an edge see both
    faces round them, so their own planes lean half way and fit neither face;
    they grow into thin faces of their own, and a bar test across such a strip
    sees two small turns instead of one sharp one. A face, smallest first, whose
    voxels lie on the planes of bigger faces it touches (STRIP_SHARE), many of
    them on two at once (STRIP_EDGE), is dissolved: each voxel joins the touching
    face whose plane it lies nearest. A small real face, such as a ball's
    pentagon, lies on its neighbours' planes only round its rim."""
    count = len(normals)
    if count < 2:
        return label, normals
    size = np.bincount(label[label >= 0], minlength=count)
    centre = np.zeros((count, 3))
    np.add.at(centre, label[label >= 0], points[label >= 0])
    centre /= np.maximum(size, 1)[:, None]
    la, lb = label[a], label[b]
    touch = (la >= 0) & (lb >= 0) & (la != lb)
    nbrs: list[set[int]] = [set() for _ in range(count)]
    for f, g in zip(la[touch].tolist(), lb[touch].tolist()):
        nbrs[f].add(g)
        nbrs[g].add(f)
    alive = size > 0
    for f in np.argsort(size, kind="stable"):
        if not alive[f]:
            continue
        bigger = [g for g in nbrs[f] if alive[g] and size[g] > size[f]]
        if not bigger:
            continue
        rows = np.nonzero(label == f)[0]
        g = np.array(bigger)
        dist = np.abs(((points[rows][:, None, :] - centre[g][None]) * normals[g][None]).sum(axis=2))
        near = (dist <= PLANE_TOLERANCE).sum(axis=1)
        if (near >= 1).mean() < STRIP_SHARE or (near >= 2).mean() < STRIP_EDGE:
            continue
        best = dist.argmin(axis=1)
        label[rows] = g[best]
        np.add.at(size, g[best], 1)
        alive[f] = False
        for h in nbrs[f]:
            nbrs[h].discard(f)
            nbrs[h].update(x for x in nbrs[f] if x != h)
    remap = np.cumsum(alive) - 1
    label = np.where(label >= 0, remap[np.maximum(label, 0)], -1)
    return label, normals[alive]


def _pairs(index: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a, b): every pair of listed cells that touch (26 ways), as row numbers."""
    shape = np.asarray(index.shape)
    a_all, b_all = [], []
    for d in _HALF26:
        nb = cells + d
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        j = np.full(len(cells), -1, dtype=np.int64)
        j[ok] = index[nb[ok, 0], nb[ok, 1], nb[ok, 2]]
        hit = j >= 0
        a_all.append(np.nonzero(hit)[0])
        b_all.append(j[hit])
    return np.concatenate(a_all), np.concatenate(b_all)


QUADRIC_MIN = 12  # a face needs this many voxels before its bend is fitted; smaller ones stay flat


def _frame_of(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors across the normal `n`."""
    t1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def bends(cells: np.ndarray, label: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """(F, 13) for every face its centre (3), its two across vectors (6) and the
    bend h = a u^2 + b uv + c v^2 + d u + e v (4 of these, then d and e): the
    surface's height over the face's plane, fitted to its voxels. A flat face
    has no bend; a patch of a ball bends with the ball."""
    out = np.zeros((len(normals), 14))
    order = np.argsort(label, kind="stable")
    starts = np.searchsorted(label[order], np.arange(len(normals) + 1))
    points = cells.astype(np.float64)
    for f in range(len(normals)):
        rows = order[starts[f]:starts[f + 1]]
        n = normals[f]
        t1, t2 = _frame_of(n)
        c = points[rows].mean(axis=0) if len(rows) else np.zeros(3)
        out[f, :3], out[f, 3:6], out[f, 6:9] = c, t1, t2
        if len(rows) < QUADRIC_MIN:
            continue
        d = points[rows] - c
        u, v, h = d @ t1, d @ t2, d @ n
        m = np.stack([u * u, u * v, v * v, u, v, np.ones(len(rows))], axis=1)
        coef, *_ = np.linalg.lstsq(m, h, rcond=None)
        out[f, 9:14] = coef[:5]
    return out


def bent_normals(bend: np.ndarray, normals: np.ndarray, faces_of: np.ndarray, at: np.ndarray) -> np.ndarray:
    """(N, 3) the surface normal of face `faces_of[i]` at the point `at[i]`, from its bend."""
    b = bend[faces_of]
    d = at - b[:, :3]
    t1, t2 = b[:, 3:6], b[:, 6:9]
    u = (d * t1).sum(axis=1)
    v = (d * t2).sum(axis=1)
    hu = 2.0 * b[:, 9] * u + b[:, 10] * v + b[:, 12]
    hv = b[:, 10] * u + 2.0 * b[:, 11] * v + b[:, 13]
    n = normals[faces_of] - hu[:, None] * t1 - hv[:, None] * t2
    return n / np.linalg.norm(n, axis=1, keepdims=True)


@dataclass
class Faces:
    """A model's surface split into flat faces (`faces`), kept so that Crease's
    angle and width can change without splitting it again. `bend` (see `bends`)
    lets a curved surface, split into patches, show no edge where it is smooth."""

    filled: np.ndarray
    cells: np.ndarray
    label: np.ndarray
    normals: np.ndarray
    bend: np.ndarray | None = None

    @classmethod
    def of(cls, voxels: np.ndarray, limit: np.ndarray | None = None) -> "Faces":
        filled = np.asarray(voxels) != 0
        cells, label, normals = faces(filled, limit)
        return cls(filled, cells, label, normals, bends(cells, label, normals))

    def edges(self, angle: float = 20.0, width: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(edge cells (N, 3), their faces' normals (N, 3)): the voxels where two
        faces meet at `angle` degrees or more, measured where they meet (each
        face's surface normal there, from its bend). `width` 1 takes one side
        of each edge (the same side all along it, see `_side`), 2 both sides."""
        if len(self.normals) < 2:
            return self.cells[:0], np.zeros((0, 3))
        index = np.full(self.filled.shape, -1, dtype=np.int64)
        index[self.cells[:, 0], self.cells[:, 1], self.cells[:, 2]] = np.arange(len(self.cells))
        a, b = _pairs(index, self.cells)
        la, lb = self.label[a], self.label[b]
        meet = (la >= 0) & (lb >= 0) & (la != lb)
        a, b, la, lb = a[meet], b[meet], la[meet], lb[meet]
        if self.bend is None:
            na, nb = self.normals[la], self.normals[lb]
        else:
            mid = (self.cells[a] + self.cells[b]) / 2.0
            na = bent_normals(self.bend, self.normals, la, mid)
            nb = bent_normals(self.bend, self.normals, lb, mid)
        sharp = (na * nb).sum(axis=1) <= np.cos(np.radians(angle))
        edge = np.zeros(len(self.cells), dtype=bool)
        if width >= 2:
            edge[a[sharp]] = True
            edge[b[sharp]] = True
        else:
            side = _side(self.normals[la], self.normals[lb])
            edge[a[sharp & (side >= 0)]] = True
            edge[b[sharp & (side <= 0)]] = True
        return self.cells[edge], self.normals[self.label[edge]]


def _side(na: np.ndarray, nb: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
    """(N,) +1 where a bar of width 1 goes on face a, -1 on face b, 0 on both:
    the face that looks more up or down, then more along y, then along x.
    Only sizes count, not signs, so a model symmetric in an axis gets
    symmetric bars."""
    out = np.zeros(len(na), dtype=np.int64)
    d = np.abs(na) - np.abs(nb)
    for axis in (2, 1, 0):
        open_ = out == 0
        out[open_ & (d[:, axis] > tolerance)] = 1
        out[open_ & (d[:, axis] < -tolerance)] = -1
    return out


def raise_cells(filled: np.ndarray, cells: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """(M, 3) the empty cells just outside `cells`, bars one voxel high over the
    edges: the face neighbour along the axis the normal leans out on most, and
    any other the normal leans out on clearly (RAISE_LEAN)."""
    out = []
    shape = np.asarray(filled.shape)
    lean = normals @ _N6.T  # (N, 6)
    most = lean.argmax(axis=1) if len(normals) else np.zeros(0, dtype=np.int64)
    for k, d in enumerate(_N6):
        keep = (most == k) | (lean[:, k] > RAISE_LEAN)
        nb = cells[keep] + d
        nb = nb[np.all((nb >= 0) & (nb < shape), axis=1)]
        out.append(nb[~filled[nb[:, 0], nb[:, 1], nb[:, 2]]])
    cat = np.concatenate(out)
    return np.unique(cat, axis=0) if len(cat) else cat


def crease(voxels: np.ndarray, colour: int, angle: float = 20.0, width: int = 1, raised: bool = True,
           limit: np.ndarray | None = None, found: Faces | None = None,
           mirror=(False, False, False), paint: bool | None = None) -> np.ndarray:
    """A copy of `voxels` with bars in `colour` on the sharp edges: raised one
    voxel over them (`raised`), the edge voxels painted (`paint`, by default
    when not raised), or both. `found` is the split into faces when the caller
    has it already. On each `mirror` axis every bar also goes on its mirror
    image, so a symmetric model gets symmetric bars, as a brush stroke repeats
    on the other side."""
    found = found if found is not None else Faces.of(voxels, limit)
    paint = not raised if paint is None else paint
    out = np.array(voxels, copy=True)
    edge, normals = found.edges(angle, width)
    bars = []  # (cells, whether they go on empty cells)
    if raised:
        bars.append((raise_cells(found.filled, edge, normals), True))
    if paint:
        bars.append((edge, False))
    for cells, over in bars:
        if any(mirror) and len(cells):
            from .edit import mirror_cells

            cells = mirror_cells(cells, voxels.shape, tuple(bool(m) for m in mirror))
            cells = cells[np.all((cells >= 0) & (cells < np.asarray(voxels.shape)), axis=1)]
            here = found.filled[cells[:, 0], cells[:, 1], cells[:, 2]]
            cells = cells[here != over]  # a raised bar's image goes on an empty cell, a painted one on a voxel
            if limit is not None and not over:
                cells = cells[limit[cells[:, 0], cells[:, 1], cells[:, 2]]]
        if len(cells):
            out[cells[:, 0], cells[:, 1], cells[:, 2]] = colour
    return out
