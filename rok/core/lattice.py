"""The cage's maths: free-form deformation of a voxel block.

A box of control points surrounds the selection. The forward map is the
tensor-product spline volume F(u) = sum B_i(u_x) B_j(u_y) B_k(u_z) P[i, j, k],
with `B` the clamped B-spline basis of degree min(3, n - 1). Moving a point
bends the space inside the box. The voxels follow by an inverse lookup: every
target cell asks which rest cell lands on it, found with exactly four Newton
steps, so there are no holes and no invented colours.

The numeric kernel is fixed (plan, section 4) so that the numpy path here and
the Rust path (native/src/live.rs, `live_deform`) agree cell for cell. Both
work in f64, with the same basis recurrence, the same loop order over the
(degree + 1)^3 control points of a span, the same multiply order, the same
adjugate solve, the same start, step count and accept test. Details the plan
leaves open are pinned here and in the Rust file alike:

- The basis is evaluated on the span `floor(t * spans)` clamped to the valid
  spans, so a Newton step that leaves [0, 1] continues the end span's
  polynomial (no clamping of u).
- A step with |det J| under DET_FRACTION of the rest box's volume uses
  det = copysign(DET_FRACTION * volume, det) and is halved (DAMP).
- Cell centres are `cell + 0.5`; the start is `(x - lo) / size`.

Pure numpy: nothing from the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_DEGREE = 3
NEWTON_STEPS = 4
DET_FRACTION = 1e-3  # |det J| below this share of the rest box's volume is damped
DAMP = 0.5  # the factor a damped step is scaled by
ACCEPT_RESIDUAL = 0.5  # cells: max |F(u) - x| of an accepted solve
MIN_COUNT, MAX_COUNT = 2, 6
CHUNK = 1 << 16  # target cells solved per numpy pass


@dataclass
class Lattice:
    """The cage: the rest box `lo .. lo + size` (ints), the number of points
    per axis, and the points (nx, ny, nz, 3) as floats in grid units."""

    lo: tuple[int, int, int]
    size: tuple[int, int, int]
    counts: tuple[int, int, int]
    points: np.ndarray
    bernstein: bool = False  # one full-degree Bernstein span per axis (every point pulls the whole volume)

    def copy(self) -> "Lattice":
        return Lattice(self.lo, self.size, self.counts, self.points.copy(), self.bernstein)

    @property
    def hi(self) -> tuple[int, int, int]:
        """The last rest cell on each axis (inclusive)."""
        return tuple(int(l + s - 1) for l, s in zip(self.lo, self.size))


# --- the basis ----------------------------------------------------------------------------


def degree(n: int, bernstein: bool = False) -> int:
    """The basis degree: min(3, n - 1), or n - 1 with `bernstein` (one span)."""
    return int(n) - 1 if bernstein else min(MAX_DEGREE, int(n) - 1)


def knots(n: int, bernstein: bool = False) -> np.ndarray:
    """The clamped uniform knot vector for `n` points: p + 1 zeros, the
    interior knots j / spans, p + 1 ones. With `bernstein` the degree is n - 1,
    so there is no interior knot and the basis is the Bernstein polynomials."""
    p = degree(n, bernstein)
    spans = n - p
    inner = [j / spans for j in range(1, spans)]
    return np.array([0.0] * (p + 1) + inner + [1.0] * (p + 1), dtype=np.float64)


def greville(n: int, bernstein: bool = False) -> np.ndarray:
    """Where the `n` points sit at rest: the knot averages. With them F is the
    identity. For 2 to 4 points (one span: Bernstein) that is even spacing; at
    5 and 6 it is not, because a clamped spline's points lie nearer the ends."""
    p = degree(n, bernstein)
    U = knots(n, bernstein)
    return np.array([sum(U[i + 1 : i + p + 1]) / p for i in range(n)], dtype=np.float64)


def _span_first(n: int, t: np.ndarray, bernstein: bool = False) -> np.ndarray:
    spans = n - degree(n, bernstein)
    with np.errstate(invalid="ignore", over="ignore"):
        first = np.floor(t * spans)
    first = np.where(np.isfinite(first), first, 0.0)
    return np.clip(first, 0, spans - 1).astype(np.int64)


def _funs(U: np.ndarray, s: np.ndarray, t: np.ndarray, p: int) -> list[np.ndarray]:
    """The p + 1 nonzero basis functions of degree p on knot span `s` at `t`
    (Piegl and Tiller, A2.2), vectorised over t."""
    N = [np.ones_like(t)]
    left = [None] * (p + 1)
    right = [None] * (p + 1)
    for j in range(1, p + 1):
        left[j] = t - U[s + 1 - j]
        right[j] = U[s + j] - t
        saved = np.zeros_like(t)
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N.append(saved)
    return N


def basis(n: int, t, bernstein: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(first, values, derivatives) of the basis at each `t`: `first` is the
    index of the first of the p + 1 points that weigh in, `values` and
    `derivatives` are (N, p + 1). The compact form the kernel uses."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    p = degree(n, bernstein)
    U = knots(n, bernstein)
    first = _span_first(n, t, bernstein)
    s = first + p
    top = _funs(U, s, t, p)
    low = _funs(U, s, t, p - 1)
    d = []
    for a in range(p + 1):
        i = s - p + a
        t1 = low[a - 1] / (U[i + p] - U[i]) if a >= 1 else np.zeros_like(t)
        t2 = low[a] / (U[i + p + 1] - U[i + 1]) if a <= p - 1 else np.zeros_like(t)
        d.append(p * (t1 - t2))
    return first, np.stack(top, axis=1), np.stack(d, axis=1)


def weights(n: int, t, bernstein: bool = False) -> np.ndarray:
    """(N, n) weight of each of the `n` points at each `t` in [0, 1]."""
    first, vals, _ = basis(n, t, bernstein)
    return _scatter(n, first, vals)


def weights_d(n: int, t, bernstein: bool = False) -> np.ndarray:
    """(N, n) derivative of each point's weight."""
    first, _, ders = basis(n, t, bernstein)
    return _scatter(n, first, ders)


def _scatter(n: int, first: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.zeros((len(first), n), dtype=np.float64)
    rows = np.arange(len(first))
    for a in range(vals.shape[1]):
        out[rows, first + a] = vals[:, a]
    return out


# --- the lattice --------------------------------------------------------------------------


def rest_lattice(bounds, counts, bernstein: bool = False) -> Lattice:
    """The points at rest on the box `bounds = (lo, hi)` (inclusive cells), so
    the corner points sit at `lo` and at `hi + 1`. `bernstein` picks the
    full-degree basis (see `Lattice.bernstein`)."""
    lo = tuple(int(v) for v in bounds[0])
    hi = tuple(int(v) for v in bounds[1])
    counts = _counts(counts)
    size = tuple(h + 1 - l for l, h in zip(lo, hi))
    axes = [lo[a] + size[a] * greville(counts[a], bernstein) for a in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).astype(np.float64)
    return Lattice(lo, size, counts, pts, bool(bernstein))


def offset_lattice(lattice: Lattice, offsets) -> Lattice:
    """A copy of `lattice` with points moved by `offsets`, a dict of
    `"i,j,k": [dx, dy, dz]` (grid units); the format of the fixtures and of the
    MCP tool."""
    out = lattice.copy()
    for key, delta in offsets.items():
        i, j, k = (int(v) for v in str(key).split(","))
        if not (0 <= i < out.counts[0] and 0 <= j < out.counts[1] and 0 <= k < out.counts[2]):
            raise ValueError(f"no point {key!r} in a {out.counts} cage")
        out.points[i, j, k] += np.asarray(delta, dtype=np.float64)
    return out


def _counts(counts) -> tuple[int, int, int]:
    if isinstance(counts, (int, np.integer)):
        counts = (counts,) * 3
    out = tuple(int(c) for c in counts)
    if len(out) != 3 or any(c < MIN_COUNT or c > MAX_COUNT for c in out):
        raise ValueError(f"the cage takes {MIN_COUNT} to {MAX_COUNT} points per axis, got {counts!r}")
    return out


def point_id(i: int, j: int, k: int) -> str:
    return f"CAGE_{i}_{j}_{k}"


def parse_point_id(name) -> tuple[int, int, int] | None:
    """(i, j, k) of a point id, or None when `name` is not one."""
    if not isinstance(name, str) or not name.startswith("CAGE_"):
        return None
    parts = name[5:].split("_")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def point_ids(counts) -> list[str]:
    nx, ny, nz = _counts(counts)
    return [point_id(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]


def edge_points(counts) -> np.ndarray:
    """(nx, ny, nz) bool: the points on a box edge (corners and edge middles),
    which draw full size. Face centres and the centre draw smaller."""
    nx, ny, nz = _counts(counts)
    ends = [np.zeros(n, dtype=np.int64) for n in (nx, ny, nz)]
    for a, n in enumerate((nx, ny, nz)):
        ends[a][0] = ends[a][n - 1] = 1
    count = ends[0][:, None, None] + ends[1][None, :, None] + ends[2][None, None, :]
    return count >= 2


def edge_polylines(lattice: Lattice, segments: int = 8) -> list[np.ndarray]:
    """The cage's lines for the overlay: along each axis, the deformed lattice
    line through every row of points (the curve F at the rows' rest places),
    sampled `segments` times per span. A list of (M, 3) polylines."""
    counts = lattice.counts
    rest = [greville(n, lattice.bernstein) for n in counts]
    out = []
    for axis in range(3):
        n = counts[axis]
        t = np.linspace(0.0, 1.0, segments * (n - degree(n, lattice.bernstein)) + 1)
        b, c = [a for a in range(3) if a != axis]
        for vb in rest[b]:
            for vc in rest[c]:
                u = np.zeros((len(t), 3), dtype=np.float64)
                u[:, axis], u[:, b], u[:, c] = t, vb, vc
                out.append(_eval(lattice, u, jac=False)[0])
    return out


# --- evaluation ---------------------------------------------------------------------------


def _eval(lattice: Lattice, u: np.ndarray, jac: bool = True):
    """F(u) (N, 3) and, with `jac`, J (N, 3, 3) with J[:, a, b] = dF_a / du_b.
    The sum runs over a, b, c in order and multiplies (wx * wy) * wz first: the
    Rust kernel does the same."""
    counts = lattice.counts
    bx, by, bz = [basis(counts[a], u[:, a], lattice.bernstein) for a in range(3)]
    (fx, vx, dx), (fy, vy, dy), (fz, vz, dz) = bx, by, bz
    P = lattice.points
    N = len(u)
    F = np.zeros((N, 3), dtype=np.float64)
    J = np.zeros((N, 3, 3), dtype=np.float64) if jac else None
    for a in range(vx.shape[1]):
        for b in range(vy.shape[1]):
            for c in range(vz.shape[1]):
                p = P[fx + a, fy + b, fz + c]
                F += ((vx[:, a] * vy[:, b]) * vz[:, c])[:, None] * p
                if jac:
                    J[:, :, 0] += ((dx[:, a] * vy[:, b]) * vz[:, c])[:, None] * p
                    J[:, :, 1] += ((vx[:, a] * dy[:, b]) * vz[:, c])[:, None] * p
                    J[:, :, 2] += ((vx[:, a] * vy[:, b]) * dz[:, c])[:, None] * p
    return F, J


def forward(lattice: Lattice, x) -> np.ndarray:
    """The deformed place of each rest position `x` (N, 3), in grid units."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    u = (x - np.asarray(lattice.lo, dtype=np.float64)) / np.asarray(lattice.size, dtype=np.float64)
    return _eval(lattice, u, jac=False)[0]


def _solve(J: np.ndarray, r: np.ndarray, det_min: float) -> np.ndarray:
    """J^-1 r by the adjugate, damped when J is near singular."""
    m = [[J[:, i, j] for j in range(3)] for i in range(3)]
    c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2]
    c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02
    adj = (
        (c00, m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][1] * m[1][2] - m[0][2] * m[1][1]),
        (c01, m[0][0] * m[2][2] - m[0][2] * m[2][0], m[0][2] * m[1][0] - m[0][0] * m[1][2]),
        (c02, m[0][1] * m[2][0] - m[0][0] * m[2][1], m[0][0] * m[1][1] - m[0][1] * m[1][0]),
    )
    small = np.abs(det) < det_min
    det_used = np.where(small, np.copysign(det_min, det), det)
    factor = np.where(small, DAMP, 1.0)
    step = np.empty_like(r)
    for i in range(3):
        step[:, i] = (((adj[i][0] * r[:, 0] + adj[i][1] * r[:, 1]) + adj[i][2] * r[:, 2]) / det_used) * factor
    return step


def _det_min(lattice: Lattice) -> float:
    s = lattice.size
    return DET_FRACTION * float(s[0]) * float(s[1]) * float(s[2])


def newton(lattice: Lattice, x: np.ndarray, u0: np.ndarray, steps: int = NEWTON_STEPS) -> np.ndarray:
    """`steps` Newton steps of F(u) = x from `u0`, no early exit."""
    u = u0.copy()
    det_min = _det_min(lattice)
    with np.errstate(all="ignore"):
        for _ in range(steps):
            F, J = _eval(lattice, u)
            u = u - _solve(J, F - x, det_min)
    return u


def _accepted(lattice: Lattice, u: np.ndarray, x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        F = _eval(lattice, u, jac=False)[0]
        inside = np.all((u >= 0.0) & (u <= 1.0), axis=1)
        near = np.max(np.abs(F - x), axis=1) <= ACCEPT_RESIDUAL
    return inside & near


def inverse(lattice: Lattice, x) -> tuple[np.ndarray, np.ndarray]:
    """(u, ok) for each position `x` (N, 3): the rest-box coordinate that F
    sends to `x`, and whether the solve is accepted (u in [0, 1]^3, residual
    at most half a cell). A failed solve is tried once more from the centre of
    the span the start falls in."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    lo = np.asarray(lattice.lo, dtype=np.float64)
    size = np.asarray(lattice.size, dtype=np.float64)
    u0 = (x - lo) / size
    u = newton(lattice, x, u0)
    ok = _accepted(lattice, u, x)
    bad = np.flatnonzero(~ok)
    if len(bad):
        start = np.empty((len(bad), 3), dtype=np.float64)
        for a in range(3):
            spans = lattice.counts[a] - degree(lattice.counts[a], lattice.bernstein)
            first = _span_first(lattice.counts[a], u0[bad, a], lattice.bernstein)
            start[:, a] = (first + 0.5) / spans
        u2 = newton(lattice, x[bad], start)
        ok2 = _accepted(lattice, u2, x[bad])
        u[bad] = u2
        ok[bad] = ok2
    return u, ok


def rest_cells(lattice: Lattice, u: np.ndarray) -> np.ndarray:
    """The rest cell of each u (N, 3): floor(lo + u * size), the u = 1 edge
    clamped to the last cell (the free turn's rule). A u outside [0, 1] (a
    rejected solve) is clamped into the box too, so the index is always valid."""
    lo = np.asarray(lattice.lo, dtype=np.float64)
    size = np.asarray(lattice.size, dtype=np.float64)
    with np.errstate(all="ignore"):
        q = np.floor(lo + u * size)
    q = np.where(np.isfinite(q), q, lo)
    return np.clip(q, lo, np.asarray(lattice.hi, dtype=np.float64)).astype(np.int64)


# --- the cells a deformation visits -------------------------------------------------------


def target_boxes(lattice: Lattice, shape) -> np.ndarray:
    """(K, 6) int64 inclusive boxes (lo xyz, hi xyz), clipped to the grid: one
    per span of the volume, each the bounding box of that span's control
    points grown by one cell. The volume lies inside the convex hull of a
    span's points, so a cell outside every box can not be accepted. Empty when
    every box falls outside the grid."""
    counts = lattice.counts
    p = [degree(n, lattice.bernstein) for n in counts]
    spans = [n - q for n, q in zip(counts, p)]
    shape = np.asarray(shape, dtype=np.int64)
    out = []
    for i in range(spans[0]):
        for j in range(spans[1]):
            for k in range(spans[2]):
                block = lattice.points[i : i + p[0] + 1, j : j + p[1] + 1, k : k + p[2] + 1].reshape(-1, 3)
                lo = np.floor(block.min(axis=0)).astype(np.int64) - 1
                hi = np.ceil(block.max(axis=0)).astype(np.int64)
                lo = np.maximum(lo, 0)
                hi = np.minimum(hi, shape - 1)
                if np.all(hi >= lo):
                    out.append(np.concatenate([lo, hi]))
    return np.array(out, dtype=np.int64).reshape(-1, 6)


def deformed_bounds(lattice: Lattice, shape) -> tuple[np.ndarray, np.ndarray] | None:
    """The box of cells `deform_grid` can touch (inclusive, clipped to the
    grid): the union of the span boxes. None when it is outside the grid."""
    boxes = target_boxes(lattice, shape)
    if not len(boxes):
        return None
    return boxes[:, :3].min(axis=0), boxes[:, 3:].max(axis=0)


def target_cells(lattice: Lattice, shape) -> np.ndarray:
    """(M, 3) the cells in the union of the span boxes, in x, y, z order."""
    boxes = target_boxes(lattice, shape)
    if not len(boxes):
        return np.zeros((0, 3), dtype=np.int64)
    lo = boxes[:, :3].min(axis=0)
    hi = boxes[:, 3:].max(axis=0)
    mask = np.zeros(tuple(int(v) for v in hi - lo + 1), dtype=bool)
    for b in boxes:
        a, c = b[:3] - lo, b[3:] - lo + 1
        mask[a[0] : c[0], a[1] : c[1], a[2] : c[2]] = True
    return np.argwhere(mask) + lo


# --- the deformation ----------------------------------------------------------------------


_NATIVE: list = []


def _native():
    """rok_native's fused deformation, or None (no native module, or ROK_PURE_DEFORM set)."""
    if not _NATIVE:
        import os

        mod = None
        if not os.environ.get("ROK_PURE_DEFORM"):
            try:
                import rok_native

                mod = rok_native if hasattr(rok_native, "live_deform_fast") else None
            except ImportError:
                mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


def live_deform_grid(rest_voxels: np.ndarray, rest_mask: np.ndarray, lattice: Lattice, exact: bool = False):
    """(want, new mask): the grid under the cage, from the voxels and the selection
    at rest. One parallel pass in the native module when it is there; `deform_grid`
    is the reference. By default the native pass finds each cell's rest place by
    checked interpolation (live.rs `live_deform_fast`: the exact solve at the corners
    of 8-cell blocks, the blend verified at each block's centre, any block that fails
    solved cell by cell), which is what makes a dense 128 cube bend in under 30 ms; it
    differs from `deform_grid` only where a rest place lies within 0.02 cell of a rest
    cell's edge. `exact=True` runs the native kernel that equals `deform_grid` cell for
    cell. The shape of gizmo.live_turn_grid."""
    native = _native()
    if native is None:
        return deform_grid(rest_voxels, rest_mask, lattice)
    want = np.ascontiguousarray(rest_voxels, dtype=np.uint8)
    mask = np.ascontiguousarray(rest_mask, dtype=bool)
    boxes = np.ascontiguousarray(target_boxes(lattice, rest_voxels.shape), dtype=np.int64)
    run = native.live_deform if exact else native.live_deform_fast
    return run(
        want, mask, tuple(int(v) for v in lattice.lo), tuple(int(v) for v in lattice.size),
        tuple(int(v) for v in lattice.counts), np.ascontiguousarray(lattice.points, dtype=np.float64), boxes,
        bool(lattice.bernstein),
    )


def deform_grid(rest_voxels: np.ndarray, rest_mask: np.ndarray, lattice: Lattice):
    """(want, new mask): the grid after the deformation, from the rest voxels
    and the rest selection: the selection cleared, every target cell whose
    inverse lands on a selected, filled rest cell takes its colour, what falls
    outside the grid is lost. `new mask` is the cells written. The same
    contract as gizmo._live_turn_grid_numpy."""
    want = rest_voxels.copy()
    want[rest_mask] = 0
    new_mask = np.zeros(rest_mask.shape, dtype=bool)
    cells = target_cells(lattice, rest_voxels.shape)
    for start in range(0, len(cells), CHUNK):
        chunk = cells[start : start + CHUNK]
        u, ok = inverse(lattice, chunk.astype(np.float64) + 0.5)
        rc = rest_cells(lattice, u)
        r = (rc[:, 0], rc[:, 1], rc[:, 2])
        hit = ok & rest_mask[r] & (rest_voxels[r] != 0)
        t = chunk[hit]
        want[t[:, 0], t[:, 1], t[:, 2]] = rest_voxels[r][hit]
        new_mask[t[:, 0], t[:, 1], t[:, 2]] = True
    return want, new_mask
