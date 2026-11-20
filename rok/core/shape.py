"""Footprints for the Box, Line and Center modes, and the axis drag helper.

Imports numpy and .edit only. All cell functions return (N, 3) int64 arrays.
"""

from __future__ import annotations

import numpy as np

from .edit import BOX, CENTER, LINE, Cell, line_cells

_DRAG_EPS = 1e-9


def _free_axes(axis: int) -> tuple[int, int]:
    u, v = (i for i in range(3) if i != axis)
    return u, v


def _plane_cells(axis: int, fixed: int, u_range, v_range, u: int, v: int) -> np.ndarray:
    """All cells of u_range x v_range at coordinate[axis] == fixed, row-major."""
    us = np.arange(u_range[0], u_range[1] + 1, dtype=np.int64)
    vs = np.arange(v_range[0], v_range[1] + 1, dtype=np.int64)
    if us.size == 0 or vs.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    gu, gv = np.meshgrid(us, vs, indexing="ij")
    cells = np.empty((us.size * vs.size, 3), dtype=np.int64)
    cells[:, axis] = int(fixed)
    cells[:, u] = gu.ravel()
    cells[:, v] = gv.ravel()
    return cells


def rect_cells(a: Cell, b: Cell, axis: int) -> np.ndarray:
    """Every cell of the inclusive rectangle between a and b, at a[axis].

    Any corner order. Row-major in the two free axes (lower axis index outer).
    """
    u, v = _free_axes(axis)
    u_range = (min(int(a[u]), int(b[u])), max(int(a[u]), int(b[u])))
    v_range = (min(int(a[v]), int(b[v])), max(int(a[v]), int(b[v])))
    return _plane_cells(axis, int(a[axis]), u_range, v_range, u, v)


def center_cells(c: Cell, b: Cell, axis: int, size) -> np.ndarray:
    """Rectangle around c with half extents |b - c| per free axis, clipped to the grid."""
    u, v = _free_axes(axis)
    du = abs(int(b[u]) - int(c[u]))
    dv = abs(int(b[v]) - int(c[v]))
    u_range = (max(0, int(c[u]) - du), min(int(size[u]) - 1, int(c[u]) + du))
    v_range = (max(0, int(c[v]) - dv), min(int(size[v]) - 1, int(c[v]) + dv))
    return _plane_cells(axis, int(c[axis]), u_range, v_range, u, v)


def line_shape(a: Cell, b: Cell, axis: int) -> np.ndarray:
    """[a] + edit.line_cells(a, b, axis) as an array. One cell when a == b."""
    cells = [tuple(int(x) for x in a)] + line_cells(a, b, axis)
    return np.array(cells, dtype=np.int64).reshape(-1, 3)


def line3d_cells(a: Cell, b: Cell) -> np.ndarray:
    """3D Bresenham from a to b, both included, in order. One cell when a == b.

    Steps every axis by its own error term, so a diagonal jump takes one step
    and the line has no gaps in any direction.
    """
    x = [int(v) for v in a]
    end = [int(v) for v in b]
    delta = [abs(end[i] - x[i]) for i in range(3)]
    step = [1 if end[i] > x[i] else -1 for i in range(3)]
    longest = max(delta)
    cells = [tuple(x)]
    if longest == 0:
        return np.array(cells, dtype=np.int64).reshape(-1, 3)
    err = [2 * delta[i] - longest for i in range(3)]
    for _ in range(longest):
        for i in range(3):
            if err[i] >= 0:
                x[i] += step[i]
                err[i] -= 2 * longest
            err[i] += 2 * delta[i]
        cells.append(tuple(x))
    return np.array(cells, dtype=np.int64).reshape(-1, 3)


def shape_cells(mode: str, a: Cell, b: Cell, axis: int, size) -> np.ndarray:
    """The footprint of a shape mode from its anchor a to the current cell b."""
    if mode == BOX:
        return rect_cells(a, b, axis)
    if mode == CENTER:
        return center_cells(a, b, axis, size)
    if mode == LINE:
        return line3d_cells(a, b)  # the end follows the surface; axis is not used
    raise ValueError(f"not a shape mode: {mode!r}")


def shape_label(mode: str, a: Cell, b: Cell, axis: int) -> str:
    """HUD text for a shape: BOX "w x h", CENTER the full extents, LINE the cell count."""
    u, v = _free_axes(axis)
    du = abs(int(b[u]) - int(a[u]))
    dv = abs(int(b[v]) - int(a[v]))
    if mode == BOX:
        return f"{du + 1} x {dv + 1}"
    if mode == CENTER:
        return f"{2 * du + 1} x {2 * dv + 1}"
    if mode == LINE:
        return str(int(line3d_cells(a, b).shape[0]))
    raise ValueError(f"not a shape mode: {mode!r}")


# The solids a Box or Center footprint can grow into during the height stage:
# the primitives. Each fills the box of the footprint and the height.
BOX_SOLID = "BOX"
CYLINDER = "CYLINDER"
PYRAMID = "PYRAMID"
CONE = "CONE"
SPHERE = "SPHERE"
DOME = "DOME"
WEDGE = "WEDGE"
ROOF = "ROOF"
TUBE = "TUBE"
HEX_PRISM = "HEX_PRISM"
TORUS = "TORUS"
STAIRS = "STAIRS"
CAPSULE = "CAPSULE"
HALF_CYLINDER = "HALF_CYLINDER"
DIAMOND = "DIAMOND"
OCTAGON_PRISM = "OCTAGON_PRISM"
ROCK = "ROCK"
HILL = "HILL"
GEAR = "GEAR"
SPIRAL_RAMP = "SPIRAL_RAMP"
TOWER = "TOWER"
BUCKYBALL = "BUCKYBALL"
FRAMED_BUCKYBALL = "FRAMED_BUCKYBALL"
# The grid order, six across: round, then pointed and angular, then the rest.
SOLID_ITEMS = (
    (BOX_SOLID, "Box", "Straight sides, the footprint all the way up"),
    (CYLINDER, "Cylinder", "Round: the ellipse inside the footprint, all the way up"),
    (SPHERE, "Sphere", "A ball that fills the box"),
    (DOME, "Dome", "The top half of a ball, flat on the footprint"),
    (BUCKYBALL, "Buckyball", "A ball of 32 flat faces, five- and six-sided, its top ridge along the longer side"),
    (FRAMED_BUCKYBALL, "Framed Buckyball",
     "A buckyball with raised bars on every edge, in a darker shade: flat panes in a frame, as on a canopy"),
    (CAPSULE, "Capsule", "A pill along the longest side of the box, round at both ends"),
    (HALF_CYLINDER, "Half Cylinder", "A barrel vault lying along the longer side"),
    (PYRAMID, "Pyramid", "The footprint tapering to a point at the top"),
    (CONE, "Cone", "The ellipse inside the footprint tapering to a point at the top"),
    (DIAMOND, "Diamond", "A double pyramid, pointed at the top and the bottom"),
    (OCTAGON_PRISM, "Octagon Prism", "An eight-sided column"),
    (HEX_PRISM, "Hex Prism", "A six-sided column"),
    (TUBE, "Tube", "A cylinder with a wall one voxel thick"),
    (WEDGE, "Wedge", "A ramp rising along the longer side"),
    (ROOF, "Roof", "A ridge along the longer side, sloping down both ways"),
    (STAIRS, "Stairs", "Steps rising along the longer side"),
    (TORUS, "Torus", "A ring with a hole in the middle, filling the box"),
    (ROCK, "Rock", "A lumpy boulder, different on every drag"),
    (HILL, "Hill", "A lumpy mound, different on every drag"),
    (GEAR, "Gear", "A cog wheel with teeth round the rim and a hole for the axle"),
    (SPIRAL_RAMP, "Spiral Ramp", "A ramp that winds up round a centre column"),
    (TOWER, "Tower", "A round tower, a little narrower at the top, with battlements"),
)
LUMPY = (ROCK, HILL)  # take a seed: a new lump pattern on every drag
FACETED = (BUCKYBALL, FRAMED_BUCKYBALL)  # built whole from face planes, not layer by layer
WALL_MAX = 4  # the thickest hollow wall, in voxels
SOLIDS = tuple(ident for ident, _name, _desc in SOLID_ITEMS)
_EPS = 1e-9
TORUS_RING = 0.5  # the ring's width over the half-width; the hole is the other half


def square_corner(a: Cell, b: Cell, axis: int) -> Cell:
    """The far corner moved so the footprint is square: both free extents become the
    larger one, each keeping its direction from `a`."""
    u, v = _free_axes(axis)
    du = int(b[u]) - int(a[u])
    dv = int(b[v]) - int(a[v])
    m = max(abs(du), abs(dv))
    out = list(int(c) for c in b)
    out[u] = int(a[u]) + (m if du >= 0 else -m)
    out[v] = int(a[v]) + (m if dv >= 0 else -m)
    return (out[0], out[1], out[2])


def solid_cells(
    footprint: np.ndarray, axis: int, step: int, height: int, kind: str, size,
    wall: int = 0, seed: int = 0, along: int | None = None,
) -> np.ndarray:
    """The cells of a solid grown from a rectangular footprint (N, 3) in the layer
    across `axis`, `height` layers along `step` (+1 or -1): the primitive `kind`
    (one of SOLIDS) filling the box of the footprint and the height. Clipped to
    the grid.

    `wall` above 0 hollows it: only the cells within `wall` steps of its
    outside stay, top and bottom included. `seed` picks the lump pattern of
    a Rock or a Hill. `along` (a grid axis across `axis`) is the way a shape
    with a run (Half Cylinder, Wedge, Roof, Stairs, Hex Prism) runs; None runs
    it along the footprint's longer side."""
    return solid_parts(footprint, axis, step, height, kind, size, wall, seed, along)[0]


def solid_parts(
    footprint: np.ndarray, axis: int, step: int, height: int, kind: str, size,
    wall: int = 0, seed: int = 0, along: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(cells, frame): `solid_cells`, and for each cell whether it is a bar of
    the frame (the Framed Buckyball's; no other solid has one)."""
    footprint = np.asarray(footprint, dtype=np.int64).reshape(-1, 3)
    if footprint.shape[0] == 0:
        return footprint, np.zeros(0, dtype=bool)
    height = max(1, int(height))
    u, v = _free_axes(axis)
    lo = footprint.min(axis=0)
    hi = footprint.max(axis=0)
    # The whole bounding rectangle, once; each layer keeps what its shape allows.
    us = np.arange(lo[u], hi[u] + 1)
    vs = np.arange(lo[v], hi[v] + 1)
    gu, gv = np.meshgrid(us, vs, indexing="ij")
    gu = gu.ravel()
    gv = gv.ravel()
    frame_block = None
    if kind in FACETED:
        from .facets import bucky_grid

        # The ball's top ridge runs along its second axis: turn it to run along the longer side
        nu, nv = len(us), len(vs)
        ridge_u = nu > nv if along is None else along == u
        filled, frame = bucky_grid((nv, nu, height) if ridge_u else (nu, nv, height), kind == FRAMED_BUCKYBALL)
        if ridge_u:
            filled, frame = filled.transpose(1, 0, 2), frame.transpose(1, 0, 2)
        block = filled.transpose(2, 0, 1).reshape(height, -1)  # (layer, cells of the rectangle)
        frame_block = frame.transpose(2, 0, 1).reshape(height, -1)
    else:
        keep_layer = _solid_rule(
            kind, gu - lo[u], gv - lo[v], int(hi[u] - lo[u] + 1), int(hi[v] - lo[v] + 1), height, seed,
            None if along is None else along == u,
        )
        block = np.stack([keep_layer(k) for k in range(height)])  # (height, cells of the rectangle)
    wall = max(0, min(int(wall), WALL_MAX))
    if wall and block.any():
        from .morph import erode  # the same shell as the Hollow command: one face step per voxel

        shaped = block.reshape(height, len(us), len(vs))
        block = (shaped & ~erode(shaped, wall)).reshape(height, -1)
    # Every layer at once (a loop over the layers cost a big box's preview 7 ms a move).
    ks, idx = np.nonzero(block)
    if ks.size == 0:
        return footprint[:0], np.zeros(0, dtype=bool)
    out = np.empty((ks.size, 3), dtype=np.int64)
    out[:, u] = gu[idx]
    out[:, v] = gv[idx]
    out[:, axis] = int(lo[axis]) + int(step) * ks
    bars = frame_block[ks, idx] if frame_block is not None else np.zeros(ks.size, dtype=bool)
    size = np.asarray(size, dtype=np.int64)
    # Clip to the grid only when the block can reach past it: the test over
    # every cell is the dearest step for a big block.
    top = int(lo[axis]) + int(step) * (height - 1)
    if lo.min() >= 0 and (hi < size).all() and 0 <= top < int(size[axis]):
        return out, bars
    inside = np.all((out >= 0) & (out < size), axis=1)
    return out[inside], bars[inside]


def solid_grid(kind: str, size, colour: int, seed: int = 0, frame_colour: int | None = None) -> np.ndarray:
    """A grid of `size` that holds the primitive `kind` filling it, standing on
    its bottom layer, in `colour` (a frame's bars in `frame_colour`): a new
    model of that shape."""
    sx, sy, sz = (int(v) for v in size)
    gx, gy = np.meshgrid(np.arange(sx), np.arange(sy), indexing="ij")
    footprint = np.stack([gx.ravel(), gy.ravel(), np.zeros(sx * sy, dtype=np.int64)], axis=1)
    cells, bars = solid_parts(footprint, 2, 1, sz, kind, (sx, sy, sz), seed=seed)
    voxels = np.zeros((sx, sy, sz), dtype=np.uint8)
    voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = colour
    if frame_colour is not None and bars.any():
        f = cells[bars]
        voxels[f[:, 0], f[:, 1], f[:, 2]] = frame_colour
    return voxels


def _solid_rule(kind: str, pu: np.ndarray, pv: np.ndarray, nu: int, nv: int, height: int, seed: int = 0,
                along_u: bool | None = None):
    """keep(k): the bool mask of the footprint cells that layer k of the solid keeps.

    pu, pv are the cells' places in the footprint (0 .. nu - 1, 0 .. nv - 1).
    Distances are from cell centres; an extent is half the width plus half a
    voxel, so a round shape touches the footprint's edge cells. `along_u` says
    whether a shape with a run runs along u; None picks the longer side."""
    du = pu - (nu - 1) / 2.0
    dv = pv - (nv - 1) / 2.0
    hu = (nu - 1) / 2.0
    hv = (nv - 1) / 2.0
    eu = hu + 0.5
    ev = hv + 0.5
    ellipse = (du / eu) ** 2 + (dv / ev) ** 2  # 1 at the footprint's edge
    # The run goes along "l", across "s" (wedge, roof, stairs, hex, half cylinder):
    # the longer side unless the caller names one.
    long_u = nu >= nv if along_u is None else bool(along_u)
    pl, nl = (pu, nu) if long_u else (pv, nv)
    ds, hs = (dv, hv) if long_u else (du, hu)
    dl, el = (du, eu) if long_u else (dv, ev)
    es = hs + 0.5
    full = np.ones(pu.shape, dtype=bool)

    def taper(k: int) -> float:
        """1 at the base, 0 at the top layer."""
        return 1.0 if height == 1 else (height - 1 - k) / (height - 1)

    def mid(k: int) -> float:
        """The layer's centre height over the height, 0 .. 1."""
        return (k + 0.5) / height

    if kind == CYLINDER:
        return lambda k: ellipse <= 1.0 + _EPS
    if kind == PYRAMID:
        def pyramid(k):
            t = taper(k)
            return (np.abs(du) <= hu * t + 0.5 + _EPS) & (np.abs(dv) <= hv * t + 0.5 + _EPS)
        return pyramid
    if kind == CONE:
        def cone(k):
            t = taper(k)
            return (du / (hu * t + 0.5)) ** 2 + (dv / (hv * t + 0.5)) ** 2 <= 1.0 + _EPS
        return cone
    if kind == SPHERE:
        return lambda k: ellipse + ((mid(k) - 0.5) * 2.0) ** 2 <= 1.0 + _EPS
    if kind == DOME:
        return lambda k: ellipse + mid(k) ** 2 <= 1.0 + _EPS
    if kind == WEDGE:
        # Column l rises to (l + 1) / nl of the height: the back column is full height.
        top = (pl + 1) / nl * height
        return lambda k: k + 0.5 <= top + _EPS
    if kind == ROOF:
        return lambda k: np.abs(ds) <= hs * taper(k) + 0.5 + _EPS
    if kind == TUBE:
        if nu <= 2 or nv <= 2:
            return lambda k: ellipse <= 1.0 + _EPS  # too thin for a hole
        inner = (du / (eu - 1.0)) ** 2 + (dv / (ev - 1.0)) ** 2
        ring = (ellipse <= 1.0 + _EPS) & (inner > 1.0 + _EPS)
        return lambda k: ring
    if kind == HEX_PRISM:
        # Flat on the longer sides, pointed at the ends of the longer axis.
        hexagon = (np.abs(ds) <= es + _EPS) & (np.abs(dl) / el + 0.5 * np.abs(ds) / es <= 1.0 + _EPS)
        return lambda k: hexagon
    if kind == TORUS:
        # A ring filling the box: its cross-section is an oval as tall as the
        # height and TORUS_RING of the half-width across, out to the footprint's
        # edge, so the hole is the rest of the way to the centre.
        if nu < 3 or nv < 3:
            return lambda k: ellipse <= 1.0 + _EPS  # too small for a hole: a disc, never nothing
        rho = np.sqrt(ellipse)  # 0 at the centre, 1 at the footprint's edge
        half = TORUS_RING / 2.0
        centre = 1.0 - half

        def torus(k):
            dz = ((k + 0.5) - height / 2.0) / (height / 2.0)
            return ((rho - centre) / half) ** 2 + dz**2 <= 1.0 + _EPS
        return torus
    if kind == CAPSULE:
        # Along the longest of the three sides; round ends as wide as the other two.
        ez = height / 2.0
        dz = lambda k: (k + 0.5) - ez  # noqa: E731
        spans = {"u": eu, "v": ev, "z": ez}
        along = max(spans, key=spans.get)
        cross = [a for a in spans if a != along]
        r = min(spans[a] for a in cross)

        def capsule(k):
            d = {"u": du, "v": dv, "z": np.full(du.shape, dz(k))}
            beyond = np.maximum(0.0, np.abs(d[along]) - (spans[along] - r)) / max(r, 1e-9)
            return (d[cross[0]] / spans[cross[0]]) ** 2 + (d[cross[1]] / spans[cross[1]]) ** 2 + beyond**2 <= 1.0 + _EPS
        return capsule
    if kind == HALF_CYLINDER:
        # Its round top runs along the longer side; the flat side on the footprint.
        return lambda k: (ds / es) ** 2 + mid(k) ** 2 <= 1.0 + _EPS
    if kind == DIAMOND:
        # Measured from each cell's edge nearest the centre, so an even base
        # (its centre between cells) keeps its middle cells.
        near_u = np.maximum(0.0, np.abs(du) - 0.5) / max(hu, 0.5)
        near_v = np.maximum(0.0, np.abs(dv) - 0.5) / max(hv, 0.5)
        hz = (height - 1) / 2.0

        def diamond(k):
            near_z = max(0.0, abs(k - hz) - 0.5) / max(hz, 0.5)
            return near_u + near_v + near_z <= 1.0 + _EPS
        return diamond
    if kind == OCTAGON_PRISM:
        # A regular octagon squeezed to the footprint: the corners cut at 45 degrees.
        octagon = np.abs(du) / eu + np.abs(dv) / ev <= np.sqrt(2.0) + _EPS
        return lambda k: octagon
    if kind in LUMPY:
        return _lumpy_rule(kind, du / eu, dv / ev, height, seed)
    if kind in (GEAR, SPIRAL_RAMP, TOWER):
        if nu < 5 or nv < 5:
            return lambda k: ellipse <= 1.0 + _EPS  # too small for teeth or a ramp: a disc
        rho = np.sqrt(ellipse)  # 0 at the centre, 1 at the footprint's edge
        angle = np.arctan2(dv / ev, du / eu)
        rim = np.pi * (eu + ev)  # about the rim's length in cells
        if kind == GEAR:
            teeth = max(6, int(round(rim / 4.0)))  # a tooth and a gap about every four cells
            outer = np.where(np.cos(teeth * angle) > 0.0, 1.0, 0.78)
            gear = (rho <= outer + _EPS) & (rho >= 0.22)
            return lambda k: gear
        if kind == SPIRAL_RAMP:
            turns = max(1.0, height / 12.0)
            width = max(1.0 / 8.0, 1.5 * turns / height)  # thick enough that the layers join
            around = (angle + np.pi) / (2.0 * np.pi)
            column = rho <= 0.28 + _EPS
            ring = rho <= 1.0 + _EPS

            def ramp(k):
                phase = np.mod(around - (k + 0.5) / height * turns, 1.0)
                return column | (ring & (phase < width))
            return ramp
        # TOWER: the body narrows a little upwards; a wider course and a ring
        # of battlements finish the top.
        crown = max(1, height // 10)
        merlons = max(4, int(round(rim / 6.0)))
        merlon = np.cos(merlons * angle) > 0.0

        def tower(k):
            if height >= 6 and k >= height - crown:
                return (rho <= 1.0 + _EPS) & (rho >= 0.76) & merlon
            if height >= 6 and k == height - crown - 1:
                return rho <= 1.0 + _EPS
            return rho <= 0.86 - 0.08 * k / max(height - 1, 1) + _EPS
        return tower
    if kind == STAIRS:
        # As many steps as fit with a tread of at least two voxels; each step
        # rises by an equal share of the height, the back step full height.
        steps = max(1, min(height, nl // 2))
        tread = nl / steps
        top = (np.floor(pl / tread) + 1) / steps * height
        return lambda k: k + 0.5 <= top + _EPS
    return lambda k: full


LUMP = 0.28  # how far a Rock or a Hill bulges in and out, over its radius


def _lumpy_rule(kind: str, x: np.ndarray, y: np.ndarray, height: int, seed: int):
    """keep(k) for a Rock or a Hill: a ball or a dome whose radius wobbles with
    a smooth pattern of a few waves over the direction. `x`, `y` are the
    cells' places over the half-widths. A Rock rests on a flat cut, a Hill is
    a dome; both stay inside the box, and the middle of the base is always in."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    waves = [(rng.normal(size=3) * 2.2, rng.uniform(0, 2 * np.pi), rng.uniform(0.4, 1.0)) for _ in range(5)]
    total = sum(w for _d, _p, w in waves)
    r2_base = x**2 + y**2
    core = r2_base <= max((1.0 - 2 * LUMP) ** 2, float(r2_base.min())) + _EPS  # never nothing

    def radius(px, py, pz):
        n = sum(w * np.sin(d[0] * px + d[1] * py + d[2] * pz + p) for d, p, w in waves) / total
        return (1.0 - LUMP) + LUMP * n  # 1 - 2 LUMP .. 1: never past the box

    if kind == ROCK:
        base = 0.3  # the ball's centre sits this far up, so it rests on a flat cut

        def rock(k):
            z = ((k + 0.5) / height - base) / (1.0 - base)
            r2 = x**2 + y**2 + z**2
            return (r2 <= radius(x, y, z) ** 2 + _EPS) | ((k == 0) & core)
        return rock

    def hill(k):
        z = (k + 0.5) / height
        r2 = x**2 + y**2 + z**2
        return (r2 <= radius(x, y, z) ** 2 + _EPS) | ((k == 0) & core)
    return hill


def axis_drag_param(centre, normal, origin, direction) -> float | None:
    """Closest approach of the ray origin + t * direction to the line centre + s * normal.

    Returns s, the distance along the unit normal. None when the ray runs along
    the normal (the closest point is not defined). Pure Python floats.
    """
    n = [float(normal[i]) for i in range(3)]
    d = [float(direction[i]) for i in range(3)]
    w = [float(origin[i]) - float(centre[i]) for i in range(3)]
    b = n[0] * d[0] + n[1] * d[1] + n[2] * d[2]
    c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    e = d[0] * w[0] + d[1] * w[1] + d[2] * w[2]
    f = n[0] * w[0] + n[1] * w[1] + n[2] * w[2]
    denom = c - b * b
    if denom < _DRAG_EPS:
        return None
    # From d/dt and d/ds of |w + t*d - s*n|^2 == 0. Positive s lies on the
    # normal side of the centre.
    return (c * f - b * e) / denom


BRIDGE_SOLID = "SOLID"
BRIDGE_LINES = "LINES"
BRIDGE_KINDS = (BRIDGE_SOLID, BRIDGE_LINES)
BRIDGE_PAIRS = 40000  # lines per bridge at most; larger parts are thinned evenly


def bridge_cells(a: np.ndarray, b: np.ndarray, size, kind: str = BRIDGE_SOLID) -> np.ndarray:
    """(N, 3): cells that fill the gap between parts `a` and `b`, inside the grid,
    the two parts themselves left out.

    SOLID (default): a box loft of the facing sides. Side walls stay
    flat when the sides share that extent; a free axis steps only where the two
    sides differ on it.

    LINES: every cell on a straight line between a cell of `a` and a cell of `b`
    (the old fill). Large parts are thinned evenly so the pair count stays at
    BRIDGE_PAIRS.
    """
    kind = str(kind)
    if kind == BRIDGE_LINES:
        return _bridge_line_cells(a, b, size)
    if kind != BRIDGE_SOLID:
        raise ValueError(f"not a bridge kind: {kind!r}")
    return _bridge_slab_cells(a, b, size)


def _lerp_int(start: int, end: int, i: int, n: int) -> int:
    """Integer sample `i` of `n` along start→end. One sample is the midpoint."""
    if n <= 1:
        extra = 1 if start + end >= 0 else -1
        return (start + end + extra) // 2
    denom = n - 1
    return start + (i * (end - start) + denom // 2) // denom


def _aabb_gap(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> int:
    """Empty cells between two inclusive ranges. 0 when they touch, -1 when they overlap."""
    if a_hi < b_lo:
        return int(b_lo - a_hi - 1)
    if b_hi < a_lo:
        return int(a_lo - b_hi - 1)
    return -1


def _facing_axis(a_lo, a_hi, b_lo, b_hi) -> int | None:
    """Axis with the largest AABB gap. None when the boxes touch or overlap."""
    gaps = [_aabb_gap(int(a_lo[i]), int(a_hi[i]), int(b_lo[i]), int(b_hi[i])) for i in range(3)]
    axis = int(max(range(3), key=lambda i: (gaps[i], -i)))
    if gaps[axis] <= 0:
        return None
    return axis


def _face_at(cells: np.ndarray, axis: int, coord: int) -> np.ndarray:
    face = cells[cells[:, axis] == int(coord)]
    return face if len(face) else cells


def _clip_out_parts(cells: np.ndarray, a: np.ndarray, b: np.ndarray, size) -> np.ndarray:
    if len(cells) == 0:
        return cells.reshape(0, 3) if cells.ndim == 1 else cells[:0]
    size = np.asarray(size, dtype=np.int64)
    out = cells[np.all((cells >= 0) & (cells < size), axis=1)]
    if len(out) == 0:
        return out
    stride = np.array([int(size[1]) * int(size[2]), int(size[2]), 1], dtype=np.int64)
    own = np.concatenate([a, b]) @ stride
    return out[~np.isin(out @ stride, own)]


def _bridge_slab_cells(a: np.ndarray, b: np.ndarray, size) -> np.ndarray:
    """AABB loft between the facing sides of `a` and `b`."""
    a = np.asarray(a, dtype=np.int64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.int64).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        return a[:0]
    a_lo, a_hi = a.min(axis=0), a.max(axis=0)
    b_lo, b_hi = b.min(axis=0), b.max(axis=0)
    axis = _facing_axis(a_lo, a_hi, b_lo, b_hi)
    if axis is None:
        return a[:0]
    if int(a_hi[axis]) < int(b_lo[axis]):
        start, end = a, b
        x0, x1 = int(a_hi[axis]), int(b_lo[axis])
    else:
        start, end = b, a
        x0, x1 = int(b_hi[axis]), int(a_lo[axis])
    start_face = _face_at(start, axis, x0)
    end_face = _face_at(end, axis, x1)
    s_lo, s_hi = start_face.min(axis=0), start_face.max(axis=0)
    e_lo, e_hi = end_face.min(axis=0), end_face.max(axis=0)
    u, v = _free_axes(axis)
    n = x1 - x0 - 1
    if n <= 0:
        return a[:0]
    layers = []
    for i, x in enumerate(range(x0 + 1, x1)):
        u0 = _lerp_int(int(s_lo[u]), int(e_lo[u]), i, n)
        u1 = _lerp_int(int(s_hi[u]), int(e_hi[u]), i, n)
        v0 = _lerp_int(int(s_lo[v]), int(e_lo[v]), i, n)
        v1 = _lerp_int(int(s_hi[v]), int(e_hi[v]), i, n)
        layer = _plane_cells(axis, x, (min(u0, u1), max(u0, u1)), (min(v0, v1), max(v0, v1)), u, v)
        if len(layer):
            layers.append(layer)
    if not layers:
        return a[:0]
    out = np.unique(np.concatenate(layers), axis=0)
    return _clip_out_parts(out, a, b, size)


def _bridge_line_cells(a: np.ndarray, b: np.ndarray, size) -> np.ndarray:
    """All-pairs 3D lines between `a` and `b` (the old bridge fill)."""
    a = np.asarray(a, dtype=np.int64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.int64).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        return a[:0]
    while len(a) * len(b) > BRIDGE_PAIRS:
        if len(a) >= len(b):
            a = a[::2]
        else:
            b = b[::2]
    cells = [line3d_cells(tuple(pa), tuple(pb)) for pa in a for pb in b]
    out = np.unique(np.concatenate(cells), axis=0)
    return _clip_out_parts(out, a, b, size)

