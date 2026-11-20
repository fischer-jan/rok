"""Glyphs for the primitives Gear, Spiral Ramp, Tower, Buckyball and Framed
Buckyball and the tools Lathe, Twist, Remove Slice, Heightmap, Drop Voxels and
Crease, drawn here rather than generated.

    uv run python scripts/icons/make_icons.py GEAR SPIRAL_RAMP TOWER LATHE TWIST REMOVE_SLICE HEIGHTMAP
    uv run python scripts/icons/make_icons.py BUCKYBALL FRAMED_BUCKYBALL CREASE

Outlines at alpha 1 and top faces at FACE, as in the generated set. The round
shapes are prisms of a plane shape seen from the usual isometric angle.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anim_glyphs import FACE, LINE as LINE_HALF, SIDE, Canvas, _iso_cube  # noqa: E402
from select_glyphs import C30, _block  # noqa: E402



def _voxels(cv: Canvas, voxels: np.ndarray, s: float, origin=None) -> None:
    """Every voxel as a block, far to near, so nearer blocks hide farther ones.
    Centred on the canvas unless `origin` is given."""
    sx, sy, sz = voxels.shape
    if origin is None:
        # The drawing spans x - y across and (x + y) / 2 + z up, in blocks.
        width = (sx + sy) * C30 * s
        height = ((sx + sy) * 0.5 + sz) * s
        origin = (0.5 - width / 2 + sy * C30 * s, 0.5 - height / 2)
    cells = [tuple(c) for c in np.argwhere(voxels > 0)]
    for cell in sorted(cells, key=lambda c: (-(c[0] + c[1]) + c[2], c[2])):
        x, y, z = cell
        covered = (x > 0 and voxels[x - 1, y, z]) and (y > 0 and voxels[x, y - 1, z]) and (z + 1 < sz and voxels[x, y, z + 1])
        if not covered:
            _block(cv, origin, s, cell)


def _iso(x: float, y: float, z: float, origin=(0.5, 0.2)) -> tuple[float, float]:
    """A point of the unit shape (x, y across, z up) on the canvas."""
    return (origin[0] + (x - y) * C30, origin[1] + (x + y) * 0.5 + z)


def _plane_mask(cv: Canvas, inside, z: float, origin) -> np.ndarray:
    """The pixels whose point on the plane at height z is inside the shape:
    inside(x, y) takes arrays of plane coordinates."""
    u = (cv.x - origin[0]) / C30  # x - y
    w = (cv.y - origin[1] - z) / 0.5  # x + y
    return inside((u + w) / 2.0, (w - u) / 2.0)


def _edge(mask: np.ndarray, px: int) -> np.ndarray:
    """The pixels of the mask's border, about px wide."""
    h, w = mask.shape
    padded = np.pad(mask, px)  # no wrapping round the canvas edge
    grown = mask.copy()
    shrunk = mask.copy()
    for dy in range(-px, px + 1):
        for dx in range(-px, px + 1):
            if dx * dx + dy * dy > px * px:
                continue
            moved = padded[px + dy : px + dy + h, px + dx : px + dx + w]
            grown |= moved
            shrunk &= moved
    return grown & ~shrunk


def _extrude(cv: Canvas, inside, z0: float, z1: float, origin, fill_top: bool = True) -> None:
    """A prism of the shape from z0 up to z1: its outline, its top outlined
    and filled, and what it hides cleared first."""
    top = _plane_mask(cv, inside, z1, origin)
    body = top.copy()
    for k in range(24):
        body |= _plane_mask(cv, inside, z0 + (z1 - z0) * k / 23.0, origin)
    px = max(1, int(round(0.021 * cv.side)))
    cv.a[body] = 0.0
    if fill_top:
        cv.a[top] = FACE
    cv.a = np.maximum(cv.a, _edge(body, px).astype(np.float32))
    cv.a = np.maximum(cv.a, _edge(top, px).astype(np.float32))


def gear(side: int = SIDE) -> np.ndarray:
    """Gear: a flat cog wheel with square teeth and an axle hole."""
    cv = Canvas(side)

    def inside(x, y):
        r = np.hypot(x, y)
        a = np.arctan2(y, x)
        outer = np.where(np.cos(10 * a) > 0.0, 0.42, 0.33)
        return (r <= outer) & (r >= 0.1)

    _extrude(cv, inside, 0.0, 0.12, (0.5, 0.3))
    return cv.a


def spiral_ramp(side: int = SIDE) -> np.ndarray:
    """Spiral Ramp: a column with a ramp winding up round it."""
    cv = Canvas(side)
    origin = (0.5, 0.08)
    steps = 14
    for k in range(steps):  # far to near is hard on a spiral: bottom to top reads well
        a0 = 2 * math.pi * 1.5 * k / steps

        def inside(x, y, a0=a0):
            r = np.hypot(x, y)
            a = np.mod(np.arctan2(y, x) - a0, 2 * math.pi)
            return (r <= 0.34) & (r >= 0.1) & (a < 2 * math.pi / 7)

        z = 0.05 * k
        _extrude(cv, inside, z, z + 0.05, origin)
    _extrude(cv, lambda x, y: np.hypot(x, y) <= 0.1, 0.0, 0.05 * steps + 0.05, origin)
    return cv.a


def tower(side: int = SIDE) -> np.ndarray:
    """Tower: a round body, a wider course at the top, battlements on it."""
    cv = Canvas(side)
    origin = (0.5, 0.04)
    _extrude(cv, lambda x, y: np.hypot(x, y) <= 0.2, 0.0, 0.5, origin, fill_top=False)
    _extrude(cv, lambda x, y: np.hypot(x, y) <= 0.26, 0.5, 0.57, origin)

    def merlons(x, y):
        r = np.hypot(x, y)
        a = np.arctan2(y, x)
        return (r <= 0.26) & (r >= 0.17) & (np.cos(6 * a) > 0.3)

    _extrude(cv, merlons, 0.57, 0.66, origin)
    return cv.a


def lathe(side: int = SIDE) -> np.ndarray:
    """Lathe: a vase turned round its centre line, a turning arrow over it."""
    cv = Canvas(side)
    origin = (0.5, 0.05)
    profile = (0.2, 0.28, 0.3, 0.26, 0.16, 0.12, 0.16)
    for k, r in enumerate(profile):
        _extrude(cv, lambda x, y, r=r: np.hypot(x, y) <= r, 0.068 * k, 0.068 * (k + 1), origin,
                 fill_top=k == len(profile) - 1)
    cv.arc((0.5, 0.88), 0.22, math.radians(205), math.radians(335), half=0.018)
    tip = (0.5 + 0.22 * math.cos(math.radians(335)), 0.88 + 0.22 * math.sin(math.radians(335)))
    cv.fill([(tip[0] + 0.05, tip[1] + 0.05), (tip[0] + 0.03, tip[1] - 0.07), (tip[0] - 0.07, tip[1] - 0.02)], alpha=1.0)
    return cv.a


def twist(side: int = SIDE) -> np.ndarray:
    """Twist: four square layers, each turned further than the one below."""
    cv = Canvas(side)
    for k in range(4):
        angle = math.radians(22.0 * k)
        cy = 0.2 + 0.19 * k
        pts = []
        for corner in range(4):
            a = angle + math.radians(45 + 90 * corner)
            x, y = 0.3 * math.cos(a), 0.3 * math.sin(a)
            pts.append((0.5 + (x - y) * C30, cy + (x + y) * 0.5))
        if k == 3:
            cv.fill(pts, alpha=FACE)
        cv.path(pts, closed=True)
    cv.line((0.5, 0.12), (0.5, 0.86), half=0.012)
    return cv.a


def remove_slice(side: int = SIDE) -> np.ndarray:
    """Remove Slice: a column of blocks with its middle layer pulled out to the
    side, arrows closing the gap."""
    cv = Canvas(side)
    s = 0.2
    origin = (0.34, 0.05)
    _block(cv, origin, s, (0, 0, 0))
    _block(cv, origin, s, (0, 0, 2.2))
    # The slice that goes, out to the right, filled.
    _block(cv, (0.72, 0.2), s * 0.85, (0, 0, 0), faces="all")
    for x in (0.14, 0.54):
        cv.line((x, 0.72), (x, 0.52), half=0.014)
        cv.fill([(x - 0.04, 0.56), (x + 0.04, 0.56), (x, 0.49)], alpha=1.0)
    return cv.a


def heightmap(side: int = SIDE) -> np.ndarray:
    """Heightmap: a small picture, and block columns rising from it."""
    cv = Canvas(side)
    heights = np.array([[1, 2, 3], [1, 2, 2], [1, 1, 2]])
    v = np.zeros((3, 3, 3), np.uint8)
    for x in range(3):
        for y in range(3):
            v[x, y, : heights[x, y]] = 1
    _voxels(cv, v, 0.15, origin=(0.58, 0.04))
    # The picture: a square with a sun and a hill line, top left.
    frame = [(0.04, 0.62), (0.34, 0.62), (0.34, 0.94), (0.04, 0.94)]
    cv.path(frame, closed=True, half=0.016)
    cv.disc((0.25, 0.85), 0.035)
    cv.path([(0.04, 0.68), (0.13, 0.78), (0.2, 0.72), (0.34, 0.8)], half=0.014)
    return cv.a


def shaders(side: int = SIDE) -> np.ndarray:
    """Shaders: a block with a four-pointed sparkle over it."""
    cv = Canvas(side)
    from select_glyphs import _block

    _block(cv, (0.36, 0.08), 0.26, (0, 0, 0))
    cx, cy, r, w = 0.74, 0.74, 0.19, 0.055
    for tip, left, right in (
        ((cx, cy + r), (cx - w, cy), (cx + w, cy)),
        ((cx, cy - r), (cx + w, cy), (cx - w, cy)),
        ((cx - r, cy), (cx, cy - w), (cx, cy + w)),
        ((cx + r, cy), (cx, cy + w), (cx, cy - w)),
    ):
        cv.fill([tip, left, right], alpha=1.0)
    return cv.a


def cage(side: int = SIDE) -> np.ndarray:
    """Cage: a box with a dot on each corner and on the middle of its near edge."""
    cv = Canvas(side)
    c = _iso_cube(cv, 0.5, 0.74, 0.46, face=False)
    for name in ("front", "bottom", "left_t", "right_t", "left_b", "right_b", "back"):
        cv.disc(c[name], 0.062)
    mid = ((c["front"][0] + c["bottom"][0]) / 2, (c["front"][1] + c["bottom"][1]) / 2)
    cv.disc(mid, 0.04)
    return cv.a


def drop_voxels(side: int = SIDE) -> np.ndarray:
    """Drop Voxels: a cube resting on the floor line, a second one falling beside it
    with three streaks above, each voxel on its own."""
    cv = Canvas(side)
    cv.line((0.08, 0.10), (0.92, 0.10))  # the floor
    _iso_cube(cv, 0.29, 0.34, 0.20)  # the one that has landed: its bottom corner is on the floor
    _iso_cube(cv, 0.69, 0.60, 0.20)  # the one in the air
    for dx, y0, y1 in ((-0.10, 0.88, 0.95), (0.0, 0.90, 0.98), (0.10, 0.88, 0.95)):
        cv.line((0.69 + dx, y0), (0.69 + dx, y1))
    return cv.a


def _bucky_faces(tilt: float = 0.5, turn: float = 0.35):
    """The buckyball's faces seen from the front: [(screen points, five-sided)]
    for the faces that face the eye, in the canvas's 0 .. 1 square."""
    from rok.core.facets import bucky_planes, bucky_vertices

    normals, offsets = bucky_planes()
    verts = bucky_vertices()
    ct, st, cu, su = math.cos(tilt), math.sin(tilt), math.cos(turn), math.sin(turn)
    rz = np.array([[cu, -su, 0.0], [su, cu, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    rot = rx @ rz  # the eye looks along +y after the turn: screen x is x, screen y is z
    out = []
    for n, d in zip(normals, offsets):
        on = verts[np.abs(verts @ n - d) < 1e-6]
        nr = rot @ n
        if nr[1] > -0.05:  # faces away from the eye (the eye is at -y)
            continue
        centre = on.mean(axis=0)
        t1 = on[0] - centre
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        ang = np.arctan2((on - centre) @ t2, (on - centre) @ t1)
        loop = on[np.argsort(ang)] @ rot.T
        out.append(([(0.5 + 0.42 * p[0], 0.5 + 0.42 * p[2]) for p in loop], len(on) == 5))
    return out


def buckyball(side: int = SIDE) -> np.ndarray:
    """Buckyball: the ball's outline with its edges, the five-sided faces filled."""
    cv = Canvas(side)
    for pts, five in _bucky_faces():
        if five:
            cv.fill(pts)
        cv.path(pts, closed=True, half=0.6 * LINE_HALF)
    return cv.a


def framed_buckyball(side: int = SIDE) -> np.ndarray:
    """Framed Buckyball: every face a pane, the edges thick bars."""
    cv = Canvas(side)
    for pts, _five in _bucky_faces():
        cv.fill(pts)
    for pts, _five in _bucky_faces():
        cv.path(pts, closed=True, half=1.25 * LINE_HALF)
    return cv.a


def crease(side: int = SIDE) -> np.ndarray:
    """Crease: a block whose near edges carry raised bars."""
    cv = Canvas(side)
    c = _iso_cube(cv, 0.5, 0.62, 0.42)
    for p, q in ((c["front"], c["bottom"]), (c["front"], c["left_t"]), (c["front"], c["right_t"])):
        cv.line(p, q, half=2.2 * LINE_HALF)
    return cv.a


GLYPHS = {
    "BUCKYBALL": buckyball,
    "FRAMED_BUCKYBALL": framed_buckyball,
    "CREASE": crease,
    "CAGE": cage,
    "DROP_VOXELS": drop_voxels,
    "SHADERS": shaders,
    "GEAR": gear,
    "SPIRAL_RAMP": spiral_ramp,
    "TOWER": tower,
    "LATHE": lathe,
    "TWIST": twist,
    "REMOVE_SLICE": remove_slice,
    "HEIGHTMAP": heightmap,
}
