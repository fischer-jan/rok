"""The little moving pictures on the preset cards. No GPU.

Each preset is shown as what it does to a voxel cube, seen from the usual
isometric angle: Spin turns it, Swing rocks a bar over and back, Bob floats
it, Shake jitters it, Glow brightens its colour, Blink closes two eyes on
it, Flow runs colours along a row of cubes. `frame(name, t)` gives the
filled polygons of the picture at `t` seconds, back to front, in a box from
-1 to 1 on both axes; the drawing scales them into the card.

A card that is not under the mouse shows `REST_T`, a pose that already says
what the motion is; the one under the mouse plays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

COS30 = math.cos(math.radians(30))
SIN30 = 0.5
VIEW = np.array([-1.0, -1.0, 1.0]) / math.sqrt(3.0)  # toward the viewer
LIGHT = np.array([-0.35, -0.75, 1.0]) / np.linalg.norm([-0.35, -0.75, 1.0])

PART_RGB = (0.60, 0.50, 0.98)  # the part presets' cube: Animate's violet, lit
SHADOW_RGBA = (0.0, 0.0, 0.0, 0.35)
GLOW_RGB = (1.0, 0.56, 0.24)
EYE_RGB = (0.10, 0.08, 0.16)
FLOW_RGBS = (
    (0.98, 0.36, 0.42), (1.0, 0.64, 0.25), (0.98, 0.86, 0.32), (0.40, 0.86, 0.55), (0.38, 0.62, 1.0),
)

REST_T = {"SPIN": 0.18, "SWING": 0.2, "BOB": 0.3, "SHAKE": 0.0, "GLOW": 0.5, "BLINK": 0.0, "FLOW": 0.0}

# A cube's faces: corner indices (counter-clockwise seen from outside) and the normal.
_CORNERS = np.array(
    [[x, y, z] for z in (-0.5, 0.5) for y in (-0.5, 0.5) for x in (-0.5, 0.5)], dtype=np.float64
)
_FACES = (
    ((4, 5, 7, 6), (0, 0, 1)),  # top
    ((0, 2, 3, 1), (0, 0, -1)),  # bottom
    ((0, 1, 5, 4), (0, -1, 0)),
    ((3, 2, 6, 7), (0, 1, 0)),
    ((2, 0, 4, 6), (-1, 0, 0)),
    ((1, 3, 7, 5), (1, 0, 0)),
)


@dataclass(frozen=True)
class Shape:
    """One filled polygon of the picture: points in the -1..1 box and an RGBA."""

    points: tuple[tuple[float, float], ...]
    rgba: tuple[float, float, float, float]


def project(p) -> tuple[float, float]:
    """A world point to the picture: isometric, z up."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return ((x - y) * COS30, (x + y) * SIN30 + z)


def rotation(axis, angle: float) -> np.ndarray:
    """The 3x3 turn by `angle` radians about `axis` (Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _shade(rgb, normal) -> tuple[float, float, float, float]:
    lit = 0.52 + 0.48 * max(0.0, float(np.dot(normal, LIGHT)))
    return (min(1.0, rgb[0] * lit), min(1.0, rgb[1] * lit), min(1.0, rgb[2] * lit), 1.0)


def box(centre, size, rgb, turn: np.ndarray | None = None, pivot=None) -> list[tuple[float, Shape]]:
    """The faces of a box that face the viewer, each with its depth (bigger:
    farther). `size` is (sx, sy, sz); `turn` a 3x3 about `pivot` (the box
    centre when None)."""
    corners = _CORNERS * np.asarray(size, dtype=np.float64) + np.asarray(centre, dtype=np.float64)
    normals = [np.asarray(n, dtype=np.float64) for _i, n in _FACES]
    if turn is not None:
        p = np.asarray(centre if pivot is None else pivot, dtype=np.float64)
        corners = (corners - p) @ turn.T + p
        normals = [turn @ n for n in normals]
    out = []
    for (index, _n), normal in zip(_FACES, normals):
        if float(np.dot(normal, VIEW)) <= 1e-6:
            continue
        pts = corners[list(index)]
        centre_pt = pts.mean(axis=0)
        depth = float(centre_pt[0] + centre_pt[1] - centre_pt[2])
        out.append((depth, Shape(tuple(project(q) for q in pts), _shade(rgb, normal))))
    return out


def ellipse(cx: float, cy: float, rx: float, ry: float, rgba, steps: int = 24) -> Shape:
    return Shape(
        tuple((cx + rx * math.cos(2 * math.pi * k / steps), cy + ry * math.sin(2 * math.pi * k / steps))
              for k in range(steps)),
        tuple(rgba),
    )


def _shadow(scale: float = 1.0) -> Shape:
    x, y = project((0.0, 0.0, -0.62))
    return ellipse(x, y, 0.62 * scale, 0.3 * scale, SHADOW_RGBA)


def _ordered(faces: list[tuple[float, Shape]]) -> list[Shape]:
    return [shape for _d, shape in sorted(faces, key=lambda f: -f[0])]


def _hash(n: int) -> float:
    """A repeatable number in -1..1 for integer `n`."""
    return math.sin(n * 12.9898 + 78.233) * 43758.5453 % 2.0 - 1.0


def frame(name: str, t: float) -> tuple[Shape, ...]:
    """The picture of preset `name` at `t` seconds, back to front."""
    tau = 2 * math.pi
    if name == "SPIN":
        turn = rotation((0, 0, 1), t * tau / 2.0)
        return (_shadow(),) + tuple(_ordered(box((0, 0, 0), (1, 1, 1), PART_RGB, turn)))
    if name == "SWING":
        angle = math.radians(38) * math.sin(t * tau / 1.2)
        turn = rotation((1, 1, 0), angle)
        base = box((0, 0, -0.52), (0.62, 0.62, 0.24), tuple(c * 0.7 for c in PART_RGB))
        bar = box((0, 0, 0.12), (0.2, 0.2, 1.08), PART_RGB, turn, pivot=(0, 0, -0.42))
        return (_shadow(0.8),) + tuple(_ordered(base)) + tuple(_ordered(bar))
    if name == "BOB":
        z = 0.3 * math.sin(t * tau / 1.2)
        return (_shadow(1.0 - 0.35 * (z + 0.3)),) + tuple(_ordered(box((0, 0, z + 0.08), (0.9, 0.9, 0.9), PART_RGB)))
    if name == "SHAKE":
        n = int(t * 14)
        dx, dy = 0.09 * _hash(n), 0.09 * _hash(n + 101)
        return (_shadow(),) + tuple(_ordered(box((dx, dy, 0), (1, 1, 1), PART_RGB)))
    if name == "GLOW":
        k = 0.5 + 0.5 * math.sin(t * tau / 2.0 - math.pi / 2)
        rgb = tuple(min(1.0, c * (0.45 + 0.75 * k)) for c in GLOW_RGB)
        halo = ellipse(0.0, 0.1, 1.0, 1.0, GLOW_RGB + (0.10 + 0.28 * k,), 32)
        return (halo, _shadow()) + tuple(_ordered(box((0, 0, 0), (1, 1, 1), rgb)))
    if name == "BLINK":
        cube = _ordered(box((0, 0, 0), (1, 1, 1), (0.95, 0.80, 0.62)))
        phase = t % 1.6
        open_ = 1.0 if phase > 0.18 else abs(phase - 0.09) / 0.09
        eyes = []
        for u in (-0.22, 0.2):
            # On the -y face (y = -0.5): x across, z up.
            h = 0.2 * max(0.08, open_)
            pts = [(u - 0.07, -0.5, 0.08 - h / 2), (u + 0.07, -0.5, 0.08 - h / 2),
                   (u + 0.07, -0.5, 0.08 + h / 2), (u - 0.07, -0.5, 0.08 + h / 2)]
            eyes.append(Shape(tuple(project(p) for p in pts), EYE_RGB + (1.0,)))
        return (_shadow(),) + tuple(cube) + tuple(eyes)
    if name == "FLOW":
        step = int(t * 6)
        faces = []
        for k in range(5):
            rgb = FLOW_RGBS[(k - step) % len(FLOW_RGBS)]
            x = -0.84 + 0.42 * k
            faces += box((x, -x * 0.0, 0), (0.36, 0.36, 0.36), rgb)
        shadow = ellipse(*project((0, 0, -0.28)), 1.0, 0.26, SHADOW_RGBA)
        return (shadow,) + tuple(_ordered(faces))
    return (_shadow(),) + tuple(_ordered(box((0, 0, 0), (1, 1, 1), PART_RGB)))


def bounds(shapes) -> tuple[float, float, float, float]:
    """(min x, min y, max x, max y) of every point."""
    xs = [p[0] for s in shapes for p in s.points]
    ys = [p[1] for s in shapes for p in s.points]
    return (min(xs), min(ys), max(xs), max(ys))
