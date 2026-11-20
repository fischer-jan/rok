"""The transform gizmo: handle geometry, projection, hit test and the drag.

Rok draws the gizmo itself (see the Phase 3 design, Transform gizmo). This
module is pure numpy: it builds the geometry the overlay draws, projects the
anchors for the hit test and turns a drag into a `GizmoResult`. Imports .edit,
.shape and .select only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import shape
from .edit import SELECT, Cell, SetMany
from .geom2d import polyline_distance
from .gesture_base import Gesture
from .select import (
    MovePreview,
    Selection,
    flip_cells,
    flipped_cells,
    move_cells,
    rotate_cells,
    turned_cells,
)

HANDLES = (
    "MOVE_X",
    "MOVE_Y",
    "MOVE_Z",
    "PLANE_YZ",
    "PLANE_XZ",
    "PLANE_XY",
    "ROT_X",
    "ROT_Y",
    "ROT_Z",
    "FLIP_X",
    "FLIP_Y",
    "FLIP_Z",
    "CENTER",
    "SCALE_ALL",
)
# Rok's axis colours, x, y, z: coral, light green, sky blue. Every axis
# drawing uses them (the gizmos, the guides, the agent's pictures).
AXIS_COLOURS = ((0.94, 0.40, 0.33), (0.55, 0.80, 0.30), (0.35, 0.60, 0.95))
AXIS_NAMES = "XYZ"
# In the scene the model itself is the drag handle, and a centre square there
# caught the second click of a double-click: the scene's gizmo has none. Its
# centre holds a small sphere instead, which scales the model in all three
# dimensions (SCALE_ALL); inside a model the gizmo has no sphere.
GIZMO_SKIP_IN_SCENE = ("CENTER",)
GIZMO_SKIP_IN_MODEL = ("SCALE_ALL",)
# In Animate mode the gizmo moves and turns a part, each drag a key: no flip
# and no stretch (they would edit the voxels). Shift on the centre square moves
# the part's pivot.
GIZMO_SKIP_IN_ANIM = ("FLIP_X", "FLIP_Y", "FLIP_Z", "SCALE_ALL")


def skip_for(level: str, animating: bool = False) -> tuple[str, ...]:
    """The handles the gizmo leaves out at `level`, or in Animate mode."""
    if animating:
        return GIZMO_SKIP_IN_ANIM
    return GIZMO_SKIP_IN_SCENE if level == "SCENE" else GIZMO_SKIP_IN_MODEL
STRETCH_HANDLES = ("FLIP_X", "FLIP_Y", "FLIP_Z")  # in the scene the cubes stretch, not flip

SELECTION = "SELECTION"
MODEL = "MODEL"

MOVE = "MOVE"
ROTATE = "ROTATE"
FLIP = "FLIP"
PIVOT = "PIVOT"
NONE = "NONE"

# Layout as fractions of the arrow length L.
LINE_END = 0.82
CONE_RADIUS = 0.07
CONE_SEGMENTS = 12
FLIP_AT = 0.68
FLIP_SIDE = 0.09
RING_RADIUS = 0.55
RING_SEGMENTS = 48
PLANE_AT = 0.28
PLANE_SIDE = 0.13
CENTER_SIDE = 0.1
MOVE_ANCHOR = (LINE_END + 1.0) / 2  # the middle of the cone

FADE_DOT = 0.05  # a ring is edge-on under this |view_dir . axis|
RING_HIT = 0.6  # ring hit distance as a fraction of radius_px
# A ring drag turns freely: tenths of a degree, a snap to the
# quarter turns within SNAP_DEG, 15 degree steps with Ctrl, a tenth of the
# speed with Shift.
STEP_DEG = 0.1
SNAP_DEG = 4.0
COARSE_DEG = 15.0
FINE = 0.1
_EPS = 1e-9

_MOVE = ("MOVE_X", "MOVE_Y", "MOVE_Z")
_PLANE = ("PLANE_YZ", "PLANE_XZ", "PLANE_XY")
_ROT = ("ROT_X", "ROT_Y", "ROT_Z")
_FLIP = ("FLIP_X", "FLIP_Y", "FLIP_Z")
# Hit test order: the centre, the cubes, the squares, the cones, then the rings.
_ANCHOR_GROUPS = (("CENTER", "SCALE_ALL"), _FLIP, _PLANE, _MOVE)


def handle_axis(handle: str) -> int:
    """The axis a handle belongs to. PLANE_bc belongs to the third axis a."""
    for group in (_MOVE, _PLANE, _ROT, _FLIP):
        if handle in group:
            return group.index(handle)
    return 0


def _unit(axis: int) -> np.ndarray:
    e = np.zeros(3, dtype=np.float64)
    e[axis] = 1.0
    return e


def _normalised(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else v


def _view_basis(view_dir) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors that span the plane facing the viewer."""
    d = _normalised(view_dir)
    up = _unit(2) if abs(d[2]) < 0.9 else _unit(1)
    right = _normalised(np.cross(d, up))
    return right, _normalised(np.cross(right, d))


def _square(centre, u, v, half: float) -> np.ndarray:
    """Two triangles of the square spanned by u and v around centre, (6, 3)."""
    c = np.asarray(centre, dtype=np.float64)
    a = c - u * half - v * half
    b = c + u * half - v * half
    cc = c + u * half + v * half
    d = c - u * half + v * half
    return np.array([a, b, cc, a, cc, d], dtype=np.float32)


def _sphere(centre, radius: float, rings: int = 6, segments: int = 10) -> np.ndarray:
    """A small UV sphere as TRIS, (rings * segments * 6, 3)."""
    c = np.asarray(centre, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, rings + 1)
    phi = np.linspace(0.0, 2 * np.pi, segments + 1)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    pts = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1) * radius + c
    a, b = pts[:-1, :-1], pts[1:, :-1]
    d, e = pts[:-1, 1:], pts[1:, 1:]
    tris = np.stack([a, b, e, a, e, d], axis=2).reshape(-1, 3)
    return tris.astype(np.float32)


def _cube(centre, side: float) -> np.ndarray:
    """The 12 triangles of an axis-aligned cube, (36, 3)."""
    c = np.asarray(centre, dtype=np.float64)
    h = side / 2
    corners = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ]
    )
    quads = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    tris = []
    for a, b, cc, d in quads:
        tris.extend([corners[a], corners[b], corners[cc]])
        tris.extend([corners[a], corners[cc], corners[d]])
    return (np.array(tris, dtype=np.float64) + c).astype(np.float32)


def _cone(pivot, axis: int, length: float) -> np.ndarray:
    """A cone along the axis from LINE_END L to L with a fan for the base."""
    e = _unit(axis)
    u = _unit((axis + 1) % 3)
    v = _unit((axis + 2) % 3)
    base = pivot + e * (LINE_END * length)
    tip = pivot + e * length
    r = CONE_RADIUS * length
    angles = np.linspace(0.0, 2 * math.pi, CONE_SEGMENTS, endpoint=False)
    rim = base + np.outer(np.cos(angles), u) * r + np.outer(np.sin(angles), v) * r
    tris = []
    for i in range(CONE_SEGMENTS):
        a = rim[i]
        b = rim[(i + 1) % CONE_SEGMENTS]
        tris.extend([a, b, tip, base, b, a])
    return np.array(tris, dtype=np.float32)


def _ring(pivot, axis: int, length: float) -> np.ndarray:
    """RING_SEGMENTS points of the circle in the plane normal to the axis."""
    u = _unit((axis + 1) % 3)
    v = _unit((axis + 2) % 3)
    r = RING_RADIUS * length
    angles = np.linspace(0.0, 2 * math.pi, RING_SEGMENTS, endpoint=False)
    pts = pivot + np.outer(np.cos(angles), u) * r + np.outer(np.sin(angles), v) * r
    return pts.astype(np.float32)


@dataclass(frozen=True)
class GizmoGeometry:
    """Everything the overlay draws, in the space the gizmo is drawn in."""

    pivot: np.ndarray  # (3,) float32
    length: float  # L, the arrow length in that space
    lines: dict[str, np.ndarray]  # LINES pairs per handle (the axis lines)
    tris: dict[str, np.ndarray]  # TRIS per handle (cones, squares, cubes, centre)
    rings: dict[str, np.ndarray]  # ROT_*: (48, 3) polyline points
    anchors: dict[str, np.ndarray]  # (k, 3) points whose projection is hit-tested


def geometry(pivot, length: float, view_dir) -> GizmoGeometry:
    """Build the handle geometry around the pivot with arrow length `length`."""
    p = np.asarray(pivot, dtype=np.float64).reshape(3)
    length = float(length)
    lines: dict[str, np.ndarray] = {}
    tris: dict[str, np.ndarray] = {}
    rings: dict[str, np.ndarray] = {}
    anchors: dict[str, np.ndarray] = {}
    for axis in range(3):
        e = _unit(axis)
        b, c = (i for i in range(3) if i != axis)
        lines[_MOVE[axis]] = np.array([p, p + e * (LINE_END * length)], dtype=np.float32)
        tris[_MOVE[axis]] = _cone(p, axis, length)
        cone_at = p + e * (MOVE_ANCHOR * length)
        anchors[_MOVE[axis]] = cone_at.reshape(1, 3).astype(np.float32)
        flip_at = p + e * (FLIP_AT * length)
        tris[_FLIP[axis]] = _cube(flip_at, FLIP_SIDE * length)
        anchors[_FLIP[axis]] = flip_at.reshape(1, 3).astype(np.float32)
        plane_at = p + (_unit(b) + _unit(c)) * (PLANE_AT * length)
        tris[_PLANE[axis]] = _square(plane_at, _unit(b), _unit(c), PLANE_SIDE * length / 2)
        anchors[_PLANE[axis]] = plane_at.reshape(1, 3).astype(np.float32)
        rings[_ROT[axis]] = _ring(p, axis, length)
    right, up = _view_basis(view_dir)
    tris["CENTER"] = _square(p, right, up, CENTER_SIDE * length / 2)
    anchors["CENTER"] = p.reshape(1, 3).astype(np.float32)
    tris["SCALE_ALL"] = _sphere(p, FLIP_SIDE * length * 0.62)  # about as big as the cubes
    anchors["SCALE_ALL"] = p.reshape(1, 3).astype(np.float32)
    return GizmoGeometry(p.astype(np.float32), length, lines, tris, rings, anchors)


def project(matrix, points, width: int, height: int) -> np.ndarray:
    """(N, 2) float64 region pixels of points under a 4 x 4 clip matrix.

    Rows with w <= 0 (behind the eye) are nan.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    if n == 0:
        return out
    clip = pts @ m[:3, :3].T + m[:3, 3]
    w = pts @ m[3, :3] + m[3, 3]
    ok = w > 0
    if ok.any():
        wk = w[ok]
        out[ok, 0] = (clip[ok, 0] / wk + 1.0) * 0.5 * float(width)
        out[ok, 1] = (clip[ok, 1] / wk + 1.0) * 0.5 * float(height)
    return out


@dataclass(frozen=True)
class GizmoScreen:
    """The projected anchors and rings of the last drawn frame."""

    pivot: np.ndarray  # (2,) region pixels
    anchors: dict[str, np.ndarray]  # (k, 2)
    rings: dict[str, np.ndarray]  # (48, 2)


def screen_geometry(geom: GizmoGeometry, matrix, width: int, height: int) -> GizmoScreen:
    """Project the anchors and the rings for the hit test."""
    pivot = project(matrix, geom.pivot.reshape(1, 3), width, height)[0]
    anchors = {k: project(matrix, v, width, height) for k, v in geom.anchors.items()}
    rings = {k: project(matrix, v, width, height) for k, v in geom.rings.items()}
    return GizmoScreen(pivot, anchors, rings)


def ring_faded(view_dir, axis: int) -> bool:
    """True when the ring about the axis is seen edge-on."""
    d = _normalised(view_dir)
    return abs(float(d[axis])) < FADE_DOT


def hit_test(
    screen: GizmoScreen | None, mx: float, my: float, radius_px: float, view_dir, skip=()
) -> str | None:
    """The handle under a region pixel, or None.

    Order: CENTER, the flip cubes, the plane squares, the cones (nearest anchor
    under radius_px wins within a group), then the rings (nearest under
    RING_HIT * radius_px). Edge-on rings are skipped.
    """
    if screen is None:
        return None
    mx = float(mx)
    my = float(my)
    for group in _ANCHOR_GROUPS:
        best = None
        best_d = float(radius_px)
        for handle in group:
            if handle in skip:
                continue
            pts = screen.anchors.get(handle)
            if pts is None or len(pts) == 0:
                continue
            d = np.hypot(pts[:, 0] - mx, pts[:, 1] - my)
            d = d[np.isfinite(d)]
            if len(d) == 0:
                continue
            dm = float(d.min())
            if dm < best_d:
                best = handle
                best_d = dm
        if best is not None:
            return best
    best = None
    best_d = RING_HIT * float(radius_px)
    for axis, handle in enumerate(_ROT):
        if ring_faded(view_dir, axis):
            continue
        pts = screen.rings.get(handle)
        if pts is None:
            continue
        dm = polyline_distance(np.asarray(pts, dtype=np.float64), mx, my)
        if dm < best_d:
            best = handle
            best_d = dm
    return best


def snap_pivot(pivot2, axis: int) -> Cell:
    """Make a quarter turn about `axis` map cells to cells.

    The two plane coordinates of the doubled pivot need the same parity; when
    they differ, the higher-numbered one gets 1 added.
    """
    p = [int(pivot2[0]), int(pivot2[1]), int(pivot2[2])]
    b, c = (i for i in range(3) if i != int(axis))
    if (p[b] - p[c]) % 2 != 0:
        p[c] += 1
    return (p[0], p[1], p[2])


def plane_point(origin, direction, point, normal) -> np.ndarray | None:
    """Where the ray origin + t * direction crosses the plane through point.

    None when the ray runs parallel to the plane or crosses it behind the origin.
    """
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    p = np.asarray(point, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    denom = float(d @ n)
    if abs(denom) < _EPS:
        return None
    t = float((p - o) @ n) / denom
    if t <= _EPS:
        return None
    return o + d * t


# ---------------------------------------------------------------- the drag


@dataclass(frozen=True)
class GizmoResult:
    """What a gizmo drag asks for. `pivot2` is the pivot used, after snap."""

    kind: str  # MOVE | ROTATE | FLIP | PIVOT | NONE
    axis: int = 0
    turns: int = 0  # ROTATE: quarter turns, 0..3; 0 with `degrees` set is a free turn
    offset: Cell = (0, 0, 0)  # MOVE
    copy: bool = False
    pivot2: Cell = (0, 0, 0)
    degrees: float = 0.0  # ROTATE: the angle, in tenths, wrapped into (-180, 180]

    @property
    def free(self) -> bool:
        """A turn that is not a whole number of quarter turns."""
        return self.kind == ROTATE and self.turns == 0 and self.degrees != 0.0


def _cell(values) -> Cell:
    return (int(values[0]), int(values[1]), int(values[2]))


def _offset_label(offset: Cell, copy: bool) -> str:
    parts = [f"{int(v):+d} {AXIS_NAMES[i]}" for i, v in enumerate(offset) if int(v) != 0]
    if not parts:
        return ""
    text = " ".join(parts)
    return f"copy {text}" if copy else text


class GizmoDrag(Gesture):
    """One press-drag-release on a gizmo handle, in the gizmo's grid space.

    The operator calls `begin_ray` with the press ray (and `set_mouse` before it
    for the rings), then `set_mouse` and `extend` on every move. `finish` applies
    a SELECTION result through `set_many`; a MODEL result is the caller's.
    """

    tool = SELECT
    hit = None
    last = None

    def __init__(
        self,
        handle: str,
        target: str,
        pivot2,
        voxels: np.ndarray,
        selection: Selection | None,
        set_many: SetMany | None,
        *,
        shift: bool,
        view_dir,
        pivot_px,
        size,
        set_grid=None,
    ) -> None:
        if handle not in HANDLES:
            raise ValueError(f"not a handle: {handle!r}")
        self.handle = handle
        self.target = target
        self.pivot2 = _cell(pivot2)
        self.voxels = voxels
        self.selection = selection
        self.set_many = set_many
        self.shift = bool(shift)
        self.view_dir = _normalised(view_dir)
        self.pivot_px = (float(pivot_px[0]), float(pivot_px[1]))
        self.size = _cell(size)
        self.axis = handle_axis(handle)
        self.pivot = np.asarray(self.pivot2, dtype=np.float64) / 2.0
        self.mouse: tuple[float, float] | None = None
        # References taken at the press.
        self.s0: float | None = None
        self.p0: np.ndarray | None = None
        self.angle0: float | None = None
        self.angle_acc = 0.0
        self.degrees = 0.0  # the turn so far, snapped (not wrapped)
        self.coarse = False  # Ctrl: 15 degree steps
        self.fine = False  # Shift: a tenth of the speed
        # The integer state.
        self.offset: Cell = (0, 0, 0)
        self.turns = 0
        self.swept = False
        self.new_pivot2: Cell = self.pivot2
        self._preview_key = None
        self._preview = None
        # A ring drag on a selection writes the turned voxels at once, on every change.
        self._live = LiveTurn(voxels, selection, set_many, set_grid)

    # -- kind

    @property
    def is_pivot(self) -> bool:
        return self.handle == "CENTER" and self.shift

    @property
    def copy(self) -> bool:
        # Shift copies, a selection and a whole model alike: a copy stays behind.
        return self.shift and self.handle in _MOVE + _PLANE

    def _kind(self) -> str:
        if self.handle in _ROT:
            return ROTATE
        if self.handle in _FLIP:
            return FLIP
        if self.is_pivot:
            return PIVOT
        return MOVE

    def _effective_turns(self) -> int:
        """A press with no sweep is one quarter turn."""
        if self.turns == 0 and not self.swept:
            return 1
        return self.turns

    # -- protocol

    def set_mouse(self, mx: float, my: float) -> None:
        self.mouse = (float(mx), float(my))

    def set_shift(self, shift: bool) -> None:
        """Shift during a ring drag turns slower; the copy flag is the press's."""
        self.fine = bool(shift)

    def _angle(self) -> float | None:
        if self.mouse is None:
            return None
        dx = self.mouse[0] - self.pivot_px[0]
        dy = self.mouse[1] - self.pivot_px[1]
        return math.atan2(dy, dx)

    def begin_ray(self, origin, direction) -> bool:
        """Store the reference of the drag. False when the handle cannot start."""
        kind = self._kind()
        if kind == FLIP:
            return True
        if kind == ROTATE:
            if ring_faded(self.view_dir, self.axis):
                return False
            self.angle0 = self._angle()  # None until the first set_mouse
            self.angle_acc = 0.0
            return True
        if self.handle in _MOVE:
            s = shape.axis_drag_param(self.pivot, _unit(self.axis), origin, direction)
            if s is None:
                return False
            self.s0 = s
            return True
        normal = self.view_dir if self.handle == "CENTER" else _unit(self.axis)
        p = plane_point(origin, direction, self.pivot, normal)
        if p is None:
            return False
        self.p0 = p
        return True

    def extend(self, origin, direction) -> bool:
        """Update the integer result. True when it changed."""
        kind = self._kind()
        if kind == FLIP:
            return False
        if kind == ROTATE:
            return self._extend_turn()
        if self.handle in _MOVE:
            if self.s0 is None:
                return False
            s = shape.axis_drag_param(self.pivot, _unit(self.axis), origin, direction)
            if s is None:
                return False
            offset = [0, 0, 0]
            offset[self.axis] = int(round(s - self.s0))
            return self._set_offset(offset)
        if self.p0 is None:
            return False
        normal = self.view_dir if self.handle == "CENTER" else _unit(self.axis)
        p = plane_point(origin, direction, self.pivot, normal)
        if p is None:
            return False
        if kind == PIVOT:
            new = np.rint(2.0 * p).astype(np.int64)
            new = np.clip(new, 0, 2 * np.asarray(self.size, dtype=np.int64))
            new_pivot2 = _cell(new)
            if new_pivot2 == self.new_pivot2:
                return False
            self.new_pivot2 = new_pivot2
            return True
        delta = np.rint(p - self.p0).astype(np.int64)
        offset = [int(v) for v in delta]
        if self.handle in _PLANE:
            offset[self.axis] = 0
        return self._set_offset(offset)

    def _set_offset(self, offset) -> bool:
        cell = _cell(offset)
        if cell == self.offset:
            return False
        self.offset = cell
        return True

    def _extend_turn(self) -> bool:
        angle = self._angle()
        if angle is None:
            return False
        if self.angle0 is None:
            self.angle0 = angle
            return False
        # Unwrap step by step so a sweep past 180 degrees keeps counting.
        step = angle - self.angle0
        step = (step + math.pi) % (2 * math.pi) - math.pi
        self.angle_acc += step * (FINE if self.fine else 1.0)
        self.angle0 = angle
        deg = math.degrees(self.angle_acc)
        if float(_unit(self.axis) @ self.view_dir) > 0:
            deg = -deg
        before = self.result()
        self.degrees = snap_degrees(deg, coarse=self.coarse)
        if self.degrees != 0.0:
            self.swept = True
        q = round(self.degrees / 90.0)
        self.turns = int(q) % 4 if abs(self.degrees - 90.0 * q) < 1e-9 else 0
        after = self.result()
        changed = (after.kind, after.turns, after.degrees) != (before.kind, before.turns, before.degrees)
        if changed and self.swept and self.target == SELECTION:
            self._show_live(after)
        return changed

    # --- the live turn of a selection ---------------------------------------------------

    @staticmethod
    def _turn_key(res: GizmoResult):
        return (res.kind, res.axis, res.turns, res.degrees, res.pivot2)

    def _show_live(self, res: GizmoResult) -> None:
        """Show `res` in the real voxels (LiveTurn); a result that is no turn
        puts the voxels at the press back."""
        if res.kind == ROTATE and res.free:
            self._live.show_free(self._turn_key(res), res.axis, res.degrees, res.pivot2)
            return
        if res.kind != ROTATE:
            turn = None
        else:
            def turn(v, s, w):
                return rotate_cells(v, s, res.axis, res.turns, res.pivot2, w)
        self._live.show(self._turn_key(res), turn)

    def result(self) -> GizmoResult:
        kind = self._kind()
        if kind == ROTATE:
            if not self.swept:  # a click on a ring: one quarter turn
                return GizmoResult(ROTATE, axis=self.axis, turns=1, degrees=90.0,
                                   pivot2=snap_pivot(self.pivot2, self.axis))
            d = wrap_degrees(self.degrees)
            if d == 0.0:
                return GizmoResult(NONE, axis=self.axis, pivot2=self.pivot2)
            turns = self.turns
            pivot2 = snap_pivot(self.pivot2, self.axis) if turns % 2 else self.pivot2
            return GizmoResult(ROTATE, axis=self.axis, turns=turns, degrees=d, pivot2=pivot2)
        if kind == FLIP:
            return GizmoResult(FLIP, axis=self.axis, pivot2=self.pivot2)
        if kind == PIVOT:
            return GizmoResult(PIVOT, pivot2=self.new_pivot2)
        if self.offset == (0, 0, 0):
            return GizmoResult(NONE, pivot2=self.pivot2, copy=self.copy)
        return GizmoResult(MOVE, offset=self.offset, copy=self.copy, pivot2=self.pivot2)

    def _label(self, res: GizmoResult) -> str:
        """The text at the mouse: "+3 X", "copy +3 X", "2 x 90 Z", "flip Y" or "pivot"."""
        if res.kind == PIVOT:
            return "pivot"
        if res.kind == ROTATE:
            return degrees_label(res.degrees, res.axis)
        if res.kind == FLIP:
            return f"flip {AXIS_NAMES[res.axis]}"
        if self.handle in _ROT:
            return degrees_label(0.0, self.axis)
        return _offset_label(res.offset, res.copy)

    def finish(self) -> int:
        """Apply a SELECTION result through set_many. MODEL and PIVOT: 0."""
        if self.target != SELECTION or self.selection is None or self.set_many is None:
            return 0
        res = self.result()
        if self.handle in _ROT:
            # The live turn is already in the voxels; a click turns now.
            if self._live.key != self._turn_key(res):
                self._show_live(res)
            return self._live.total
        if res.kind not in (MOVE, ROTATE, FLIP):
            return 0
        total = 0

        def counted(cells: np.ndarray, values: np.ndarray) -> int:
            nonlocal total
            n = int(self.set_many(cells, values))
            total += n
            return n

        if res.kind == MOVE:
            new = move_cells(self.voxels, self.selection, res.offset, res.copy, counted)
        elif res.free:
            new = free_rotate_cells(self.voxels, self.selection, res.axis, res.degrees, res.pivot2, counted)
        elif res.kind == ROTATE:
            new = rotate_cells(
                self.voxels, self.selection, res.axis, res.turns, res.pivot2, counted
            )
        else:
            new = flip_cells(self.voxels, self.selection, res.axis, res.pivot2, counted)
        self.selection.replace(new)
        return total

    def preview(self):
        """SELECTION: a move preview, or the transformed cells. MODEL: the label only;
        the Empty or the ghost shows the move."""
        res = self.result()
        if self.target != SELECTION or self.selection is None:
            return MovePreview(kind="LABEL", label=self._label(res))
        if res.kind == PIVOT:
            return MovePreview(label="pivot")
        if self.handle in _ROT:
            # The voxels themselves show the turn (see _live): only the angle.
            return MovePreview(kind="LABEL", label=self._label(res))
        if res.kind in (MOVE, NONE) and self.handle not in _ROT:
            label = _offset_label(res.offset, res.copy)
            return MovePreview(offset=res.offset, label=label)
        key = (res.kind, res.axis, res.turns, res.degrees, res.pivot2, self.selection.version)
        if key != self._preview_key:
            cells = self.selection.cells()
            if res.free:
                cells = free_turned_cells(self.voxels, self.selection, res.axis, res.degrees, res.pivot2)[0]
                label = degrees_label(res.degrees, res.axis)
            elif res.kind == ROTATE:
                cells = turned_cells(cells, res.axis, res.turns, res.pivot2)
                label = degrees_label(res.degrees, res.axis)
            elif res.kind == FLIP:
                cells = flipped_cells(cells, res.axis, res.pivot2)
                label = f"flip {AXIS_NAMES[res.axis]}"
            else:
                cells = cells[:0]
                label = ""
            self._preview_key = key
            self._preview = MovePreview(kind="CELLS", cells=cells, label=label)
        return self._preview


# --- free turns -------------------------------------------------------------------------


def wrap_degrees(degrees: float) -> float:
    """Tenths of a degree, wrapped into (-180, 180]."""
    d = round(float(degrees), 1)
    d = (d + 180.0) % 360.0 - 180.0
    if d <= -180.0:
        d += 360.0
    return round(d, 1)


def snap_degrees(degrees: float, *, coarse: bool = False) -> float:
    """A ring drag's angle: 15 degree steps with `coarse`; else the nearest
    quarter turn within SNAP_DEG, or tenths of a degree."""
    if coarse:
        return round(round(degrees / COARSE_DEG) * COARSE_DEG, 1)
    q = round(degrees / 90.0) * 90.0
    if abs(degrees - q) <= SNAP_DEG:
        return float(q)
    return round(round(degrees / STEP_DEG) * STEP_DEG, 1)


def degrees_label(degrees: float, axis: int) -> str:
    """The text at the mouse: "37.5° Z", "90° X"."""
    d = wrap_degrees(degrees)
    text = f"{d:.0f}" if d == int(d) else f"{d:.1f}"
    return f"{text}° {AXIS_NAMES[axis]}"


def free_turned_block(voxels: np.ndarray, mask: np.ndarray, bounds, axis: int, degrees: float, pivot2):
    """The voxels under `mask` (inside its inclusive `bounds`) turned by any
    angle about `pivot2` (doubled cells): (block, position) of the turned
    box, as free_rotate.rotate_models places it; 0 in the block is empty."""
    from . import free_rotate

    lo = np.asarray(bounds[0], dtype=np.int64)
    hi = np.asarray(bounds[1], dtype=np.int64) + 1
    # The selected voxels in their box, straight from the mask: no cell list
    # (a 729 000-cell selection spent 8 ms a step listing it).
    window = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    box = np.where(mask[window], voxels[window], np.uint8(0)).astype(np.uint8)
    # The turned box is clipped to the grid when it is pasted back, so it may
    # be larger than a grid (a selected 210-cube would not
    # turn, its box of 266 was over the grid's limit of 256).
    (turned, position), = free_rotate.rotate_models([(box, tuple(int(v) for v in lo))], axis, degrees, pivot2,
                                                    limit=None)
    return turned, position


def paste_block(grid: np.ndarray, block: np.ndarray, position) -> np.ndarray:
    """Write the filled cells of `block` into `grid` at `position`, clipped to
    the grid; returns the bool mask of the cells written. Block arithmetic:
    no cell list."""
    written = np.zeros(grid.shape, dtype=bool)
    pos = np.asarray(position, dtype=np.int64)
    size = np.asarray(grid.shape, dtype=np.int64)
    lo = np.maximum(pos, 0)
    hi = np.minimum(pos + np.asarray(block.shape, dtype=np.int64), size)
    if np.any(hi <= lo):
        return written
    dst = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    src = tuple(slice(int(a), int(b)) for a, b in zip(lo - pos, hi - pos))
    part = block[src]
    filled = part != 0
    grid[dst][filled] = part[filled]
    written[dst] = filled
    return written


_NATIVE: list = []


def _native():
    """rok_native's fused live turn, or None (no native module, or ROK_PURE_TURN set)."""
    if not _NATIVE:
        import os

        mod = None
        if not os.environ.get("ROK_PURE_TURN"):
            try:
                import rok_native

                mod = rok_native if hasattr(rok_native, "live_turn") else None
            except ImportError:
                mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


def live_turn_grid(press_voxels: np.ndarray, press_mask: np.ndarray, bounds, axis: int, degrees: float, pivot2):
    """(want, new mask): the grid at one angle of a live turn, from the voxels
    and the selection at the press: the selection cleared, its turned box
    pasted back, what lands outside the grid lost. One parallel pass in the
    native module when it is there; the numpy path below is the same, cell
    for cell (tests compare them)."""
    if bounds is None:
        want = press_voxels.copy()
        want[press_mask] = 0
        return want, np.zeros(press_mask.shape, dtype=bool)
    native = _native()
    if native is not None:
        from . import free_rotate

        lo = tuple(int(v) for v in bounds[0])
        hi = tuple(int(v) for v in bounds[1])
        shape = tuple(h - l + 1 for l, h in zip(lo, hi))
        snapped = free_rotate.snap(degrees)
        _tshape, position = free_rotate.turned_place(shape, lo, axis, snapped, pivot2)
        return native.live_turn(np.ascontiguousarray(press_voxels, dtype=np.uint8),
                                np.ascontiguousarray(press_mask, dtype=bool), lo, hi, int(axis), snapped,
                                tuple(int(v) for v in position))
    return _live_turn_grid_numpy(press_voxels, press_mask, bounds, axis, degrees, pivot2)


def _live_turn_grid_numpy(press_voxels, press_mask, bounds, axis: int, degrees: float, pivot2):
    want = press_voxels.copy()
    want[press_mask] = 0
    turned, position = free_turned_block(press_voxels, press_mask, bounds, axis, degrees, pivot2)
    return want, paste_block(want, turned, position)


def free_turned_cells(voxels: np.ndarray, selection: Selection, axis: int, degrees: float, pivot2):
    """The selection turned by any angle about `pivot2` (doubled cells): (cells,
    values) of the turned voxels that land inside the grid."""
    bounds = selection.bounds()
    if bounds is None:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.uint8)
    turned, position = free_turned_block(voxels, selection.mask, bounds, axis, degrees, pivot2)
    out = np.argwhere(turned != 0)
    values = turned[out[:, 0], out[:, 1], out[:, 2]]
    out = out + np.asarray(position, dtype=np.int64)
    inside = np.all((out >= 0) & (out < np.asarray(voxels.shape)), axis=1)
    return out[inside], values[inside]


class LiveTurn:
    """A selection's turn shown in the real voxels during a ring drag, always
    from the voxels at the press, so turning back and forth never wears the
    selection down. Each step is one write of the cells that differ
    (four full writes a step, restore and redo, made a 729 000-cell
    selection crawl). The turn runs on a copy of the press voxels; the copy's
    difference from the grid is written through `set_many`, so the mask and
    the section planes hold as for any write."""

    def __init__(self, voxels: np.ndarray, selection, set_many, set_grid=None) -> None:
        self.voxels = voxels
        self.selection = selection
        self.set_many = set_many
        self.set_grid = set_grid  # writes a whole grid by its difference (StrokeRecorder.set_grid), or None
        self.key = None  # the turn shown
        self.total = 0  # voxels the shown turn changed, for the stroke's report
        self._press = None  # (voxels, selection mask, selection bounds) at the press

    def _start(self):
        if self._press is None:
            self._press = (self.voxels.copy(), self.selection.mask.copy(), self.selection.bounds())
        return self._press

    def _write(self, want: np.ndarray) -> int:
        """The grid becomes `want`: one write of the cells that differ."""
        if self.set_grid is not None:
            return int(self.set_grid(want))
        diff = want != self.voxels
        cells = np.argwhere(diff)
        return int(self.set_many(cells, want[diff])) if len(cells) else 0

    def show(self, key, turn) -> None:
        """Show `turn(voxels, selection, set_many) -> new mask`, or the voxels
        at the press when `turn` is None."""
        if self.selection is None or self.set_many is None or key == self.key:
            return
        if self._press is None and turn is None:
            return  # nothing shown yet: nothing to put back
        press_voxels, press_mask, _bounds = self._start()
        want = press_voxels.copy()
        if turn is None:
            new_mask = press_mask
        else:
            scratch = Selection(press_mask.shape)
            scratch.replace(press_mask)

            def write(cells, values) -> int:
                cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
                if len(cells):
                    want[cells[:, 0], cells[:, 1], cells[:, 2]] = np.asarray(values, dtype=np.uint8)
                return len(cells)

            new_mask = turn(want, scratch, write)
        changed = self._write(want)
        self.total = 0 if turn is None else changed  # back at the press: nothing changed
        self.selection.replace(new_mask)
        self.key = key

    def show_free(self, key, axis: int, degrees: float, pivot2) -> None:
        """Show a free turn: the selection's box turned and pasted back, all in
        block arithmetic, with no cell list (the same voxels free_rotate_cells
        writes)."""
        if self.selection is None or self.set_many is None or key == self.key:
            return
        press_voxels, press_mask, bounds = self._start()
        want, new_mask = live_turn_grid(press_voxels, press_mask, bounds, axis, degrees, pivot2)
        self.total = self._write(want)
        self.selection.replace(new_mask)
        self.key = key


def free_rotate_cells(voxels: np.ndarray, selection: Selection, axis: int, degrees: float, pivot2, set_many):
    """Turn the selected voxels by any angle: their cells empty, the turned
    voxels written, and the new selection mask returned. Cells outside the grid
    are lost, as with a quarter turn."""
    old = selection.cells()
    cells, values = free_turned_cells(voxels, selection, axis, degrees, pivot2)
    if len(old):
        set_many(old, np.zeros(len(old), dtype=np.uint8))
    if len(cells):
        set_many(cells, values)
    mask = np.zeros(selection.mask.shape, dtype=bool)
    if len(cells):
        mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return mask
