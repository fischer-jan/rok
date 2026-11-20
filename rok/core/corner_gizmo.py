"""The corner gizmo: move and scale a selection from a fixed widget at the
bottom right of the view, instead of handles on the voxels.

Three axis arrows point where the world axes point on screen. The arrow
head moves the selection along that axis, one cell on a click; the box on the shaft scales it
along that axis; the box in the centre scales it along all three. Three
rings, each the circle about one axis as the view sees it, turn the
selection about that axis around its centre, freely in tenths of a degree
with a snap to the quarter turns, as the main gizmo's rings do (Ctrl: 15
degree steps, Shift: slower); the real voxels turn during the drag, and a
click on a ring is one quarter turn. A drag of
one cell's on-screen length changes the result by one cell, so the ghost
follows the mouse at the selection's own scale. Pure numpy: the overlay
draws `corner_layout`, the stroke runs `CornerDrag` through extend_screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import math

from .edit import SELECT, Cell, SetMany, in_bounds_mask
from .geom2d import polyline_distance
from .gesture_base import Gesture
from .gizmo import FINE, LiveTurn, degrees_label, snap_degrees, snap_pivot, wrap_degrees
from .select import MovePreview, Selection, move_cells, rotate_cells
from .stretch import stretch

AXES = "XYZ"
MOVE_IDS = ("CG_MOVE_X", "CG_MOVE_Y", "CG_MOVE_Z")
SCALE_IDS = ("CG_SCALE_X", "CG_SCALE_Y", "CG_SCALE_Z")
SCALE_ALL = "CG_SCALE_ALL"
ROT_IDS = ("CG_ROT_X", "CG_ROT_Y", "CG_ROT_Z")
HANDLES = MOVE_IDS + SCALE_IDS + (SCALE_ALL,) + ROT_IDS

# Points, before the UI scale.
MARGIN_RIGHT = 88.0
MARGIN_BOTTOM = 96.0
ARM = 58.0  # centre to arrow tip
BOX_AT = 0.55  # the scale box, as a fraction of the arm
TIP_HIT = 12.0
BOX_HIT = 9.0
CENTRE_HIT = 10.0
RING = 44.0  # ring radius
RING_HIT = 6.0
RING_STEPS = 48
MIN_CELL_PX = 2.0  # an axis shorter than this on screen points at the eye: hidden
CLICK_PX = 6.0  # region pixels: a press that moves less than this is a click

MOVE = "MOVE"
SCALE = "SCALE"
ROTATE = "ROTATE"
NONE = "NONE"


def is_handle(name: str | None) -> bool:
    return name in HANDLES


@dataclass(frozen=True)
class CornerLayout:
    centre: tuple[float, float]
    dirs: dict[int, tuple[float, float]]  # unit screen direction of each shown axis
    tips: dict[int, tuple[float, float]]
    boxes: dict[int, tuple[float, float]]
    scale: float
    rings: dict[int, np.ndarray] = None  # (RING_STEPS, 2) closed polyline per axis
    ring_sign: dict[int, int] = None  # +1 when the axis points at the eye: counter-clockwise is +


def corner_layout(width: float, height: float, scale: float, cell_px: dict[int, tuple[float, float]]) -> CornerLayout:
    """The widget for a region: `cell_px[axis]` is one cell along that world
    axis in region pixels at the selection; short ones are left out."""
    s = float(scale)
    centre = (float(width) - MARGIN_RIGHT * s, MARGIN_BOTTOM * s)
    dirs, tips, boxes = {}, {}, {}
    for axis, (dx, dy) in cell_px.items():
        length = float(np.hypot(dx, dy))
        if length < MIN_CELL_PX:
            continue
        ux, uy = dx / length, dy / length
        dirs[axis] = (ux, uy)
        tips[axis] = (centre[0] + ux * ARM * s, centre[1] + uy * ARM * s)
        boxes[axis] = (centre[0] + ux * ARM * BOX_AT * s, centre[1] + uy * ARM * BOX_AT * s)
    # The rings: the circle about each axis, drawn with the other two axes as
    # the view foreshortens them (each cell vector over the longest one).
    longest = max((float(np.hypot(*v)) for v in cell_px.values()), default=0.0)
    rings, signs = {}, {}
    if longest > 0.0 and len(cell_px) == 3:
        unit = {a: np.asarray(v, dtype=np.float64) / longest for a, v in cell_px.items()}
        t = np.linspace(0.0, 2.0 * math.pi, RING_STEPS, endpoint=False)
        for axis in range(3):
            b, c = (axis + 1) % 3, (axis + 2) % 3
            pts = np.outer(np.cos(t), unit[b]) + np.outer(np.sin(t), unit[c])
            if float(np.abs(pts).max()) * RING * s < 4.0:
                continue  # edge-on: a line, not a ring
            rings[axis] = (np.asarray(centre) + pts * RING * s).astype(np.float32)
            cross = unit[b][0] * unit[c][1] - unit[b][1] * unit[c][0]
            signs[axis] = 1 if cross >= 0 else -1
    return CornerLayout(centre, dirs, tips, boxes, s, rings, signs)


def hit_test(layout: CornerLayout | None, mx: float, my: float) -> str | None:
    """The handle under the pixel: an arrow tip, a box, or the centre box."""
    if layout is None:
        return None
    s = layout.scale

    def near(p, r) -> bool:
        return float(np.hypot(mx - p[0], my - p[1])) <= r * s

    for axis, tip in layout.tips.items():
        if near(tip, TIP_HIT):
            return MOVE_IDS[axis]
    for axis, box in layout.boxes.items():
        if near(box, BOX_HIT):
            return SCALE_IDS[axis]
    if near(layout.centre, CENTRE_HIT):
        return SCALE_ALL
    best, best_d = None, RING_HIT * s
    for axis, ring in (layout.rings or {}).items():
        d = polyline_distance(ring, mx, my)
        if d <= best_d:
            best, best_d = ROT_IDS[axis], d
    return best


def scaled_block(cells: np.ndarray, values: np.ndarray, lengths) -> tuple[np.ndarray, np.ndarray]:
    """The selected block resampled to `lengths` cells per axis (as a scene
    stretch: nearest cell growing, no lost cells shrinking),
    around its centre: it grows or shrinks equally on both sides, and an odd
    change puts the extra cell on the low side. Returns (cells, values)."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if len(cells) == 0:
        return cells, np.zeros(0, dtype=np.uint8)
    lo = cells.min(axis=0)
    old = cells.max(axis=0) - lo + 1
    block = np.zeros(tuple(int(v) for v in old), dtype=np.uint8)
    rel = cells - lo
    block[rel[:, 0], rel[:, 1], rel[:, 2]] = values
    new = [max(1, int(n)) for n in lengths]
    out = stretch(block, new)
    found = np.argwhere(out != 0)
    new_lo = lo + (old - np.asarray(new, dtype=np.int64)) // 2
    return found + new_lo, out[found[:, 0], found[:, 1], found[:, 2]]


class CornerDrag(Gesture):
    """One press-drag-release on the corner gizmo (the stroke's extend_screen path)."""

    tool = SELECT
    screen_drag = True

    def __init__(
        self,
        handle: str,
        voxels: np.ndarray,
        selection: Selection,
        set_many: SetMany,
        cell_px: dict[int, tuple[float, float]],
        *,
        copy: bool = False,
        centre: tuple[float, float] = (0.0, 0.0),
        ring_sign: dict[int, int] | None = None,
        set_grid=None,
    ) -> None:
        if handle not in HANDLES:
            raise ValueError(f"not a corner handle: {handle!r}")
        self.handle = handle
        self.voxels = voxels
        self.selection = selection
        self.set_many = set_many
        self.cell_px = {a: np.asarray(v, dtype=np.float64) for a, v in cell_px.items()}
        self.copy = bool(copy) and handle in MOVE_IDS
        self.start: tuple[float, float] | None = None
        self.steps = 0
        cells = selection.cells()
        self.lo = cells.min(axis=0) if len(cells) else np.zeros(3, dtype=np.int64)
        self.lengths = (cells.max(axis=0) - self.lo + 1) if len(cells) else np.ones(3, dtype=np.int64)
        self._preview_key = None
        self._preview = None
        self.centre = (float(centre[0]), float(centre[1]))
        self.ring_sign = dict(ring_sign or {})
        self.angle0: float | None = None
        self.angle_acc = 0.0
        self.swept = False
        self.degrees = 0.0  # the turn so far, snapped (not wrapped)
        self.coarse = False  # Ctrl: 15 degree steps
        self.fine = False  # Shift: a tenth of the speed
        self._live = LiveTurn(voxels, selection, set_many, set_grid)  # the turn shown in the real voxels
        # Twice the block's centre, the pivot the turns go round.
        self.pivot2 = tuple(int(2 * int(self.lo[i]) + int(self.lengths[i])) for i in range(3))

    @property
    def axis(self) -> int | None:
        if self.handle == SCALE_ALL:
            return None
        return (MOVE_IDS + SCALE_IDS + ROT_IDS).index(self.handle) % 3

    def _angle(self, mx: float, my: float) -> float:
        return math.atan2(my - self.centre[1], mx - self.centre[0])

    def set_shift(self, shift: bool) -> None:
        """Shift during a ring drag turns slower; a move's copy flag is the press's."""
        self.fine = bool(shift)

    def turns(self) -> int:
        """Quarter turns, 0..3; 0 for a free angle. A click with no sweep is one quarter turn."""
        if self.handle not in ROT_IDS:
            return 0
        if not self.swept:
            return 1
        q = round(self.degrees / 90.0)
        return int(q) % 4 if abs(self.degrees - 90.0 * q) < 1e-9 else 0

    def turn_degrees(self) -> float:
        """The angle the release keeps, wrapped into (-180, 180]; 90 for a click."""
        return 90.0 if not self.swept else wrap_degrees(self.degrees)

    def _steps_at(self, mx: float, my: float) -> int:
        if self.start is None:
            return 0
        delta = np.array([mx - self.start[0], my - self.start[1]], dtype=np.float64)
        if self.axis is not None:
            v = self.cell_px.get(self.axis)
            if v is None or float(v @ v) <= 0.0:
                return 0
            return int(round(float(delta @ v) / float(v @ v)))
        # The centre box: up and to the right grows, down and to the left shrinks,
        # one step per mean cell length.
        unit = np.mean([np.linalg.norm(v) for v in self.cell_px.values()]) if self.cell_px else 10.0
        return int(round(float(delta[0] + delta[1]) / (2.0 * max(unit, 1.0))))

    def begin_screen(self, mx: float, my: float) -> bool:
        self.start = (float(mx), float(my))
        self.angle0 = self._angle(mx, my)
        self.dragged = False
        return True

    def extend_screen(self, mx: float, my: float) -> bool:
        if self.handle in ROT_IDS:
            before = (self.swept, self.degrees)
            angle = self._angle(mx, my)
            if self.angle0 is not None:
                step = (angle - self.angle0 + math.pi) % (2 * math.pi) - math.pi
                self.angle_acc += step * (FINE if self.fine else 1.0)
            self.angle0 = angle
            deg = math.degrees(self.angle_acc) * self.ring_sign.get(self.axis, 1)
            self.degrees = snap_degrees(deg, coarse=self.coarse)
            if self.degrees != 0.0:
                self.swept = True
            changed = (self.swept, self.degrees) != before
            if changed and self.swept:
                self._show_live()
            return changed
        before = self.offset() if self.handle in MOVE_IDS else None
        if self.start is not None and not self.dragged:
            moved = float(np.hypot(mx - self.start[0], my - self.start[1]))
            self.dragged = moved >= CLICK_PX
        steps = self._steps_at(mx, my)
        if before is not None:
            self.steps = steps
            return self.offset() != before
        if steps == self.steps:
            return False
        self.steps = steps
        return True

    def _move_steps(self) -> int:
        """A click on an arrow, with no drag, is one cell forward."""
        if self.steps == 0 and not getattr(self, "dragged", True):
            return 1
        return self.steps

    def kind(self) -> str:
        if self.handle in ROT_IDS:
            return ROTATE if self.turn_degrees() != 0.0 else NONE
        if self.handle in MOVE_IDS:
            return MOVE if self._move_steps() else NONE
        if self.steps == 0:
            return NONE
        return MOVE if self.handle in MOVE_IDS else SCALE

    def offset(self) -> Cell:
        out = [0, 0, 0]
        if self.handle in MOVE_IDS:
            out[self.axis] = self._move_steps()
        return (out[0], out[1], out[2])

    def new_lengths(self) -> tuple[int, int, int]:
        old = [int(v) for v in self.lengths]
        if self.handle in SCALE_IDS:
            old[self.axis] = max(1, old[self.axis] + self.steps)
            return (old[0], old[1], old[2])
        if self.handle == SCALE_ALL:
            big = max(old)
            factor = max(1, big + self.steps) / big
            return tuple(max(1, int(round(n * factor))) for n in old)
        return (old[0], old[1], old[2])

    def _turn_pivot(self):
        return snap_pivot(self.pivot2, self.axis) if self.turns() % 2 else self.pivot2

    def label(self) -> str:
        if self.kind() == ROTATE:
            return degrees_label(self.turn_degrees(), self.axis)
        if self.kind() == MOVE:
            text = f"{self._move_steps():+d} {AXES[self.axis]}"
            return f"copy {text}" if self.copy else text
        if self.kind() == SCALE:
            a, b = [int(v) for v in self.lengths], self.new_lengths()
            if self.axis is not None:
                return f"{AXES[self.axis]} {a[self.axis]} to {b[self.axis]}"
            return f"{a[0]} x {a[1]} x {a[2]} to {b[0]} x {b[1]} x {b[2]}"
        return ""

    def _scaled(self):
        cells = self.selection.cells()
        values = self.voxels[cells[:, 0], cells[:, 1], cells[:, 2]]
        new_cells, new_values = scaled_block(cells, values, self.new_lengths())
        inside = in_bounds_mask(new_cells, self.voxels.shape)
        return cells, new_cells[inside], new_values[inside]

    def preview(self):
        if self.handle in ROT_IDS:
            # The voxels themselves show the turn (see _live): only the angle.
            return MovePreview(kind="LABEL", label=self.label())
        if self.kind() != SCALE:
            return MovePreview(offset=self.offset(), label=self.label())
        key = (self.new_lengths(), self.selection.version)
        if key != self._preview_key:
            _src, cells, _vals = self._scaled()
            self._preview_key = key
            self._preview = MovePreview(kind="CELLS", cells=cells, label=self.label())
        return self._preview

    def finish(self) -> int:
        kind = self.kind()
        if kind == NONE:
            if self.handle in ROT_IDS:
                self._live.show(("REST",), None)  # turned back to where it started
            return 0
        total = 0

        def counted(cells, values) -> int:
            nonlocal total
            n = int(self.set_many(cells, values))
            total += n
            return n

        if kind == MOVE:
            new = move_cells(self.voxels, self.selection, self.offset(), self.copy, counted)
            self.selection.replace(new)
            return total
        if kind == ROTATE:
            if self._live.key != self._turn_key():
                self._show_live()  # a click: one quarter turn now
            return self._live.total
        src, cells, values = self._scaled()
        if len(src):
            counted(src, np.zeros(len(src), dtype=np.uint8))
        if len(cells):
            counted(cells, values)
        mask = np.zeros(self.voxels.shape, dtype=bool)
        if len(cells):
            mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
        self.selection.replace(mask)
        return total

    # --- the live turn ------------------------------------------------------------------

    def _turn_key(self):
        return (self.turns(), self.turn_degrees())

    def _show_live(self) -> None:
        """Show the turn in the real voxels (LiveTurn), always from the voxels at the press."""
        if self.kind() == ROTATE and not self.turns():
            self._live.show_free(self._turn_key(), self.axis, self.turn_degrees(), self.pivot2)
            return
        if self.kind() != ROTATE:
            turn = None
        else:
            turns, pivot2 = self.turns(), self._turn_pivot()

            def turn(v, s, w):
                return rotate_cells(v, s, self.axis, turns, pivot2, w)
        self._live.show(self._turn_key(), turn)
