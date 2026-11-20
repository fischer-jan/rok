"""Shape, Face and screen gestures, the drag preview and the gesture factory.

Imports numpy, .edit, .shape, .region, .raycast and .select only. Every
gesture follows the protocol from the design: begin(hit), extend(origin,
direction), finish() and preview(). Shape and Face gestures write on finish;
Face + Paint and Face + Select act during the drag. A ScreenGesture takes
pixels through begin_screen and extend_screen instead of rays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from . import gradient, pattern, region, shape
from .edit import (
    ATTACH,
    BOX,
    BY_COLOUR,
    CENTER,
    COLOUR_CONNECTED_CORNERS,
    COLOUR_MODES,
    CONNECTED_CORNERS,
    ERASE,
    FACE,
    LINE,
    NO_MIRROR,
    PAINT,
    PATTERN,
    STAMP_MODES,
    RECT,
    SCREEN_MODES,
    SELECT,
    SELECT_ONLY_MODES,
    VOXEL,
    Cell,
    Mirror,
    PlaneLock,
    SetMany,
    Setter,
    Stroke,
    apply_cells,
    clamp_free,
    in_bounds_mask,
    target_cell,
)
from .gesture_base import Gesture
from .raycast import Hit, raycast
from .select import ColourGesture, ConnectedGesture, PickGesture, Project, Selection, exposed_cells, screen_cells, seen_cells

LASSO_MIN_STEP = 3.0  # px between two lasso points
LASSO_MAX_VERTICES = 96  # the polygon is decimated to this on finish
ROUND_MAX = 3  # the rounding steps of a Rect: flat, round top, round ends, ellipse
ROUND_LABELS = ("R to round", "round top", "round ends", "ellipse")
ROUND_SEGMENTS = 24  # points along each rounded end


def _empty_cells() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.int64)


def _empty_points() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


@dataclass(frozen=True, eq=False)
class Preview:
    """What the overlay draws during a drag. Same field names as select.MovePreview."""

    kind: str  # 'PLANE' | 'CELLS' | 'MOVE' | 'SCREEN'
    tool: str
    cells: np.ndarray = field(default_factory=_empty_cells)  # (N, 3); empty for MOVE
    normal: Cell = (0, 0, 1)  # PLANE: the side the face quads are drawn on
    box: tuple[Cell, Cell] | None = None  # PLANE: outline from lo to hi (inclusive)
    label: str = ""  # drawn next to the mouse
    offset: Cell = (0, 0, 0)  # MOVE: translation of the selection
    points: np.ndarray = field(default_factory=_empty_points)  # SCREEN: (N, 2) pixels
    # The height stage's level: (axis, coordinate) of the plane at the top of the
    # block, drawn across the grid so the height can be lined up with other things.
    level: tuple[int, float] | None = None
    # Select only: this drag would remove cells from the selection instead of
    # adding them, so the overlay draws the outline in the Erase colour.
    subtract: bool = False


def _as_cell(values) -> Cell:
    return (int(values[0]), int(values[1]), int(values[2]))


def _axis_of(normal) -> int:
    for i in range(3):
        if int(normal[i]) != 0:
            return i
    raise ValueError("normal has no non-zero component")


class ShapeGesture(Gesture):
    """Box, Line and Center: a preview during the drag, one write on release.

    Box and Center are locked to the face plane of the first hit, so the shape
    lies in one voxel layer. Line follows the surface: its end is the cell under
    the mouse on whatever face is there, so it can run in any direction. A click without a drag is a one-cell shape.

    Box and Center have a second stage: after the release the mouse sets the
    height. Attach, and Select on the floor or a wall, start outward. Erase,
    Paint and Select on a voxel start inward. The drag can cross to the other
    side of the rectangle. A click confirms.
    """

    def __init__(
        self,
        voxels: np.ndarray,
        mode: str,
        tool: str,
        colour: int,
        mirror: Mirror,
        set_many: SetMany,
        selection: Selection | None,
        subtract: bool = False,
    ) -> None:
        self.voxels = voxels
        self.size = tuple(int(d) for d in voxels.shape)
        self.mode = mode
        self.tool = tool
        self.colour = int(colour)
        self.mirror = (bool(mirror[0]), bool(mirror[1]), bool(mirror[2]))
        self.set_many = set_many
        self.selection = selection
        self.subtract = bool(subtract)  # Select only: remove the footprint instead of adding it
        self.hit: Hit | None = None
        self.last: Cell | None = None
        self.lock = PlaneLock(2, 0.0, 0)
        self.anchor: Cell = (0, 0, 0)
        self.current: Cell = (0, 0, 0)
        self.height = 1  # layers along the normal; 1 until the height stage raises it
        self.side = 1  # +1 grows the tool's usual way; -1 grows the other way
        self.height_stage = False
        self._cells_cache: tuple | None = None  # (state key, cells) of the last cells()
        # A whole-grid write (StrokeRecorder.set_grid): a plain box writes as a
        # block through it, no cell list. None: every write lists its cells.
        self.set_grid = None
        self.solid = shape.BOX_SOLID  # what a Box or Center grows into (shape.SOLIDS)
        self.square = False  # Shift during the footprint drag: equal extents
        self.viewer: dict | None = None  # Select: take only what this camera sees (ScreenGesture)
        self.wall = 0  # Attach: a hollow primitive's wall in voxels, 0 for solid
        self.frame_colour: int | None = None  # Attach: a framed primitive's bars; None: the colour
        self._bars = None  # (N,) which of cells() are bars of a frame, or None
        self.seed = 0  # the lump pattern of a Rock or a Hill; begin() draws a new one

    def begin(self, hit: Hit) -> bool:
        """Lock the plane and set the anchor. Nothing is written."""
        cell = target_cell(self.tool, hit, self.size)
        if cell is None:
            return False
        self.lock = PlaneLock.from_hit(hit, cell)
        self.hit = hit
        self.anchor = self.current = self.last = cell
        self.side = 1
        self.seed = int(np.random.default_rng().integers(1 << 31))  # a new Rock on every drag
        return True

    def has_height_stage(self) -> bool:
        """Box and Center have the height stage with every action."""
        return self.mode in (BOX, CENTER) and self.hit is not None

    def set_shift(self, shift: bool) -> None:
        self.square = bool(shift)  # Shift during the footprint drag: a square or round base

    def begin_height(self) -> bool:
        """Start the height stage after the footprint. False when this shape has none."""
        if not self.has_height_stage():
            return False
        self.height_stage = True
        self.height = 1
        self.side = 1
        return True

    def _usual_axis(self) -> np.ndarray:
        """Unit vector of the tool's usual height: the normal for Attach and for
        a Select that starts on the floor or a wall; into the model for the rest."""
        n = np.asarray(self.hit.normal, dtype=np.float64)
        if self.tool == ATTACH or (self.tool == SELECT and self.hit.wall):
            return n
        return -n

    def _height_axis(self) -> np.ndarray:
        """Unit vector the height grows along now. The drag can reverse it."""
        return self._usual_axis() * self.side

    def _room(self, side: int) -> int:
        """How many layers fit from the footprint along `side` of the usual axis."""
        axis = self.lock.axis
        layer = int(self.lock.layer)
        if float(self._usual_axis()[axis]) * side > 0:
            return self.size[axis] - layer
        return layer + 1

    def max_height(self) -> int:
        """How many layers fit in the grid from the footprint along the height now."""
        return self._room(self.side)

    def extend_height(self, origin, direction) -> bool:
        """Set the height from the mouse ray: the point of the height line nearest the
        ray, in whole layers from the footprint. The line runs either way from the
        rectangle. True when the height or its side changed."""
        if not self.height_stage:
            return False
        v = self._usual_axis()
        d = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(d))
        if norm == 0.0:
            return False
        d = d / norm
        p = np.asarray(self.anchor, dtype=np.float64) + 0.5
        o = np.asarray(origin, dtype=np.float64)
        vd = float(v @ d)
        denom = 1.0 - vd * vd
        if denom < 1e-6:
            return False  # the ray runs along the height line: no reading
        w = o - p
        t = (float(w @ v) - float(w @ d) * vd) / denom
        # t is in cells from the footprint's centre, along the usual axis.
        # The other side starts once the ray reaches the next cell's centre.
        if t >= 0.0:
            side, along = 1, t
        else:
            side, along = -1, -t
        height = int(math.floor(along)) + 1
        height = max(1, min(self._room(side), height))
        if height == 1:
            side = 1  # the rectangle alone: keep the usual face
        if height == self.height and side == self.side:
            return False
        self.side = side
        self.height = height
        return True

    def extend(self, origin, direction) -> bool:
        """Move the far corner. True when it changed."""
        if self.mode == LINE:
            hit = raycast(self.voxels, origin, direction)
            cell = target_cell(self.tool, hit, self.size)
            if cell is None:
                return False  # off the grid: the line keeps its last end
        else:
            cell = self.lock.cell_at(origin, direction)
            if cell is None:
                return False
            if self.square:
                cell = shape.square_corner(self.anchor, cell, self.lock.axis)
            # A drag off the grid stops at the border.
            cell = clamp_free(cell, self.lock.axis, self.size)
        if cell == self.current:
            return False
        self.current = self.last = cell
        return True

    def footprint(self) -> np.ndarray:
        """(N, 3): the shape from the anchor to the current cell, one layer."""
        return shape.shape_cells(
            self.mode, self.anchor, self.current, self.lock.axis, self.size
        )

    def plain_box(self):
        """(lo, hi), inclusive, of the block when it is a plain filled box: a
        Box or Center footprint, the Box solid, no hollow wall. Its real
        voxels' surface is its six faces and its write is a block, so neither
        needs a cell list (a 200 x 200 base felt slow; the
        hollow box and the round solids keep the per-cell path). None otherwise."""
        if self.mode not in (BOX, CENTER) or self.solid != shape.BOX_SOLID or self.wall or self.hit is None:
            return None
        axis = self.lock.axis
        u, v = (i for i in range(3) if i != axis)
        a, b = self.anchor, self.current
        lo, hi = [0, 0, 0], [0, 0, 0]
        if self.mode == BOX:
            lo[u], hi[u] = min(int(a[u]), int(b[u])), max(int(a[u]), int(b[u]))
            lo[v], hi[v] = min(int(a[v]), int(b[v])), max(int(a[v]), int(b[v]))
        else:
            du, dv = abs(int(b[u]) - int(a[u])), abs(int(b[v]) - int(a[v]))
            lo[u], hi[u] = max(0, int(a[u]) - du), min(int(self.size[u]) - 1, int(a[u]) + du)
            lo[v], hi[v] = max(0, int(a[v]) - dv), min(int(self.size[v]) - 1, int(a[v]) + dv)
        step = int(self._height_axis()[axis])
        k0, k1 = int(a[axis]), int(a[axis]) + step * (int(self.height) - 1)
        lo[axis], hi[axis] = max(0, min(k0, k1)), min(int(self.size[axis]) - 1, max(k0, k1))
        if any(hi[i] < lo[i] for i in range(3)):
            return None
        return tuple(lo), tuple(hi)

    def _mirrored_boxes(self, box) -> list:
        """The box and its mirror images, each once."""
        lo, hi = box
        boxes = [(tuple(lo), tuple(hi))]
        for axis in range(3):
            if not self.mirror[axis]:
                continue
            n = int(self.size[axis]) - 1
            for blo, bhi in list(boxes):
                mlo, mhi = list(blo), list(bhi)
                mlo[axis], mhi[axis] = n - bhi[axis], n - blo[axis]
                if (tuple(mlo), tuple(mhi)) not in boxes:
                    boxes.append((tuple(mlo), tuple(mhi)))
        return boxes

    def _finish_box(self, box) -> int | None:
        """A plain box written as a block (or selected by a mask); None when
        this stroke needs the cell path after all."""
        if self.tool == SELECT:
            if self.selection is None:
                return 0
            if self.viewer is not None:
                return None  # what the camera sees: the cells' lines of sight
            lo, hi = box
            window = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
            mask = np.zeros(self.size, dtype=bool)
            mask[window] = self.voxels[window] != 0
            if self.subtract:
                self.selection.remove_mask(mask)
            else:
                self.selection.add_mask(mask)
            return 0
        if self.set_grid is None:
            return None
        want = np.array(self.voxels, copy=True)
        colour = np.uint8(self.colour)
        for lo, hi in self._mirrored_boxes(box):
            sub = want[tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))]
            if self.tool == ATTACH:
                sub[sub == 0] = colour  # as apply_cells: an empty cell takes the colour
            elif self.tool == ERASE:
                sub[...] = 0
            else:
                sub[sub != 0] = colour  # Paint: a filled cell takes the colour
        return int(self.set_grid(want))

    def cells(self) -> np.ndarray:
        """(N, 3): the footprint grown into the chosen solid, `height` layers along
        the height axis. Line and a plain box of height 1 are the footprint itself.
        Built once per state: the preview and finish share it (else
        a 490 000-cell box was listed twice on its click)."""
        key = (self.mode, self.anchor, self.current, self.lock.axis, self.height, self.side, self.wall,
               self.solid, self.seed, None if self.hit is None else self.hit.normal)
        if self._cells_cache is not None and self._cells_cache[0] == key:
            return self._cells_cache[1]
        cells = self.footprint()
        self._bars = None
        if self.mode != LINE and len(cells) and self.hit is not None \
                and not (self.solid == shape.BOX_SOLID and self.height <= 1):
            step = int(self._height_axis()[self.lock.axis])
            cells, bars = shape.solid_parts(
                cells, self.lock.axis, step, self.height, self.solid, self.size, wall=self.wall, seed=self.seed
            )
            self._bars = bars if bars.any() else None
        self._cells_cache = (key, cells)
        return cells

    def preview(self) -> Preview:
        label = shape.shape_label(self.mode, self.anchor, self.current, self.lock.axis)
        if self.height_stage:
            # The block: the cells themselves (a plain box: its six faces, through
            # `box`), the height in the label, and the level plane at the top of
            # the block across the grid.
            axis = self.lock.axis
            sign = int(self._height_axis()[axis])
            base = float(self.anchor[axis]) + (1.0 if sign < 0 else 0.0)
            level = (axis, base + sign * float(self.height))
            hollow = ""
            if self.tool == ATTACH:
                hollow = f" · hollow {self.wall}" if self.wall else " · scroll for hollow"
            box = self.plain_box()
            cells = _empty_cells() if box is not None else self.cells()
            return Preview(
                kind="CELLS", tool=self.tool, cells=cells, box=box,
                label=f"{label} x {self.height}{hollow}", level=level, subtract=self.subtract,
            )
        cells = self.cells()
        if self.mode == LINE and self.tool != SELECT:
            # A 3D line: draw the cells themselves, no plane or box.
            return Preview(kind="CELLS", tool=self.tool, cells=cells, label=label)
        lo = cells.min(axis=0).tolist()
        hi = cells.max(axis=0).tolist()
        normal = self.hit.normal if self.hit is not None else (0, 0, 1)
        return Preview(
            kind="PLANE",
            tool=self.tool,
            cells=cells,
            normal=normal,
            box=(_as_cell(lo), _as_cell(hi)),
            label=label,
            subtract=self.subtract,
        )

    def finish(self) -> int:
        """Write the shape, or add the filled voxels inside it to the selection.

        Select works like the other actions: the footprint, then the height
        stage sets how deep the block reaches, and the filled cells in it are
        selected (or dropped with subtract)."""
        box = self.plain_box()
        if box is not None:
            done = self._finish_box(box)
            if done is not None:
                return done
        if self.tool == SELECT:
            if self.selection is not None:
                cells = self.cells()
                if len(cells):
                    filled = cells[self.voxels[cells[:, 0], cells[:, 1], cells[:, 2]] != 0]
                    if self.viewer is not None and len(filled):
                        filled = filled[seen_cells(self.voxels, filled, **self.viewer)]
                    if self.subtract:
                        self.selection.remove_cells(filled)
                    else:
                        self.selection.add_cells(filled)
            return 0
        cells = self.cells()
        if self._bars is not None and self.tool == ATTACH and self.frame_colour is not None:
            # A framed primitive: the faces in the colour, the bars in the frame's
            return (apply_cells(self.voxels, ATTACH, self.colour, cells[~self._bars], self.set_many, self.mirror)
                    + apply_cells(self.voxels, ATTACH, self.frame_colour, cells[self._bars], self.set_many,
                                  self.mirror))
        return apply_cells(
            self.voxels, self.tool, self.colour, cells, self.set_many, self.mirror
        )


class FaceGesture(Gesture):
    """Face mode: the connected coplanar same-colour region under the mouse.

    Attach extrudes it by N layers and Erase pushes it in by N layers; a click
    is one layer, a drag along the normal sets N. Paint and Select act on the
    region at once and on every further region the drag crosses.
    """

    def __init__(
        self,
        voxels: np.ndarray,
        tool: str,
        colour: int,
        mirror: Mirror,
        set_many: SetMany,
        selection: Selection | None,
        subtract: bool = False,
    ) -> None:
        self.voxels = voxels
        self.size = tuple(int(d) for d in voxels.shape)
        self.tool = tool
        self.colour = int(colour)
        self.mirror = (bool(mirror[0]), bool(mirror[1]), bool(mirror[2]))
        self.set_many = set_many
        self.selection = selection
        self.subtract = bool(subtract)  # Select only: remove the region instead of adding it
        self.hit: Hit | None = None
        self.last: Cell | None = None
        self.base = _empty_cells()
        self.first = 1
        self.normal: Cell = (0, 0, 1)
        self.axis = 2
        self.centre = np.zeros(3, dtype=np.float64)
        self.steps = 1
        self.s0: float | None = None
        self.painted: set[Cell] = set()

    def begin(self, hit: Hit) -> bool:
        """Find the region. Paint and Select act on it now; the rest wait for finish."""
        if hit.wall:
            if self.tool != ATTACH:
                return False
            self.base = region.wall_region(self.voxels, hit.target, hit.normal)
            self.first = 0
        else:
            self.base = region.face_region(self.voxels, hit.voxel, hit.normal)
            self.first = 1
        if len(self.base) == 0:
            return False
        self.normal = hit.normal
        self.axis = _axis_of(hit.normal)
        self.centre = self.base.mean(axis=0) + 0.5
        self.hit = hit
        self.last = target_cell(self.tool, hit, self.size)
        self.steps = 1
        self.s0 = None
        self.painted = set()
        if self.tool == PAINT:
            apply_cells(
                self.voxels, PAINT, self.colour, self.base, self.set_many, self.mirror
            )
            self._remember(self.base)
        elif self.tool == SELECT:
            self._select(self.base)
            self._remember(self.base)
        return True

    def _select(self, cells: np.ndarray) -> None:
        """Add cells to the selection, or with subtract, remove them from it."""
        if self.selection is None:
            return
        if self.subtract:
            self.selection.remove_cells(cells)
        else:
            self.selection.add_cells(cells)

    def _remember(self, cells: np.ndarray) -> None:
        self.painted.update(_as_cell(c) for c in cells.tolist())

    def extend(self, origin, direction) -> bool:
        if self.tool in (ATTACH, ERASE):
            return self._extend_layers(origin, direction)
        return self._extend_regions(origin, direction)

    def _extend_layers(self, origin, direction) -> bool:
        """Set the layer count from the drag along the normal. True when it changed."""
        s = shape.axis_drag_param(self.centre, self.normal, origin, direction)
        if s is None:
            return False
        if self.s0 is None:
            # begin has only the hit, so the reference comes from the first move.
            self.s0 = s
            return False
        delta = s - self.s0 if self.tool == ATTACH else self.s0 - s
        steps = max(1, min(self.size[self.axis], 1 + round(delta)))
        if steps == self.steps:
            return False
        self.steps = steps
        return True

    def _extend_regions(self, origin, direction) -> bool:
        """Paint or select every region the drag crosses, each once."""
        hit = raycast(self.voxels, origin, direction, walls=False)
        if hit is None or hit.voxel in self.painted:
            return False
        cells = region.face_region(self.voxels, hit.voxel, hit.normal)
        if self.tool == PAINT:
            changed = apply_cells(
                self.voxels, PAINT, self.colour, cells, self.set_many, self.mirror
            )
        else:
            self._select(cells)
            changed = 1
        self._remember(cells)
        self.last = hit.voxel
        return changed > 0

    def layers(self) -> np.ndarray:
        """(N, 3): the cells Attach adds or Erase removes, inside the grid."""
        normal = np.asarray(self.normal, dtype=np.int64)
        if self.tool == ATTACH:
            ks = range(self.first, self.first + self.steps)
            parts = [self.base + k * normal for k in ks]
        else:
            parts = [self.base - k * normal for k in range(self.steps)]
        cells = np.concatenate(parts) if parts else _empty_cells()
        return cells[in_bounds_mask(cells, self.size)]

    def preview(self) -> Preview | None:
        if self.tool == ATTACH:
            label = f"+{self.steps}"
        elif self.tool == ERASE:
            label = f"-{self.steps}"
        else:
            return None
        return Preview(
            kind="CELLS",
            tool=self.tool,
            cells=self.layers(),
            normal=self.normal,
            label=label,
        )

    def finish(self) -> int:
        if self.tool not in (ATTACH, ERASE):
            return 0
        return apply_cells(
            self.voxels, self.tool, self.colour, self.layers(), self.set_many, self.mirror
        )


def rounded_rect(start, end, step: int, segments: int = ROUND_SEGMENTS) -> np.ndarray:
    """The corners of the rectangle from `start` to `end`, with its ends rounded
    by `step` (0 flat, 1 round top, 2 round top and bottom, 3 ellipse). The
    rounding lies inside the rectangle, so the drag shows the extent you get. A
    round end is a half ellipse as wide as the rectangle and as tall as half the
    width, or as the rectangle allows; "top" is the upper side on the screen
    (the larger y). (P, 2) float64, counter-clockwise from the bottom left."""
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    xl, xr = min(x0, x1), max(x0, x1)
    yb, yt = min(y0, y1), max(y0, y1)
    w, h = xr - xl, yt - yb
    step = max(0, min(ROUND_MAX, int(step)))
    if step == 0 or w <= 0 or h <= 0:
        return np.array([[xl, yb], [xr, yb], [xr, yt], [xl, yt]], dtype=np.float64)
    rx = w / 2.0
    if step == 1:
        ry_top, ry_bottom = min(rx, h), 0.0
    elif step == 2:
        ry_top = ry_bottom = min(rx, h / 2.0)
    else:
        ry_top = ry_bottom = h / 2.0
    cx = (xl + xr) / 2.0
    t = np.linspace(0.0, np.pi, segments + 1)
    top = np.stack([cx + rx * np.cos(t), (yt - ry_top) + ry_top * np.sin(t)], axis=1)  # right to left over the top
    if ry_bottom > 0:
        u = np.linspace(np.pi, 2.0 * np.pi, segments + 1)
        bottom = np.stack([cx + rx * np.cos(u), (yb + ry_bottom) + ry_bottom * np.sin(u)], axis=1)  # left to right
    else:
        bottom = np.array([[xl, yb], [xr, yb]])
    return np.concatenate([bottom, top], axis=0)


class ScreenGesture(Gesture):
    """Rect and Lasso: a polygon drawn on screen takes every filled voxel whose
    centre projects inside it, interior cells included. One write on release.

    The operator feeds pixels: `begin_screen` at the press, `extend_screen` on
    every move. `begin` and `extend` exist for the protocol and do nothing.
    """

    screen_drag = True

    def __init__(
        self,
        voxels: np.ndarray,
        mode: str,
        tool: str,
        colour: int,
        set_many: SetMany,
        selection: Selection | None,
        project: Project,
        mirror: Mirror = NO_MIRROR,
        subtract: bool = False,
    ) -> None:
        if mode not in SCREEN_MODES:
            raise ValueError(f"not a screen mode: {mode!r}")
        self.voxels = voxels
        self.mode = mode
        self.tool = tool
        self.colour = int(colour)
        self.set_many = set_many
        self.selection = selection
        self.project = project
        self.mirror = (bool(mirror[0]), bool(mirror[1]), bool(mirror[2]))
        self.subtract = bool(subtract)  # Select only: remove the picked cells instead
        # Select only: the camera ({"eye": ...} or {"toward": ...}) when only the
        # voxels it sees are taken, "Visible only"; None takes every one inside.
        self.viewer: dict | None = None
        self.hit: Hit | None = None
        self.last: Cell | None = None
        self.start: tuple[float, float] = (0.0, 0.0)
        self.end: tuple[float, float] = (0.0, 0.0)
        self.points: list[tuple[float, float]] = []
        self.round = 0  # a Rect only: 0 flat .. ROUND_MAX, set by R, the arrow keys or scrolling during the drag

    def set_round(self, step: int) -> bool:
        """Round the rectangle's ends (Rect only). True when the step changed."""
        step = max(0, min(ROUND_MAX, int(step)))
        if self.mode != RECT or step == self.round:
            return False
        self.round = step
        return True

    def step_round(self, delta: int, wrap: bool = False) -> bool:
        """Move the rounding `delta` steps (Rect only). With `wrap`, past the last
        step comes the first (the R key cycles); else it stops at the ends."""
        step = self.round + int(delta)
        if wrap:
            step %= ROUND_MAX + 1
        return self.set_round(step)

    def begin(self, hit: Hit | None) -> bool:
        """Any hit or None: the gesture starts anywhere on screen."""
        self.hit = hit
        return True

    def begin_screen(self, mx: float, my: float) -> None:
        self.start = self.end = (float(mx), float(my))
        self.points = [self.start]

    def extend(self, origin, direction) -> bool:
        return False

    def extend_screen(self, mx: float, my: float) -> bool:
        """Move the far corner, or add a lasso point. True when the polygon changed."""
        p = (float(mx), float(my))
        if self.mode == RECT:
            if p == self.end:
                return False
            self.end = p
            return True
        last = self.points[-1] if self.points else self.start
        if math.hypot(p[0] - last[0], p[1] - last[1]) < LASSO_MIN_STEP:
            return False
        self.points.append(p)
        return True

    def _lattice_rect(self):
        """The dragged rectangle moved onto the cells it picks: (start, end) in
        pixels, or None. A rounded end is symmetric about the middle of the
        rectangle, but cells are picked by their centres, which sit anywhere
        against that middle, so the picked shape came out up to half a cell
        lopsided. Fitting the rectangle to the picked columns and rows fixes the
        middle. It needs a regular lattice (a straight-on view); in perspective
        the cell centres are not evenly spaced and the drag stays as drawn."""
        (x0, y0), (x1, y1) = self.start, self.end
        plain = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        cells = screen_cells(self.voxels, self.project, plain)
        if len(cells) == 0:
            return None
        px = np.asarray(self.project(cells.astype(np.float64) + 0.5), dtype=np.float64)
        if not np.isfinite(px).all():
            return None
        fitted = []
        for axis in (0, 1):
            values = np.unique(np.round(px[:, axis], 6))
            if len(values) < 2:
                return None
            gaps = np.diff(values)
            pitch = float(gaps.min())
            if pitch <= 1e-6 or np.abs(gaps / pitch - np.round(gaps / pitch)).max() > 1e-3:
                return None  # not an even lattice (perspective): leave the drag alone
            fitted.append((float(values[0]) - pitch / 2.0, float(values[-1]) + pitch / 2.0))
        (xl, xr), (yb, yt) = fitted
        return (xl, yb), (xr, yt)

    def polygon(self) -> np.ndarray:
        """(P, 2) float64: the four corners, or the lasso points."""
        if self.mode == RECT:
            if self.round == 0:
                (x0, y0), (x1, y1) = self.start, self.end
                return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
            return rounded_rect(self.start, self.end, self.round)
        return np.array(self.points, dtype=np.float64).reshape(-1, 2)

    def preview(self) -> Preview:
        return Preview(
            kind="SCREEN", tool=self.tool, points=self.polygon().astype(np.float32),
            subtract=self.subtract, label=ROUND_LABELS[self.round] if self.mode == RECT else "",
        )

    def finish(self) -> int:
        """Pick the cells inside the polygon and select, erase or paint them."""
        polygon = self.polygon()
        if self.mode == RECT:
            if self.start[0] == self.end[0] or self.start[1] == self.end[1]:
                return 0  # an empty rectangle selects nothing
            if self.round:
                fitted = self._lattice_rect()
                if fitted is not None:
                    polygon = rounded_rect(fitted[0], fitted[1], self.round)
        else:
            if len(polygon) < 3:
                return 0
            step = max(1, math.ceil(len(polygon) / LASSO_MAX_VERTICES))
            polygon = polygon[::step]
        if self.tool == SELECT and self.viewer is not None:
            # What the camera sees: only a cell with an open face can be seen,
            # so only those are candidates (a solid model's
            # inside is most of its cells, and every one was tested).
            cells = screen_cells(self.voxels, self.project, polygon, cells=exposed_cells(self.voxels))
            if len(cells):
                cells = cells[seen_cells(self.voxels, cells, **self.viewer)]
        else:
            cells = screen_cells(self.voxels, self.project, polygon)
        if len(cells) == 0:
            return 0
        if self.tool == SELECT:
            if self.selection is not None:
                if self.subtract:
                    self.selection.remove_cells(cells)
                else:
                    self.selection.add_cells(cells)
            return 0
        if self.tool == ATTACH:
            return 0
        return apply_cells(
            self.voxels, self.tool, self.colour, cells, self.set_many, self.mirror
        )


class PatternGesture(Gesture):
    """Pattern mode: stamp a box of voxels (the clipboard, turned) on the surface.

    A click stamps once. A drag stamps again whenever the box moves to another
    place, as the voxel brush does, so the writes happen during the drag and
    `finish` has nothing to do. Attach fills empty cells only, in the current
    colour (the stamp is a shape then), and the ray skips the voxels it placed; Erase clears the filled cells under the box and the
    ray still stops at them, so one stroke goes one stamp deep; Paint gives
    them the pattern's colours; Select selects them.

    `holes` (Attach only): the box sinks into the surface as
    Erase's does; its filled cells are built (a voxel already there stays) and
    its empty cells are cleared, so a window frame cuts its opening too.
    Carving the shape out is Erase's stamp: the app asks for that.
    """

    def __init__(
        self,
        voxels: np.ndarray,
        tool: str,
        box: np.ndarray,
        mirror: Mirror,
        set_many: SetMany,
        selection: Selection | None,
        remap: np.ndarray | None = None,
        subtract: bool = False,
        colour: int = 0,
        holes: bool = False,
    ) -> None:
        self.voxels = voxels
        self.holes = bool(holes) and tool == ATTACH
        self.size = tuple(int(d) for d in voxels.shape)
        self.tool = tool
        self.colour = int(colour)  # Attach stamps the shape in this colour
        self.box = np.asarray(box, dtype=np.uint8)
        self.mirror = (bool(mirror[0]), bool(mirror[1]), bool(mirror[2]))
        self.set_many = set_many
        self.selection = selection
        self.remap = remap
        self.subtract = bool(subtract)  # Select only: remove the stamped cells instead
        self.hit: Hit | None = None
        self.last: Cell | None = None
        self.origin: Cell | None = None
        self.placed: set[Cell] = set()  # Attach: the ray sees through these
        self.erased: set[Cell] = set()  # Erase: the ray still stops at these

    def _place(self, hit: Hit) -> bool:
        """Move the box to this hit. False when it cannot go there or did not move."""
        origin = pattern.placement(self.box, hit, ERASE if self.holes else self.tool)
        if origin is None or origin == self.origin:
            return False
        self.hit = hit
        self.origin = origin
        target = target_cell(self.tool, hit, self.size)
        self.last = target if target is not None else hit.voxel
        return True

    def begin(self, hit: Hit) -> bool:
        if not self._place(hit):
            return False
        self._stamp()
        return True

    def extend(self, origin, direction) -> bool:
        """Stamp again when the box lands somewhere new. True when a cell changed."""
        if self.holes:
            # Into the surface, as Erase: the ray stops at the cells this stroke
            # cleared and sees through the ones it built.
            hit = raycast(self.voxels, origin, direction, walls=False, blocked=self.erased, ignore=self.placed)
        elif self.tool == ATTACH:
            hit = raycast(self.voxels, origin, direction, walls=True, ignore=self.placed)
        else:
            hit = raycast(self.voxels, origin, direction, walls=False, blocked=self.erased)
        if hit is None or not self._place(hit):
            return False
        return self._stamp()

    def cells(self) -> np.ndarray:
        """(N, 3): the filled cells of the box at its current place."""
        if self.origin is None:
            return _empty_cells()
        return pattern.box_cells(self.box, self.origin, self.size)[0]

    def _stamp(self) -> bool:
        cells, values = pattern.box_cells(self.box, self.origin, self.size, self.remap)
        if self.tool == ATTACH and self.colour > 0:
            values = np.full(len(cells), self.colour, dtype=np.uint8)  # the shape, in the current colour
        if self.tool == SELECT:
            if self.selection is None or len(cells) == 0:
                return False
            filled = cells[self.voxels[cells[:, 0], cells[:, 1], cells[:, 2]] != 0]
            version = self.selection.version
            if self.subtract:
                self.selection.remove_cells(filled)
            else:
                self.selection.add_cells(filled)
            return self.selection.version != version
        written = pattern.stamp(self.voxels, self.tool, cells, values, self.set_many, self.mirror)
        if self.holes:
            empties = pattern.empty_cells(self.box, self.origin, self.size)
            cleared = pattern.stamp(self.voxels, ERASE, empties, np.zeros(len(empties), np.uint8),
                                    self.set_many, self.mirror)
            self.erased.update(map(tuple, cleared.tolist()))
            self.placed.update(map(tuple, written.tolist()))
            return len(written) + len(cleared) > 0
        if len(written) == 0:
            return False
        if self.tool == ATTACH:
            self.placed.update(map(tuple, written.tolist()))
        elif self.tool == ERASE:
            self.erased.update(map(tuple, written.tolist()))
        return True

    def preview(self) -> Preview:
        sx, sy, sz = self.box.shape
        return Preview(
            kind="CELLS", tool=self.tool, cells=self.cells(),
            label=f"{sx} x {sy} x {sz}", subtract=self.subtract,
        )

    def finish(self) -> int:
        """Nothing deferred: every stamp was written during the drag."""
        return 0


class GradientGesture(Gesture):
    """Paint's Gradient mode: drag across the model and its voxels take a ramp
    of colours, the start colour where the drag began, the end colour where it
    is now. The real voxels change on every move; the drag's writes are one
    stroke, so one undo step.

    `cells` are the voxels it paints (the selection's, or every filled one),
    `slots` the palette indices of the ramp, first to last. Where the mouse
    leaves the model the end slides on the plane through the start that faces
    the view.
    """

    def __init__(self, voxels: np.ndarray, cells: np.ndarray, slots, set_many: SetMany, dither: bool = False) -> None:
        self.voxels = voxels
        self.cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        self.slots = np.asarray(slots, dtype=np.uint8)
        self.set_many = set_many
        self.dither = bool(dither)
        self.tool = PAINT
        self.hit: Hit | None = None
        self.last = None
        self.start: np.ndarray | None = None
        self.end: np.ndarray | None = None
        self.start_px: tuple[float, float] | None = None
        self.mouse_px: tuple[float, float] | None = None
        self.written = 0

    def set_mouse(self, mx: float, my: float) -> None:
        self.mouse_px = (float(mx), float(my))
        if self.start_px is None:
            self.start_px = self.mouse_px

    def begin(self, hit: Hit) -> bool:
        if hit is None or hit.wall or len(self.cells) == 0:
            return False
        self.hit = hit
        self.last = hit.voxel
        self.start = np.asarray(hit.voxel, dtype=np.float64) + 0.5
        return True

    def extend(self, origin, direction) -> bool:
        if self.start is None:
            return False
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        hit = raycast(self.voxels, origin, direction, walls=False)
        if hit is not None:
            end = np.asarray(hit.voxel, dtype=np.float64) + 0.5
            self.last = hit.voxel
        else:  # off the model: on the plane through the start that faces the view
            dd = float(direction @ direction) or 1.0
            end = origin + direction * float((self.start - origin) @ direction) / dd
        if self.end is not None and np.allclose(end, self.end):
            return False
        self.end = end
        if float(np.linalg.norm(end - self.start)) < 0.75:
            return False  # not dragged yet
        band = gradient.bands(self.cells, self.start, end, len(self.slots), self.dither)
        values = self.slots[band]
        current = self.voxels[self.cells[:, 0], self.cells[:, 1], self.cells[:, 2]]
        change = current != values
        if not change.any():
            return False
        self.set_many(self.cells[change], values[change])
        self.written += int(change.sum())
        return True

    def preview(self) -> Preview:
        points = np.zeros((0, 2), dtype=np.float32)
        if self.start_px is not None and self.mouse_px is not None:
            points = np.array([self.start_px, self.mouse_px], dtype=np.float32)
        return Preview(kind="SCREEN", tool=PAINT, points=points, label=f"{len(self.slots)} colours")

    def finish(self) -> int:
        """Every write happened during the drag."""
        return self.written


def make_gesture(
    mode: str,
    tool: str,
    voxels: np.ndarray,
    colour: int,
    mirror: Mirror,
    setter: Setter,
    set_many: SetMany,
    selection: Selection | None,
    project: Project | None = None,
    radius: int = 1,
    pattern_box: np.ndarray | None = None,
    remap: np.ndarray | None = None,
    subtract: bool = False,
    holes: bool = False,
    palette: np.ndarray | None = None,
    tolerance: int = 0,
):
    """The gesture object for a mode and an effective tool.

    None for a screen mode with Attach: Rect and Lasso work with Erase, Paint
    and Select only. None for Pattern mode without a `pattern_box` (an empty
    clipboard). `project` is needed for the screen modes, `radius` is the
    brush size in Voxel mode (every tool), `remap` fits the pattern's colours in.
    `subtract` matters for a Select gesture only: it removes cells from the
    selection instead of adding them.
    """
    if mode in STAMP_MODES:
        if pattern_box is None:
            return None
        # Clone keeps the copied colours: no current colour for Attach.
        return PatternGesture(
            voxels, tool, pattern_box, mirror, set_many, selection, remap, subtract,
            colour if mode == PATTERN else 0, holes
        )
    if mode in SCREEN_MODES:
        if tool == ATTACH:
            return None
        return ScreenGesture(
            voxels, mode, tool, colour, set_many, selection, project, mirror, subtract
        )
    if mode in SELECT_ONLY_MODES:
        if tool == SELECT and mode in COLOUR_MODES:
            return ColourGesture(voxels, selection, joined=mode != BY_COLOUR,
                                 corners=mode == COLOUR_CONNECTED_CORNERS, subtract=subtract,
                                 palette=palette, tolerance=tolerance)
        if tool == SELECT:
            return ConnectedGesture(voxels, selection, mode == CONNECTED_CORNERS, subtract)
        mode = VOXEL  # the other tools have no Connected mode: they work one voxel at a time
    if mode == VOXEL:
        if tool == SELECT:
            return PickGesture(voxels, selection, radius, subtract)
        return Stroke(voxels, tool, colour, setter, mirror, brush=radius, set_many=set_many)
    if mode == FACE:
        return FaceGesture(voxels, tool, colour, mirror, set_many, selection, subtract)
    return ShapeGesture(voxels, mode, tool, colour, mirror, set_many, selection, subtract)
