"""Tool rules and strokes. Writes go through an injected setter, so this module
is testable alone. It imports numpy and .raycast only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .gesture_base import Gesture
from .raycast import Hit, raycast

ATTACH = "ATTACH"
ERASE = "ERASE"
PAINT = "PAINT"
SELECT = "SELECT"
TOOLS = (ATTACH, ERASE, PAINT, SELECT)
TOOL_ITEMS = (
    (ATTACH, "Attach", "Add a voxel on the hovered face"),
    (ERASE, "Erase", "Remove the hovered voxel"),
    (PAINT, "Paint", "Recolour the hovered voxel"),
    (SELECT, "Select", "Select voxels; drag a selected voxel to move it"),
)

VOXEL = "VOXEL"
FACE = "FACE"
BOX = "BOX"
LINE = "LINE"
CENTER = "CENTER"
RECT = "RECT"
LASSO = "LASSO"
PATTERN = "PATTERN"
MODES = (VOXEL, FACE, BOX, LINE, CENTER, RECT, LASSO, PATTERN)
# Attach alone: the pattern stamp with the copied voxels' own colours (Pattern
# builds the copied shape in the current colour). Not in MODES: the Tab wheel
# keeps its eight; Attach's flyout offers it.
CLONE = "CLONE"
STAMP_MODES = (PATTERN, CLONE)
# Paint alone: a drag paints a ramp of colours across the model. Not in MODES either.
GRADIENT = "GRADIENT"
# Select alone: a click takes every voxel joined to the clicked one, face to
# face (CONNECTED) or also by an edge or a corner (CONNECTED_CORNERS). Not in
# MODES: the Tab wheel keeps its eight; Select's flyout offers these two.
CONNECTED = "CONNECTED"
CONNECTED_CORNERS = "CONNECTED_CORNERS"
BY_COLOUR = "BY_COLOUR"  # Select: every voxel of the clicked voxel's colour
COLOUR_CONNECTED = "COLOUR_CONNECTED"  # ...only those joined to it face to face
COLOUR_CONNECTED_CORNERS = "COLOUR_CONNECTED_CORNERS"  # ...joined by a face, an edge or a corner
SELECT_ONLY_MODES = (CONNECTED, CONNECTED_CORNERS, BY_COLOUR, COLOUR_CONNECTED, COLOUR_CONNECTED_CORNERS)
COLOUR_MODES = (BY_COLOUR, COLOUR_CONNECTED, COLOUR_CONNECTED_CORNERS)  # the clicked colour only
MODE_ITEMS = (
    (VOXEL, "Voxel", "One voxel; drag to continue"),
    (FACE, "Face", "The connected face of one colour; drag to extrude"),
    (BOX, "Box", "Drag a rectangle on the face plane; release, then set its height either way and click"),
    (LINE, "Line", "Drag a straight line on the face plane"),
    (CENTER, "Center", "Drag a rectangle that grows from the centre; release, then set its height either way"),
    (RECT, "Marquee", "Drag a rectangle on screen; every voxel inside, through the model, is taken"),
    (LASSO, "Lasso", "Draw a loop on screen; every voxel inside is taken"),
    (PATTERN, "Pattern", "Stamp the copied voxels on the surface; R turns them, Shift+R tips them"),
    (CLONE, "Clone", "Stamp the copied voxels with their own colours; R turns them, Shift+R tips them"),
    (GRADIENT, "Gradient", "Drag across the model: a ramp of colours from the current colour to the end colour"),
    (CONNECTED, "Connected", "Every voxel joined to the clicked one face to face"),
    (CONNECTED_CORNERS, "Connected by corners", "Every voxel joined to the clicked one by a face, an edge or a corner"),
    (BY_COLOUR, "By colour", "Every voxel of the clicked voxel's colour"),
    (COLOUR_CONNECTED, "By colour, connected by faces",
     "The clicked voxel's colour, where it joins the clicked voxel face to face"),
    (COLOUR_CONNECTED_CORNERS, "By colour, connected by corners",
     "The clicked voxel's colour, where it joins the clicked voxel by a face, an edge or a corner"),
)
# Modes that work on the screen instead of on a face. Not for Attach.
SCREEN_MODES = (RECT, LASSO)
def modes_for(tool: str) -> tuple[str, ...]:
    """The modes that do something with `tool`: every mode, minus the screen modes for Attach."""
    if tool == ATTACH:
        return tuple(m for m in MODES if m not in SCREEN_MODES)
    return MODES
# Hover-only tool name: the press activates the hovered model.
ACTIVATE = "ACTIVATE"

Cell = tuple[int, int, int]
Mirror = tuple[bool, bool, bool]
NO_MIRROR: Mirror = (False, False, False)
# (x, y, z, value) -> True when the value changed.
Setter = Callable[[int, int, int, int], bool]
# (cells (N, 3), values (N,)) -> number of cells changed.
SetMany = Callable[[np.ndarray, np.ndarray], int]

_PLANE_EPS = 1e-9


def effective_tool(base: str, alt: bool, shift: bool) -> str:
    """Option/Alt gives ERASE, else Shift gives PAINT (or keeps SELECT), else the base.

    Select never erases: with Option held (alone or with Shift) it stays
    SELECT and takes voxels out of the selection. Ctrl never erases.
    """
    if base == SELECT and alt:
        return SELECT
    if alt:
        return ERASE
    if shift:
        return SELECT if base == SELECT else PAINT
    return base


def select_subtract(tool: str, alt: bool, shift: bool) -> bool:
    """True when an effective SELECT tool should remove cells instead of adding
    them: Option held, with or without Shift. Takes the tool `effective_tool`
    returned."""
    return tool == SELECT and bool(alt)


def press_tool(base: str, forced: str | None, alt: bool, shift: bool) -> tuple[str, bool]:
    """(tool, subtract) for a press. `forced` is the tool a key binding names
    (right-click names ERASE), None for the header tool with its modifiers.
    Select never erases: a right-click there deselects, as Option does."""
    if forced is None:
        tool = effective_tool(base, alt, shift)
        return tool, select_subtract(tool, alt, shift)
    if forced == ERASE and base == SELECT:
        return SELECT, True
    return forced, False


def select_extending(tool: str, shift: bool, subtract: bool) -> bool:
    """True when Select is adding (Shift) or subtracting (Option+Shift).

    Spatial cage handles then yield so a press next to the selection is a pick.
    """
    return tool == SELECT and (bool(shift) or bool(subtract))


def in_bounds(cell: Cell, size) -> bool:
    """True when the cell lies inside the grid. z < 0 is never in bounds."""
    for i in range(3):
        if cell[i] < 0 or cell[i] >= int(size[i]):
            return False
    return True


def in_bounds_mask(cells: np.ndarray, size) -> np.ndarray:
    """(N,) bool: True for each row of an (N, 3) cell array inside the grid."""
    cells = np.asarray(cells).reshape(-1, 3)
    limit = np.asarray([int(size[0]), int(size[1]), int(size[2])])
    return np.all((cells >= 0) & (cells < limit), axis=1)


def target_cell(tool: str, hit: Hit | None, size) -> Cell | None:
    """The cell a tool acts on for a hit, or None when there is nothing to do."""
    if hit is None:
        return None
    if tool == ATTACH:
        cell = hit.target
        return cell if in_bounds(cell, size) else None
    if tool == SELECT:
        return hit.target if hit.wall else hit.voxel
    if hit.wall:
        return None
    return hit.voxel


def clamp_free(cell: Cell, axis: int, size) -> Cell:
    """Clamp the two coordinates other than axis into [0, size - 1]."""
    out = list(cell)
    for i in range(3):
        if i != axis:
            out[i] = max(0, min(int(size[i]) - 1, int(out[i])))
    return (out[0], out[1], out[2])


def plane_crossing(
    origin, direction, axis: int, plane: float, layer: int
) -> Cell | None:
    """Cell where the ray crosses coordinate[axis] == plane, in the given layer.

    No bounds test: the cell can lie outside the grid. None when the ray runs
    parallel to the plane or crosses it behind the origin.
    """
    d_axis = float(direction[axis])
    if d_axis == 0.0:
        return None
    t = (float(plane) - float(origin[axis])) / d_axis
    if t <= _PLANE_EPS:
        return None
    cell = [0, 0, 0]
    for i in range(3):
        if i == axis:
            cell[i] = int(layer)
        else:
            cell[i] = math.floor(float(origin[i]) + t * float(direction[i]))
    return (cell[0], cell[1], cell[2])


def plane_cell(
    origin, direction, axis: int, plane: float, layer: int, size
) -> Cell | None:
    """`plane_crossing`, or None when the cell lies outside the grid."""
    cell = plane_crossing(origin, direction, axis, plane, layer)
    if cell is None or not in_bounds(cell, size):
        return None
    return cell


def clamp_to_border(cell: Cell, axis: int, size) -> Cell:
    """Clamp the two free coordinates to [-1, size], one cell past the grid.

    A ray that grazes the plane can cross it far away. The clamp keeps the line walk
    short; the first out-of-bounds cell ends the walk anyway.
    """
    out = list(cell)
    for i in range(3):
        if i != axis:
            out[i] = max(-1, min(int(size[i]), out[i]))
    return (out[0], out[1], out[2])


@dataclass(frozen=True)
class PlaneLock:
    """The face plane a drag is locked to, and the voxel layer it writes into."""

    axis: int
    plane: float
    layer: int

    @classmethod
    def from_hit(cls, hit: Hit, cell: Cell) -> PlaneLock:
        axis = 0
        for i in range(3):
            if hit.normal[i] != 0:
                axis = i
                break
        plane = float(hit.voxel[axis] + (1 if hit.normal[axis] > 0 else 0))
        return cls(axis, plane, int(cell[axis]))

    def cell_at(self, origin, direction) -> Cell | None:
        return plane_crossing(origin, direction, self.axis, self.plane, self.layer)


def line_cells(a: Cell, b: Cell, axis: int) -> list[Cell]:
    """2D Bresenham from a to b in the two coordinates other than axis.

    Excludes a, includes b, in order. Empty when a == b. No gaps for any jump.
    """
    u, v = (i for i in range(3) if i != axis)
    x0, y0 = int(a[u]), int(a[v])
    x1, y1 = int(b[u]), int(b[v])
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    cells: list[Cell] = []
    fixed = int(a[axis])
    while x != x1 or y != y1:
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        cell = [0, 0, 0]
        cell[axis] = fixed
        cell[u] = x
        cell[v] = y
        cells.append((cell[0], cell[1], cell[2]))
    return cells


# ---------------------------------------------------------------- mirror


def mirror_images(cell: Cell, size, axes) -> tuple[Cell, ...]:
    """The cell and its mirror images, source first, without duplicates.

    For each active axis in x, y, z order the image of every cell so far is
    appended. A cell on the mirror plane of an odd axis maps to itself.
    """
    out: list[Cell] = [(int(cell[0]), int(cell[1]), int(cell[2]))]
    for i in range(3):
        if not axes[i]:
            continue
        for c in list(out):
            image = list(c)
            image[i] = int(size[i]) - 1 - c[i]
            image_cell = (image[0], image[1], image[2])
            if image_cell not in out:
                out.append(image_cell)
    return tuple(out)


def _mirror_all(cell: Cell, size, axes) -> list[Cell]:
    """The cell's image for every set of active axes, in a fixed order and with
    repeats, so the images of two cells pair up (the ends of a segment)."""
    out: list[Cell] = [(int(cell[0]), int(cell[1]), int(cell[2]))]
    for i in range(3):
        if axes[i]:
            out += [tuple(int(size[i]) - 1 - c[i] if k == i else c[k] for k in range(3)) for c in out]
    return out


def unique_cells(cells: np.ndarray, size) -> np.ndarray:
    """The distinct rows of an (N, 3) cell array inside `size`, sorted as
    np.unique(axis=0) sorts them, through a flag per cell of the grid: no
    sort (the click that built a 500 000-cell box spent
    200 ms sorting its cells). Cells outside the grid are dropped."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if cells.shape[0] == 0:
        return cells
    cells = cells[in_bounds_mask(cells, size)]
    if cells.shape[0] == 0:
        return cells
    flags = np.zeros(tuple(int(s) for s in size), dtype=bool)
    flags[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return np.argwhere(flags)


def mirror_cells(cells: np.ndarray, size, axes) -> np.ndarray:
    """Vectorised `mirror_images` for an (N, 3) array. Sorted, unique (M, 3).
    Cells outside the grid stay only while none is mirrored."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    for i in range(3):
        if not axes[i]:
            continue
        image = cells.copy()
        image[:, i] = int(size[i]) - 1 - cells[:, i]
        cells = np.concatenate([cells, image])
    if cells.shape[0] and not in_bounds_mask(cells, size).all():
        return np.unique(cells, axis=0)  # the rare caller with cells past the grid keeps them
    return unique_cells(cells, size)


# ---------------------------------------------------------------- rules


BRUSH_MAX = 50  # brush sizes 1..50: a sphere 2 * size - 1 voxels across (was 8)

_SPHERES: dict[int, np.ndarray] = {}
_SHELLS: dict[int, np.ndarray] = {}


def sphere_offsets(size: int) -> np.ndarray:
    """(N, 3) int64 offsets of a brush sphere. Size 1 is the one centre cell;
    size s has radius r = s - 1 and holds every cell with dx² + dy² + dz² <= r² + r,
    which is round from size 3 on (size 2 is the centre and its 18 neighbours
    that share a face or an edge)."""
    s = max(1, min(BRUSH_MAX, int(size)))
    found = _SPHERES.get(s)
    if found is None:
        r = s - 1
        rng = np.arange(-r, r + 1)
        d = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3)
        found = d[(d * d).sum(axis=1) <= r * r + r].astype(np.int64)
        _SPHERES[s] = found
    return found


def sphere_shell(size: int) -> np.ndarray:
    """The sphere's outer cells only (a face neighbour lies outside): what the
    hover preview draws."""
    s = max(1, min(BRUSH_MAX, int(size)))
    found = _SHELLS.get(s)
    if found is None:
        mask = sphere_mask(s)
        padded = np.pad(mask, 1)
        inner = mask.copy()  # cells whose six face neighbours are all inside
        for axis in range(3):
            for step in (-1, 1):
                inner &= np.roll(padded, step, axis=axis)[1:-1, 1:-1, 1:-1]
        found = np.argwhere(mask & ~inner).astype(np.int64) - (s - 1)
        _SHELLS[s] = found
    return found


_SPHERE_MASKS: dict[int, np.ndarray] = {}


def sphere_mask(size: int) -> np.ndarray:
    """The brush sphere as a bool box (2 r + 1 on a side), the centre in the middle."""
    s = max(1, min(BRUSH_MAX, int(size)))
    found = _SPHERE_MASKS.get(s)
    if found is None:
        r = s - 1
        off = sphere_offsets(s) + r
        found = np.zeros((2 * r + 1,) * 3, dtype=bool)
        found[off[:, 0], off[:, 1], off[:, 2]] = True
        _SPHERE_MASKS[s] = found
    return found


CAPSULE_MIN = 1.5  # a stroke sweeps a capsule between hits farther apart than this many cells
_CAPSULE_NATIVE: list = []  # [rok_native or None], once looked up


def _capsule_native():
    """rok_native's capsule_mask, or None (no native module, or ROK_PURE_CAPSULE set)."""
    if not _CAPSULE_NATIVE:
        import os

        mod = None
        if not os.environ.get("ROK_PURE_CAPSULE"):
            try:
                import rok_native

                mod = rok_native if hasattr(rok_native, "capsule_mask") else None
            except ImportError:
                mod = None
        _CAPSULE_NATIVE.append(mod)
    return _CAPSULE_NATIVE[0]


def capsule_mask(a, b, size: int, lo, dims) -> np.ndarray:
    """The brush swept from cell a to cell b, as a bool box at `lo` with `dims`:
    every cell whose squared distance to the segment is at most r * r + r, the
    rule of `sphere_offsets`, so both ends are the brush sphere itself."""
    r = max(1, min(BRUSH_MAX, int(size))) - 1
    limit = float(r * r + r)
    a3 = tuple(float(v) for v in a)
    b3 = tuple(float(v) for v in b)
    lo3 = tuple(int(v) for v in lo)
    dims3 = tuple(int(v) for v in dims)
    native = _capsule_native()
    if native is not None:
        return np.asarray(native.capsule_mask(a3, b3, limit, lo3, dims3)).reshape(dims3)
    grids = np.meshgrid(*[np.arange(lo3[k], lo3[k] + dims3[k], dtype=np.float64) for k in range(3)], indexing="ij")
    p = np.stack(grids, axis=-1) - np.asarray(a3)
    ab = np.asarray(b3) - np.asarray(a3)
    len2 = float(ab @ ab)
    t = np.clip((p @ ab) / len2, 0.0, 1.0) if len2 > 0 else np.zeros(dims3)
    d = p - t[..., None] * ab
    return (d * d).sum(axis=-1) <= limit


class CellMask:
    """A set of cells as a bool grid: `cell in mask`, `add`, and whole boxes
    marked by the brush's write. A big brush would fill a Python set with a
    million tuples."""

    def __init__(self, size) -> None:
        self.grid = np.zeros(tuple(int(v) for v in size), dtype=bool)

    def __contains__(self, cell) -> bool:
        x, y, z = int(cell[0]), int(cell[1]), int(cell[2])
        s = self.grid.shape
        return 0 <= x < s[0] and 0 <= y < s[1] and 0 <= z < s[2] and bool(self.grid[x, y, z])

    def add(self, cell) -> None:
        self.grid[int(cell[0]), int(cell[1]), int(cell[2])] = True

    def update(self, cells) -> None:
        for c in cells:
            self.add(c)

    def __len__(self) -> int:
        return int(np.count_nonzero(self.grid))


def brush_sphere(centre, size: int, grid_size) -> np.ndarray:
    """(N, 3) cells of the brush sphere around centre, clipped to the grid."""
    cells = sphere_offsets(size) + np.asarray(centre, dtype=np.int64)
    return cells[in_bounds_mask(cells, grid_size)]


def apply_tool(
    voxels: np.ndarray, tool: str, colour: int, cell: Cell, setter: Setter
) -> bool:
    """Apply one tool to one cell. True when the value changed.

    ATTACH fills an empty cell, ERASE clears a filled cell, PAINT recolours a
    filled cell of another colour. SELECT and out-of-bounds cells write nothing.
    """
    if tool == SELECT or not in_bounds(cell, voxels.shape):
        return False
    x, y, z = (int(cell[0]), int(cell[1]), int(cell[2]))
    current = int(voxels[x, y, z])
    colour = int(colour)
    if tool == ATTACH:
        if current != 0:
            return False
        return bool(setter(x, y, z, colour))
    if tool == ERASE:
        if current == 0:
            return False
        return bool(setter(x, y, z, 0))
    if current == 0 or current == colour:
        return False
    return bool(setter(x, y, z, colour))


def apply_cells(
    voxels: np.ndarray,
    tool: str,
    colour: int,
    cells: np.ndarray,
    set_many: SetMany,
    mirror: Mirror = NO_MIRROR,
) -> int:
    """Vectorised `apply_tool` for a cell array. Returns the number of cells changed.

    `set_many` receives only the cells whose value changes.
    """
    size = voxels.shape
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if any(mirror):
        cells = mirror_cells(cells, size, mirror)
        cells = cells[in_bounds_mask(cells, size)]  # an image of a cell past the grid is dropped
    else:
        cells = unique_cells(cells, size)  # inside the grid, no doubles
    if tool == SELECT or len(cells) == 0:
        return 0
    cur = voxels[cells[:, 0], cells[:, 1], cells[:, 2]]
    colour = int(colour)
    if tool == ATTACH:
        m = cur == 0
        value = colour
    elif tool == ERASE:
        m = cur != 0
        value = 0
    else:
        m = (cur != 0) & (cur != colour)
        value = colour
    if not m.any():
        return 0
    return int(set_many(cells[m], np.full(int(m.sum()), value, np.uint8)))


# ---------------------------------------------------------------- strokes


JOIN_REACH = 1.8  # neighbouring cells, the diagonal of a cube too: one run of a stroke


class Stroke(Gesture):
    """One click-and-drag with one tool in Voxel mode. Each source cell is
    applied at most once; writes happen at once, so `finish` has nothing to do.

    `brush` above 1 applies a sphere around each source cell (see
    `sphere_offsets`) in one `set_many` batch instead of cell by cell.
    """

    def __init__(
        self,
        voxels: np.ndarray,
        tool: str,
        colour: int,
        setter: Setter,
        mirror: Mirror = NO_MIRROR,
        brush: int = 1,
        set_many: SetMany | None = None,
        write_brush=None,
    ) -> None:
        """`write_brush(lo, mask, mode, colour, mark)` writes the brush as one
        box (StrokeRecorder.write_brush); without it the sphere's cells go
        through `set_many` one list at a time."""
        self.voxels = voxels
        self.size = tuple(int(d) for d in voxels.shape)
        self.tool = tool
        self.colour = int(colour)
        self.setter = setter
        self.mirror = (bool(mirror[0]), bool(mirror[1]), bool(mirror[2]))
        self.hit: Hit | None = None
        self.last: Cell | None = None
        self.lock = PlaneLock(2, 0.0, 0)
        self.visited: set[Cell] = set()
        self.brush = max(1, min(BRUSH_MAX, int(brush))) if set_many is not None else 1
        self.set_many = set_many
        self.write_brush = None
        self.erased: set[Cell] | CellMask = set()
        self.attached: set[Cell] | CellMask = set()
        # The brush centres in order, for a designed stroke: a new run starts
        # where the stroke jumped, so a replay never draws across the jump.
        self.runs: list[list[Cell]] = []
        self._noted: set[Cell] = set()
        self._last_noted: Cell | None = None
        self.shape_exp: float | None = None  # a p-norm brush; None is the sphere
        # Erase and Paint fill the way from the last hit to the next one when
        # the two lie at most this many cells apart (set by the caller from the
        # mouse's step on screen); farther is a jump past an edge. 0: no fill.
        self.max_gap = 0.0
        if write_brush is not None:
            self.use_box_writes(write_brush)

    def use_box_writes(self, write_brush) -> None:
        """Write the brush as boxes (StrokeRecorder.write_brush) instead of
        cell lists: for a brush above size 1, before the stroke begins."""
        if self.brush > 1 and write_brush is not None:
            self.write_brush = write_brush
            self.erased = CellMask(self.size)
            self.attached = CellMask(self.size)

    @property
    def axis(self) -> int:
        return self.lock.axis

    @property
    def plane(self) -> float:
        return self.lock.plane

    @property
    def layer(self) -> int:
        return self.lock.layer

    def begin(self, hit: Hit) -> bool:
        """Start the stroke on the first hit. False when there is no target cell."""
        cell = target_cell(self.tool, hit, self.size)
        if cell is None:
            return False
        self.lock = PlaneLock.from_hit(hit, cell)
        self.hit = hit
        self.last = cell
        self.apply_cell(cell)
        return True

    def extend(self, origin, direction) -> bool:
        """Continue the stroke along a new mouse ray. True when a voxel changed."""
        if self.tool == ERASE:
            # Erased cells stay solid for picking, so one stroke peels one layer.
            hit = raycast(
                self.voxels, origin, direction, walls=False, blocked=self.erased
            )
            if hit is None:
                return False
            changed = self._fill_to(hit.voxel)
            self.last = hit.voxel
            return changed
        if self.tool == PAINT:
            hit = raycast(self.voxels, origin, direction, walls=False)
            if hit is None:
                return False
            changed = self._fill_to(hit.voxel)
            self.last = hit.voxel
            return changed
        if self.tool != ATTACH:
            return False
        # Attach follows the surface. The voxels placed this stroke count as empty
        # for the ray, so it keeps hitting the face they sit on. A hit on a voxel
        # face in another plane moves the lock there; in the same plane the drag
        # walks a line, so a fast move leaves no gaps.
        hit = raycast(self.voxels, origin, direction, walls=True, ignore=self.attached)
        if hit is not None and not hit.wall and in_bounds(hit.target, self.size):
            lock = PlaneLock.from_hit(hit, hit.target)
            if lock != self.lock:
                return self._switch(lock, hit.target)
            return self._walk(hit.target)
        # No voxel under the mouse: stay in the plane. The crossing may lie outside
        # the grid: a drag that runs off the edge must still fill the cells up to
        # the border. When it does, and the ray hit the floor or a wall, the drag
        # continues there (down a side face onto the floor, for example).
        cell = self.lock.cell_at(origin, direction)
        changed = False
        if cell is not None:
            changed = self._walk(clamp_to_border(cell, self.lock.axis, self.size))
        if hit is not None and hit.wall and (cell is None or not in_bounds(cell, self.size)):
            lock = PlaneLock.from_hit(hit, hit.target)
            if lock != self.lock and in_bounds(hit.target, self.size):
                changed = self._switch(lock, hit.target) or changed
        return changed

    def _fill_to(self, cell: Cell) -> bool:
        """Apply the brush on the way from the last hit to `cell`, then at `cell`,
        so a fast cut never breaks. The stamps stand
        close enough that the groove's floor shows no ridge: a step of s over a
        sphere of radius r leaves a ridge s * s / (8 r) high, so s = 2 sqrt(r)
        keeps it under half a cell; a brush of 1 steps cell by cell."""
        changed = False
        if self.last is not None and self.max_gap > 0:
            a = np.asarray(self.last, dtype=np.float64)
            b = np.asarray(cell, dtype=np.float64)
            dist = float(np.linalg.norm(b - a))
            sphere = self.shape_exp is None or abs(self.shape_exp - 2.0) < 1e-6
            if CAPSULE_MIN < dist <= self.max_gap and self.brush > 1 and self.write_brush is not None and sphere:
                # One capsule: a row of spheres still left a bead per stamp on
                # the groove's floor (a brush-50 cut). Hits a
                # cell or so apart take the sphere: spheres that close leave no
                # bead, and 64 tiny capsules a move cost 7 ms with a brush of 5.
                self.visited.add(cell)
                self._note(cell)
                return self._apply_capsule(self.last, cell)
            radius = self.brush - 1
            spacing = 1.0 if radius < 1 else max(1.0, 2.0 * math.sqrt(radius))
            if spacing < dist <= self.max_gap:
                n = int(math.ceil(dist / spacing))
                for k in range(1, n):
                    p = a + (b - a) * (k / n)
                    c = (int(round(p[0])), int(round(p[1])), int(round(p[2])))
                    if in_bounds(c, self.size) and self.apply_cell(c):
                        changed = True
        return self.apply_cell(cell) or changed

    def _walk(self, cell: Cell) -> bool:
        """Apply the line from the last cell to `cell` in the locked plane."""
        changed = False
        start = self.last if self.last is not None else cell
        for c in line_cells(start, cell, self.lock.axis):
            if not in_bounds(c, self.size):
                break
            if self.apply_cell(c):
                changed = True
            self.last = c
        return changed

    def _switch(self, lock: PlaneLock, target: Cell) -> bool:
        """Move the drag to another face. When the last cell already lies in the new
        layer (the two faces meet there), the line continues; else the drag jumps."""
        self.lock = lock
        if self.last is not None and self.last[lock.axis] == lock.layer:
            return self._walk(target)
        self._last_noted = None  # a jump: the next centre starts a new run
        changed = self.apply_cell(target)
        self.last = target
        return changed

    def _note(self, cell: Cell) -> None:
        """Remember a brush centre, once, in the order the stroke reached it. A
        centre farther from the last one than the stroke fills across (a jump
        past an edge) starts a new run."""
        noted = (int(cell[0]), int(cell[1]), int(cell[2]))
        last = self._last_noted
        reach = max(float(self.max_gap), JOIN_REACH)
        jumped = last is None or not self.runs or math.dist(last, noted) > reach
        if jumped:
            self.runs.append([])
        self._last_noted = noted
        # A new run starts at its first centre, also when an earlier run passed over that cell.
        if noted in self._noted and not jumped:
            return
        self._noted.add(noted)
        self.runs[-1].append(noted)

    def _brush_mask(self) -> np.ndarray:
        """The brush box: a p-norm when the Stroke Designer set one, else the sphere."""
        if self.shape_exp is None or abs(self.shape_exp - 2.0) < 1e-6:
            return sphere_mask(self.brush)
        from .stroke_design import shape_mask

        return shape_mask(self.brush, self.shape_exp)

    def apply_cell(self, cell: Cell) -> bool:
        """Apply the tool to a source cell and its mirror images, once per stroke.

        True when any image changed.
        """
        self._note(cell)
        if cell in self.visited:
            return False
        self.visited.add(cell)
        if self.brush > 1:
            return self._apply_sphere(cell)
        changed = False
        for image in mirror_images(cell, self.size, self.mirror):
            if apply_tool(self.voxels, self.tool, self.colour, image, self.setter):
                changed = True
                if self.tool == ERASE:
                    self.erased.add(image)
                elif self.tool == ATTACH:
                    self.attached.add(image)
        return changed

    def _apply_sphere(self, cell: Cell) -> bool:
        """The brush sphere around cell and its mirror images, in one batch."""
        if self.write_brush is not None:
            return self._apply_sphere_box(cell)
        if self.shape_exp is not None and abs(self.shape_exp - 2.0) > 1e-6:
            from .stroke_design import shape_offsets

            cells = shape_offsets(self.brush, self.shape_exp) + np.asarray(cell, dtype=np.int64)
            cells = cells[in_bounds_mask(cells, self.size)]
        else:
            cells = brush_sphere(cell, self.brush, self.size)
        if any(self.mirror):
            cells = mirror_cells(cells, self.size, self.mirror)
            cells = cells[in_bounds_mask(cells, self.size)]
        if len(cells) == 0 or self.tool == SELECT:
            return False
        cur = self.voxels[cells[:, 0], cells[:, 1], cells[:, 2]]
        if self.tool == ATTACH:
            m = cur == 0
        elif self.tool == ERASE:
            m = cur != 0
        else:  # PAINT
            m = (cur != 0) & (cur != self.colour)
        hit = cells[m]
        if len(hit) == 0:
            return False
        value = 0 if self.tool == ERASE else self.colour
        n = int(self.set_many(hit, np.full(len(hit), value, dtype=np.uint8)))
        if self.tool == ERASE:
            self.erased.update(map(tuple, hit.tolist()))
        elif self.tool == ATTACH:
            self.attached.update(map(tuple, hit.tolist()))
        return n > 0

    def _apply_capsule(self, a: Cell, b: Cell) -> bool:
        """The brush swept from a to b, one box write per mirror image."""
        from .grid import BRUSH_ATTACH, BRUSH_ERASE, BRUSH_PAINT

        mode = {ATTACH: BRUSH_ATTACH, ERASE: BRUSH_ERASE, PAINT: BRUSH_PAINT}.get(self.tool)
        if mode is None:
            return False
        mark = self.erased.grid if self.tool == ERASE else self.attached.grid if self.tool == ATTACH else None
        r = self.brush - 1
        changed = 0
        for ia, ib in zip(_mirror_all(a, self.size, self.mirror), _mirror_all(b, self.size, self.mirror)):
            clip_lo = [max(min(ia[k], ib[k]) - r, 0) for k in range(3)]
            clip_hi = [min(max(ia[k], ib[k]) + r, self.size[k] - 1) for k in range(3)]
            if any(clip_lo[k] > clip_hi[k] for k in range(3)):
                continue
            dims = [clip_hi[k] - clip_lo[k] + 1 for k in range(3)]
            mask = capsule_mask(ia, ib, self.brush, clip_lo, dims)
            changed += int(self.write_brush(tuple(clip_lo), mask, mode, self.colour, mark))
        return changed > 0

    def _apply_sphere_box(self, cell: Cell) -> bool:
        """The sphere as one box write per mirror image: no cell list."""
        from .grid import BRUSH_ATTACH, BRUSH_ERASE, BRUSH_PAINT

        mode = {ATTACH: BRUSH_ATTACH, ERASE: BRUSH_ERASE, PAINT: BRUSH_PAINT}.get(self.tool)
        if mode is None:
            return False
        mark = self.erased.grid if self.tool == ERASE else self.attached.grid if self.tool == ATTACH else None
        full = self._brush_mask()
        r = self.brush - 1
        changed = 0
        for image in mirror_images(cell, self.size, self.mirror):
            lo = [int(image[k]) - r for k in range(3)]
            hi = [int(image[k]) + r for k in range(3)]
            clip_lo = [max(lo[k], 0) for k in range(3)]
            clip_hi = [min(hi[k], self.size[k] - 1) for k in range(3)]
            if any(clip_lo[k] > clip_hi[k] for k in range(3)):
                continue
            mask = full[tuple(slice(clip_lo[k] - lo[k], clip_hi[k] - lo[k] + 1) for k in range(3))]
            changed += int(self.write_brush(tuple(clip_lo), mask, mode, self.colour, mark))
        return changed > 0

    def finish(self) -> int:
        """Nothing deferred: every cell was written during the drag."""
        return 0

    def preview(self) -> None:
        return None


class ShaderStroke(Stroke):
    """The shader brush: a Paint stroke that writes a
    shader's result. `target` is what the shader makes of the whole model;
    the brush sphere around each cell the stroke passes takes the target's
    value in every cell where it differs, filled or empty, so a shader that
    adds voxels adds them where the brush goes. Shaders work from the cell's
    place, so strokes that overlap join without a seam."""

    def __init__(self, voxels: np.ndarray, target: np.ndarray, setter: Setter, mirror: Mirror = NO_MIRROR,
                 brush: int = 1, set_many: SetMany | None = None) -> None:
        super().__init__(voxels, PAINT, 0, setter, mirror, brush=brush, set_many=set_many)
        self.target = np.asarray(target, dtype=np.uint8)
        self.brush = max(1, min(BRUSH_MAX, int(brush)))

    def apply_cell(self, cell: Cell) -> bool:
        if cell in self.visited:
            return False
        self.visited.add(cell)
        cells = brush_sphere(cell, self.brush, self.size)
        if any(self.mirror):
            cells = mirror_cells(cells, self.size, self.mirror)
            cells = cells[in_bounds_mask(cells, self.size)]
        if len(cells) == 0:
            return False
        want = self.target[cells[:, 0], cells[:, 1], cells[:, 2]]
        m = self.voxels[cells[:, 0], cells[:, 1], cells[:, 2]] != want
        if not m.any():
            return False
        return int(self.set_many(cells[m], want[m])) > 0
