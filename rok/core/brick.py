"""Bricks: boxes of cells one cell high.

A brick model is a grid with a brick table (`VoxelGrid.bricks`). Every brick
is one cell high, so a brick is a plain box of voxels in one colour. The
table says which cells belong together; the look draws a seam between
bricks.

A grid with a table is locked: its cells take no write unless the writer
opens `unlocked`. Brick mode, the Convert commands, undo and redo and the
linked-copy sync do; every other path (a command, a live shader, the MCP)
gets `BrickLocked`, so nothing can change a brick model by mistake and the
table never goes stale.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .grid import VoxelGrid

LOCKED_MESSAGE = ("This model is built from bricks. Build on it in Brick mode (Edit menu), "
                  "or convert it to voxels.")

# The seams byte (BrickTable.field): one bit per side, "the neighbour on
# that side is the same brick".
SAME_BITS = ((0, 1, 0b000001), (0, -1, 0b000010), (1, 1, 0b000100),
             (1, -1, 0b001000), (2, 1, 0b010000), (2, -1, 0b100000))


class BrickLocked(ValueError):
    """A write to a brick model's cells outside `unlocked`."""

    def __init__(self, message: str = LOCKED_MESSAGE) -> None:
        super().__init__(message)


_OPEN: dict[int, int] = {}  # id(grid) -> how many `unlocked` blocks hold it open
_ALL = [0]  # how many `unlocked()` blocks (every grid) are open


@contextlib.contextmanager
def unlocked(grid: VoxelGrid | None = None):
    """Let writes reach a brick grid inside the block: `grid`'s, or every
    grid's with no argument (undo, redo and the linked-copy sync, which
    replay or mirror a change that was allowed when it was made)."""
    if grid is None:
        _ALL[0] += 1
    else:
        _OPEN[id(grid)] = _OPEN.get(id(grid), 0) + 1
    try:
        yield
    finally:
        if grid is None:
            _ALL[0] -= 1
        else:
            left = _OPEN[id(grid)] - 1
            if left:
                _OPEN[id(grid)] = left
            else:
                del _OPEN[id(grid)]


def is_open(grid: VoxelGrid) -> bool:
    """True when a write may reach `grid`: it has no table, or it is unlocked."""
    return getattr(grid, "bricks", None) is None or _ALL[0] > 0 or id(grid) in _OPEN


def check(grid: VoxelGrid) -> None:
    """Raise BrickLocked when `grid` is a brick grid and not unlocked."""
    if not is_open(grid):
        raise BrickLocked()


# ---------------------------------------------------------------- the table


@dataclass(frozen=True)
class Brick:
    """One brick: the cells x0 .. x0 + w - 1, y0 .. y0 + d - 1 at height z."""

    x: int
    y: int
    z: int
    w: int
    d: int
    colour: int

    def cells(self) -> np.ndarray:
        """(w * d, 3) int64 cells."""
        xs, ys = np.meshgrid(np.arange(self.x, self.x + self.w), np.arange(self.y, self.y + self.d), indexing="ij")
        out = np.empty((self.w * self.d, 3), dtype=np.int64)
        out[:, 0] = xs.ravel()
        out[:, 1] = ys.ravel()
        out[:, 2] = self.z
        return out

    def holds(self, cell) -> bool:
        x, y, z = (int(v) for v in cell)
        return z == self.z and self.x <= x < self.x + self.w and self.y <= y < self.y + self.d

    def name(self) -> str:
        """"2×4": the footprint, short side first."""
        a, b = sorted((self.w, self.d))
        return f"{a}×{b}"


class BrickTable:
    """The bricks of one grid. No two bricks share a cell; a filled cell in
    no brick is a loose cell (shown and pulled as a 1×1)."""

    def __init__(self, bricks=()) -> None:
        self.bricks: list[Brick] = list(bricks)

    def __len__(self) -> int:
        return len(self.bricks)

    def __eq__(self, other) -> bool:
        return isinstance(other, BrickTable) and sorted(self.bricks, key=_key) == sorted(other.bricks, key=_key)

    def __repr__(self) -> str:
        return f"BrickTable({len(self.bricks)} bricks)"

    def at(self, cell) -> int | None:
        """The index of the brick holding `cell`, or None."""
        for i, b in enumerate(self.bricks):
            if b.holds(cell):
                return i
        return None

    def cells(self, i: int) -> np.ndarray:
        return self.bricks[i].cells()

    def cell_map(self, size) -> np.ndarray:
        """int32 grid of `size`: each cell's brick index, -1 where no brick."""
        out = np.full(tuple(int(v) for v in size), -1, dtype=np.int32)
        for i, b in enumerate(self.bricks):
            out[b.x:b.x + b.w, b.y:b.y + b.d, b.z] = i
        return out

    def field(self, voxels: np.ndarray) -> np.ndarray:
        """The seams byte of every cell (uint8, the grid's shape): the
        same-brick bit of each side. A loose cell has no same-brick bits."""
        voxels = np.asarray(voxels)
        ids = self.cell_map(voxels.shape)
        out = np.zeros(voxels.shape, dtype=np.uint8)
        inside = ids >= 0
        for axis, step, bit in SAME_BITS:
            other = np.full(voxels.shape, -2, dtype=np.int32)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if step > 0:
                dst[axis], src[axis] = slice(None, -1), slice(1, None)
            else:
                dst[axis], src[axis] = slice(1, None), slice(None, -1)
            other[tuple(dst)] = ids[tuple(src)]
            out[inside & (other == ids)] |= bit
        return out

    def add(self, brick: Brick) -> int:
        self.bricks.append(brick)
        return len(self.bricks) - 1

    def remove(self, i: int) -> Brick:
        return self.bricks.pop(i)

    def copy(self) -> BrickTable:
        return BrickTable(self.bricks)

    def paint(self, voxels: np.ndarray) -> np.ndarray:
        """`voxels` with every brick's cells in its colour (a uint8 copy)."""
        out = np.array(voxels, dtype=np.uint8, copy=True)
        for b in self.bricks:
            out[b.x:b.x + b.w, b.y:b.y + b.d, b.z] = b.colour
        return out

    def to_json(self) -> list:
        return [[b.x, b.y, b.z, b.w, b.d, b.colour] for b in self.bricks]

    @classmethod
    def from_json(cls, data) -> BrickTable:
        bricks = []
        for item in data:
            if len(item) != 6:
                raise ValueError(f"not a brick: {item!r}")
            x, y, z, w, d, colour = (int(v) for v in item)
            if w < 1 or d < 1 or not 0 <= colour <= 255:
                raise ValueError(f"not a brick: {item!r}")
            bricks.append(Brick(x, y, z, w, d, colour))
        return cls(bricks)

    def check(self, voxels: np.ndarray) -> str | None:
        """None when the table fits `voxels`: every brick inside the grid, no
        two bricks on one cell, each brick's cells filled in its colour.
        Else the first fault."""
        size = np.asarray(voxels).shape
        seen = np.zeros(size, dtype=bool)
        for b in self.bricks:
            if b.x < 0 or b.y < 0 or b.z < 0 or b.x + b.w > size[0] or b.y + b.d > size[1] or b.z >= size[2]:
                return f"{b} is outside the grid"
            box = (slice(b.x, b.x + b.w), slice(b.y, b.y + b.d), b.z)
            if seen[box].any():
                return f"{b} overlaps another brick"
            seen[box] = True
            if (voxels[box] != b.colour).any():
                return f"{b} does not match the voxels"
        return None

    # --- the table follows the voxels -------------------------------------

    def transformed(self, R, size) -> BrickTable:
        """The table after the grid of `size` was turned or flipped by the
        signed permutation R, as grid_ops.apply_rotation turns the voxels: a
        cell c goes to R c + s. R must keep the up axis up (a brick stays one
        cell high); else ValueError."""
        R = np.asarray(R, dtype=np.int64).reshape(3, 3)
        if R[2, 2] == 0:
            raise ValueError("a brick model turns only about the upright axis")
        perm = [int(np.nonzero(R[i])[0][0]) for i in range(3)]
        new_size = [int(size[perm[i]]) for i in range(3)]
        s = np.array([new_size[i] - 1 if R[i, perm[i]] < 0 else 0 for i in range(3)], dtype=np.int64)
        out = []
        for b in self.bricks:
            lo = R @ np.array([b.x, b.y, b.z]) + s
            hi = R @ np.array([b.x + b.w - 1, b.y + b.d - 1, b.z]) + s
            a, c = np.minimum(lo, hi), np.maximum(lo, hi)
            out.append(replace(b, x=int(a[0]), y=int(a[1]), z=int(a[2]),
                               w=int(c[0] - a[0] + 1), d=int(c[1] - a[1] + 1)))
        return BrickTable(out)

    def shifted(self, offset) -> BrickTable:
        """The table after the grid's low corner moved by `offset` cells (a
        crop that keeps every brick: Fit)."""
        ox, oy, oz = (int(v) for v in offset)
        return BrickTable(replace(b, x=b.x - ox, y=b.y - oy, z=b.z - oz) for b in self.bricks)


def _key(b: Brick):
    return (b.z, b.y, b.x, b.w, b.d, b.colour)


def upright(orient) -> bool:
    """True when a linked copy's orient keeps the up axis up, so a brick
    table can follow it."""
    return int(np.asarray(orient, dtype=np.int64).reshape(3, 3)[2, 2]) != 0


def member_table(table: BrickTable, size, orient, other_orient) -> BrickTable:
    """A linked copy's table from another copy's: back to the shared shape,
    then out by the copy's own orient (as link.sync does with the voxels)."""
    R = np.asarray(orient, dtype=np.int64).reshape(3, 3)
    source = table.transformed(R.T, size)
    source_size = tuple(int(v) for v in np.abs(R.T) @ np.asarray(size, dtype=np.int64))
    return source.transformed(np.asarray(other_orient, dtype=np.int64).reshape(3, 3), source_size)


# ---------------------------------------------------------------- placing


def turned(w: int, d: int, turn: int) -> tuple[int, int, int, int]:
    """(ox, oy, w', d') of a w × d footprint turned `turn` quarter turns
    clockwise (seen from above) about the centre of its first cell: the
    footprint covers x0 + ox .. + w' - 1 and y0 + oy .. + d' - 1 when the
    first cell is at (x0, y0)."""
    turn = int(turn) % 4
    if turn == 0:
        return 0, 0, w, d
    if turn == 1:
        return 0, -(w - 1), d, w
    if turn == 2:
        return -(w - 1), -(d - 1), w, d
    return -(d - 1), 0, d, w


def land(voxels: np.ndarray, x0: int, y0: int, z: int, w: int, d: int) -> int:
    """The height a w × d footprint at (x0, y0) comes to rest at from height
    `z`: it slides up one cell at a time while any of its cells is filled.
    Cells outside the grid in x or y count as free (fits refuses them); the
    result may be past the top."""
    sx, sy, sz = voxels.shape
    xa, xb = max(x0, 0), min(x0 + w, sx)
    ya, yb = max(y0, 0), min(y0 + d, sy)
    z = max(int(z), 0)
    if xa >= xb or ya >= yb:
        return z
    column = (voxels[xa:xb, ya:yb, :] != 0).any(axis=(0, 1))
    while z < sz and column[z]:
        z += 1
    return z


def fits(grid: VoxelGrid, x0: int, y0: int, z: int, w: int, d: int) -> str | None:
    """None when a brick fits at (x0, y0, z), else the reason: outside the
    grid, in a hidden slice, on a filled cell, or with no filled cell under
    its footprint (on the floor, z = 0, it always has support)."""
    sx, sy, sz = grid.size
    if x0 < 0 or y0 < 0 or x0 + w > sx or y0 + d > sy or z < 0 or z >= sz:
        return "outside the grid"
    box = Brick(x0, y0, z, w, d, 1)
    if grid.sections and grid.is_cut(box.cells()).any():
        return "in a hidden slice"
    if (grid.voxels[x0:x0 + w, y0:y0 + d, z] != 0).any():
        return "on a filled cell"
    if z > 0 and not (grid.voxels[x0:x0 + w, y0:y0 + d, z - 1] != 0).any():
        return "no support"
    return None


def mirror_boxes(x0: int, y0: int, z: int, w: int, d: int, size, axes) -> list[tuple[int, int, int]]:
    """The (x, y, z) origins of a footprint and of its mirror copies across
    the grid's middle planes on the `axes` that are on (the cell c goes to
    size - 1 - c, as the voxel tools mirror). The source first; a copy that
    is the source's own box is left out."""
    out = [(x0, y0, z)]
    for mx in ((False, True) if axes[0] else (False,)):
        for my in ((False, True) if axes[1] else (False,)):
            for mz in ((False, True) if axes[2] else (False,)):
                if not (mx or my or mz):
                    continue
                box = (int(size[0]) - x0 - w if mx else x0,
                       int(size[1]) - y0 - d if my else y0,
                       int(size[2]) - 1 - z if mz else z)
                if box not in out:
                    out.append(box)
    return out


def plan(grid: VoxelGrid, x0: int, y0: int, z: int, w: int, d: int, axes=(False, False, False)):
    """Where a click puts a w × d brick whose footprint starts at (x0, y0)
    from height `z`: ([(x, y, z), ...], None) for the brick and each mirror
    copy, each landed from its own start height, or (boxes, reason) when the
    click is refused (the boxes show where the ghost is)."""
    starts = mirror_boxes(x0, y0, z, w, d, grid.size, axes)
    boxes = []
    reason = None
    for bx, by, bz in starts:
        at = land(grid.voxels, bx, by, bz, w, d)
        boxes.append((bx, by, at))
        why = fits(grid, bx, by, at, w, d)
        if why is not None and reason is None:
            reason = why
    if reason is None:
        taken = np.zeros(grid.size, dtype=bool)
        for bx, by, bz in boxes:
            if taken[bx:bx + w, by:by + d, bz].any():
                reason = "overlaps the mirror copy"
                break
            taken[bx:bx + w, by:by + d, bz] = True
    return boxes, reason


# ---------------------------------------------------------------- converting


def to_bricks(voxels: np.ndarray) -> BrickTable:
    """A 1×1 brick for every filled cell, in its colour."""
    cells = np.argwhere(np.asarray(voxels) != 0)
    return BrickTable(Brick(int(x), int(y), int(z), 1, 1, int(voxels[x, y, z])) for x, y, z in cells)


def convert_to_bricks(grid: VoxelGrid) -> BrickTable:
    """Make `grid` a brick grid of 1×1s and clear its mask. No cell changes.
    Returns the table."""
    table = to_bricks(grid.voxels)
    grid.bricks = table
    grid.lock = None
    grid.mark_all_dirty()
    return table


def convert_to_voxels(grid: VoxelGrid) -> BrickTable | None:
    """Make `grid` a voxel grid again: the table goes, the cells stay.
    Returns the table it had."""
    table = grid.bricks
    grid.bricks = None
    grid.mark_all_dirty()
    return table


def changed_cells(before: BrickTable | None, after: BrickTable | None) -> np.ndarray:
    """(N, 3) cells of the bricks that are in one table and not the other."""
    a = set(before.bricks) if before is not None else set()
    b = set(after.bricks) if after is not None else set()
    diff = a ^ b
    if not diff:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate([brick.cells() for brick in diff])


def put(grid: VoxelGrid, boxes, w: int, d: int, colour: int, write=None) -> list[Brick]:
    """Set a w × d brick at each (x, y, z) of `boxes` (from `plan`): its
    cells take `colour` and the table gets the brick. A voxel grid becomes a
    brick grid with an empty table first. `write(cells, value)` writes the
    cells (a StrokeRecorder's set_many, for undo); grid.set_voxels without.
    Returns the bricks."""
    if grid.bricks is None:
        grid.bricks = BrickTable()
        grid.lock = None
        grid.mark_all_dirty()
    bricks = [Brick(int(x), int(y), int(z), int(w), int(d), int(colour)) for x, y, z in boxes]
    if not bricks:
        return []
    cells = np.concatenate([b.cells() for b in bricks])
    with unlocked(grid):
        if write is None:
            grid.set_voxels(cells, colour, ignore_lock=True)
        else:
            write(cells, colour)
    for b in bricks:
        grid.bricks.add(b)
    return bricks


def click(grid: VoxelGrid, ax: int, ay: int, z: int, w: int, d: int, turn: int = 0, colour: int = 1,
          axes=(False, False, False)) -> tuple[list[Brick], str | None]:
    """A Place click with its first cell at (ax, ay) from height z: the
    bricks it set, or ([], reason) when it was refused."""
    ox, oy, tw, td = turned(w, d, turn)
    boxes, reason = plan(grid, ax + ox, ay + oy, z, tw, td, axes)
    if reason is not None:
        return [], reason
    return put(grid, boxes, tw, td, colour), None


def carry(source: VoxelGrid, grid: VoxelGrid) -> VoxelGrid:
    """Give `grid`, a copy of `source`'s voxels, `source`'s brick table too.
    Returns `grid`."""
    grid.bricks = None if source.bricks is None else source.bricks.copy()
    return grid


# The bin's Random colours: each placed brick takes one of these, never the
# same one twice in a row. Rok's own toy-box set, bright and easy to tell apart.
RANDOM_COLOURS = (
    ("Yellow", (242, 194, 27)),
    ("Red", (214, 58, 47)),
    ("Blue", (42, 93, 176)),
    ("Orange", (240, 138, 36)),
    ("Green", (46, 139, 71)),
    ("Light green", (140, 203, 78)),
    ("Pink", (238, 143, 184)),
    ("Light blue", (124, 195, 232)),
    ("White", (242, 242, 238)),
    ("Dark grey", (74, 77, 82)),
)


def next_random(previous: int | None, rng) -> int:
    """An index into RANDOM_COLOURS other than `previous`; `rng` is a random.Random."""
    choices = [i for i in range(len(RANDOM_COLOURS)) if i != previous]
    return rng.choice(choices)
