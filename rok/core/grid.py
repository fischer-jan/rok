"""Dense uint8 voxel grid with 32^3 chunk dirty tracking."""

from __future__ import annotations

from .palette import default_palette
from .vox_read import VoxModel

import numpy as np

CHUNK_SIZE = 32

# The brush's write modes (VoxelGrid.write_brush).
BRUSH_ATTACH, BRUSH_ERASE, BRUSH_PAINT = 0, 1, 2
MAX_DIM = 256


class VoxelGrid:
    """One dense (x, y, z) uint8 array. 0 is empty. Chunks are remesh units only."""

    def __init__(
        self,
        size: tuple[int, int, int],
        voxels: np.ndarray | None = None,
        palette: np.ndarray | None = None,
    ) -> None:
        self.size = _checked_size(size)
        if voxels is None:
            self.voxels = np.zeros(self.size, dtype=np.uint8)
        else:
            if tuple(voxels.shape) != self.size:
                raise ValueError(
                    f"voxels shape {voxels.shape} does not match size {self.size}"
                )
            self.voxels = np.array(voxels, dtype=np.uint8, copy=True)
        if palette is None:
            self.palette = default_palette()
        else:
            pal = np.array(palette, dtype=np.uint8, copy=True)
            if pal.shape != (256, 4):
                raise ValueError(f"palette shape {pal.shape} is not (256, 4)")
            self.palette = pal
        # While a section is set, `_visible` is a copy of `voxels` with the cut
        # cells zeroed, kept in step by every write.
        # The section planes, one per axis at most: axis -> (coord, from_positive).
        # from_positive cuts the cells at coord and beyond; otherwise the cells
        # below coord. Cut cells are out of view and take no write. Not saved,
        # not in undo. `section_axis` is the plane the keys move.
        self.sections: dict[int, tuple[int, bool]] = {}
        self.section_axis = 2
        self._visible: np.ndarray | None = None
        # The mask (it must protect voxels reliably): a bool
        # array of the grid's shape; a locked cell takes no write from
        # set_voxel or set_voxels, whichever command sends it. Undo writes
        # chunks and a whole-grid change replaces the array, so neither is
        # stopped; a lock of another shape is not used. `locked_writes`
        # counts the writes it stopped, for a note.
        self.lock: np.ndarray | None = None
        self.locked_writes = 0
        # A brick model's table (rok.core.brick.BrickTable), or None for a
        # voxel model. While it is set the cells are locked: the write methods
        # raise brick.BrickLocked unless the writer opened brick.unlocked.
        self.bricks = None
        self._dirty: set[tuple[int, int, int]] = set()
        self.mark_all_dirty()

    def copy(self) -> VoxelGrid:
        """A new grid with this one's voxels, mask and brick table (the
        palette is shared, as the scene's models share it)."""
        out = VoxelGrid(self.size, self.voxels, self.palette)
        out.palette = self.palette
        out.lock = None if self.lock is None else self.lock.copy()
        out.bricks = None if self.bricks is None else self.bricks.copy()
        return out

    def _check_bricks(self) -> None:
        if self.bricks is not None:
            from . import brick

            brick.check(self)

    @classmethod
    def from_model(cls, model: VoxModel) -> VoxelGrid:
        return cls(model.size, model.voxels, model.palette)

    @classmethod
    def empty(cls, size: tuple[int, int, int] = (32, 32, 32)) -> VoxelGrid:
        return cls(size)

    def to_model(self) -> VoxModel:
        return VoxModel(
            size=self.size,
            voxels=self.voxels.copy(),
            palette=self.palette.copy(),
        )

    @property
    def visible(self) -> np.ndarray:
        """The voxels the mesh sees: `voxels` itself while nothing is cut."""
        if self._visible is None:
            return self.voxels
        return self._visible

    @property
    def tool_voxels(self) -> np.ndarray:
        """The voxels the tools see: `visible`. A cut cell is empty there and takes
        no write, so a ray passes the cut and nothing lands in it."""
        return self.visible

    @property
    def section(self) -> int | None:
        """The z plane's height when it cuts from above, for the old callers."""
        cut = self.sections.get(2)
        return cut[0] if cut is not None and cut[1] else None

    def cut_slices(self, bounds=None):
        """For each cut axis, (axis, slice) of the cut cells inside `bounds`
        (chunk slices; the whole grid when None), in local coordinates."""
        out = []
        for axis, (coord, from_positive) in self.sections.items():
            start = 0 if bounds is None else int(bounds[axis].start)
            stop = int(self.size[axis]) if bounds is None else int(bounds[axis].stop)
            if from_positive:
                lo, hi = max(coord - start, 0), stop - start
            else:
                lo, hi = 0, max(min(coord, stop) - start, 0)
            if lo < hi:
                out.append((axis, slice(lo, hi)))
        return out

    def is_cut(self, cells: np.ndarray) -> np.ndarray:
        """(N,) bool: which of the (N, 3) cells lie beyond a section plane."""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        cut = np.zeros(len(cells), dtype=bool)
        for axis, (coord, from_positive) in self.sections.items():
            cut |= cells[:, axis] >= coord if from_positive else cells[:, axis] < coord
        return cut

    def _zero_cut(self, block: np.ndarray, bounds=None) -> None:
        for axis, sl in self.cut_slices(bounds):
            index = [slice(None)] * 3
            index[axis] = sl
            block[tuple(index)] = 0

    def set_section(self, coord: int | None, axis: int = 2, from_positive: bool = True) -> None:
        """Put the section plane of `axis` at `coord` (1 .. size - 1), cutting the
        cells at coord and beyond, or below coord with from_positive False. None
        takes that plane away. The plane becomes the one the keys move."""
        axis = int(axis)
        self.section_axis = axis
        if coord is not None:
            coord = int(coord)
            if coord < 1 or coord >= self.size[axis]:
                coord = None
        new = None if coord is None else (coord, bool(from_positive))
        if self.sections.get(axis) == new:
            return
        if new is None:
            self.sections.pop(axis, None)
        else:
            self.sections[axis] = new
        self._rebuild_visible()

    def _rebuild_visible(self) -> None:
        if not self.sections:
            self._visible = None
        else:
            visible = self.voxels.copy()
            self._zero_cut(visible)
            self._visible = visible
        self.mark_all_dirty()

    @property
    def chunk_counts(self) -> tuple[int, int, int]:
        return tuple((d + CHUNK_SIZE - 1) // CHUNK_SIZE for d in self.size)

    def chunk_bounds(self, chunk: tuple[int, int, int]) -> tuple[slice, slice, slice]:
        cx, cy, cz = chunk
        sx, sy, sz = self.size
        x0, y0, z0 = cx * CHUNK_SIZE, cy * CHUNK_SIZE, cz * CHUNK_SIZE
        return (
            slice(x0, min(x0 + CHUNK_SIZE, sx)),
            slice(y0, min(y0 + CHUNK_SIZE, sy)),
            slice(z0, min(z0 + CHUNK_SIZE, sz)),
        )

    def active_lock(self) -> np.ndarray | None:
        """The lock when it fits the grid and locks something, else None."""
        lock = self.lock
        if lock is None or lock.shape != self.voxels.shape:
            return None
        return lock

    def get_voxel(self, x: int, y: int, z: int) -> int:
        return int(self.voxels[x, y, z])

    def set_voxel(self, x: int, y: int, z: int, index: int) -> bool:
        """Set one voxel. Return True when the value changed."""
        self._check_bricks()
        value = np.uint8(index)
        if self.voxels[x, y, z] == value:
            return False
        if self.sections and bool(self.is_cut(np.array([[x, y, z]]))[0]):
            return False  # beyond a section plane: out of view, so no write
        lock = self.active_lock()
        if lock is not None and lock[x, y, z]:
            self.locked_writes += 1
            return False  # masked
        if self._visible is not None:
            self._visible[x, y, z] = value
        self.voxels[x, y, z] = value
        self._mark_dirty_at(x, y, z)
        return True

    def set_voxels(self, cells: np.ndarray, values, *, ignore_cut: bool = False, ignore_lock: bool = False) -> int:
        """Set many voxels at once. `values` is a scalar or an (N,) array.

        Writes only the cells whose value changes and, by default, never a cell
        beyond a section plane. `ignore_cut=True` writes a cut cell's index too
        (for a palette swap's remap, which must reach every voxel of the swapped
        colours or a cut one keeps the old colour once the plane moves); its
        entry in the visible copy stays zero either way, so it is still cut after
        the write. `ignore_lock=True` writes a locked (masked) cell too: a
        palette remap keeps every voxel looking as it did, and a linked copy
        follows its source. Returns the changed count.
        """
        self._check_bricks()
        cells = np.asarray(cells, dtype=np.intp).reshape(-1, 3)
        if cells.shape[0] == 0:
            return 0
        values = np.asarray(values, dtype=np.uint8)
        if values.ndim == 0:
            values = np.broadcast_to(values, (cells.shape[0],))
        elif values.shape != (cells.shape[0],):
            raise ValueError(
                f"values shape {values.shape} does not match {cells.shape[0]} cells"
            )
        if (cells < 0).any() or (cells >= np.array(self.size)).any():
            raise IndexError("set_voxels: a cell is outside the grid")
        xs, ys, zs = cells[:, 0], cells[:, 1], cells[:, 2]
        changed = self.voxels[xs, ys, zs] != values
        if self.sections and not ignore_cut:
            changed &= ~self.is_cut(cells)  # beyond a section plane: out of view, no write
        lock = None if ignore_lock else self.active_lock()
        if lock is not None:
            locked = changed & lock[xs, ys, zs]
            if locked.any():
                self.locked_writes += int(np.count_nonzero(locked))
                changed &= ~locked
        count = int(np.count_nonzero(changed))
        if count == 0:
            return 0
        hit = cells[changed]
        new = values[changed]
        self.voxels[hit[:, 0], hit[:, 1], hit[:, 2]] = new
        if self._visible is not None:
            shown = np.ones(len(hit), dtype=bool)
            if self.sections:
                shown &= ~self.is_cut(hit)
            self._visible[hit[shown, 0], hit[shown, 1], hit[shown, 2]] = new[shown]
        self._mark_dirty_cells(hit)
        return count

    def _cut_mask(self) -> np.ndarray | None:
        """A bool grid of the cells beyond the section planes, or None without planes."""
        if not self.sections:
            return None
        cut = np.zeros(self.size, dtype=bool)
        for axis, sl in self.cut_slices():
            index = [slice(None)] * 3
            index[axis] = sl
            cut[tuple(index)] = True
        return cut

    def _changed_mask(self, want: np.ndarray, ignore_cut: bool, ignore_lock: bool) -> tuple[np.ndarray, int]:
        """Where a write of the whole grid `want` would change a voxel: the
        cells that differ, less the cut ones and the locked ones; and how
        many the lock stopped (as set_voxels counts them: among the uncut)."""
        want = np.asarray(want, dtype=np.uint8)
        if want.shape != self.voxels.shape:
            raise ValueError(f"set_grid: shape {want.shape} is not the grid's {self.voxels.shape}")
        changed = want != self.voxels
        if not ignore_cut:
            cut = self._cut_mask()
            if cut is not None:
                changed &= ~cut
        lock = None if ignore_lock else self.active_lock()
        stopped = 0
        if lock is not None:
            stopped = int(np.count_nonzero(changed & lock))
            changed &= ~lock
        return changed, stopped

    def _native_args(self, want, ignore_cut: bool, ignore_lock: bool):
        """(want, lock, cut) as the native write takes them, or None without the module."""
        native = _native()
        if native is None:
            return None
        want = np.ascontiguousarray(want, dtype=np.uint8)
        if want.shape != self.voxels.shape:
            raise ValueError(f"set_grid: shape {want.shape} is not the grid's {self.voxels.shape}")
        lock = None if ignore_lock else self.active_lock()
        cut = self._cut_mask()  # the visible copy never shows a cut cell, ignore_cut or not
        lock = None if lock is None else np.ascontiguousarray(lock, dtype=bool)
        return want, lock, cut

    def changed_chunks(self, want: np.ndarray, *, ignore_cut: bool = False,
                       ignore_lock: bool = False) -> list[tuple[int, int, int]]:
        """The chunks a write of `want` (set_grid) would mark dirty."""
        args = self._native_args(want, ignore_cut, ignore_lock)
        if args is not None:
            want, lock, cut = args
            return [tuple(int(v) for v in c)
                    for c in _native().changed_chunks(np.ascontiguousarray(self.voxels), want, lock, cut,
                                                      bool(ignore_cut), CHUNK_SIZE)]
        return self._changed_chunks_numpy(want, ignore_cut=ignore_cut, ignore_lock=ignore_lock)

    def _changed_chunks_numpy(self, want, *, ignore_cut: bool = False, ignore_lock: bool = False):
        return _chunks_touched(self._changed_mask(want, ignore_cut, ignore_lock)[0], self.chunk_counts)

    def set_grid(self, want: np.ndarray, *, ignore_cut: bool = False, ignore_lock: bool = False) -> int:
        """Write the whole grid `want`: every cell that differs changes, as
        set_voxels would change it (never a cut cell, never a locked one, by
        default). Mask arithmetic throughout, no cell list (a
        live turn's write listed every differing cell); one parallel pass
        in the native module when it is there. Returns the count."""
        self._check_bricks()
        args = self._native_args(want, ignore_cut, ignore_lock)
        if args is not None:
            want, lock, cut = args
            if not self.voxels.flags.c_contiguous:
                self.voxels = np.ascontiguousarray(self.voxels)
            if self._visible is not None and not self._visible.flags.c_contiguous:
                self._visible = np.ascontiguousarray(self._visible)
            count, stopped, dirty = _native().apply_grid(self.voxels, self._visible, want, lock, cut,
                                                         bool(ignore_cut), CHUNK_SIZE)
            self.locked_writes += int(stopped)
            self._dirty.update(tuple(int(v) for v in c) for c in dirty)
            return int(count)
        return self._set_grid_numpy(want, ignore_cut=ignore_cut, ignore_lock=ignore_lock)

    def write_brush(self, lo, mask: np.ndarray, mode: int, colour: int, mark: np.ndarray | None = None) -> int:
        """The brush: write one box of the grid, `mask` (bool, the box's shape)
        saying which cells the brush covers. BRUSH_ATTACH fills empty cells
        with `colour`, BRUSH_ERASE clears filled ones, BRUSH_PAINT recolours
        filled cells of another colour. Cut and locked cells stay (the lock
        counts in locked_writes). `mark`, a bool grid, gets True for every
        written cell. No cell list: a brush of size 50 covers a million cells. Returns the changed count."""
        self._check_bricks()
        lo = tuple(int(v) for v in lo)
        mask = np.ascontiguousarray(mask, dtype=bool)
        dims = mask.shape
        if len(dims) != 3 or any(lo[k] < 0 or lo[k] + dims[k] > self.size[k] for k in range(3)):
            raise ValueError(f"write_brush: box {lo} + {dims} is not inside {self.size}")
        if mode not in (BRUSH_ATTACH, BRUSH_ERASE, BRUSH_PAINT):
            raise ValueError(f"write_brush: mode {mode}")
        cut = None
        if self.sections:
            cut = np.zeros(dims, dtype=bool)
            box = tuple(slice(lo[k], lo[k] + dims[k]) for k in range(3))
            for axis, sl in self.cut_slices(box):
                index = [slice(None)] * 3
                index[axis] = sl
                cut[tuple(index)] = True
        native = _native()
        if native is not None and hasattr(native, "brush_write"):
            if not self.voxels.flags.c_contiguous:
                self.voxels = np.ascontiguousarray(self.voxels)
            if self._visible is not None and not self._visible.flags.c_contiguous:
                self._visible = np.ascontiguousarray(self._visible)
            lock = self.active_lock()
            lock = None if lock is None else np.ascontiguousarray(lock, dtype=bool)
            count, stopped, dirty = native.brush_write(self.voxels, self._visible, lock, cut, lo, mask, int(mode),
                                                       int(colour), mark, CHUNK_SIZE)
            self.locked_writes += int(stopped)
            self._dirty.update(tuple(int(v) for v in c) for c in dirty)
            return int(count)
        return self._write_brush_numpy(lo, mask, mode, colour, cut, mark)

    def _write_brush_numpy(self, lo, mask, mode: int, colour: int, cut, mark) -> int:
        """write_brush in numpy: the fallback, and what the native path is tested against."""
        dims = mask.shape
        box = tuple(slice(lo[k], lo[k] + dims[k]) for k in range(3))
        cur = self.voxels[box]
        if mode == BRUSH_ATTACH:
            hit = mask & (cur == 0)
        elif mode == BRUSH_ERASE:
            hit = mask & (cur != 0)
        else:
            hit = mask & (cur != 0) & (cur != colour)
        if cut is not None:
            hit &= ~cut
        lock = self.active_lock()
        if lock is not None:
            locked = hit & lock[box]
            stopped = int(np.count_nonzero(locked))
            if stopped:
                self.locked_writes += stopped
                hit &= ~locked
        count = int(np.count_nonzero(hit))
        if count == 0:
            return 0
        value = 0 if mode == BRUSH_ERASE else int(colour)
        cur[hit] = value
        if self._visible is not None:
            self._visible[box][hit] = value
        if mark is not None:
            mark[box][hit] = True
        changed = np.zeros(self.size, dtype=bool)
        changed[box] = hit
        self._dirty.update(_chunks_touched(changed, self.chunk_counts))
        return count

    def _set_grid_numpy(self, want: np.ndarray, *, ignore_cut: bool = False, ignore_lock: bool = False) -> int:
        """set_grid in numpy: the fallback, and what the native path is tested against."""
        want = np.asarray(want, dtype=np.uint8)
        changed, stopped = self._changed_mask(want, ignore_cut, ignore_lock)
        self.locked_writes += stopped
        count = int(np.count_nonzero(changed))
        if count == 0:
            return 0
        self.voxels[changed] = want[changed]
        if self._visible is not None:
            shown = changed
            cut = self._cut_mask()
            if cut is not None:
                shown = changed & ~cut
            self._visible[shown] = want[shown]
        self._dirty.update(_chunks_touched(changed, self.chunk_counts))
        return count

    def replace_voxels(self, voxels: np.ndarray) -> None:
        """Replace the whole array, possibly with a new size. Marks every chunk dirty."""
        self._check_bricks()
        voxels = np.asarray(voxels)
        if voxels.ndim != 3:
            raise ValueError(f"voxels shape {voxels.shape} is not 3-dimensional")
        self.size = _checked_size(voxels.shape)
        self.voxels = np.array(voxels, dtype=np.uint8, copy=True)
        if self.sections:
            self.sections = {}
            self._visible = None
        self.mark_all_dirty()

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        sx, sy, sz = self.size
        return 0 <= x < sx and 0 <= y < sy and 0 <= z < sz

    def chunk_of(self, x: int, y: int, z: int) -> tuple[int, int, int]:
        return (x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE)

    def read_chunk(self, chunk: tuple[int, int, int]) -> np.ndarray:
        """Return a copy of the chunk contents."""
        return self.voxels[self.chunk_bounds(chunk)].copy()

    def write_chunk(self, chunk: tuple[int, int, int], data: np.ndarray) -> None:
        """Replace the chunk contents (undo, redo, revert). Mark a neighbour dirty only
        when its boundary changed."""
        bounds = self.chunk_bounds(chunk)
        old = self.voxels[bounds]
        if tuple(data.shape) != old.shape:
            raise ValueError(f"chunk data shape {data.shape} does not match {old.shape}")
        new = np.asarray(data, dtype=np.uint8)
        changed_faces = _changed_faces(old, new)
        self.voxels[bounds] = new
        if self._visible is not None:
            # The cells beyond a section plane stay out of view.
            shown = new.copy()
            self._zero_cut(shown, bounds)
            self._visible[bounds] = shown
        self._dirty.add(tuple(chunk))
        counts = self.chunk_counts
        for axis, step in changed_faces:
            neighbor = list(chunk)
            neighbor[axis] += step
            if 0 <= neighbor[axis] < counts[axis]:
                self._dirty.add(tuple(neighbor))

    def dirty_chunks(self) -> frozenset[tuple[int, int, int]]:
        return frozenset(self._dirty)

    def clear_dirty(self) -> None:
        self._dirty.clear()

    def mark_all_dirty(self) -> None:
        nx, ny, nz = self.chunk_counts
        self._dirty = {(cx, cy, cz) for cx in range(nx) for cy in range(ny) for cz in range(nz)}

    def _mark_dirty_at(self, x: int, y: int, z: int) -> None:
        cx, cy, cz = x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE
        self._dirty.add((cx, cy, cz))
        counts = self.chunk_counts
        coords = (x, y, z)
        origin = (cx, cy, cz)
        for axis, coord in enumerate(coords):
            if coord % CHUNK_SIZE == 0:
                neighbor = list(origin)
                neighbor[axis] -= 1
                if neighbor[axis] >= 0:
                    self._dirty.add(tuple(neighbor))
            if (coord + 1) % CHUNK_SIZE == 0:
                neighbor = list(origin)
                neighbor[axis] += 1
                if neighbor[axis] < counts[axis]:
                    self._dirty.add(tuple(neighbor))

    def chunks_of(self, cells: np.ndarray) -> list[tuple[int, int, int]]:
        """Return the distinct chunks of an (N, 3) array of cells."""
        cells = np.asarray(cells, dtype=np.intp).reshape(-1, 3)
        return _unique_chunks(cells // CHUNK_SIZE, self.chunk_counts)

    def mark_dirty_cells(self, cells: np.ndarray) -> None:
        """Mark the chunks of an (N, 3) array of cells dirty, as a write
        would: for a change the voxels do not show (a brick table's)."""
        cells = np.asarray(cells, dtype=np.intp).reshape(-1, 3)
        if cells.shape[0]:
            self._mark_dirty_cells(cells)

    def _mark_dirty_cells(self, cells: np.ndarray) -> None:
        """Vectorised `_mark_dirty_at` for an (N, 3) array of cells: the cells'
        chunks and, for a cell on a chunk's face, the chunk beyond it. One
        flag per chunk, set through linear keys: no (N, 3) copies and no sort."""
        counts = tuple(int(c) for c in self.chunk_counts)
        chunks = cells // CHUNK_SIZE
        keys = np.ravel_multi_index(tuple(chunks.T), counts)
        flags = np.zeros(counts[0] * counts[1] * counts[2], dtype=bool)
        flags[keys] = True
        strides = (counts[1] * counts[2], counts[2], 1)
        for axis in range(3):
            coord = cells[:, axis]
            low = (coord % CHUNK_SIZE == 0) & (chunks[:, axis] > 0)
            flags[keys[low] - strides[axis]] = True
            high = ((coord + 1) % CHUNK_SIZE == 0) & (chunks[:, axis] < counts[axis] - 1)
            flags[keys[high] + strides[axis]] = True
        cx, cy, cz = np.unravel_index(np.flatnonzero(flags), counts)
        self._dirty.update((int(x), int(y), int(z)) for x, y, z in zip(cx, cy, cz))


_NATIVE: list = []  # [module or None], once looked up


def _native():
    """rok_native's grid write, or None (no native module, or ROK_PURE_GRID set)."""
    if not _NATIVE:
        import os

        mod = None
        if not os.environ.get("ROK_PURE_GRID"):
            try:
                import rok_native

                mod = rok_native if hasattr(rok_native, "apply_grid") else None
            except ImportError:
                mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


def _chunks_touched(changed: np.ndarray, counts) -> list[tuple[int, int, int]]:
    """The chunks of a bool grid's True cells and of their face neighbours: the
    chunks `_mark_dirty_cells` marks for those cells, by mask arithmetic."""
    if not changed.any():
        return []
    grown = changed.copy()
    for axis in range(3):
        lo = tuple(slice(1, None) if i == axis else slice(None) for i in range(3))
        hi = tuple(slice(None, -1) if i == axis else slice(None) for i in range(3))
        grown[lo] |= changed[hi]
        grown[hi] |= changed[lo]
    counts = tuple(int(c) for c in counts)
    shape = np.asarray(grown.shape, dtype=np.int64)
    pad = np.asarray(counts, dtype=np.int64) * CHUNK_SIZE - shape
    if pad.any():
        grown = np.pad(grown, [(0, int(p)) for p in pad], constant_values=False)
    blocks = grown.reshape(counts[0], CHUNK_SIZE, counts[1], CHUNK_SIZE, counts[2], CHUNK_SIZE).any(axis=(1, 3, 5))
    return [(int(x), int(y), int(z)) for x, y, z in np.argwhere(blocks)]


def _unique_chunks(chunks: np.ndarray, counts: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Distinct rows of an (N, 3) chunk array: a linear key per row, marked in
    a flag per chunk. No sort: a write of 700 000 cells spent most of its
    time in np.unique here (a live turn of a big selection)."""
    if chunks.shape[0] == 0:
        return []
    flags = np.zeros(int(np.prod(counts)), dtype=bool)
    flags[np.ravel_multi_index(tuple(chunks.T), counts)] = True
    keys = np.flatnonzero(flags)
    cx, cy, cz = np.unravel_index(keys, counts)
    return [(int(x), int(y), int(z)) for x, y, z in zip(cx, cy, cz)]


def _checked_size(size) -> tuple[int, int, int]:
    """Return `size` as an int triple, or raise when a dimension is outside 1..MAX_DIM."""
    sx, sy, sz = (int(d) for d in size)
    for dim, name in ((sx, "x"), (sy, "y"), (sz, "z")):
        if dim < 1 or dim > MAX_DIM:
            raise ValueError(f"grid {name}={dim} is out of range 1..{MAX_DIM}")
    return (sx, sy, sz)


def _changed_faces(old: np.ndarray, new: np.ndarray) -> list[tuple[int, int]]:
    """Return (axis, -1 or +1) for each boundary slice that differs between old and new."""
    faces: list[tuple[int, int]] = []
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(0, 1)
        high[axis] = slice(-1, None)
        if not np.array_equal(old[tuple(low)], new[tuple(low)]):
            faces.append((axis, -1))
        if not np.array_equal(old[tuple(high)], new[tuple(high)]):
            faces.append((axis, 1))
    return faces
