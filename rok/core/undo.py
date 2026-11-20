"""Stroke recording and a bounded undo stack.

Phase 2 records (StrokeRecord, GridRecord) act on one VoxelGrid. Phase 3 records
(ModelRecord, SceneRecord, PaletteRecord, SelectionRecord, LayerRecord, AnimRecord)
act on a VoxScene: the stack passes the target through, so each record gets the
object it was made for.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .grid import VoxelGrid
from .select import Selection

if TYPE_CHECKING:
    from .anim import Timeline
    from .colour_store import Snapshot
    from .material import MaterialTable
    from .scene import SceneModel, VoxScene

Chunk = tuple[int, int, int]
Cell = tuple[int, int, int]


@dataclass
class StrokeRecord:
    """The chunks one stroke changed. Maps chunk -> (before, after)."""

    chunks: dict[Chunk, tuple[np.ndarray, np.ndarray]]

    def undo(self, grid: VoxelGrid) -> None:
        for chunk, (before, _after) in self.chunks.items():
            grid.write_chunk(chunk, before)

    def redo(self, grid: VoxelGrid) -> None:
        for chunk, (_before, after) in self.chunks.items():
            grid.write_chunk(chunk, after)

    @property
    def nbytes(self) -> int:
        return sum(before.nbytes + after.nbytes for before, after in self.chunks.values())


@dataclass
class GridRecord:
    """A whole-grid change (flip, rotate, resize). Holds full copies of both states."""

    before: np.ndarray
    after: np.ndarray

    def undo(self, grid: VoxelGrid) -> None:
        grid.replace_voxels(self.before)

    def redo(self, grid: VoxelGrid) -> None:
        grid.replace_voxels(self.after)

    @property
    def nbytes(self) -> int:
        return self.before.nbytes + self.after.nbytes


@dataclass
class ModelRecord:
    """A change to one model of a scene: its voxels, its position or its name.

    A None position or name means "unchanged". `ModelRecord(uid, stroke)` is enough
    for a plain voxel edit.
    """

    uid: int
    inner: StrokeRecord | GridRecord | None = None
    position_before: Cell | None = None
    position_after: Cell | None = None
    name_before: str | None = None
    name_after: str | None = None
    # Linked copies: a turn or flip in the scene changes the orient; Make
    # unique changes the link (and resets the orient). None: unchanged.
    orient_before: tuple | None = None
    orient_after: tuple | None = None
    link_before: int | None = None
    link_after: int | None = None
    frame: int = 0  # the frame the voxel change is on (0: the model's own grid)

    def undo(self, scene: VoxScene) -> None:
        model = scene.by_uid(self.uid)
        if self.inner is not None:
            self.inner.undo(model.frame_grid(self.frame))
        self._restore(model, self.position_before, self.name_before)
        self._relink(model, self.orient_before, self.link_before)

    def redo(self, scene: VoxScene) -> None:
        model = scene.by_uid(self.uid)
        if self.inner is not None:
            self.inner.redo(model.frame_grid(self.frame))
        self._restore(model, self.position_after, self.name_after)
        self._relink(model, self.orient_after, self.link_after)

    @staticmethod
    def _relink(model: SceneModel, orient: tuple | None, link: int | None) -> None:
        if orient is not None:
            model.orient = orient
        if link is not None:
            model.link = link

    @property
    def moved(self) -> bool:
        return self.position_before is not None and self.position_before != self.position_after

    @property
    def nbytes(self) -> int:
        return self.inner.nbytes if self.inner is not None else 0

    @staticmethod
    def _restore(model: SceneModel, position: Cell | None, name: str | None) -> None:
        if position is not None:
            model.position = tuple(int(p) for p in position)
        if name is not None:
            model.name = name


@dataclass
class BrickRecord:
    """A change to one model's brick table (rok.core.brick): a placed,
    pulled or painted brick goes with the cells' record in one step; Convert
    is this record alone, as it changes no cell. None is a voxel model."""

    uid: int
    before: object = None  # BrickTable | None
    after: object = None

    def _set(self, scene: VoxScene, table) -> None:
        from . import brick

        grid = scene.by_uid(self.uid).grid
        old = grid.bricks
        grid.bricks = None if table is None else table.copy()
        if (old is None) != (table is None):
            grid.mark_all_dirty()  # the look comes or goes
        else:
            grid.mark_dirty_cells(brick.changed_cells(old, table))

    def undo(self, scene: VoxScene) -> None:
        self._set(scene, self.before)

    def redo(self, scene: VoxScene) -> None:
        self._set(scene, self.after)

    @property
    def nbytes(self) -> int:
        return 64 * (len(self.before or ()) + len(self.after or ()))


# Rough size of one material table entry, for the byte budget.
MATERIAL_ENTRY_BYTES = 200


@dataclass
class PaletteRecord:
    """A palette edit. Holds full (256, 4) copies; index says which slot changed.

    The material snapshots are copies of the whole table; None means the edit
    did not touch the table.
    """

    index: int | None  # None = whole palette (import, merge)
    before: np.ndarray
    after: np.ndarray
    session: int  # merge key: edits of one popup session collapse into one record
    materials_before: MaterialTable | None = None
    materials_after: MaterialTable | None = None

    def undo(self, scene: VoxScene) -> None:
        scene.palette[...] = self.before  # in place: every grid aliases it
        if self.materials_before is not None:
            scene.materials.restore(self.materials_before)

    def redo(self, scene: VoxScene) -> None:
        scene.palette[...] = self.after
        if self.materials_after is not None:
            scene.materials.restore(self.materials_after)

    @property
    def nbytes(self) -> int:
        total = self.before.nbytes + self.after.nbytes
        for table in (self.materials_before, self.materials_after):
            if table is not None:
                total += len(table) * MATERIAL_ENTRY_BYTES
        return total


@dataclass
class SceneRecord:
    """A change to the model list: add, duplicate, delete, split, add from file.

    `inner` is a change that came with it (the split's source stroke, the merge's
    palette record). It is undone after the list is restored and redone before.

    `timeline_before` and `timeline_after` are the whole timeline, so a deleted
    model's track goes with it and undo brings the track back. `None` means the
    timeline was not touched; the caller passes deep copies (Task 2.4 on).
    """

    before: tuple[SceneModel, ...]
    after: tuple[SceneModel, ...]
    active_before: int
    active_after: int
    inner: ModelRecord | PaletteRecord | None = None
    timeline_before: Timeline | None = None
    timeline_after: Timeline | None = None

    def undo(self, scene: VoxScene) -> None:
        scene.models = list(self.before)
        if self.timeline_before is not None:
            scene.timeline = copy.deepcopy(self.timeline_before)
        if self.inner is not None:
            self.inner.undo(scene)

    def redo(self, scene: VoxScene) -> None:
        if self.inner is not None:
            self.inner.redo(scene)
        scene.models = list(self.after)
        if self.timeline_after is not None:
            scene.timeline = copy.deepcopy(self.timeline_after)

    @property
    def nbytes(self) -> int:
        """The arrays of models present in exactly one of the two lists, plus inner
        and the timeline snapshots (when the delete took a track with it)."""
        before = {m.uid: m for m in self.before}
        after = {m.uid: m for m in self.after}
        total = 0
        for uid in set(before) ^ set(after):
            model = before.get(uid) or after[uid]
            total += model.grid.voxels.nbytes
        if self.inner is not None:
            total += self.inner.nbytes
        for timeline in (self.timeline_before, self.timeline_after):
            if timeline is not None:
                total += _timeline_nbytes(timeline)
        return total


class SelectionRecord:
    """A change to one model's selection mask.

    Holds the mask before and after as packed bits, one bit per cell: a 256^3
    grid costs 2 MB per mask here, not 16 MB, so a Select click does not push
    voxel strokes out of the undo budget. A None-op (equal masks) is not
    pushed: callers use `selection_record`, which returns None then. `before`
    and `after` unpack to whole bool masks.
    """

    def __init__(self, uid: int, before: np.ndarray, after: np.ndarray) -> None:
        self.uid = int(uid)
        self._before_shape, self._before = self._pack(before)
        self._after_shape, self._after = self._pack(after)

    @staticmethod
    def _pack(mask: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        return tuple(int(s) for s in mask.shape), np.packbits(mask.reshape(-1))

    @staticmethod
    def _unpack(shape: tuple[int, ...], packed: np.ndarray) -> np.ndarray:
        count = int(np.prod(shape)) if shape else 0
        return np.unpackbits(packed, count=count).astype(bool).reshape(shape)

    @property
    def before(self) -> np.ndarray:
        return self._unpack(self._before_shape, self._before)

    @property
    def after(self) -> np.ndarray:
        return self._unpack(self._after_shape, self._after)

    def undo(self, scene: VoxScene) -> None:
        self._restore(scene, self._before_shape, self._before)

    def redo(self, scene: VoxScene) -> None:
        self._restore(scene, self._after_shape, self._after)

    def _restore(self, scene: VoxScene, shape: tuple[int, ...], packed: np.ndarray) -> None:
        model = scene.by_uid(self.uid)
        if model.selection.size != shape:
            model.selection = Selection(shape)
        model.selection.replace(self._unpack(shape, packed))

    @property
    def nbytes(self) -> int:
        return int(self._before.nbytes + self._after.nbytes)


@dataclass
class LayerRecord:
    """A layer shown or hidden, or models moved to another layer: the LAYR
    bytes and each moved model's layer, before and after."""

    layers_before: list[bytes]
    layers_after: list[bytes]
    moves: tuple[tuple[int, int, int], ...] = ()  # (uid, layer before, layer after)

    def undo(self, scene: VoxScene) -> None:
        scene.layers = list(self.layers_before)
        for uid, before, _after in self.moves:
            scene.by_uid(uid).layer = before

    def redo(self, scene: VoxScene) -> None:
        scene.layers = list(self.layers_after)
        for uid, _before, after in self.moves:
            scene.by_uid(uid).layer = after

    @property
    def nbytes(self) -> int:
        return sum(len(c) for c in self.layers_before) + sum(len(c) for c in self.layers_after)


# Rough size of one keyframe, for the byte budget: a frame number, a small
# tuple or scalar value, and an ease string.
ANIM_KEY_BYTES = 64


def _timeline_nbytes(timeline: Timeline) -> int:
    return ANIM_KEY_BYTES * sum(
        len(prop.keys) for track in timeline.tracks for prop in track.properties.values()
    )


@dataclass
class AnimRecord:
    """A change to the whole timeline: keys added or moved, a preset applied.

    Deep-copies both snapshots at construction, so a caller can pass the live
    timeline itself, before and after mutating it in place, without first
    taking its own copy. Undo and redo deep-copy again on apply, so the stored
    snapshots stay good after repeated undo/redo.
    """

    before: Timeline
    after: Timeline

    def __post_init__(self) -> None:
        self.before = copy.deepcopy(self.before)
        self.after = copy.deepcopy(self.after)

    def undo(self, scene: VoxScene) -> None:
        scene.timeline = copy.deepcopy(self.before)

    def redo(self, scene: VoxScene) -> None:
        scene.timeline = copy.deepcopy(self.after)

    @property
    def nbytes(self) -> int:
        return _timeline_nbytes(self.before) + _timeline_nbytes(self.after)


@dataclass
class FramesRecord:
    """A part's frames (SceneModel.states) before and after: a frame added or deleted."""

    uid: int
    before: list
    after: list

    def __post_init__(self) -> None:
        self.before = [np.array(v, copy=True) for v in self.before]
        self.after = [np.array(v, copy=True) for v in self.after]

    def undo(self, scene: VoxScene) -> None:
        model = scene.by_uid(self.uid)
        model.states = [np.array(v, copy=True) for v in self.before]
        model.frame_grids.clear()

    def redo(self, scene: VoxScene) -> None:
        model = scene.by_uid(self.uid)
        model.states = [np.array(v, copy=True) for v in self.after]
        model.frame_grids.clear()

    @property
    def nbytes(self) -> int:
        return sum(v.nbytes for v in self.before) + sum(v.nbytes for v in self.after)


@dataclass
class ClipsRecord:
    """A change to the list of clips (added, renamed, duplicated, deleted,
    another one picked): the whole list and the active index, before and after."""

    before: list
    before_active: int
    after: list
    after_active: int

    def __post_init__(self) -> None:
        self.before = copy.deepcopy(self.before)
        self.after = copy.deepcopy(self.after)

    def undo(self, scene: VoxScene) -> None:
        scene.clips = copy.deepcopy(self.before)
        scene.active_clip = self.before_active

    def redo(self, scene: VoxScene) -> None:
        scene.clips = copy.deepcopy(self.after)
        scene.active_clip = self.after_active

    @property
    def nbytes(self) -> int:
        return sum(_timeline_nbytes(c) for c in self.before) + sum(_timeline_nbytes(c) for c in self.after)


@dataclass
class ColourRecord:
    """The scene's colours (rok.core.colour_store) before and after: new
    colours, an edited colour, a new order, leases. `session` merges the
    edits of one colour popup drag into one step."""

    before: Snapshot
    after: Snapshot
    session: int = 0

    def undo(self, scene: VoxScene) -> None:
        scene.colours.restore(self.before)

    def redo(self, scene: VoxScene) -> None:
        scene.colours.restore(self.after)

    @property
    def nbytes(self) -> int:
        return 200 * (len(self.before.entries) + len(self.after.entries)) + 2048


@dataclass
class LightRecord:
    """The scene's light (rok.core.light) before and after. A LightSetup does
    not change in place, so the record keeps the two objects."""

    before: object  # LightSetup | None
    after: object

    def undo(self, scene: VoxScene) -> None:
        scene.light = self.before

    def redo(self, scene: VoxScene) -> None:
        scene.light = self.after

    @property
    def nbytes(self) -> int:
        return 2048


@dataclass
class CompoundRecord:
    """Several records that undo as one step (a paste that also changed the palette).

    Undo runs the parts in reverse order, redo in order. A voxel write that also
    changed the selection is one CompoundRecord of the ModelRecord then the
    SelectionRecord, so one undo takes both back.
    """

    parts: tuple[ModelRecord | PaletteRecord | SelectionRecord | AnimRecord, ...]

    def undo(self, scene: VoxScene) -> None:
        for part in reversed(self.parts):
            part.undo(scene)

    def redo(self, scene: VoxScene) -> None:
        for part in self.parts:
            part.redo(scene)

    def first(self, kind):
        """The first part of class `kind`, or None."""
        for part in self.parts:
            if isinstance(part, kind):
                return part
        return None

    @property
    def nbytes(self) -> int:
        return sum(part.nbytes for part in self.parts)


Record = (
    StrokeRecord
    | GridRecord
    | ModelRecord
    | SceneRecord
    | PaletteRecord
    | SelectionRecord
    | LayerRecord
    | AnimRecord
    | LightRecord
    | BrickRecord
    | CompoundRecord
)


class MaskRecord(SelectionRecord):
    """A change to one model's mask (grid.lock), packed as a selection is. Like
    a selection, a mask is not saved in the file, so it is no file change."""

    def _restore(self, scene: VoxScene, shape: tuple[int, ...], packed: np.ndarray) -> None:
        mask = self._unpack(shape, packed)
        scene.by_uid(self.uid).grid.lock = mask if mask.any() else None


def mask_record(uid: int, before: np.ndarray | None, after: np.ndarray | None, shape) -> MaskRecord | None:
    """A MaskRecord when the masks differ, else None. None is no mask; `shape`
    is the grid's, for a side that has none."""
    empty = np.zeros(tuple(shape), dtype=bool)
    b = empty if before is None else before
    a = empty if after is None else after
    if b.shape == a.shape and np.array_equal(b, a):
        return None
    return MaskRecord(uid, b, a)


def selection_record(uid: int, before: np.ndarray, after: np.ndarray) -> SelectionRecord | None:
    """A SelectionRecord when the masks differ, else None."""
    if np.array_equal(before, after):
        return None
    return SelectionRecord(uid, before, after)


def changes_saved_data(record: Record | None) -> bool:
    """False for a step that only moved the selection: nothing in the file changed.

    A selection is not saved, so undoing or redoing one must not mark a clean
    document as modified.
    """
    if record is None:
        return False
    if isinstance(record, SelectionRecord):
        return False
    if isinstance(record, CompoundRecord):
        return any(not isinstance(part, SelectionRecord) for part in record.parts)
    return True


def stacked_record(*parts: Record | None) -> Record | None:
    """One record, a CompoundRecord of several, or None when every part is None."""
    present = tuple(p for p in parts if p is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return CompoundRecord(present)


class StrokeRecorder:
    """Records one stroke. Snapshots a chunk the first time the stroke touches it."""

    def __init__(self, grid: VoxelGrid) -> None:
        self._grid = grid
        self._before: dict[Chunk, np.ndarray] = {}

    def set(self, x: int, y: int, z: int, value: int) -> bool:
        """Snapshot the chunk on first touch, then set the voxel."""
        chunk = self._grid.chunk_of(x, y, z)
        if chunk not in self._before:
            self._before[chunk] = self._grid.read_chunk(chunk)
        return self._grid.set_voxel(x, y, z, value)

    def set_many(self, cells: np.ndarray, values, *, ignore_cut: bool = False, ignore_lock: bool = False) -> int:
        """Snapshot every chunk the cells touch on first touch, then set the voxels.

        `ignore_cut` is passed straight to `grid.set_voxels`: a palette swap
        needs it so a voxel beyond a section plane is remapped too, not skipped.
        """
        for chunk in self._grid.chunks_of(cells):
            if chunk not in self._before:
                self._before[chunk] = self._grid.read_chunk(chunk)
        return self._grid.set_voxels(cells, values, ignore_cut=ignore_cut, ignore_lock=ignore_lock)

    def write_brush(self, lo, mask: np.ndarray, mode: int, colour: int, mark=None) -> int:
        """Snapshot every chunk the brush's box touches, then write it
        (VoxelGrid.write_brush): the box write for a big brush."""
        from .grid import CHUNK_SIZE

        lo = tuple(int(v) for v in lo)
        hi = tuple(lo[k] + int(mask.shape[k]) - 1 for k in range(3))
        for cx in range(lo[0] // CHUNK_SIZE, hi[0] // CHUNK_SIZE + 1):
            for cy in range(lo[1] // CHUNK_SIZE, hi[1] // CHUNK_SIZE + 1):
                for cz in range(lo[2] // CHUNK_SIZE, hi[2] // CHUNK_SIZE + 1):
                    chunk = (cx, cy, cz)
                    if chunk not in self._before:
                        self._before[chunk] = self._grid.read_chunk(chunk)
        return self._grid.write_brush(lo, mask, mode, colour, mark)

    def set_grid(self, want: np.ndarray, *, ignore_cut: bool = False, ignore_lock: bool = False) -> int:
        """Snapshot every chunk a write of the whole grid `want` touches, then
        write it (VoxelGrid.set_grid): the mask-based write for a live turn."""
        for chunk in self._grid.changed_chunks(want, ignore_cut=ignore_cut, ignore_lock=ignore_lock):
            if chunk not in self._before:
                self._before[chunk] = self._grid.read_chunk(chunk)
        return self._grid.set_grid(want, ignore_cut=ignore_cut, ignore_lock=ignore_lock)

    def finish(self) -> StrokeRecord | None:
        """Return the record of changed chunks, or None when nothing changed."""
        chunks: dict[Chunk, tuple[np.ndarray, np.ndarray]] = {}
        for chunk, before in self._before.items():
            after = self._grid.read_chunk(chunk)
            if not np.array_equal(before, after):
                chunks[chunk] = (before, after)
        if not chunks:
            return None
        return StrokeRecord(chunks)

    def revert(self) -> None:
        """Write every snapshot back to the grid."""
        for chunk, before in self._before.items():
            self._grid.write_chunk(chunk, before)


# The name of a record with no name, by its kind; a CompoundRecord takes its first part's.
KIND_NAMES = {
    "StrokeRecord": "Voxels", "GridRecord": "Voxels", "ModelRecord": "Voxels", "PaletteRecord": "Colours",
    "SceneRecord": "Scene", "SelectionRecord": "Selection", "MaskRecord": "Mask", "LayerRecord": "Layers",
    "AnimRecord": "Animation", "FramesRecord": "Frames", "ClipsRecord": "Clips",
    "LiveRecord": "Live Shader", "ColourRecord": "Colours", "PivotRecord": "Pivot", "ParentRecord": "Parent",
    "LightRecord": "Light", "BrickRecord": "Bricks",
}


def kind_name(record) -> str:
    if isinstance(record, CompoundRecord) and record.parts:
        return kind_name(record.parts[0])
    return next((KIND_NAMES[c.__name__] for c in type(record).__mro__ if c.__name__ in KIND_NAMES), "Change")


@contextlib.contextmanager
def naming(label: str):
    """Records pushed inside get `label` as their name, unless they have one.
    The innermost wins: an action that runs another names that one's records."""
    old = UndoStack.label
    UndoStack.label = label
    try:
        yield
    finally:
        UndoStack.label = old


def _replay():
    """Undo and redo replay a change that was allowed when it was made, so
    they may write a brick model's cells."""
    from . import brick

    return brick.unlocked()


class UndoStack:
    """A bounded undo stack. The oldest record drops when the count or byte budget is full.

    Each pushed record gets a `label` (its name in the history: the action,
    tool or agent call that made it, see `naming`, else its kind) and `when`
    (time.time() of the push), unless it has them already.

    `after_apply(record, undone)`, when set, runs after every push, undo and
    redo: the app sets it to keep linked copies in line (link.py).
    """

    after_apply = None  # set once for every stack; see the class docstring
    # `after_push(record)`, when set, runs after a push, before after_apply,
    # and may return a record that takes the pushed one's place (one undo
    # step): the app reruns live shaders there (rok/app/live_shaders.py).
    after_push = None
    label = ""  # the name of the next record pushed without one; set by `naming`

    def __init__(self, capacity: int = 64, max_bytes: int = 256 * 1024 * 1024) -> None:
        self._undo: collections.deque[Record] = collections.deque(maxlen=capacity)
        self._redo: list[Record] = []
        self._max_bytes = max_bytes
        self.dropped = 0  # records dropped for the count or the byte budget
        self.pushes = 0  # every push and every fold into the top record; it only grows
        self.last_applied: Record | None = None  # the record undo() or redo() last applied
        self.colours = None  # the ColourStore this stack follows (track)
        self._colours_seen: tuple[int, Snapshot] | None = None

    def track(self, store) -> None:
        """Follow `store`: a change to it since the last record travels with the
        next record pushed, as a ColourRecord in the same step. So a colour a
        command added, or a lease taken when you picked a colour, is undone
        with the voxels that use it."""
        self.colours = store
        self._colours_seen = None if store is None else (store.version, store.snapshot())
        if store is not None:
            store.on_sync.append(self._outside_change)

    def _outside_change(self, version_before: int) -> None:
        """The store took on palette rows written outside it. With no command's
        change pending, they are the new start, not part of the next step."""
        if self.colours is not None and self._colours_seen is not None and self._colours_seen[0] == version_before:
            self._colours_now()

    def _colours_changed(self):
        """A ColourRecord of the store's change since the last record, or None."""
        store = self.colours
        if store is None or self._colours_seen is None or store.version == self._colours_seen[0]:
            return None
        record = ColourRecord(self._colours_seen[1], store.snapshot())
        self._colours_seen = (store.version, record.after)
        return record

    def push_colours(self, session: int = 0) -> bool:
        """Push the store's change since the last record as its own step: a
        command that changed only colours. With a `session`, the change folds
        into the top record of the same session (one popup drag, one step).
        False when nothing changed."""
        record = self._colours_changed()
        if record is None:
            return False
        record.session = session
        top = self.peek()
        if session and not self._redo and isinstance(top, ColourRecord) and top.session == session:
            top.after = record.after
            self.pushes += 1
            return True
        self.push(record)
        return True

    def _colours_now(self) -> None:
        if self.colours is not None:
            self._colours_seen = (self.colours.version, self.colours.snapshot())

    def push(self, record: Record, merge: bool = False) -> None:
        """Push a record. With `merge`, a PaletteRecord of the same slot and session
        as the top record folds into it. Never after an undo: the redo list must be empty.
        """
        self.pushes += 1
        if merge and not self._redo and self._mergeable(record):
            top = self._undo[-1]
            top.after = record.after
            if record.materials_after is not None:
                top.materials_after = record.materials_after
                if top.materials_before is None:
                    top.materials_before = record.materials_before
            return
        colours = self._colours_changed()
        if colours is not None and not isinstance(record, ColourRecord):
            record.colour_change = colours  # undone and redone with the record
        elif colours is not None:
            record.before = colours.before
        if not getattr(record, "label", ""):
            record.label = UndoStack.label or kind_name(record)
        record.when = time.time()
        if len(self._undo) == self._undo.maxlen:
            self.dropped += 1  # the deque drops the oldest on append
        self._undo.append(record)
        self._redo.clear()
        if UndoStack.after_push is not None:
            amended = UndoStack.after_push(record)
            if amended is not None:
                amended.label, amended.when = record.label, record.when
                if getattr(record, "colour_change", None) is not None:
                    amended.colour_change = record.colour_change
                self._undo[-1] = record = amended
        if UndoStack.after_apply is not None:
            UndoStack.after_apply(record, False)
        # Drop the oldest records over the byte budget. The newest always stays.
        total = sum(r.nbytes for r in self._undo)
        while total > self._max_bytes and len(self._undo) > 1:
            total -= self._undo.popleft().nbytes
            self.dropped += 1

    def _mergeable(self, record: Record) -> bool:
        top = self.peek()
        return (
            isinstance(record, PaletteRecord)
            and isinstance(top, PaletteRecord)
            and top.index == record.index
            and top.session == record.session
        )

    def peek(self) -> Record | None:
        """The record the next undo would apply, or None."""
        return self._undo[-1] if self._undo else None

    def peek_redo(self) -> Record | None:
        """The record the next redo would apply, or None."""
        return self._redo[-1] if self._redo else None

    def undo(self, target) -> bool:
        """Undo the top record on `target`: a VoxelGrid for StrokeRecord and GridRecord,
        a VoxScene for ModelRecord, SceneRecord, PaletteRecord and SelectionRecord."""
        if not self._undo:
            return False
        record = self._undo.pop()
        with _replay():
            record.undo(target)
        change = getattr(record, "colour_change", None)
        if change is not None and self.colours is not None:
            self.colours.restore(change.before)
        self._colours_now()
        self._redo.append(record)
        self.last_applied = record
        if UndoStack.after_apply is not None:
            UndoStack.after_apply(record, True)
        return True

    def redo(self, target) -> bool:
        if not self._redo:
            return False
        record = self._redo.pop()
        with _replay():
            record.redo(target)
        change = getattr(record, "colour_change", None)
        if change is not None and self.colours is not None:
            self.colours.restore(change.after)
        self._colours_now()
        self._undo.append(record)
        self.last_applied = record
        if UndoStack.after_apply is not None:
            UndoStack.after_apply(record, False)
        return True

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self.last_applied = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def __len__(self) -> int:
        return len(self._undo)

    def steps(self) -> list[Record]:
        """Every record, oldest first: the done ones, then the undone ones
        in the order redo brings them back."""
        return list(self._undo) + self._redo[::-1]

    @property
    def position(self) -> int:
        """How many of `steps()` are done."""
        return len(self._undo)

    def go_to(self, position: int, target, after=None) -> int:
        """Undo or redo until `position` steps are done. `after(undone)` runs
        after each step. Returns the number of steps applied."""
        position = max(0, min(len(self._undo) + len(self._redo), int(position)))
        applied = 0
        while len(self._undo) > position and self.undo(target):
            applied += 1
            if after is not None:
                after(True)
        while len(self._undo) < position and self.redo(target):
            applied += 1
            if after is not None:
                after(False)
        return applied
