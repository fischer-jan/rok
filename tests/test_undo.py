"""Stroke recording and the bounded undo stack."""

import copy

import numpy as np

from rok.core import anim
from rok.core.grid import CHUNK_SIZE, VoxelGrid
from rok.core.grid_ops import rotate_model
from rok.core.material import DEFAULT, MaterialTable
from rok.core.scene import VoxScene
from rok.core.undo import (
    MATERIAL_ENTRY_BYTES,
    AnimRecord,
    CompoundRecord,
    GridRecord,
    ModelRecord,
    PaletteRecord,
    SceneRecord,
    SelectionRecord,
    StrokeRecord,
    StrokeRecorder,
    UndoStack,
    changes_saved_data,
    selection_record,
    stacked_record,
)
from rok.core.vox_read import read_scene, read_vox

from helpers import FIXTURES

GOLDEN = FIXTURES / "phase3" / "two_models.vox"


def _grid_64() -> VoxelGrid:
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    return grid


def test_recorder_snapshots_on_first_touch_only():
    grid = _grid_64()
    grid.set_voxel(1, 1, 1, 7)
    rec = StrokeRecorder(grid)
    assert rec.set(1, 1, 1, 2) is True
    assert rec.set(2, 2, 2, 3) is True
    assert rec.set(2, 2, 2, 3) is False
    record = rec.finish()
    assert record is not None
    assert set(record.chunks) == {(0, 0, 0)}
    before, after = record.chunks[(0, 0, 0)]
    assert before[1, 1, 1] == 7
    assert before[2, 2, 2] == 0
    assert after[1, 1, 1] == 2
    assert after[2, 2, 2] == 3


def test_finish_drops_unchanged_chunks():
    grid = _grid_64()
    rec = StrokeRecorder(grid)
    rec.set(1, 1, 1, 5)
    rec.set(40, 1, 1, 0)
    record = rec.finish()
    assert record is not None
    assert set(record.chunks) == {(0, 0, 0)}


def test_finish_returns_none_when_edits_cancel_out():
    grid = _grid_64()
    rec = StrokeRecorder(grid)
    rec.set(1, 1, 1, 5)
    rec.set(1, 1, 1, 0)
    assert rec.finish() is None

    empty = StrokeRecorder(grid)
    assert empty.finish() is None


def test_revert_restores_grid():
    rng = np.random.default_rng(1)
    start = rng.integers(0, 4, size=(64, 32, 32), dtype=np.uint8)
    grid = VoxelGrid((64, 32, 32), start)
    rec = StrokeRecorder(grid)
    rec.set(1, 1, 1, 200)
    rec.set(50, 3, 3, 201)
    rec.set(31, 31, 31, 202)
    assert not np.array_equal(grid.voxels, start)
    rec.revert()
    assert np.array_equal(grid.voxels, start)


def test_boundary_edit_snapshots_own_chunk_only():
    grid = _grid_64()
    rec = StrokeRecorder(grid)
    rec.set(CHUNK_SIZE, 0, 0, 3)
    record = rec.finish()
    assert record is not None
    assert set(record.chunks) == {(1, 0, 0)}


def _stroke(grid: VoxelGrid, x: int, value: int) -> StrokeRecord:
    rec = StrokeRecorder(grid)
    rec.set(x, 1, 1, value)
    record = rec.finish()
    assert record is not None
    return record


def test_push_clears_redo():
    grid = _grid_64()
    stack = UndoStack()
    assert not stack.can_undo
    assert not stack.can_redo
    stack.push(_stroke(grid, 1, 1))
    assert stack.undo(grid)
    assert stack.can_redo
    stack.push(_stroke(grid, 2, 2))
    assert not stack.can_redo
    assert len(stack) == 1


def test_undo_redo_restore_bytes_and_mark_dirty():
    grid = _grid_64()
    stack = UndoStack()
    original = grid.voxels.copy()
    stack.push(_stroke(grid, 1, 1))
    stack.push(_stroke(grid, 40, 2))
    edited = grid.voxels.copy()
    grid.clear_dirty()

    assert stack.undo(grid)
    assert grid.dirty_chunks() == {(1, 0, 0)}
    assert stack.undo(grid)
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}
    assert np.array_equal(grid.voxels, original)
    assert not stack.undo(grid)
    assert len(stack) == 0

    grid.clear_dirty()
    assert stack.redo(grid)
    assert grid.dirty_chunks() == {(0, 0, 0)}
    assert stack.redo(grid)
    assert np.array_equal(grid.voxels, edited)
    assert not stack.redo(grid)
    assert len(stack) == 2


def test_clear_empties_both_sides():
    grid = _grid_64()
    stack = UndoStack()
    stack.push(_stroke(grid, 1, 1))
    stack.push(_stroke(grid, 2, 2))
    stack.undo(grid)
    stack.clear()
    assert len(stack) == 0
    assert not stack.can_undo
    assert not stack.can_redo


def test_capacity_drops_oldest():
    grid = VoxelGrid((128, 32, 32))
    stack = UndoStack(capacity=64)
    for i in range(65):
        stack.push(_stroke(grid, i, 1))
    assert len(stack) == 64
    for _ in range(64):
        assert stack.undo(grid)
    assert not stack.undo(grid)
    assert grid.get_voxel(0, 1, 1) == 1
    assert int(grid.voxels.sum()) == 1


def test_golden_random_strokes_round_trip():
    model = read_vox(FIXTURES / "robot.vox")
    grid = VoxelGrid.from_model(model)
    rng = np.random.default_rng(42)
    stack = UndoStack()
    sx, sy, sz = grid.size
    pushed = 0
    for _ in range(30):
        rec = StrokeRecorder(grid)
        for _ in range(int(rng.integers(1, 6))):
            x = int(rng.integers(0, sx))
            y = int(rng.integers(0, sy))
            z = int(rng.integers(0, sz))
            rec.set(x, y, z, int(rng.integers(0, 256)))
        record = rec.finish()
        if record is not None:
            stack.push(record)
            pushed += 1
    assert pushed > 0
    edited = grid.voxels.copy()
    assert not np.array_equal(edited, model.voxels)

    while stack.undo(grid):
        pass
    assert np.array_equal(grid.voxels, model.voxels)

    while stack.redo(grid):
        pass
    assert np.array_equal(grid.voxels, edited)
    assert len(stack) == pushed


def _fixture_grid(stem: str) -> VoxelGrid:
    return VoxelGrid.from_model(read_vox(FIXTURES / f"{stem}.vox"))


def _box(x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> np.ndarray:
    """Every cell of the half-open box as an (N, 3) array."""
    xs, ys, zs = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), np.arange(z0, z1), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)


def _flipped(grid: VoxelGrid) -> np.ndarray:
    """Flip on Y. The fixture models are symmetric on X, so an X flip would be a no-op."""
    return np.ascontiguousarray(grid.voxels[:, ::-1, :])


def _resized(grid: VoxelGrid, size: tuple[int, int, int]) -> np.ndarray:
    new = np.zeros(size, dtype=np.uint8)
    keep = tuple(slice(0, min(a, b)) for a, b in zip(grid.size, size))
    new[keep] = grid.voxels[keep]
    return new


def _grid_change(grid: VoxelGrid, after: np.ndarray) -> GridRecord:
    """Apply a whole-grid change the way flip, rotate and resize do, and return its record."""
    record = GridRecord(grid.voxels.copy(), np.array(after, dtype=np.uint8, copy=True))
    grid.replace_voxels(after)
    return record


def test_set_many_snapshots_touched_chunks_once():
    grid = VoxelGrid((96, 64, 32))
    rec = StrokeRecorder(grid)
    cells = np.array([[1, 1, 1], [2, 2, 2], [40, 40, 5], [70, 1, 1], [CHUNK_SIZE, 40, 5]])
    assert rec.set_many(cells, 6) == 5
    assert set(rec._before) == {(0, 0, 0), (1, 1, 0), (2, 0, 0)}
    first = {chunk: data.copy() for chunk, data in rec._before.items()}
    assert first[(0, 0, 0)][1, 1, 1] == 0

    assert rec.set_many(np.array([[3, 3, 3], [41, 41, 5]]), 7) == 2
    assert set(rec._before) == set(first)
    for chunk, data in first.items():
        assert np.array_equal(rec._before[chunk], data)

    record = rec.finish()
    assert record is not None
    assert set(record.chunks) == set(first)
    before, after = record.chunks[(0, 0, 0)]
    assert before[1, 1, 1] == 0 and after[1, 1, 1] == 6
    assert before[3, 3, 3] == 0 and after[3, 3, 3] == 7


def test_set_many_and_set_share_snapshots():
    grid = _grid_64()
    rec = StrokeRecorder(grid)
    rec.set(1, 1, 1, 3)
    rec.set_many(np.array([[2, 2, 2], [40, 2, 2]]), 4)
    assert set(rec._before) == {(0, 0, 0), (1, 0, 0)}
    assert rec._before[(0, 0, 0)][1, 1, 1] == 0
    rec.set(41, 2, 2, 5)
    assert rec._before[(1, 0, 0)][8, 2, 2] == 0
    assert rec.set_many(np.zeros((0, 3), dtype=np.intp), 9) == 0


def test_set_many_ignore_cut_reaches_a_cut_cell():
    """A palette swap's remap must land on a cut voxel too, not skip it."""
    grid = _grid_64()
    grid.set_voxel(4, 4, 4, 3)
    grid.set_section(4)  # cuts z = 4 and above
    grid.clear_dirty()
    rec = StrokeRecorder(grid)
    assert rec.set_many(np.array([[4, 4, 4]]), 9) == 0  # the plain path still refuses it
    assert rec.set_many(np.array([[4, 4, 4]]), 9, ignore_cut=True) == 1
    assert grid.voxels[4, 4, 4] == 9
    assert grid.visible[4, 4, 4] == 0  # unchanged: the cell is still cut
    record = rec.finish()
    assert record is not None
    before, after = record.chunks[(0, 0, 0)]
    assert before[4, 4, 4] == 3 and after[4, 4, 4] == 9


def test_revert_after_set_many_restores_fixture():
    model = read_vox(FIXTURES / "robot.vox")
    grid = VoxelGrid.from_model(model)
    rec = StrokeRecorder(grid)
    assert rec.set_many(_box(0, 24, 0, 30, 0, 20), 0) > 0
    assert int(grid.voxels.sum()) == 0
    rec.set_many(_box(2, 10, 3, 12, 1, 5), 17)
    rec.set(0, 0, 0, 200)
    rec.revert()
    assert grid.voxels.tobytes() == model.voxels.tobytes()
    assert grid.size == model.size


def test_finish_drops_unchanged_chunks_after_set_many():
    grid = _grid_64()
    grid.set_voxel(40, 1, 1, 5)
    rec = StrokeRecorder(grid)
    rec.set_many(np.array([[1, 1, 1], [40, 1, 1]]), 5)
    assert set(rec._before) == {(0, 0, 0), (1, 0, 0)}
    record = rec.finish()
    assert record is not None
    assert set(record.chunks) == {(0, 0, 0)}

    rec = StrokeRecorder(grid)
    rec.set_many(np.array([[40, 1, 1]]), 5)
    assert rec.finish() is None


def test_stroke_record_undo_redo_directly():
    grid = _grid_64()
    rec = StrokeRecorder(grid)
    rec.set(1, 1, 1, 5)
    rec.set_many(np.array([[40, 1, 1], [CHUNK_SIZE, 0, 0]]), 6)
    record = rec.finish()
    assert record is not None
    edited = grid.voxels.copy()
    grid.clear_dirty()

    record.undo(grid)
    assert int(grid.voxels.sum()) == 0
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}

    grid.clear_dirty()
    record.redo(grid)
    assert np.array_equal(grid.voxels, edited)
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}


def test_grid_record_flip_through_stack():
    grid = _fixture_grid("bird")
    original = grid.voxels.copy()
    stack = UndoStack()
    flipped = _flipped(grid)
    assert not np.array_equal(flipped, original)
    stack.push(_grid_change(grid, flipped))
    assert np.array_equal(grid.voxels, flipped)
    grid.clear_dirty()

    assert stack.undo(grid)
    assert grid.size == (80, 68, 22)
    assert grid.voxels.tobytes() == original.tobytes()
    assert grid.dirty_chunks() == {(cx, cy, 0) for cx in range(3) for cy in range(3)}

    assert stack.redo(grid)
    assert grid.voxels.tobytes() == flipped.tobytes()
    assert len(stack) == 1


def test_grid_record_resize_through_stack():
    grid = _fixture_grid("bird")
    original = grid.voxels.copy()
    stack = UndoStack()
    small = _resized(grid, (40, 40, 22))
    stack.push(_grid_change(grid, small))
    assert grid.size == (40, 40, 22)
    big = _resized(grid, (100, 40, 33))
    stack.push(_grid_change(grid, big))
    assert grid.size == (100, 40, 33)
    assert grid.chunk_counts == (4, 2, 2)
    grid.clear_dirty()

    assert stack.undo(grid)
    assert grid.size == (40, 40, 22)
    assert grid.voxels.tobytes() == small.tobytes()
    assert grid.dirty_chunks() == {(cx, cy, 0) for cx in range(2) for cy in range(2)}
    assert stack.undo(grid)
    assert grid.size == (80, 68, 22)
    assert grid.voxels.tobytes() == original.tobytes()
    assert not stack.undo(grid)

    assert stack.redo(grid)
    assert grid.size == (40, 40, 22)
    assert stack.redo(grid)
    assert grid.size == (100, 40, 33)
    assert grid.voxels.tobytes() == big.tobytes()
    assert not stack.redo(grid)


def test_grid_record_holds_copies():
    grid = VoxelGrid((4, 4, 4))
    after = np.zeros((4, 4, 4), dtype=np.uint8)
    after[1, 1, 1] = 3
    record = GridRecord(grid.voxels.copy(), after.copy())
    record.redo(grid)
    grid.set_voxel(1, 1, 1, 9)
    assert record.after[1, 1, 1] == 3
    record.undo(grid)
    assert int(grid.voxels.sum()) == 0
    grid.set_voxel(0, 0, 0, 1)
    assert record.before[0, 0, 0] == 0


def test_nbytes_of_both_record_types():
    grid = VoxelGrid((64, 33, 32))
    rec = StrokeRecorder(grid)
    rec.set(1, 1, 1, 5)
    rec.set(40, CHUNK_SIZE, 1, 6)
    record = rec.finish()
    assert record is not None
    assert set(record.chunks) == {(0, 0, 0), (1, 1, 0)}
    assert record.nbytes == 2 * (32 * 32 * 32) + 2 * (32 * 1 * 32)

    before = np.zeros((32, 32, 32), dtype=np.uint8)
    after = np.zeros((256, 256, 256), dtype=np.uint8)
    assert GridRecord(before, after).nbytes == 32**3 + 256**3
    assert GridRecord(before, before.copy()).nbytes == 64 * 1024


def test_byte_budget_evicts_oldest_never_newest():
    grid = VoxelGrid((16, 16, 16))
    one = 2 * 16 * 16 * 16
    stack = UndoStack(max_bytes=one)
    stack.push(_grid_change(grid, _resized(grid, (16, 16, 16))))
    assert len(stack) == 1
    stack.push(_grid_change(grid, _resized(grid, (16, 16, 16))))
    assert len(stack) == 1

    stack = UndoStack(max_bytes=3 * one)
    records = [_grid_change(grid, _resized(grid, (16, 16, 16))) for _ in range(5)]
    for record in records:
        stack.push(record)
    assert len(stack) == 3
    assert list(stack._undo) == records[2:]

    # A record bigger than the whole budget stays as the only record.
    huge = _grid_change(grid, _resized(grid, (64, 64, 64)))
    stack.push(huge)
    assert len(stack) == 1
    assert stack._undo[0] is huge
    assert stack.undo(grid)
    assert grid.size == (16, 16, 16)


def test_byte_budget_counts_stroke_records_too():
    grid = VoxelGrid((128, 32, 32))
    stack = UndoStack(max_bytes=5 * 32**3)
    for i in range(4):
        stack.push(_stroke(grid, i * 32, 1))
    assert len(stack) == 2
    assert stack.undo(grid)
    assert stack.undo(grid)
    assert not stack.undo(grid)
    assert grid.get_voxel(0, 1, 1) == 1
    assert grid.get_voxel(32, 1, 1) == 1
    assert grid.get_voxel(64, 1, 1) == 0
    assert grid.get_voxel(96, 1, 1) == 0


def test_capacity_cap_still_64_by_default():
    grid = VoxelGrid((128, 32, 32))
    stack = UndoStack()
    for i in range(70):
        stack.push(_stroke(grid, i, 1))
    assert len(stack) == 64
    for _ in range(64):
        assert stack.undo(grid)
    assert not stack.undo(grid)
    assert int(grid.voxels.sum()) == 6


def test_golden_mixed_stack_round_trip():
    model = read_vox(FIXTURES / "robot.vox")
    grid = VoxelGrid.from_model(model)
    rng = np.random.default_rng(2026)
    stack = UndoStack()
    sx, sy, sz = grid.size
    for _ in range(20):
        rec = StrokeRecorder(grid)
        for _ in range(int(rng.integers(1, 6))):
            x = int(rng.integers(0, sx))
            y = int(rng.integers(0, sy))
            z = int(rng.integers(0, sz))
            rec.set(x, y, z, int(rng.integers(0, 256)))
        record = rec.finish()
        if record is not None:
            stack.push(record)
    pushed_strokes = len(stack)
    assert pushed_strokes > 0

    stack.push(_grid_change(grid, _flipped(grid)))
    stack.push(_grid_change(grid, _resized(grid, (40, 30, 33))))
    assert grid.size == (40, 30, 33)

    rec = StrokeRecorder(grid)
    assert rec.set_many(_box(30, 40, 5, 12, 25, 33), 42) > 0
    record = rec.finish()
    assert record is not None
    stack.push(record)
    assert len(stack) == pushed_strokes + 3

    edited = grid.voxels.copy()
    edited_size = grid.size

    steps = 0
    while stack.undo(grid):
        steps += 1
    assert steps == pushed_strokes + 3
    assert grid.size == model.size
    assert grid.voxels.tobytes() == model.voxels.tobytes()

    while stack.redo(grid):
        pass
    assert grid.size == edited_size
    assert grid.voxels.tobytes() == edited.tobytes()
    assert len(stack) == pushed_strokes + 3


def test_last_applied_names_the_record_of_the_last_undo_or_redo():
    """Undo drops the selection after a GridRecord step; it reads this field."""
    grid = _fixture_grid("bird")
    stack = UndoStack()
    assert stack.last_applied is None

    recorder = StrokeRecorder(grid)
    recorder.set(0, 0, 0, 7)
    stroke = recorder.finish()
    stack.push(stroke)
    flip = _grid_change(grid, _flipped(grid))
    stack.push(flip)
    assert stack.last_applied is None  # push applies nothing

    assert stack.undo(grid)
    assert stack.last_applied is flip
    assert stack.undo(grid)
    assert stack.last_applied is stroke
    assert not stack.undo(grid)
    assert stack.last_applied is stroke  # a refused step changes nothing

    assert stack.redo(grid)
    assert stack.last_applied is stroke
    assert stack.redo(grid)
    assert stack.last_applied is flip

    stack.clear()
    assert stack.last_applied is None


# --- Phase 3: records over a scene ------------------------------------------------


def _scene_two() -> VoxScene:
    """The robot and the bird as two models over one palette."""
    scene = VoxScene.single(_fixture_grid("robot"), (0, 0, 0))
    scene.add(_fixture_grid("bird"), "Bird", (40, 0, 0))
    return scene


def _golden_scene() -> VoxScene:
    return VoxScene.from_data(read_scene(GOLDEN))


def _stroke_in(scene: VoxScene, uid: int, x: int, value: int) -> ModelRecord:
    grid = scene.by_uid(uid).grid
    return ModelRecord(uid, _stroke(grid, x, value))


def _palette_edit(scene: VoxScene, index: int, rgba, session: int) -> PaletteRecord:
    before = scene.palette.copy()
    scene.palette[index] = rgba
    return PaletteRecord(index, before, scene.palette.copy(), session)


def _snapshot(scene: VoxScene) -> dict:
    return {
        "uids": [m.uid for m in scene.models],
        "names": [m.name for m in scene.models],
        "positions": [m.position for m in scene.models],
        "voxels": [m.grid.voxels.copy() for m in scene.models],
        "palette": scene.palette.copy(),
    }


def _assert_state(scene: VoxScene, snap: dict) -> None:
    assert [m.uid for m in scene.models] == snap["uids"]
    assert [m.name for m in scene.models] == snap["names"]
    assert [m.position for m in scene.models] == snap["positions"]
    for model, voxels in zip(scene.models, snap["voxels"]):
        assert model.grid.size == voxels.shape
        assert model.grid.voxels.tobytes() == voxels.tobytes()
    assert np.array_equal(scene.palette, snap["palette"])
    for model in scene.models:
        assert model.grid.palette is scene.palette


def test_model_record_routes_by_uid():
    scene = _scene_two()
    robot, bird = scene.models
    start_robot = robot.grid.voxels.copy()
    start_bird = bird.grid.voxels.copy()
    stack = UndoStack()

    stack.push(_stroke_in(scene, bird.uid, 3, 9))
    stack.push(_stroke_in(scene, robot.uid, 3, 8))
    assert bird.grid.get_voxel(3, 1, 1) == 9
    assert robot.grid.get_voxel(3, 1, 1) == 8
    scene.models.reverse()  # order in the list does not matter, the uid does

    assert stack.undo(scene)
    assert robot.grid.voxels.tobytes() == start_robot.tobytes()
    assert bird.grid.get_voxel(3, 1, 1) == 9
    assert stack.last_applied.uid == robot.uid
    assert stack.undo(scene)
    assert bird.grid.voxels.tobytes() == start_bird.tobytes()
    assert stack.redo(scene)
    assert bird.grid.get_voxel(3, 1, 1) == 9
    assert robot.grid.get_voxel(3, 1, 1) == 0


def test_model_record_restores_position_and_name():
    scene = _scene_two()
    bird = scene.models[1]
    bird.position = (41, -2, 5)
    bird.name = "Red"
    record = ModelRecord(bird.uid, None, (40, 0, 0), (41, -2, 5), "Bird", "Red")
    assert record.nbytes == 0
    assert record.moved
    stack = UndoStack()
    stack.push(record)
    assert stack.undo(scene)
    assert bird.position == (40, 0, 0) and bird.name == "Bird"
    assert stack.redo(scene)
    assert bird.position == (41, -2, 5) and bird.name == "Red"

    # A stroke record leaves position and name alone.
    stroke = _stroke_in(scene, bird.uid, 5, 4)
    assert not stroke.moved
    stack.push(stroke)
    stack.undo(scene)
    assert bird.position == (41, -2, 5) and bird.name == "Red"


def test_model_record_with_grid_record_and_move():
    scene = _scene_two()
    robot = scene.models[0]
    original = robot.grid.voxels.copy()
    flipped = _flipped(robot.grid)
    record = ModelRecord(robot.uid, _grid_change(robot.grid, flipped), (0, 0, 0), (2, 0, 0))
    robot.position = (2, 0, 0)
    assert record.nbytes == 2 * original.nbytes
    stack = UndoStack()
    stack.push(record)
    assert stack.undo(scene)
    assert robot.grid.voxels.tobytes() == original.tobytes()
    assert robot.position == (0, 0, 0)
    assert stack.redo(scene)
    assert robot.grid.voxels.tobytes() == flipped.tobytes()
    assert robot.position == (2, 0, 0)


def test_scene_record_restores_list_order_and_active():
    scene = _scene_two()
    robot, bird = scene.models
    before = tuple(scene.models)
    extra = scene.add(VoxelGrid((8, 8, 8)), "Third", (0, 50, 0))
    scene.models.reverse()
    after = tuple(scene.models)
    record = SceneRecord(before, after, robot.uid, extra.uid)
    assert record.nbytes == 8 * 8 * 8

    assert record.undo(scene) is None
    assert scene.models == [robot, bird]
    record.redo(scene)
    assert scene.models == [extra, bird, robot]
    assert record.active_before == robot.uid and record.active_after == extra.uid


def test_scene_record_nbytes_counts_a_deleted_model_once():
    scene = _scene_two()
    robot, bird = scene.models
    before = tuple(scene.models)
    scene.remove(1)
    after = tuple(scene.models)
    record = SceneRecord(before, after, bird.uid, robot.uid)
    assert record.nbytes == bird.grid.voxels.nbytes
    assert record.nbytes == 80 * 68 * 22

    # Reorder only: nothing held.
    assert SceneRecord(before, tuple(reversed(before)), 1, 1).nbytes == 0
    # Inner records add their bytes.
    palette = PaletteRecord(None, scene.palette.copy(), scene.palette.copy(), 0)
    assert SceneRecord(before, after, 1, 1, palette).nbytes == bird.grid.voxels.nbytes + 2 * 1024


def test_scene_record_undoes_inner_after_the_list_and_redoes_it_before():
    scene = _scene_two()
    robot, bird = scene.models
    before = tuple(scene.models)
    stroke = _stroke_in(scene, robot.uid, 2, 7)
    scene.remove(1)
    after = tuple(scene.models)
    stack = UndoStack()
    stack.push(SceneRecord(before, after, bird.uid, robot.uid, stroke))

    assert stack.undo(scene)
    assert scene.models == [robot, bird]
    assert robot.grid.get_voxel(2, 1, 1) == 0
    assert stack.redo(scene)
    assert scene.models == [robot]
    assert robot.grid.get_voxel(2, 1, 1) == 7


# --- AnimRecord: the timeline as an undo step ---------------------------------


def _position_track(scene: VoxScene, uid: int) -> anim.Track:
    """Add a bare POSITION track for `uid` to the scene's timeline."""
    track = anim.Track(anim.Target(anim.PART, uid))
    scene.timeline.tracks.append(track)
    return track


def test_anim_record_undo_and_redo_put_the_whole_timeline_back():
    scene = _scene_two()
    bird = scene.models[1]
    track = _position_track(scene, bird.uid)
    before = copy.deepcopy(scene.timeline)
    anim.set_key(track, anim.POSITION, anim.Key(0, (0, 0, 0)))
    anim.set_key(track, anim.POSITION, anim.Key(10, (1, 0, 0)))
    record = AnimRecord(before, scene.timeline)

    stack = UndoStack()
    stack.push(record)
    assert stack.undo(scene)
    assert scene.timeline.track_of(bird.uid).keys(anim.POSITION) == []
    assert stack.redo(scene)
    values = [key.value for key in scene.timeline.track_of(bird.uid).keys(anim.POSITION)]
    assert values == [(0, 0, 0), (1, 0, 0)]


def test_anim_record_copies_at_construction_so_later_edits_do_not_reach_it():
    scene = _scene_two()
    track = _position_track(scene, scene.models[0].uid)
    anim.set_key(track, anim.POSITION, anim.Key(0, (0, 0, 0)))
    # The caller hands over the live timeline itself, not a copy of its own.
    record = AnimRecord(scene.timeline, scene.timeline)
    anim.set_key(track, anim.POSITION, anim.Key(5, (2, 2, 2)))  # mutated after construction

    assert [k.value for k in record.before.tracks[0].keys(anim.POSITION)] == [(0, 0, 0)]
    assert [k.value for k in record.after.tracks[0].keys(anim.POSITION)] == [(0, 0, 0)]
    live_values = [k.value for k in scene.timeline.tracks[0].keys(anim.POSITION)]
    assert live_values == [(0, 0, 0), (2, 2, 2)]


def test_anim_record_copies_again_on_undo_so_a_later_edit_does_not_reach_it():
    scene = _scene_two()
    track = _position_track(scene, scene.models[0].uid)
    before = copy.deepcopy(scene.timeline)
    anim.set_key(track, anim.POSITION, anim.Key(0, (0, 0, 0)))
    record = AnimRecord(before, scene.timeline)

    record.undo(scene)
    scene.timeline.tracks[0].properties[anim.POSITION] = anim.Property(anim.POSITION)
    anim.set_key(scene.timeline.tracks[0], anim.POSITION, anim.Key(9, (3, 3, 3)))
    assert record.before.tracks[0].keys(anim.POSITION) == []


def test_changes_saved_data_is_true_for_an_anim_record():
    scene = VoxScene()
    assert changes_saved_data(AnimRecord(scene.timeline, scene.timeline)) is True


def test_anim_record_is_allowed_in_compound_record_parts():
    scene = _scene_two()
    palette_part = PaletteRecord(5, scene.palette.copy(), scene.palette.copy(), 0)
    anim_part = AnimRecord(scene.timeline, scene.timeline)
    record = CompoundRecord((palette_part, anim_part))
    assert record.first(AnimRecord) is anim_part


def test_a_stack_of_stroke_anim_stroke_undoes_in_order():
    scene = _scene_two()
    bird = scene.models[1]
    grid = bird.grid
    stroke_a = _stroke(grid, 1, 5)
    track = _position_track(scene, bird.uid)
    before = copy.deepcopy(scene.timeline)
    anim.set_key(track, anim.POSITION, anim.Key(0, (0, 0, 0)))
    anim_record = AnimRecord(before, scene.timeline)
    stroke_b = _stroke(grid, 2, 6)

    stack = UndoStack()
    stack.push(stroke_a)
    stack.push(anim_record)
    stack.push(stroke_b)

    assert stack.undo(grid)
    assert grid.get_voxel(2, 1, 1) == 0
    assert stack.undo(scene)
    assert scene.timeline.track_of(bird.uid).keys(anim.POSITION) == []
    assert stack.undo(grid)
    assert grid.get_voxel(1, 1, 1) == 0
    assert not stack.can_undo


def test_scene_record_carries_the_deleted_models_track():
    scene = _scene_two()
    robot, bird = scene.models
    track = _position_track(scene, bird.uid)
    anim.set_key(track, anim.POSITION, anim.Key(0, (0, 0, 0)))
    before_models = tuple(scene.models)
    timeline_before = copy.deepcopy(scene.timeline)

    scene.remove(1)
    scene.timeline.tracks.remove(track)
    after_models = tuple(scene.models)
    timeline_after = copy.deepcopy(scene.timeline)

    record = SceneRecord(
        before_models,
        after_models,
        robot.uid,
        robot.uid,
        timeline_before=timeline_before,
        timeline_after=timeline_after,
    )
    stack = UndoStack()
    stack.push(record)

    assert stack.undo(scene)
    assert scene.models == [robot, bird]
    assert scene.timeline.track_of(bird.uid) is not None

    assert stack.redo(scene)
    assert scene.models == [robot]
    assert scene.timeline.track_of(bird.uid) is None


def test_scene_record_without_timeline_snapshots_leaves_the_timeline_alone():
    scene = _scene_two()
    robot, bird = scene.models
    before = tuple(scene.models)
    scene.models.reverse()
    after = tuple(scene.models)
    marker = scene.timeline
    record = SceneRecord(before, after, robot.uid, bird.uid)  # old call shape, no timeline
    record.undo(scene)
    assert scene.timeline is marker
    record.redo(scene)
    assert scene.timeline is marker


def test_palette_record_writes_in_place():
    scene = _scene_two()
    palette = scene.palette
    stack = UndoStack()
    stack.push(_palette_edit(scene, 5, (1, 2, 3, 255), 1))
    assert stack.undo(scene)
    assert scene.palette is palette
    assert tuple(scene.palette[5]) != (1, 2, 3, 255)
    for model in scene.models:
        assert model.grid.palette is palette
    assert stack.redo(scene)
    assert tuple(scene.palette[5]) == (1, 2, 3, 255)


def test_palette_record_merges_inside_a_session_only():
    scene = _scene_two()
    original = scene.palette.copy()
    stack = UndoStack()
    stack.push(_palette_edit(scene, 5, (10, 0, 0, 255), 1), merge=True)
    first = stack.peek()
    assert isinstance(first, PaletteRecord)
    stack.push(_palette_edit(scene, 5, (20, 0, 0, 255), 1), merge=True)
    stack.push(_palette_edit(scene, 5, (30, 0, 0, 255), 1), merge=True)
    assert len(stack) == 1
    assert stack.peek() is first
    assert tuple(first.after[5]) == (30, 0, 0, 255)
    assert np.array_equal(first.before, original)

    # Another index in the same session: a new record.
    stack.push(_palette_edit(scene, 6, (0, 10, 0, 255), 1), merge=True)
    assert len(stack) == 2
    # The same index in a new session: a new record.
    stack.push(_palette_edit(scene, 6, (0, 20, 0, 255), 2), merge=True)
    assert len(stack) == 3
    # merge=False never merges.
    stack.push(_palette_edit(scene, 6, (0, 30, 0, 255), 2), merge=False)
    assert len(stack) == 4
    # Whole-palette records (index None) merge with each other inside a session.
    whole = PaletteRecord(None, scene.palette.copy(), scene.palette.copy(), 2)
    scene.palette[7] = (0, 0, 10, 255)
    whole.after = scene.palette.copy()
    stack.push(whole, merge=True)
    stack.push(PaletteRecord(None, whole.after.copy(), whole.after.copy(), 2), merge=True)
    assert len(stack) == 5
    assert stack.peek() is whole
    # index None does not merge with a slot record of the same session.
    stack.push(_palette_edit(scene, 7, (0, 0, 20, 255), 2), merge=True)
    assert len(stack) == 6

    edited = scene.palette.copy()
    while stack.undo(scene):
        pass
    assert np.array_equal(scene.palette, original)
    while stack.redo(scene):
        pass
    assert np.array_equal(scene.palette, edited)


def test_palette_record_never_merges_after_an_undo():
    scene = _scene_two()
    stack = UndoStack()
    stack.push(_palette_edit(scene, 5, (10, 0, 0, 255), 1), merge=True)
    stack.push(_palette_edit(scene, 5, (20, 0, 0, 255), 1), merge=True)
    assert len(stack) == 1
    assert stack.undo(scene)
    assert len(stack) == 0 and stack.can_redo
    stack.push(_palette_edit(scene, 5, (30, 0, 0, 255), 1), merge=True)
    assert len(stack) == 1 and not stack.can_redo
    assert tuple(stack.peek().before[5]) != (10, 0, 0, 255)

    # Once a record sits on top again after redo, an edit of the same session merges.
    stack.undo(scene)
    stack.redo(scene)
    stack.push(_palette_edit(scene, 5, (40, 0, 0, 255), 1), merge=True)
    assert len(stack) == 1


def test_merge_needs_a_palette_record_on_top():
    scene = _scene_two()
    stack = UndoStack()
    assert stack.peek() is None
    stack.push(_palette_edit(scene, 5, (10, 0, 0, 255), 1), merge=True)  # empty stack: pushed
    assert len(stack) == 1
    stroke = _stroke_in(scene, 1, 1, 3)
    stack.push(stroke, merge=True)  # not a palette record: pushed
    assert len(stack) == 2
    stack.push(_palette_edit(scene, 5, (20, 0, 0, 255), 1), merge=True)  # top is a stroke: pushed
    assert len(stack) == 3
    assert stack.peek() is not stroke


# --- materials in palette records --------------------------------------------------


def _scene_with_materials() -> VoxScene:
    return VoxScene.from_data(read_scene(FIXTURES / "robot.vox"))


def _material_edit(scene: VoxScene, index: int, material, session: int) -> PaletteRecord:
    """A popup edit of the slot's type: the palette stays, the table changes."""
    before = scene.materials.copy()
    scene.materials.set(index, material)
    return PaletteRecord(
        index, scene.palette.copy(), scene.palette.copy(), session,
        materials_before=before, materials_after=scene.materials.copy(),
    )


def test_palette_record_with_snapshots_restores_the_table_in_place():
    scene = _scene_with_materials()
    table = scene.materials
    start = table.copy()
    metal = DEFAULT.with_type("METAL").with_value(b"_rough", 0.3)
    stack = UndoStack()
    record = _material_edit(scene, 5, metal, 1)
    scene.palette[5] = (9, 9, 9, 255)
    record.after = scene.palette.copy()
    stack.push(record)
    assert scene.materials.get(5) == metal

    assert stack.undo(scene)
    assert scene.materials is table
    assert scene.materials == start
    assert scene.materials.get(5) == DEFAULT
    assert tuple(scene.palette[5]) == tuple(record.before[5])

    assert stack.redo(scene)
    assert scene.materials is table
    assert scene.materials.get(5) == metal
    assert scene.materials == record.materials_after
    assert tuple(scene.palette[5]) == (9, 9, 9, 255)


def test_session_fold_keeps_the_first_materials_before():
    scene = _scene_with_materials()
    start = scene.materials.copy()
    original = scene.palette.copy()
    stack = UndoStack()
    # A colour drag, a type change, another colour drag: one popup session.
    stack.push(_palette_edit(scene, 5, (10, 0, 0, 255), 1), merge=True)
    top = stack.peek()
    assert top.materials_before is None and top.materials_after is None
    glass = DEFAULT.with_type("GLASS")
    stack.push(_material_edit(scene, 5, glass, 1), merge=True)
    assert len(stack) == 1
    assert stack.peek() is top
    assert top.materials_before == start
    assert top.materials_after.get(5) == glass
    emit = DEFAULT.with_type("EMIT")
    stack.push(_material_edit(scene, 5, emit, 1), merge=True)
    assert len(stack) == 1
    assert top.materials_before == start  # the table as the popup found it
    assert top.materials_after.get(5) == emit
    stack.push(_palette_edit(scene, 5, (20, 0, 0, 255), 1), merge=True)
    assert len(stack) == 1
    assert tuple(top.after[5]) == (20, 0, 0, 255)
    assert top.materials_after.get(5) == emit  # a colour edit leaves the snapshots

    assert stack.undo(scene)
    assert np.array_equal(scene.palette, original)
    assert scene.materials == start
    assert stack.redo(scene)
    assert tuple(scene.palette[5]) == (20, 0, 0, 255)
    assert scene.materials.get(5) == emit


def test_palette_record_without_snapshots_leaves_the_table_alone():
    scene = _scene_with_materials()
    table = scene.materials
    metal = DEFAULT.with_type("METAL")
    scene.materials.set(9, metal)
    edited = scene.materials.copy()
    stack = UndoStack()
    stack.push(_palette_edit(scene, 9, (1, 2, 3, 255), 1))
    assert stack.undo(scene)
    assert scene.materials is table
    assert scene.materials == edited
    assert stack.redo(scene)
    assert scene.materials == edited


def test_whole_palette_record_restores_the_table():
    scene = _scene_with_materials()
    table = scene.materials
    start = table.copy()
    palette_before = scene.palette.copy()
    incoming = read_scene(FIXTURES / "bird.vox")
    scene.palette[...] = incoming.palette
    scene.materials.restore(incoming.materials)
    record = PaletteRecord(
        None, palette_before, scene.palette.copy(), 0,
        materials_before=start, materials_after=scene.materials.copy(),
    )
    stack = UndoStack()
    stack.push(record)
    assert stack.undo(scene)
    assert scene.materials is table
    assert scene.materials == start
    assert np.array_equal(scene.palette, palette_before)
    assert stack.redo(scene)
    assert scene.materials == incoming.materials
    assert np.array_equal(scene.palette, incoming.palette)
    # A snapshot is a copy: a later table edit does not reach the record.
    scene.materials.set(1, DEFAULT.with_type("METAL"))
    assert record.materials_after.get(1) == DEFAULT


def test_palette_record_nbytes_counts_the_snapshots():
    scene = _scene_with_materials()
    plain = _palette_edit(scene, 5, (1, 2, 3, 255), 1)
    palette_bytes = plain.before.nbytes + plain.after.nbytes
    assert plain.nbytes == palette_bytes
    both = _material_edit(scene, 5, DEFAULT.with_type("METAL"), 1)
    assert both.nbytes == palette_bytes + 2 * 256 * MATERIAL_ENTRY_BYTES
    one = PaletteRecord(5, plain.before, plain.after, 1, materials_after=MaterialTable())
    assert one.nbytes == palette_bytes
    one.materials_after.set(5, DEFAULT)
    assert one.nbytes == palette_bytes + 256 * MATERIAL_ENTRY_BYTES


def test_byte_budget_counts_scene_records():
    scene = _scene_two()
    robot, bird = scene.models
    before = tuple(scene.models)
    scene.remove(1)
    after = tuple(scene.models)
    stack = UndoStack(max_bytes=bird.grid.voxels.nbytes + 100)
    stack.push(_stroke_in(scene, robot.uid, 1, 5))
    stack.push(SceneRecord(before, after, bird.uid, robot.uid))
    assert len(stack) == 1  # the stroke dropped; the newest always stays
    assert isinstance(stack.peek(), SceneRecord)


def test_golden_two_models_mixed_round_trip():
    scene = _golden_scene()
    robot, car = scene.models
    assert (robot.name, car.name) == ("Robot", "Car")
    start = _snapshot(scene)
    stack = UndoStack()
    active = robot.uid

    # Strokes in both models.
    rng = np.random.default_rng(7)
    for model in (robot, car, robot):
        rec = StrokeRecorder(model.grid)
        sx, sy, sz = model.grid.size
        for _ in range(8):
            rec.set(int(rng.integers(0, sx)), int(rng.integers(0, sy)), int(rng.integers(0, sz)), int(rng.integers(1, 256)))
        record = rec.finish()
        assert record is not None
        stack.push(ModelRecord(model.uid, record))

    # A model move.
    old = car.position
    car.position = (old[0] + 3, old[1] - 2, old[2] + 1)
    stack.push(ModelRecord(car.uid, None, old, car.position))

    # A gizmo rotate of the robot model: one quarter turn about Z around an even pivot.
    pivot2 = tuple(2 * p + d for p, d in zip(robot.position, robot.grid.size))
    turned, new_pos = rotate_model(robot.grid.voxels, robot.position, 2, 1, pivot2)
    record = ModelRecord(robot.uid, _grid_change(robot.grid, turned), robot.position, new_pos)
    robot.position = new_pos
    stack.push(record)
    assert robot.grid.size == (30, 24, 20)

    # A palette edit session of 20 merged steps on a used colour.
    used = int(np.flatnonzero(scene.used_counts()[1:])[0]) + 1
    depth = len(stack)
    for step in range(20):
        stack.push(_palette_edit(scene, used, (step * 10, 255 - step * 10, 7, 255), 3), merge=True)
    assert len(stack) == depth + 1

    # A duplicate of the active model.
    before = tuple(scene.models)
    copy = scene.add(VoxelGrid(robot.grid.size, robot.grid.voxels), scene.unique_name("Robot"),
                     (robot.position[0] + robot.grid.size[0] + 2, robot.position[1], robot.position[2]))
    stack.push(SceneRecord(before, tuple(scene.models), active, copy.uid))
    active = copy.uid
    assert copy.name == "Robot 2"

    # A delete of the car model.
    before = tuple(scene.models)
    scene.remove(scene.index_of(car.uid))
    stack.push(SceneRecord(before, tuple(scene.models), active, active))

    # A split: the top half of the copy goes into a new model.
    before = tuple(scene.models)
    filled = int(np.count_nonzero(copy.grid.voxels))
    z0 = copy.grid.size[2] // 2
    part = np.zeros_like(copy.grid.voxels)
    part[:, :, z0:] = copy.grid.voxels[:, :, z0:]
    rec = StrokeRecorder(copy.grid)
    cells = np.argwhere(part != 0)
    assert rec.set_many(cells, 0) > 0
    piece = scene.add(VoxelGrid(part.shape, part), scene.unique_name(copy.name + " part"), copy.position)
    stack.push(SceneRecord(before, tuple(scene.models), active, piece.uid, ModelRecord(copy.uid, rec.finish())))

    edited = _snapshot(scene)
    assert [m.name for m in scene.models] == ["Robot", "Robot 2", "Robot 2 part"]
    assert int(np.count_nonzero(copy.grid.voxels)) + int(np.count_nonzero(piece.grid.voxels)) == filled
    assert 0 < int(np.count_nonzero(piece.grid.voxels)) < filled
    assert len(stack) == depth + 4

    steps = 0
    while stack.undo(scene):
        steps += 1
    assert steps == depth + 4
    _assert_state(scene, start)
    assert robot.grid.size == (24, 30, 20)

    steps = 0
    while stack.redo(scene):
        steps += 1
    assert steps == depth + 4
    _assert_state(scene, edited)

    # Halfway back: the palette session is one step.
    for _ in range(4):
        assert stack.undo(scene)
    assert np.array_equal(scene.palette, start["palette"])
    assert stack.redo(scene)
    assert tuple(scene.palette[used]) == (190, 65, 7, 255)


# --- CompoundRecord (a paste that also changed the palette) -------------------


def test_compound_record_undoes_parts_in_reverse_and_redoes_in_order():
    scene = _scene_two()
    robot = scene.models[0]
    original = scene.palette.copy()
    palette_part = _palette_edit(scene, 5, (10, 20, 30, 255), 1)
    stroke_part = _stroke_in(scene, robot.uid, 3, 5)
    record = CompoundRecord((palette_part, stroke_part))
    assert record.first(PaletteRecord) is palette_part
    assert record.first(ModelRecord) is stroke_part
    assert record.first(SceneRecord) is None
    assert record.nbytes == palette_part.nbytes + stroke_part.nbytes

    stack = UndoStack()
    stack.push(record)
    assert len(stack) == 1
    assert stack.undo(scene)
    assert np.array_equal(scene.palette, original)
    assert robot.grid.get_voxel(3, 1, 1) == 0
    assert stack.last_applied is record
    assert stack.redo(scene)
    assert tuple(scene.palette[5]) == (10, 20, 30, 255)
    assert robot.grid.get_voxel(3, 1, 1) == 5


def test_compound_record_order_matters_for_the_calls():
    calls = []

    class Part:
        def __init__(self, name):
            self.name = name
            self.nbytes = 1

        def undo(self, scene):
            calls.append(("undo", self.name))

        def redo(self, scene):
            calls.append(("redo", self.name))

    record = CompoundRecord((Part("a"), Part("b")))
    record.undo(None)
    record.redo(None)
    assert calls == [("undo", "b"), ("undo", "a"), ("redo", "a"), ("redo", "b")]
    assert record.nbytes == 2


# --- SelectionRecord ----------------------------------------------------------


def test_selection_record_factory_skips_equal_masks():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    assert selection_record(1, mask, mask.copy()) is None
    other = mask.copy()
    other[2, 2, 2] = True
    record = selection_record(1, mask, other)
    assert record is not None
    assert record.uid == 1
    assert np.array_equal(record.before, mask)
    assert np.array_equal(record.after, other)


def test_selection_record_holds_copies():
    scene = _scene_two()
    model = scene.models[0]
    before = model.selection.mask.copy()
    after = (model.grid.voxels != 0)
    record = SelectionRecord(model.uid, before, after)
    record.redo(scene)
    model.selection.clear()
    assert record.after.any()
    record.undo(scene)
    model.selection.replace(np.ones(model.grid.size, dtype=bool))
    assert not record.before.any()


def test_selection_record_undo_redo_restores_mask_and_bumps_version():
    scene = _scene_two()
    model = scene.models[0]
    before = model.selection.mask.copy()
    assert model.selection.is_empty()
    model.selection.replace(model.grid.voxels != 0)
    after = model.selection.mask.copy()
    assert after.any()
    record = selection_record(model.uid, before, after)
    assert record is not None
    # Packed bits: one bit per cell for each mask, not one byte.
    assert record.nbytes == 2 * -(-before.size // 8)
    assert record.nbytes < (before.nbytes + after.nbytes) // 4

    stack = UndoStack()
    stack.push(record)
    version = model.selection.version
    assert stack.undo(scene)
    assert model.selection.is_empty()
    assert model.selection.version != version
    assert stack.last_applied is record
    assert stack.redo(scene)
    assert np.array_equal(model.selection.mask, after)
    assert not stack.redo(scene)


def test_selection_record_keeps_masks_of_two_shapes():
    """A whole-model turn changes the grid shape; the record restores each side's mask."""
    scene = _scene_two()
    model = scene.models[0]
    before = np.zeros(model.grid.size, dtype=bool)
    before[1, 2, 3] = True
    after_shape = (model.grid.size[1], model.grid.size[0], model.grid.size[2])
    after = np.zeros(after_shape, dtype=bool)
    record = SelectionRecord(model.uid, before, after)
    assert record.before.shape == before.shape and record.after.shape == after_shape
    record.redo(scene)
    assert model.selection.size == after_shape and model.selection.is_empty()
    record.undo(scene)
    assert model.selection.size == before.shape
    assert np.array_equal(model.selection.mask, before)


def test_changes_saved_data_is_false_for_a_selection_only_step():
    mask = np.zeros((4, 4, 4), dtype=bool)
    other = mask.copy()
    other[1, 1, 1] = True
    sel = selection_record(1, mask, other)
    assert not changes_saved_data(None)
    assert not changes_saved_data(sel)
    assert not changes_saved_data(stacked_record(sel, selection_record(2, mask, other)))
    grid = VoxelGrid((4, 4, 4))
    recorder = StrokeRecorder(grid)
    recorder.set_many(np.array([[1, 1, 1]], dtype=np.int32), 3)
    model_part = ModelRecord(1, recorder.finish())
    assert changes_saved_data(model_part)
    assert changes_saved_data(stacked_record(model_part, sel))


def test_selection_record_routes_by_uid():
    scene = _scene_two()
    robot, bird = scene.models
    robot.selection.replace(robot.grid.voxels != 0)
    bird.selection.replace(bird.grid.voxels != 0)
    robot_mask = robot.selection.mask.copy()
    bird_mask = bird.selection.mask.copy()
    empty = np.zeros(robot.grid.size, dtype=bool)
    stack = UndoStack()
    stack.push(selection_record(robot.uid, robot_mask, empty))
    robot.selection.clear()
    assert stack.undo(scene)
    assert np.array_equal(robot.selection.mask, robot_mask)
    assert np.array_equal(bird.selection.mask, bird_mask)


def test_compound_voxel_and_selection_one_step():
    scene = _scene_two()
    model = scene.models[0]
    sel_before = model.selection.mask.copy()
    voxels_before = model.grid.voxels.copy()
    stroke = _stroke_in(scene, model.uid, 3, 9)
    model.selection.replace(model.grid.voxels != 0)
    sel_after = model.selection.mask.copy()
    record = stacked_record(stroke, selection_record(model.uid, sel_before, sel_after))
    assert isinstance(record, CompoundRecord)
    assert record.first(ModelRecord) is stroke
    assert isinstance(record.first(SelectionRecord), SelectionRecord)

    stack = UndoStack()
    stack.push(record)
    assert len(stack) == 1
    assert stack.undo(scene)
    assert np.array_equal(model.grid.voxels, voxels_before)
    assert model.selection.is_empty()
    assert stack.redo(scene)
    assert model.grid.get_voxel(3, 1, 1) == 9
    assert np.array_equal(model.selection.mask, sel_after)
    assert len(stack) == 1


def test_stacked_record_none_one_or_many():
    assert stacked_record(None, None) is None
    scene = _scene_two()
    stroke = _stroke_in(scene, scene.models[0].uid, 1, 4)
    assert stacked_record(None, stroke, None) is stroke
    mask = np.zeros(scene.models[0].grid.size, dtype=bool)
    other = mask.copy()
    other[0, 0, 0] = True
    sel = selection_record(scene.models[0].uid, mask, other)
    stacked = stacked_record(stroke, sel)
    assert isinstance(stacked, CompoundRecord)
    assert stacked.parts == (stroke, sel)


def test_selection_only_does_not_drop_a_later_voxel_undo():
    """A Select-tool step sits on the stack; the stroke under it still undoes."""
    scene = _scene_two()
    model = scene.models[0]
    start = model.grid.voxels.copy()
    stack = UndoStack()
    stack.push(_stroke_in(scene, model.uid, 3, 9))
    before = model.selection.mask.copy()
    model.selection.replace(model.grid.voxels != 0)
    stack.push(selection_record(model.uid, before, model.selection.mask.copy()))
    assert len(stack) == 2
    assert stack.undo(scene)
    assert model.selection.is_empty()
    assert model.grid.get_voxel(3, 1, 1) == 9
    assert stack.undo(scene)
    assert np.array_equal(model.grid.voxels, start)

