"""Bricks: the table, landing, Mirror, the lock, undo, the file and the seams byte."""

import itertools
import json

import numpy as np
import pytest

from rok.core import brick, grid_ops, link, rokfile
from rok.core.brick import Brick, BrickLocked, BrickTable
from rok.core.grid import VoxelGrid
from rok.core.scene import VoxScene
from rok.core.undo import (KIND_NAMES, BrickRecord, GridRecord, ModelRecord, StrokeRecorder, UndoStack,
                              kind_name, stacked_record)
from rok.core.vox_read import read_vox

from helpers import FIXTURES

HERE = FIXTURES / "brick"


def _house():
    data = json.loads((HERE / "house.json").read_text())
    return data, read_vox(HERE / "house.vox")


def _built(clicks, size, axes=(False, False, False)):
    grid = VoxelGrid(tuple(size))
    for x, y, z, w, d, turn, colour in clicks:
        placed, reason = brick.click(grid, x, y, z, w, d, turn, colour, axes)
        assert reason is None, (x, y, z, w, d, turn, reason)
    return grid


# --- the golden house ---------------------------------------------------------


def test_the_house_clicks_make_the_house_table_and_its_voxels():
    data, model = _house()
    grid = _built(data["clicks"], data["size"])
    assert grid.bricks == BrickTable.from_json(data["bricks"])
    assert np.array_equal(grid.voxels, model.voxels)
    assert grid.bricks.check(grid.voxels) is None
    assert np.array_equal(grid.bricks.paint(np.zeros_like(grid.voxels)), model.voxels)


def test_the_left_half_with_mirror_x_makes_the_same_house():
    mirror = json.loads((HERE / "house_mirror.json").read_text())
    data, model = _house()
    grid = _built(mirror["clicks"], mirror["size"], tuple(mirror["axes"]))
    assert grid.bricks == BrickTable.from_json(mirror["bricks"]) == BrickTable.from_json(data["bricks"])
    assert np.array_equal(grid.voxels, model.voxels)


def test_the_house_converts_to_1x1s_and_back_without_a_changed_cell():
    _data, model = _house()
    grid = VoxelGrid(model.size, model.voxels)
    grid.lock = np.ones(grid.size, dtype=bool)
    table = brick.convert_to_bricks(grid)
    assert len(table) == int((model.voxels != 0).sum())
    assert all(b.w == b.d == 1 for b in table.bricks)
    assert grid.lock is None and table.check(grid.voxels) is None
    assert np.array_equal(grid.voxels, model.voxels)
    assert brick.convert_to_voxels(grid) is table
    assert grid.bricks is None
    assert np.array_equal(grid.voxels, model.voxels)


# --- landing ------------------------------------------------------------------


def _grid_with(*bricks, size=(8, 8, 8)):
    grid = VoxelGrid(size)
    brick.put(grid, [], 1, 1, 1)  # a brick grid with an empty table
    for b in bricks:
        brick.put(grid, [(b.x, b.y, b.z)], b.w, b.d, b.colour)
    return grid


def test_a_brick_lands_on_the_floor():
    grid = VoxelGrid((8, 8, 8))
    assert brick.plan(grid, 2, 2, 0, 2, 4) == ([(2, 2, 0)], None)


def test_a_brick_lands_on_top_of_another_and_slides_up_past_a_stack():
    grid = _grid_with(Brick(0, 0, 0, 4, 2, 5), Brick(0, 0, 1, 2, 2, 5))
    # From the floor it slides up past both bricks under its footprint.
    assert brick.plan(grid, 1, 0, 0, 2, 2) == ([(1, 0, 2)], None)
    # Started from the top face of the lower brick (z = 1), it rises one more.
    assert brick.plan(grid, 2, 0, 1, 2, 2) == ([(2, 0, 1)], None)


def test_a_brick_goes_under_an_eave_and_into_a_hole():
    # An eave: a brick at z = 3 held by a post; a brick started at z = 0 beside the post stays under it.
    grid = _grid_with(Brick(0, 0, 0, 1, 1, 5), Brick(0, 0, 1, 1, 1, 5), Brick(0, 0, 2, 1, 1, 5),
                      Brick(0, 0, 3, 4, 1, 5))
    assert brick.plan(grid, 2, 0, 0, 2, 1) == ([(2, 0, 0)], None)
    # A hole one brick wide in a wall: started at the hole's height it fills the hole.
    wall = _grid_with(Brick(0, 0, 0, 6, 1, 5), Brick(0, 0, 1, 2, 1, 5), Brick(4, 0, 1, 2, 1, 5),
                      Brick(0, 0, 2, 6, 1, 5))
    assert brick.plan(wall, 2, 0, 1, 2, 1) == ([(2, 0, 1)], None)


def test_a_brick_with_nothing_under_it_is_refused():
    grid = _grid_with(Brick(0, 0, 0, 2, 2, 5))
    assert brick.plan(grid, 4, 4, 1, 2, 2)[1] == "no support"
    # One cell of support is enough.
    assert brick.plan(grid, 1, 1, 1, 2, 2) == ([(1, 1, 1)], None)


def test_a_brick_outside_the_grid_or_past_the_top_is_refused():
    grid = VoxelGrid((4, 4, 2))
    assert brick.plan(grid, 3, 0, 0, 2, 1)[1] == "outside the grid"
    assert brick.plan(grid, -1, 0, 0, 2, 1)[1] == "outside the grid"
    full = _grid_with(Brick(0, 0, 0, 2, 2, 5), Brick(0, 0, 1, 2, 2, 5), size=(4, 4, 2))
    assert brick.plan(full, 0, 0, 0, 2, 2)[1] == "outside the grid"


def test_a_brick_in_a_hidden_slice_is_refused():
    grid = _grid_with(Brick(0, 0, 0, 2, 2, 5), Brick(0, 0, 1, 2, 2, 5))
    grid.set_section(2, axis=2)  # z 2 and up are cut away
    assert brick.plan(grid, 0, 0, 0, 2, 2)[1] == "in a hidden slice"
    assert brick.plan(grid, 4, 4, 0, 2, 2) == ([(4, 4, 0)], None)


@pytest.mark.parametrize("w,d", [(1, 1), (2, 4), (3, 1), (1, 6)])
def test_a_turn_keeps_the_first_cell_under_the_mouse(w, d):
    for turn in range(4):
        ox, oy, tw, td = brick.turned(w, d, turn)
        assert (tw, td) == ((w, d) if turn % 2 == 0 else (d, w))
        assert ox <= 0 < ox + tw and oy <= 0 < oy + td  # the first cell is in the footprint
    # A full turn is the start again; R then Shift+R is no turn.
    assert brick.turned(w, d, 4) == brick.turned(w, d, 0)


def test_a_turn_is_clockwise_seen_from_above():
    # A 1 x 3 along +x, turned once clockwise (seen from +z), runs along -y.
    assert brick.turned(3, 1, 1) == (0, -2, 1, 3)


# --- Mirror -------------------------------------------------------------------


def test_mirror_x_places_two_bricks_and_a_middle_brick_once():
    grid = VoxelGrid((10, 4, 4))
    placed, reason = brick.click(grid, 0, 0, 0, 2, 1, 0, 3, (True, False, False))
    assert reason is None and sorted((b.x, b.y) for b in placed) == [(0, 0), (8, 0)]
    placed, reason = brick.click(grid, 3, 2, 0, 4, 1, 0, 3, (True, False, False))
    assert reason is None and [(b.x, b.w) for b in placed] == [(3, 4)]


def test_a_mirror_copy_that_overlaps_the_brick_refuses_the_whole_click():
    grid = VoxelGrid((10, 4, 4))
    placed, reason = brick.click(grid, 3, 0, 0, 3, 1, 0, 3, (True, False, False))
    assert placed == [] and reason == "overlaps the mirror copy"
    assert not grid.voxels.any() and not grid.bricks


def test_a_refused_mirror_copy_refuses_the_whole_click():
    grid = _grid_with(Brick(0, 0, 0, 2, 1, 5), size=(10, 4, 4))
    # The brick lands on the one at x 0..1; its copy at x 8..9 has nothing under it.
    placed, reason = brick.click(grid, 0, 0, 1, 2, 1, 0, 3, (True, False, False))
    assert placed == [] and reason == "no support"
    assert len(grid.bricks) == 1


def test_each_mirror_copy_lands_from_its_own_height():
    grid = _grid_with(Brick(8, 0, 0, 2, 1, 5), size=(10, 4, 4))
    placed, reason = brick.click(grid, 0, 0, 0, 2, 1, 0, 3, (True, False, False))
    assert reason is None and sorted((b.x, b.z) for b in placed) == [(0, 0), (8, 1)]


# --- the lock -----------------------------------------------------------------


def _locked_grid():
    grid = VoxelGrid((8, 8, 8))
    brick.put(grid, [(0, 0, 0)], 2, 2, 5)
    return grid


@pytest.mark.parametrize("write", [
    lambda g: g.set_voxel(4, 4, 4, 1),
    lambda g: g.set_voxels(np.array([[4, 4, 4]]), 1),
    lambda g: g.set_grid(np.ones(g.size, dtype=np.uint8)),
    lambda g: g.write_brush((4, 4, 4), np.ones((1, 1, 1), dtype=bool), 0, 1),
    lambda g: g.replace_voxels(np.zeros((4, 4, 4), dtype=np.uint8)),
    lambda g: StrokeRecorder(g).set_many(np.array([[4, 4, 4]]), 1),
])
def test_every_write_to_a_brick_grid_raises_outside_unlocked(write):
    grid = _locked_grid()
    before = grid.voxels.copy()
    with pytest.raises(BrickLocked) as err:
        write(grid)
    assert "built from bricks" in str(err.value) and isinstance(err.value, ValueError)
    assert np.array_equal(grid.voxels, before)
    with brick.unlocked(grid):
        write(grid)  # inside the block the write goes through
    with brick.unlocked():
        assert brick.is_open(grid)
    assert not brick.is_open(grid)


def test_unlocking_one_grid_leaves_the_others_locked():
    a, b = _locked_grid(), _locked_grid()
    with brick.unlocked(a):
        with brick.unlocked(a):
            a.set_voxel(5, 5, 5, 1)
        a.set_voxel(5, 5, 6, 1)  # still open: the outer block holds it
        with pytest.raises(BrickLocked):
            b.set_voxel(5, 5, 5, 1)
    with pytest.raises(BrickLocked):
        a.set_voxel(5, 5, 7, 1)


def test_a_voxel_grid_is_never_locked_and_undo_writes_a_brick_grid():
    grid = VoxelGrid((8, 8, 8))
    grid.set_voxel(1, 1, 1, 3)
    locked = _locked_grid()
    rec = StrokeRecorder(locked)
    with brick.unlocked(locked):
        rec.set_many(np.array([[5, 5, 0]]), 9)
    record = rec.finish()
    record.undo(locked)  # chunk writes: no unlock needed
    assert locked.voxels[5, 5, 0] == 0
    stack = UndoStack()
    whole = GridRecord(before=locked.voxels.copy(), after=np.zeros((8, 8, 8), dtype=np.uint8))
    with brick.unlocked(locked):
        locked.replace_voxels(whole.after)
    stack.push(whole)
    stack.undo(locked)  # replace_voxels inside the stack's replay
    assert locked.voxels[0, 0, 0] == 5
    stack.redo(locked)
    assert not locked.voxels.any()


# --- the table ----------------------------------------------------------------


def test_the_table_round_trips_through_json_and_refuses_a_bad_brick():
    data, _model = _house()
    table = BrickTable.from_json(data["bricks"])
    assert BrickTable.from_json(json.loads(json.dumps(table.to_json()))) == table
    with pytest.raises(ValueError):
        BrickTable.from_json([[0, 0, 0, 0, 2, 5]])
    with pytest.raises(ValueError):
        BrickTable.from_json([[0, 0, 0, 1, 2, 5, 1]])  # a brick has six values


def test_at_and_cell_map_find_the_brick_of_a_cell():
    table = BrickTable([Brick(0, 0, 0, 2, 4, 5), Brick(2, 0, 0, 1, 1, 6)])
    assert table.at((1, 3, 0)) == 0 and table.at((2, 0, 0)) == 1 and table.at((2, 1, 0)) is None
    ids = table.cell_map((4, 4, 2))
    assert (ids == 0).sum() == 8 and ids[2, 0, 0] == 1 and ids[3, 3, 1] == -1
    assert table.bricks[0].name() == "2×4" and Brick(0, 0, 0, 4, 2, 1).name() == "2×4"


def test_check_finds_an_overlap_and_a_wrong_colour():
    voxels = np.zeros((4, 4, 2), dtype=np.uint8)
    voxels[0:2, 0, 0] = 5
    assert BrickTable([Brick(0, 0, 0, 2, 1, 5)]).check(voxels) is None
    assert "overlaps" in BrickTable([Brick(0, 0, 0, 2, 1, 5), Brick(1, 0, 0, 1, 1, 5)]).check(voxels)
    assert "does not match" in BrickTable([Brick(0, 0, 0, 2, 1, 6)]).check(voxels)
    assert "outside" in BrickTable([Brick(3, 0, 0, 2, 1, 5)]).check(voxels)


def test_the_seams_byte_by_hand():
    table = BrickTable([Brick(0, 0, 0, 2, 1, 5), Brick(2, 0, 0, 1, 1, 5), Brick(0, 0, 1, 1, 1, 6)])
    voxels = table.paint(np.zeros((4, 2, 3), dtype=np.uint8))
    voxels[3, 1, 0] = 7  # a loose cell
    field = table.field(voxels)
    # The 2x1's left cell: the same brick on +x only.
    assert field[0, 0, 0] == 0b0000001
    # Its right cell: the same brick on -x; the 1x1 beside it is another brick.
    assert field[1, 0, 0] == 0b0000010
    # The 1x1, the brick on top, the loose cell, and air: no neighbour of the same brick.
    assert field[2, 0, 0] == 0
    assert field[0, 0, 1] == 0
    assert field[3, 1, 0] == 0
    assert field[3, 0, 0] == 0


def test_a_z_turn_or_a_flip_moves_the_table_with_the_voxels():
    data, model = _house()
    table = BrickTable.from_json(data["bricks"])
    for R in [link.turn_matrix(2, k) for k in range(4)] + [link.flip_matrix(0), link.flip_matrix(1)]:
        voxels, _s = grid_ops.apply_rotation(model.voxels, R)
        moved = table.transformed(R, model.size)
        assert moved.check(voxels) is None
        assert np.array_equal(moved.paint(np.zeros_like(voxels)), voxels)
    with pytest.raises(ValueError):
        table.transformed(link.turn_matrix(0, 1), model.size)
    assert table.shifted((1, 1, 0)).bricks[0].x == table.bricks[0].x - 1


def test_scene_turns_with_grid_ops_rotate_model_match_the_table():
    data, model = _house()
    table = BrickTable.from_json(data["bricks"])
    for k in range(4):
        voxels, _pos = grid_ops.rotate_model(model.voxels, (0, 0, 0), 2, k, (10, 10, 8))
        assert table.transformed(link.turn_matrix(2, k), model.size).check(voxels) is None


def test_a_linked_copy_gets_the_table_turned_by_its_orient():
    data, model = _house()
    for k in range(4):
        orient = link.as_orient(link.turn_matrix(2, k))
        scene = VoxScene.single(VoxelGrid(model.size, model.voxels), (0, 0, 0))
        turned = link.from_source(model.voxels, orient)
        scene.add(VoxelGrid(turned.shape, turned), "copy", (20, 0, 0))
        a, b = scene.models
        a.link = b.link = 1
        b.orient = orient
        a.grid.bricks = BrickTable.from_json(data["bricks"])
        # A brick placed in the original shows in the copy, with the copy's table.
        with brick.unlocked(a.grid):
            brick.put(a.grid, [(0, 0, 0)], 1, 1, 9)
        assert link.sync(scene, a.uid) == [b.uid]
        assert b.grid.bricks.check(b.grid.voxels) is None
        assert len(b.grid.bricks) == len(a.grid.bricks)
        # Convert to voxels in the copy takes the table off the original too.
        b.grid.bricks = None
        link.sync(scene, b.uid)
        assert a.grid.bricks is None


# --- undo, the file -----------------------------------------------------------


def test_a_click_is_one_undo_step_with_its_table():
    scene = VoxScene.single(VoxelGrid((8, 8, 8)), (0, 0, 0))
    model = scene.models[0]
    stack = UndoStack()
    before = model.grid.bricks
    rec = StrokeRecorder(model.grid)
    placed, reason = brick.click(model.grid, 1, 1, 0, 2, 4, 0, 5)
    assert reason is None
    # (The click wrote through its own set_voxels; a recorder of the same
    # chunks, made before, gives the record the app pushes.)
    rec._before = {(0, 0, 0): np.zeros((8, 8, 8), dtype=np.uint8)}
    record = stacked_record(ModelRecord(model.uid, rec.finish()), BrickRecord(model.uid, before, model.grid.bricks.copy()))
    stack.push(record)
    assert kind_name(record) == "Voxels" and KIND_NAMES["BrickRecord"] == "Bricks"
    assert kind_name(BrickRecord(1)) == "Bricks"
    stack.undo(scene)
    assert model.grid.bricks is None and not model.grid.voxels.any()
    stack.redo(scene)
    assert len(model.grid.bricks) == 1 and (model.grid.voxels == 5).sum() == 8


def test_the_rok_file_keeps_the_tables():
    data, model = _house()
    scene = VoxScene.single(VoxelGrid(model.size, model.voxels), (0, 0, 0))
    scene.add(VoxelGrid((4, 4, 4)), "plain", (20, 0, 0))
    scene.models[0].grid.bricks = BrickTable.from_json(data["bricks"])
    assert "bricks" in rokfile.needs_rok(scene)
    back, _doc = rokfile.loads(rokfile.dumps(scene, {}))
    assert back.models[0].grid.bricks == scene.models[0].grid.bricks
    assert back.models[1].grid.bricks is None
    plain = VoxScene.single(VoxelGrid((4, 4, 4)), (0, 0, 0))
    assert "bricks" not in rokfile.needs_rok(plain)


def test_a_table_that_does_not_fit_the_voxels_is_not_a_rok_file():
    data, model = _house()
    scene = VoxScene.single(VoxelGrid(model.size, model.voxels), (0, 0, 0))
    text = rokfile.pack(scene, {})
    meta = json.loads(text)
    meta["bricks"] = [[0, [[0, 0, 7, 2, 2, 5, 1]]]]  # air at the top of the grid
    with pytest.raises(rokfile.RokFileError):
        rokfile.unpack(json.dumps(meta))


def test_changed_cells_lists_the_bricks_that_differ():
    a = BrickTable([Brick(0, 0, 0, 2, 1, 5), Brick(4, 0, 0, 1, 1, 5)])
    b = BrickTable([Brick(0, 0, 0, 2, 1, 6), Brick(4, 0, 0, 1, 1, 5)])
    cells = brick.changed_cells(a, b)
    assert sorted(map(tuple, cells.tolist())) == [(0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0)]
    assert len(brick.changed_cells(None, None)) == 0
    assert len(brick.changed_cells(None, a)) == 3


def test_mirror_boxes_cover_every_axis_combination():
    boxes = brick.mirror_boxes(0, 0, 0, 1, 1, (4, 4, 4), (True, True, True))
    assert len(boxes) == 8 and set(boxes) == set(itertools.product((0, 3), (0, 3), (0, 3)))
