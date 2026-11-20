"""The mask protects voxels from every command ("the mask
does not reliably protect voxels from deletion"). The grid itself refuses a
write to a masked cell (VoxelGrid.lock); commands that move the whole grid
take the mask along."""

from __future__ import annotations

import numpy as np
import pytest
from test_app_tools import app, empty, put  # noqa: F401 (fixtures)

from rok.app import opkit, select_ops
from rok.app.model_ops import MASK_LOSS
from rok.app.tools import MASK_NOTE
from rok.app.object import document
from rok.core import gizmo
from rok.core.grid import VoxelGrid
from rok.core.undo import StrokeRecorder

# --- the grid -------------------------------------------------------------------------------


def test_a_locked_cell_takes_no_write_but_undo_and_remaps_reach_it():
    grid = VoxelGrid((4, 4, 4))
    grid.set_voxels(np.array([(1, 1, 1), (2, 1, 1)]), 5)
    grid.lock = np.zeros((4, 4, 4), bool)
    grid.lock[1, 1, 1] = True
    assert not grid.set_voxel(1, 1, 1, 0)
    assert grid.set_voxels(np.array([(1, 1, 1), (2, 1, 1)]), 0) == 1
    assert grid.voxels[1, 1, 1] == 5 and grid.voxels[2, 1, 1] == 0
    assert grid.locked_writes == 2
    assert grid.set_voxels(np.array([(1, 1, 1)]), 7, ignore_lock=True) == 1  # a palette remap
    rec = StrokeRecorder(grid)
    rec.set_many(np.array([(3, 3, 3)]), 9)
    record = rec.finish()
    grid.voxels[1, 1, 1] = 5
    record.undo(grid)  # undo writes chunks: the lock does not stop it
    assert grid.voxels[3, 3, 3] == 0
    grid.lock = np.ones((2, 2, 2), bool)  # a lock of another shape is not used
    assert grid.set_voxel(1, 1, 1, 0)


# --- the commands ---------------------------------------------------------------------------

BLOCK = [(x, y, z) for x in range(2, 9) for y in range(2, 9) for z in range(0, 4)]
MASKED = [(x, y, z) for x in range(2, 5) for y in range(2, 9) for z in range(0, 4)]


def _masked_block() -> np.ndarray:
    """A 7 x 7 x 4 block, colour 5, with its left part masked (colour 9).
    Returns the voxels before."""
    put(BLOCK, colour=5)
    put(MASKED, colour=9)
    opkit.call("rok.deselect")
    document.selection.add_cells(np.array(MASKED))
    opkit.call("rok.mask_selection")
    assert select_ops.mask_of() is not None and int(select_ops.mask_of().sum()) == len(MASKED)
    return document.grid.voxels.copy()


def _select_all() -> None:
    opkit.call("rok.deselect")
    document.selection.add_cells(np.argwhere(document.grid.voxels != 0))


def _kept(before: np.ndarray) -> bool:
    mask = select_ops.mask_of()
    return mask is not None and np.array_equal(document.grid.voxels[mask], before[mask])


@pytest.mark.parametrize("run", [
    lambda: (_select_all(), opkit.call("rok.selection_delete")),
    lambda: (_select_all(), opkit.call("rok.cut")),
    lambda: (_select_all(), opkit.call("rok.paint_selection")),
    lambda: opkit.call("rok.hollow", "EXEC", wall=1),
    lambda: opkit.call("rok.shrink", "EXEC", count=2, dialog=False),
    lambda: opkit.call("rok.smooth"),
    lambda: opkit.call("rok.bevel"),
    lambda: opkit.call("rok.clear_all"),
    lambda: opkit.call("rok.inverse"),
    lambda: opkit.call("rok.fill_all"),
    lambda: opkit.call("rok.fill_volume"),
], ids=["delete", "cut", "paint_selection", "hollow", "shrink", "smooth", "bevel", "clear_all", "inverse", "fill_all",
        "fill_volume"])
def test_no_command_changes_a_masked_voxel(empty, run):  # noqa: F811
    before = _masked_block()
    document.colour = 3
    run()
    assert _kept(before)
    assert not np.array_equal(document.grid.voxels, before)  # the rest did change


def test_split_leaves_masked_voxels_in_the_model(empty):  # noqa: F811
    before = _masked_block()
    _select_all()
    opkit.call("rok.model_split")
    source = document.scene.models[0]
    assert np.array_equal(source.grid.voxels[select_ops.mask_of(source.uid)], before[select_ops.mask_of(source.uid)])
    part = document.scene.models[-1]
    assert not np.any(part.grid.voxels == 9)  # the masked voxels did not go with it


def test_a_moved_selection_leaves_its_masked_voxels_behind(empty):  # noqa: F811
    before = _masked_block()
    _select_all()
    opkit.call("rok.transform", "EXEC", target=gizmo.SELECTION, kind=gizmo.MOVE, offset=(0, 0, 3))
    assert _kept(before)
    assert np.count_nonzero(document.grid.voxels == 9) == len(MASKED)  # not copied along either


def test_drop_to_floor_lets_the_rest_fall_and_the_masked_voxels_stay(empty):  # noqa: F811
    put([(6, 6, 6)], colour=5)
    before = _masked_block()
    opkit.call("rok.drop_to_floor", "EXEC")
    assert _kept(before)


@pytest.mark.parametrize("run", [
    lambda: opkit.call("rok.flip", "EXEC", axis="X"),
    lambda: opkit.call("rok.rotate", "EXEC", axis="Z"),
    lambda: opkit.call("rok.transform", "EXEC", target=gizmo.MODEL, kind=gizmo.ROTATE, axis=2, turns=1),
    lambda: opkit.call("rok.scale", "EXEC", mode="UP"),
    lambda: opkit.call("rok.fit_grid"),
    lambda: opkit.call("rok.diagonal"),
], ids=["flip", "rotate", "gizmo_turn", "scale_up", "fit_grid", "diagonal"])
def test_the_mask_moves_with_the_model_and_undo_brings_both_back(empty, run):  # noqa: F811
    before = _masked_block()
    mask_before = select_ops.mask_of().copy()
    run()
    mask = select_ops.mask_of()
    assert mask is not None and mask.shape == document.grid.voxels.shape
    assert np.all(document.grid.voxels[mask] == 9)  # it still covers the masked voxels ...
    assert np.count_nonzero(document.grid.voxels == 9) == np.count_nonzero(mask)  # ... and only them
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)
    assert np.array_equal(select_ops.mask_of(), mask_before)


def test_a_resize_that_would_cut_masked_voxels_is_refused(empty):  # noqa: F811
    before = _masked_block()
    assert opkit.call("rok.resize", "EXEC", size=(3, 16, 8)) == {"CANCELLED"}
    assert np.array_equal(document.grid.voxels, before)
    assert opkit.call("rok.resize", "EXEC", size=(12, 12, 8)) == {"FINISHED"}  # cuts no masked voxel
    assert np.count_nonzero(document.grid.voxels[select_ops.mask_of()] == 9) == len(MASKED)


def test_masking_and_clearing_are_undo_steps(empty):  # noqa: F811
    _masked_block()
    opkit.call("rok.mask_clear")
    assert select_ops.mask_of() is None
    opkit.call("rok.undo")
    assert select_ops.mask_of() is not None
    opkit.call("rok.undo")  # the masking itself
    assert select_ops.mask_of() is None
    assert document.selection.count() == len(MASKED)  # the selection it was made from is back


def test_a_merge_keeps_masked_voxels_on_top(empty):  # noqa: F811
    _masked_block()
    first = document.model
    opkit.call("rok.model_add", "EXEC")
    other = document.model
    other.position = tuple(first.position)
    document.grid.set_voxels(np.array(BLOCK), 7)  # the later model covers the whole block
    document.set_active(first.uid)
    assert opkit.call("rok.model_merge_active", "EXEC") == {"FINISHED"}
    mask = select_ops.mask_of()
    assert int(mask.sum()) == len(MASKED) and np.all(document.grid.voxels[mask] == 9)
    assert np.count_nonzero(document.grid.voxels == 7) == len(BLOCK) - len(MASKED)
    opkit.call("rok.undo")
    assert int(select_ops.mask_of(first.uid).sum()) == len(MASKED)


# --- whole-grid commands that can resample a masked cell -----------------


def _fresh() -> None:
    opkit.call("rok.new", "EXEC", size=(8, 8, 8), discard=True)


def _mask_cells(cells) -> None:
    document.selection.clear()
    document.selection.add_cells(np.asarray(cells, dtype=np.int64).reshape(-1, 3))
    assert opkit.call("rok.mask_selection") == {"FINISHED"}


def _mask_colours() -> list[int]:
    mask = select_ops.mask_of()
    assert mask is not None and mask.shape == document.grid.voxels.shape
    return sorted(int(c) for c in document.grid.voxels[mask])


def test_lathe_keeps_a_masked_cell(empty):  # noqa: F811
    _fresh()
    put([(2, 2, 4)], 9)
    put([(6, 2, 4)], 5)  # same ring, and later in the grid, so it would win
    _mask_cells([(2, 2, 4)])
    before = document.grid.voxels.copy()
    mask_before = select_ops.mask_of().copy()
    assert opkit.call("rok.lathe", "EXEC", axis="Z") == {"FINISHED"}
    assert document.grid.voxels[2, 2, 4] == 9
    assert select_ops.mask_of()[2, 2, 4]
    assert int((document.grid.voxels == 5).sum()) > 1  # the other voxel still spun
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)
    assert np.array_equal(select_ops.mask_of(), mask_before)


def test_lathe_of_a_round_model_leaves_a_masked_cell_alone(empty):  # noqa: F811
    _fresh()
    put([(6, 2, 4)], 5)
    assert opkit.call("rok.lathe", "EXEC", axis="Z") == {"FINISHED"}
    assert document.grid.voxels[2, 2, 4] == 5
    put([(2, 2, 4)], 9)
    _mask_cells([(2, 2, 4)])
    before = document.grid.voxels.copy()
    count = len(document.undo._undo)
    assert opkit.call("rok.lathe", "EXEC", axis="Z") == {"CANCELLED"}
    assert empty.message == "Already round"
    assert np.array_equal(document.grid.voxels, before)
    assert len(document.undo._undo) == count


def test_scale_down_refuses_when_a_neighbour_would_repaint_the_mask(empty):  # noqa: F811
    _fresh()
    put([(2, 2, 4)], 9)
    put([(3, 2, 4)], 5)  # the sample in the middle of that block
    _mask_cells([(2, 2, 4)])
    before = document.grid.voxels.copy()
    count = len(document.undo._undo)
    assert opkit.call("rok.scale", "EXEC", mode="DOWN") == {"CANCELLED"}
    assert empty.message == MASK_LOSS
    assert np.array_equal(document.grid.voxels, before)
    assert select_ops.mask_of()[2, 2, 4]
    assert len(document.undo._undo) == count


def test_scale_down_keeps_one_masked_voxel(empty):  # noqa: F811
    _fresh()
    put([(2, 2, 4)], 9)
    _mask_cells([(2, 2, 4)])
    before = document.grid.voxels.copy()
    mask_before = select_ops.mask_of().copy()
    assert opkit.call("rok.scale", "EXEC", mode="DOWN") == {"FINISHED"}
    assert _mask_colours() == [9]
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)
    assert np.array_equal(select_ops.mask_of(), mask_before)


def test_scale_down_refuses_when_masked_cells_would_fold_into_one(empty):  # noqa: F811
    _fresh()
    block = [(x, y, z) for x in range(2, 4) for y in range(2, 4) for z in range(4, 6)]
    put(block, 9)
    _mask_cells(block)
    before = document.grid.voxels.copy()
    assert opkit.call("rok.scale", "EXEC", mode="DOWN") == {"CANCELLED"}
    assert empty.message == MASK_LOSS
    assert np.array_equal(document.grid.voxels, before)
    assert int(select_ops.mask_of().sum()) == 8


def test_twist_keeps_a_masked_voxel_that_stays_in_the_grid(empty):  # noqa: F811
    _fresh()
    put([(2, 2, 4)], 9)
    _mask_cells([(2, 2, 4)])
    before = document.grid.voxels.copy()
    mask_before = select_ops.mask_of().copy()
    assert opkit.call("rok.twist", "EXEC", angle=90, axis="Z") == {"FINISHED"}
    assert _mask_colours() == [9]
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)
    assert np.array_equal(select_ops.mask_of(), mask_before)


def test_twist_refuses_when_a_masked_voxel_would_leave_the_grid(empty):  # noqa: F811
    _fresh()
    put([(0, 0, 7)], 9)
    _mask_cells([(0, 0, 7)])
    before = document.grid.voxels.copy()
    count = len(document.undo._undo)
    assert opkit.call("rok.twist", "EXEC", angle=45, axis="Z") == {"CANCELLED"}
    assert empty.message == MASK_LOSS
    assert np.array_equal(document.grid.voxels, before)
    assert select_ops.mask_of()[0, 0, 7]
    assert len(document.undo._undo) == count


def test_twist_refuses_when_one_masked_colour_replaces_another(empty):  # noqa: F811
    _fresh()
    put([(0, 0, 7)], 9)
    put([(4, 4, 7)], 4)
    _mask_cells([(0, 0, 7), (4, 4, 7)])
    before = document.grid.voxels.copy()
    assert opkit.call("rok.twist", "EXEC", angle=45, axis="Z") == {"CANCELLED"}
    assert empty.message == MASK_LOSS
    assert np.array_equal(document.grid.voxels, before)
    assert sorted(_mask_colours()) == [4, 9]


def test_a_quarter_turn_keeps_a_masked_voxel(empty):  # noqa: F811
    _fresh()
    put([(2, 2, 4)], 9)
    _mask_cells([(2, 2, 4)])
    before = document.grid.voxels.copy()
    mask_before = select_ops.mask_of().copy()
    assert opkit.call("rok.rotate_degrees", "EXEC", x=0, y=0, z=90) == {"FINISHED"}
    assert _mask_colours() == [9]
    assert int((document.grid.voxels == 9).sum()) == 1
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)
    assert np.array_equal(select_ops.mask_of(), mask_before)


def test_a_free_turn_refuses_when_the_masked_voxel_would_be_lost(empty):  # noqa: F811
    _fresh()
    put([(2, 2, 4)], 9)
    _mask_cells([(2, 2, 4)])
    before = document.grid.voxels.copy()
    count = len(document.undo._undo)
    assert opkit.call("rok.rotate_degrees", "EXEC", x=0, y=0, z=45) == {"CANCELLED"}
    assert empty.message == MASK_LOSS
    assert document.grid.size == (8, 8, 8)
    assert np.array_equal(document.grid.voxels, before)
    assert select_ops.mask_of()[2, 2, 4]
    assert len(document.undo._undo) == count


def test_smooth_builds_nothing_against_a_masked_model(empty):  # noqa: F811
    opkit.call("rok.new", "EXEC", size=(16, 16, 16), discard=True)
    grid = document.grid
    grid.voxels[3:10, 3:10, 3:10] = 5
    grid.voxels[6, 6, 9] = 0  # a dent in the top face
    _mask_cells(np.argwhere(grid.voxels != 0))
    before = grid.voxels.copy()
    assert opkit.call("rok.smooth", "EXEC") == {"CANCELLED"}
    assert empty.message == MASK_NOTE
    assert np.array_equal(grid.voxels, before)
    assert grid.voxels[6, 6, 9] == 0


def test_smooth_rounds_the_rest_and_keeps_a_masked_corner(empty):  # noqa: F811
    opkit.call("rok.new", "EXEC", size=(16, 16, 16), discard=True)
    grid = document.grid
    grid.voxels[2:9, 2:9, 2:9] = 5
    _mask_cells([(2, 2, 2)])
    assert opkit.call("rok.smooth", "EXEC") == {"FINISHED"}
    assert grid.voxels[2, 2, 2] == 5  # the masked corner stays
    assert grid.voxels[8, 8, 8] == 0  # the far corner goes
    assert grid.voxels[1, 2, 2] == 0 and grid.voxels[2, 1, 2] == 0 and grid.voxels[2, 2, 1] == 0
    assert empty.message == MASK_NOTE


def test_a_free_turn_of_a_selection_leaves_the_masked_voxel(empty):  # noqa: F811
    _fresh()
    put([(2, 2, 4)], 9)
    put([(6, 2, 4)], 5)
    _mask_cells([(2, 2, 4)])
    document.selection.add_cells(np.array([[2, 2, 4], [6, 2, 4]]))
    assert opkit.call("rok.rotate_degrees", "EXEC", x=0, y=0, z=45) == {"FINISHED"}
    assert document.grid.voxels[2, 2, 4] == 9
    assert select_ops.mask_of()[2, 2, 4]
