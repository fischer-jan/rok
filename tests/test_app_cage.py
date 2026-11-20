"""The lattice cage in the app: the Model menu command, the chip row, the drag
through the stroke operator, and when the cage resets or hides (pattern:
test_app_tools; every test drives the app as a user would)."""

from __future__ import annotations

import numpy as np
import pytest
from test_app_tools import H, W, app, click, drag, empty, key, put  # noqa: F401  (fixtures)

from rok.app import cage_ops, opkit, overlay
from rok.app.object import document
from rok.app.state import state
from rok.core import cage_chips, lattice
from rok.core.chrome import COMMAND_OF_MODEL_ID, COMMANDS, DEFAULT_MODEL_ITEMS

BLOCK = [(x, y, z) for x in range(4, 10) for y in range(4, 10) for z in range(0, 4)]


def block_model(app_):
    put(BLOCK, colour=6)
    app_.redraw()


def select_all_of_block():
    document.selection.replace(document.grid.voxels != 0)


def show_cage(app_, count=3):
    select_all_of_block()
    opkit.call("rok.cage", "EXEC")
    app_.redraw()
    return state.cage


def host(region_xy) -> dict:
    """A host pixel from region pixels (origin bottom left)."""
    return {"x": float(region_xy[0]), "y": H - float(region_xy[1])}


def point_px(index: str) -> dict:
    _ptr, ids, pts, _depth = state.cage.layout
    return host(pts[ids.index(index)])


def chip_px(name: str) -> dict:
    x, y, w, h = dict(state.cage.chips[1].chips)[name]
    return host((x + w / 2, y + h / 2))


# --- the command ------------------------------------------------------------------------------


def test_cage_is_in_the_model_menu_by_default_and_in_cmd_k():
    assert "cage" in DEFAULT_MODEL_ITEMS
    assert COMMAND_OF_MODEL_ID["MODEL_CAGE"] == "cage"
    cmd = [c for c in COMMANDS if c.id == "cage"]
    assert len(cmd) == 1 and cmd[0].name == "Cage" and "lattice" in cmd[0].keywords


def test_the_command_puts_the_cage_on_the_selection(empty):
    block_model(empty)
    document.selection.replace(np.isin(np.arange(16)[:, None, None] * np.ones((16, 16, 8)), np.arange(4, 7)) & (document.grid.voxels != 0))
    opkit.call("rok.cage", "EXEC")
    cage = state.cage
    assert cage is not None and not cage.whole and cage.counts == (3, 3, 3)
    assert cage.lattice.lo == (4, 4, 0) and cage.lattice.size == (3, 6, 4)
    assert cage.lattice.points.shape == (3, 3, 3, 3)
    assert np.array_equal(cage.rest_voxels, document.grid.visible)


def test_with_nothing_selected_the_cage_sits_on_the_models_filled_bounds(empty):
    block_model(empty)
    document.selection.clear()
    opkit.call("rok.cage", "EXEC")
    cage = state.cage
    assert cage.whole and cage.lattice.lo == (4, 4, 0) and cage.lattice.size == (6, 6, 4)
    assert int(cage.rest_mask.sum()) == len(BLOCK)


def test_the_command_does_nothing_on_an_empty_model(empty):
    assert not opkit.ops.rok.cage.poll()


def test_running_it_again_while_it_shows_starts_it_over(empty):
    block_model(empty)
    cage = show_cage(empty)
    cage.lattice.points[2, 2, 2] += 3.0  # as if dragged
    opkit.call("rok.cage", "EXEC")
    assert np.allclose(state.cage.lattice.points, lattice.rest_lattice(((4, 4, 0), (9, 9, 3)), 3).points)


# --- the chips ----------------------------------------------------------------------------------


def set_points(app_, x, y, z):
    """Click the Points chip, type the three numbers into its dialog and press Apply."""
    click(app_, chip_px("CAGE_POINTS"))
    assert app_.dialog is not None and app_.dialog.op.command_id == "rok.cage_points"
    op = app_.dialog.op
    op.points_x, op.points_y, op.points_z = x, y, z
    app_.dialog._finish(True)
    app_.redraw()


def test_the_chip_row_draws_above_the_cage_and_the_points_dialog_gives_64_points(empty):
    block_model(empty)
    show_cage(empty)
    assert state.cage.chips is not None and len(state.cage.chips[1].chips) == 7
    click(empty, chip_px("CAGE_POINTS"))
    op = empty.dialog.op
    assert (op.points_x, op.points_y, op.points_z) == (3, 3, 3)  # it opens on the cage's counts
    empty.redraw()  # the dialog lays out
    empty.dialog._finish(False)
    assert state.cage.counts == (3, 3, 3)  # Cancel changes nothing
    set_points(empty, 4, 4, 4)
    assert state.cage.counts == (4, 4, 4) and state.cage.lattice.points.shape == (4, 4, 4, 3)
    assert state.cage.lattice.points.reshape(-1, 3).shape[0] == 64
    assert document.selection.mask.sum() == len(BLOCK)  # a press on a chip is not a stroke


def test_the_points_dialog_gives_a_4_by_3_by_2_cage_and_keeps_inside_2_to_6(empty):
    block_model(empty)
    show_cage(empty)
    set_points(empty, 4, 3, 3)
    assert state.cage.counts == (4, 3, 3) and state.cage.lattice.points.shape == (4, 3, 3, 3)
    assert len(lattice.point_ids(state.cage.counts)) == 36
    set_points(empty, 4, 3, 2)
    assert state.cage.counts == (4, 3, 2)
    set_points(empty, 9, 1, 3)  # the number is clamped to 2..6
    assert state.cage.counts == (6, 2, 3)
    assert document.selection.mask.sum() == len(BLOCK)


def test_the_soft_chip_switches_to_the_bernstein_basis_and_back_and_keeps_the_counts(empty):
    block_model(empty)
    show_cage(empty)
    assert not state.cage.lattice.bernstein
    set_points(empty, 5, 5, 5)
    base = state.cage.lattice.points.copy()
    click(empty, chip_px("CAGE_SOFT"))
    lat = state.cage.lattice
    assert lat.bernstein and lat.counts == (5, 5, 5) and not np.allclose(lat.points, base)  # even places, not Greville
    assert np.allclose(lat.points[:, 0, 0, 0], np.linspace(4, 10, 5))  # x of the points: the box 4..10 spread evenly
    before = document.grid.voxels.copy()
    assert (before != 0).sum() == len(BLOCK)  # a chip is not an edit
    set_points(empty, 4, 4, 4)
    assert state.cage.lattice.bernstein and state.cage.counts == (4, 4, 4)  # a count keeps the basis
    empty.redraw()
    start = point_px("CAGE_3_3_3")  # a corner of the 4-point cage
    drag(empty, [start, {"x": start["x"] + 30.0, "y": start["y"]}, {"x": start["x"] + 60.0, "y": start["y"]}])
    assert state.cage.lattice.bernstein and not np.array_equal(document.grid.voxels, before)  # it bends
    click(empty, chip_px("CAGE_RESET"))
    assert not state.cage.lattice.bernstein and state.cage.counts == (3, 3, 3)  # Reset: every chip back to its default
    click(empty, chip_px("CAGE_SOFT"))
    assert state.cage.lattice.bernstein


def test_reset_puts_the_points_back_and_done_hides_the_cage(empty):
    block_model(empty)
    cage = show_cage(empty)
    cage.lattice.points[1, 1, 1] += 2.0
    empty.redraw()
    click(empty, chip_px("CAGE_RESET"))
    assert np.allclose(state.cage.lattice.points, lattice.rest_lattice(((4, 4, 0), (9, 9, 3)), 3).points)
    click(empty, chip_px("CAGE_DONE"))
    assert state.cage is None


def test_return_and_escape_hide_the_cage_and_the_next_escape_deselects(empty):
    block_model(empty)
    show_cage(empty)
    key(empty, "Enter")
    assert state.cage is None and not document.selection.is_empty()
    show_cage(empty)
    key(empty, "Escape")
    assert state.cage is None and not document.selection.is_empty()  # the cage went first
    key(empty, "Escape")
    assert document.selection.is_empty()


# --- the drag ------------------------------------------------------------------------------------


def corner_drag(app_, dx=60.0, dy=0.0, release=True):
    start = point_px("CAGE_2_2_2")
    drag(app_, [start, {"x": start["x"] + dx / 2, "y": start["y"] + dy / 2},
                {"x": start["x"] + dx, "y": start["y"] + dy}], release=release)


def test_a_drag_on_a_point_changes_the_grid_and_is_one_undo_step(empty):
    block_model(empty)
    show_cage(empty)
    before = document.grid.voxels.copy()
    undo_depth = len(document.undo._undo)
    corner_drag(empty)
    assert len(document.undo._undo) == undo_depth + 1
    after = document.grid.voxels.copy()
    assert not np.array_equal(before, after)
    assert set(np.unique(after)) - {0} <= {6}  # no invented colours
    assert not np.allclose(state.cage.lattice.points[2, 2, 2], lattice.rest_lattice(((4, 4, 0), (9, 9, 3)), 3).points[2, 2, 2])
    assert state.cage is not None and not state.stroke_active
    key(empty, "z", ctrl=True)  # one step takes the whole drag back
    assert np.array_equal(document.grid.voxels, before)
    assert len(document.undo._undo) == undo_depth  # the drag was exactly one step


def test_a_drag_with_mirror_x_moves_the_mirrored_point_and_keeps_the_model_symmetric(empty):
    from rok.app.opkit import CONTEXT

    put([(x, y, z) for x in range(5, 11) for y in range(4, 10) for z in range(0, 4)], colour=6)  # centred in x of 16
    empty.redraw()
    show_cage(empty)
    CONTEXT.prefs.rok_mirror = (True, False, False)
    rest = state.cage.lattice.points.copy()
    corner_drag(empty, dx=40.0, dy=0.0)
    pts = state.cage.lattice.points
    d_right, d_left = pts[2, 2, 2] - rest[2, 2, 2], pts[0, 2, 2] - rest[0, 2, 2]
    assert np.abs(d_right).max() > 0 and np.allclose(d_left, d_right * (-1, 1, 1))
    v = document.grid.visible
    assert np.array_equal(v, v[::-1])  # symmetric in x about the grid's middle plane
    CONTEXT.prefs.rok_mirror = (False, False, False)


def test_a_cage_on_one_wing_with_mirror_x_bends_the_other_wing_too(empty):
    from rok.app.opkit import CONTEXT

    wing = [(x, y, z) for x in range(1, 6) for y in range(4, 10) for z in range(0, 4)]
    put(wing, colour=6)
    put([(15 - x, y, z) for x, y, z in wing], colour=7)  # its mirror image in a 16 grid
    empty.redraw()
    document.selection.replace(document.grid.voxels == 6)
    CONTEXT.prefs.rok_mirror = (True, False, False)
    opkit.call("rok.cage", "EXEC")
    empty.redraw()  # draws the mirrored cage's lines too
    before = document.grid.visible.copy()
    try:
        start = point_px("CAGE_0_2_2")
        drag(empty, [start, {"x": start["x"] + 15, "y": start["y"]}, {"x": start["x"] + 30, "y": start["y"]}])
        v = document.grid.visible
        assert not np.array_equal(v[8:], before[8:])  # the other wing bent
        assert np.array_equal(v != 0, (v != 0)[::-1])  # the same shape on both sides
        assert set(np.unique(v[8:])) <= {0, 7}  # each wing keeps its own colour
        assert not document.selection.mask[8:].any()  # the selection stays on the dragged wing
    finally:
        CONTEXT.prefs.rok_mirror = (False, False, False)


def test_undo_resets_the_cage_to_the_box_on_the_restored_voxels(empty):
    block_model(empty)
    show_cage(empty)
    corner_drag(empty)
    key(empty, "z", ctrl=True)
    cage = state.cage
    assert cage is not None
    assert np.allclose(cage.lattice.points, lattice.rest_lattice(((4, 4, 0), (9, 9, 3)), 3).points)
    assert np.array_equal(cage.rest_voxels, document.grid.visible)


def test_a_second_drag_bends_the_rest_not_the_bend(empty):
    block_model(empty)
    show_cage(empty)
    corner_drag(empty)
    empty.redraw()
    start = point_px("CAGE_0_0_0")
    drag(empty, [start, {"x": start["x"] - 20.0, "y": start["y"]}, {"x": start["x"] - 40.0, "y": start["y"]}])
    cage = state.cage
    want, _ = lattice.live_deform_grid(cage.rest_voxels, cage.rest_mask, cage.lattice)  # what a drag runs
    assert np.array_equal(document.grid.visible, want)


def test_escape_during_a_drag_puts_the_voxels_and_points_back(empty):
    block_model(empty)
    show_cage(empty)
    before = document.grid.voxels.copy()
    rest = state.cage.lattice.points.copy()
    corner_drag(empty, release=False)
    assert not np.array_equal(document.grid.voxels, before)
    key(empty, "Escape")
    assert np.array_equal(document.grid.voxels, before)
    assert state.cage is not None and np.allclose(state.cage.lattice.points, rest) and state.cage.live is None


def test_a_press_beside_the_cage_goes_to_the_tool_and_a_paint_stroke_resets_the_rest(empty):
    block_model(empty)
    show_cage(empty)
    cage = state.cage
    cage.lattice.points[1, 1, 1] += 1.0  # the points have moved
    opkit.call("rok.set_tool", tool="PAINT")
    document.colour = 9
    from test_app_tools import on_top

    click(empty, on_top(empty, (5, 5, 3)))
    assert document.grid.voxels[5, 5, 3] == 9  # the tool painted
    assert state.cage is not None
    assert np.array_equal(state.cage.rest_voxels, document.grid.visible)  # the rest is the voxels now
    assert np.allclose(state.cage.lattice.points[1, 1, 1], lattice.rest_lattice(((4, 4, 0), (9, 9, 3)), 3).points[1, 1, 1])


def test_a_new_selection_resets_the_cage(empty):
    block_model(empty)
    show_cage(empty)
    cage = state.cage
    document.selection.replace(np.isin(np.indices(document.grid.size)[0], (4, 5)) & (document.grid.voxels != 0))
    cage_ops.refresh()
    assert state.cage is not None and state.cage is not cage
    assert state.cage.lattice.size == (2, 6, 4)


def test_the_cage_hides_when_the_mode_leaves_model(empty):
    from rok.app.level_ops import set_level

    block_model(empty)
    show_cage(empty)
    set_level("SCENE")
    assert state.cage is None


# --- the hit test ----------------------------------------------------------------------------------


def test_the_hit_test_finds_the_point_nearest_the_eye_when_two_overlap(empty):
    block_model(empty)
    show_cage(empty)
    from rok.app.opkit import CONTEXT

    region = CONTEXT.region
    ptr, ids, pts, depth = state.cage.layout
    pts = pts.copy()
    depth = depth.copy()
    pts[1] = pts[0]  # two points on the same pixel
    depth[0], depth[1] = 50.0, 10.0
    state.cage.layout = (ptr, ids, pts, depth)
    assert overlay.cage_hit(region, *pts[0]) == ids[1]
    depth[0], depth[1] = 5.0, 10.0
    assert overlay.cage_hit(region, *pts[0]) == ids[0]
    assert overlay.cage_hit(region, pts[0][0] + 80, pts[0][1] + 80) is None


def test_hovering_a_point_names_it_and_a_chip_lights(empty):
    block_model(empty)
    show_cage(empty)
    empty.handle("mouse_move", point_px("CAGE_2_0_1"))
    assert state.hover_handle == "CAGE_2_0_1" and state.hover is None
    empty.handle("mouse_move", chip_px("CAGE_LOCK_0"))
    assert state.hover_pill == "CAGE_LOCK_0"


# --- Reset: an undo of the whole cage session --------------------------------


def test_reset_puts_the_voxels_and_the_selection_back_as_they_were_before_the_first_drag(empty):
    block_model(empty)
    show_cage(empty)
    voxels, mask = document.grid.voxels.copy(), document.selection.mask.copy()
    corner_drag(empty)
    empty.redraw()
    start = point_px("CAGE_0_0_0")
    drag(empty, [start, {"x": start["x"] - 20.0, "y": start["y"]}, {"x": start["x"] - 40.0, "y": start["y"]}])
    assert not np.array_equal(document.grid.voxels, voxels)  # two drags bent it
    click(empty, chip_px("CAGE_RESET"))
    assert np.array_equal(document.grid.voxels, voxels) and np.array_equal(document.selection.mask, mask)
    cage = state.cage
    assert cage is not None and cage.counts == (3, 3, 3)
    assert np.allclose(cage.lattice.points, lattice.rest_lattice(((4, 4, 0), (9, 9, 3)), 3).points)
    assert np.array_equal(cage.rest_voxels, voxels)


def test_reset_is_one_undo_step_and_undo_brings_the_bend_back(empty):
    block_model(empty)
    show_cage(empty)
    original = document.grid.voxels.copy()
    corner_drag(empty)
    bent = document.grid.voxels.copy()
    click(empty, chip_px("CAGE_RESET"))
    assert np.array_equal(document.grid.voxels, original)
    key(empty, "z", ctrl=True)
    assert np.array_equal(document.grid.voxels, bent)  # one step took the whole reset back
    key(empty, "z", ctrl=True)
    assert np.array_equal(document.grid.voxels, original)  # and the next is the drag


def test_a_new_point_count_is_still_the_same_session_for_reset(empty):
    block_model(empty)
    show_cage(empty)
    original = document.grid.voxels.copy()
    corner_drag(empty)
    empty.redraw()
    set_points(empty, 4, 4, 4)  # the count starts the points again, on the bent voxels
    assert state.cage.counts == (4, 4, 4) and not np.array_equal(document.grid.voxels, original)
    click(empty, chip_px("CAGE_RESET"))
    assert np.array_equal(document.grid.voxels, original) and state.cage.counts == (3, 3, 3)


def test_reset_after_other_edits_goes_back_only_to_where_the_cage_started(empty):
    block_model(empty)
    show_cage(empty)
    corner_drag(empty)
    key(empty, "z", ctrl=True)  # an outside change: the cage starts over on the voxels now
    after_undo = document.grid.voxels.copy()
    corner_drag(empty)
    click(empty, chip_px("CAGE_RESET"))
    assert np.array_equal(document.grid.voxels, after_undo)


def test_reset_without_a_drag_changes_nothing_and_pushes_no_undo_step(empty):
    block_model(empty)
    show_cage(empty)
    before = document.grid.voxels.copy()
    steps = len(document.undo._undo) if hasattr(document.undo, "_undo") else None
    click(empty, chip_px("CAGE_RESET"))
    assert np.array_equal(document.grid.voxels, before)
    if steps is not None:
        assert len(document.undo._undo) == steps


# --- the axis locks ------------------------------------------------------------


def dragged_offset(app_, index=(2, 2, 2)):
    """The corner point's move after a drag to the right, in grid units."""
    before = state.cage.lattice.points[index].copy()
    corner_drag(app_, 60.0, 0.0)
    return state.cage.lattice.points[index] - before


def test_a_lock_chip_lights_and_a_second_click_unlights_it(empty):
    block_model(empty)
    show_cage(empty)
    assert state.cage.locks == (False, False, False)
    click(empty, chip_px("CAGE_LOCK_1"))
    assert state.cage.locks == (False, True, False)
    empty.redraw()  # draws lit
    click(empty, chip_px("CAGE_LOCK_1"))
    assert state.cage.locks == (False, False, False)
    assert document.selection.mask.sum() == len(BLOCK)  # a chip is not an edit


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_locked_axis_does_not_move_in_a_drag(empty, axis):
    block_model(empty)
    show_cage(empty)
    free = dragged_offset(empty)
    assert np.count_nonzero(free) >= 1  # without a lock the drag moves the point
    key(empty, "z", ctrl=True)  # back to the box on the original voxels
    click(empty, chip_px("CAGE_RESET"))
    empty.redraw()
    click(empty, chip_px(f"CAGE_LOCK_{axis}"))
    locked = dragged_offset(empty)
    assert locked[axis] == 0.0


def test_two_locks_leave_one_axis_and_the_mouse_moves_the_point_along_it(empty):
    block_model(empty)
    show_cage(empty)
    click(empty, chip_px("CAGE_LOCK_1"))
    click(empty, chip_px("CAGE_LOCK_2"))
    off = dragged_offset(empty)
    assert off[1] == 0.0 and off[2] == 0.0
    assert off[0] != 0.0  # the mouse still moves it, along X


def test_three_locks_freeze_the_cage_and_the_bend_is_nothing(empty):
    block_model(empty)
    show_cage(empty)
    before = document.grid.voxels.copy()
    for a in range(3):
        click(empty, chip_px(f"CAGE_LOCK_{a}"))
    off = dragged_offset(empty)
    assert not off.any() and np.array_equal(document.grid.voxels, before)


def test_locks_stay_over_a_new_point_count_and_reset_clears_them(empty):
    block_model(empty)
    show_cage(empty)
    click(empty, chip_px("CAGE_LOCK_0"))
    set_points(empty, 4, 4, 4)
    assert state.cage.locks == (True, False, False)
    click(empty, chip_px("CAGE_RESET"))
    assert state.cage.locks == (False, False, False)  # Reset: every chip back to its default
    click(empty, chip_px("CAGE_LOCK_2"))
    click(empty, chip_px("CAGE_DONE"))
    show_cage(empty)
    assert state.cage.locks == (False, False, False)  # a new cage starts free


def test_turning_mirror_off_after_a_bend_keeps_the_other_wing_bent(empty):
    from rok.app.opkit import CONTEXT

    wing = [(x, y, z) for x in range(1, 6) for y in range(4, 10) for z in range(0, 4)]
    put(wing, colour=6)
    put([(15 - x, y, z) for x, y, z in wing], colour=7)
    empty.redraw()
    document.selection.replace(document.grid.voxels == 6)
    CONTEXT.prefs.rok_mirror = (True, False, False)
    opkit.call("rok.cage", "EXEC")
    empty.redraw()
    try:
        start = point_px("CAGE_0_2_2")
        drag(empty, [start, {"x": start["x"] + 15, "y": start["y"]}, {"x": start["x"] + 30, "y": start["y"]}])
        other = document.grid.visible[8:].copy()
        CONTEXT.prefs.rok_mirror = (False, False, False)
        empty.handle("mouse_move", {"x": 5.0, "y": 5.0})  # any event: the cage starts again from the bend
        empty.redraw()
        assert state.cage is not None and state.cage.copies == ()
        start = point_px("CAGE_2_2_2")
        drag(empty, [start, {"x": start["x"] + 10, "y": start["y"]}, {"x": start["x"] + 20, "y": start["y"]}])
        assert np.array_equal(document.grid.visible[8:], other)  # the other wing kept its bend
    finally:
        CONTEXT.prefs.rok_mirror = (False, False, False)
