"""The selection, clipboard, gizmo, layer and level modules
(rok/app/select_ops.py, clip_ops.py, gizmo_ops.py, layer_ops.py, level_ops.py)
against the real fixture models. The host only records."""

from __future__ import annotations

from rok.app import opkit  # noqa: E402

import numpy as np
import pytest

from rok.app import gizmo_ops, opkit
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.opkit import CONTEXT
from rok.app.state import Hover, state
from rok.core import edit, gizmo, link, morph, region
from rok.core.raycast import Hit
from rok.core.undo import CompoundRecord, SelectionRecord
from test_foundation import FIXTURES, BIRD, RecordingHost, key

TWO = FIXTURES / "phase3" / "two_models.vox"


def _start(path):
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(path))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    return a


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("ROK_CONFIG", str(tmp_path / "config"))
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    yield
    document.reset()
    state.__init__()


@pytest.fixture
def app(config):
    a = _start(BIRD)
    document.colour = 7
    return a


@pytest.fixture
def two(config):
    return _start(TWO)


def records() -> int:
    return len(document.undo._undo)


def run(idname: str, **props) -> set:
    return opkit.call(idname, "EXEC", **props)


def undo(app) -> None:
    app.handle("menu", {"id": "rok.undo"})


def hover_on(cell) -> None:
    x, y, z = cell
    state.hover = Hover(hit=Hit((x, y, z), (0, 0, 1), 1.0), tool=edit.ATTACH, target=(x, y, z + 1))


def a_filled_cell():
    return tuple(int(c) for c in np.argwhere(document.grid.visible)[0])


# --- selection ------------------------------------------------------------------------


def test_select_all_selects_every_voxel_and_undo_clears_it(app):
    count = records()
    assert run("rok.select_all") == {"FINISHED"}
    assert np.array_equal(document.selection.mask, document.grid.visible != 0)
    assert records() == count + 1 and isinstance(document.undo.peek(), SelectionRecord)
    assert not document.modified  # a selection is not saved data
    assert CONTEXT.prefs.rok_tool != edit.SELECT  # the tool stays as it was
    undo(app)
    assert document.selection.is_empty()


def test_deselect_polls_false_without_a_selection(app):
    assert not opkit.ops.rok.deselect.poll()
    run("rok.select_all")
    assert run("rok.deselect") == {"FINISHED"} and document.selection.is_empty()
    undo(app)
    assert document.selection.count() == int((document.grid.visible != 0).sum())


def test_invert_twice_is_the_same_selection(app):
    cell = a_filled_cell()
    document.selection.add_cells(np.asarray([cell]))
    run("rok.select_invert")
    assert not document.selection.contains(cell)
    assert document.selection.count() == int((document.grid.visible != 0).sum()) - 1
    run("rok.select_invert")
    assert document.selection.count() == 1 and document.selection.contains(cell)


def test_grow_and_shrink_selection(app):
    grid = document.grid
    cell = a_filled_cell()
    document.selection.add_cells(np.asarray([cell]))
    start = document.selection.mask.copy()
    run("rok.select_grow")
    grown = document.selection.mask.copy()
    assert np.array_equal(grown, start | (morph.dilate(start, 1) & (grid.visible != 0)))
    run("rok.select_shrink")
    assert document.selection.contains(cell)
    assert document.selection.count() < int(grown.sum())


def test_ctrl_plus_and_minus_grow_and_shrink_the_selection(app):
    """Grow and shrink the selection from the keyboard. The +
    is Shift+= on a US keyboard and a key of its own on a German one; both
    work, with Ctrl or with Cmd."""
    before = document.grid.visible.copy()
    cell = a_filled_cell()
    document.selection.add_cells(np.asarray([cell]))
    one = document.selection.count()
    key(app, "+", ctrl=True, shift=True)  # a US keyboard's Shift+=
    grown = document.selection.count()
    assert grown > one
    key(app, "+", meta=True)  # a German keyboard's own + key, with Cmd
    assert document.selection.count() > grown
    key(app, "-", ctrl=True)
    key(app, "-", meta=True)
    assert document.selection.count() == one and document.selection.contains(cell)
    assert np.array_equal(document.grid.visible, before)  # the voxels are untouched


def test_select_by_colour_uses_the_hovered_voxel(app):
    grid = document.grid
    cell = a_filled_cell()
    hover_on(cell)
    run("rok.select_colour")
    assert np.array_equal(document.selection.mask, grid.visible == grid.visible[cell])


def test_select_connected_from_the_hover(app):
    grid = document.grid
    cell = a_filled_cell()
    hover_on(cell)
    run("rok.select_connected")
    expected = region.connected_blobs(grid.visible, np.asarray([cell]))
    assert np.array_equal(document.selection.mask, expected)


def test_delete_selection_removes_the_voxels_in_one_step(app):
    grid = document.grid
    before = grid.voxels.copy()
    hover_on(a_filled_cell())
    run("rok.select_colour")
    picked = document.selection.mask.copy()
    count = records()
    assert run("rok.selection_delete") == {"FINISHED"}
    assert not grid.voxels[picked].any() and document.selection.is_empty()
    assert records() == count + 1 and isinstance(document.undo.peek(), CompoundRecord)
    assert document.modified
    undo(app)
    assert np.array_equal(grid.voxels, before)
    assert np.array_equal(document.selection.mask, picked)


def test_fill_paints_the_joined_colour_under_the_mouse(app):
    grid = document.grid
    assert not opkit.ops.rok.fill.poll()  # no hover
    cell = a_filled_cell()
    hover_on(cell)
    expected = region.connected_colour(grid.visible, cell)
    before = grid.voxels.copy()
    assert run("rok.fill") == {"FINISHED"}
    assert (grid.voxels[expected] == document.colour).all()
    assert np.array_equal(grid.voxels[~expected], before[~expected])
    undo(app)
    assert np.array_equal(grid.voxels, before)


def test_the_section_plane_toggles(app):
    grid = document.grid
    assert run("rok.section", axis="Z", action="TOGGLE") == {"FINISHED"}
    assert grid.sections[2][0] == grid.size[2] // 2
    run("rok.section", axis="CURRENT", action="DOWN")
    assert grid.sections[2][0] != grid.size[2] // 2
    run("rok.section", axis="Z", action="TOGGLE")
    assert 2 not in grid.sections


# --- the clipboard ------------------------------------------------------------------------


def test_copy_needs_a_selection(app):
    assert run("rok.copy") == {"CANCELLED"}
    assert app.message == "Select something first"


def test_copy_clear_paste_brings_the_model_back(app):
    grid = document.grid
    before = grid.voxels.copy()
    run("rok.select_all")
    assert run("rok.copy") == {"FINISHED"}
    assert state.clipboard.count() == int((before != 0).sum())
    run("rok.clear_all")
    count = records()
    assert run("rok.paste") == {"FINISHED"}
    assert np.array_equal(grid.voxels, before)
    assert np.array_equal(document.selection.mask, before != 0)
    assert records() == count + 1
    assert app.message == f"Pasted {int((before != 0).sum())} voxels"
    undo(app)
    assert not grid.voxels.any()


def test_cut_removes_and_paste_puts_back(app):
    grid = document.grid
    before = grid.voxels.copy()
    hover_on(a_filled_cell())
    run("rok.select_colour")
    picked = document.selection.mask.copy()
    assert run("rok.cut") == {"FINISHED"}
    assert not grid.voxels[picked].any()
    run("rok.paste")
    assert np.array_equal(grid.voxels, before)


def test_paste_into_another_file_fits_the_palette_in_one_undo_step(config):
    _start(FIXTURES / "robot.vox")
    run("rok.select_all")
    run("rok.copy")
    clip = state.clipboard
    app = CONTEXT.app
    assert run("rok.open", filepath=str(BIRD)) == {"FINISHED"}  # the clipboard survives Open
    assert document.name == "bird.vox"
    assert state.clipboard is clip
    palette = document.scene.palette.copy()
    voxels = document.grid.voxels.copy()
    count = records()
    assert run("rok.paste") == {"FINISHED"}
    assert records() == count + 1
    record = document.undo.peek()
    assert not np.array_equal(document.scene.palette, palette)  # the two files' palettes differ
    assert record.colour_change is not None  # the new colours undo with the voxels
    undo(app)
    assert np.array_equal(document.scene.palette, palette)
    assert np.array_equal(document.grid.voxels, voxels)


def test_paste_with_an_empty_clipboard_says_so(app):
    assert run("rok.paste") == {"CANCELLED"}
    assert app.message == "Nothing to paste"


# --- the gizmo ------------------------------------------------------------------------


def test_refresh_shows_the_gizmo_for_a_selection_with_the_select_tool(app):
    model = document.model
    lo = np.array([10, 10, 2])
    mask = np.zeros(model.grid.size, dtype=bool)
    mask[10:14, 10:16, 2:5] = True
    document.selection.replace(mask)
    gizmo_ops.refresh()
    assert state.gizmo is None  # not with Attach
    CONTEXT.prefs.rok_tool = edit.SELECT
    gizmo_ops.refresh()
    g = state.gizmo
    assert g.target == gizmo.SELECTION and g.uid == model.uid
    assert g.pivot2 == tuple(int(v) for v in (2 * lo + np.array([4, 6, 3])))
    assert g.length_px == gizmo_ops.GIZMO_LENGTH_PX
    document.selection.clear()
    gizmo_ops.refresh()
    assert state.gizmo is None  # one model and no selection: nothing to move


def test_transform_moves_the_selection_and_undo_puts_it_back(app):
    CONTEXT.prefs.rok_tool = edit.SELECT
    grid = document.grid
    before = grid.voxels.copy()
    hover_on(a_filled_cell())
    run("rok.select_colour")
    picked = document.selection.mask.copy()
    count = records()
    assert run("rok.transform", target=gizmo.SELECTION, kind=gizmo.MOVE, offset=(0, 0, 1)) == {"FINISHED"}
    assert records() == count + 1 and document.modified
    assert np.array_equal(document.selection.mask[:, :, 1:], picked[:, :, :-1] & (grid.visible[:, :, 1:] != 0))
    assert state.gizmo is not None and state.gizmo.target == gizmo.SELECTION
    undo(app)
    assert np.array_equal(grid.voxels, before)


def test_transform_turns_the_whole_model(two):
    model = document.model
    voxels = model.grid.voxels.copy()
    position = tuple(model.position)
    assert run("rok.transform", target=gizmo.MODEL, kind=gizmo.ROTATE, axis=2, turns=1) == {"FINISHED"}
    assert model.grid.size == (voxels.shape[1], voxels.shape[0], voxels.shape[2])
    assert model.orient != link.IDENTITY  # a linked copy would turn alone
    undo(two)
    assert np.array_equal(model.grid.voxels, voxels) and tuple(model.position) == position


def test_a_model_drag_shows_the_model_moved_until_it_ends(two):
    model = document.model
    gizmo_ops.begin_model_drag()
    gizmo_ops.update_model_drag(gizmo.GizmoResult(kind=gizmo.MOVE, offset=(3, 0, 0)))
    assert state.model_drag_offset == (3, 0, 0)
    turn = state.turn_preview[model.uid]
    assert np.allclose(turn[:3, 3], (3, 0, 0))
    two.host.calls.clear()
    two.handle("tick", {})
    assert two.host.named("set_model_turn")
    gizmo_ops.end_model_drag(cancel=True)
    assert model.uid not in state.turn_preview and state.model_drag_offset == (0, 0, 0)


def _world_centres(model, matrix=None):
    c = np.argwhere(model.grid.voxels != 0) + 0.5
    if matrix is not None:
        c = c @ matrix[:3, :3].T + matrix[:3, 3]
    return {tuple(np.round(p + np.asarray(model.position), 6)) for p in c}


def test_a_turn_drag_turns_the_real_voxels_where_the_release_puts_them(two):
    model = document.model
    before = model.grid.voxels.copy()
    gizmo_ops.begin_model_drag()
    res = gizmo.GizmoResult(kind=gizmo.ROTATE, axis=2, turns=1, degrees=90.0,
                            pivot2=gizmo.snap_pivot(gizmo_ops.model_pivot2(model), 2))
    gizmo_ops.update_model_drag(res)
    assert state.gizmo_ghost is None and not state.turn_preview  # no stand-in: the voxels turn
    shown = _world_centres(model)
    assert shown != _world_centres_of(before, model)
    gizmo_ops.end_model_drag(cancel=True)
    assert np.array_equal(model.grid.voxels, before)  # Esc gives the press back
    run("rok.transform", target=gizmo.MODEL, kind=gizmo.ROTATE, axis=2, turns=1)
    assert _world_centres(document.model) == shown


def _world_centres_of(voxels, model):
    c = np.argwhere(voxels != 0) + 0.5
    return {tuple(np.round(p + np.asarray(model.position), 6)) for p in c}


def test_the_pivot_can_move_and_come_back(app):
    CONTEXT.prefs.rok_tool = edit.SELECT
    run("rok.select_all")
    saved = gizmo_ops.save_pivot()
    gizmo_ops.set_pivot((4, 6, 8))
    assert document.pivot_override == (2.0, 3.0, 4.0) and state.gizmo.pivot2 == (4, 6, 8)
    gizmo_ops.restore_pivot(saved)
    assert document.pivot_override is None


def test_the_scene_shows_the_model_gizmo(two):
    assert state.level == "SCENE"
    gizmo_ops.refresh()
    assert state.gizmo.target == gizmo.MODEL
    assert state.gizmo.pivot2 == tuple(int(v) for v in document.model.grid.size)


# --- levels -------------------------------------------------------------------------------


def test_a_file_of_two_models_opens_in_the_scene_and_enter_steps_in(two):
    assert state.level == "SCENE"
    assert opkit.call("rok.level", "INVOKE", action="IN") == {"FINISHED"}
    assert state.level == "MODEL"
    assert state.gizmo is None  # Attach, no selection, two models: the model gizmo only with Select
    assert opkit.call("rok.level", "INVOKE", action="OUT") == {"FINISHED"}
    assert state.level == "SCENE" and state.last_inside == document.active_uid


def test_tab_steps_through_model_scene_and_animate(two):
    tab = {"key": "Tab", "code": "Tab", "text": "\t", "repeat": False}
    shift = {"shift": True, "ctrl": False, "alt": False, "meta": False}
    assert state.level == "SCENE" and not state.animating
    two.handle("key_down", tab)
    assert state.animating  # the scene's next is Animate
    two.handle("key_down", tab)
    assert state.level == "MODEL" and not state.animating
    run("rok.select_all")  # unlike Esc, Tab leaves at once; the selection stays in its model
    two.handle("key_down", tab)
    assert state.level == "SCENE" and not document.selection.is_empty()
    two.handle("modifiers", shift)  # Shift+Tab steps back
    two.handle("key_down", tab)
    assert state.level == "MODEL" and not state.animating
    two.handle("key_down", tab)
    assert state.animating


def test_tab_still_turns_the_cutting_plane_with_the_cut_tool(two):
    tab = {"key": "Tab", "code": "Tab", "text": "\t", "repeat": False}
    run("rok.scene_tool", tool="CUT")
    before = state.cut_other
    two.handle("key_down", tab)
    assert state.level == "SCENE" and state.cut_other != before


def test_step_out_waits_while_something_is_selected(two):
    run("rok.level", action="IN")
    run("rok.select_all")
    assert run("rok.level", action="OUT") == {"PASS"}
    assert state.level == "MODEL"
    run("rok.deselect")
    run("rok.level", action="OUT")
    assert state.level == "SCENE"


def test_the_scene_fades_the_other_models_inside_one(two):
    from rok.app.object import style_of

    run("rok.level", action="IN")
    other = next(m for m in document.scene.models if m.uid != document.active_uid)
    assert style_of(other)[0] == 1.0
    run("rok.level", action="OUT")
    assert style_of(other)[0] == 0.0


def test_step_into_makes_the_model_active(two):
    from rok.app import level_ops

    other = next(m for m in document.scene.models if m.uid != document.active_uid)
    level_ops.step_into(other.uid)
    assert document.active_uid == other.uid and state.level == "MODEL"


def test_selection_operators_do_not_run_in_the_scene(two):
    assert state.level == "SCENE"
    assert run("rok.select_all") == {"CANCELLED"}


# --- layers -------------------------------------------------------------------------------


def test_a_model_moves_to_a_layer_and_hiding_it_keeps_a_visible_model_active(two):
    first = document.active_uid
    assert run("rok.model_to_layer", layer=1) == {"FINISHED"}
    assert int(document.model.layer) == 1
    assert run("rok.layer_toggle", layer=1) == {"FINISHED"}
    assert document.scene.is_hidden(document.scene.by_uid(first))
    assert document.active_uid != first  # the visible model took over
    assert run("rok.model_edit_from_list", uid=first) == {"FINISHED"}
    assert document.active_uid == first and not document.scene.is_hidden(document.model)
    for _ in range(3):
        undo(two)
    assert int(document.scene.by_uid(first).layer) == 0


def test_escape_deselects_first_then_steps_out_to_the_scene(app):
    assert state.level == "MODEL"
    opkit.call("rok.select_all", "EXEC")
    sel = document.selection
    assert not sel.is_empty()
    key(app, "Escape")  # Esc deselects first
    assert sel.is_empty() and state.level == "MODEL"
    from rok.app import level_ops, select_ops

    select_ops.last_deselect -= level_ops.ESC_GUARD_S  # a moment later
    key(app, "Escape")  # then steps out to the scene
    assert state.level == "SCENE"
