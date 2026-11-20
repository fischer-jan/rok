"""The whole-model operations (rok/app/model_ops.py): each one changes the
grid as the core says, pushes one undo record, and undo puts the model back.
The dialogs open with their defaults and Enter runs them. The host only records."""

from __future__ import annotations

import numpy as np
import pytest

from rok.app import opkit
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import Hover, state
from rok.core import edit, grid_ops, morph, reshape
from rok.core.raycast import Hit
from test_foundation import BIRD, RecordingHost, key


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ROK_CONFIG", str(tmp_path / "config"))
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(BIRD))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    document.colour = 7
    yield a
    document.reset()
    state.__init__()


def records() -> int:
    return len(document.undo._undo)


def snapshot():
    model = document.model
    return model.grid.voxels.copy(), tuple(model.position)


def assert_restored(app, before) -> None:
    app.handle("menu", {"id": "rok.undo"})
    voxels, position = before
    assert np.array_equal(document.grid.voxels, voxels)
    assert tuple(document.model.position) == position


def run(idname: str, **props) -> set:
    return opkit.call(idname, "EXEC", **props)


# --- the menu bar ---------------------------------------------------------------------


def test_the_menu_bar_leaves_the_model_commands_to_the_rail(app):
    menus = app.host.named("set_menu")[-1][1][0]
    titles = [title for title, _items in menus]
    assert titles[:4] == ["Rok", "File", "Edit", "View"]
    assert not {"Select", "Model", "Scene"} & set(titles)  # the rail has them
    assert dict(menus)["View"][0][0] == "rok.model_menu"  # Customize comes first


def test_a_menu_item_with_properties_runs_the_operator_with_them(app):
    before = snapshot()
    app.handle("menu", {"id": "rok.flip(axis='Y')"})
    assert np.array_equal(document.grid.voxels, grid_ops.flip(before[0], 1))


# --- operations that replace the whole grid ------------------------------------------------


def _expect(name, v, colour):
    return {
        "flip": lambda: grid_ops.flip(v, 1),
        "rotate": lambda: grid_ops.rotate90(v, 2),
        "fill_all": lambda: grid_ops.fill_all(v, colour),
        "fill_volume": lambda: grid_ops.fill_volume(v.shape, colour),
        "clear_all": lambda: grid_ops.fill_volume(v.shape, 0),
        "inverse": lambda: grid_ops.inverse(v, colour),
        "loop": lambda: grid_ops.loop(v, 0, 3),
        "scale_up": lambda: grid_ops.scale_up(v),
        "scale_down": lambda: grid_ops.scale_down(v),
        "repeat": lambda: grid_ops.repeat(v, (2, 1, 1)),
        "diagonal": lambda: grid_ops.diagonal(v),
        "lathe": lambda: reshape.lathe(v, 2),
        "twist": lambda: reshape.twist(v, 90.0, 2),
        "resize": lambda: grid_ops.resize(v, (40, 50, 30)),
        "mirror_half": lambda: grid_ops.mirror_half(v, 1, True),
        "grow": lambda: morph.grow(v, 2, limit=None, round_corners=False),
        "shrink": lambda: morph.shrink(v, 1, limit=None, round_corners=True),
        "hollow": lambda: morph.hollow(v, 2, limit=None),
        "smooth": lambda: morph.smooth(v, limit=None),
    }[name]()


CASES = [
    ("flip", "rok.flip", {"axis": "Y"}),
    ("rotate", "rok.rotate", {"axis": "Z"}),
    ("fill_all", "rok.fill_all", {}),
    ("fill_volume", "rok.fill_volume", {}),
    ("clear_all", "rok.clear_all", {}),
    ("inverse", "rok.inverse", {}),
    ("loop", "rok.loop", {"axis": "X", "offset": 3}),
    ("scale_up", "rok.scale", {"mode": "UP"}),
    ("scale_down", "rok.scale", {"mode": "DOWN"}),
    ("repeat", "rok.repeat", {"counts": (2, 1, 1)}),
    ("diagonal", "rok.diagonal", {}),
    ("lathe", "rok.lathe", {"axis": "Z"}),
    ("twist", "rok.twist", {"angle": 90, "axis": "Z"}),
    ("resize", "rok.resize", {"size": (40, 50, 30)}),
    ("mirror_half", "rok.mirror_half", {"axis": "Y", "from_positive": True}),
    ("grow", "rok.grow", {"count": 2}),
    ("shrink", "rok.shrink", {"count": 1, "round_corners": True}),
    ("hollow", "rok.hollow", {"wall": 2}),
    ("smooth", "rok.smooth", {}),
]


@pytest.mark.parametrize("name, idname, props", CASES, ids=[c[0] for c in CASES])
def test_an_operation_changes_the_grid_as_the_core_says_and_undo_restores_it(app, name, idname, props):
    before = snapshot()
    expected = _expect(name, before[0].copy(), document.colour)
    count = records()
    document.modified = False
    app.host.calls.clear()
    assert run(idname, **props) == {"FINISHED"}
    assert not np.array_equal(document.grid.voxels, before[0])  # the case tests a change
    assert np.array_equal(document.grid.voxels, expected)
    assert records() == count + 1
    assert document.modified
    # The change reaches the GPU by the end of the event.
    app.handle("tick", {})
    assert app.host.named("set_model") or app.host.named("update_region")
    assert not document.grid.dirty_chunks()
    assert_restored(app, before)


def test_rotate_keeps_the_box_centre_in_place(app):
    model = document.model
    size, position = model.grid.size, tuple(model.position)
    run("rok.rotate", axis="Z")
    centre = tuple(position[i] + size[i] // 2 for i in range(3))
    new = model.grid.size
    assert tuple(model.position) == tuple(centre[i] - new[i] // 2 for i in range(3))


def test_an_operation_that_changes_nothing_pushes_nothing(app):
    count = records()
    document.modified = False
    run("rok.loop", axis="X", offset=0)
    assert records() == count and not document.modified


def test_fit_grid_shrinks_to_the_voxels_and_keeps_their_world_place(app):
    before = snapshot()
    run("rok.resize", size=(90, 70, 30))
    run("rok.loop", axis="Z", offset=4)
    model = document.model
    world = {tuple(c + np.asarray(model.position)) for c in np.argwhere(model.grid.voxels)}
    assert run("rok.fit_grid") == {"FINISHED"}
    assert model.grid.size == before[0].shape
    assert {tuple(c + np.asarray(model.position)) for c in np.argwhere(model.grid.voxels)} == world


def test_drop_to_floor_moves_the_voxels_down(app):
    assert run("rok.drop_to_floor") == {"CANCELLED"}  # the bird is on its floor
    run("rok.resize", size=(80, 68, 30))
    run("rok.loop", axis="Z", offset=5)
    before = snapshot()
    expected, fall = grid_ops.drop_to_floor(before[0])
    assert fall == 5
    assert run("rok.drop_to_floor") == {"FINISHED"}
    assert np.array_equal(document.grid.voxels, expected)
    assert_restored(app, before)


def test_a_whole_grid_operation_drops_the_selection(app):
    run("rok.select_all")
    run("rok.flip", axis="Y")
    assert document.selection.is_empty()


def test_grow_on_a_selection_adds_the_new_layer_to_it(app):
    grid = document.grid
    sel = np.zeros(grid.size, dtype=bool)
    sel[30:40, 30:40, :] = grid.visible[30:40, 30:40, :] != 0
    document.selection.replace(sel)
    expected = morph.grow(grid.visible, 1, limit=sel, round_corners=False)
    run("rok.grow", count=1)
    assert np.array_equal(grid.voxels, expected)
    assert document.selection.count() > int(sel.sum())


def test_bevel_takes_the_edges_off(app):
    before = snapshot()
    remove = morph.bevel_mask(before[0] != 0, limit=None)
    expected = before[0].copy()
    expected[remove] = 0
    assert run("rok.bevel") == {"FINISHED"}
    assert np.array_equal(document.grid.voxels, expected)
    assert_restored(app, before)


def test_bridge_needs_two_parts(app):
    assert run("rok.bridge") == {"CANCELLED"}
    assert "Select two separate parts" in app.message


def test_bridge_fills_between_two_parts(app):
    run("rok.clear_all")
    grid = document.grid
    two = np.zeros(grid.size, dtype=np.uint8)
    two[10:13, 10:13, 2:5] = 3
    two[20:23, 10:13, 2:5] = 3
    grid.replace_voxels(two)
    document.selection.replace(two != 0)
    count = records()
    assert run("rok.bridge") == {"FINISHED"}
    assert records() == count + 1
    assert (grid.voxels[13:20, 10:13, 2:5] == document.colour).all()


def test_remove_slice_takes_the_selected_layers_out(app):
    grid = document.grid
    before = snapshot()
    sel = np.zeros(grid.size, dtype=bool)
    sel[10:12, :, :] = True
    document.selection.replace(sel)
    axis, first, last = reshape.slice_axis(document.selection.mask)
    expected = reshape.remove_slices(before[0], axis, first, last)
    assert run("rok.remove_slice") == {"FINISHED"}
    assert np.array_equal(grid.voxels, expected)
    assert_restored(app, before)


def test_remove_slice_without_a_selection_says_what_to_do(app):
    assert run("rok.remove_slice") == {"CANCELLED"}
    assert "Select the layers" in app.message


# --- dialogs -----------------------------------------------------------------------------


@pytest.mark.parametrize("idname, defaults", [
    ("rok.loop", {"axis": "X", "offset": 1}),
    ("rok.repeat", {"counts": (1, 1, 1)}),
    ("rok.hollow", {"wall": 1}),
    ("rok.grow", {"count": 1, "round_corners": False}),
    ("rok.shrink", {"count": 1, "round_corners": False}),
    ("rok.twist", {"angle": 90, "axis": "Z"}),
    ("rok.rotate_degrees", {"x": 0.0, "y": 0.0, "z": 45.0}),
])
def test_a_dialog_opens_with_its_defaults(app, idname, defaults):
    app.handle("menu", {"id": idname})
    assert app.dialog is not None and app.dialog.op.command_id == idname
    for name, value in defaults.items():
        assert getattr(app.dialog.op, name) == value
    app.redraw()  # the dialog lays out without an error
    assert any(h.kind == "ok" for h in app.dialog.hots)
    key(app, "Escape")
    assert app.dialog is None


def test_resize_opens_with_the_grid_size_and_enter_resizes(app):
    app.handle("menu", {"id": "rok.resize"})
    assert tuple(app.dialog.op.size) == document.grid.size == (80, 68, 22)
    app.dialog.op.size = (64, 64, 32)
    count = records()
    key(app, "Enter")
    assert app.dialog is None
    assert document.grid.size == (64, 64, 32) and records() == count + 1


def test_loop_dialog_enter_runs_it(app):
    before = snapshot()
    app.handle("menu", {"id": "rok.loop"})
    app.dialog.op.axis = "Y"
    app.dialog.op.offset = 4
    key(app, "Enter")
    assert np.array_equal(document.grid.voxels, grid_ops.loop(before[0], 1, 4))


def test_grow_without_the_dialog_runs_at_once(app):
    before = snapshot()
    assert opkit.call("rok.grow", "INVOKE", dialog=False) == {"FINISHED"}
    assert app.dialog is None
    assert np.array_equal(document.grid.voxels, morph.grow(before[0], 1, limit=None, round_corners=False))


def test_short_enum_buttons_share_one_row(app):
    app.handle("menu", {"id": "rok.loop"})
    app.redraw()
    rows = {h.rect[1] for h in app.dialog.hots if h.kind == "enum"}
    assert len(rows) == 1


def test_a_dialog_label_names_the_scope(app):
    app.handle("menu", {"id": "rok.hollow"})
    app.redraw()
    texts = [t[4] for t in _dialog_texts(app)]
    assert "Applies to the whole model" in texts


def _dialog_texts(app):
    from rok.app import draw, gfx

    d = draw.DrawList()
    with gfx.pixels(d, app.region):
        app.dialog.draw()
    return d.texts


def _slab_with_hover():
    """A 10 x 10 x 3 slab in an empty grid, and a hover on its top face."""
    grid = document.grid
    slab = np.zeros(grid.size, dtype=np.uint8)
    slab[5:15, 5:15, 0:3] = 4
    grid.replace_voxels(slab)
    state.hover = Hover(hit=Hit((9, 9, 2), (0, 0, 1), 1.0), tool=edit.ATTACH, target=(9, 9, 3))


def test_inset_takes_the_face_from_the_hover_and_raises_it(app):
    assert opkit.call("rok.inset", "INVOKE") == {"CANCELLED"}  # no face under the mouse
    _slab_with_hover()
    app.handle("menu", {"id": "rok.inset"})
    op = app.dialog.op
    assert tuple(op.voxel) == (9, 9, 2) and tuple(op.normal) == (0, 0, 1) and op.has_face
    assert (op.width, op.depth) == (1, 1)
    before = snapshot()
    count = records()
    key(app, "Enter")
    assert records() == count + 1
    grid = document.grid
    # The inside (8 x 8) rises one layer with the current colour; the rim stays.
    assert (grid.voxels[6:14, 6:14, 3] == document.colour).all()
    assert int(np.count_nonzero(grid.voxels[:, :, 3])) == 64
    assert_restored(app, before)


# --- typed grid size ------------------------------------------------------------------------


def test_typing_a_grid_size_resizes_on_enter(app):
    assert opkit.call("rok.grid_size_type", "INVOKE") == {"RUNNING"}
    assert state.size_edit is not None
    for ch in "40":
        key(app, ch)
    key(app, "Enter")
    assert state.size_edit is None and all(type(m).command_id in ("rok.voxel_mode", "rok.nav_cube") for m in app.modals)
    assert document.grid.size == (40, 40, 40)


def test_escape_leaves_the_typed_grid_size(app):
    opkit.call("rok.grid_size_type", "INVOKE")
    key(app, "4")
    key(app, "Escape")
    assert state.size_edit is None and document.grid.size == (80, 68, 22)


def test_no_model_operation_runs_during_a_stroke(app):
    state.stroke_active = True
    assert run("rok.flip", axis="X") == {"CANCELLED"}
    state.stroke_active = False


def _floating_setup():
    """An empty 8 x 8 x 16 grid: a slab at z 2 under column (2, 2), a lone voxel
    over it at z 12, and another at (5, 5, 14) over nothing."""
    run("rok.new", size=(8, 8, 16), discard=True)
    grid = document.grid
    grid.set_voxels(np.array([[2, 2, 2], [3, 2, 2], [2, 2, 12], [5, 5, 14]]), np.array([9, 9, 4, 4]))
    from rok.app import object as voxel_object

    voxel_object.flush_dirty()
    mask = np.zeros(grid.size, dtype=bool)
    mask[2, 2, 12] = mask[5, 5, 14] = True
    document.selection.replace(mask)
    return grid


def test_the_selection_falls_as_one_piece_until_one_voxel_lands(app):
    grid = _floating_setup()
    count = records()
    assert run("rok.drop_to_floor") == {"FINISHED"}
    # (2, 2, 12) lands on the slab at z 2 after 9 cells; (5, 5, 14) falls the same 9, not to the floor.
    assert int(grid.voxels[2, 2, 3]) == 4 and int(grid.voxels[5, 5, 5]) == 4
    assert int(grid.voxels[2, 2, 12]) == 0 and int(grid.voxels[5, 5, 14]) == 0
    assert int(grid.voxels[2, 2, 2]) == 9  # the unselected slab stays
    assert sorted(map(tuple, document.selection.cells().tolist())) == [(2, 2, 3), (5, 5, 5)]
    assert records() == count + 1
    run("rok.undo")
    assert int(grid.voxels[2, 2, 12]) == 4 and int(grid.voxels[2, 2, 3]) == 0
    assert document.selection.contains((2, 2, 12))


def test_drop_voxels_lets_each_selected_voxel_fall_on_its_own(app):
    grid = _floating_setup()  # slab voxels at z 2, and selected voxels at (2,2,12) and (5,5,14)
    count = records()
    assert run("rok.drop_voxels") == {"FINISHED"}
    assert int(grid.voxels[2, 2, 3]) == 4  # onto the slab, as the piece drop does
    assert int(grid.voxels[5, 5, 0]) == 4  # but this one reaches the floor: it does not stay level with the other
    assert int(grid.voxels[2, 2, 12]) == 0 and int(grid.voxels[5, 5, 14]) == 0
    assert int(grid.voxels[2, 2, 2]) == 9 and int(grid.voxels[3, 2, 2]) == 9  # the unselected slab stays
    assert sorted(map(tuple, document.selection.cells().tolist())) == [(2, 2, 3), (5, 5, 0)]
    assert records() == count + 1
    run("rok.undo")
    assert int(grid.voxels[2, 2, 12]) == 4 and int(grid.voxels[5, 5, 14]) == 4 and int(grid.voxels[5, 5, 0]) == 0
    assert document.selection.contains((2, 2, 12)) and document.selection.contains((5, 5, 14))


def test_drop_voxels_needs_a_selection_and_does_nothing_when_all_rest(app):
    grid = _floating_setup()
    document.selection.replace(np.zeros(grid.size, dtype=bool))
    assert run("rok.drop_voxels") == {"CANCELLED"}  # nothing selected
    grid.set_voxels(np.array([[7, 7, 0]]), np.array([5]))
    mask = np.zeros(grid.size, dtype=bool)
    mask[7, 7, 0] = True  # it rests on the floor
    document.selection.replace(mask)
    assert run("rok.drop_voxels") == {"CANCELLED"}


def test_drop_voxels_piles_a_selected_column_in_order(app):
    grid = _floating_setup()
    grid.set_voxels(np.array([[6, 6, 4], [6, 6, 9], [6, 6, 13]]), np.array([1, 2, 3]))
    mask = np.zeros(grid.size, dtype=bool)
    mask[6, 6, 4] = mask[6, 6, 9] = mask[6, 6, 13] = True
    document.selection.replace(mask)
    assert run("rok.drop_voxels") == {"FINISHED"}
    assert [int(grid.voxels[6, 6, z]) for z in range(4)] == [1, 2, 3, 0]  # a pile, the lowest first


def test_a_resting_selection_does_not_fall(app):
    grid = _floating_setup()
    grid.set_voxels(np.array([[0, 0, 0], [0, 0, 1]]), np.array([9, 4]))
    mask = np.zeros(grid.size, dtype=bool)
    mask[0, 0, 1] = True  # it sits on the voxel at z 0
    document.selection.replace(mask)
    assert run("rok.drop_to_floor") == {"CANCELLED"}
    mask[:] = False
    mask[0, 0, 0] = True  # and that one on the floor
    document.selection.replace(mask)
    assert run("rok.drop_to_floor") == {"CANCELLED"}


def test_from_the_rail_it_falls_faster_and_faster_bounces_and_a_key_ends_it(app, monkeypatch):
    from rok.app import model_ops

    grid = _floating_setup()
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    assert opkit.call("rok.drop_to_floor", "INVOKE") == {"RUNNING"}
    heights = []
    for _ in range(12):
        clock[0] += 1.0 / 60.0
        app.handle("tick", {})
        heights.append(int(np.argwhere(grid.voxels[5, 5] == 4)[0][0]))
    drops = [heights[i] - heights[i + 1] for i in range(len(heights) - 1)]
    assert heights[0] > 5 and drops[-1] >= drops[0]  # it speeds up
    assert model_ops.bounce_height(9) == 2 and model_ops.fall_offset(model_ops.fall_seconds(9) - 0.04, 9) < 9
    key(app, "a")  # any key: straight to the end
    assert int(grid.voxels[5, 5, 5]) == 4 and not state.stroke_active


def test_drop_voxels_from_the_rail_falls_with_gravity_bounces_and_is_one_undo_step(app, monkeypatch):
    from rok.app import model_ops

    grid = _floating_setup()  # selected voxels at (2,2,12) -> lands on the slab at 3; (5,5,14) -> the floor
    count = records()
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    assert opkit.call("rok.drop_voxels", "INVOKE") == {"RUNNING"}
    assert state.stroke_active
    heights = {5: [], 2: []}
    for _ in range(20):
        clock[0] += 1.0 / 60.0
        app.handle("tick", {})
        heights[5].append(int(np.argwhere(grid.voxels[5, 5] == 4)[0][0]))
        heights[2].append(int(np.argwhere(grid.voxels[2, 2] == 4)[0][0]))
    assert heights[5][0] > heights[5][-1] or heights[5][0] > 0  # it fell in steps, not at once
    assert heights[5][0] > 0 and heights[2][0] > 3  # both were still in the air after the first frames
    assert document.selection.contains((2, 2, heights[2][-1]))  # the selection came along
    # The voxel with the longer fall bounces: it rises again before it rests.
    long_fall = heights[5]
    assert min(long_fall) == 0 and long_fall[-1] > 0  # it reached the floor, then bounced up again
    key(app, "a")  # any key: straight to the end
    assert not state.stroke_active
    assert int(grid.voxels[2, 2, 3]) == 4 and int(grid.voxels[5, 5, 0]) == 4
    assert int(grid.voxels[2, 2, 12]) == 0 and int(grid.voxels[5, 5, 14]) == 0
    assert records() == count + 1  # the frames in between are not undo steps
    run("rok.undo")
    assert int(grid.voxels[2, 2, 12]) == 4 and int(grid.voxels[5, 5, 14]) == 4 and int(grid.voxels[5, 5, 0]) == 0
    assert model_ops.fall_offset(0.0, 12, 0) == 0


def test_a_falling_pile_keeps_every_voxel_in_every_frame(app, monkeypatch):
    """The voxels of one column fall and bounce each on its own, but none is ever written over."""
    grid = _floating_setup()
    grid.set_voxels(np.array([[6, 6, 4], [6, 6, 9], [6, 6, 13]]), np.array([1, 2, 3]))
    mask = np.zeros(grid.size, dtype=bool)
    mask[6, 6, 4] = mask[6, 6, 9] = mask[6, 6, 13] = True
    document.selection.replace(mask)
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    assert opkit.call("rok.drop_voxels", "INVOKE") == {"RUNNING"}
    for _ in range(40):
        clock[0] += 1.0 / 60.0
        app.handle("tick", {})
        column = grid.voxels[6, 6]
        assert sorted(int(v) for v in column[column != 0]) == [1, 2, 3]
    key(app, "a")
    assert [int(grid.voxels[6, 6, z]) for z in range(4)] == [1, 2, 3, 0]


def test_rotate_by_degrees_turns_the_selection_or_the_whole_model_in_one_undo_step(app):
    """A typed angle for whoever finds the rings slow on a big
    model; several axes at once turn X, then Y, then Z, about one pivot."""
    from rok.core import free_rotate, gizmo
    from rok.core.select import Selection
    from rok.app import gizmo_ops

    grid = document.grid
    start = grid.voxels.copy()
    # The whole model, two axes: as two turns in a row, one undo step.
    assert opkit.call("rok.rotate_degrees", "EXEC", x=20.0, y=0.0, z=30.0) == {"FINISHED"}
    items = [(start, (0, 0, 0))]
    items = free_rotate.rotate_models(items, 0, 20.0)
    items = free_rotate.rotate_models(items, 2, 30.0)
    assert np.array_equal(document.grid.voxels, items[0][0])
    opkit.call("rok.undo", "EXEC")
    assert np.array_equal(document.grid.voxels, start)
    # A selection: about its own centre, like the gizmo's ring.
    cells = np.argwhere(start != 0)
    some = cells[cells[:, 0] < int(np.median(cells[:, 0]))]
    document.selection.add_cells(some)
    pivot2 = gizmo_ops.selection_pivot2(document.model)
    fresh = start.copy()
    fresh_sel = Selection(fresh.shape)
    fresh_sel.add_cells(some)
    from helpers import ArraySetMany
    new_mask = gizmo.free_rotate_cells(fresh, fresh_sel, 2, 30.0, pivot2, ArraySetMany(fresh))
    assert opkit.call("rok.rotate_degrees", "EXEC", x=0.0, y=0.0, z=30.0) == {"FINISHED"}
    assert np.array_equal(document.grid.voxels, fresh)
    assert np.array_equal(document.selection.mask, new_mask)
    opkit.call("rok.undo", "EXEC")
    assert np.array_equal(document.grid.voxels, start)
    assert opkit.call("rok.rotate_degrees", "EXEC", x=0.0, y=0.0, z=0.0) == {"CANCELLED"}  # nothing to turn
