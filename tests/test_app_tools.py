"""The editing heart: rok.stroke in every mode, the hover, the keys that
switch tools and modes, the view keys and the overlay's two passes.

Every test drives the app through `app.handle(...)` with a host that only
records, as a user would: mouse moves, presses and keys. Host pixels have
their origin at the top left; `camera.project` gives them for a world point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from helpers import FIXTURES

from rok.app import opkit, overlay, tools
from rok.app.app import RokApp
from rok.app.object import document, model_origin
from rok.app.opkit import CONTEXT
from rok.app.state import state
from rok.core import edit, hud
from rok.core.clipboard import Clip
from rok.core.material import MaterialTable

BIRD = FIXTURES / "bird.vox"
CAR = FIXTURES / "car.vox"
W, H = 1440.0, 900.0
NO_MODS = {"shift": False, "ctrl": False, "alt": False, "meta": False}


class RecordingHost:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "measure_text":
                return (len(args[0]) * 7.0, 16.0)
            return None

        return record

    def named(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ROK_CONFIG", str(tmp_path / "config"))
    from rok.app import io_ops, recent
    from rok.core import new_sizes

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setattr(io_ops, "_last_new_size", new_sizes.size_of(new_sizes.DEFAULT))
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(BIRD))
    a.handle("start", {"width": W, "height": H, "scale": 2.0})
    yield a
    document.reset()
    state.__init__()


@pytest.fixture
def empty(app):
    """An empty 16 x 16 x 8 grid, framed."""
    opkit.call("rok.new", "EXEC", size=(16, 16, 8))
    app.handle("resize", {"width": W, "height": H})
    return app


# --- driving helpers ----------------------------------------------------------------


def mods(app, **on) -> None:
    app.handle("modifiers", {**NO_MODS, **on})


def key(app, name: str, **on) -> None:
    mods(app, **on)
    app.handle("key_down", {"key": name, "code": name, "text": name if len(name) == 1 else "", "repeat": False})
    app.handle("key_up", {"key": name, "code": name, "text": "", "repeat": False})
    mods(app)


def at(app, world) -> dict:
    """Host pixel of a world point."""
    x, y = app.camera.project([world])[0]
    return {"x": float(x), "y": float(y)}


def on_floor(app, x: int, y: int) -> dict:
    """The centre of floor cell (x, y) of the active model."""
    o = np.asarray(model_origin(document.model), dtype=np.float64)
    return at(app, o + (x + 0.5, y + 0.5, 0.0))


def on_top(app, cell) -> dict:
    """The centre of the top face of a voxel of the active model."""
    o = np.asarray(model_origin(document.model), dtype=np.float64)
    x, y, z = cell
    return at(app, o + (x + 0.5, y + 0.5, z + 1.0))


def click(app, p, button: str = "left", **on) -> None:
    app.handle("mouse_move", p)  # point first: a held key and a move would move the view
    mods(app, **on)
    app.handle("mouse_down", {"button": button})
    app.handle("mouse_up", {"button": button})
    mods(app)


def drag(app, points, button: str = "left", release: bool = True) -> None:
    app.handle("mouse_move", points[0])
    app.handle("mouse_down", {"button": button})
    for p in points[1:]:
        app.handle("mouse_move", p)
    if release:
        app.handle("mouse_up", {"button": button})


def filled(colour=None) -> set:
    v = document.grid.voxels
    mask = v != 0 if colour is None else v == colour
    return {tuple(int(c) for c in cell) for cell in np.argwhere(mask)}


def put(cells, colour: int = 5) -> None:
    """Voxels written straight into the grid, as a loaded file would have them."""
    arr = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    document.grid.set_voxels(arr, np.full(len(arr), colour, dtype=np.uint8))
    document.stroke_serial += 1
    document.edit_serial += 1


def modal_names(app) -> list[str]:
    return [type(op).command_id for op in app.modals]


# --- the hover ------------------------------------------------------------------------


def test_the_hover_operator_runs_from_the_start(app):
    assert modal_names(app) == ["rok.voxel_mode", "rok.nav_cube"]
    assert state.running


def test_the_hover_names_the_floor_cell_a_click_would_fill(empty):
    empty.handle("mouse_move", on_floor(empty, 3, 4))
    hover = state.hover
    assert hover is not None and hover.tool == "ATTACH" and hover.hit.wall
    assert hover.target == (3, 4, 0)


def test_option_turns_the_hover_into_erase(empty):
    put([(3, 4, 0)])
    empty.handle("mouse_move", on_top(empty, (3, 4, 0)))
    assert state.hover.tool == "ATTACH" and state.hover.target == (3, 4, 1)
    mods(empty, alt=True)
    assert state.hover.tool == "ERASE" and state.hover.target == (3, 4, 0)


def test_with_option_the_empty_floor_shows_no_hover_face(empty):
    from rok.app.overlay import shows_face

    empty.handle("mouse_move", on_floor(empty, 3, 4))
    assert shows_face(state.hover)  # Attach builds on the floor: the face shows
    mods(empty, alt=True)
    assert state.hover.tool == "ERASE" and state.hover.target is None
    assert not shows_face(state.hover)  # nothing to erase there: no face anywhere on the walls
    mods(empty, shift=True)
    assert state.hover.tool == "PAINT" and not shows_face(state.hover)
    put([(3, 4, 0)])
    mods(empty)
    empty.handle("mouse_move", on_top(empty, (3, 4, 0)))
    mods(empty, alt=True)
    assert shows_face(state.hover)  # on a voxel it shows


def test_the_hover_follows_a_view_change(empty):
    empty.handle("mouse_move", on_floor(empty, 3, 4))
    before = state.hover.target
    empty.handle("scroll", {"dx": 120.0, "dy": 0.0, "precise": True})  # two fingers orbit
    assert state.hover is not None and state.hover.target != before


# --- Voxel mode -------------------------------------------------------------------------


def test_a_click_attaches_one_voxel_in_the_current_colour(empty):
    document.colour = 7
    click(empty, on_floor(empty, 3, 4))
    assert filled() == {(3, 4, 0)} and filled(7) == {(3, 4, 0)}
    assert document.modified and not state.stroke_active
    assert modal_names(empty) == ["rok.voxel_mode", "rok.nav_cube"]


def test_ctrl_z_takes_a_stroke_back_and_redo_puts_it_again(empty):
    click(empty, on_floor(empty, 3, 4))
    click(empty, on_floor(empty, 6, 6))
    key(empty, "z", ctrl=True)
    assert filled() == {(3, 4, 0)}
    key(empty, "z", ctrl=True, shift=True)
    assert filled() == {(3, 4, 0), (6, 6, 0)}


def test_a_drag_fills_every_cell_it_crosses_as_one_undo_step(empty):
    drag(empty, [on_floor(empty, x, 2) for x in range(2, 9)])
    assert {(x, 2, 0) for x in range(2, 9)} <= filled()
    key(empty, "z", ctrl=True)
    assert filled() == set()


def test_one_undo_removes_a_whole_designed_stroke(empty):
    from rok.core.stroke_design import default_design, pack

    wm = CONTEXT.prefs
    wm.rok_brush_size = 4
    wm.rok_stroke_design = True
    wm.rok_stroke_curves = pack(default_design())
    try:
        drag(empty, [on_floor(empty, x, 3) for x in range(1, 14)])
    finally:
        wm.rok_stroke_design = False  # the later tests draw plain strokes
    cells = filled()
    assert len(cells) > 14
    end = max(c[0] for c in cells)
    start = min(c[0] for c in {c for c in cells if c[1] == 3 and c[2] == 0} or cells)
    assert sum(1 for c in cells if c[0] == end) == 1  # one voxel at the end
    assert sum(1 for c in cells if c[0] == start) > 1  # wide at the start
    key(empty, "z", ctrl=True)
    assert filled() == set()


def test_the_right_button_erases(empty):
    put([(3, 4, 0), (3, 4, 1)])
    click(empty, on_top(empty, (3, 4, 1)), button="right")
    assert filled() == {(3, 4, 0)}


def test_option_click_erases(empty):
    put([(3, 4, 0), (3, 4, 1)])
    click(empty, on_top(empty, (3, 4, 1)), alt=True)
    assert filled() == {(3, 4, 0)}


def test_paint_recolours_and_adds_nothing(empty):
    put([(3, 4, 0), (4, 4, 0)], colour=5)
    key(empty, "g")
    assert CONTEXT.prefs.rok_tool == "PAINT"
    document.colour = 9
    click(empty, on_top(empty, (3, 4, 0)))
    assert filled() == {(3, 4, 0), (4, 4, 0)} and filled(9) == {(3, 4, 0)}


def test_mirror_x_repeats_the_attach_on_the_other_side(empty):
    opkit.call("rok.toggle_mirror", axis=0)
    assert tuple(CONTEXT.prefs.rok_mirror) == (True, False, False)
    click(empty, on_floor(empty, 2, 5))
    assert filled() == {(2, 5, 0), (13, 5, 0)}


def test_a_fast_paint_or_erase_stroke_leaves_no_gaps(empty):
    """A Voxel stroke samples the
    mouse's path between two events, so one long jump paints the whole row."""
    slab = [(x, y, z) for x in range(16) for y in range(16) for z in range(2)]
    put(slab)
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "PAINT", "VOXEL"
    document.colour = 9
    drag(empty, [on_top(empty, (1, 8, 1)), on_top(empty, (14, 8, 1))])  # one event from one end to the other
    row = document.grid.visible[1:15, 8, 1]
    assert (row == 9).all(), row.tolist()
    wm.rok_tool = "ERASE"
    drag(empty, [on_top(empty, (1, 4, 1)), on_top(empty, (14, 4, 1))])
    assert (document.grid.visible[1:15, 4, 1] == 0).all()
    assert (document.grid.visible[1:15, 4, 0] != 0).all()  # one layer only


@pytest.mark.parametrize("brush", [1, 3])
def test_a_flick_faster_than_the_rays_still_cuts_a_whole_groove(empty, monkeypatch, brush):
    """A cut never breaks, however fast the stroke. With two rays between events the hits lie far apart; the stroke
    fills the way between them in the voxels."""
    monkeypatch.setattr(tools, "STROKE_SAMPLES", 2)
    put([(x, y, z) for x in range(16) for y in range(16) for z in range(4)])
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_brush_size = "ERASE", "VOXEL", brush
    drag(empty, [on_top(empty, (1, 8, 3)), on_top(empty, (14, 8, 3))])
    assert (document.grid.visible[1:15, 8, 3] == 0).all(), document.grid.visible[:, 8, 3].tolist()


def test_a_stroke_does_not_fill_a_jump_past_an_edge(empty):
    """Two hits far apart in depth are two surfaces, not one: the way between
    them goes through the air or the model, so the stroke leaves it."""
    put([(x, y, z) for x in range(4) for y in range(16) for z in range(8)]  # a tall wall at the back
        + [(x, y, 0) for x in range(4, 16) for y in range(16)])  # and the floor in front
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "PAINT", "VOXEL"
    document.colour = 9
    drag(empty, [on_top(empty, (3, 8, 7)), on_top(empty, (12, 8, 0))])  # from the wall's top onto the floor
    v = document.grid.visible
    assert (v[4:16, :, 1:] == 0).all()  # nothing grew in the air
    assert (v[0:3, 8, 0:7] != 9).all()  # the wall's inside kept its colour


def test_a_bigger_brush_erases_a_sphere(empty):
    slab = [(x, y, z) for x in range(16) for y in range(16) for z in range(3)]
    put(slab)
    CONTEXT.prefs.rok_brush_size = 2
    click(empty, on_top(empty, (8, 8, 2)), button="right")
    gone = set(slab) - filled()
    assert (8, 8, 2) in gone and len(gone) > 1
    assert all(max(abs(a - b) for a, b in zip(c, (8, 8, 2))) <= 1 for c in gone)


# --- the shapes ---------------------------------------------------------------------------


def _box(empty, a, b, lift_px: float):
    drag(empty, [on_floor(empty, *a), on_floor(empty, *b)])
    assert state.stroke_active  # the height stage waits
    p = on_floor(empty, *b)
    empty.handle("mouse_move", {"x": p["x"], "y": p["y"] - lift_px})


def test_box_sets_a_footprint_then_a_height_and_builds_on_the_click(empty):
    key(empty, "b")
    assert CONTEXT.prefs.rok_mode == "BOX"
    _box(empty, (2, 2), (5, 4), 120.0)
    assert filled() == set()  # nothing written before the click
    click(empty, {"x": empty.mouse[0], "y": empty.mouse[1]})
    cells = filled()
    xs, ys, zs = (sorted({c[i] for c in cells}) for i in range(3))
    assert xs == [2, 3, 4, 5] and ys == [2, 3, 4] and zs[0] == 0 and len(zs) >= 2
    assert len(cells) == 4 * 3 * len(zs)  # a solid block
    assert not state.stroke_active
    key(empty, "z", ctrl=True)
    assert filled() == set()


def test_the_box_preview_is_one_layer_sent_only_on_a_change(empty):
    """Raising a box's height stuttered. The preview's cells
    were rebuilt, hashed and uploaded every frame; now they are a keyed layer
    of their own, sent once per change and cleared with the preview."""
    def sends():
        return [c for c in empty.host.calls if c[0] == "set_draw" and c[1] and c[1][0] == "overlay.preview"]

    key(empty, "b")
    _box(empty, (2, 2), (9, 9), 60.0)
    empty.redraw()
    n = len(sends())
    assert n >= 1
    empty.redraw()
    empty.redraw()
    assert len(sends()) == n  # frames without a change send nothing
    p = on_floor(empty, 9, 9)
    empty.handle("mouse_move", {"x": p["x"], "y": p["y"] - 120.0})
    empty.redraw()
    assert len(sends()) > n  # a new height: built and sent once more
    assert not any(c[0] == "clear_draw" and c[1][0] == "overlay.preview" for c in empty.host.calls)
    key(empty, "Escape")
    empty.redraw()
    assert state.preview is None
    assert any(c[0] == "clear_draw" and c[1][0] == "overlay.preview" for c in empty.host.calls)


def test_the_hover_region_is_one_layer_sent_only_when_the_hover_changes(app):
    """A By-colour hover over a big model rebuilt the blob's
    faces every frame while the mouse rested. The region is a keyed layer now."""
    def sends():
        return [c for c in app.host.calls if c[0] == "set_draw" and c[1] and c[1][0] == "overlay.hover"]

    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "SELECT", "BY_COLOUR"
    cells = np.argwhere(document.grid.visible != 0)
    top = cells[np.argmax(cells[:, 2])]
    app.handle("mouse_move", on_top(app, top))
    assert state.hover is not None and state.hover.cells is not None and len(state.hover.cells) > 1
    app.redraw()
    n = len(sends())
    assert n >= 1
    app.redraw()
    app.redraw()
    assert len(sends()) == n  # the mouse rests: nothing sent
    # Another voxel of the same colour: the same blob, the same layer.
    same = cells[document.grid.visible[cells[:, 0], cells[:, 1], cells[:, 2]] == document.grid.visible[tuple(top)]]
    other = next((c for c in same if abs(int(c[0]) - int(top[0])) > 2 and c[2] == top[2]), None)
    if other is not None:
        app.handle("mouse_move", on_top(app, other))
        app.redraw()
        if state.hover is not None and state.hover.cells is not None and state.hover.cells is not None:
            assert len(sends()) <= n + 1
    # Off the model: the layer goes.
    app.handle("mouse_move", {"x": 4.0, "y": app.height / 2})
    app.redraw()
    assert any(c[0] == "clear_draw" and c[1][0] == "overlay.hover" for c in app.host.calls)


def test_scrolling_in_the_height_stage_makes_the_box_hollow(empty):
    key(empty, "b")
    _box(empty, (2, 2), (7, 7), 150.0)
    empty.handle("scroll", {"dx": 0.0, "dy": 1.0, "precise": False})
    assert CONTEXT.prefs.rok_hollow == 1
    click(empty, {"x": empty.mouse[0], "y": empty.mouse[1]})
    cells = filled()
    zs = sorted({c[2] for c in cells})
    assert len(zs) >= 3
    assert (4, 4, zs[len(zs) // 2]) not in cells  # the inside stays empty
    assert (2, 4, zs[len(zs) // 2]) in cells  # the wall stands


def test_escape_during_the_drag_drops_the_box(empty):
    key(empty, "b")
    drag(empty, [on_floor(empty, 2, 2), on_floor(empty, 5, 4)], release=False)
    assert state.preview is not None
    key(empty, "Escape")
    empty.handle("mouse_up", {"button": "left"})
    assert filled() == set() and state.preview is None and not state.stroke_active
    assert modal_names(empty) == ["rok.voxel_mode", "rok.nav_cube"]


def test_escape_in_the_height_stage_drops_the_box(empty):
    key(empty, "b")
    _box(empty, (2, 2), (5, 4), 80.0)
    key(empty, "Escape")
    assert filled() == set() and not state.stroke_active
    assert not document.undo.can_undo  # no record was pushed


def test_center_grows_the_footprint_from_the_press(empty):
    key(empty, "c")
    assert CONTEXT.prefs.rok_mode == "CENTER"
    _box(empty, (7, 7), (9, 8), 60.0)
    click(empty, {"x": empty.mouse[0], "y": empty.mouse[1]})
    xs = sorted({c[0] for c in filled()})
    ys = sorted({c[1] for c in filled()})
    assert xs == [5, 6, 7, 8, 9] and ys == [6, 7, 8]


def test_line_draws_a_straight_row(empty):
    key(empty, "l")
    drag(empty, [on_floor(empty, 2, 3), on_floor(empty, 5, 4), on_floor(empty, 9, 3)])
    assert filled() == {(x, 3, 0) for x in range(2, 10)}


def test_face_extrudes_the_face_under_the_mouse(empty):
    put([(x, y, 0) for x in range(4, 8) for y in range(4, 8)])
    key(empty, "f")
    click(empty, on_top(empty, (5, 5, 0)))
    assert {(x, y, 1) for x in range(4, 8) for y in range(4, 8)} <= filled()
    assert len(filled()) == 32


def test_pattern_stamps_the_clipboard_and_r_turns_it(empty):
    box = np.zeros((2, 1, 1), dtype=np.uint8)
    box[:, 0, 0] = 3
    state.clipboard = Clip(voxels=box, origin=(0, 0, 0), palette=document.scene.palette.copy(),
                           materials=MaterialTable())
    key(empty, "p")
    assert CONTEXT.prefs.rok_mode == "PATTERN"
    key(empty, "r")  # turns the pattern; the tool stays Attach
    assert CONTEXT.prefs.rok_tool == "ATTACH"
    assert state.pattern_rotation != (1, 0, 0, 0, 1, 0, 0, 0, 1)
    click(empty, on_floor(empty, 6, 6))
    cells = filled()
    assert len(cells) == 2 and len({c[0] for c in cells}) == 1  # turned: along y


# --- selection ----------------------------------------------------------------------------


def test_select_starts_in_marquee_and_a_rect_over_the_view_takes_every_voxel(app):
    key(app, "z")
    wm = CONTEXT.prefs
    assert wm.rok_tool == "SELECT" and wm.rok_mode == "RECT"
    drag(app, [{"x": 120.0, "y": 2.0}, {"x": 700.0, "y": 450.0}, {"x": W - 2.0, "y": H - 2.0}])
    sel = document.selection
    assert int(sel.mask.sum()) == int(np.count_nonzero(document.grid.voxels))
    assert not document.modified  # a selection changes no saved data
    key(app, "z", ctrl=True)
    assert document.selection.is_empty()


def test_a_rect_over_part_of_the_view_takes_only_what_projects_inside(app):
    key(app, "z")
    drag(app, [{"x": 700.0, "y": 420.0}, {"x": 760.0, "y": 480.0}])
    cells = np.argwhere(document.selection.mask)
    assert 0 < len(cells) < np.count_nonzero(document.grid.voxels)
    centres = np.asarray(model_origin(document.model)) + cells + 0.5
    px = app.camera.project(centres)
    assert np.all((px[:, 0] > 690) & (px[:, 0] < 770) & (px[:, 1] > 410) & (px[:, 1] < 490))


def test_marquee_under_attach_switches_to_select_and_says_so(app):
    key(app, "m")
    wm = CONTEXT.prefs
    assert wm.rok_mode == "RECT" and wm.rok_tool == "SELECT"
    assert "Select" in (tools.fresh_note() or "")


def test_lasso_with_erase_removes_what_lies_inside_the_loop(app):
    key(app, "r")
    key(app, "o")
    wm = CONTEXT.prefs
    assert wm.rok_tool == "ERASE" and wm.rok_mode == "LASSO"
    before = np.count_nonzero(document.grid.voxels)
    drag(app, [{"x": 120.0, "y": 2.0}, {"x": W - 2.0, "y": 2.0}, {"x": W - 2.0, "y": H - 2.0}, {"x": 120.0, "y": H - 2.0}])
    assert before > 0 and np.count_nonzero(document.grid.voxels) == 0


def test_select_voxel_click_then_drag_moves_the_selection(empty):
    put([(4, 4, 0)], colour=6)
    key(empty, "z")
    key(empty, "v")
    click(empty, on_top(empty, (4, 4, 0)))
    assert document.selection.contains((4, 4, 0))
    # A drag on the selected voxel moves it along the floor.
    drag(empty, [on_top(empty, (4, 4, 0)), on_top(empty, (6, 4, 0))])
    assert filled(6) == {(6, 4, 0)}


# --- keys -------------------------------------------------------------------------------


@pytest.mark.parametrize("name, tool", [("t", "ATTACH"), ("r", "ERASE"), ("g", "PAINT"), ("z", "SELECT")])
def test_the_tool_keys(app, name, tool):
    key(app, "g")  # away from Attach first
    key(app, name)
    assert CONTEXT.prefs.rok_tool == tool


@pytest.mark.parametrize("name, mode", [("v", "VOXEL"), ("f", "FACE"), ("b", "BOX"), ("l", "LINE"),
                                        ("c", "CENTER"), ("p", "PATTERN")])
def test_the_mode_keys(app, name, mode):
    key(app, name)
    assert CONTEXT.prefs.rok_mode == mode
    assert CONTEXT.prefs.rok_tool == "ATTACH"


def test_r_is_erase_outside_pattern_mode_and_turns_nothing(app):
    key(app, "r")
    assert CONTEXT.prefs.rok_tool == "ERASE"
    assert state.pattern_rotation == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_a_tapped_view_key_turns_the_view_one_step(app):
    yaw = app.camera.yaw
    key(app, "d")
    assert math.degrees(app.camera.yaw - yaw) == pytest.approx(15.0, abs=1.0)
    assert modal_names(app) == ["rok.voxel_mode", "rok.nav_cube"] and not app.timers
    distance = app.camera.distance
    key(app, "e")
    assert app.camera.distance < distance


def test_saved_views_come_back_from_the_view_menu(app):
    """Saved views live in View > Saved Views; the keys save selections."""
    menus = dict(app.host.named("set_menu")[-1][1][0])
    saved = next(item[1] for item in menus["View"] if item is not None and item[0] == "Saved Views")
    assert menus["View"][4][0] == "Saved Views"  # after Customize, the Animation switch and Show History
    ids = [i[0] for i in saved if i]
    assert ids == [f"rok.view_save(slot={s})" for s in (1, 2, 3, 4)] + [f"rok.view_go(slot={s})" for s in (1, 2, 3, 4)]
    app.handle("menu", {"id": "rok.view_save(slot=2)"})
    yaw, target = app.camera.yaw, app.camera.target.copy()
    key(app, "a")
    app.camera.target = target + 5.0
    assert app.camera.yaw != yaw
    app.handle("menu", {"id": "rok.view_go(slot=2)"})
    assert app.camera.yaw == pytest.approx(yaw) and np.allclose(app.camera.target, target)


@pytest.mark.parametrize("save", [{"ctrl": True}, {"meta": True}])
def test_ctrl_number_saves_the_selection_and_the_number_brings_it_back(empty, save):
    put([(3, 4, 0), (4, 4, 0), (5, 4, 0)])
    opkit.call("rok.deselect")
    document.selection.add_cells(np.array([(3, 4, 0), (4, 4, 0)]))
    key(empty, "3", **save)
    assert "Selection 3 saved" in empty.message
    opkit.call("rok.deselect")
    assert document.selection.is_empty()
    yaw = empty.camera.yaw
    key(empty, "3")
    assert {tuple(c) for c in document.selection.cells()} == {(3, 4, 0), (4, 4, 0)}
    assert empty.camera.yaw == yaw  # the view stays
    opkit.call("rok.undo")  # the recall is one undo step
    assert document.selection.is_empty()
    key(empty, "4")
    assert "Selection 4 is empty" in empty.message


def test_a_saved_selection_belongs_to_its_model_and_drops_erased_voxels(empty):
    put([(3, 4, 0), (4, 4, 0)])
    opkit.call("rok.deselect")
    document.selection.add_cells(np.array([(3, 4, 0), (4, 4, 0)]))
    key(empty, "1", ctrl=True)
    document.grid.voxels[4, 4, 0] = 0  # erased since
    opkit.call("rok.deselect")
    key(empty, "1")
    assert {tuple(c) for c in document.selection.cells()} == {(3, 4, 0)}
    other = (document.active_uid, 1)
    assert other in document.saved_selections
    opkit.call("rok.new", "EXEC", size=(8, 8, 8))
    assert document.saved_selections == {}  # a new file starts without them


def test_holding_ctrl_orbits_without_a_button(app):  # Cmd and Option: test_app_navigation.py
    app.handle("mouse_move", {"x": 700.0, "y": 400.0})
    mods(app, ctrl=True)
    yaw = app.camera.yaw
    app.handle("mouse_move", {"x": 760.0, "y": 400.0})
    assert app.camera.yaw != yaw
    mods(app)
    assert modal_names(app) == ["rok.voxel_mode", "rok.nav_cube"]


# --- the overlay and the strip ----------------------------------------------------------


def _layer(app, name):
    calls = [c for c in app.host.named("set_draw") if c[1][0] == name]
    return calls[-1][2] if calls else None


def test_the_overlay_draws_the_hover_and_the_strip(empty):
    empty.handle("mouse_move", on_floor(empty, 3, 4))
    view = _layer(empty, "overlay.view")
    pixel = _layer(empty, "overlay.pixel")
    assert view is not None and len(view["lines3_depth"]) > 0
    texts = [t[4] for t in pixel["texts"]]
    assert any("16 x 16 x 8" in t for t in texts) and "Attach" in texts
    assert not overlay._disabled


def test_a_rect_drag_draws_its_dashed_outline(app):
    key(app, "z")
    drag(app, [{"x": 600.0, "y": 300.0}, {"x": 800.0, "y": 500.0}], release=False)
    pixel = _layer(app, "overlay.pixel")
    assert len(pixel["lines2"]) > 0
    app.handle("mouse_up", {"button": "left"})


def test_a_swatch_click_picks_its_colour_and_edits_nothing(app):
    layout = overlay.strip_layout(CONTEXT.region)
    index, (x, y, w, h) = layout.swatches[-1]  # an unused colour, maybe without a slot
    ident = overlay.entry_of_key(index)
    voxels = document.grid.voxels.copy()
    click(app, {"x": x + w / 2, "y": H - (y + h / 2)})  # region pixels have y up
    assert document.scene.colours.id_of(document.colour) == ident
    assert np.array_equal(document.grid.voxels, voxels) and not document.modified


def test_a_cell_of_the_open_256_grid_picks_its_colour(app):
    CONTEXT.prefs.rok_show_palette_grid = True
    grid = overlay.palette_layout(CONTEXT.region)
    cx, cy, cw, ch = hud.cell_rect(grid, 200)
    click(app, {"x": cx + cw / 2, "y": H - (cy + ch / 2)})
    assert document.colour == 200 and not document.modified


def test_the_grid_size_label_is_a_hover_target(app):
    parts = state.label_parts[1]
    x, y, w, h = parts["GRID_SIZE"]
    app.handle("mouse_move", {"x": x + w / 2, "y": H - (y + h / 2)})
    assert state.hover_strip == ("GRID_SIZE", 0) and state.hover is None


def test_a_selection_draws_the_cage_pills_and_the_corner_gizmo(app):
    key(app, "z")
    drag(app, [{"x": 120.0, "y": 2.0}, {"x": W - 2.0, "y": H - 2.0}])
    app.redraw()
    assert state.cage_screen is not None and state.cage_screen.strip is not None
    assert state.corner is not None


def _fake_op(monkeypatch, idname: str, calls: list) -> None:
    """A stand-in for an operator another module owns, for this test only."""

    class Fake(opkit.Action):
        command_id = idname

        def execute(self, context):
            calls.append(idname)
            return {"FINISHED"}

    monkeypatch.setitem(opkit.REGISTRY, idname, Fake)


def test_a_click_on_the_grid_size_opens_the_size_field(app, monkeypatch):
    calls: list = []
    _fake_op(monkeypatch, "rok.grid_size_type", calls)
    x, y, w, h = state.label_parts[1]["GRID_SIZE"]
    voxels = document.grid.voxels.copy()
    click(app, {"x": x + w / 2, "y": H - (y + h / 2)})
    assert calls == ["rok.grid_size_type"]
    assert np.array_equal(document.grid.voxels, voxels)


def test_the_typed_size_draws_as_a_field(app):
    state.size_edit, state.size_edit_fresh = "40 x 40 x 20", False
    app.redraw()
    texts = [t[4] for t in _layer(app, "overlay.pixel")["texts"]]
    assert "40 x 40 x 20" in texts


def _press_pill(app, name: str) -> None:
    """Click the bar's button `name`."""
    app.redraw()
    (x, y, w, h), = [rect for n, rect in state.cage_screen.strip.pills if n == name]
    click(app, {"x": x + w / 2, "y": H - (y + h / 2)})


def test_a_cage_pill_runs_its_command(app, monkeypatch):
    calls: list = []
    _fake_op(monkeypatch, "rok.model_split", calls)
    key(app, "z")
    drag(app, [{"x": 120.0, "y": 2.0}, {"x": W - 2.0, "y": H - 2.0}])
    app.redraw()
    assert [n for n, _r in state.cage_screen.strip.pills][:2] == ["PAINT", "MASK"]  # the bar is always there
    selected = int(document.selection.mask.sum())
    _press_pill(app, "SPLIT")
    assert calls == ["rok.model_split"]
    assert int(document.selection.mask.sum()) == selected  # the press started no stroke


def test_the_bevel_and_smooth_buttons_change_the_selection(empty):
    from rok.core import morph

    cells = [(x, y, z) for x in range(4, 7) for y in range(4, 7) for z in range(3)]
    put(cells, colour=5)
    key(empty, "z")
    _select(cells)
    before = document.grid.voxels.copy()
    limit = document.selection.mask.copy()
    _press_pill(empty, "BEVEL")
    expected = before.copy()
    expected[morph.bevel_mask(before != 0, limit=limit)] = 0
    assert np.array_equal(document.grid.voxels, expected)
    beveled = document.grid.voxels.copy()
    kept = document.selection.mask.copy()
    _press_pill(empty, "SMOOTH")
    assert np.array_equal(document.grid.voxels, morph.smooth(beveled, limit=kept))


def test_every_bar_button_has_its_command(empty):
    from rok.app import tools as tools_mod
    from rok.core import cage

    for name in cage.GLYPHS:
        op = tools_mod._CAGE_GLYPH_OPS[name]
        assert f"rok.{op}" in opkit.REGISTRY, name


def test_the_grow_and_shrink_buttons_change_the_selection_not_the_voxels(empty):
    cells = [(x, y, 0) for x in range(2, 8) for y in range(2, 8)]
    put(cells, colour=5)
    key(empty, "z")
    _select([(x, y, 0) for x in range(4, 6) for y in range(4, 6)])
    voxels = document.grid.voxels.copy()
    before = int(document.selection.mask.sum())
    _press_pill(empty, "SEL_GROW")
    grown = int(document.selection.mask.sum())
    assert grown > before
    _press_pill(empty, "SEL_SHRINK")
    assert int(document.selection.mask.sum()) < grown
    assert np.array_equal(document.grid.voxels, voxels)  # no voxel was added or taken


def test_the_invert_pill_selects_what_was_not_selected_and_a_second_click_goes_back(empty):
    app = empty
    put([(3, 4, 0), (4, 4, 0), (5, 4, 0)], colour=5)
    key(app, "z")
    _select([(4, 4, 0)])
    app.redraw()
    filled_all = document.grid.visible != 0
    before = document.selection.mask.copy()

    def press():
        _press_pill(app, "INVERT")

    press()
    assert np.array_equal(document.selection.mask, filled_all & ~before)  # the rest of the voxels
    press()
    assert np.array_equal(document.selection.mask, before)
    key(app, "z", ctrl=True)
    assert np.array_equal(document.selection.mask, filled_all & ~before)  # one undo step per click


def test_the_corner_gizmo_moves_the_selection(empty):
    put([(4, 4, 0)], colour=6)
    key(empty, "z")
    key(empty, "v")
    click(empty, on_top(empty, (4, 4, 0)))
    empty.redraw()
    _pointer, layout, cell_px = state.corner
    tx, ty = layout.tips[0]
    cx, cy = cell_px[0]
    # Backwards along the arrow: the corner sits near the right edge, where a
    # drag forwards would warp the cursor round (an endless drag).
    points = [{"x": tx + cx * k, "y": H - (ty + cy * k)} for k in (0.0, -1.0, -2.0, -3.0)]
    drag(empty, points)
    assert filled(6) == {(1, 4, 0)}
    key(empty, "z", ctrl=True)
    assert filled(6) == {(4, 4, 0)}


def test_x_during_a_selection_move_locks_it_to_x(empty):
    put([(4, 4, 0)], colour=6)
    key(empty, "z")
    key(empty, "v")
    click(empty, on_top(empty, (4, 4, 0)))
    drag(empty, [on_top(empty, (4, 4, 0)), on_top(empty, (5, 5, 0))], release=False)
    key(empty, "x")  # the lock: the move keeps to the x axis
    empty.handle("mouse_move", on_top(empty, (6, 7, 0)))
    empty.handle("mouse_up", {"button": "left"})
    cells = filled(6)
    assert len(cells) == 1 and next(iter(cells))[1:] == (4, 0)


def test_connected_mode_selects_the_blob_under_the_click(empty):
    blob = [(2, 2, 0), (3, 2, 0), (3, 3, 0)]
    put(blob)
    put([(8, 8, 0)])
    key(empty, "z")
    opkit.call("rok.set_mode", mode="CONNECTED")
    empty.handle("mouse_move", on_top(empty, (3, 2, 0)))
    assert state.hover.cells is not None and len(state.hover.cells) == 3
    click(empty, on_top(empty, (3, 2, 0)))
    assert {tuple(int(v) for v in c) for c in document.selection.cells()} == set(blob)


# --- the scene level ------------------------------------------------------------------


def _centre_of(app, model) -> dict:
    o = np.asarray(model_origin(model), dtype=np.float64)
    return at(app, o + np.asarray(model.grid.size, dtype=np.float64) / 2.0)


def _top_voxel_of(app, model) -> dict:
    """A point on the top face of a filled voxel far from the model's centre."""
    filled = np.argwhere(model.grid.voxels != 0)
    centre = np.asarray(model.grid.size) / 2.0
    far = filled[np.argmax(np.linalg.norm((filled - centre)[:, :2], axis=1) - 0.01 * filled[:, 2])]
    o = np.asarray(model_origin(model), dtype=np.float64)
    column = model.grid.voxels[far[0], far[1]]
    top = int(np.nonzero(column)[0].max())
    return at(app, o + np.array([far[0] + 0.5, far[1] + 0.5, top + 1.0]))


def test_in_the_scene_a_click_picks_a_model_and_a_drag_moves_it(app):
    opkit.call("rok.open", "EXEC", filepath=str(FIXTURES / "phase3" / "two_models.vox"))
    assert state.level == "SCENE"
    first, second = document.scene.models
    document.set_active(first.uid)
    voxels = [m.grid.voxels.copy() for m in document.scene.models]
    app.handle("mouse_move", _centre_of(app, second))
    assert state.hover_model == second.uid and state.hover is None
    click(app, _centre_of(app, second))
    assert document.active_uid == second.uid
    before = tuple(second.position)
    # Grab the hull away from the centre: the picked model's gizmo sits there,
    # and its centre sphere stretches the model.
    p = _top_voxel_of(app, second)
    drag(app, [p, {"x": p["x"] + 40.0, "y": p["y"]}, {"x": p["x"] + 120.0, "y": p["y"]}])
    assert tuple(second.position) != before
    # The scene moves whole models: no voxel changed.
    assert all(np.array_equal(m.grid.voxels, v) for m, v in zip(document.scene.models, voxels))


def test_in_the_scene_a_ring_drag_turns_the_real_voxels_live(app):
    from rok.app.state import gizmo_screen_for

    opkit.call("rok.open", "EXEC", filepath=str(FIXTURES / "phase3" / "two_models.vox"))
    first, second = document.scene.models
    click(app, _centre_of(app, second))
    assert document.active_uid == second.uid
    app.redraw()
    ring = np.asarray(gizmo_screen_for(app.region).rings["ROT_Z"], dtype=np.float64)
    host = [{"x": float(x), "y": float(app.height - y)} for x, y in ring[:6]]
    before = second.grid.voxels.copy()
    drag(app, host, release=False)
    assert not state.turn_preview  # no stand-in on the GPU
    live = second.grid.voxels.copy()
    assert live.shape != before.shape  # the real voxels turn during the drag
    assert app.host.named("set_model")  # and go to the GPU at once
    app.handle("mouse_up", {"button": "left"})
    assert np.array_equal(second.grid.voxels, live)  # the release keeps what was shown
    opkit.call("rok.undo", "EXEC")
    assert np.array_equal(document.scene.by_uid(second.uid).grid.voxels, before)


# --- Marquee rounding -------------------------------------------------------------------


def _marquee(app, rounds: int, key_name: str = "r", tool: str = "SELECT", **on) -> int:
    """Drag a marquee over the middle of the model, press `key_name` `rounds` times, let go: cells picked.

    The marquee tests open the car: it fills the corners of its box, so a
    rounded corner leaves out cells."""
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = tool, "RECT"
    opkit.call("rok.deselect")
    cx, cy = (float(v) for v in app.camera.project([app.camera.target])[0])  # the model's middle on screen
    a, b = {"x": cx - W * 0.2, "y": cy - H * 0.25}, {"x": cx + W * 0.2, "y": cy + H * 0.25}
    drag(app, [a, {"x": cx, "y": cy}, b], release=False)
    for _ in range(rounds):
        key(app, key_name, **on)
    app.handle("mouse_up", {"button": "left"})
    return document.selection.count() if document.selection is not None else 0


def test_r_during_a_marquee_rounds_it(app):
    opkit.call("rok.open", "EXEC", filepath=str(CAR))
    plain = _marquee(app, 0)
    top = _marquee(app, 1)
    ends = _marquee(app, 2)
    ellipse = _marquee(app, 3)
    assert plain > top > ends >= ellipse > 0
    assert _marquee(app, 4) == plain  # R goes round: back to flat
    assert _marquee(app, 1, "r", shift=True) == ellipse  # Shift+R steps back


def test_up_and_down_round_the_marquee_too(app):
    opkit.call("rok.open", "EXEC", filepath=str(CAR))
    assert _marquee(app, 2, "ArrowUp") == _marquee(app, 2)
    assert _marquee(app, 5, "ArrowUp") == _marquee(app, 3)  # Up stops at the ellipse


@pytest.mark.parametrize("tool", ["ERASE", "PAINT"])
def test_erase_and_paint_take_a_rounded_marquee_too(app, tool):
    """A marquee for Erase and Paint that works as Select's."""
    picked = {}
    for rounds in (0, 3):
        document.reset()
        state.__init__()
        opkit.call("rok.open", "EXEC", filepath=str(CAR))
        before = document.grid.voxels.copy()
        document.colour = 7
        _marquee(app, 0, tool="SELECT")  # what the same drag selects, flat
        opkit.call("rok.deselect")
        _marquee(app, rounds, tool=tool)
        after = document.grid.voxels
        changed = (after != before).sum()
        assert changed > 0
        if tool == "ERASE":
            assert ((after == 0) | (after == before)).all()
        else:
            assert (after[after != before] == 7).all()
        picked[rounds] = changed
    assert picked[3] < picked[0]  # the ellipse takes less than the rectangle


# --- paint and the selection ------------------------------------------


def _select(cells) -> None:
    opkit.call("rok.deselect")
    if document.selection is not None:
        document.selection.add_cells(np.asarray(cells))


def test_paint_selection_colours_every_selected_voxel_in_one_undo(empty):
    put([(3, 4, 0), (4, 4, 0), (5, 4, 0)], colour=5)
    _select([(3, 4, 0), (4, 4, 0)])
    document.colour = 9
    depth = len(document.undo._undo)
    key(empty, "p", shift=True)  # Shift+P
    assert filled(9) == {(3, 4, 0), (4, 4, 0)} and filled(5) == {(5, 4, 0)}
    assert len(document.undo._undo) == depth + 1
    opkit.call("rok.undo")
    assert filled(5) == {(3, 4, 0), (4, 4, 0), (5, 4, 0)}


def test_paint_selection_without_a_selection_says_so(empty):
    put([(3, 4, 0)], colour=5)
    opkit.call("rok.deselect")
    assert opkit.call("rok.paint_selection") == {"CANCELLED"}
    assert empty.message == "Select something first"


def test_the_paint_pill_runs_paint_selection():
    from rok.app import tools as tools_mod
    from rok.core import cage

    assert "PAINT" in cage.GLYPHS and tools_mod._CAGE_GLYPH_OPS["PAINT"] == "paint_selection"
    assert cage.pill_hud("PAINT")[0] == "Paint"


def test_paint_stays_inside_the_selection(empty):
    put([(3, 4, 0), (4, 4, 0)], colour=5)
    _select([(4, 4, 0)])
    key(empty, "g")
    document.colour = 9
    click(empty, on_top(empty, (3, 4, 0)))  # outside the selection: nothing
    assert filled(9) == set()
    assert tools.INSIDE_NOTE in (empty.message or "")
    click(empty, on_top(empty, (4, 4, 0)))  # inside: painted
    assert filled(9) == {(4, 4, 0)}


def test_paint_without_a_selection_paints_anywhere(empty):
    put([(3, 4, 0)], colour=5)
    opkit.call("rok.deselect")
    key(empty, "g")
    document.colour = 9
    click(empty, on_top(empty, (3, 4, 0)))
    assert filled(9) == {(3, 4, 0)}


# --- the pattern's negative space ---------------------------------------


def _pattern_from(cells) -> None:
    """Copy these voxels as the pattern, then clear the grid but for a 3-thick floor."""
    put(cells, colour=6)
    opkit.call("rok.deselect")
    document.selection.add_cells(np.array(cells))
    assert opkit.call("rok.copy") == {"FINISHED"}
    opkit.call("rok.deselect")
    document.grid.voxels[:] = 0
    put([(x, y, z) for x in range(16) for y in range(16) for z in range(3)], colour=2)


def test_n_and_the_rail_switch_step_through_add_carve_and_holes(empty):
    from rok.app import chrome
    from rok.core import chrome as kit

    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "ATTACH", "PATTERN"
    assert wm.rok_pattern_space == "ADD"
    key(empty, "n")
    assert wm.rok_pattern_space == "CARVE"
    key(empty, "n", shift=True)
    assert wm.rok_pattern_space == "ADD"
    region = CONTEXT.region
    lay = chrome.layout_for(region, CONTEXT)
    assert lay.by_id(kit.STAMP_ID) is not None
    chrome._run(CONTEXT, kit.STAMP_ID, False, lay, region)
    chrome._run(CONTEXT, kit.STAMP_ID, False, lay, region)
    assert wm.rok_pattern_space == "HOLES"
    wm.rok_pattern_space = "ADD"


def test_carve_cuts_the_patterns_shape_out(empty):
    _pattern_from([(1, 1, 3), (2, 1, 3), (1, 2, 3)])  # an L of three
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_pattern_space = "ATTACH", "PATTERN", "CARVE"
    try:
        before = (document.grid.voxels != 0).sum()
        click(empty, on_top(empty, (8, 8, 2)))
        assert (document.grid.voxels != 0).sum() == before - 3  # the L, carved
        assert (document.grid.voxels[:, :, 3:] == 0).all()  # nothing added
    finally:
        wm.rok_pattern_space = "ADD"


def test_holes_cut_the_patterns_empty_cells(empty):
    ring = [(x, y, 3) for x in range(1, 4) for y in range(1, 4) if (x, y) != (2, 2)]
    _pattern_from(ring)  # a frame with an empty middle, one cell thick
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_pattern_space = "ATTACH", "PATTERN", "HOLES"
    try:
        before = (document.grid.voxels != 0).sum()
        click(empty, on_top(empty, (8, 8, 2)))
        assert (document.grid.voxels != 0).sum() == before - 1  # the opening, in the top layer
        assert (document.grid.voxels[:, :, 3:] == 0).all()  # the frame sank in: nothing on top
    finally:
        wm.rok_pattern_space = "ADD"


# --- the mask -------------------------------------------------------------


def _mask(cells) -> None:
    opkit.call("rok.deselect")
    document.selection.add_cells(np.array(cells))
    assert opkit.call("rok.mask_selection") == {"FINISHED"}
    assert document.selection.is_empty()  # the mask is not a selection


def test_masked_voxels_take_no_erase_or_paint(empty):
    put([(3, 4, 0), (4, 4, 0)], colour=5)
    _mask([(3, 4, 0)])
    key(empty, "g")  # Paint
    document.colour = 9
    click(empty, on_top(empty, (3, 4, 0)))
    assert filled(9) == set()
    from rok.app import tools as tools_mod

    assert tools_mod.MASK_NOTE in (empty.message or "")
    click(empty, on_top(empty, (4, 4, 0)))
    assert filled(9) == {(4, 4, 0)}
    click(empty, on_top(empty, (3, 4, 0)), alt=True)  # Option: erase
    assert (3, 4, 0) in filled()


def test_nothing_attaches_onto_a_masked_voxel(empty):
    put([(3, 4, 0), (4, 4, 0)], colour=5)
    _mask([(3, 4, 0)])
    click(empty, on_top(empty, (3, 4, 0)))  # Attach onto the masked one: nothing
    assert filled() == {(3, 4, 0), (4, 4, 0)}
    click(empty, on_top(empty, (4, 4, 0)))  # onto the other: built
    assert (4, 4, 1) in filled()


def test_clear_mask_and_the_keys(empty):
    put([(3, 4, 0)], colour=5)
    opkit.call("rok.deselect")
    document.selection.add_cells(np.array([(3, 4, 0)]))
    key(empty, "m", shift=True)  # Shift+M masks
    from rok.app import select_ops

    assert select_ops.mask_of() is not None
    from rok.app import chrome
    from rok.core import chrome as kit

    lay = chrome.layout_for(CONTEXT.region, CONTEXT)
    chip = lay.by_id(kit.MASK_ID)
    assert chip is not None and chip.glyph == "MASK"  # the rail shows the mask
    key(empty, "m", alt=True)  # Option+M clears
    assert select_ops.mask_of() is None
    assert chrome.layout_for(CONTEXT.region, CONTEXT).by_id(kit.MASK_ID) is None
    opkit.call("rok.new", "EXEC", size=(8, 8, 8))
    assert select_ops.mask_of() is None


def test_the_mask_chips_on_the_strip():
    from rok.app import tools as tools_mod
    from rok.core import cage, star
    from pathlib import Path

    assert "MASK" in cage.GLYPHS and tools_mod._CAGE_GLYPH_OPS["MASK"] == "mask_selection"
    icons = Path(__file__).resolve().parents[1] / "assets" / "icons" / "chrome" / "1x"
    assert (icons / "MASK.png").exists() and (icons / "MASK@2x.png").exists()  # the bandit's mask
    assert not any(r.id == "SELECTION" for r in star.model_rays())  # the star has no Selection arm


def test_paint_selection_leaves_masked_voxels(empty):
    put([(3, 4, 0), (4, 4, 0)], colour=5)
    _mask([(3, 4, 0)])
    document.selection.add_cells(np.array([(3, 4, 0), (4, 4, 0)]))
    document.colour = 9
    key(empty, "p", shift=True)
    assert filled(9) == {(4, 4, 0)} and filled(5) == {(3, 4, 0)}


# --- a big selection stays fast -------------------------------------------


def _skin_calls(app, kind="set_draw"):
    return [c for c in app.host.calls if c[0] == kind and str(c[1][0] if c[1] else c[2].get("layer", "")).startswith("overlay.selection/")]


def test_dirty_blocks_are_those_of_the_changed_cells_and_their_neighbours():
    """The skin's block diff sorted millions of rows for a
    live turn; now mask arithmetic, with the same blocks as before."""
    rng = np.random.default_rng(5)
    changed = np.zeros((70, 50, 33), dtype=bool)
    cells = rng.integers(0, [70, 50, 33], size=(300, 3))
    changed[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    changed[31:34, 0, 32] = True  # across a block seam and on the grid's edge
    near = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
    cells = np.argwhere(changed)
    around = (cells[:, None, :] + near[None, :, :]).reshape(-1, 3)
    around = around[((around >= 0) & (around < changed.shape)).all(axis=1)]
    expected = {tuple(b) for b in (around // overlay.BLOCK).tolist()}
    assert overlay._dirty_blocks(changed) == expected
    assert overlay._dirty_blocks(np.zeros((4, 4, 4), dtype=bool)) == set()


def test_the_direct_skin_path_draws_the_same_colours_and_points_as_the_view_path():
    """A skin's geometry goes to the GPU in grid space with
    the model's origin as the layer offset; the colours still take gfx's sRGB
    step, as gfx.tris gives it to the view path."""
    from rok.app import draw, gfx

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    tris, lines = hud.mask_surface(mask, (0, 0, 0), (4, 4, 4))
    origin = (10.0, 20.0, 30.0)
    direct = draw.DrawList()
    overlay._direct(direct, origin, tris, (0.2, 0.4, 0.6, 0.5), lines, (0.2, 0.4, 0.6, 0.9), 1.5)
    view = draw.DrawList()
    with gfx.view(view), overlay._space(origin, None, np.zeros(3)):
        overlay._tris(tris, (0.2, 0.4, 0.6, 0.5), bias=False)
        overlay._lines(lines, (0.2, 0.4, 0.6, 0.9), 1.5, bias=False)
    a, b = direct.arrays(), view.arrays()
    t_direct = a["tris3_depth"].reshape(-1, 7)
    t_view = b["tris3_depth"].reshape(-1, 7)
    assert np.allclose(t_direct[:, :3] + np.asarray(origin, dtype=np.float32), t_view[:, :3], atol=1e-4)
    assert np.array_equal(t_direct[:, 3:], t_view[:, 3:])  # the same sRGB colours
    l_direct = a["lines3_depth"].reshape(-1, 11)
    l_view = b["lines3_depth"].reshape(-1, 11)
    assert np.array_equal(l_direct[:, 6:], l_view[:, 6:])  # colour and width
    assert direct.offset == origin and view.offset == (0.0, 0.0, 0.0)


def test_a_layers_offset_follows_its_content_to_the_host():
    """The host makes a new layer when its content first arrives,
    with no offset, so an offset sent before the content went nowhere: the
    skin blocks and the Box preview drew at the grid's zero corner."""
    from rok.app import draw

    host = RecordingHost()
    layers = draw.Layers(host)
    d = draw.DrawList()
    d.tris3(np.zeros((3, 3), np.float32), (1.0, 0.0, 0.0, 1.0))
    d.key = ("skin", 1)
    d.offset = (-24.0, -24.0, 0.0)
    assert layers.send("overlay.selection/0.0.0", 5, d)
    names = [c[0] for c in host.calls if c[0] in ("set_draw", "set_draw_offset")]
    assert names == ["set_draw", "set_draw_offset"]
    # The same content again: nothing sent; a new offset alone: only the offset.
    assert not layers.send("overlay.selection/0.0.0", 5, d)
    d.offset = (-30.0, -24.0, 0.0)
    assert not layers.send("overlay.selection/0.0.0", 5, d)
    names = [c[0] for c in host.calls if c[0] in ("set_draw", "set_draw_offset")]
    assert names == ["set_draw", "set_draw_offset", "set_draw_offset"]
    # New content for a shifted layer: the offset comes again, after it.
    d.key = ("skin", 2)
    assert layers.send("overlay.selection/0.0.0", 5, d)
    names = [c[0] for c in host.calls if c[0] in ("set_draw", "set_draw_offset")]
    assert names[-2:] == ["set_draw", "set_draw_offset"]


def test_a_big_selection_is_sent_once_and_then_only_where_it_changed(app):
    opkit.call("rok.select_all")
    app.redraw()
    first = len(_skin_calls(app))
    assert first > 1  # in blocks
    app.redraw()
    app.handle("mouse_move", {"x": 700.0, "y": 420.0})
    app.redraw()
    assert len(_skin_calls(app)) == first  # nothing changed: nothing sent
    cell = tuple(int(v) for v in document.selection.cells()[0])
    document.selection.remove_cells(np.array([cell]))
    app.redraw()
    assert 1 <= len(_skin_calls(app)) - first <= 8  # the block of that cell, and maybe its neighbours
    opkit.call("rok.deselect")
    app.redraw()
    assert _skin_calls(app, "clear_draw")  # the blocks go away


def test_the_moving_copy_is_shifted_on_the_gpu(app):
    from rok.core.gesture import Preview

    opkit.call("rok.select_all")
    app.redraw()
    state.preview = Preview(kind="MOVE", tool="SELECT", offset=(2, 0, 0))
    try:
        app.redraw()
        sent = len([c for c in app.host.calls if c[0] == "set_draw" and c[1] and c[1][0] == "overlay.move"])
        state.preview = Preview(kind="MOVE", tool="SELECT", offset=(5, 0, 0))
        app.redraw()
        again = len([c for c in app.host.calls if c[0] == "set_draw" and c[1] and c[1][0] == "overlay.move"])
        offsets = [c for c in app.host.calls if c[0] == "set_draw_offset" and c[1][0] == "overlay.move"]
    finally:
        state.preview = None
    assert sent == again  # the copy was not sent again
    assert offsets[-1][1][1:] == (5.0, 0.0, 0.0)  # only its shift changed


def _allow(mask, voxels, normal):
    from rok.app.tools import _not_onto

    return _not_onto(mask, voxels, lambda: normal)


def test_the_mask_stops_attach_from_every_side():
    """The mask stopped Attach only along one direction."""
    v = np.zeros((10, 10, 10), np.uint8)
    v[:, :, 0] = 2  # an unmasked floor
    v[5, 5, 1:3] = 3  # a masked pillar, two high
    mask = v == 3
    allow = _allow(mask, v, (0, 0, 1))  # a press on the floor's top face
    cells = np.array([(4, 5, 1), (6, 5, 1), (5, 4, 1), (5, 6, 2), (2, 2, 1), (5, 5, 3)])
    ok = allow(cells)
    assert not ok[:4].any()  # beside the pillar, on any side: not built
    assert ok[4]  # away from it: built
    assert not ok[5]  # on top of it: not built


def test_a_box_beside_the_mask_leaves_no_voxel_floating():
    v = np.zeros((10, 10, 10), np.uint8)
    v[:, :, 0] = 2
    v[5, 5, 1] = 3  # a masked voxel, one high
    mask = v == 3
    allow = _allow(mask, v, (0, 0, 1))
    column = np.array([(4, 5, 1), (4, 5, 2), (4, 5, 3)])  # a box's column beside it
    ok = allow(column)
    assert not ok[0]  # touches the mask
    assert not ok[1] and not ok[2]  # over a cell that is not built: nothing floats
    beyond = allow(np.array([(3, 5, 1), (3, 5, 2), (3, 5, 3)]))
    assert beyond.all()  # one cell further: the whole column


def test_attach_with_a_big_brush_never_touches_the_mask(empty):
    put([(x, y, 0) for x in range(16) for y in range(16)], colour=2)
    put([(8, 8, 1)], colour=3)
    _mask([(8, 8, 1)])
    CONTEXT.prefs.rok_brush_size = 3
    try:
        click(empty, on_top(empty, (7, 8, 0)))  # right beside the masked voxel
    finally:
        CONTEXT.prefs.rok_brush_size = 1
    new = {c for c in filled() if c[2] >= 1} - {(8, 8, 1)}
    assert new  # something was built
    for x, y, z in new:
        assert all((x + dx, y + dy, z + dz) != (8, 8, 1)
                   for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))


def test_strokes_put_their_colours_on_the_stars_colour_arm(empty):
    from rok.app import marking, palette_ui
    from rok.core import star

    assert palette_ui.recent_colours() == []  # a new file starts with none
    put([(3, 4, 0), (4, 4, 0)], colour=5)
    key(empty, "g")
    for colour in (9, 11, 9, 12, 13):
        document.colour = colour
        click(empty, on_top(empty, (3, 4, 0)))
    assert palette_ui.recent_colours() == [13, 12, 9]  # three, newest first, each once
    arm = next(r for r in star.rays_for(state.level, False, colours=marking._recent_chips()) if r.id == "COLOURS")
    assert [c.id for c in arm.chips] == [f"{star.STAR_COLOUR}{i}" for i in (13, 12, 9)]
    assert f"{star.STAR_COLOUR}13" in marking._active_ids(CONTEXT)  # the current colour shows as on
    marking._apply(CONTEXT, None, f"{star.STAR_COLOUR}9")
    assert document.colour == 9
    opkit.call("rok.new", "EXEC", size=(16, 16, 8))
    assert palette_ui.recent_colours() == []  # another file: its slots are other colours


def test_erase_stays_inside_the_selection(empty):
    put([(3, 4, 0), (4, 4, 0)], colour=5)
    _select([(4, 4, 0)])
    key(empty, "r")
    click(empty, on_top(empty, (3, 4, 0)))  # outside the selection: nothing
    assert filled() == {(3, 4, 0), (4, 4, 0)}
    assert tools.ERASE_INSIDE_NOTE in (empty.message or "")
    click(empty, on_top(empty, (4, 4, 0)))  # inside: erased
    assert filled() == {(3, 4, 0)}
    opkit.call("rok.deselect")
    click(empty, on_top(empty, (3, 4, 0)))  # nothing selected: erases anywhere
    assert filled() == set()


def _ghosts(app):
    """The ghost boxes the host holds after the last set_ghost call: (idx, [(origin, box), ...])."""
    calls = app.host.named("set_ghost")
    return (calls[-1][1][0], calls[-1][1][1]) if calls else None


def test_the_voxel_hover_shows_the_result_as_ghost_voxels(empty):
    """The hover shows what a click would do, not a red outline. The renderer gets the new voxel as a ghost and
    shades it as it shades every voxel, so the preview is the real thing."""
    document.colour = 7
    empty.handle("mouse_move", on_floor(empty, 3, 4))
    assert state.hover is not None and state.hover.target == (3, 4, 0)
    empty.redraw()
    idx, boxes = _ghosts(empty)
    assert idx == 0 and len(boxes) == 1
    origin, box = boxes[0]
    assert origin == (3, 4, 0) and box.shape == (1, 1, 1) and box[0, 0, 0] == 7
    layer = _layer(empty, "overlay.hover")
    assert layer is None or len(layer["tris3_depth"]) == 0  # nothing drawn over it
    # No outline for the voxel tools: no Attach-green line in the view layer.
    lines = np.asarray(_layer(empty, "overlay.view")["lines3_depth"]).reshape(-1, 11)  # 2 points, rgba, width
    green = overlay.TOOL_COLOURS["ATTACH"] + (overlay.EDGE_ALPHA,)
    assert not np.isclose(lines[:, 6:10], green, atol=1e-3).all(axis=1).any()
    n = len(empty.host.named("set_ghost"))
    empty.redraw()
    assert len(empty.host.named("set_ghost")) == n  # the mouse rests: nothing sent
    # Another colour: the ghost is sent again in that colour.
    document.colour = 9
    empty.redraw()
    assert _ghosts(empty)[1][0][1][0, 0, 0] == 9
    # Off the model: the ghost goes.
    empty.handle("mouse_move", {"x": 4.0, "y": empty.height / 2})
    empty.redraw()
    assert _ghosts(empty) == (0, [])


def test_a_paint_hover_ghosts_the_filled_cells_and_their_mirror_images(app):
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_brush_size = "PAINT", "VOXEL", 2
    wm.rok_mirror = (True, False, False)
    document.colour = 5
    v = document.grid.visible
    cells = np.argwhere(v != 0)
    cells = cells[np.abs(cells[:, 0] - v.shape[0] // 2) > 6]  # away from the mirror plane
    top = cells[np.argmax(cells[:, 2])]
    app.handle("mouse_move", on_top(app, top))
    assert state.hover is not None and state.hover.tool == "PAINT"
    app.redraw()
    idx, boxes = _ghosts(app)
    assert idx == 0 and len(boxes) == 2
    for origin, box in boxes:
        cells = np.argwhere(box != 0) + origin
        assert (v[cells[:, 0], cells[:, 1], cells[:, 2]] != 0).all()  # paint touches filled cells only
        assert (box[box != 0] == 5).all()
    (o1, b1), (o2, b2) = boxes
    assert o2[0] == v.shape[0] - 1 - (o1[0] + b1.shape[0] - 1)  # the image across X


def test_shaded_cells_light_the_top_and_darken_the_rest():
    """The fallback for a host without ghosts: faces in the colour, shaded per direction."""
    tris, colours = overlay._shaded_cells(np.array([[2, 3, 4]]), (8, 8, 8), (0.5, 0.2, 0.8), 0.9)
    assert tris.shape == (36, 3) and colours.shape == (36, 4)
    top = colours[np.isclose(tris[:, 2], tris[:, 2].max())]
    assert len(top) == 6 and np.allclose(top[:, :3], (0.5, 0.2, 0.8))
    assert np.allclose(colours[:, 3], 0.9) and (colours[:, :3] <= np.array([0.5, 0.2, 0.8]) + 1e-6).all()
    assert overlay._shaded_cells(np.zeros((0, 3)), (8, 8, 8), (1, 1, 1), 1.0)[0].shape == (0, 3)


def test_the_erase_hover_blanks_the_voxel_it_would_take(app):
    """Erase shows the voxel gone, not a red marker."""
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "ERASE", "VOXEL"
    cells = np.argwhere(document.grid.visible != 0)
    top = cells[np.argmax(cells[:, 2])]
    app.handle("mouse_move", on_top(app, top))
    assert state.hover is not None and state.hover.target == tuple(int(v) for v in top)
    app.redraw()
    call = app.host.named("set_ghost")[-1]
    idx, boxes = call[1][0], call[1][1]
    assert idx == 0 and call[2] == {"erase": True}
    (origin, box), = boxes
    assert origin == tuple(int(v) for v in top) and box.shape == (1, 1, 1) and box[0, 0, 0] != 0
    layer = _layer(app, "overlay.hover")
    assert len(layer["tris3_depth"]) == 0  # no red tint drawn over it
    # A red cage where the voxel stood: its 6 faces, 4 edges each.
    rows = np.asarray(layer["lines3_depth"]).reshape(-1, 11)
    red = rows[np.all(np.isclose(rows[:, 6:9], overlay.TOOL_COLOURS["ERASE"]), axis=1)]
    assert len(red) == 24
    ends = red[:, :6].reshape(-1, 3)
    assert np.allclose(ends.max(axis=0) - ends.min(axis=0), 1.0, atol=0.01)  # one cell, in world space


def test_a_bigger_brush_previews_the_whole_sphere(empty):
    CONTEXT.prefs.rok_brush_size = 2
    empty.handle("mouse_move", on_floor(empty, 6, 6))
    empty.redraw()
    idx, boxes = _ghosts(empty)
    (origin, box), = boxes
    assert origin == (5, 5, 0) and box.shape == (3, 3, 2)  # the sphere of size 2, cut at the floor
    assert (box != 0).sum() == len([c for c in edit.sphere_offsets(2) if c[2] >= 0])


def test_a_big_brush_writes_as_boxes_and_its_hover_ghosts_are_boxes(app):
    """A brush of 50 on a 256-cube took 28 ms a move; the
    sphere writes as one box per mirror image now, and so does its ghost."""
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_brush_size = "ERASE", "VOXEL", 12
    wm.rok_mirror = (True, False, False)
    v = document.grid.visible
    cells = np.argwhere(v != 0)
    cells = cells[np.abs(cells[:, 0] - v.shape[0] // 2) > 14]
    top = cells[np.argmax(cells[:, 2])]
    app.handle("mouse_move", on_top(app, top))
    app.redraw()
    idx, boxes = _ghosts(app)
    assert idx == 0 and len(boxes) == 2  # the sphere and its mirror image
    for origin, box in boxes:
        assert box.ndim == 3 and max(box.shape) <= 23  # a clipped 23-box
        inside = np.argwhere(box != 0) + origin
        assert (v[inside[:, 0], inside[:, 1], inside[:, 2]] != 0).all()  # Erase ghosts filled cells only
    before = int((v != 0).sum())
    drag(app, [on_top(app, top), on_top(app, top)])
    after = int((document.grid.visible != 0).sum())
    assert before - after > 100  # the sphere and its image are gone


def test_a_big_erase_brush_gets_a_box_not_a_cage_of_every_face(app):
    """The cage must not slow Erase down. Past ERASE_CAGE_LIMIT
    cells it is the 12 edges of the box around them."""
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_brush_size = "ERASE", "VOXEL", 30
    document.grid.voxels[...] = 5  # a solid block: the brush takes a big part of it
    document.edit_serial += 1
    size = np.asarray(document.grid.voxels.shape)
    top = (int(size[0]) // 2, int(size[1]) // 2, int(size[2]) - 1)
    app.handle("mouse_move", on_top(app, top))
    app.redraw()
    _idx, boxes = _ghosts(app)
    (origin, box), = boxes
    assert np.count_nonzero(box) > overlay.ERASE_CAGE_LIMIT
    rows = np.asarray(_layer(app, "overlay.hover")["lines3_depth"]).reshape(-1, 11)
    red = rows[np.all(np.isclose(rows[:, 6:9], overlay.TOOL_COLOURS["ERASE"]), axis=1)]
    assert len(red) == 12
    ends = red[:, :6].reshape(-1, 3)
    assert np.allclose(ends.max(axis=0) - ends.min(axis=0), box.shape, atol=0.01)  # the box around the erased cells


def test_a_box_base_previews_as_one_quad_however_big(app):
    """A ROK_PROFILE run: a Box base on a 256-cube cost 15 to
    30 ms a move as one quad per cell. A base that fills its box is one quad;
    its mirror image is one box."""
    from rok.app import gfx
    from rok.app.draw import DrawList
    from rok.core.gesture import Preview

    cells = np.stack(np.meshgrid(np.arange(10, 110), np.arange(20, 220), [0], indexing="ij"), -1).reshape(-1, 3)
    preview = Preview(kind="PLANE", tool="ATTACH", cells=cells, normal=(0, 0, 1), box=((10, 20, 0), (109, 219, 0)))
    for mirror, quads in (((False, False, False), 1), ((True, False, False), 7)):
        d = DrawList()
        with gfx.view(d):
            overlay._draw_preview(preview, None, (256, 256, 256), mirror, 1.0)
        tris = sum(len(a) for a in d.parts["tris3_depth"] + d.parts["tris3_top"])
        assert tris == quads * 6 * 7  # 6 vertices a quad, 7 floats a vertex (x, y, z, r, g, b, a)


def test_a_fast_big_cut_has_a_flat_floor(app, monkeypatch):
    """A brush-50 cut still showed a bead per stamp on its
    floor. Between two hits the brush is swept as one capsule, so the floor
    lies at one depth along the whole groove."""
    monkeypatch.setattr(tools, "STROKE_SAMPLES", 2)
    opkit.call("rok.new", "EXEC", size=(64, 64, 32))
    app.handle("resize", {"width": W, "height": H})
    put([(x, y, z) for x in range(64) for y in range(64) for z in range(24)])
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_brush_size = "ERASE", "VOXEL", 8
    drag(app, [on_top(app, (8, 32, 23)), on_top(app, (56, 32, 23))])
    v = document.grid.visible
    for y in range(26, 39):  # every row across the groove, not only its middle
        cut = (v[14:50, y, :] == 0).sum(axis=1)  # the cells cut in each column of the row
        assert cut.min() == cut.max(), (y, cut.tolist())


def test_clone_stamps_the_copied_colours_and_its_ghost_shows_them(app):
    from rok.app import pattern_ops

    opkit.call("rok.open", "EXEC", filepath=str(CAR))
    voxels = document.grid.voxels
    front = np.zeros(voxels.shape, bool)
    front[:, 2:5, :] = voxels[:, 2:5, :] != 0  # the bumper and the lights: several colours
    copied = set(np.unique(voxels[front]).tolist())
    opkit.call("rok.select_all")
    document.selection.clear()
    document.selection.add_mask(front)
    opkit.call("rok.copy")
    opkit.call("rok.deselect")
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "ATTACH", "CLONE"
    app.handle("mouse_move", {"x": W * 0.40, "y": H * 0.45})
    hover = state.hover
    assert hover is not None and hover.mode == "CLONE" and len(hover.cells) > 0
    groups = pattern_ops.hover_colours(hover.cells)
    assert groups is not None and len(groups) == len(copied)  # one ghost colour per copied colour
    before = document.grid.voxels.copy()
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})
    written = document.grid.voxels[(before == 0) & (document.grid.voxels != 0)]
    assert len(written) > 0 and set(np.unique(written).tolist()) == copied
    wm.rok_tool = "ERASE"
    assert wm.rok_mode == "PATTERN"  # Clone is Attach's


def test_esc_deselects_and_steps_out_only_on_a_new_press_a_moment_later(app, monkeypatch):
    import time as time_mod

    from rok.app import level_ops

    clock = [time_mod.monotonic()]
    monkeypatch.setattr(time_mod, "monotonic", lambda: clock[0])
    opkit.call("rok.open", "EXEC", filepath=str(CAR))
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "SELECT", "COLOUR_CONNECTED"
    drag(app, [{"x": W * 0.45, "y": H * 0.45}])
    assert document.selection.count() > 0

    def esc(repeat=False):
        app.handle("key_down", {"key": "Escape", "code": "Escape", "text": "", "repeat": repeat})

    esc()  # deselects
    for _ in range(5):
        esc(repeat=True)  # the key held down: its repeats do nothing more
    assert document.selection.count() == 0 and state.level == "MODEL"
    app.handle("key_up", {"key": "Escape", "code": "Escape", "text": "", "repeat": False})
    clock[0] += 0.1
    esc()  # a quick second press right after the deselect: still the model
    assert state.level == "MODEL"
    clock[0] += level_ops.ESC_GUARD_S
    esc()  # a new press a moment later steps out to the scene
    assert state.level == "SCENE"


def test_a_gradient_drag_paints_a_ramp_of_colours_in_one_undo_step(app):
    from rok.app import gradient_ops, settings
    from rok.core import gradient

    opkit.call("rok.open", "EXEC", filepath=str(CAR))
    settings.put(gradient_ops.STEPS_KEY, "4")
    gradient_ops.set_end((20, 200, 60))
    document.colour = int(np.flatnonzero(document.used_counts()[1:])[0]) + 1
    start_rgb = gradient_ops.current_rgb()
    before = document.grid.voxels.copy()
    palette_before = document.scene.palette.copy()
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "PAINT", "GRADIENT"
    drag(app, [{"x": W * 0.35, "y": H * 0.45}, {"x": W * 0.45, "y": H * 0.47}, {"x": W * 0.6, "y": H * 0.5}])
    voxels = document.grid.voxels
    assert (voxels != 0).sum() == (before != 0).sum()  # Paint: no voxel added or removed
    shown = {tuple(int(c) for c in document.scene.palette[v, :3]) for v in np.unique(voxels[voxels != 0])}
    assert set(gradient.ramp(start_rgb, (20, 200, 60), 4)) <= shown  # all four colours of the ramp
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)
    assert np.array_equal(document.scene.palette, palette_before)  # its colours went back with it
    wm.rok_tool = "ERASE"
    assert wm.rok_mode == "VOXEL"  # Gradient is Paint's


def test_the_tolerance_field_shows_for_select_by_colour_and_widens_the_pick(app):
    from rok.app import chrome
    from rok.core import chrome as kit

    opkit.call("rok.open", "EXEC", filepath=str(CAR))
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode = "SELECT", "BY_COLOUR"
    assert chrome.layout_for(app.region).by_id(kit.TOLERANCE_ID) is not None
    drag(app, [{"x": W * 0.45, "y": H * 0.45}])
    exact = document.selection.count()
    opkit.call("rok.deselect")
    wm.rok_colour_tolerance = 100
    drag(app, [{"x": W * 0.45, "y": H * 0.45}])
    assert document.selection.count() > exact  # close colours count too
    wm.rok_mode = "VOXEL"
    layout = chrome.layout_for(app.region)
    assert layout.by_id(kit.TOLERANCE_ID) is None and layout.by_id(kit.SIZE_ID) is not None
    wm.rok_colour_tolerance = 0


def test_steady_hand_makes_the_brush_trail_the_mouse(empty):
    """With a string the brush starts at the press and stops short of where
    the mouse ends; with 0 it reaches the end."""
    slab = [(x, y, z) for x in range(16) for y in range(16) for z in range(4)]
    put(slab)
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_brush_size = "PAINT", "VOXEL", 1
    document.colour = 9
    path = [on_top(empty, (1, 8, 3)), on_top(empty, (14, 8, 3))]
    drag(empty, path)
    assert (document.grid.visible[1:15, 8, 3] == 9).all()  # no string: the whole way
    put(slab)  # the same slab again, in its old colour
    wm.rok_steady_hand, wm.rok_steady_length = True, 120
    drag(empty, path)
    row = document.grid.visible[:, 8, 3]
    assert row[1] == 9 and row[14] != 9  # it started at the press and stopped short


def test_steady_hand_shows_its_string_during_the_stroke_only(empty):
    put([(x, y, z) for x in range(16) for y in range(16) for z in range(4)])
    wm = CONTEXT.prefs
    wm.rok_tool, wm.rok_mode, wm.rok_brush_size = "PAINT", "VOXEL", 1
    wm.rok_steady_hand, wm.rok_steady_length = True, 120
    path = [on_top(empty, (1, 8, 3)), on_top(empty, (14, 8, 3))]
    assert state.steady_line is None
    drag(empty, path, release=False)
    (bx, by), (mx, my) = state.steady_line
    assert mx == pytest.approx(path[1]["x"], abs=1.0) and abs(bx - mx) + abs(by - my) > 1.0  # brush behind the mouse
    empty.redraw()  # the string is drawn without error
    empty.handle("mouse_up", {"button": "left"})
    assert state.steady_line is None
    wm.rok_steady_hand = False
    drag(empty, path, release=False)
    assert state.steady_line is None  # off: no string
    empty.handle("mouse_up", {"button": "left"})
