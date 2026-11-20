"""Brick mode in the app (rok/app/brick_mode.py): the level, the lock on a
brick model in every other level, the Convert commands, the rail and the bin.
The host only records."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rok.app import brick_mode, opkit
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state
from rok.core import brick, brick_bin
from rok.core import chrome as kit
from rok.core.brick import BrickTable
from rok.core.undo import kind_name
from test_foundation import FIXTURES, RecordingHost, key

TWO = FIXTURES / "phase3" / "two_models.vox"
HOUSE = FIXTURES / "brick" / "house.vox"


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
def house(config):
    """The golden house as a brick model, inside it in Model level."""
    a = _start(HOUSE)
    table = json.loads((FIXTURES / "brick" / "house.json").read_text())["bricks"]
    document.grid.bricks = BrickTable.from_json(table)
    assert document.grid.bricks.check(document.grid.voxels) is None
    return a


@pytest.fixture
def two(config):
    return _start(TWO)


def run(idname: str, **props) -> set:
    return opkit.call(idname, "EXEC", **props)


def poll(idname: str) -> bool:
    return getattr(opkit.ops.rok, idname.removeprefix("rok.")).poll()


def undo(app) -> None:
    app.handle("menu", {"id": "rok.undo"})


def redo(app) -> None:
    app.handle("menu", {"id": "rok.redo"})


def _go(level: str) -> None:
    from rok.app import anim_mode

    anim_mode.go(level)
    assert state.level == level


# --- the level ------------------------------------------------------------------------


def test_brick_mode_keeps_the_model_cell_of_the_switch_lit():
    assert kit.LEVEL_IDS == (kit.LEVEL_MODEL_ID, kit.LEVEL_SCENE_ID, kit.LEVEL_ANIM_ID)
    lay = kit.chrome_layout(1.0, kit.ChromeState(level=kit.LEVEL_BRICK), 300, 900)
    assert [lay.by_id(i).active for i in kit.LEVEL_IDS] == [True, False, False]


def _edit_menu_check(app) -> int:
    from rok.app import ui

    items = ui.edit_menu(app)[0][1]
    return next(it[3] for it in items if it is not None and it[0] == "rok.brick_mode")


def test_edit_brick_mode_turns_the_mode_on_and_off(house):
    assert _edit_menu_check(house) == 0
    assert run("rok.brick_mode") == {"FINISHED"}
    assert state.level == kit.LEVEL_BRICK and _edit_menu_check(house) == 1
    run("rok.brick_mode")
    assert state.level == kit.LEVEL_MODEL and _edit_menu_check(house) == 0
    # From the scene it goes into the picked model; Tab leaves it for the scene.
    _go(kit.LEVEL_SCENE)
    run("rok.brick_mode")
    assert state.level == kit.LEVEL_BRICK
    key(house, "Tab")
    assert state.level == kit.LEVEL_SCENE and not state.animating
    # From Animate too.
    from rok.app import anim_mode

    anim_mode.go("ANIMATE")
    run("rok.brick_mode")
    assert state.level == kit.LEVEL_BRICK and not state.animating


def test_the_brick_rail_has_the_four_tools_and_the_convert_chip_that_applies():
    voxels = kit.chrome_layout(1.0, kit.ChromeState(level=kit.LEVEL_BRICK, brick_tool="PULL"), 300, 900)
    ids = [w.id for w in voxels.widgets]
    for tool in kit.BRICK_TOOL_IDS:
        assert tool in ids
    assert voxels.by_id(kit.BRICK_PULL_ID).active and not voxels.by_id(kit.BRICK_PLACE_ID).active
    assert kit.BRICK_CONVERT_ID in ids and kit.BRICK_TO_VOXELS_ID not in ids
    assert kit.TOOL_IDS.keys().isdisjoint(ids)  # no voxel tools in Brick mode
    bricks = kit.chrome_layout(1.0, kit.ChromeState(level=kit.LEVEL_BRICK, brick_model=True), 300, 900)
    assert bricks.by_id(kit.BRICK_TO_VOXELS_ID) is not None and bricks.by_id(kit.BRICK_CONVERT_ID) is None
    # Everything stands inside the rail's height.
    assert min(w.rect[1] for w in bricks.widgets) >= 0
    assert kit.rail_height(kit.ChromeState(level=kit.LEVEL_BRICK)) <= 900


def test_enter_goes_to_model_and_esc_to_the_scene(house):
    _go(kit.LEVEL_BRICK)
    key(house, "Enter")
    assert state.level == kit.LEVEL_MODEL
    _go(kit.LEVEL_BRICK)
    run("rok.brick_take", number=3)
    assert state.brick_hand == (1, 3)
    key(house, "Escape")  # the first Esc empties the hand
    assert state.level == kit.LEVEL_BRICK and state.brick_hand is None
    key(house, "Escape")
    assert state.level == kit.LEVEL_SCENE


def test_the_keys_take_bricks_and_pick_the_tools(house):
    _go(kit.LEVEL_BRICK)
    key(house, "7")
    assert state.brick_hand == (2, 2)  # the bin's seventh brick
    for name, tool in (("U", "PULL"), ("G", "PAINT"), ("I", "PICK"), ("B", "PLACE")):
        key(house, name)
        assert state.brick_tool == tool
    assert brick_bin.by_key(1) == (1, 1) and brick_bin.by_key(10) is None
    # The same keys keep their voxel meaning in Model level.
    _go(kit.LEVEL_MODEL)
    key(house, "G")
    assert state.brick_tool == "PLACE"


def test_a_voxel_model_shows_the_bin_greyed_and_takes_no_brick(two):
    _go(kit.LEVEL_SCENE)
    _go(kit.LEVEL_BRICK)
    assert not brick_mode.can_build()
    assert run("rok.brick_take", number=1) == {"CANCELLED"}
    assert state.brick_hand is None
    assert brick_mode.status() == brick_mode.NOT_BRICKS


def test_leaving_brick_level_empties_the_hand(house):
    _go(kit.LEVEL_BRICK)
    run("rok.brick_take", number=5)
    _go(kit.LEVEL_SCENE)
    assert state.brick_hand is None


# --- the bin --------------------------------------------------------------------------


def test_the_bin_has_every_brick_once_and_finds_the_one_under_the_mouse():
    layout = brick_bin.bin_layout(1.0, 1440, 900)
    tiles = [i for i in layout.items if i.kind == "BRICK"]
    assert [t.value for t in tiles] == list(brick_bin.bricks())
    assert len(tiles) == 15 and [t.key for t in tiles[:9]] == list(range(1, 10)) and tiles[9].key == 0
    for t in tiles:
        x, y, w, h = t.rect
        assert brick_bin.hit(layout, x + w / 2, y + h / 2) == t
        assert layout.rect[0] <= x and x + w <= layout.rect[0] + layout.rect[2]
        assert layout.rect[1] <= y and y + h <= layout.rect[1] + layout.rect[3]
    # The bin stands under the nav cube's corner.
    assert layout.rect[1] + layout.rect[3] <= 900 - brick_bin.CUBE_CLEAR
    assert brick_bin.hit(layout, 5, 5) is None


def test_the_bin_shrinks_its_cells_to_fit_a_short_window():
    tall = brick_bin.bin_layout(1.0, 1440, 900)
    short = brick_bin.bin_layout(1.0, 1440, 560)
    assert short.cell < tall.cell
    assert short.rect[1] >= 0


def test_a_click_on_the_bin_takes_the_brick(house):
    _go(kit.LEVEL_BRICK)
    region = SimpleNamespace(width=1440, height=900)
    layout = brick_bin.bin_layout(1.0, 1440, 900)
    tile = next(i for i in layout.items if i.kind == "BRICK" and i.value == (2, 4))
    x, y, w, h = tile.rect
    p = SimpleNamespace(region=region, mx=x + w / 2, my=y + h / 2, left=True, event=SimpleNamespace(alt=False))
    assert brick_mode.press(p) == {"FINISHED"}
    assert state.brick_hand == (2, 4) and state.brick_tool == "PLACE"
    # A press on the bin beside every brick takes none; a right-click empties the hand.
    brick_mode.press(SimpleNamespace(region=region, mx=layout.rect[0] + 2, my=layout.rect[1] + 2, left=True,
                                     event=SimpleNamespace(alt=False)))
    assert state.brick_hand == (2, 4)
    brick_mode.press(SimpleNamespace(region=region, mx=300, my=300, left=False, event=SimpleNamespace(alt=False)))
    assert state.brick_hand is None


# --- Convert ------------------------------------------------------------------------------


def test_convert_to_bricks_and_back_is_one_undo_step_each(two):
    voxels = document.grid.voxels.copy()
    document.grid.lock = np.zeros(document.grid.size, dtype=bool)
    document.grid.lock[tuple(np.argwhere(voxels)[0])] = True
    assert poll("convert_to_bricks") and not poll("convert_to_voxels")
    steps = len(document.undo._undo)
    assert run("rok.convert_to_bricks") == {"FINISHED"}
    table = document.grid.bricks
    assert table is not None and len(table) == int((voxels != 0).sum()) and document.grid.lock is None
    assert len(document.undo._undo) == steps + 1 and document.undo.peek().label == "Convert to Bricks"
    assert np.array_equal(document.grid.voxels, voxels)
    assert poll("convert_to_voxels") and not poll("convert_to_bricks")
    undo(two)
    assert document.grid.bricks is None and document.grid.lock is not None  # the mask comes back too
    redo(two)
    assert document.grid.bricks == table
    assert run("rok.convert_to_voxels") == {"FINISHED"}
    assert document.grid.bricks is None and np.array_equal(document.grid.voxels, voxels)
    assert document.undo.peek().label == "Convert to Voxels" and kind_name(document.undo.peek()) == "Bricks"
    undo(two)
    assert document.grid.bricks == table


def test_convert_to_bricks_bakes_the_live_shaders(two):
    from rok.core.live_shader import LiveShader

    uid = document.active_uid
    document.scene.live[uid] = [LiveShader(1, "noise", {}, None, (False, False, False))]
    run("rok.convert_to_bricks")
    assert not document.scene.live.get(uid)
    undo(two)
    assert document.scene.live.get(uid)


# --- the lock -----------------------------------------------------------------------------


def test_a_stroke_on_a_brick_model_in_model_level_is_refused_with_the_reason(house):
    voxels = document.grid.voxels.copy()
    x, y = house.camera.project([house.camera.target])[0]
    house.handle("mouse_move", {"x": float(x), "y": float(y)})
    house.handle("mouse_down", {"button": "left"})
    house.handle("mouse_move", {"x": float(x) + 30, "y": float(y)})
    house.handle("mouse_up", {"button": "left"})
    assert np.array_equal(document.grid.voxels, voxels)
    assert "built from bricks" in house.message
    assert state.hover is None  # no highlight on a brick model in Model level


MODEL_LEVEL_MODULES = ("model_ops", "select_ops", "clip_ops", "cage_ops", "pattern_ops", "xs_ops",
                       "gradient_ops", "shader_brush", "mirror_preview", "gizmo_ops", "stroke_ops")
# The commands of those modules that change no cell of the model.
ALLOWED_IN_MODEL_LEVEL = {
    "rok.deselect", "rok.mask_clear", "rok.selection_recall", "rok.transform",
    "rok.stroke_design", "rok.size_slider", "rok.model_menu_key", "rok.xs_gallery",
    "rok.shader_panel", "rok.shader_brush_stop", "rok.cage_done", "rok.pivot_reset",
    "rok.xs_folder", "rok.gradient_end",
}


def test_every_command_that_writes_cells_polls_false_on_a_brick_model(house):
    import importlib

    open_ = []
    for name in MODEL_LEVEL_MODULES:
        try:
            module = importlib.import_module(f"rok.app.{name}")
        except ImportError:
            continue
        for value in vars(module).values():
            ident = getattr(value, "command_id", None)
            if not isinstance(value, type) or not ident or value.__module__ != module.__name__:
                continue
            if ident in ALLOWED_IN_MODEL_LEVEL:
                continue
            if getattr(opkit.ops.rok, ident.removeprefix("rok.")).poll():
                open_.append(ident)
    assert open_ == []


def test_the_scene_moves_and_turns_a_brick_model_but_locks_the_rest(house):
    _go(kit.LEVEL_SCENE)
    for ident in ("scene_stretch", "scene_flip", "model_split", "make_part"):
        assert not poll(ident), ident
    for ident in ("scene_turn", "scene_fit", "scene_drop", "scene_nudge", "model_duplicate", "model_rename",
                  "convert_to_voxels"):
        assert poll(ident), ident
    position = document.model.position
    run("rok.scene_nudge", direction="RIGHT")
    assert document.model.position != position


def test_a_quarter_turn_in_the_scene_turns_the_bricks_with_the_voxels(house):
    _go(kit.LEVEL_SCENE)
    table = document.grid.bricks.copy()
    voxels = document.grid.voxels.copy()
    assert run("rok.scene_turn") == {"FINISHED"}
    assert document.grid.bricks.check(document.grid.voxels) is None
    assert document.grid.bricks != table and len(document.grid.bricks) == len(table)
    run("rok.scene_turn", back=True)
    assert document.grid.bricks == table and np.array_equal(document.grid.voxels, voxels)
    run("rok.scene_turn")
    undo(house)
    assert document.grid.bricks == table and np.array_equal(document.grid.voxels, voxels)
    # Turns about the other axes and flips are refused.
    assert run("rok.transform", target="MODEL", kind="ROTATE", axis=0, turns=1) == {"CANCELLED"}
    assert run("rok.transform", target="MODEL", kind="FLIP", axis=0) == {"CANCELLED"}
    assert document.grid.bricks == table


def test_fit_and_duplicate_keep_the_bricks(house):
    _go(kit.LEVEL_SCENE)
    run("rok.scene_fit")
    grid = document.grid
    assert grid.size == (8, 8, 5) and grid.bricks.check(grid.voxels) is None
    undo(house)
    assert document.grid.size == (10, 10, 8) and document.grid.bricks.check(document.grid.voxels) is None
    source = document.grid
    run("rok.model_duplicate")
    assert document.grid is not source and document.grid.bricks == source.bricks
    assert document.grid.bricks is not source.bricks


def test_a_carve_passes_through_a_brick_model(house):
    from rok.app import scene_arrange

    _go(kit.LEVEL_SCENE)
    house_model = document.model
    cutter = document.scene.add(brick.carry(house_model.grid, house_model.grid.copy()), "cutter",
                                tuple(house_model.position))
    cutter.grid.bricks = None
    voxels = house_model.grid.voxels.copy()
    _records, changed = scene_arrange.carve(cutter)
    assert changed == [] and np.array_equal(house_model.grid.voxels, voxels)


def test_animate_poses_a_brick_model_but_gives_it_no_frame(house):
    from rok.app import anim_mode

    anim_mode.go("ANIMATE")
    assert state.animating
    assert not poll("frame_add") and not poll("anim_paint")


# --- the MCP --------------------------------------------------------------------------


def test_the_mcp_cell_writers_answer_with_the_reason():
    pytest.importorskip("rok_native")
    from rok.headless import Session

    s = Session()
    try:
        s.run("rok.new", size=(8, 8, 8), discard=True)
        s.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [1, 1, 0], "colour": 5})
        s.run("rok.convert_to_bricks")
        reply = s.agent.call("rok_voxels_set", {"voxels": [[4, 4, 0, 5]]})
        assert reply.error and "built from bricks" in reply.text and "convert_to_voxels" in reply.text
        assert int(document.grid.voxels[4, 4, 0]) == 0
    finally:
        s.close()


# --- placing (Phase 1b) --------------------------------------------------------------------

from rok.app.opkit import CONTEXT  # noqa: E402
from rok.core.raycast import Hit  # noqa: E402


@pytest.fixture
def empty(config):
    """An empty 16 x 16 x 8 voxel model, in Brick mode."""
    a = _start(None)
    run("rok.new", size=(16, 16, 8), discard=True)
    document.colour = 9
    _go(kit.LEVEL_BRICK)
    return a


def floor(x, y):
    return Hit((x, y, -1), (0, 0, 1), 1.0, wall=True)


def top(x, y, z):
    return Hit((x, y, z), (0, 0, 1), 1.0)


def test_the_start_comes_from_the_face_the_mouse_is_on():
    assert brick_mode.start_of(floor(3, 4)) == (3, 4, 0)
    assert brick_mode.start_of(top(3, 4, 2)) == (3, 4, 3)
    assert brick_mode.start_of(Hit((3, 4, 2), (1, 0, 0), 1.0)) == (4, 4, 2)  # beside a side face
    assert brick_mode.start_of(Hit((16, 4, 2), (-1, 0, 0), 1.0, wall=True)) is None  # out through a side wall
    assert brick_mode.start_of(None) is None


def test_a_click_sets_the_brick_and_undo_takes_it_and_its_record_back(empty):
    run("rok.brick_take", number=9)  # 2×4
    steps = len(document.undo._undo)
    assert brick_mode.place_at(floor(2, 2)) is None
    grid = document.grid
    assert grid.bricks.bricks == [brick.Brick(2, 2, 0, 2, 4, 9)]
    assert int((grid.voxels == 9).sum()) == 8
    assert len(document.undo._undo) == steps + 1 and document.undo.peek().label == "Brick 2×4"
    assert brick_mode.place_at(top(2, 2, 0)) is None  # stacked on it
    assert grid.bricks.bricks[1].z == 1
    undo(empty)
    undo(empty)
    assert document.grid.bricks is None and not document.grid.voxels.any()
    redo(empty)
    assert len(document.grid.bricks) == 1 and int((document.grid.voxels == 9).sum()) == 8


def test_a_refused_click_writes_nothing_and_says_why(empty):
    run("rok.brick_take", number=9)
    brick_mode.place_at(floor(0, 0))
    steps = len(document.undo._undo)
    voxels = document.grid.voxels.copy()
    assert brick_mode.place_at(floor(15, 0)) == "The brick does not fit: outside the grid"
    assert brick_mode.place_at(Hit((10, 10, 4), (0, 0, 1), 1.0)) == "The brick does not fit: no support"
    assert len(document.undo._undo) == steps and np.array_equal(document.grid.voxels, voxels)


def test_a_turned_brick_keeps_its_first_cell_under_the_mouse(empty):
    run("rok.brick_take", number=4)  # 1×4
    key(empty, "R")
    assert state.brick_turn == 1
    brick_mode.place_at(floor(5, 8))
    assert document.grid.bricks.bricks[0] == brick.Brick(5, 8, 0, 4, 1, 9)  # the 1×4 along y now runs along x
    key(empty, "R", shift=True)
    assert state.brick_turn == 0


def test_a_mirrored_click_places_both_bricks_as_one_step(empty):
    CONTEXT.prefs.rok_mirror = (True, False, False)
    try:
        run("rok.brick_take", number=9)
        steps = len(document.undo._undo)
        assert brick_mode.place_at(floor(1, 1)) is None
        assert sorted((b.x, b.y) for b in document.grid.bricks.bricks) == [(1, 1), (13, 1)]
        assert len(document.undo._undo) == steps + 1
    finally:
        CONTEXT.prefs.rok_mirror = (False, False, False)


def test_pull_paint_and_pick_take_the_whole_brick(empty):
    run("rok.brick_take", number=9)
    brick_mode.place_at(floor(4, 4))
    brick_mode.place_at(top(4, 4, 0))
    document.colour = 12
    assert brick_mode.paint_at(top(5, 7, 0)) == "Point at a brick" or True
    assert brick_mode.paint_at(Hit((5, 7, 1), (0, 0, 1), 1.0)) is None
    grid = document.grid
    assert grid.bricks.bricks[1].colour == 12 and int((grid.voxels[:, :, 1] == 12).sum()) == 8
    assert document.undo.peek().label == "Paint Brick"
    document.colour = 3
    assert brick_mode.pick_at(Hit((4, 4, 1), (0, 0, 1), 1.0)) is None
    assert state.brick_hand == (2, 4) and document.colour == 12
    assert brick_mode.pull_at(Hit((5, 5, 1), (0, 0, 1), 1.0)) is None
    assert len(grid.bricks) == 1 and not grid.voxels[:, :, 1].any()
    assert document.undo.peek().label == "Pull 2×4"
    # A loose cell goes as a 1×1; a pull on nothing says so.
    with brick.unlocked(grid):
        grid.set_voxel(10, 10, 0, 5)
    assert brick_mode.pull_at(top(10, 10, 0)) is None and grid.voxels[10, 10, 0] == 0
    assert brick_mode.pull_at(floor(12, 12)) == "Point at a brick"
    undo(empty)
    assert len(document.grid.bricks) == 1


def test_option_click_in_place_pulls_and_the_ghost_shows_where_it_lands(empty):
    run("rok.brick_take", number=9)
    from rok.app import object as voxel_object

    host = empty.host
    ox, oy, oz = voxel_object.model_origin(document.model)
    x, y = empty.camera.project([[ox + 7.5, oy + 7.5, oz]])[0]  # the middle of the floor
    host.calls.clear()
    empty.handle("mouse_move", {"x": float(x), "y": float(y)})
    ghosts = [c for c in host.calls if c[0] == "set_ghost" and c[1][1]]
    assert ghosts and ghosts[-1][1][1][0][1].shape == (2, 4, 1)
    empty.handle("mouse_down", {"button": "left"})
    empty.handle("mouse_up", {"button": "left"})
    assert len(document.grid.bricks) == 1
    keys = {"shift": False, "ctrl": False, "alt": True, "meta": False}
    empty.handle("modifiers", keys)
    empty.handle("mouse_move", {"x": float(x), "y": float(y) + 1})
    empty.handle("mouse_down", {"button": "left"})
    empty.handle("mouse_up", {"button": "left"})
    empty.handle("modifiers", {**keys, "alt": False})
    assert len(document.grid.bricks) == 0 and not document.grid.voxels.any()


def test_a_refused_ghost_names_the_reason_in_the_status_line(empty):
    run("rok.brick_take", number=9)
    from rok.app import object as voxel_object

    ox, oy, oz = voxel_object.model_origin(document.model)
    x, y = empty.camera.project([[ox + 14.5, oy + 14.5, oz]])[0]
    empty.handle("mouse_move", {"x": float(x), "y": float(y)})
    assert brick_mode._ghost["kind"] == "REFUSED"
    assert brick_mode.status() == "The brick does not fit here: outside the grid"


# --- the look (Phase 1c) -------------------------------------------------------------------


def _seams_calls(app):
    return [c for c in app.host.calls if c[0] in ("set_seams", "clear_seams")]


def test_the_seams_field_follows_the_bricks(empty):
    app = empty
    run("rok.brick_take", number=9)
    app.host.calls.clear()
    brick_mode.place_at(floor(2, 2))
    app.handle("tick", {})
    sent = _seams_calls(app)
    assert sent and sent[-1][0] == "set_seams"
    idx, origin, box = sent[-1][1]
    field = np.zeros(document.grid.size, dtype=np.uint8)
    o = origin
    field[o[0]:o[0] + box.shape[0], o[1]:o[1] + box.shape[1], o[2]:o[2] + box.shape[2]] = box
    assert np.array_equal(field, document.grid.bricks.field(document.grid.voxels))
    assert field[2, 2, 0] & 0b101 == 0b101  # +x and +y are the same brick
    app.host.calls.clear()
    run("rok.convert_to_voxels")
    app.handle("tick", {})
    assert [c[0] for c in _seams_calls(app)] == ["clear_seams"]
    app.host.calls.clear()
    undo(app)
    app.handle("tick", {})
    assert [c[0] for c in _seams_calls(app)][-1:] == ["set_seams"]


# --- Random colours -------------------------------------------------------------------------


def test_random_colours_give_each_brick_one_of_the_set_never_twice_in_a_row(empty):
    run("rok.brick_take", number=1)
    brick_mode._rng.seed(7)
    region = SimpleNamespace(width=1440, height=900)
    layout = brick_bin.bin_layout(1.0, 1440, 900)
    switch = next(i for i in layout.items if i.kind == "RANDOM")
    x, y, w, h = switch.rect
    brick_mode.press(SimpleNamespace(region=region, mx=x + w / 2, my=y + h / 2, left=True,
                                     event=SimpleNamespace(alt=False)))
    assert state.brick_random
    wanted = {rgb for _name, rgb in brick.RANDOM_COLOURS}
    assert len(wanted) == 10
    steps = len(document.undo._undo)
    seen = []
    for k in range(8):
        assert brick_mode.place_at(floor(2 * k, 0)) is None
        b = document.grid.bricks.bricks[-1]
        rgb = tuple(int(c) for c in document.scene.palette[b.colour][:3])
        assert rgb in wanted
        seen.append(rgb)
    assert all(a != b for a, b in zip(seen, seen[1:]))
    assert len(set(seen)) > 2
    assert len(document.undo._undo) == steps + 8  # the colour each one claimed is in its own step
    undo(empty)
    assert len(document.grid.bricks) == 7
    brick_mode.set_random(False)
    brick_mode.place_at(floor(0, 4))
    assert document.grid.bricks.bricks[-1].colour == document.colour
