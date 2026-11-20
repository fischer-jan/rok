"""The foundation of the app: opkit, the dispatch, dialogs, the view shims,
the painter, and the file and undo actions. The host only records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app import gfx, opkit
from rok.app.app import RokApp
from rok.app.draw import DrawList
from rok.app.object import document
from rok.app.opkit import CONTEXT, Event
from rok.app.state import ModeState, state
from rok.app.view3d import world_to_region, ray_origin, ray_direction
from rok.core.undo import ModelRecord, StrokeRecorder
from rok.core.vox_read import read_scene

FIXTURES = Path(__file__).parent / "fixtures"
BIRD = FIXTURES / "bird.vox"


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
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    from rok.app import io_ops
    from rok.core import new_sizes

    monkeypatch.setattr(io_ops, "_last_new_size", new_sizes.size_of(new_sizes.DEFAULT))
    document.reset()
    state.__init__()
    host = RecordingHost()
    a = RokApp(host, str(BIRD))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    yield a
    document.reset()
    state.__init__()


def key(app, name: str, **mods) -> None:
    for m in ("shift", "ctrl", "alt", "meta"):
        mods.setdefault(m, False)
    app.handle("modifiers", mods)
    app.handle("key_down", {"key": name, "code": name, "text": name if len(name) == 1 else "", "repeat": False})
    app.handle("key_up", {"key": name, "code": name, "text": "", "repeat": False})


# --- opkit ------------------------------------------------------------------------------


def test_an_operator_gets_its_defaults_and_the_call_values():
    class Op(opkit.Action):
        command_id = "test.op"
        count = opkit.IntProp(default=3, min=1, max=5)
        axis = opkit.EnumProp(items=[("X", "X", ""), ("Y", "Y", "")])

    op = Op(count=9)
    assert op.count == 5 and op.axis == "X"


def test_poll_decides_whether_a_call_runs():
    ran = []

    @opkit.register
    class Guarded(opkit.Action):
        command_id = "test.guarded"
        allowed = False

        @classmethod
        def poll(cls, context):
            return cls.allowed

        def execute(self, context):
            ran.append(1)
            return {"FINISHED"}

    assert opkit.ops.test.guarded() == {"CANCELLED"} and not ran
    Guarded.allowed = True
    assert opkit.ops.test.guarded.poll()
    assert opkit.ops.test.guarded() == {"FINISHED"} and ran == [1]


# --- the view shims ---------------------------------------------------------------------


@pytest.mark.parametrize("x, y", [(0, 0), (720, 450), (1400, 880), (100, 700)])
def test_region_pixels_have_their_origin_at_the_bottom_left(app, x, y):
    region, view_state = CONTEXT.region, CONTEXT.view_state
    o = ray_origin(region, view_state, (x, y))
    d = ray_direction(region, view_state, (x, y))
    back = world_to_region(region, view_state, o + d * 40.0)
    assert np.allclose(back, (x, y), atol=1e-6)
    # The top of the region is up on the screen.
    top = world_to_region(region, view_state, view_state.orbit_centre + np.array([0.0, 0.0, 5.0]))
    mid = world_to_region(region, view_state, view_state.orbit_centre)
    assert top[1] > mid[1]


# --- the painter --------------------------------------------------------------------------


def test_the_painter_flips_region_pixels_to_host_pixels(app):
    d = DrawList()
    with gfx.pixels(d, CONTEXT.region):
        gfx.pen.rect_fill((10, 20, 30, 40), (1, 0, 0, 1))
    tris = d.arrays()["tris2"].reshape(-1, 6)
    ys = sorted(set(tris[:, 1].tolist()))
    assert ys == [900 - 60, 900 - 20]


def test_blf_draws_at_the_baseline(app):
    d = DrawList()
    with gfx.pixels(d, CONTEXT.region):
        gfx.fonts.size(0, 20)
        gfx.fonts.position(0, 100, 50, 0)
        gfx.fonts.draw(0, "Hi")
    (x, top, size, _c, text), = d.texts
    assert (x, text, size) == (100.0, "Hi", 20.0)
    assert top == pytest.approx(900 - 50 - 20 * gfx.BASELINE)


def test_the_view_pass_translates_and_goes_on_top(app):
    d = DrawList()
    with gfx.view(d):
        with gfx.translated((10, 0, 0)):
            gfx.pen.lines([[0, 0, 0], [1, 0, 0]], (1, 1, 1, 1), 2.0)
        with gfx.on_top():
            gfx.pen.tris([[0, 0, 0], [1, 0, 0], [0, 1, 0]], (1, 1, 1, 1))
    line = d.arrays()["lines3_depth"]
    assert line[0] == 10.0 and line[3] == 11.0
    assert len(d.arrays()["tris3_top"]) == 3 * 7


# --- the app, the dispatch and the file operators ------------------------------------------


def test_start_opens_the_file_on_the_floor_and_names_the_window(app):
    assert document.name == "bird.vox"
    lo, hi = document.scene.world_bounds()
    assert lo[2] == 0
    assert app.host.named("set_title")[-1][1][0] == "bird.vox — Rok"
    assert app.host.named("set_model") and app.host.named("set_camera")
    menus = [title for title, _items in app.host.named("set_menu")[-1][1][0]]
    assert menus[:4] == ["Rok", "File", "Edit", "View"]


def test_new_opens_a_dialog_and_create_makes_the_chosen_size(app):
    app.handle("menu", {"id": "rok.new"})
    assert app.dialog is not None
    app.dialog.op.preset = "TILE"  # its update callback fills the size
    assert tuple(app.dialog.op.size) == (64, 64, 16)
    key(app, "Enter")
    assert app.dialog is None
    assert document.grid.size == (64, 64, 16) and document.name == "Untitled.vox"


def test_escape_closes_the_dialog_without_running_it(app):
    app.handle("menu", {"id": "rok.new"})
    key(app, "Escape")
    assert app.dialog is None and document.name == "bird.vox"


def test_a_click_on_a_dialog_field_types_a_new_value(app):
    app.handle("menu", {"id": "rok.new"})
    app.redraw()  # lay the dialog out
    field = next(h for h in app.dialog.hots if h.kind == "field" and h.index == 2)
    x, y, w, h = field.rect
    app.handle("mouse_move", {"x": x + w / 2, "y": y + h / 2})
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})
    assert app.dialog.editing is not None
    for _ in range(3):
        key(app, "Backspace")
    for ch in "48":
        key(app, ch)
    key(app, "Enter")  # takes the value
    assert tuple(app.dialog.op.size) == (32, 32, 48)
    key(app, "Enter")  # creates
    assert document.grid.size == (32, 32, 48)


def test_open_asks_first_when_the_scene_is_modified(app):
    document.modified = True
    app.handle("menu", {"id": "rok.open"})
    (_n, (token, _title, _text, buttons), _kw), = app.host.named("ask")
    assert buttons == ["Discard", "Cancel"]
    app.handle("dialog", {"token": token, "answer": "Discard"})
    (_n, args, _kw), = app.host.named("file_dialog")
    assert args[1] is False  # an open dialog
    app.handle("dialog", {"token": args[0], "path": str(FIXTURES / "robot.vox")})
    assert document.name == "robot.vox" and not document.modified


def test_save_as_writes_a_vox_and_clears_the_modified_flag(app, tmp_path):
    document.modified = True
    app.handle("menu", {"id": "rok.save_as"})
    (_n, args, _kw), = app.host.named("file_dialog")
    assert args[1] is True and args[4] == "bird.vox"
    app.handle("dialog", {"token": args[0], "path": str(tmp_path / "copy")})
    out = tmp_path / "copy.vox"
    assert out.exists() and not document.modified
    saved, original = read_scene(out), read_scene(BIRD)
    # The same colours on the same cells; a .vox save numbers them in the strip's order.
    assert np.array_equal(saved.palette[saved.models[0].voxels], original.palette[original.models[0].voxels])
    used = len(np.unique(original.models[0].voxels)) - 1
    assert set(np.unique(saved.models[0].voxels)) == set(range(used + 1))  # colours 1..n, nothing unused


def test_a_bad_file_leaves_the_document_and_says_why(app, tmp_path):
    bad = tmp_path / "bad.vox"
    bad.write_bytes(b"not a vox")
    assert opkit.call("rok.open", "EXEC", filepath=str(bad)) == {"CANCELLED"}
    assert document.name == "bird.vox"
    assert "Cannot open bad.vox" in app.message


def test_quit_asks_when_modified_and_quits_on_yes(app):
    document.modified = True
    app.handle("close", {})
    (_n, (token, *_rest), _kw), = app.host.named("ask")
    assert not app.host.named("quit")
    app.handle("dialog", {"token": token, "answer": "Quit"})
    assert app.host.named("quit")


def test_undo_and_redo_go_through_the_stack_and_reach_the_gpu(app):
    grid = document.grid
    rec = StrokeRecorder(grid)
    rec.set(1, 1, 1, 7)
    document.undo.push(ModelRecord(document.active_uid, rec.finish()))
    app.flush()
    app.host.calls.clear()
    app.handle("menu", {"id": "rok.undo"})
    assert grid.voxels[1, 1, 1] == 0 and app.host.named("update_region")
    app.handle("menu", {"id": "rok.redo"})
    assert grid.voxels[1, 1, 1] == 7


def test_ctrl_drag_orbits_the_view_through_the_keymap(app):
    yaw = app.camera.yaw
    app.handle("mouse_move", {"x": 700.0, "y": 400.0})
    app.handle("modifiers", {"shift": False, "ctrl": True, "alt": False, "meta": False})
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", {"x": 760.0, "y": 400.0})
    app.handle("mouse_up", {"button": "left"})
    assert app.camera.yaw != yaw
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": False})
    # Only the always-on hover operator is left: the orbit ended with the key.
    assert [type(op).command_id for op in app.modals] == ["rok.voxel_mode", "rok.nav_cube"]


def test_a_modal_operator_that_passes_through_lets_the_keymap_act(app):
    seen = []

    class Watch(opkit.Action):
        command_id = "test.watch"

        def modal(self, context, event):
            seen.append(event.type)
            return {"PASS"}

    app.modal_add(Watch())
    app.handle("menu", {"id": "rok.new"})
    key(app, "Escape")
    assert "ESCAPE" in seen and app.dialog is None


def test_a_timer_sends_timer_events_to_the_modals(app, monkeypatch):
    got = []

    class Tick(opkit.Action):
        command_id = "test.tick"

        def modal(self, context, event):
            if event.type == "TIMER":
                got.append(1)
                return {"FINISHED"}
            return {"PASS"}

    timer = CONTEXT.prefs.add_timer(0.0)
    app.modal_add(Tick())
    app.handle("tick", {})
    assert got == [1]
    CONTEXT.prefs.remove_timer(timer)
    assert not app.timers


@pytest.mark.parametrize("button", [True, False], ids=["ctrl-drag", "ctrl-hold"])
def test_moving_the_mouse_up_with_ctrl_lowers_the_camera(app, button):
    """The vertical orbit was the wrong way round."""
    app.handle("mouse_move", {"x": 700.0, "y": 450.0})
    app.handle("modifiers", {"shift": False, "ctrl": True, "alt": False, "meta": False})
    if button:
        app.handle("mouse_down", {"button": "left"})
    pitch = app.camera.pitch
    app.handle("mouse_move", {"x": 700.0, "y": 400.0})  # up the screen
    app.handle("mouse_move", {"x": 700.0, "y": 350.0})
    if button:
        app.handle("mouse_up", {"button": "left"})
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": False})
    assert app.camera.pitch < pitch


def test_an_error_in_a_stroke_unblocks_the_tools_and_is_logged(app, tmp_path, monkeypatch):
    """A failing action must not leave a stroke running: every tool would wait
    for it. The window says so, and error.log in the settings folder keeps the
    traceback."""
    from rok.app import recent
    from rok.app.state import state

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")

    class Broken(opkit.Action):
        command_id = "test.broken_stroke"

        def invoke(self, context, event):
            state.stroke_active = True
            context.prefs.add_modal(self)
            return {"RUNNING"}

        def modal(self, context, event):
            raise RuntimeError("broken stroke")

    opkit.register(Broken)
    try:
        opkit.call("test.broken_stroke", "INVOKE")
        assert state.stroke_active
        with pytest.raises(RuntimeError):
            app.handle("mouse_move", {"x": 10.0, "y": 10.0})
    finally:
        opkit.REGISTRY.pop("test.broken_stroke", None)
    assert not state.stroke_active
    assert "broken stroke" in app.message
    log = (tmp_path / "config" / "error.log").read_text()
    assert "RuntimeError: broken stroke" in log and "event mouse_move" in log
