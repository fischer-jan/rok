"""How the mouse moves the view: a held key and a plain move (Ctrl orbits,
Cmd pans, Option zooms), the wheel, and Rok > Settings (navigation.py,
view_ops.HoldMoveAction). A move with a button held belongs to its drag,
and the trackpad never changes."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from rok.app import navigation
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state

FIXTURES = Path(__file__).parent / "fixtures"
NO_KEYS = {"shift": False, "ctrl": False, "alt": False, "meta": False}


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


def start(tmp_path, monkeypatch, settings_file: str | None = None) -> RokApp:
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    if settings_file:
        (tmp_path / "config").mkdir(exist_ok=True)
        shutil.copy(FIXTURES / "settings" / settings_file, tmp_path / "config" / "settings.txt")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "bird.vox"))
    a.handle("start", {"width": 1200.0, "height": 800.0, "scale": 2.0})
    return a


@pytest.fixture
def app(tmp_path, monkeypatch):
    a = start(tmp_path, monkeypatch)
    yield a
    document.reset()
    state.__init__()


def keys(app, **held) -> None:
    app.handle("modifiers", {**NO_KEYS, **{("meta" if k == "cmd_key" else k): v for k, v in held.items()}})


def drag(app, button: str, **held) -> list[str]:
    """Press `button` with `held` keys at the window's middle, move up and right,
    let go. Returns the modal operators that ran during the drag."""
    app.handle("mouse_move", {"x": 600.0, "y": 400.0})
    keys(app, **held)
    app.handle("mouse_down", {"button": button})
    running = [type(op).command_id for op in app.modals]
    app.handle("mouse_move", {"x": 640.0, "y": 370.0})
    app.handle("mouse_up", {"button": button})
    keys(app)
    return running


def view(app) -> tuple:
    cam = app.camera
    return (round(cam.yaw, 6), round(cam.pitch, 6), round(cam.distance, 6), tuple(round(float(v), 6) for v in cam.target))


def apply(app, **props) -> None:
    app.handle("menu", {"id": "rok.preferences"})
    assert app.dialog is not None and app.dialog.op.command_id == "rok.preferences"
    for name, value in props.items():
        setattr(app.dialog.op, name, value)
    app.redraw()  # the dialog lays out, with the problem line if there is one
    app.dialog._finish(True)


def wheel(app, dy: float, **held) -> None:
    keys(app, **held)
    app.handle("scroll", {"dx": 0.0, "dy": dy, "precise": False})
    keys(app)


def hold(app, key: str, to=(660.0, 360.0)) -> None:
    """Hold `key` and move the mouse from the middle, no button."""
    app.handle("mouse_move", {"x": 600.0, "y": 400.0})
    keys(app, **{key: True})
    app.handle("mouse_move", {"x": to[0], "y": to[1]})
    keys(app)


def test_ctrl_orbits_cmd_pans_and_option_zooms_on_a_plain_move(app):
    before = view(app)
    hold(app, "ctrl")
    orbited = view(app)
    assert orbited[0] != before[0] and orbited[3] == before[3]
    hold(app, "cmd_key")
    panned = view(app)
    assert panned[3] != orbited[3] and panned[:2] == orbited[:2]
    distance = app.camera.distance
    hold(app, "alt")  # the mouse went up
    assert app.camera.distance < distance and view(app)[:2] == panned[:2]
    assert [type(op).command_id for op in app.modals] == ["rok.voxel_mode", "rok.nav_cube"]  # all ended


@pytest.mark.parametrize("key, other", [("ctrl", "shift"), ("cmd_key", "shift"), ("alt", "ctrl"), ("ctrl", "alt")])
def test_a_hold_key_moves_the_view_only_when_it_is_the_only_one_held(app, key, other):
    before = view(app)
    app.handle("mouse_move", {"x": 600.0, "y": 400.0})
    keys(app, **{key: True})
    keys(app, **{key: True, other: True})
    app.handle("mouse_move", {"x": 660.0, "y": 360.0})
    assert view(app) == before
    keys(app, **{key: True})  # the second key comes up: the view moves again
    app.handle("mouse_move", {"x": 600.0, "y": 400.0})
    keys(app)
    assert view(app) != before


def test_a_move_with_a_button_held_never_moves_the_view(app):
    before = view(app)
    assert "rok.stroke" in drag(app, "left", alt=True)  # Option + left drag erases
    assert view(app) == before
    keys(app)
    app.handle("mouse_move", {"x": 600.0, "y": 400.0})
    app.handle("mouse_down", {"button": "left"})  # a stroke runs
    keys(app, alt=True)  # Option during it
    app.handle("mouse_move", {"x": 640.0, "y": 360.0})
    app.handle("mouse_up", {"button": "left"})
    keys(app)
    assert view(app) == before


def test_ctrl_left_drag_on_nothing_orbits_and_draws_nothing(app):
    before, voxels = view(app), document.grid.voxels.copy()
    assert "rok.stroke" not in drag(app, "left", ctrl=True)
    assert view(app)[0] != before[0] and (document.grid.voxels == voxels).all()
    assert "rok.stroke" not in drag(app, "left", ctrl=True, shift=True)  # Ctrl+Shift drag is not a stroke


def test_the_wheel_zooms_and_settings_can_turn_it(app):
    distance = app.camera.distance
    wheel(app, 30.0)
    assert app.camera.distance < distance  # wheel up moves closer
    apply(app, invert_wheel=True)
    distance = app.camera.distance
    wheel(app, 30.0)
    assert app.camera.distance > distance
    assert "rok.stroke" in drag(app, "right")  # the right button erases


def test_the_settings_item_is_in_the_app_menu_with_cmd_comma(app):
    rok_menu = next(items for title, items in navigation_menus(app) if title == "Rok")
    assert ("rok.preferences", "Settings…", "CmdOrCtrl+,", -1) in rok_menu


def test_the_app_menu_has_the_system_hide_items(app):
    rok_menu = next(items for title, items in navigation_menus(app) if title == "Rok")
    ids = [item[0] for item in rok_menu if item is not None]
    assert ids.index("@hide") < ids.index("@hide_others") < ids.index("@show_all") < ids.index("rok.quit")


def navigation_menus(app):
    from rok.app.handlers import registry

    return [m for fn in registry.menus for m in fn(app)]


def test_the_next_start_reads_the_wheel_setting(tmp_path, monkeypatch):
    app = start(tmp_path, monkeypatch, "inverted_wheel.txt")
    try:
        distance = app.camera.distance
        wheel(app, 30.0)
        assert app.camera.distance > distance  # inverted: wheel up moves away
        app.handle("menu", {"id": "rok.preferences"})
        assert app.dialog.op.invert_wheel is True
    finally:
        document.reset()
        state.__init__()


def test_the_trackpad_never_changes(tmp_path, monkeypatch):
    app = start(tmp_path, monkeypatch, "inverted_wheel.txt")
    try:
        before = view(app)
        app.handle("scroll", {"dx": 20.0, "dy": 0.0, "precise": True})
        assert view(app)[0] != before[0]  # two fingers orbit
        keys(app, ctrl=True)
        target = view(app)[3]
        app.handle("scroll", {"dx": 20.0, "dy": 10.0, "precise": True})
        assert view(app)[3] != target  # Ctrl + two fingers pan
        keys(app, alt=True)
        distance = app.camera.distance
        app.handle("scroll", {"dx": 0.0, "dy": 30.0, "precise": True})
        assert app.camera.distance != distance  # Option + two fingers zoom
        keys(app)
        distance = app.camera.distance
        app.handle("pinch", {"delta": 0.2})
        assert app.camera.distance < distance  # a pinch zooms
    finally:
        document.reset()
        state.__init__()


def test_steady_hand_is_saved_and_read_on_the_next_start(app, tmp_path, monkeypatch):
    from rok.app import settings
    from rok.app.opkit import CONTEXT

    assert not CONTEXT.prefs.rok_steady_hand  # off until the user turns it on
    apply(app, steady_on=True, steady_length=45)
    assert settings.get("steady_hand") == "on" and settings.get("steady_length") == "45"
    assert CONTEXT.prefs.rok_steady_hand and int(CONTEXT.prefs.rok_steady_length) == 45
    document.reset()
    state.__init__()
    (tmp_path / "second").mkdir()
    again = start(tmp_path / "second", monkeypatch, "steady_hand.txt")
    try:
        assert CONTEXT.prefs.rok_steady_hand and int(CONTEXT.prefs.rok_steady_length) == 60
        again.handle("menu", {"id": "rok.preferences"})
        assert again.dialog.op.steady_on and again.dialog.op.steady_length == 60
        again.redraw()  # the dialog lays out with its cards, chips and slider
    finally:
        document.reset()
        state.__init__()


def test_the_string_slider_follows_a_click_and_a_drag(app):
    from rok.core import steady_hand

    app.handle("menu", {"id": "rok.preferences"})
    app.redraw()
    slider = next(h for h in app.dialog.hots if h.kind == "slider")
    x, y, w, h = slider.rect
    mid = y + h / 2
    app.handle("mouse_move", {"x": x + w, "y": mid})
    app.handle("mouse_down", {"button": "left"})
    assert app.dialog.op.steady_length == steady_hand.MAX_LENGTH  # the far end
    app.handle("mouse_move", {"x": x - 50, "y": mid})  # a drag past the start stays in range
    assert app.dialog.op.steady_length == steady_hand.MIN_LENGTH
    app.handle("mouse_up", {"button": "left"})
    app.handle("mouse_move", {"x": x + w, "y": mid})
    assert app.dialog.op.steady_length == steady_hand.MIN_LENGTH  # the button is up: no change


def test_the_model_flyout_stays_above_the_tool_readout(app):
    from rok.app import chrome, overlay
    from rok.app.opkit import CONTEXT
    from rok.core import chrome as kit

    region = CONTEXT.region
    layout = chrome.layout_for(region)
    assert layout.state.readout_top == overlay.readout_top(region) > 0
    flyout = kit.flyout_widgets(layout, kit.MODEL_ID)
    assert flyout and min(w.rect[1] for w in flyout) >= layout.state.readout_top
