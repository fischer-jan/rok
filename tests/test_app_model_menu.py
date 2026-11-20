"""View > Customize Rail Menu and Keys (rok/app/model_menu.py): the dialog picks
which commands the rail's Model flyout shows and gives them keys; both are
saved in the settings folder and the keys run the commands."""

from __future__ import annotations

import json

import pytest

from rok.app import chrome as app_chrome
from rok.app import model_menu
from rok.app.app import RokApp
from rok.app.handlers import registry
from rok.app.object import document
from rok.app.state import state
from rok.core.chrome import DEFAULT_MODEL_ITEMS, MODEL_COMMAND_IDS
from test_foundation import BIRD, RecordingHost, key


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ROK_CONFIG", str(tmp_path / "config"))
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    document.reset()
    state.__init__()
    model_menu._shown = None
    a = RokApp(RecordingHost(), str(BIRD))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    yield a
    model_menu.save(DEFAULT_MODEL_ITEMS, {})  # leave no custom key in the shared keymap
    model_menu.install(a)
    model_menu._shown = None


def click(app, hot) -> None:
    x, y, w, h = hot.rect
    app.handle("mouse_move", {"x": x + w / 2, "y": y + h / 2})
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def key_hot(app, cid):
    app.redraw()
    return next(h for h in app.dialog.hots if h.kind == "key" and h.name == f"key_{cid}")


def test_the_flyout_starts_without_the_rare_commands(app):
    assert model_menu.items() == DEFAULT_MODEL_ITEMS
    assert app_chrome.chrome_state().model_items == DEFAULT_MODEL_ITEMS


def test_the_view_menu_opens_the_dialog(app):
    menus = dict(app.host.named("set_menu")[-1][1][0])
    assert menus["View"][0][0] == "rok.model_menu"  # the first item
    app.handle("menu", {"id": "rok.model_menu"})
    assert app.dialog is not None and app.dialog.op.command_id == "rok.model_menu"
    assert app.dialog.op.show_fill_all and not app.dialog.op.show_loop
    app.redraw()  # lays out without an error
    key(app, "Escape")
    assert app.dialog is None


def test_save_shows_the_ticked_commands_and_remembers_them(app, tmp_path):
    app.handle("menu", {"id": "rok.model_menu"})
    app.dialog.op.show_loop = True
    app.dialog.op.show_fill_all = False
    key(app, "Enter")
    shown = app_chrome.chrome_state().model_items
    assert "loop" in shown and "fill_all" not in shown
    saved = json.loads((tmp_path / "config" / model_menu.FILE_NAME).read_text())
    assert saved["shown"] == list(shown)
    model_menu._shown = None  # a new start reads the file
    assert model_menu.items() == shown


def test_a_key_field_takes_the_next_key_with_its_modifiers(app):
    app.handle("menu", {"id": "rok.model_menu"})
    click(app, key_hot(app, "loop"))
    assert app.dialog.capturing is not None
    key(app, "l", shift=True, ctrl=True)
    assert app.dialog.op.key_loop == "Shift+Ctrl+L" and app.dialog.capturing is None
    assert app.dialog is not None  # the key did not run anything
    click(app, key_hot(app, "loop"))
    key(app, "Escape")  # keeps the old key, and the dialog stays open
    assert app.dialog is not None and app.dialog.op.key_loop == "Shift+Ctrl+L"
    click(app, key_hot(app, "loop"))
    key(app, "Backspace")
    assert app.dialog.op.key_loop == ""


def test_a_saved_key_runs_its_command(app):
    app.handle("menu", {"id": "rok.model_menu"})
    app.dialog.op.key_twist = "Shift+Ctrl+T"
    key(app, "Enter")
    assert model_menu.keys() == {"twist": "Shift+Ctrl+T"}
    assert ("rok.twist", {"type": "T", "value": "PRESS", "shift": True, "ctrl": True}, None) == registry.keymap[0]
    key(app, "t", shift=True, ctrl=True)
    assert app.dialog is not None and app.dialog.op.command_id == "rok.twist"


def test_a_key_another_command_has_is_named_in_the_dialog(app):
    app.handle("menu", {"id": "rok.model_menu"})
    app.dialog.op.key_loop = "Shift+G"  # Grow's key
    assert "rok.grow" in model_menu.taken("loop", "Shift+G")
    app.redraw()
    assert any("Shift+G ran rok.grow" in (n.text or "") for n in app.dialog._layout().children)


@pytest.mark.parametrize("key_text, spec, shown", [
    ("Shift+Ctrl+L", {"type": "L", "value": "PRESS", "shift": True, "ctrl": True}, "Shift+Ctrl+L"),
    ("Cmd+DIGIT_1", {"type": "DIGIT_1", "value": "PRESS", "cmd_key": True}, "Cmd+1"),
    ("Option+PAGE_UP", {"type": "PAGE_UP", "value": "PRESS", "alt": True}, "Option+Page Up"),
])
def test_key_strings_parse_and_read(key_text, spec, shown):
    assert model_menu.parse(key_text) == spec
    assert model_menu.label(key_text) == shown


@pytest.mark.parametrize("bad", ["", "Shift+", "Hyper+L", "Shift+LEFT_SHIFT"])
def test_a_bad_key_string_is_no_key(bad):
    assert model_menu.parse(bad) is None


def test_a_broken_settings_file_gives_the_defaults(app, tmp_path):
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / model_menu.FILE_NAME).write_text("{not json")
    model_menu.load()
    assert model_menu.items() == DEFAULT_MODEL_ITEMS and model_menu.keys() == {}


def test_every_model_command_is_still_in_cmd_k():
    from rok.core.chrome import COMMANDS

    ids = {c.id for c in COMMANDS}
    assert set(MODEL_COMMAND_IDS) <= ids


def test_a_saved_list_without_known_shows_what_it_lists_and_nothing_more(app, tmp_path):
    shown = ["model_split", "fill_all", "fill_volume", "clear_all", "inverse", "fit_grid", "drop_to_floor",
             "smooth", "scale", "mirror_half", "remove_slice", "resize"]
    (tmp_path / "config" / model_menu.FILE_NAME).write_text(json.dumps({"shown": shown, "keys": {}}))
    model_menu.load()
    assert model_menu.items() == tuple(c for c in MODEL_COMMAND_IDS if c in shown)


def test_a_command_you_turned_off_stays_off_after_a_save(app, tmp_path):
    shown = [c for c in DEFAULT_MODEL_ITEMS if c != "cage"]
    model_menu.save(shown, {})
    model_menu._shown = None
    assert model_menu.items() == tuple(shown)  # "known" holds the cage, so it does not come back
