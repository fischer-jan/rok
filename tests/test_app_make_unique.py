"""A click on a linked copy's chain asks before Make Unique: the card names the
copies, Cancel keeps the link, "Don't ask again" turns the question off and
Rok > Settings turns it back on."""

from __future__ import annotations

import pytest

from rok.app import opkit, scene_tools, settings
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.scene_ops import ASK_MAKE_UNIQUE
from rok.app.state import state
from rok.core import link as link_core
from test_foundation import RecordingHost
from tests.helpers import FIXTURES


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ROK_CONFIG", str(tmp_path / "config"))
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "robot.vox"))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    yield a
    document.reset()
    state.__init__()


def linked_copy() -> int:
    """A linked copy of the model, as a Shift-drag leaves it; its uid."""
    token = scene_tools.begin_copy([document.model])
    scene_tools.finish_copy(token, [])
    return token[2][0]


def is_linked(uid: int) -> bool:
    return link_core.is_linked(document.scene, document.scene.by_uid(uid))


def test_the_chain_asks_first_and_cancel_keeps_the_link(app):
    uid = linked_copy()
    opkit.call("rok.model_make_unique", "INVOKE", uid=uid)
    assert app.dialog is not None and app.dialog.op.command_id == "rok.model_make_unique"
    app.dialog._finish(False)
    assert is_linked(uid)
    opkit.call("rok.model_make_unique", "INVOKE", uid=uid)
    app.dialog._finish(True)
    assert not is_linked(uid) and settings.get(ASK_MAKE_UNIQUE, "on") == "on"


def test_dont_ask_again_turns_the_question_off_and_settings_turns_it_on(app):
    uid = linked_copy()
    opkit.call("rok.model_make_unique", "INVOKE", uid=uid)
    app.dialog.op.dont_ask = True
    app.dialog._finish(True)
    assert not is_linked(uid) and settings.get(ASK_MAKE_UNIQUE) == "off"
    uid = linked_copy()
    opkit.call("rok.model_make_unique", "INVOKE", uid=uid)
    assert app.dialog is None and not is_linked(uid)  # at once, no question
    opkit.call("rok.preferences", "INVOKE")
    assert app.dialog.op.ask_unique is False
    app.dialog.op.ask_unique = True
    app.dialog._finish(True)
    assert settings.get(ASK_MAKE_UNIQUE) == "on"
