"""The recovery copy: written after a change, deleted when the work is saved or
given up, and offered back after a crash."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from rok.app import recovery, welcome
from rok.app.object import document
from rok.app.state import state
from rok.core import rokfile
from test_app_ui import BIRD, _make_app, _reset_ui

WALK = Path(__file__).parent / "fixtures" / "anim" / "walk.rok"


@pytest.fixture
def app(tmp_path, monkeypatch):
    a = _make_app(tmp_path, monkeypatch, BIRD)
    yield a
    _clean()


def _clean() -> None:
    recovery.discard()
    document.reset()
    state.__init__()
    _reset_ui()
    welcome._skip_at_start = False


def _dead_pid(after: int = 99_000) -> int:
    """A pid above `after` that no process has now."""
    pid = after + 1
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except PermissionError:
            pass
        pid += 1


def _edit() -> None:
    grid = document.scene.models[0].grid
    grid.set_voxel(0, 0, 0, 7)
    document.modified = True


def _write_due(app) -> None:
    """Run the waiting write as its timer would, and wait for the worker."""
    assert recovery._timer is not None and recovery._timer in app.timers
    recovery._timer.callback()
    if recovery._worker is not None:
        recovery._worker.join(10)


def test_a_change_writes_a_copy_that_opens(app):
    _edit()
    assert recovery._timer.interval == recovery.WAIT_S
    _write_due(app)
    assert recovery._timer is None
    scene, doc = rokfile.read(recovery.copy_path())
    assert scene.models[0].grid.get_voxel(0, 0, 0) == 7
    meta = json.loads(recovery.meta_path().read_text())
    assert meta["path"] == str(BIRD) and meta["name"] == "bird.vox"


def test_the_write_waits_for_a_stroke(app):
    _edit()
    state.stroke_active = True
    _write_due(app)
    assert not recovery.copy_path().exists()
    assert recovery._timer.next_at > time.monotonic() + recovery.RETRY_S - 0.5
    state.stroke_active = False
    _write_due(app)
    assert recovery.copy_path().exists()


def test_a_second_change_does_not_push_the_write_back(app):
    _edit()
    timer = recovery._timer
    _edit()
    assert recovery._timer is timer and len([t for t in app.timers if t is timer]) == 1


def test_saving_deletes_the_copy(app):
    _edit()
    _write_due(app)
    document.modified = False  # what Save does after it wrote the file
    assert not recovery.copy_path().exists() and not recovery.meta_path().exists()
    assert recovery._timer is None


def test_quit_deletes_the_copy(app):
    _edit()
    _write_due(app)
    from rok.app import opkit

    opkit.call("rok.quit", "EXEC")
    assert not recovery.copy_path().exists()
    assert app.host.named("quit")


def test_a_write_that_ends_after_a_discard_leaves_nothing(app):
    _edit()
    generation = recovery._generation
    recovery.discard()
    recovery._write(document.scene, {}, {"path": "", "name": "x", "time": time.time()}, generation)
    assert not recovery.copy_path().exists()


def test_an_animated_scene_comes_back_whole(app):
    scene, _doc = rokfile.read(WALK)
    from rok.app.object import apply_scene

    from rok.app.opkit import CONTEXT

    apply_scene(CONTEXT, scene)
    document.modified = True
    recovery.write_now(wait=True)
    back, _ = rokfile.read(recovery.copy_path())
    assert len(back.models) == len(scene.models)
    assert back.timeline.has_keys() and len(back.timeline.tracks) == len(scene.timeline.tracks)
    for a, b in zip(scene.models, back.models):
        assert np.array_equal(a.grid.voxels, b.grid.voxels)


def _leave_copy(pid: int, path: str = "/somewhere/ship.vox", age_s: float = 60.0) -> None:
    """A copy as a crashed Rok leaves it."""
    scene, doc = rokfile.read(WALK)
    recovery.folder().mkdir(parents=True, exist_ok=True)
    rokfile.write(recovery.copy_path(pid), scene, doc)
    meta = {"path": path, "name": os.path.basename(path), "time": time.time() - age_s}
    recovery.meta_path(pid).write_text(json.dumps(meta))


def _question(app):
    asks = app.host.named("ask")
    assert asks, "no question"
    token, title, message, buttons = asks[-1][1]
    return token, title, message, buttons


def test_a_copy_of_a_dead_rok_is_offered_and_restored(tmp_path, monkeypatch):
    pid = _dead_pid()
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    _leave_copy(pid)
    app = _make_app(tmp_path, monkeypatch, None)
    try:
        token, title, message, buttons = _question(app)
        assert title == "Restore unsaved work?" and "ship.vox" in message
        assert buttons == [recovery.RESTORE, recovery.DISCARD]
        assert welcome._skip_at_start
        app.handle("dialog", {"token": token, "answer": recovery.RESTORE})
        assert len(document.scene.models) == len(rokfile.read(WALK)[0].models)
        assert document.modified and document.filepath == "/somewhere/ship.vox"
        assert not recovery.copy_path(pid).exists() and not recovery.meta_path(pid).exists()
        # This Rok's own copy is there at once, so a second crash loses nothing.
        back, _ = rokfile.read(recovery.copy_path())
        assert len(back.models) == len(document.scene.models)
        assert json.loads(recovery.meta_path().read_text())["path"] == "/somewhere/ship.vox"
    finally:
        _clean()


def test_discard_deletes_the_copy_and_asks_about_the_next(tmp_path, monkeypatch):
    first = _dead_pid()
    second = _dead_pid(first)
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    _leave_copy(first, age_s=10)
    _leave_copy(second, path="/somewhere/older.vox", age_s=3600)
    app = _make_app(tmp_path, monkeypatch, None)
    try:
        token, _title, message, _ = _question(app)
        assert "ship.vox" in message  # the newest first
        app.handle("dialog", {"token": token, "answer": recovery.DISCARD})
        assert not recovery.copy_path(first).exists()
        assert recovery.copy_path(second).exists()
        token, _title, message, _ = _question(app)
        assert "older.vox" in message
        app.handle("dialog", {"token": token, "answer": recovery.DISCARD})
        assert not recovery.copy_path(second).exists()
        assert not document.modified
    finally:
        _clean()


def test_a_copy_of_a_running_rok_is_left_alone(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    _leave_copy(os.getppid())
    assert recovery.left_behind() == []
    assert recovery.copy_path(os.getppid()).exists()


def test_old_copies_and_half_written_files_go(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    old = _dead_pid()
    _leave_copy(old, age_s=(recovery.KEEP_DAYS + 1) * 86400)
    broken = _dead_pid(old)
    (recovery.folder() / f"{broken}.rok.part").write_bytes(b"half")
    assert recovery.left_behind() == []
    assert not recovery.copy_path(old).exists()
    assert not (recovery.folder() / f"{broken}.rok.part").exists()


def test_a_copy_that_does_not_read_is_reported_and_deleted(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    pid = _dead_pid()
    _leave_copy(pid)
    recovery.copy_path(pid).write_bytes(b"not a zip")
    app = _make_app(tmp_path, monkeypatch, None)
    try:
        token, *_ = _question(app)
        app.handle("dialog", {"token": token, "answer": recovery.RESTORE})
        assert not recovery.copy_path(pid).exists()
        assert "could not be read" in app.message
    finally:
        _clean()
