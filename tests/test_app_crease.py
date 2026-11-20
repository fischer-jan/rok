"""Model > Crease in the app: the dialog shows the bars in the real voxels while
its values change, Cancel puts the voxels back, OK is one undo step, and Mirror
repeats each bar on the other side."""

from __future__ import annotations

import numpy as np
import pytest

from rok.app import opkit
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state
from rok.core.chrome import COMMANDS, DEFAULT_MODEL_ITEMS
from test_foundation import RecordingHost
from tests.helpers import FIXTURES

CANOPY = FIXTURES / "facets" / "canopy.vox"
FRAME = 165


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ROK_CONFIG", str(tmp_path / "config"))
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(CANOPY))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    document.colour = FRAME
    yield a
    opkit.CONTEXT.prefs.rok_mirror = (False, False, False)
    document.reset()
    state.__init__()


def records() -> int:
    return len(document.undo._undo)


def open_crease(app_):
    opkit.call("rok.crease", "INVOKE")
    assert app_.dialog is not None and app_.dialog.op.command_id == "rok.crease"
    return app_.dialog.op


def test_crease_is_in_the_model_menu_and_cmd_k():
    assert "crease" in DEFAULT_MODEL_ITEMS
    assert [c.name for c in COMMANDS if c.id == "crease"] == ["Crease..."]


def test_the_dialog_shows_the_bars_live_and_cancel_puts_the_voxels_back(app):
    before = document.grid.voxels.copy()
    depth = records()
    op = open_crease(app)
    assert (document.grid.voxels == FRAME).sum() > 0  # shown at once, in the real voxels
    assert records() == depth  # no undo step while it shows
    shown = (document.grid.voxels == FRAME).sum()
    op.angle = 60  # only the sharpest edges
    assert 0 < (document.grid.voxels == FRAME).sum() < shown
    op.angle = 20
    op.width = 2
    assert (document.grid.voxels == FRAME).sum() > shown
    app.dialog._finish(False)
    assert np.array_equal(document.grid.voxels, before) and records() == depth


def test_ok_is_one_undo_step_and_undo_takes_the_bars_away(app):
    before = document.grid.voxels.copy()
    depth = records()
    op = open_crease(app)
    op.bars = "PAINT"
    app.dialog._finish(True)
    after = document.grid.voxels.copy()
    assert records() == depth + 1
    assert np.array_equal(after != 0, before != 0) and (after == FRAME).any()  # Paint adds no voxel
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)


def test_with_mirror_on_the_bars_are_symmetric(app):
    opkit.CONTEXT.prefs.rok_mirror = (True, True, False)
    open_crease(app)
    app.dialog._finish(True)
    v = document.grid.voxels
    assert np.array_equal(v, v[::-1]) and np.array_equal(v, v[:, ::-1])


def test_with_a_selection_the_bars_stay_on_it(app):
    sel = document.selection
    mask = np.zeros(document.grid.voxels.shape, dtype=bool)
    mask[:16] = document.grid.voxels[:16] != 0
    sel.replace(mask)
    open_crease(app).bars = "PAINT"
    app.dialog._finish(True)
    v = document.grid.voxels
    assert (v[:16] == FRAME).any() and not (v[16:] == FRAME).any()


def test_raise_and_paint_adds_bars_and_paints_the_edges_under_them(app):
    before = document.grid.voxels.copy()
    depth = records()
    open_crease(app).bars = "BOTH"
    app.dialog._finish(True)
    after = document.grid.voxels
    assert records() == depth + 1
    assert (after[before == 0] == FRAME).any()  # raised bars on empty cells
    assert (after[before != 0] == FRAME).any()  # painted edge voxels under them
