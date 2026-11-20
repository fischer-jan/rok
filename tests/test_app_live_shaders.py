"""Live shaders in the app (rok/app/live_shaders.py), on the real GPU.

The shader is tests/fixtures/live_shaders/snow_top.txt: every voxel with
air above it turns colour 9."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app import opkit, xs_ops
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state

FIXTURES = Path(__file__).parent / "fixtures"
HOUSE = FIXTURES / "looks" / "house.vox"

rok_native = pytest.importorskip("rok_native")


class RecordingHost:
    def __getattr__(self, name):
        def record(*args, **kwargs):
            if name == "measure_text":
                return (len(args[0]) * 7.0, 16.0)
            return None

        return record


@pytest.fixture
def app(tmp_path, monkeypatch):
    try:
        rok_native.ShaderRunner()
    except Exception as exc:  # no GPU here
        pytest.skip(f"no GPU: {exc}")
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setattr(xs_ops, "user_shader_folder", lambda: FIXTURES / "live_shaders")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(HOUSE))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    yield a
    document.reset()
    state.__init__()


def snowed(voxels: np.ndarray) -> np.ndarray:
    """What snow_top makes of `voxels`."""
    out = voxels.copy()
    air = np.concatenate([voxels[:, :, 1:] == 0, np.ones(voxels.shape[:2] + (1,), bool)], axis=2)
    out[(voxels > 0) & air] = 9
    return out


def edit(cells, values) -> None:
    """A plain voxel edit as one undo step, as the tools make it."""
    from rok.app.model_ops import _apply_cells

    new = document.grid.visible.copy()
    for (x, y, z), v in zip(cells, values):
        new[x, y, z] = v
    _apply_cells(new)


def go_live() -> set:
    return opkit.call("rok.xs_run", "EXEC", shader="snow_top", live=True)


def free_top(voxels: np.ndarray) -> tuple[int, int, int]:
    """An empty cell right above a voxel, under the model's top."""
    x, y, z = np.argwhere((voxels[:, :, :-1] > 0) & (voxels[:, :, 1:] == 0))[0]
    return int(x), int(y), int(z) + 1


def test_a_live_shader_runs_again_after_an_edit(app):
    base = document.grid.voxels.copy()
    assert go_live() == {"FINISHED"}
    assert np.array_equal(document.grid.voxels, snowed(base))
    cell = free_top(base)
    edit([cell], [3])
    mine = base.copy()
    mine[cell] = 3
    assert np.array_equal(document.grid.voxels, snowed(mine))  # the snow moved up
    opkit.call("rok.undo")  # the edit and the rerun are one step
    assert np.array_equal(document.grid.voxels, snowed(base))
    opkit.call("rok.redo")
    assert np.array_equal(document.grid.voxels, snowed(mine))
    opkit.call("rok.undo")
    opkit.call("rok.undo")  # and going live is one step
    assert np.array_equal(document.grid.voxels, base)
    assert not document.scene.live


def test_switch_off_remove_and_bake(app):
    base = document.grid.voxels.copy()
    go_live()
    live = document.scene.live[document.active_uid][0]
    opkit.call("rok.live_toggle", "EXEC", live=live.id)
    assert np.array_equal(document.grid.voxels, base)
    opkit.call("rok.live_toggle", "EXEC", live=live.id)
    assert np.array_equal(document.grid.voxels, snowed(base))
    opkit.call("rok.live_remove", "EXEC", live=live.id)
    assert np.array_equal(document.grid.voxels, base) and not document.scene.live
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, snowed(base)) and document.scene.live
    assert opkit.call("rok.live_bake", "EXEC") == {"FINISHED"}
    assert not document.scene.live
    cell = free_top(base)
    edit([cell], [3])  # baked: no rerun, the snow under the new voxel stays
    expect = snowed(base)
    expect[cell] = 3
    assert np.array_equal(document.grid.voxels, expect)


def test_your_paint_on_a_shaders_voxel_stays_when_it_goes(app):
    base = document.grid.voxels.copy()
    go_live()
    x, y, z = np.argwhere(document.grid.voxels == 9)[0]
    live = document.scene.live[document.active_uid][0]
    edit([(x, y, z)], [5])  # you paint over the snow (it reruns, and snows there again)
    opkit.call("rok.live_remove", "EXEC", live=live.id)
    expect = base.copy()
    expect[x, y, z] = 5  # your paint, not the colour from before the snow
    assert np.array_equal(document.grid.voxels, expect)


def _click(app, rect) -> None:
    x, y, w, h = rect
    app.handle("mouse_move", {"x": x + w / 2, "y": app.region.height - (y + h / 2)})
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def test_the_panels_live_button_and_a_change_from_the_layers_list(app):
    from rok.app import chrome, shader_panel

    base = document.grid.voxels.copy()
    top = snowed(base) == 9
    opkit.call("rok.shader_panel", "INVOKE", shader="paint_top")
    layout = shader_panel._layout(shader_panel._session, app.region)
    assert layout.live is not None
    _click(app, layout.live)
    assert shader_panel._session is None
    (live,) = document.scene.live[document.active_uid]
    assert (document.grid.voxels[top] == 9).all()
    # The Layers list shows it under the model.
    state = chrome._layer_state()
    assert (document.active_uid, live.id, "Paint Top", True, "") in state["live"]
    # Open it again: the panel changes it (Update), with the model shown without it.
    opkit.call("rok.shader_panel", "INVOKE", live=live.id)
    session = shader_panel._session
    assert session.live is not None and shader_panel._layout(session, app.region).live is None
    session.values["a0"] = 12.0
    shader_panel._preview(session)
    assert (document.grid.voxels[top] == 12).all()
    _click(app, shader_panel._layout(session, app.region).run)  # Update
    assert (document.grid.voxels[top] == 12).all()
    assert document.scene.live[document.active_uid][0].settings["a0"] == 12.0
    # Esc after a change puts the model back with the shader as it is.
    opkit.call("rok.shader_panel", "INVOKE", live=live.id)
    shader_panel._session.values["a0"] = 30.0
    shader_panel._preview(shader_panel._session)
    shader_panel._close()
    assert (document.grid.voxels[top] == 12).all()
    # Remove, from the panel (Back's place).
    opkit.call("rok.shader_panel", "INVOKE", live=live.id)
    _click(app, shader_panel._layout(shader_panel._session, app.region).back)
    assert np.array_equal(document.grid.voxels, base) and not document.scene.live


def test_save_writes_a_rok_file_next_to_the_vox_and_it_opens_live(app, tmp_path):
    import shutil

    vox = tmp_path / "house.vox"
    shutil.copy(HOUSE, vox)
    document.filepath = str(vox)
    base = document.grid.voxels.copy()
    go_live()
    assert opkit.call("rok.save", "EXEC") == {"FINISHED"}
    rok = tmp_path / "house.rok"
    assert rok.exists() and document.filepath == str(rok) and not document.modified
    assert vox.read_bytes() == HOUSE.read_bytes()  # the .vox stays as it was
    assert opkit.call("rok.open", "EXEC", filepath=str(rok)) == {"FINISHED"}
    assert np.array_equal(document.grid.voxels, snowed(base))
    assert document.scene.live[document.active_uid][0].shader == "snow_top"
    cell = free_top(base)
    edit([cell], [3])  # still live after the file round trip
    mine = base.copy()
    mine[cell] = 3
    assert np.array_equal(document.grid.voxels, snowed(mine))
    # Save As a .vox: written as asked, the shaders' voxels in it.
    out = tmp_path / "flat.vox"
    assert opkit.call("rok.save_as", "EXEC", filepath=str(out), check_existing=False) == {"FINISHED"}
    assert out.exists() and document.filepath == str(out)
