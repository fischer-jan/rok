"""CAD view: the menu, the camera, and the voxel length."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app import cad_mode, ui
from rok.app.app import RokApp
from rok.app.object import document, model_origin
from rok.app.opkit import CONTEXT
from rok.app.state import state
from rok.app.view_ops import ViewKeyAction

FIXTURES = Path(__file__).parent / "fixtures"
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


def _reset_cad() -> None:
    cad_mode._saved = None
    cad_mode._cache.clear()
    cad_mode._edges.clear()
    cad_mode._hits.clear()
    cad_mode._edit = None
    cad_mode._edit_fresh = False
    cad_mode._selected = None
    cad_mode._hover = None
    cad_mode._hidden = False
    cad_mode._ghost.clear()


@pytest.fixture
def app(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    _reset_cad()
    a = RokApp(RecordingHost(), str(FIXTURES / "bird.vox"))
    a.handle("start", {"width": W, "height": H, "scale": 2.0})
    yield a
    _reset_cad()
    document.reset()
    state.__init__()


@pytest.fixture
def empty(app):
    """An empty 16 x 16 x 8 grid, framed."""
    from rok.app import opkit

    opkit.call("rok.new", "EXEC", size=(16, 16, 8))
    app.handle("resize", {"width": W, "height": H})
    return app


def mods(app, **on) -> None:
    app.handle("modifiers", {**NO_MODS, **on})


def key(app, name: str, **on) -> None:
    mods(app, **on)
    app.handle("key_down", {"key": name, "code": name, "text": name if len(name) == 1 else "", "repeat": False})
    app.handle("key_up", {"key": name, "code": name, "text": "", "repeat": False})
    mods(app)


def click_region(app, x: float, y: float) -> None:
    """A left click at region pixels (origin bottom left)."""
    app.handle("mouse_move", {"x": float(x), "y": float(app.region.height - y)})
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def on_floor(app, x: int, y: int) -> dict:
    o = np.asarray(model_origin(document.model), dtype=np.float64)
    px, py = app.camera.project([o + (x + 0.5, y + 0.5, 0.0)])[0]
    return {"x": float(px), "y": float(py)}


def click(app, p) -> None:
    app.handle("mouse_move", p)
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def filled() -> set:
    v = document.grid.voxels
    return {tuple(int(c) for c in cell) for cell in np.argwhere(v != 0)}


def camera_of(app) -> tuple:
    cam = app.main_camera
    return (cam.ortho, cam.locked, cam.yaw, cam.pitch, cam.distance, np.array(cam.target, copy=True))


def test_the_menu_turns_the_cad_view_on_and_puts_the_camera_back(app):
    before = camera_of(app)
    app.handle("menu", {"id": "rok.cad_view"})
    assert app.cad and app.main_camera.ortho and not app.main_camera.locked
    assert app.host.named("set_cad")[-1][1] == (True,)
    assert app.host.named("set_cad_field")  # the fitted faces go to the GPU
    item = next(it for it in ui.view_menu(app)[0][1] if isinstance(it, tuple) and it[0] == "rok.cad_view")
    assert item[3] == 1
    assert ui._hint(CONTEXT) == cad_mode.HINT
    app.main_camera.yaw += 0.4
    app.handle("menu", {"id": "rok.cad_view"})
    assert not app.cad
    assert app.host.named("set_cad")[-1][1] == (False,)
    after = camera_of(app)
    assert after[0] == before[0] and after[1] == before[1]
    assert after[2] == pytest.approx(before[2])
    assert after[3] == pytest.approx(before[3])
    assert after[4] == pytest.approx(before[4])
    assert np.allclose(after[5], before[5])


def test_the_cad_view_ends_the_rendered_view_and_does_not_bring_it_back(app):
    app.handle("menu", {"id": "rok.rendered_view"})
    assert app.rendered and not app.cad
    app.handle("menu", {"id": "rok.cad_view"})
    assert app.cad and not app.rendered
    app.handle("menu", {"id": "rok.cad_view"})
    assert not app.cad and not app.rendered and not app.quad


def test_the_cad_view_ends_the_quad_view_and_does_not_bring_it_back(app):
    yaw = app.main_camera.yaw
    app.handle("menu", {"id": "rok.quad_view"})
    assert app.quad
    app.handle("menu", {"id": "rok.cad_view"})
    assert app.cad and not app.quad and app.main_camera.ortho
    app.main_camera.yaw += 0.4
    app.handle("menu", {"id": "rok.cad_view"})
    assert not app.cad and not app.quad and not app.main_camera.ortho
    assert app.main_camera.yaw == pytest.approx(yaw)


def test_the_rendered_view_and_the_quad_view_end_the_cad_view(app):
    app.handle("menu", {"id": "rok.cad_view"})
    app.handle("menu", {"id": "rok.rendered_view"})
    assert app.rendered and not app.cad and not app.main_camera.ortho
    app.handle("menu", {"id": "rok.rendered_view"})
    app.handle("menu", {"id": "rok.cad_view"})
    app.handle("menu", {"id": "rok.quad_view"})
    assert app.quad and not app.cad


def test_a_click_does_not_paint_while_the_cad_view_is_on(empty):
    empty.handle("menu", {"id": "rok.cad_view"})
    click(empty, on_floor(empty, 3, 4))
    assert filled() == set()
    empty.handle("menu", {"id": "rok.cad_view"})
    click(empty, on_floor(empty, 3, 4))
    assert filled() == {(3, 4, 0)}


def test_the_length_bar_keeps_metres_and_feet_as_one_length(app):
    app.handle("menu", {"id": "rok.cad_view"})
    scene = document.scene
    assert scene.cell_metres is None
    x, y, w, h = cad_mode._hits["field"]
    click_region(app, x + w / 2, y + h / 2)
    assert cad_mode.editing()
    assert ui._hint(CONTEXT) == cad_mode.TYPE_HINT
    assert not ViewKeyAction.poll(CONTEXT)
    yaw = app.main_camera.yaw
    key(app, "a")
    assert app.main_camera.yaw == yaw
    for ch in "0.25":
        key(app, ch)
    key(app, "Enter")
    assert not cad_mode.editing()
    assert scene.cell_metres == pytest.approx(0.25)
    assert scene.cell_unit == "m"
    assert document.modified
    document.modified = False
    cad_mode.set_length(scene, "0.25")
    assert not document.modified
    fx, fy, fw, fh = cad_mode._hits["ft"]
    click_region(app, fx + fw / 2, fy + fh / 2)
    assert scene.cell_unit == "ft" and scene.cell_metres == pytest.approx(0.25)
    assert document.modified
    # A length that is not positive stays uncommitted.
    click_region(app, x + w / 2, y + h / 2)
    key(app, "0")
    key(app, "Enter")
    assert cad_mode.editing() and scene.cell_metres == pytest.approx(0.25)
    key(app, "Escape")
    assert not cad_mode.editing()


def _project(app, point) -> tuple[float, float]:
    """Region pixels (origin bottom left) of a point in the active model's cells."""
    origin = np.asarray(model_origin(document.model), dtype=np.float64)
    px, py = app.camera.project([origin + np.asarray(point, dtype=np.float64)])[0]
    return float(px), float(app.region.height - py)


def test_a_click_on_an_edge_shows_its_length(empty):
    grid = document.grid
    grid.voxels[6:10, 6:9, 3:5] = 1
    empty.handle("menu", {"id": "rok.cad_view"})
    assert ui._hint(CONTEXT) == cad_mode.HINT
    px, py = _project(empty, (8.0, 6.0, 5.0))
    document.modified = False
    click_region(empty, px, py)
    edge = cad_mode.selected_edge()
    assert edge is not None and edge[1:] == (0, 6, 5, 6, 10)
    assert cad_mode.edge_label(document.scene) == "X 4 cells"
    assert cad_mode._hover is not None and cad_mode._hover[1:] == (0, 6, 5, 6, 10)
    assert not document.modified
    assert ui._hint(CONTEXT) == "Edge X 4 cells. Click empty space or press Esc to clear it."
    cad_mode.set_length(document.scene, "0.25")
    assert cad_mode.edge_label(document.scene) == "X 1 m"
    # The length field keeps the edge. Esc leaves the field, then clears the edge.
    x, y, w, h = cad_mode._hits["field"]
    click_region(empty, x + w / 2.0, y + h / 2.0)
    assert cad_mode.editing() and cad_mode.selected_edge() is not None
    key(empty, "Escape")
    assert not cad_mode.editing() and cad_mode.selected_edge() is not None
    key(empty, "Escape")
    assert cad_mode.selected_edge() is None
    click_region(empty, px, py)
    assert cad_mode.selected_edge() is not None
    ex, ey = _project(empty, (0.5, 0.5, 0.5))
    click_region(empty, ex, ey)
    assert cad_mode.selected_edge() is None
    assert int(grid.voxels[0, 0, 0]) == 0
    click_region(empty, px, py)
    click_region(empty, px, py)
    assert cad_mode.selected_edge() is None
    click_region(empty, px, py)
    empty.handle("menu", {"id": "rok.cad_view"})
    assert cad_mode.selected_edge() is None and not empty.cad


def test_hidden_edges_are_optional(empty):
    document.grid.voxels[6:10, 6:9, 3:5] = 1
    empty.handle("menu", {"id": "rok.cad_view"})
    assert not cad_mode.hidden()
    ids = [it[0] for it in ui.view_menu(empty)[0][1] if isinstance(it, tuple)]
    assert "rok.cad_hidden" not in ids
    assert ui._hint(CONTEXT) == cad_mode.HINT
    x, y, w, h = cad_mode._hits["hidden"]
    document.modified = False
    click_region(empty, x + w / 2.0, y + h / 2.0)
    assert cad_mode.hidden() and not document.modified
    assert empty.host.named("set_cad_hidden")[-1][1] == (True,)
    assert ui._hint(CONTEXT) == "See-through: hidden edges are dotted lines. Click an edge for its length."
    runs = cad_mode._hidden_runs(empty, empty.camera)
    assert len(runs) > 0 and np.isfinite(runs).all()
    again = cad_mode._hidden_runs(empty, empty.camera)
    assert len(again) == len(runs)
    empty.handle("menu", {"id": "rok.cad_view"})
    assert cad_mode.hidden()
    assert empty.host.named("set_cad_hidden")[-1][1] == (False,)
    empty.handle("menu", {"id": "rok.cad_view"})
    assert empty.host.named("set_cad_hidden")[-1][1] == (True,)
    click_region(empty, x + w / 2.0, y + h / 2.0)
    assert not cad_mode.hidden()
    assert empty.host.named("set_cad_hidden")[-1][1] == (False,)


def test_an_undo_of_a_voxel_keeps_the_voxel_length(empty):
    empty.handle("menu", {"id": "rok.cad_view"})
    cad_mode.set_length(document.scene, "0.25")
    empty.handle("menu", {"id": "rok.cad_view"})
    click(empty, on_floor(empty, 3, 4))
    assert filled() == {(3, 4, 0)}
    key(empty, "z", ctrl=True)
    assert filled() == set()
    assert document.scene.cell_metres == pytest.approx(0.25)
    assert document.scene.cell_unit == "m"
