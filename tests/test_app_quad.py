"""View > Quad View (rok/app/quad.py and the app's views) against a recording host."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app import nav_cube, opkit
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.opkit import CONTEXT
from rok.app.state import state

FIXTURES = Path(__file__).parent / "fixtures"
W, H = 1400.0, 900.0


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
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "robot.vox"))
    a.handle("start", {"width": W, "height": H, "scale": 2.0})
    opkit.call("rok.quad_view", "EXEC")
    yield a
    a.set_quad(False)
    document.reset()
    state.__init__()


def _hidden(vox, cell, n) -> bool:
    """True when another voxel lies between the cell and the viewer."""
    p = np.asarray(cell) + n
    while np.all((p >= 0) & (p < np.asarray(vox.shape))):
        if vox[tuple(p)]:
            return True
        p = p + n
    return False


def view(app, name: str):
    return next(v for v in app.views if v.name == name)


def host_point_of(app, view_state, world) -> dict:
    """The host pixel (origin top left of the window) where `world` shows in a view."""
    x, y = view_state.camera.project([world])[0]
    rx, ry = view_state.from_camera(app.region, x, y)
    return {"x": float(rx), "y": float(app.height - ry)}


def test_the_quad_view_has_four_views_and_the_host_gets_them(app):
    assert [v.name for v in app.views] == ["Top", "Perspective", "Front", "Right"]
    app.after_event()
    views = app.host.named("set_views")[-1][1][0]
    assert len(views) == 4
    rects = sorted((x, y, w, h) for x, y, w, h, *_rest in views)
    assert rects == [(0.0, 0.0, W / 2, H / 2), (0.0, H / 2, W / 2, H / 2), (W / 2, 0.0, W / 2, H / 2), (W / 2, H / 2, W / 2, H / 2)]
    pixels = {v.name: v.camera.pixel for v in app.views}
    assert pixels["Perspective"] == 0.0 and all(pixels[n] > 0 for n in ("Top", "Front", "Right"))


def test_the_mouse_works_in_the_view_under_it_and_a_drag_stays_there(app):
    app.handle("mouse_move", {"x": W * 0.25, "y": H * 0.25})
    assert CONTEXT.view_state.name == "Top"
    app.handle("mouse_down", {"button": "middle"})
    app.handle("mouse_move", {"x": W * 0.75, "y": H * 0.75})
    assert CONTEXT.view_state.name == "Top"
    app.handle("mouse_up", {"button": "middle"})
    app.handle("mouse_move", {"x": W * 0.75, "y": H * 0.76})
    assert CONTEXT.view_state.name == "Right"


@pytest.mark.parametrize("name, normal", [("Top", (0, 0, 1)), ("Front", (0, -1, 0)), ("Right", (1, 0, 0))])
def test_a_click_in_a_side_view_attaches_on_the_face_that_view_sees(app, name, normal):
    model = document.model
    vox = model.grid.voxels
    axis = int(np.flatnonzero(normal)[0])
    sign = int(np.sign(normal[axis]))
    # A filled cell whose neighbour toward the view is inside the grid and empty,
    # and that nothing hides from the view.
    n = np.asarray(normal)
    ahead = np.argwhere(vox != 0) + n
    inside = np.all((ahead >= 0) & (ahead < np.asarray(vox.shape)), axis=1)
    filled = np.argwhere(vox != 0)[inside]
    filled = np.array([c for c in filled if not _hidden(vox, c, n)])
    # The outermost filled cell along the view's direction, near the middle.
    mid = np.asarray(vox.shape) / 2.0
    others = [k for k in range(3) if k != axis]
    order = np.argsort(-sign * filled[:, axis] * 1000 + np.linalg.norm(filled[:, others] - mid[others], axis=1))
    cell = filled[order[0]]
    view_state = view(app, name)
    face = np.asarray(model.position, float) + cell + 0.5 + 0.5 * np.asarray(normal, float)
    p = host_point_of(app, view_state, face)
    app.handle("mouse_move", p)
    assert CONTEXT.view_state is view_state
    before = vox.copy()
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})
    changed = np.argwhere(model.grid.voxels != before)
    assert changed.tolist() == [list(cell + np.asarray(normal))]


def test_the_side_views_do_not_turn_but_zoom_and_pan(app):
    top = view(app, "Top")
    app.handle("mouse_move", host_point_of(app, top, top.camera.target))
    yaw, pitch, dist = top.camera.yaw, top.camera.pitch, top.camera.distance
    app.handle("scroll", {"dx": 30.0, "dy": 20.0, "precise": True})  # two fingers orbit elsewhere
    assert (top.camera.yaw, top.camera.pitch) == (yaw, pitch)
    app.handle("pinch", {"delta": 0.2})
    assert top.camera.distance < dist
    target = top.camera.target.copy()
    bx, by = nav_cube.buttons(app.region, top)["PAN"]
    app.handle("mouse_move", {"x": bx, "y": app.height - by})
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", {"x": bx + 40, "y": app.height - by})
    app.handle("mouse_up", {"button": "left"})
    assert not np.allclose(top.camera.target, target)


def test_only_the_perspective_view_has_the_cube_and_the_quad_button(app):
    assert nav_cube.has_cube(view(app, "Perspective"))
    assert not nav_cube.has_cube(view(app, "Front"))
    assert "QUAD" in nav_cube.buttons(app.region, view(app, "Perspective"))
    assert "QUAD" not in nav_cube.buttons(app.region, view(app, "Top"))


def test_the_quad_button_and_the_command_switch_back_to_one_view(app):
    persp = view(app, "Perspective")
    bx, by = nav_cube.buttons(app.region, persp)["QUAD"]
    app.handle("mouse_move", {"x": bx, "y": app.height - by})
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})
    assert not app.quad and app.views == [app.view_state]
    assert app.camera.width == W and app.camera.height == H
    app.handle("menu", {"id": "rok.quad_view"})
    assert app.quad
    assert "Quad View" not in str(app.host.named("set_menu")[-1][1][0])  # the nav cube has it


def test_a_resize_lays_the_four_views_out_again(app):
    app.handle("resize", {"width": 1000.0, "height": 600.0, "scale": 2.0})
    assert view(app, "Right").rect == (500.0, 0.0, 500.0, 300.0)
    assert view(app, "Top").camera.width == 500.0
