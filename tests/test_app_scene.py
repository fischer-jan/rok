"""The scene's model list and whole-model tools (rok/app/scene_ops.py,
scene_tools.py), run as operators against a host that only records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app import opkit
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state
from rok.core import png
from rok.core.vox_read import read_scene

FIXTURES = Path(__file__).parent / "fixtures"
TWO = FIXTURES / "phase3" / "two_models.vox"


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
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(TWO))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    yield a
    document.reset()
    state.__init__()


def run(idname: str, how: str = "EXEC", **props) -> set:
    return opkit.call(idname, how, **props)


def names() -> list[str]:
    return [m.name for m in document.scene.models]


def test_a_file_of_two_models_opens_in_the_scene(app):
    assert state.level == "SCENE" and names() == ["Robot", "Car"]


def test_new_model_adds_an_empty_grid_right_of_the_scene_and_undo_removes_it(app):
    _lo, hi = document.scene.world_bounds()
    assert run("rok.model_add") == {"FINISHED"}
    new = document.model
    assert len(document.scene.models) == 3 and new.grid.size == (32, 32, 32)
    assert new.position[0] > hi[0] and not new.grid.voxels.any()
    assert document.modified
    run("rok.undo")
    assert names() == ["Robot", "Car"]
    # The GPU has the model list of the scene again.
    assert app.host.named("remove_model") or app.host.named("set_model")


def test_duplicate_copies_the_active_model_beside_it(app):
    document.set_active(document.scene.models[0].uid)
    src = document.model
    run("rok.model_duplicate")
    copy = document.model
    assert copy.uid != src.uid and np.array_equal(copy.grid.voxels, src.grid.voxels)
    assert copy.position[0] == src.position[0] + src.grid.size[0] + 2


def test_rename_opens_a_dialog_with_the_name_and_renames(app):
    run("rok.model_rename", "INVOKE")
    assert app.dialog is not None and app.dialog.op.name == document.model.name
    app.dialog.op.name = "Scout"
    app.dialog._finish(True)
    assert document.model.name == "Scout"
    run("rok.undo")
    assert document.model.name != "Scout"


def test_delete_keeps_at_least_one_model(app):
    run("rok.model_delete")
    assert len(document.scene.models) == 1
    assert run("rok.model_delete") == {"CANCELLED"}
    assert "A scene needs one model" in app.message


def test_merge_all_makes_one_model_of_every_voxel(app):
    total = sum(int((m.grid.voxels != 0).sum()) for m in document.scene.models)
    run("rok.model_merge_all")
    assert len(document.scene.models) == 1
    assert int((document.model.grid.voxels != 0).sum()) == total


def test_add_from_file_brings_the_file_models_next_to_the_scene(app):
    op_result = run("rok.model_add_file", filepath=str(FIXTURES / "robot.vox"))
    assert op_result == {"FINISHED"} and len(document.scene.models) == 3
    added = document.model
    src = read_scene(FIXTURES / "robot.vox").models[0].voxels
    assert added.grid.size == src.shape and int((added.grid.voxels != 0).sum()) == int((src != 0).sum())


def test_new_from_heightmap_makes_columns_as_tall_as_the_pixels_are_bright(app, tmp_path):
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[0, 0, :3] = 255  # one white pixel: the tallest column
    path = tmp_path / "hill.png"
    path.write_bytes(png.encode(rgba))
    assert run("rok.heightmap_new", filepath=str(path), height=10, image_colours=False) == {"FINISHED"}
    vox = document.model.grid.voxels
    heights = (vox != 0).sum(axis=2)
    assert heights.max() == 10


def test_new_terrain_makes_tiles_that_meet_without_a_seam(app):
    pytest.importorskip("rok_native")  # the terrain runs a shader on the GPU
    before = len(document.scene.models)
    result = run("rok.terrain_new", across=2, deep=1, size=32, height=16, seed=7)
    assert result == {"FINISHED"}
    tiles = document.scene.models[before:]
    assert len(tiles) == 2 and all(t.grid.voxels.any() for t in tiles)
    a, b = tiles
    # Side by side along x; the ground runs on across the edge.
    assert b.position[0] == a.position[0] + 32
    edge_a = (a.grid.voxels[-1] != 0).sum(axis=1)
    edge_b = (b.grid.voxels[0] != 0).sum(axis=1)
    assert np.abs(edge_a.astype(int) - edge_b.astype(int)).max() <= 3


def test_drop_to_floor_moves_the_picked_model_onto_z_zero(app):
    robot = document.scene.models[0]
    car = document.scene.models[1]
    document.set_active(car.uid)
    lifted = tuple(car.position)
    assert run("rok.scene_drop") in ({"FINISHED"}, {"CANCELLED"})
    lowest = int(np.nonzero((car.grid.voxels != 0).any(axis=(0, 1)))[0].min())
    assert car.position[2] + lowest == 0
    assert robot.position[2] == 0 or robot.position != lifted


def test_fit_shrinks_the_grid_to_its_voxels_and_keeps_them_in_place(app):
    model = document.model
    world = {tuple(np.add(c, model.position)) for c in np.argwhere(model.grid.voxels != 0)}
    grown = np.zeros(tuple(s + 4 for s in model.grid.size), dtype=np.uint8)
    grown[2:-2, 2:-2, 2:-2] = model.grid.voxels
    model.grid.replace_voxels(grown)
    model.position = tuple(p - 2 for p in model.position)
    assert run("rok.scene_fit") == {"FINISHED"}
    again = {tuple(np.add(c, model.position)) for c in np.argwhere(model.grid.voxels != 0)}
    assert again == world


def test_cut_splits_a_model_in_two_models_and_keeps_every_voxel(app):
    model = document.model
    world = {tuple(np.add(c, model.position)) for c in np.argwhere(model.grid.voxels != 0)}
    axis = int(np.argmax(model.grid.size))
    at = model.grid.size[axis] // 2
    others = {m.uid for m in document.scene.models if m.uid != model.uid}
    assert run("rok.scene_cut", uid=model.uid, axis=axis, at=at) == {"FINISHED"}
    assert len(document.scene.models) == 3
    parts = [m for m in document.scene.models if m.uid not in others]
    again = set()
    for m in parts:
        again |= {tuple(np.add(c, m.position)) for c in np.argwhere(m.grid.voxels != 0)}
    assert again == world  # every voxel kept its place
    run("rok.undo")
    assert len(document.scene.models) == 2


def test_the_scene_menu_is_not_in_the_menu_bar(app):
    titles = [t for t, _items in app.host.named("set_menu")[-1][1][0]]
    assert "Scene" not in titles  # the rail has the scene commands


def test_a_moving_model_and_its_box_stay_together(app, monkeypatch):
    """In a scene move the box ran ahead of the model (twice the
    offset) and snapped back on release. The model shows moved through its
    preview matrix; the box is drawn with that matrix and must not add the offset too."""
    from rok.app import gizmo_ops, overlay
    from rok.app.object import model_origin
    from rok.core import gizmo

    seen = []
    monkeypatch.setattr(overlay, "draw_scene", lambda size, origin, *a, **k: seen.append((origin, k)))
    model = document.model
    start = np.asarray(model_origin(model), dtype=float)
    gizmo_ops.begin_model_drag()
    state.stroke_active = True
    try:
        gizmo_ops.update_model_drag(gizmo.GizmoResult(kind=gizmo.MOVE, offset=(5, 0, 2)))
        app.redraw()
    finally:
        state.stroke_active = False
    origin, kw = seen[-1]
    shown = np.asarray(origin, dtype=float)
    matrix = kw.get("matrix")
    if matrix is not None:
        shown = shown + np.asarray(matrix)[:3, 3]
    assert np.allclose(shown, start + (5, 0, 2))  # where the model shows
    gs = kw.get("gizmo_state")
    if gs is not None:
        assert np.allclose(np.asarray(gs.origin, dtype=float), start + (5, 0, 2))  # the gizmo too
