"""The scene's group tools in the app (rok/app/scene_arrange.py): Pick Linked
Copies, Align, Array and Carve with Model. Each is one undo step; a card's
Cancel leaves no trace."""

from __future__ import annotations

import numpy as np

from rok.app import scene_tools
from rok.app.object import document
from rok.app.state import state
from rok.core import layers as layers_core
from rok.core import link as link_core
from test_app_scene import app, run  # noqa: F401 - the fixture

assert app


def models():
    return {m.name: m for m in document.scene.models}


def pick_both():
    scene = document.scene
    state.group.clear()
    state.group |= {m.uid for m in scene.models}
    document.set_active(scene.models[0].uid)


def undo():
    run("rok.undo")


def test_align_lines_the_group_up_on_the_picked_model_as_one_step(app):
    pick_both()
    robot, car = models()["Robot"], models()["Car"]
    car_before = tuple(car.position)
    assert run("rok.scene_align", axis="1", how="MIN") == {"FINISHED"}
    assert car.position[1] == robot.position[1] and car.position[0] == car_before[0]
    undo()
    assert tuple(car.position) == car_before


def test_align_card_cancel_moves_nothing(app):
    pick_both()
    car = models()["Car"]
    before = tuple(car.position)
    run("rok.scene_align", "INVOKE")
    app.dialog.op.how = "MAX"
    app.dialog._finish(False)
    assert tuple(car.position) == before and not state.turn_preview


def test_array_makes_linked_copies_in_a_row_and_undo_takes_them_away(app):
    count = len(document.scene.models)
    robot = models()["Robot"]
    document.set_active(robot.uid)
    state.group.clear()
    assert run("rok.scene_array", count=3, axis="0", direction="1", gap=2) == {"FINISHED"}
    scene = document.scene
    assert len(scene.models) == count + 2
    copies = [m for m in scene.models if m.link == robot.link and m is not robot]
    step = int(robot.grid.size[0]) + 2
    assert sorted(int(m.position[0]) - int(robot.position[0]) for m in copies) == [step, 2 * step]
    undo()
    assert len(scene.models) == count and not link_core.is_linked(scene, robot)


def test_array_card_shows_copies_live_and_cancel_removes_them(app):
    count = len(document.scene.models)
    document.set_active(models()["Robot"].uid)
    state.group.clear()
    run("rok.scene_array", "INVOKE")
    assert len(document.scene.models) == count + 2  # shown at once
    app.dialog.op.count = 5
    assert len(document.scene.models) == count + 4
    app.dialog._finish(False)
    assert len(document.scene.models) == count and len(document.undo._undo) == 0


def test_pick_linked_copies_picks_the_whole_group(app):
    robot = models()["Robot"]
    document.set_active(robot.uid)
    state.group.clear()
    run("rok.scene_array", count=3)
    state.group.clear()
    assert run("rok.scene_pick_linked") == {"FINISHED"}
    assert scene_tools.group_uids() == {m.uid for m in link_core.members(document.scene, robot.link)}
    document.set_active(models()["Car"].uid)
    state.group.clear()
    assert run("rok.scene_pick_linked") == {"CANCELLED"}  # not a linked copy


def test_carve_cuts_the_cutters_shape_out_and_hides_the_cutter(app):
    scene = document.scene
    robot, car = models()["Robot"], models()["Car"]
    car.position = tuple(robot.position)  # the car overlaps the robot
    before = robot.grid.voxels.copy()
    hole = (before != 0) & np.pad(car.grid.voxels != 0, [(0, max(0, b - c)) for b, c in
                                                          zip(before.shape, car.grid.voxels.shape)])[
        tuple(slice(0, n) for n in before.shape)]
    assert hole.any() and (~hole & (before != 0)).any()  # a real hole, and voxels that stay
    document.set_active(car.uid)
    state.group.clear()
    assert run("rok.scene_carve") == {"FINISHED"}
    assert (robot.grid.voxels[hole] == 0).all() and (robot.grid.voxels[~hole] == before[~hole]).all()
    layer = next(i for i in layers_core.infos(scene.layers) if i.id == int(car.layer))
    assert layer.name == "Cutters" and layer.hidden and document.active_uid == robot.uid
    undo()
    assert (robot.grid.voxels == before).all() and not scene.is_hidden(car)


def test_carve_reaches_every_linked_copy(app):
    scene = document.scene
    robot, car = models()["Robot"], models()["Car"]
    document.set_active(robot.uid)
    state.group.clear()
    run("rok.scene_array", count=2, gap=10)
    copy_ = next(m for m in scene.models if m.link == robot.link and m is not robot)
    car.position = tuple(robot.position)
    document.set_active(car.uid)
    carved_before = (robot.grid.voxels != 0).sum()
    run("rok.scene_carve")
    assert (robot.grid.voxels != 0).sum() < carved_before
    assert (copy_.grid.voxels == robot.grid.voxels).all()  # the copy took the hole too


def test_carve_without_overlap_changes_nothing(app):
    robot, car = models()["Robot"], models()["Car"]
    car.position = (int(robot.position[0]) + 500, 0, 0)
    document.set_active(car.uid)
    state.group.clear()
    assert run("rok.scene_carve") == {"CANCELLED"}


def test_a_carve_drop_uses_up_the_cutter_as_one_step(app):
    from rok.app import scene_arrange

    scene = document.scene
    robot, car = models()["Robot"], models()["Car"]
    before = robot.grid.voxels.copy()
    count = len(scene.models)
    note = scene_arrange.carve_drop(car, tuple(robot.position), keep=False)
    assert note.startswith("Carved 1 model") and len(scene.models) == count - 1
    assert (robot.grid.voxels != before).any() and document.active_uid == robot.uid
    undo()
    assert len(scene.models) == count and (robot.grid.voxels == before).all()


def test_a_carve_drop_with_shift_keeps_the_cutter_where_it_was(app):
    from rok.app import scene_arrange

    robot, car = models()["Robot"], models()["Car"]
    start = tuple(car.position)
    count = len(document.scene.models)
    scene_arrange.carve_drop(car, tuple(robot.position), keep=True)
    assert len(document.scene.models) == count and tuple(car.position) == start
    far = (int(robot.position[0]) + 500, 0, 0)
    assert scene_arrange.carve_drop(car, far, keep=False).startswith("Drop the model onto another")
    assert len(document.scene.models) == count  # nothing under it: nothing used up


def test_a_linked_copy_carves_its_original_and_is_made_unique_in_the_same_step(app):
    from rok.app import scene_arrange

    scene = document.scene
    robot = models()["Robot"]
    document.set_active(robot.uid)
    state.group.clear()
    run("rok.scene_array", count=3, gap=10)
    cutter, other = [m for m in scene.models if m.link == robot.link and m is not robot]
    link, shape, before = robot.link, cutter.grid.voxels.copy(), robot.grid.voxels.copy()
    drop = (int(robot.position[0]) + 3, int(robot.position[1]), int(robot.position[2]))
    note = scene_arrange.carve_drop(cutter, drop, keep=True)
    assert note.startswith("Carved"), note
    assert (robot.grid.voxels != before).any() and (other.grid.voxels == robot.grid.voxels).all()
    assert cutter.link != link and (cutter.grid.voxels == shape).all()  # unique, its shape kept
    undo()
    assert cutter.link == link and (robot.grid.voxels == before).all() and (other.grid.voxels == before).all()


def test_a_linked_copy_that_misses_stays_linked(app):
    from rok.app import scene_arrange

    robot = models()["Robot"]
    document.set_active(robot.uid)
    state.group.clear()
    run("rok.scene_array", count=2, gap=10)
    cutter = next(m for m in document.scene.models if m.link == robot.link and m is not robot)
    far = (int(robot.position[0]) + 500, 0, 0)
    assert scene_arrange.carve_drop(cutter, far, keep=True).startswith("Drop the model onto another")
    assert cutter.link == robot.link


def test_a_carve_drag_from_a_gizmo_arrow_moves_along_that_axis_only(app):
    from rok.app import opkit

    car = models()["Car"]
    document.set_active(car.uid)
    start = tuple(int(p) for p in car.position)
    region = opkit.CONTEXT.region
    x, y = region.width / 2, region.height / 2
    event = type("E", (), {"mouse_region_x": x, "mouse_region_y": y})()
    op = scene_tools.SceneDragAction()
    op.carve, op.keep, op.axis, op.pressed, op.grab_x, op.grab_y = True, True, 0, car.uid, int(x), int(y)
    assert op.invoke(opkit.CONTEXT, event) == {"RUNNING"}
    move = type("M", (), {"type": "MOUSE_MOVE", "value": "NOTHING", "alt": False, "cmd_key": False,
                          "mouse_region_x": x + 150, "mouse_region_y": y + 90})()
    op.modal(opkit.CONTEXT, move)
    op.modal(opkit.CONTEXT, move)
    assert op._offset[0] != 0 and op._offset[1:] == (0, 0)
    op._finish(opkit.CONTEXT, cancel=True)
    assert tuple(int(p) for p in car.position) == start


def test_ctrl_press_on_a_gizmo_arrow_carves_along_it_and_other_handles_pass(app, monkeypatch):
    from rok.app import opkit
    from rok.app import scene_arrange
    from rok.app.tools import StrokeAction

    car = models()["Car"]
    document.set_active(car.uid)
    calls = []
    monkeypatch.setattr(opkit, "call", lambda idname, how, **props: calls.append((idname, props)) or {"RUNNING"})
    event = type("E", (), {"mouse_region_x": 10, "mouse_region_y": 20})()
    op = scene_arrange.SceneCarveDragAction()
    op.keep = True
    monkeypatch.setattr(StrokeAction, "handle_at", staticmethod(lambda *a: "MOVE_Z"))
    assert op.invoke(opkit.CONTEXT, event) == {"RUNNING"}
    assert calls == [("rok.scene_drag", {"carve": True, "keep": True, "pressed": car.uid, "axis": 2,
                                          "grab_x": 10, "grab_y": 20})]
    monkeypatch.setattr(StrokeAction, "handle_at", staticmethod(lambda *a: "ROT_X"))
    assert op.invoke(opkit.CONTEXT, event) == {"PASS"}


def test_ctrl_press_beside_the_models_orbits_and_on_a_model_starts_a_carve_drag(app, monkeypatch):
    from rok.app import mode, opkit

    names_ = [item.idname for item in app.keymap()]
    assert names_.index("rok.scene_carve_drag") < names_.index("rok.stroke")  # before the strokes
    monkeypatch.setattr(mode, "models_hit", lambda *a: ([], {}))
    assert run("rok.scene_carve_drag", "INVOKE") == {"PASS"}
    car = models()["Car"]
    calls = []
    monkeypatch.setattr(mode, "models_hit", lambda *a: ([car.uid], {car.uid: 1.0}))
    monkeypatch.setattr(opkit, "call", lambda idname, how, **props: calls.append((idname, props)) or {"RUNNING"})
    scene_arrange_mod = __import__("rok.app.scene_arrange", fromlist=["x"])
    op = scene_arrange_mod.SceneCarveDragAction()
    op.keep = True
    event = type("E", (), {"mouse_region_x": 10, "mouse_region_y": 20})()
    assert op.invoke(opkit.CONTEXT, event) == {"RUNNING"}
    assert calls == [("rok.scene_drag", {"carve": True, "keep": True, "pressed": car.uid, "grab_x": 10, "grab_y": 20})]
    assert document.active_uid == car.uid
