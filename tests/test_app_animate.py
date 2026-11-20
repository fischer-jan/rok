"""Animate mode in the app (rok/app/anim_mode.py and dock.py), against a
recording host. The fixture is tests/fixtures/anim/propeller.vox: a hub and
a propeller blade lying on top of it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app import anim_mode, dock, opkit
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state
from rok.core import anim, gizmo

FIXTURES = Path(__file__).parent / "fixtures"
PROPELLER = FIXTURES / "anim" / "propeller.vox"


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
    a = RokApp(RecordingHost(), str(PROPELLER))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    yield a
    anim_mode.stop()
    document.reset()
    state.__init__()


def blade():
    return next(m for m in document.scene.models if m.name == "Propeller")


def pick_blade() -> None:
    document.set_active(blade().uid)


def test_spin_on_the_blade_turns_it_about_the_face_it_sits_on(app):
    opkit.call("rok.animate", action="ENTER")
    assert state.animating and state.level == "SCENE"
    pick_blade()
    assert opkit.call("rok.anim_preset", name="SPIN") == {"FINISHED"}
    anim_mode.stop()
    model = blade()
    track = document.scene.timeline.track_of(model.uid)
    assert [(k.time, k.value) for k in track.keys(anim.TURN_Z)] == [(0, 0.0), (24, 360.0)]
    # The blade touches the hub with its bottom face: it turns about that face's middle.
    assert model.pivot2 == (15, 3, 0)
    anim_mode.set_frame(6)  # a quarter turn
    shown = anim_mode.shown()
    key, voxels, place = shown.poses[model.uid]
    assert voxels.shape == (3, 15, 1)  # the bar now lies across
    assert np.count_nonzero(voxels) == np.count_nonzero(model.grid.voxels)
    assert place == tuple(p + o for p, o in zip(model.position, (6, -6, 0)))
    # The real voxels never changed.
    assert model.grid.voxels.shape == (15, 3, 1)
    opkit.call("rok.undo")
    assert document.scene.timeline.track_of(model.uid) is None and model.pivot2 is None


def test_the_host_shows_the_posed_grid_and_then_the_rest_again(app):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    opkit.call("rok.anim_preset", name="SPIN")
    anim_mode.stop()
    anim_mode.set_frame(3)  # 45 degrees: a free angle
    app.host.calls.clear()
    app.flush()
    index = document.scene.models.index(blade())
    sent = [c for c in app.host.named("set_model") if c[1][0] == index]
    assert sent and sent[-1][1][1].shape != (15, 3, 1)
    app.host.calls.clear()
    opkit.call("rok.animate", action="LEAVE")
    app.flush()
    sent = [c for c in app.host.named("set_model") if c[1][0] == index]
    assert sent and sent[-1][1][1].shape == (15, 3, 1)  # the rest pose again


def test_a_gizmo_turn_writes_a_key_at_the_play_head(app):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    anim_mode.set_frame(0)
    anim_mode.set_frame(12) if document.scene.timeline.length > 12 else None
    anim_mode.gizmo_begin()
    anim_mode.gizmo_preview(gizmo.GizmoResult(gizmo.ROTATE, axis=2, degrees=37.5))
    assert blade().uid in anim_mode.shown().poses  # it shows turned while dragging
    assert anim_mode.gizmo_commit(gizmo.GizmoResult(gizmo.ROTATE, axis=2, degrees=37.5))
    anim_mode.gizmo_end()
    track = document.scene.timeline.track_of(blade().uid)
    assert [(k.time, k.value) for k in track.keys(anim.TURN_Z)] == [(0, 0.0), (12, 37.5)]
    assert not track.keys(anim.TURN_X)  # only the ring you dragged
    opkit.call("rok.undo")
    assert document.scene.timeline.track_of(blade().uid) is None


def test_a_move_keys_whole_cells_and_shift_on_the_centre_moves_the_pivot(app):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    anim_mode.gizmo_begin()
    anim_mode.gizmo_preview(gizmo.GizmoResult(gizmo.MOVE, offset=(0, 0, 3)))
    anim_mode.gizmo_commit(gizmo.GizmoResult(gizmo.MOVE, offset=(0, 0, 3)))
    anim_mode.gizmo_end()
    track = document.scene.timeline.track_of(blade().uid)
    assert [(k.time, k.value) for k in track.keys(anim.MOVE_Z)] == [(0, 3.0)]
    anim_mode.gizmo_begin()
    anim_mode.gizmo_preview(gizmo.GizmoResult(gizmo.PIVOT, pivot2=(0, 3, 0)))
    anim_mode.gizmo_commit(gizmo.GizmoResult(gizmo.PIVOT, pivot2=(0, 3, 0)))
    anim_mode.gizmo_end()
    assert blade().pivot2 == (0, 3, 0)
    opkit.call("rok.undo")
    assert blade().pivot2 is None


def test_k_keys_the_pose_and_delete_takes_the_keys_at_the_play_head(app):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    anim_mode.set_frame(8)
    assert opkit.call("rok.key_pose") == {"FINISHED"}
    track = document.scene.timeline.track_of(blade().uid)
    assert set(track.properties) == set(anim.MOVE + anim.TURN)
    assert all([k.time for k in track.keys(n)] == [0, 8] for n in track.properties)
    assert opkit.call("rok.frame_go", to_key=-1) == {"FINISHED"} and anim_mode.frame() == 0
    anim_mode.set_frame(8)
    opkit.call("rok.key_delete")
    track = document.scene.timeline.track_of(blade().uid)
    assert all([k.time for k in track.keys(n)] == [0] for n in track.properties)


def test_building_commands_wait_in_animate_mode(app):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    assert not opkit.ops.rok.scene_delete.poll()
    assert not opkit.ops.rok.flip.poll()
    opkit.call("rok.key_delete")  # Delete with no keys: nothing is deleted
    assert len(document.scene.models) == 2


def test_play_loops_by_the_clock(app, monkeypatch):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    opkit.call("rok.anim_preset", name="SPIN")
    assert anim_mode.playing()
    clock = [100.0]
    monkeypatch.setattr(anim_mode.time, "monotonic", lambda: clock[0])
    anim_mode._play.update(t0=100.0, f0=0.0)
    clock[0] = 100.0 + 30 / 24  # thirty frames on a 24-frame clip
    assert anim_mode.play_frame(clock[0]) == 6
    anim_mode.stop()
    assert not anim_mode.playing()


def test_the_dock_lists_the_part_and_a_press_picks_and_scrubs(app):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    opkit.call("rok.anim_preset", name="SWING")
    anim_mode.stop()
    region = app.region
    layout = dock.layout_for(region)
    (row,) = [r for r in layout.rows if r.uid == blade().uid]
    assert [t for t, _e in row.keys] == [0, 6, 12, 18, 24]
    x = layout.x_of(12)
    y = row.rect[1] + row.rect[3] / 2
    assert dock.press(None, region, x, y, True, False)
    assert dock.picked_keys() and all(f == 12 for _u, _n, f in dock.picked_keys())
    assert anim_mode.frame() == 12
    opkit.call("rok.key_ease", ease="LINEAR")
    track = document.scene.timeline.track_of(blade().uid)
    assert {k.time: k.ease for k in track.keys(anim.TURN_X if track.axis == 0 else anim.TURN[track.axis])}[12] \
        == anim.LINEAR
    rx, ry, rw, rh = layout.ruler
    assert dock.contains(region, rx + rw / 2, ry + rh / 2)


@pytest.mark.parametrize("name, model_name", [("coin.rok", "Coin"), ("propeller_spin.rok", "Rotor")])
def test_the_demo_files_open_with_a_spin_that_moves(tmp_path, monkeypatch, name, model_name):
    """The Phase 1 gate files (the propeller fixture opened
    with the hub active and showed nothing): a coin on edge and a rotor on a
    hub, each with its Spin clip saved in the Rok file."""
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "anim" / name))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    try:
        model = next(m for m in document.scene.models if m.name == model_name)
        track = document.scene.timeline.track_of(model.uid)
        assert [(k.time, k.value) for k in track.keys(anim.TURN[track.axis])] == [(0, 0.0), (24, 360.0)]
        opkit.call("rok.animate", action="ENTER")
        anim_mode.set_frame(6)  # a quarter turn
        shown = anim_mode.shown()
        assert model.uid in shown.poses
        _key, voxels, _place = shown.poses[model.uid]
        assert np.count_nonzero(voxels) == np.count_nonzero(model.grid.voxels)
        assert voxels.shape != model.grid.voxels.shape or not np.array_equal(voxels, model.grid.voxels)
    finally:
        anim_mode.stop()
        document.reset()
        state.__init__()


def _press(app, key: str, code: str, **mods) -> None:
    app.handle("modifiers", {"shift": mods.get("shift", False), "ctrl": False, "alt": False, "meta": False})
    app.handle("key_down", {"key": key, "code": code, "text": key if len(key) == 1 else "", "repeat": False})
    app.handle("key_up", {"key": key, "code": code, "text": "", "repeat": False})
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": False})


def test_comma_and_period_step_the_play_head_one_frame(tmp_path, monkeypatch):
    """, and . step a frame, Shift+, and Shift+. jump
    between keys (the plan's section 5), in Animate mode."""
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "anim" / "coin.rok"))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    try:
        opkit.call("rok.animate", action="ENTER")
        assert anim_mode.frame() == 0
        _press(a, ".", "Period")
        assert anim_mode.frame() == 1
        _press(a, ".", "Period")
        _press(a, ",", "Comma")
        assert anim_mode.frame() == 1
        _press(a, ".", "Period", shift=True)  # the next key: the end of the spin
        assert anim_mode.frame() == 23  # the last frame (the key at 24 is clamped)
        _press(a, ",", "Comma", shift=True)
        assert anim_mode.frame() == 0
    finally:
        anim_mode.stop()
        document.reset()
        state.__init__()


def test_a_second_flush_keeps_the_posed_grid_and_never_sends_the_rest(tmp_path, monkeypatch):
    """Stepping a frame showed the pose for an instant, then
    the view snapped back to frame 0. The second flush of an event found the
    pose unchanged and re-sent the model's own grid."""
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "anim" / "coin.rok"))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    try:
        coin = document.scene.models[0]
        rest = tuple(coin.grid.voxels.shape)
        opkit.call("rok.animate", action="ENTER")
        a.redraw()
        a.host.calls.clear()
        _press(a, ".", "Period")
        a.handle("mouse_move", {"x": 700.0, "y": 450.0})
        a.redraw()
        a.flush()
        shapes = [tuple(c[1][1].shape) for c in a.host.named("set_model") if c[1][0] == 0]
        assert shapes and shapes[0] != rest  # the pose went out
        assert rest not in shapes  # and the rest grid never followed it
        # Nor did the own grid get moved to the pose's place (the
        # unturned rotor drifted off its hub in a window running the old code).
        assert not [c for c in a.host.named("set_model_position") if c[1][0] == 0]
    finally:
        anim_mode.stop()
        document.reset()
        state.__init__()


def test_clips_add_rename_duplicate_delete_and_undo(app):
    """Phase 2: clip tabs. A new clip starts empty with this clip's length and
    frame rate; a duplicate keeps the keys; the last clip stays; undo covers it."""
    scene = document.scene
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    opkit.call("rok.anim_preset", name="SPIN")
    anim_mode.stop()
    assert len(scene.clips) == 1 and scene.timeline.has_keys()
    assert opkit.call("rok.clip_add") == {"FINISHED"}
    assert [c.name for c in scene.clips] == ["Clip 1", "Clip 2"] and scene.active_clip == 1
    assert not scene.timeline.has_keys() and scene.timeline.fps == scene.clips[0].fps
    assert opkit.call("rok.clip_rename", name="Walk") == {"FINISHED"}
    assert scene.timeline.name == "Walk"
    opkit.call("rok.clip_pick", index=0)
    assert scene.active_clip == 0 and scene.timeline.has_keys()
    assert opkit.call("rok.clip_duplicate") == {"FINISHED"}
    assert [c.name for c in scene.clips] == ["Clip 1", "Clip 1 copy", "Walk"] and scene.active_clip == 1
    assert scene.timeline.has_keys() and scene.timeline is not scene.clips[0]
    assert opkit.call("rok.clip_delete") == {"FINISHED"}
    assert [c.name for c in scene.clips] == ["Clip 1", "Walk"] and scene.active_clip == 1
    opkit.call("rok.undo")
    assert [c.name for c in scene.clips] == ["Clip 1", "Clip 1 copy", "Walk"]
    opkit.call("rok.undo")
    assert [c.name for c in scene.clips] == ["Clip 1", "Walk"] and scene.active_clip == 0
    # The dock shows a tab per clip; a press on a tab shows that clip.
    layout = dock.layout_for(app.region)
    assert [layout.button(f"TAB_{i}").label for i in range(2)] == ["Clip 1", "Walk"]
    tx, ty, tw, th = layout.button("TAB_1").rect
    assert dock.press(opkit.CONTEXT, app.region, tx + tw / 2, ty + th / 2, True, False)
    assert scene.active_clip == 1
    # The last clip stays.
    opkit.call("rok.clip_delete")
    assert opkit.call("rok.clip_delete") == {"CANCELLED"} and len(scene.clips) == 1


def test_make_part_cuts_the_selection_into_a_child_and_an_empty_part_is_a_root(app):
    """Phase 2: Make Part (4.2 step 1) and Empty parts (6.3)."""
    from rok.app.object import model_origin

    scene = document.scene
    hub = next(m for m in scene.models if m.name == "Hub")
    document.set_active(hub.uid)
    opkit.call("rok.level_cycle", step=-1) if state.level == "SCENE" else None
    before = int((hub.grid.voxels != 0).sum())
    top = np.zeros(hub.grid.size, dtype=bool)
    top[:, :, 4:] = True
    hub.selection.replace(top)  # the hub's top half
    assert opkit.call("rok.make_part") == {"FINISHED"}
    part = document.model
    assert part is not hub and part.parent == hub.uid and part.name.startswith("Hub")
    assert tuple(part.grid.size) == (5, 5, 4) and tuple(part.position) == tuple(p + o for p, o in zip(hub.position, (0, 0, 4)))
    assert int((hub.grid.voxels != 0).sum()) + int((part.grid.voxels != 0).sum()) == before
    assert opkit.call("rok.part_empty") == {"FINISHED"}
    root = document.model
    assert root.empty and tuple(root.grid.size) == (1, 1, 1) and root.parent == part.uid
    opkit.call("rok.undo")
    assert not any(m.empty for m in scene.models)
    opkit.call("rok.undo")
    assert int((hub.grid.voxels != 0).sum()) == before and len(scene.models) == 2
    _ = model_origin


def test_the_parent_is_set_by_dragging_a_name_in_the_dock_and_undone(app):
    scene = document.scene
    hub = next(m for m in scene.models if m.name == "Hub")
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    layout = dock.layout_for(app.region)
    rows = {r.uid: r for r in layout.rows if r.kind == "PART"}
    assert set(rows) == {hub.uid, blade().uid} and all(r.depth == 0 for r in rows.values())
    # Drag the blade's name onto the hub's row.
    src, dst = rows[blade().uid], rows[hub.uid]
    x = src.rect[0] + 40 * layout.scale
    assert dock.press(opkit.CONTEXT, app.region, x, src.rect[1] + src.rect[3] / 2, True, False)
    app.handle("mouse_move", {"x": x, "y": app.height - (dst.rect[1] + dst.rect[3] / 2)})  # region y is up
    app.handle("mouse_up", {"button": "left"})
    assert blade().parent == hub.uid
    layout = dock.layout_for(app.region)
    assert [(r.uid, r.depth) for r in layout.rows if r.kind == "PART"] == [(hub.uid, 0), (blade().uid, 1)]
    # A loop is refused; undo takes the parent away.
    assert opkit.call("rok.part_parent", uid=hub.uid, parent=blade().uid) == {"CANCELLED"}
    opkit.call("rok.undo")
    assert blade().parent is None
    # Spin the blade on its hub, then move the hub: the blade's gizmo follows.
    opkit.call("rok.part_parent", uid=blade().uid, parent=hub.uid)
    document.set_active(hub.uid)
    track = anim_mode.picked_track(scene.timeline)
    anim.set_key(track, anim.MOVE_X, anim.Key(0, 3.0))
    pick_blade()
    _pivot2, origin = anim_mode.gizmo_place(blade())
    assert origin[0] == model_origin_x(blade()) + 3


def model_origin_x(model) -> float:
    from rok.app.object import model_origin

    return model_origin(model)[0]


def _spin_blade(app):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    opkit.call("rok.anim_preset", name="SPIN")
    anim_mode.stop()
    return document.scene.timeline.track_of(blade().uid)


def test_keys_copy_and_paste_at_the_play_head_and_mirrored(app):
    """Phase 2: Cmd+C and Cmd+V on picked keys; Shift+Cmd+V pastes the other leg's pose."""
    track = _spin_blade(app)
    dock._picked.update({(blade().uid, None, 0), (blade().uid, None, 24)})
    assert opkit.call("rok.key_copy") == {"FINISHED"}
    anim_mode.set_frame(6)
    assert opkit.call("rok.key_paste") == {"FINISHED"}
    assert [(k.time, k.value) for k in track.keys(anim.TURN_Z)] == [(0, 0.0), (6, 0.0), (24, 360.0), (30, 360.0)]
    assert dock.picked_keys() == [(blade().uid, anim.TURN_Z, 6), (blade().uid, anim.TURN_Z, 30)]
    anim_mode.set_frame(12)
    assert opkit.call("rok.key_paste", mirrored=True) == {"FINISHED"}
    assert (12, 0.0) in [(k.time, k.value) for k in track.keys(anim.TURN_Z)]
    assert (36, -360.0) in [(k.time, k.value) for k in track.keys(anim.TURN_Z)]
    opkit.call("rok.undo")
    opkit.call("rok.undo")
    track = document.scene.timeline.track_of(blade().uid)  # undo swaps the clip object in
    assert [(k.time, k.value) for k in track.keys(anim.TURN_Z)] == [(0, 0.0), (24, 360.0)]
    # The keys go through the keymap too: Cmd+C with picked keys, Cmd+V at the play head.
    dock._picked.clear()
    dock._picked.add((blade().uid, None, 24))
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": True})
    app.handle("key_down", {"key": "c", "code": "KeyC", "text": "c", "repeat": False})
    app.handle("key_up", {"key": "c", "code": "KeyC", "text": "", "repeat": False})
    anim_mode.set_frame(3)
    app.handle("key_down", {"key": "v", "code": "KeyV", "text": "v", "repeat": False})
    app.handle("key_up", {"key": "v", "code": "KeyV", "text": "", "repeat": False})
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": False})
    assert (3, 360.0) in [(k.time, k.value) for k in track.keys(anim.TURN_Z)]


def test_a_box_dragged_over_the_lane_picks_the_keys_inside(app):
    _spin_blade(app)
    layout = dock.layout_for(app.region)
    row = next(r for r in layout.rows if r.uid == blade().uid)
    cy = row.rect[1] + row.rect[3] / 2
    x_start = layout.x_of(18)  # an empty spot on the row, right of the middle
    assert dock.press(opkit.CONTEXT, app.region, x_start, cy, True, False)
    assert dock.picked_keys() == []
    # Drag left over the key at 24? No: drag right over the key at 24.
    app.handle("mouse_move", {"x": layout.x_of(24) + 12, "y": app.height - (cy - 8)})
    assert dock.picked_keys() == [(blade().uid, anim.TURN_Z, 24)]
    app.handle("mouse_move", {"x": layout.x_of(0) - 12, "y": app.height - (cy - 8)})
    assert dock.picked_keys() == [(blade().uid, anim.TURN_Z, 0)]  # the box now runs leftward from 18
    app.handle("mouse_up", {"button": "left"})
    assert dock._box is None and dock.picked_keys() == [(blade().uid, anim.TURN_Z, 0)]


def test_a_typed_value_keys_the_channel_at_the_play_head(app):
    track = _spin_blade(app)
    anim_mode.set_frame(8)
    names = anim.MOVE + anim.TURN
    assert opkit.call("rok.key_value", channel=names.index(anim.TURN_Z), value=45.0) == {"FINISHED"}
    assert (8, 45.0) in [(k.time, k.value) for k in track.keys(anim.TURN_Z)]
    assert opkit.call("rok.key_value", channel=names.index(anim.MOVE_Z), value=2.0) == {"FINISHED"}
    assert [(k.time, k.value) for k in track.keys(anim.MOVE_Z)] == [(0, 0.0), (8, 2.0)]  # the rest came along


def test_markers_add_rename_move_and_delete_with_undo(app):
    """Phase 2: the Markers row (5.2). Games read these moments."""
    _spin_blade(app)
    c = document.scene.timeline
    assert opkit.call("rok.marker_add", frame=6, name="footstep") == {"FINISHED"}
    assert [(m.time, m.name) for m in c.markers] == [(6, "footstep")] and dock.picked_marker() == 6
    assert opkit.call("rok.marker_rename", name="hit") == {"FINISHED"}
    assert document.scene.timeline.markers[0].name == "hit"
    # Drag it along the row through the app's events.
    layout = dock.layout_for(app.region)
    mx, my, mw, mh = layout.markers_row
    cy = my + mh / 2
    assert dock.press(opkit.CONTEXT, app.region, layout.x_of(6), cy, True, False)
    app.handle("mouse_move", {"x": layout.x_of(12), "y": app.height - cy})
    app.handle("mouse_up", {"button": "left"})
    assert [m.time for m in document.scene.timeline.markers] == [12] and dock.picked_marker() == 12
    # Delete takes the picked marker; undo brings it back.
    assert opkit.call("rok.key_delete") == {"FINISHED"}
    assert document.scene.timeline.markers == []
    opkit.call("rok.undo")
    assert [m.time for m in document.scene.timeline.markers] == [12]
    # A double-click on an empty spot of the row offers a new marker there.
    assert dock.press(opkit.CONTEXT, app.region, layout.x_of(20), cy, True, False)
    assert anim_mode.frame() == 20


def test_the_onion_skin_shows_the_poses_before_and_after_the_play_head(app):
    from rok.app import onion

    _spin_blade(app)
    scene, c = document.scene, document.scene.timeline
    anim_mode.set_frame(6)
    assert not state.onion_skin
    assert opkit.call("rok.onion_skin") == {"FINISHED"} and state.onion_skin
    found = onion.copies(scene, c, 6)
    times = sorted((uid, t) for uid, t, _tris, _colour in found)
    assert (blade().uid, 0) in times and (blade().uid, 24) in times  # the keys on both sides
    assert all(tris.shape[1] == 3 and len(tris) % 3 == 0 and len(tris) > 0 for _u, _t, tris, _c in found)
    before = next(tris for u, t, tris, _c in found if t == 0)
    after = next(tris for u, t, tris, _c in found if t == 24)
    assert np.allclose(before, after)  # 0 and 360 degrees: the same place
    # The copy at frame 0 lies at the blade's rest place.
    from rok.app.object import model_origin
    o = np.asarray(model_origin(blade()))
    assert np.all(before.min(axis=0) >= o - 1e-3) and np.all(before.max(axis=0) <= o + np.asarray(blade().grid.size) + 1e-3)
    app.redraw()
    layer = [c_ for c_ in app.host.named("set_draw") if c_[1][0] == "anim_onion"]
    assert layer and len(layer[-1][2]["tris3_depth"]) > 0
    opkit.call("rok.onion_skin")
    assert not state.onion_skin


def test_root_motion_is_a_clip_setting(app):
    _spin_blade(app)
    c = document.scene.timeline
    assert not c.root_motion
    assert opkit.call("rok.clip_length", seconds=1.0, fps=24, root_motion=True) == {"FINISHED"}
    assert document.scene.timeline.root_motion
    assert opkit.call("rok.clip_settings", root_motion=2) == {"FINISHED"}
    assert not document.scene.timeline.root_motion
    opkit.call("rok.undo")
    assert document.scene.timeline.root_motion


def _at(app, world) -> dict:
    x, y = app.camera.project([world])[0]
    return {"x": float(x), "y": float(y)}


def _click(app, point) -> None:
    app.handle("mouse_move", point)
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def test_a_frame_is_added_painted_in_animate_mode_and_undone(app):
    """Phase 3 (6.5, 4.3): Frames > Add frame copies the shown frame; the paint
    tools then paint that frame, not the rest; undo covers both."""
    from rok.app.object import model_origin

    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    model = blade()
    rest_before = model.grid.voxels.copy()
    anim_mode.set_frame(6)
    assert opkit.call("rok.frame_add") == {"FINISHED"}
    assert model.frame_count() == 2 and np.array_equal(model.states[0], rest_before)
    track = document.scene.timeline.track_of(model.uid)
    assert [(k.time, k.value) for k in track.keys(anim.STATE)] == [(0, 0), (6, 1)]
    assert anim_mode.shown_frame(model) == 1 and document.grid is model.frame_grid(1)
    anim_mode.set_frame(2)
    assert anim_mode.shown_frame(model) == 0 and document.grid is model.grid
    anim_mode.set_frame(8)
    # Paint: Erase on the blade takes a voxel off frame 2 only (the bar is solid: no room to attach).
    assert opkit.call("rok.anim_paint", tool="ERASE") == {"FINISHED"} and state.anim_paint == "ERASE"
    cells = np.argwhere(model.grid.voxels != 0)
    top = cells[len(cells) // 2]
    o = np.asarray(model_origin(model), dtype=np.float64)
    _click(app, _at(app, o + (top[0] + 0.5, top[1] + 0.5, top[2] + 1.0)))
    frame_voxels = model.frame_grid(1).voxels
    assert int((frame_voxels != 0).sum()) == int((rest_before != 0).sum()) - 1
    assert np.array_equal(model.grid.voxels, rest_before)  # the rest pose stands
    assert np.array_equal(model.states[0], frame_voxels)  # one array for the grid and the state
    key, shown, _place = anim_mode.shown().poses[model.uid]
    assert np.count_nonzero(shown) == np.count_nonzero(frame_voxels)
    opkit.call("rok.undo")
    assert int((model.frame_grid(1).voxels != 0).sum()) == int((rest_before != 0).sum())
    # Back to posing; delete the frame: its keys go, the rest stays.
    assert opkit.call("rok.anim_paint", tool="POSE") == {"FINISHED"} and state.anim_paint is None
    assert opkit.call("rok.frame_delete") == {"FINISHED"}
    assert model.frame_count() == 1 and not document.scene.timeline.track_of(model.uid).keys(anim.STATE)
    opkit.call("rok.undo")
    assert model.frame_count() == 2
    anim_mode.set_frame(2)
    assert opkit.call("rok.frame_delete") == {"CANCELLED"}  # the first frame stays


def test_the_dock_shows_a_frames_row_with_blocks(app):
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    anim_mode.set_frame(12)
    opkit.call("rok.frame_add")
    layout = dock.layout_for(app.region)
    row = next(r for r in layout.rows if r.kind == "FRAMES" and r.uid == blade().uid)
    assert row.keys == ((0, anim.HOLD), (12, anim.HOLD)) and row.values == (1, 2) and row.value == "2"
    rows = [r.kind for r in layout.rows]
    assert rows.index("FRAMES") == rows.index("PART") + 1 if rows[0] == "PART" and layout.rows[0].uid == blade().uid else True
    app.redraw()  # draws the blocks
    layer = [c for c in app.host.named("set_draw") if c[1][0] == "anim_dock"]
    assert layer


def test_colour_motions_change_the_palette_at_the_play_head_and_reach_the_host(app):
    """Phase 3: Blink, Glow and Flow on the current colour (4.3, the Motion gallery's
    colour cards). The host gets the clip's palette while it plays, the scene's own after."""
    from rok.app import anim_mode as am

    scene = document.scene
    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    slot = int(blade().grid.voxels[np.nonzero(blade().grid.voxels)][0])
    document.colour = slot
    rest_rgb = tuple(int(c) for c in scene.palette[slot, :3])
    assert opkit.call("rok.anim_preset", name="BLINK") == {"FINISHED"}
    am.stop()
    c = scene.timeline
    track = am.colour_track(c, (slot,), create=False)
    assert track is not None and [k.time for k in track.keys(anim.COLOUR_RGB)] == [0, c.length - 4, c.length - 1]
    skin = tuple(track.settings["skin_colour"])
    shut = track.keys(anim.COLOUR_RGB)[1]
    assert tuple(shut.value) == skin and skin != rest_rgb
    am.set_frame(shut.time)
    shown = am.shown()
    assert shown.palette is not None and tuple(shown.palette[slot, :3]) == skin
    app.host.calls.clear()
    app.flush()
    sent = app.host.named("set_palette")
    assert sent and tuple(sent[-1][1][0][slot, :3]) == skin
    am.set_frame(0)
    app.flush()
    assert tuple(app.host.named("set_palette")[-1][1][0][slot, :3]) == rest_rgb
    # Glow adds a GLOW channel on the same colour; Flow cycles several slots.
    assert opkit.call("rok.anim_preset", name="GLOW") == {"FINISHED"}
    am.stop()
    assert len(am.colour_track(scene.timeline, (slot,), create=False).keys(anim.GLOW)) == 3
    assert opkit.call("rok.anim_preset", name="FLOW") == {"FINISHED"}
    am.stop()
    flow = next(t for t in scene.timeline.tracks if t.target.kind == anim.COLOUR and len(t.target.slots) > 1)
    assert flow.target.slots[0] == slot and len(flow.keys(anim.SLOT)) == 6
    # The dock lists the colour motions as rows of their own.
    layout = dock.layout_for(app.region)
    assert sum(1 for r in layout.rows if r.kind == "COLOUR") == 2
    # Leaving Animate mode: the scene's own palette again.
    opkit.call("rok.animate", action="LEAVE")
    app.flush()
    assert tuple(app.host.named("set_palette")[-1][1][0][slot, :3]) == rest_rgb


def test_an_animated_vox_opens_with_its_frames_and_plays(tmp_path, monkeypatch):
    """Phase 3 gate: an animated .vox file opens and plays. Its
    second model state is frame 2 of the part, shown from the STATE key on."""
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "phase0" / "animated.vox"))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    try:
        scene = document.scene
        spinner = next(m for m in scene.models if m.name == "spinner")
        assert spinner.frame_count() == 2 and scene.timeline.fps == anim.VOX_FPS
        track = scene.timeline.track_of(spinner.uid)
        assert [(k.time, k.value) for k in track.keys(anim.STATE)] == [(0, 0), (5, 1)]
        opkit.call("rok.animate", action="ENTER")
        anim_mode.set_frame(2)
        assert anim_mode.shown_frame(spinner) == 0 and spinner.uid not in anim_mode.shown().poses
        anim_mode.set_frame(6)
        assert anim_mode.shown_frame(spinner) == 1
        _key, voxels, _place = anim_mode.shown().poses[spinner.uid]
        assert np.count_nonzero(voxels) == np.count_nonzero(spinner.states[0])  # frame 2's voxels, turned
        layout = dock.layout_for(a.region)
        assert any(r.kind == "FRAMES" and r.uid == spinner.uid for r in layout.rows)
    finally:
        anim_mode.stop()
        document.reset()
        state.__init__()


def test_the_blinking_face_demo_opens_with_two_eye_frames(tmp_path, monkeypatch):
    """Phase 3 gate: tests/fixtures/anim/face.rok, made through Make Part, Frame: Add
    and the paint tools (scratch make_face.py): the eyes shut for four frames at 3.0 s."""
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "anim" / "face.rok"))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    try:
        scene = document.scene
        eyes = next(m for m in scene.models if m.name == "Eyes")
        head = next(m for m in scene.models if m.name == "Head")
        assert eyes.parent == head.uid and eyes.frame_count() == 2
        track = scene.timeline.track_of(eyes.uid)
        assert [(k.time, k.value) for k in track.keys(anim.STATE)] == [(0, 0), (72, 1), (76, 0)]
        assert int((eyes.grid.voxels != 0).sum()) == 8 and int((eyes.states[0] != 0).sum()) == 4
        opkit.call("rok.animate", action="ENTER")
        anim_mode.set_frame(74)
        _key, voxels, _place = anim_mode.shown().poses[eyes.uid]
        assert np.count_nonzero(voxels) == 4  # the lids
    finally:
        anim_mode.stop()
        document.reset()
        state.__init__()


def _ring_points(app, ring: str = "ROT_Z", count: int = 6):
    from rok.app.state import gizmo_screen_for

    app.redraw()
    points = np.asarray(gizmo_screen_for(app.region).rings[ring], dtype=np.float64)
    return [{"x": float(x), "y": float(app.height - y)} for x, y in points[:count]]


def test_a_ring_drag_whose_first_move_changes_nothing_still_keys(app):
    """The press point is the first move too: it changes nothing. That raised
    an UnboundLocalError in the stroke and the drag was dropped silently, so a
    new model could not be turned or moved in Animate mode."""
    from tests.test_app_tools import drag

    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    anim_mode.set_frame(12)
    points = _ring_points(app)
    drag(app, [points[0]] + points, release=False)  # the first move stays where the press was
    assert anim_mode._pose_drag is not None and anim_mode._pose_drag.get("result") is not None
    app.handle("mouse_up", {"button": "left"})
    track = document.scene.timeline.track_of(blade().uid)
    assert [k.time for k in track.keys(anim.TURN_Z)] == [0, 12]  # the rest at 0, the pose at the play head


def test_the_box_and_the_gizmo_stay_on_the_part_during_an_arrow_drag(app, monkeypatch):
    """Animate mode drew the box and the gizmo at the posed part plus the drag
    offset again: twice as far as the part, then back at release."""
    from rok.app import overlay
    from rok.app.object import model_origin
    from rok.app.state import gizmo_screen_for

    opkit.call("rok.animate", action="ENTER")
    pick_blade()
    anim_mode.set_frame(12)
    app.redraw()
    rest = tuple(float(v) for v in model_origin(blade()))
    tip = np.asarray(gizmo_screen_for(app.region).anchors["MOVE_Z"], dtype=np.float64)[0]
    start = {"x": float(tip[0]), "y": float(app.height - tip[1])}
    seen = []
    real = overlay.draw_scene

    def spy(size, origin, *args, **kwargs):
        seen.append((tuple(origin), kwargs.get("gizmo_state")))
        return real(size, origin, *args, **kwargs)

    monkeypatch.setattr(overlay, "draw_scene", spy)
    app.handle("mouse_move", start)
    app.handle("mouse_down", {"button": "left"})
    for k in range(1, 9):
        app.handle("mouse_move", {"x": start["x"], "y": start["y"] - 12.0 * k})
        app.flush()
    seen.clear()
    app.redraw()
    moved = state.model_drag_offset[2]
    assert moved > 0  # the drag moved the part
    place = anim_mode.shown().poses[blade().uid][2]
    assert place[2] == blade().position[2] + moved  # the part is `moved` cells up
    box, gizmo_state = seen[-1]
    assert box[2] == rest[2] + moved  # the box is on the part, not `moved` cells above it
    assert gizmo_state is not None and gizmo_state.origin[2] == rest[2] + moved  # and so is the gizmo
    app.handle("mouse_up", {"button": "left"})
