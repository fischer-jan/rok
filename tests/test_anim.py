"""The animation core (rok/core/anim.py): keys and eases, clips and their
loop modes, one channel per axis, poses about the pivot, motions,
the .vox frames as keys and back, the loss table, and the JSON."""

from pathlib import Path

import numpy as np
import pytest

from rok.core import grid_ops
from rok.core.anim import (
    COLOUR,
    COLOUR_RGB,
    EASE_IN,
    EASE_OUT,
    GLOW,
    HOLD,
    IDENTITY_ROTATION,
    LINEAR,
    LOOP,
    LOSS_TEXT,
    MOVE,
    MOVE_X,
    MOVE_Z,
    ONCE,
    PART,
    PING_PONG,
    PRESETS,
    ROTATION,
    SCALE,
    SLOT,
    SMOOTH,
    STATE,
    TURN,
    TURN_X,
    TURN_Z,
    VISIBLE,
    VOX_FPS,
    Clip,
    Key,
    Property,
    Target,
    Track,
    bake_times,
    default_anchor2,
    euler_from_matrix,
    evaluate,
    from_json,
    key_change,
    matrix_from_euler,
    move_keys,
    neighbour_colour,
    preset_keys,
    remove_keys,
    set_ease,
    set_key,
    to_json,
    track_from_vox,
    value_at,
    vox_from_track,
    vox_losses,
)
from rok.core.grid import VoxelGrid
from rok.core.material import DEFAULT, MaterialTable
from rok.core.scene import VoxScene
from rok.core.vox_read import read_scene

GOLDEN = Path(__file__).parent / "fixtures" / "phase0" / "animated.vox"


def _grid(size, fill=1) -> VoxelGrid:
    grid = VoxelGrid(size)
    grid.voxels[:] = fill
    return grid


def _keyed(track: Track, pairs) -> None:
    for name, keys in pairs:
        for key in keys:
            set_key(track, name, key)


# --- keys and eases ----------------------------------------------------------------------


def test_a_hold_key_keeps_its_value_until_the_next():
    prop = Property("p", [Key(0, 1), Key(4, 7), Key(9, 2)])
    assert [value_at(prop, t) for t in (-3, 0, 3, 4, 8.9, 9, 50)] == [1, 1, 1, 7, 7, 2, 2]


def test_every_ease_blends_from_its_key_to_the_next():
    def at(ease, t):
        return value_at(Property("p", [Key(0, 0.0, ease), Key(10, 10.0)]), t)

    assert at(LINEAR, 5) == pytest.approx(5.0)
    assert at(SMOOTH, 5) == pytest.approx(5.0) and at(SMOOTH, 2) < 2.0
    assert at(EASE_IN, 5) == pytest.approx(2.5)  # slow at the start
    assert at(EASE_OUT, 5) == pytest.approx(7.5)  # slow at the end
    assert at(HOLD, 9) == 0.0
    tuples = Property("p", [Key(0, (0.0, 10.0), LINEAR), Key(10, (10.0, 0.0))])
    assert value_at(tuples, 5) == pytest.approx((5.0, 5.0))
    assert value_at(Property("p"), 3) is None


def test_set_move_and_remove_keys():
    track = Track(Target(PART, 1))
    set_key(track, TURN_Z, Key(0, 0.0))
    set_key(track, TURN_Z, Key(5, 30.0))
    set_key(track, TURN_Z, Key(5, 45.0))  # replaces the key at 5
    set_key(track, TURN_Z, Key(9, 90.0))
    assert [(k.time, k.value) for k in track.keys(TURN_Z)] == [(0, 0.0), (5, 45.0), (9, 90.0)]
    move_keys(track, TURN_Z, [5], 4)  # lands on 9 and replaces it
    assert [(k.time, k.value) for k in track.keys(TURN_Z)] == [(0, 0.0), (9, 45.0)]
    assert remove_keys(track, TURN_Z, [0, 9]) == 2
    assert TURN_Z not in track.properties  # an empty channel goes


def test_set_ease_leaves_hold_only_channels_alone():
    track = Track(Target(PART, 1))
    set_key(track, STATE, Key(0, 0))
    set_key(track, TURN_X, Key(0, 0.0))
    set_ease(track, STATE, [0], LINEAR)
    set_ease(track, TURN_X, [0], EASE_IN)
    assert track.keys(STATE)[0].ease == HOLD and track.keys(TURN_X)[0].ease == EASE_IN


def test_the_first_key_away_from_frame_zero_keeps_the_rest_at_zero():
    track = Track(Target(PART, 1))
    key_change(track, TURN_X, 12, 30.0, 0.0)
    assert [(k.time, k.value) for k in track.keys(TURN_X)] == [(0, 0.0), (12, 30.0)]
    key_change(track, MOVE_Z, 0, 2.0, 0.0)  # at frame 0: one key
    assert [(k.time, k.value) for k in track.keys(MOVE_Z)] == [(0, 2.0)]
    key_change(track, VISIBLE, 5, False, True)
    assert all(k.ease == HOLD for k in track.keys(VISIBLE))


def test_a_key_dropped_into_a_linear_spin_stays_linear():
    track = Track(Target(PART, 1))
    _keyed(track, preset_keys("SPIN", track, Clip()))
    key_change(track, TURN_Z, 10, 90.0, 0.0)
    assert {k.time: k.ease for k in track.keys(TURN_Z)}[10] == LINEAR


# --- clips -----------------------------------------------------------------------------


def test_a_new_clip_is_one_second_at_24_with_voxel_look_on():
    clip = Clip()
    assert (clip.name, clip.fps, clip.length, clip.loop_mode, clip.voxel_look) == ("Clip 1", 24, 24, LOOP, True)
    assert clip.seconds == pytest.approx(1.0)


def test_loop_once_and_ping_pong():
    clip = Clip(length=5)
    assert [clip.time_at(t) for t in (0, 4, 5, 7)] == [0, 4, 0, 2]
    clip.loop_mode = ONCE
    assert [clip.time_at(t) for t in (-1, 2, 9)] == [0, 2, 4]
    clip.loop_mode = PING_PONG
    assert [clip.time_at(t) for t in range(10)] == [0, 1, 2, 3, 4, 3, 2, 1, 0, 1]


# --- poses -----------------------------------------------------------------------------


def test_a_spin_turns_about_the_parts_pivot():
    scene = VoxScene()
    model = scene.add(VoxelGrid((4, 4, 4)))
    model.pivot2 = (0, 0, 0)  # a corner
    track = Track(Target(PART, model.uid))
    _keyed(track, preset_keys("SPIN", track, Clip()))
    clip = Clip(tracks=[track])
    m = evaluate(clip, scene, 6).transforms[model.uid]  # a quarter of the way: 90 degrees
    assert np.allclose(m @ [0, 0, 0, 1], [0, 0, 0, 1])  # the pivot stays
    assert np.allclose(m @ [1, 0, 0, 1], [0, 1, 0, 1])
    assert np.allclose(evaluate(clip, scene, 24).transforms[model.uid], np.eye(4))  # it loops


def test_without_a_pivot_a_part_turns_about_its_middle():
    scene = VoxScene()
    model = scene.add(VoxelGrid((4, 2, 2)))
    track = Track(Target(PART, model.uid))
    set_key(track, TURN_Z, Key(0, 180.0))
    m = evaluate(Clip(tracks=[track]), scene, 0).transforms[model.uid]
    assert np.allclose(m @ [2, 1, 1, 1], [2, 1, 1, 1])


def test_channels_add_up_x_then_y_then_z():
    scene = VoxScene()
    model = scene.add(VoxelGrid((2, 2, 2)))
    track = Track(Target(PART, model.uid))
    set_key(track, TURN_X, Key(0, 90.0))
    set_key(track, TURN_Z, Key(0, 90.0))
    set_key(track, MOVE_X, Key(0, 3.0))
    pose = evaluate(Clip(tracks=[track]), scene, 0).poses[model.uid]
    assert np.allclose(pose.linear, matrix_from_euler((90, 0, 90)))
    assert np.allclose(pose.offset, (3, 0, 0))


def test_a_part_with_no_turn_or_move_keys_has_no_pose():
    scene = VoxScene()
    model = scene.add(VoxelGrid((4, 4, 4)))
    track = Track(Target(PART, model.uid))
    set_key(track, VISIBLE, Key(0, True))
    set_key(track, VISIBLE, Key(10, False))
    set_key(track, STATE, Key(10, 2))
    clip = Clip(length=20, tracks=[track])
    frame = evaluate(clip, scene, 12)
    assert frame.poses == {} and frame.visible == {model.uid: False} and frame.state == {model.uid: 2}


def test_a_track_whose_model_is_gone_is_skipped():
    scene = VoxScene()
    track = Track(Target(PART, 99))
    set_key(track, TURN_Z, Key(0, 10.0))
    assert evaluate(Clip(tracks=[track]), scene, 0).poses == {}


def test_size_about_a_face_pivot_keeps_the_pivot():
    scene = VoxScene()
    model = scene.add(VoxelGrid((4, 4, 4)))
    model.pivot2 = (8, 4, 4)  # the +X face's middle, (4, 2, 2)
    track = Track(Target(PART, model.uid))
    set_key(track, SCALE, Key(0, (200, 200, 200)))
    m = evaluate(Clip(tracks=[track]), scene, 0).transforms[model.uid]
    assert np.allclose(m @ [4, 2, 2, 1], [4, 2, 2, 1])


# --- motions ---------------------------------------------------------------------------


def test_the_part_motions_fill_the_clip_on_the_axis():
    clip = Clip(length=48)
    track = Track(Target(PART, 1), axis=0)
    (name, keys), = preset_keys("SPIN", track, clip)
    assert name == TURN_X and [(k.time, k.value, k.ease) for k in keys] == [(0, 0.0, LINEAR), (48, 360.0, LINEAR)]
    (name, keys), = preset_keys("SWING", track, clip)
    assert name == TURN_X and [k.value for k in keys] == [0.0, 30.0, 0.0, -30.0, 0.0] and keys[-1].time == 48
    (name, keys), = preset_keys("BOB", Track(Target(PART, 1)), clip)
    assert name == MOVE_Z and [k.time for k in keys] == [0, 12, 24, 36, 48]


def test_a_shake_with_the_same_seed_is_the_same_and_ends_at_rest():
    track = Track(Target(PART, 1))
    track.settings["seed"] = 7
    first = preset_keys("SHAKE", track, Clip())
    assert first == preset_keys("SHAKE", track, Clip())
    for name, keys in first:
        assert name in MOVE[:2]
        assert all(k.value in (-1.0, 0.0, 1.0) and k.ease == HOLD for k in keys)
        assert keys[-1].value == 0.0


def test_every_motion_has_keys():
    for name in PRESETS:
        target = Target(PART, 1) if name in ("SPIN", "SWING", "BOB", "SHAKE") else Target(COLOUR, slots=(1, 2))
        assert preset_keys(name, Track(target), Clip())


def test_default_anchor2_of_a_part_touching_nothing_is_the_box_centre_and_z():
    scene = VoxScene()
    part = scene.add(_grid((4, 6, 2)), position=(10, 0, 0))
    assert default_anchor2(part, []) == ((24, 6, 2), 2)


def test_default_anchor2_of_a_part_touching_a_plus_y_face_is_that_face_and_axis_1():
    scene = VoxScene()
    part = scene.add(_grid((4, 4, 4)), position=(0, 0, 0))
    other = scene.add(_grid((4, 4, 4)), position=(0, 4, 0))
    assert default_anchor2(part, [other]) == ((4, 8, 4), 1)


# --- colours ---------------------------------------------------------------------------


def test_a_colour_key_sets_the_slot_and_leaves_the_rest_palette():
    scene = VoxScene()
    scene.palette[9] = (1, 2, 3, 255)
    track = Track(Target(COLOUR, slots=(9,)))
    set_key(track, COLOUR_RGB, Key(0, (10, 20, 30)))
    frame = evaluate(Clip(tracks=[track]), scene, 0)
    assert list(frame.palette[9]) == [10, 20, 30, 255] and list(scene.palette[9]) == [1, 2, 3, 255]


def test_slot_keys_rotate_the_range():
    scene = VoxScene()
    slots = (1, 2, 3)
    for i, slot in enumerate(slots):
        scene.palette[slot] = (i * 10, 0, 0, 255)
    track = Track(Target(COLOUR, slots=slots))
    set_key(track, SLOT, Key(0, 1))
    frame = evaluate(Clip(tracks=[track]), scene, 0)
    assert [int(frame.palette[s, 0]) for s in slots] == [20, 0, 10]


def test_glow_on_a_light_scales_its_emission():
    scene = VoxScene()
    scene.palette[5] = (100, 50, 25, 255)
    scene.materials = MaterialTable({5: DEFAULT.with_type("EMIT")})
    track = Track(Target(COLOUR, slots=(5,)))
    _keyed(track, preset_keys("GLOW", track, Clip()))
    frame = evaluate(Clip(tracks=[track]), scene, 12)  # the middle: 200 %
    assert frame.emission[5] == pytest.approx(2.0) and frame.emission[0] == pytest.approx(1.0)


def test_neighbour_colour_is_the_most_common_colour_around_the_slot():
    scene = VoxScene()
    scene.palette[9] = (10, 20, 30, 255)
    model = scene.add(_grid((8, 8, 8), fill=9))
    model.grid.voxels[3, 3, 3] = 5
    assert neighbour_colour(scene, (5,)) == (10, 20, 30)
    assert neighbour_colour(VoxScene(), (5,)) == (0, 0, 0)


# --- the .vox frames ------------------------------------------------------------


def test_no_frames_and_no_states_make_no_track():
    assert track_from_vox(1, (), 0, ()) is None
    assert vox_from_track(None, 2) == ((), 0, [(1, 1), (2, 2)])


def test_vox_frames_become_one_hold_channel_per_axis_and_back():
    frames = ((3, 17, (4, 0, -2)), (6, IDENTITY_ROTATION, (0, 5, 0)))
    degrees_3 = euler_from_matrix(grid_ops.rotation_from_byte(17))
    track = track_from_vox(9, frames, 0, ((2, 1), (5, 2)))
    assert [(k.time, k.value) for k in track.keys(MOVE_X)] == [(0, 0), (3, 4), (6, 0)]
    assert [(k.time, k.value) for k in track.keys(MOVE_Z)] == [(0, 0), (3, -2), (6, 0)]
    for axis, name in enumerate(TURN):
        if degrees_3[axis]:
            assert [(k.time, k.value) for k in track.keys(name)] == [(0, 0), (3, degrees_3[axis]), (6, 0)]
        else:
            assert name not in track.properties  # a channel that never moves is not made
    assert [(k.time, k.value) for k in track.keys(STATE)] == [(0, 0), (2, 1), (5, 2)]
    assert all(k.ease == HOLD for p in track.properties.values() for k in p.keys)
    assert vox_from_track(track, 2) == (frames, 0, [(2, 1), (5, 2)])


def _valid_bytes():
    return [b for b in range(128) if (b & 3) != 3 and ((b >> 2) & 3) != 3 and (b & 3) != ((b >> 2) & 3)]


def test_every_rotation_byte_round_trips_and_every_reflection_raises():
    saw_rotation = saw_reflection = False
    for b in _valid_bytes():
        R = grid_ops.rotation_from_byte(b)
        if int(round(np.linalg.det(R))) == 1:
            saw_rotation = True
            degrees = euler_from_matrix(R)
            M = np.rint(matrix_from_euler(degrees)).astype(np.int64)
            assert grid_ops.rotation_to_byte(M) == b
        else:
            saw_reflection = True
            with pytest.raises(ValueError):
                euler_from_matrix(R)
    assert saw_rotation and saw_reflection


def test_a_mirroring_frame_reads_as_no_turn():
    reflection = next(b for b in _valid_bytes() if int(round(np.linalg.det(grid_ops.rotation_from_byte(b)))) == -1)
    track = track_from_vox(2, ((4, reflection, (1, 0, 0)),), 0, ())
    assert not any(name in track.properties for name in TURN)
    assert [(k.time, k.value) for k in track.keys(MOVE_X)] == [(0, 0), (4, 1)]


def test_a_vox_read_sets_the_vox_frame_rate_and_a_length_one_past_the_last_frame():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    clip = scene.timeline
    assert clip.fps == VOX_FPS
    assert clip.length == max(t for track in clip.tracks for t in track.key_times()) + 1
    assert vox_losses(scene) == ()  # whole cells and quarter turns, Hold: nothing bakes


def test_a_linear_turn_bakes_with_the_loss_text():
    scene = VoxScene()
    model = scene.add(VoxelGrid((2, 2, 2)))
    track = Track(Target(PART, model.uid), name="Fan")
    _keyed(track, preset_keys("SPIN", track, Clip()))
    scene.timeline = Clip(fps=VOX_FPS, length=4, tracks=[track])
    track.properties[TURN_Z].keys = [Key(0, 0.0, LINEAR), Key(4, 360.0, LINEAR)]
    (loss,) = vox_losses(scene)
    assert (loss.property, loss.name, loss.level) == (ROTATION, "Fan", "BAKED")
    assert bake_times(track, ROTATION, 4, VOX_FPS) == [0, 1, 2, 3]
    frames, _sf, _sfs = vox_from_track(track, 0, VOX_FPS, 4)
    assert [f[0] for f in frames] == [0, 1, 2, 3]


def test_size_and_show_keys_are_dropped():
    scene = VoxScene()
    model = scene.add(VoxelGrid((2, 2, 2)))
    track = Track(Target(PART, model.uid), name="Box")
    set_key(track, SCALE, Key(0, (100, 100, 100)))
    set_key(track, VISIBLE, Key(0, True))
    scene.timeline = Clip(tracks=[track])
    assert [(loss.property, loss.level) for loss in vox_losses(scene)] == [(SCALE, "DROPPED"), (VISIBLE, "DROPPED")]
    assert all(a and b for entry in LOSS_TEXT.values() for a, b in entry.values())


# --- JSON ------------------------------------------------------------------------------


def test_a_clip_round_trips_through_json():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    clip = scene.timeline
    clip.name, clip.loop_mode, clip.voxel_look = "Walk", PING_PONG, False
    d = to_json(clip)
    for track_json in d["tracks"]:
        track_json["target"]["model"] = scene.index_of(track_json["target"]["model"])
    assert from_json(d, uids=[m.uid for m in scene.models]) == clip


# --- parents ----------------------------------------------------------------------------


def test_a_child_follows_its_parents_turn_about_the_parents_pivot():
    """6.3: a child's pose is its parent's pose, then its own."""
    scene = VoxScene()
    body = scene.add(VoxelGrid((2, 2, 2)), "Body", (0, 0, 0))
    body.pivot2 = (0, 0, 0)  # the corner at the world origin
    arm = scene.add(VoxelGrid((2, 2, 2)), "Arm", (4, 0, 0))
    arm.parent = body.uid
    track = Track(Target(PART, body.uid))
    set_key(track, TURN_Z, Key(0, 90.0))
    frame = evaluate(Clip(tracks=[track]), scene, 0)
    assert arm.uid in frame.poses  # no keys of its own, and still posed
    m = frame.transforms[arm.uid]
    # The arm's cell (0, 0, 0) sits at world (4, 0, 0); a quarter turn about the
    # origin takes it to (0, 4, 0): in the arm's own grid that is (-4, 4, 0).
    assert np.allclose(m @ [0, 0, 0, 1], [-4, 4, 0, 1])
    assert np.allclose(m[:3, :3], matrix_from_euler((0, 0, 90)))


def test_a_childs_own_move_adds_to_its_parents():
    scene = VoxScene()
    body = scene.add(VoxelGrid((2, 2, 2)), "Body", (0, 0, 0))
    hand = scene.add(VoxelGrid((2, 2, 2)), "Hand", (5, 5, 5))
    hand.parent = body.uid
    t_body = Track(Target(PART, body.uid))
    set_key(t_body, MOVE_X, Key(0, 3.0))
    t_hand = Track(Target(PART, hand.uid))
    set_key(t_hand, MOVE_Z, Key(0, 2.0))
    frame = evaluate(Clip(tracks=[t_body, t_hand]), scene, 0)
    assert np.allclose(frame.poses[hand.uid].offset, (3, 0, 2))
    assert np.allclose(frame.poses[body.uid].offset, (3, 0, 0))


def test_a_parent_loop_does_not_hang_and_a_missing_parent_is_a_root():
    scene = VoxScene()
    a = scene.add(VoxelGrid((2, 2, 2)), "A")
    b = scene.add(VoxelGrid((2, 2, 2)), "B")
    a.parent, b.parent = b.uid, a.uid
    track = Track(Target(PART, a.uid))
    set_key(track, MOVE_X, Key(0, 1.0))
    frame = evaluate(Clip(tracks=[track]), scene, 0)
    assert a.uid in frame.poses
    assert scene.would_cycle(a.uid, b.uid) and not scene.set_parent(a.uid, a.uid)
    c = scene.add(VoxelGrid((2, 2, 2)), "C")
    c.parent = 999
    assert scene.depth_of(c.uid) == 0 and [m.name for m, _d in scene.ordered_parts()][-1] == "C"


def test_ordered_parts_lists_children_under_their_parent_with_depth():
    scene = VoxScene()
    body = scene.add(VoxelGrid((2, 2, 2)), "Body")
    head = scene.add(VoxelGrid((2, 2, 2)), "Head")
    eye = scene.add(VoxelGrid((2, 2, 2)), "Eye")
    leg = scene.add(VoxelGrid((2, 2, 2)), "Leg")
    assert scene.set_parent(head.uid, body.uid) and scene.set_parent(eye.uid, head.uid) and scene.set_parent(leg.uid, body.uid)
    assert [(m.name, d) for m, d in scene.ordered_parts()] == [("Body", 0), ("Head", 1), ("Eye", 2), ("Leg", 1)]
    assert not scene.set_parent(body.uid, eye.uid)  # a loop


def test_the_vox_bake_rounds_a_turn_to_the_nearest_quarter():
    from rok.core.anim import _rounded_rotation

    assert _rounded_rotation((-30.0, 5.0, 44.0)) == (0, 0, 0)
    assert _rounded_rotation((46.0, -46.0, 135.0)) == (90, -90, 180)
    assert _rounded_rotation((360.0, 270.0, -90.0)) == (360, 270, -90)
