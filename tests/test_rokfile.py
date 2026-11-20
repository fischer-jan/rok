"""rok/core/rokfile.py: pack and unpack the Rok file's JSON text."""

import json

import numpy as np
import pytest

from rok.core import anim, rokfile, link
from rok.core.grid import VoxelGrid
from rok.core.scene import VoxScene
from rok.core.vox_read import loads_scene, read_scene

from reference import _animated_file

from helpers import FIXTURES

GOLDEN = FIXTURES / "phase3" / "two_models.vox"


def _assert_same_scene_contents(a: VoxScene, b: VoxScene) -> None:
    """Everything pack/unpack must carry across untouched."""
    assert np.array_equal(a.palette, b.palette)
    assert a.materials == b.materials
    assert a.layers == b.layers
    assert a.extras == b.extras
    assert len(a.models) == len(b.models)
    for ma, mb in zip(a.models, b.models):
        assert np.array_equal(ma.grid.voxels, mb.grid.voxels)


def test_pack_and_unpack_round_trip_two_models():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    text = rokfile.pack(scene, {})
    again, doc = rokfile.unpack(text)
    assert doc == {}
    _assert_same_scene_contents(again, scene)


def test_pack_and_unpack_round_trip_the_animated_file_and_its_timeline():
    scene = VoxScene.from_data(loads_scene(_animated_file()))
    text = rokfile.pack(scene, {})
    again, _doc = rokfile.unpack(text)
    _assert_same_scene_contents(again, scene)
    assert len(again.timeline.tracks) == len(scene.timeline.tracks)
    for orig_track, new_track in zip(scene.timeline.tracks, again.timeline.tracks):
        assert orig_track.name == new_track.name
        for name in (anim.POSITION, anim.ROTATION, anim.STATE):
            assert orig_track.keys(name) == new_track.keys(name)


def test_a_document_dict_round_trips_unchanged():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    doc = {
        "active": 1,
        "colour": 12,
        "views": {"1": [[0.0, 0.0, 8.0], [0.7, 0.4, 0.3, 0.5], 40.0, "PERSP"]},
        "view": [[0.0, 0.0, 8.0], [0.7, 0.4, 0.3, 0.5], 40.0, "PERSP"],
        "playhead": 15,
        "expanded": [0, 2],
        "record": False,
    }
    text = rokfile.pack(scene, doc)
    _again, got = rokfile.unpack(text)
    assert got == doc


def test_a_document_dict_without_the_phase_2_keys_round_trips_too():
    """A Rok file from before Phase 2 has no playhead, expanded or record
    key; rokfile hands the dict back exactly as it found it, and it is
    blend_io.restore's business (not this module's) to default the missing
    keys to 0, [] and True."""
    scene = VoxScene.from_data(read_scene(GOLDEN))
    doc = {"active": 0, "colour": 1, "views": {}, "view": None}
    text = rokfile.pack(scene, doc)
    _again, got = rokfile.unpack(text)
    assert got == doc
    assert "playhead" not in got and "expanded" not in got and "record" not in got


@pytest.mark.parametrize("text", ['{"version": 0}', "{}", "this is not json",
                                  '{"version": 1, "scene": "not base64 of a scene"}'])
def test_a_bad_text_raises_rok_file_error(text):
    with pytest.raises(rokfile.RokFileError) as exc:
        rokfile.unpack(text)
    assert str(exc.value).startswith("Not a Rok file")


def test_a_text_without_clips_raises_rok_file_error():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    data = json.loads(rokfile.pack(scene, {}))
    del data["clips"]
    with pytest.raises(rokfile.RokFileError) as exc:
        rokfile.unpack(json.dumps(data))
    assert str(exc.value).startswith("Not a Rok file")


def test_a_text_with_a_short_model_entry_raises_rok_file_error():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    data = json.loads(rokfile.pack(scene, {}))
    data["models"][0] = data["models"][0][:2]
    with pytest.raises(rokfile.RokFileError) as exc:
        rokfile.unpack(json.dumps(data))
    assert str(exc.value).startswith("Not a Rok file")


def test_every_clip_and_the_active_one_round_trip():
    """Phase 2: a scene holds several clips."""
    scene = VoxScene.from_data(loads_scene(_animated_file()))
    walk = anim.Clip(name="Walk", fps=30, length=20, loop_mode=anim.PING_PONG, voxel_look=False, root_motion=True)
    scene.clips.append(walk)
    scene.active_clip = 1
    again, _doc = rokfile.unpack(rokfile.pack(scene, {}))
    assert [c.name for c in again.clips] == [scene.clips[0].name, "Walk"] and again.active_clip == 1
    assert again.timeline is again.clips[1]
    w = again.clips[1]
    assert (w.fps, w.length, w.loop_mode, w.voxel_look, w.root_motion) == (30, 20, anim.PING_PONG, False, True)
    assert len(again.clips[0].tracks) == len(scene.clips[0].tracks)


def _two_by_two() -> VoxelGrid:
    voxels = np.zeros((2, 2, 2), dtype=np.uint8)
    voxels[0, 0, 0] = 1
    return VoxelGrid((2, 2, 2), voxels)


def test_uids_that_skip_a_number_after_a_delete_keep_the_track_on_the_right_model():
    scene = VoxScene()
    scene.add(_two_by_two(), name="First")
    middle = scene.add(_two_by_two(), name="Middle")
    scene.add(_two_by_two(), name="Last")
    scene.remove(scene.index_of(middle.uid))
    assert [m.uid for m in scene.models] == [1, 3]

    last = scene.models[1]
    track = anim.Track(anim.Target(anim.PART, last.uid))
    anim.set_key(track, anim.MOVE_Y, anim.Key(0, 0))
    anim.set_key(track, anim.MOVE_Y, anim.Key(5, 2))
    scene.timeline.tracks.append(track)

    text = rokfile.pack(scene, {})
    again, _doc = rokfile.unpack(text)

    assert [m.name for m in again.models] == ["First", "Last"]
    assert again.timeline.track_of(again.models[0].uid) is None  # "First" carries nothing
    new_track = again.timeline.track_of(again.models[1].uid)  # "Last", whatever its new uid is
    assert new_track is not None
    assert [(k.time, k.value) for k in new_track.keys(anim.MOVE_Y)] == [(0, 0), (5, 2)]


def test_a_linked_copy_keeps_link_and_orient():
    scene = VoxScene()
    first = scene.add(_two_by_two(), name="Source")
    second = scene.add(_two_by_two(), name="Copy")
    link_id = scene.new_link()
    first.link = link_id
    second.link = link_id
    second.orient = link.as_orient(link.turn_matrix(2, 1))

    text = rokfile.pack(scene, {})
    again, _doc = rokfile.unpack(text)

    assert again.models[0].link == again.models[1].link == link_id
    assert again.models[0].orient == link.IDENTITY
    assert again.models[1].orient == second.orient


# --- the zip on disk ---------------------------------------------------------------------


def test_the_zip_round_trips_and_its_vox_opens_on_its_own(tmp_path):
    import zipfile

    scene = VoxScene.from_data(loads_scene(_animated_file()))
    path = tmp_path / "walk.rok"
    rokfile.write(path, scene, {"active": 1})
    again, doc = rokfile.read(path)
    assert doc == {"active": 1}
    _assert_same_scene_contents(again, scene)
    assert len(again.timeline.tracks) == len(scene.timeline.tracks)
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["rok.json", "scene.vox"]
        inner = VoxScene.from_data(loads_scene(z.read("scene.vox")))  # what a .vox editor gets
    assert len(inner.models) == len(scene.models)


def test_live_shaders_survive_the_zip():
    from rok.core import live_shader as live_core

    scene = VoxScene.from_data(read_scene(GOLDEN))
    model = scene.models[1]
    old = model.grid.voxels.copy()
    new = old.copy()
    new[old > 0] = 9
    selection = np.zeros(old.shape, bool)
    selection[0, :, :] = True
    live = live_core.LiveShader(3, "weathering/rust", {"a0": 2.0, "colour": 9}, selection, (True, False, False),
                                True, live_core.Diff.of(old, new))
    scene.live[model.uid] = [live]
    assert rokfile.needs_rok(scene) == ("live shaders",)
    again, _doc = rokfile.loads(rokfile.dumps(scene, {}))
    (back,) = again.live[again.models[1].uid]
    assert (back.id, back.shader, back.settings, back.mirror, back.on) == (3, "weathering/rust", live.settings,
                                                                           (True, False, False), True)
    assert np.array_equal(back.selection, selection)
    assert np.array_equal(back.diff.revert(new), old)


def test_a_plain_scene_does_not_need_a_rok_file():
    assert rokfile.needs_rok(VoxScene.from_data(read_scene(GOLDEN))) == ()


def test_a_broken_zip_is_not_a_rok_file():
    with pytest.raises(rokfile.RokFileError, match="Not a Rok file"):
        rokfile.loads(b"PK\x03\x04 not really")


def test_an_empty_part_and_its_parent_round_trip():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    root = scene.add(VoxelGrid((1, 1, 1), np.zeros((1, 1, 1), np.uint8)), "Root", (0, 0, 0))
    root.empty = True
    scene.models[0].parent = root.uid
    again, _doc = rokfile.unpack(rokfile.pack(scene, {}))
    assert again.models[-1].empty and again.models[-1].name == "Root"
    assert again.models[0].parent == again.models[-1].uid and not again.models[0].empty


def test_the_inner_vox_holds_the_rest_pose_so_a_turned_part_comes_back_unturned():
    """A leg keyed at -30 degrees came back as a horizontal grid:
    the inner .vox baked the turn into a .vox frame and the reader
    turned the rest voxels by it. The inner .vox holds the parts at rest."""
    import zipfile, io

    scene = VoxScene()
    body = scene.add(VoxelGrid((4, 4, 4), np.full((4, 4, 4), 9, np.uint8)), "Body", (0, 0, 0))
    leg = scene.add(VoxelGrid((2, 2, 8), np.full((2, 2, 8), 37, np.uint8)), "Leg", (1, 1, -8))
    leg.parent = body.uid
    track = anim.Track(anim.Target(anim.PART, leg.uid), name="Leg")
    for t, v in ((0, -30.0), (9, 30.0), (18, -30.0)):
        anim.set_key(track, anim.TURN_X, anim.Key(t, v, anim.SMOOTH))
    scene.timeline.tracks.append(track)
    scene.timeline.length = 19
    for again in (rokfile.unpack(rokfile.pack(scene, {}))[0], rokfile.loads(rokfile.dumps(scene, {}))[0]):
        back = again.models[1]
        assert tuple(back.grid.size) == (2, 2, 8) and tuple(back.position) == (1, 1, -8)
        assert np.array_equal(back.grid.voxels, leg.grid.voxels)
        assert [(k.time, k.value) for k in again.timeline.track_of(back.uid).keys(anim.TURN_X)] == [(0, -30.0), (9, 30.0), (18, -30.0)]
    with zipfile.ZipFile(io.BytesIO(rokfile.dumps(scene, {}))) as z:
        inner = VoxScene.from_data(loads_scene(z.read("scene.vox")))
    assert not inner.timeline.tracks  # at rest: no frames in the .vox
    assert tuple(inner.models[1].grid.size) == (2, 2, 8)
