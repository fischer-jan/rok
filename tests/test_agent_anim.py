"""rok_animate and rok_render (rok/agent_anim.py), headless, on the animation fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rok_native")

from rok.core import png  # noqa: E402

ANIM = Path(__file__).parent / "fixtures" / "anim"


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session()
    yield s
    s.close()


def _rgba(data: bytes) -> np.ndarray:
    return png.decode(data, max_pixels=1 << 23)[2]


def _pole(session):
    """A base with a pole on it, the pole made a part that turns about its foot."""
    session.run("rok.new", size=tuple([24, 24, 24]), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [8, 8, 0], "high": [15, 15, 3], "colour": 5})
    session.tool("rok_shape", {"kind": "box", "low": [11, 11, 4], "high": [12, 12, 12], "colour": 9})
    session.tool("rok_select", {"do": "box", "low": [11, 11, 4], "high": [12, 12, 12]})
    reply, _png = session.tool("rok_animate", {"do": "make_part", "name": "pole"})
    assert not reply.error, reply.text
    reply, _png = session.tool("rok_animate", {"do": "pivot", "pivot": [1, 1, 0]})
    assert not reply.error, reply.text


def _clip(session):
    from rok.app.object import document

    return document.scene.timeline


def test_status_reads_the_parts_clips_and_keys_of_a_rok_file(session):
    session.open(ANIM / "propeller_spin.rok")
    reply, picture = session.tool("rok_animate", {"do": "status"})
    assert not reply.error and picture is None
    assert '* 0 "Clip 1": 24 frames (0 to 23) at 24 fps' in reply.text
    assert "Rotor, pivot (10.5, 10.5, 0)" in reply.text
    assert "turn_z: 0=0 linear, 24=360 linear" in reply.text
    assert [p["name"] for p in reply.data["parts"]] == ["Hub", "Rotor"]


def test_make_part_cuts_the_selection_into_a_part_that_follows_its_model(session):
    _pole(session)
    from rok.app.object import document

    scene = document.scene
    pole, base = scene.models[1], scene.models[0]
    assert pole.name == "pole" and pole.parent == base.uid
    assert int((pole.grid.voxels != 0).sum()) == 36 and int((base.grid.voxels == 9).sum()) == 0
    assert pole.pivot2 == (2, 2, 0)


def test_a_key_poses_the_part_in_the_picture_and_is_one_undo_step(session):
    _pole(session)
    rest, rest_png = session.tool("rok_animate", {"do": "key", "part": "pole", "frame": 0, "turn": [0, 0, 0]})
    bent, bent_png = session.tool("rok_animate", {"do": "key", "part": "pole", "frame": 12, "turn": [60, 0, 0],
                                                    "ease": "linear"})
    assert not bent.error and "turn_x 60" in bent.text
    assert "poses them at frame 0" in rest.text and "poses them at frame 12" in bent.text
    track = _clip(session).tracks[0]
    assert [(k.time, k.value, k.ease) for k in track.keys("TURN_X")] == [(0, 0.0, "SMOOTH"), (12, 60.0, "LINEAR")]
    # The pole leans in the picture at frame 12: the RIGHT view (bottom right) differs from the rest pose.
    a, b = _rgba(rest_png), _rgba(bent_png)
    assert np.abs(a[500:, 500:, :3].astype(int) - b[500:, 500:, :3].astype(int)).sum() > 100_000
    # The document's voxels never move: a pose is a picture, not an edit.
    from rok.app.object import document

    assert int((document.scene.models[1].grid.voxels != 0).sum()) == 36
    session.tool("rok_scene", {"do": "undo"})
    assert [k.time for k in _clip(session).tracks[0].keys("TURN_X")] == [0]


def test_look_shows_a_strip_of_frames_one_square_tile_each(session):
    session.open(ANIM / "propeller_spin.rok")
    reply, picture = session.tool("rok_animate", {"do": "look", "frames": [0, 3, 6], "size": [600, 600]})
    assert not reply.error
    assert "3D AT FRAME 3 x 200..400, y 0..200" in reply.text
    rgba = _rgba(picture)
    assert rgba.shape == (200, 600, 4)
    # A quarter turn of the rotor between frame 0 and 6 changes the tile; the tiles are not copies.
    t0, t6 = rgba[:, :200, :3].astype(int), rgba[:, 400:, :3].astype(int)
    assert np.abs(t0 - t6)[40:].sum() > 50_000


def test_a_motion_writes_keys_over_the_clip(session):
    _pole(session)
    reply, picture = session.tool("rok_animate", {"do": "motion", "part": "pole", "name": "spin", "axis": "z"})
    assert not reply.error and "Spin on \"pole\" about z" in reply.text
    track = _clip(session).tracks[0]
    keys = track.keys("TURN_Z")
    assert keys[0].time == 0 and keys[-1].time == _clip(session).length and abs(keys[-1].value) == 360
    assert _rgba(picture).shape[1] == 1000  # a strip of four frames


def test_voxel_frames_are_made_changed_shown_and_deleted(session):
    _pole(session)
    from rok.app.object import document

    pole = document.scene.models[1]
    added, _ = session.tool("rok_animate", {"do": "voxel_frame_add", "part": "pole", "frame": 6})
    assert not added.error and len(pole.states) == 1
    changed, picture = session.tool("rok_animate", {"do": "voxel_frame_set", "part": "pole", "voxel_frame": 1,
                                                      "voxels": [[0, 0, 8, 0], [1, 0, 8, 0], [5, 5, 5, 3]]})
    assert "Set 2 cells of voxel frame 1" in changed.text and "1 outside the grid were skipped" in changed.text
    assert "It shows at frame 6" in changed.text and picture is not None
    assert int((pole.states[0] != 0).sum()) == 34 and int((pole.grid.voxels != 0).sum()) == 36
    keys = _clip(session).tracks[0].keys("STATE")
    assert [(k.time, k.value) for k in keys] == [(0, 0), (6, 1)]
    own = session.tool("rok_animate", {"do": "voxel_frame_set", "voxel_frame": 0, "voxels": [[0, 0, 0, 1]]})[0]
    assert own.error and "rok_voxels_set" in own.text
    deleted, _ = session.tool("rok_animate", {"do": "voxel_frame_delete", "part": "pole", "voxel_frame": 1})
    assert not deleted.error and pole.states == []
    assert "STATE" not in _clip(session).tracks[0].properties


def test_clip_settings_clips_and_markers(session):
    _pole(session)
    reply, _ = session.tool("rok_animate", {"do": "clip", "name": "Wave", "seconds": 2, "fps": 12, "loop": "ping_pong"})
    assert "24 frames (0 to 23) at 12 fps = 2.00 s, ping_pong" in reply.text
    session.tool("rok_animate", {"do": "marker", "frame": 5, "name": "step"})
    assert [(m.time, m.name) for m in _clip(session).markers] == [(5, "step")]
    session.tool("rok_animate", {"do": "clip_add", "name": "Idle"})
    status, _ = session.tool("rok_animate", {"do": "status", "clip": "Wave"})
    assert '* 0 "Wave"' in status.text and '  1 "Idle"' in status.text and 'Markers: 5 "step"' in status.text
    late, _ = session.tool("rok_animate", {"do": "key", "part": "pole", "frame": 30, "move": [0, 0, 2]})
    assert "past the clip's end" in late.text
    gone, _ = session.tool("rok_animate", {"do": "clip_delete", "clip": "Idle"})
    assert not gone.error and len(session.document.scene.clips) == 1
    last, _ = session.tool("rok_animate", {"do": "clip_delete"})
    assert last.error and "last clip stays" in last.text


def test_ease_delete_keys_and_a_colour_key(session):
    _pole(session)
    session.tool("rok_animate", {"do": "key", "part": "pole", "frame": 8, "move": [0, 0, 3], "turn": [0, 0, 90]})
    eased, _ = session.tool("rok_animate", {"do": "ease", "part": "pole", "ease": "ease_out", "channels": ["move_z"]})
    assert eased.text.startswith("2 keys")
    track = _clip(session).tracks[0]
    assert {k.ease for k in track.keys("MOVE_Z")} == {"EASE_OUT"} and track.keys("TURN_Z")[1].ease == "SMOOTH"
    deleted, _ = session.tool("rok_animate", {"do": "delete_keys", "part": "pole", "channels": ["turn"], "frame": 8})
    assert deleted.text == 'Deleted 3 keys of "pole".'
    keyed, _ = session.tool("rok_animate", {"do": "colour_key", "colour": 5, "frame": 10, "hex": "#ff0000"})
    assert not keyed.error
    colour = next(t for t in _clip(session).tracks if t.target.kind == "COLOUR")
    assert colour.target.slots == (5,) and colour.keys("COLOUR")[-1].value == (255, 0, 0)


def test_parent_follows_and_refuses_a_loop(session):
    _pole(session)
    session.tool("rok_animate", {"do": "empty_part", "part": "Model 1", "name": "root"})
    from rok.app.object import document

    scene = document.scene
    root = scene.models[2]
    assert root.empty and root.name == "root"
    ok, _ = session.tool("rok_animate", {"do": "parent", "part": "pole", "parent": "root"})
    assert not ok.error and scene.models[1].parent == root.uid
    loop, _ = session.tool("rok_animate", {"do": "parent", "part": "root", "parent": "pole"})
    assert loop.error and "cannot follow" in loop.text
    free, _ = session.tool("rok_animate", {"do": "parent", "part": "pole", "parent": ""})
    assert scene.models[1].parent is None and "follows nothing" in free.text


def test_export_writes_gltf_vox_a_sprite_sheet_and_a_gif(session, tmp_path):
    session.open(ANIM / "propeller_spin.rok")
    for name in ("spin.glb", "spin.vox"):
        reply, _ = session.tool("rok_animate", {"do": "export", "path": str(tmp_path / name)})
        assert not reply.error and (tmp_path / name).stat().st_size > 1000
    assert (tmp_path / "spin.glb").read_bytes()[:4] == b"glTF"
    reply, _ = session.tool("rok_animate", {"do": "export", "path": str(tmp_path / "spin.png"), "directions": 4,
                                             "cell_size": 64})
    assert not reply.error, reply.text
    sheet = json.loads((tmp_path / "spin.json").read_text())
    assert len(sheet["frames"]) == 24 * 4
    assert _rgba((tmp_path / "spin.png").read_bytes()).shape[1] % 64 == 0
    reply, _ = session.tool("rok_animate", {"do": "export", "path": str(tmp_path / "spin.gif"), "view": "top"})
    assert not reply.error and (tmp_path / "spin.gif").read_bytes()[:6] == b"GIF89a"
    bad, _ = session.tool("rok_animate", {"do": "export", "path": str(tmp_path / "spin.fbx")})
    assert bad.error and ".glb (glTF)" in bad.text


def test_render_returns_a_path_traced_picture_and_saves_it(session, tmp_path):
    session.open(ANIM / "propeller_spin.rok")
    out = tmp_path / "pictures" / "rotor.png"
    reply, picture = session.tool("rok_render", {"preset": "studio", "size": [320, 240], "samples": 4,
                                                  "path": str(out)})
    assert not reply.error and "Rendered with Studio: 320 x 240 pixels, 4 samples" in reply.text
    assert picture is None and "not sent back" in reply.text  # saved: no picture tokens
    reply, picture = session.tool("rok_render", {"preset": "studio", "size": [320, 240], "samples": 4,
                                                  "path": str(out), "return_image": True})
    assert out.read_bytes() == picture
    rgba = _rgba(picture)
    assert rgba.shape == (240, 320, 4) and rgba[..., :3].std() > 5  # a picture, not one flat colour
    iso, iso_png = session.tool("rok_render", {"preset": "iso", "size": [200, 200], "samples": 2, "frame": 6})
    assert "posed at frame 6" in iso.text
    assert (_rgba(iso_png)[..., 3] < 255).any()  # Isometric leaves the background transparent
    bad, _ = session.tool("rok_render", {"preset": "glossy"})
    assert bad.error and "studio (Studio)" in bad.text


def test_mask_unmask_drop_voxels_and_export_merge(session, tmp_path):
    session.run("rok.new", size=tuple([8, 8, 8]), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [1, 1, 3], "high": [2, 2, 3], "colour": 4})
    session.tool("rok_select", {"do": "all"})
    masked, _ = session.tool("rok_select", {"do": "mask"})
    assert masked.data["masked"] == 4
    session.tool("rok_shape", {"kind": "box", "low": [0, 0, 3], "high": [7, 7, 3], "mode": "erase"})
    assert int((session.document.grid.voxels != 0).sum()) == 4  # the mask kept them
    session.tool("rok_select", {"do": "unmask"})
    session.tool("rok_select", {"do": "all"})
    dropped, _ = session.tool("rok_edit", {"do": "drop_voxels"})
    assert not dropped.error
    assert sorted(map(tuple, np.argwhere(session.document.grid.voxels != 0)[:, 2:].tolist())) == [(0,)] * 4
    for merge in ("on", "off"):
        reply, _ = session.tool("rok_scene", {"do": "export", "path": str(tmp_path / f"{merge}.obj"), "merge": merge})
        assert not reply.error
    assert (tmp_path / "on.obj").stat().st_size < (tmp_path / "off.obj").stat().st_size


def test_render_exposure_brightens_this_picture_and_bad_values_say_the_range(session):
    session.open(ANIM / "propeller_spin.rok")
    args = {"preset": "studio", "size": [160, 120], "samples": 4, "return_image": True}
    plain, plain_png = session.tool("rok_render", args)
    bright, bright_png = session.tool("rok_render", {**args, "exposure": 1.5, "bloom": 0})
    assert not bright.error and "Look: bloom 0, exposure 1.5 stops (this picture only)" in bright.text
    assert _rgba(bright_png)[..., :3].mean() > _rgba(plain_png)[..., :3].mean() + 10
    again, again_png = session.tool("rok_render", args)  # the file's light is unchanged
    assert abs(float(_rgba(again_png)[..., :3].mean()) - float(_rgba(plain_png)[..., :3].mean())) < 3
    bad, _ = session.tool("rok_render", {**args, "exposure": 9})
    assert bad.error and "-3 to 3 stops" in bad.text
    bad, _ = session.tool("rok_render", {**args, "bloom": 2})
    assert bad.error and "0 (none" in bad.text
