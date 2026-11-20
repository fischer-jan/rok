"""The agent tools after a field report: agents built a landing
site through the MCP server and wrote down every place the tools cost them
time. One test (or a few) per problem, in the report's order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rok_native")

from rok.core import png  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "agent"
LANDSCAPE = str(FIXTURES / "landscape.vox")


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session()
    yield s
    s.close()


def _rgba(data: bytes) -> np.ndarray:
    return png.decode(data, max_pixels=1 << 22)[2]


def _red_share(rgba: np.ndarray) -> float:
    """The share of pixels that show the landscape's red post."""
    r, g, b = (rgba[..., k].astype(int) for k in range(3))
    return float(((r > 120) & (r > 2 * g) & (r > 2 * b)).mean())


def _open_landscape(session):
    reply, _png = session.tool("rok_scene", {"do": "open", "discard": True, "path": LANDSCAPE})
    assert not reply.error, reply.text


def _ground(session, size=(64, 64, 24)):
    session.run("rok.new", size=tuple(list(size)), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [size[0] - 1, size[1] - 1, 1], "colour": 10})


# --- 1. the render camera can be placed ---------------------------------------------------


def test_render_default_camera_frames_the_whole_scene_as_before(session):
    _open_landscape(session)
    reply, picture = session.tool("rok_render", {"preset": "studio", "size": [320, 180], "samples": 4})
    assert not reply.error, reply.text
    assert "from the iso side" in reply.text
    assert _red_share(_rgba(picture)) < 0.02  # the post is a small thing in the whole scene


def test_render_from_a_low_eye_looks_along_the_ground(session):
    _open_landscape(session)
    reply, picture = session.tool("rok_render", {"preset": "studio", "size": [320, 180], "samples": 4,
                                                  "eye": [8.5, -1, 4], "target": [36, 50, 6], "fov": 60})
    assert not reply.error, reply.text
    assert "Camera at [8.5, -1.0, 4.0] looking at [36.0, 50.0, 6.0], 60° field of view" in reply.text
    rgba = _rgba(picture)
    assert _red_share(rgba) > 0.15  # the post stands right in front of the camera
    # The horizon: sky in the top rows, ground in the bottom rows.
    assert rgba[:20].mean() != pytest.approx(rgba[-20:].mean(), abs=5)


def test_render_azimuth_elevation_and_a_box_to_frame(session):
    _open_landscape(session)
    reply, picture = session.tool("rok_render", {"preset": "studio", "size": [320, 180], "samples": 4,
                                                  "azimuth": 270, "elevation": 8, "low": [0, 0, 0], "high": [20, 20, 12]})
    assert not reply.error, reply.text
    assert "azimuth 270°, elevation 8°" in reply.text and "frames cells [0, 0, 0]..[20, 20, 12]" in reply.text
    assert _red_share(_rgba(picture)) > 0.03  # framed on the corner with the post


def test_render_without_the_endless_floor(session):
    _open_landscape(session)
    args = {"preset": "sunset", "size": [320, 180], "samples": 4, "eye": [8.5, -1, 4], "target": [36, 50, 6]}
    _r, with_floor = session.tool("rok_render", args)
    reply, without = session.tool("rok_render", dict(args, ground=False))
    assert "endless floor is off" in reply.text
    assert not np.array_equal(_rgba(with_floor), _rgba(without))


@pytest.mark.parametrize("args, words", [
    ({"eye": [1, 1, 1], "azimuth": 30}, "not both"),
    ({"eye": [5, 5, 5], "target": [5, 5, 5]}, "same point"),
    ({"azimuth": 10, "fov": 200}, "5 to 120"),
    ({"eye": [1, 2]}, "eye must be [x, y, z]"),
    ({"low": [0, 0, 0]}, "give both"),
])
def test_render_camera_mistakes_say_how_to_fix_them(session, args, words):
    _open_landscape(session)
    reply, picture = session.tool("rok_render", dict({"size": [128, 128], "samples": 1}, **args))
    assert reply.error and words in reply.text and picture is None


def test_look_takes_the_same_camera(session):
    _open_landscape(session)
    reply, picture = session.tool("rok_look", {"eye": [8.5, -1, 4], "target": [36, 50, 6], "fov": 60,
                                                "size": [320, 180], "facts": False})
    assert not reply.error, reply.text
    assert "CAMERA x 0..320, y 0..180" in reply.text
    assert _red_share(_rgba(picture)) > 0.15
    probe, _ = session.tool("rok_probe", {"x": 70, "y": 120})
    assert probe.data["colour"] == 12  # the post, under the placed camera's pixel


def test_free_camera_stands_at_the_eye():
    from rok.headless import free_camera

    cam = free_camera(400, 300, (0, 0, 0), (10, 10, 10), eye=(1, -5, 2), target=(5, 5, 1), fov=50)
    assert np.allclose(cam.eye(), (1, -5, 2)) and np.allclose(cam.target, (5, 5, 1))
    assert cam.fov_y == pytest.approx(np.radians(50))
    framed = free_camera(400, 300, (0, 0, 0), (10, 10, 10), azimuth=270, elevation=0)
    eye = framed.eye() - framed.target
    assert eye[1] < 0 and abs(eye[0]) < 1e-6 and abs(eye[2]) < 1e-6  # in front (-y), level


# --- 2. a shader that changes nothing says why ---------------------------------------------


def test_a_shader_held_by_a_selection_says_why_it_built_nothing(session):
    _ground(session)
    session.tool("rok_select", {"do": "box", "low": [0, 0, 0], "high": [30, 30, 1]})
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "nature/woodland"})
    first = reply.text.splitlines()[0]
    assert first.startswith("Warning: nature/woodland changed nothing")
    assert "selection of" in first and "volume=" in first and "rok_select do=none" in first
    session.tool("rok_select", {"do": "none"})


def test_a_shader_with_no_room_above_says_so(session):
    _ground(session, (64, 64, 6))
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "nature/woodland"})
    assert "changed nothing" in reply.text and "do=resize" in reply.text


def test_the_window_dialog_names_the_selection_too():
    from rok.app import xs_ops

    hint = xs_ops._no_change_hint(xs_ops.shader_by_name("nature/woodland"), 1, 1, None, on_selection=True)
    assert "selection" in hint and "Volume" in hint


# --- 3. woods: dense, small, and in one piece ----------------------------------------------


def test_join_loose_bridges_edge_and_corner_touches():
    from rok.core.morph import join_loose

    before = np.zeros((6, 6, 6), dtype=np.uint8)
    before[:, :, 0] = 1  # ground
    after = before.copy()
    after[2, 2, 1:4] = 2  # a trunk on the ground
    after[3, 3, 3] = 3  # a leaf touching the trunk edge to edge
    after[1, 1, 4] = 4  # a leaf touching it corner to corner
    after[5, 5, 5] = 5  # a leaf touching nothing: left out
    out = join_loose(before, after)
    from rok.core import region

    assert len(region.components(out != 0)) == 1
    assert out[5, 5, 5] == 0 and out[3, 3, 3] == 3 and out[1, 1, 4] == 4
    assert np.array_equal(out[before != 0], before[before != 0])  # the old voxels stay
    assert int((out != 0).sum()) == int((after != 0).sum()) - 1 + 3  # one bridge, two bridges, one left out


@pytest.mark.parametrize("shader", ["nature/woodland", "nature/autumn_grove", "nature/jungle"])
def test_tree_shaders_leave_no_loose_parts(session, shader):
    _ground(session)
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": shader, "args": {"Size": 8, "Seed": 1}})
    assert reply.data["added"] > 500
    check, _ = session.tool("rok_check", {})
    assert "separate parts" not in check.text, check.text


def test_small_dense_woods(session):
    _ground(session)
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "nature/woodland",
                                             "args": {"Size": 6, "Density": 100, "Clearings": 0}})
    top = reply.data["facts"]["bounds"]["high"][2]
    assert reply.data["added"] > 1500 and top <= 12  # many trees, none taller than about 9 cells
    notes, _ = session.tool("rok_edit", {"do": "shaders", "shader": "nature/woodland"})
    assert "Density to 100 and Clearings to 0" in notes.text


# --- 4. open replaces the scene; import adds to it -----------------------------------------


def test_open_and_new_refuse_to_drop_unsaved_work(session):
    _ground(session)
    for args in ({"do": "open", "path": LANDSCAPE}, {"do": "new", "size": [8, 8, 8]}):
        reply, _png = session.tool("rok_scene", args)
        assert reply.error and "cannot be undone" in reply.text and "do=import" in reply.text
        assert session.agent.doc.model.grid.size == (64, 64, 24)  # the work is still there


def test_import_adds_a_file_beside_the_scene_and_undo_takes_it_out(session):
    _ground(session)
    reply, _png = session.tool("rok_scene", {"do": "import", "path": str(FIXTURES / "crate.vox")})
    assert not reply.error, reply.text
    models = session.tool("rok_scene", {"do": "list"})[0].data["models"]
    assert len(models) == 2 and models[1]["active"] and models[1]["voxels"] == 216
    assert models[0]["voxels"] == 64 * 64 * 2  # the ground stays
    session.tool("rok_scene", {"do": "undo"})
    assert len(session.tool("rok_scene", {"do": "list"})[0].data["models"]) == 1


def test_import_of_a_missing_file_says_so(session):
    reply, _png = session.tool("rok_scene", {"do": "import", "path": "/no/such/ship.vox"})
    assert reply.error and "there is no file" in reply.text


# --- 5. rok_voxels_set in a batch, and 8. steps sent as a string --------------------------


@pytest.mark.parametrize("step", [
    {"tool": "rok_voxels_set", "args": {"voxels": [[1, 1, 2, 7], [2, 1, 2, 7]]}},
    {"tool": "rok_voxels_set", "voxels": [[1, 1, 2, 7], [2, 1, 2, 7]]},  # the arguments next to "tool"
    {"tool": "rok_voxels_set", "params": {"voxels": [[1, 1, 2, 7], [2, 1, 2, 7]]}},
])
def test_voxels_set_runs_in_a_batch(session, step):
    _ground(session, (8, 8, 8))
    reply, _png = session.tool("rok_batch", {"steps": [step]})
    assert not reply.error, reply.text
    assert session.agent.doc.grid.voxels[1, 1, 2] == 7 and session.agent.doc.grid.voxels[2, 1, 2] == 7


def test_a_batch_step_with_args_twice_shows_the_right_form(session):
    _ground(session, (8, 8, 8))
    reply, _png = session.tool("rok_batch", {"steps": [{"tool": "rok_voxels_set", "args": {}, "voxels": [[1, 1, 1, 3]]}]})
    assert reply.error and 'put every argument in "args"' in reply.text and '"tool": "rok_shape"' in reply.text


def test_batch_steps_as_a_string(session):
    _ground(session, (8, 8, 8))
    good = '[{"tool": "rok_voxels_set", "args": {"voxels": [[4, 4, 3, 9]]}}]'
    reply, _png = session.tool("rok_batch", {"steps": good})
    assert not reply.error and session.agent.doc.grid.voxels[4, 4, 3] == 9
    bad, _png = session.tool("rok_batch", {"steps": "[{'tool': 'rok_shape'}]"})
    assert bad.error and "must be an array of step objects, not a string" in bad.text


def test_voxels_set_without_voxels_shows_both_forms(session):
    _ground(session, (8, 8, 8))
    reply, _png = session.tool("rok_voxels_set", {})
    assert reply.error and '"voxels": [[0, 0, 0, 12]' in reply.text and '"args": {"voxels"' in reply.text


# --- 6. quad in a list ---------------------------------------------------------------------


def test_look_takes_quad_in_a_list(session):
    _ground(session, (8, 8, 8))
    plain, _ = session.tool("rok_look", {"views": "quad", "size": [200, 200]})
    listed, _ = session.tool("rok_look", {"views": ["quad"], "size": [200, 200]})
    assert not listed.error and listed.text.splitlines()[-1] == plain.text.splitlines()[-1]
    mixed, _ = session.tool("rok_look", {"views": ["quad", "iso"]})
    assert mixed.error and "give it alone" in mixed.text


# --- 7. every colour in the facts ----------------------------------------------------------


def test_facts_list_every_colour_in_use(session):
    reply, _png = session.tool("rok_scene", {"do": "open", "discard": True, "path": str(FIXTURES / "many_colours.vox")})
    facts = reply.data["facts"]
    assert sorted(facts["colours"]) == list(range(1, 25))
    assert facts["colours"][24] == 16 and facts["colours"][1] == 1  # column k has min(k, 16) cells
    assert facts["where"][24] == {"hex": facts["where"][24]["hex"], "low": [23, 0, 0], "high": [23, 3, 3]}
    assert facts["where"][3] == {"hex": facts["where"][3]["hex"], "low": [2, 0, 0], "high": [2, 2, 0]}
    assert "  24 #" in reply.text


# --- 9. big boxes --------------------------------------------------------------------------


def test_a_256_wide_ground_slab_in_one_shape(session):
    session.run("rok.new", size=tuple([256, 256, 16]), discard=True)
    reply, _png = session.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [255, 255, 5], "colour": 10})
    assert not reply.error, reply.text
    assert reply.data["added"] == 256 * 256 * 6
    too_big, _ = session.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [255, 255, 64]})
    assert too_big.error and "rok_batch" in too_big.text


# --- 10 and 11. rok_command ---------------------------------------------------------------


def test_command_list_carries_the_rows_in_the_data(session):
    reply, _png = session.tool("rok_command", {"do": "list", "filter": "terrain_new"})
    rows = reply.data["commands"]
    assert reply.data["count"] == len(rows) == 1 and rows[0].startswith("rok.terrain_new(across=3")


def test_command_run_refuses_a_parameter_it_does_not_have(session):
    _ground(session, (8, 8, 8))
    reply, _png = session.tool("rok_command", {"do": "run", "id": "rok.terrain_new", "props": {"tiles": 2}})
    assert reply.error and "has no parameter tiles" in reply.text and "across=3" in reply.text
    assert len(session.agent.doc.scene.models) == 1


def test_terrain_new_takes_its_props(session):
    _ground(session, (8, 8, 8))
    reply, _png = session.tool("rok_command", {"do": "run", "id": "rok.terrain_new",
                                                "props": {"across": 2, "deep": 1, "size": 16, "height": 8, "seed": 3}})
    assert not reply.error, reply.text
    tiles = session.agent.doc.scene.models[1:]
    assert len(tiles) == 2 and all(tuple(m.grid.size) == (16, 16, 8) for m in tiles)


# --- 12. one tree in a volume of a big grid ------------------------------------------------


@pytest.mark.parametrize("shader", ["nature/trees", "nature/foliage", "nature/mushrooms"])
def test_a_one_thing_shader_builds_inside_an_off_centre_volume(session, shader):
    _ground(session, (128, 128, 48))
    volume = [5, 5, 2, 30, 30, 30]
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": shader, "volume": volume})
    assert not reply.error and reply.data["added"] > 100, reply.text
    v = session.agent.doc.grid.voxels.copy()
    v[:, :, :2] = 0
    cells = np.argwhere(v)
    assert cells.min(axis=0).tolist() >= volume[:3] and (cells.max(axis=0) <= volume[3:]).all()
