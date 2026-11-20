"""Voxel art scenes (rok/agent_art.py): the rules a scene the LLM starts keeps,
the flat views, the palette with ramps and the craft feedback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rok_native")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session()
    yield s
    s.close()


def _call(session, tool, args):
    reply, png = session.tool(tool, args)
    return reply, png


def _art(session, size=(24, 24, 24), budget=None, colours=None):
    args = {"do": "new", "discard": True, "size": list(size)}
    if budget:
        args["budget"] = budget
    reply, _ = _call(session, "rok_scene", args)
    assert not reply.error, reply.text
    if colours:
        reply, _ = _call(session, "rok_edit", {"do": "palette", "colours": colours})
        assert not reply.error, reply.text
    return reply


def _box(session, lo, hi, colour):
    return _call(session, "rok_shape", {"kind": "box", "low": list(lo), "high": list(hi), "colour": colour})[0]


def test_a_new_scene_is_a_voxel_art_scene_of_at_most_64_a_side(session):
    reply, _ = _call(session, "rok_scene", {"do": "new", "discard": True, "size": [80, 16, 16]})
    assert reply.error and "64" in reply.text
    reply = _art(session)
    assert "Voxel art scene" in reply.text and "blockout" in reply.text
    # the app's New, reached through rok_command, starts one too
    reply, _ = _call(session, "rok_command", {"do": "run", "id": "rok.new", "props": {"size": [100, 8, 8]}})
    assert reply.error and "64" in reply.text


def test_a_resize_past_the_limit_is_refused_though_no_voxel_changes(session):
    """A resize that keeps every voxel in place changes no cell, and the rules still apply."""
    _art(session, colours={"wood": "#8a5a2b"})
    assert not _box(session, (0, 0, 0), (3, 3, 3), 1).error
    reply, _ = _call(session, "rok_edit", {"do": "resize", "size": [200, 200, 200]})
    assert reply.error and "undone" in reply.text
    assert tuple(session.document.grid.size) == (24, 24, 24)
    assert int((session.document.grid.voxels != 0).sum()) == 64


def test_a_resize_past_the_limit_is_refused_when_the_undo_list_is_full(session):
    _art(session, colours={"wood": "#8a5a2b"})
    undo = session.document.undo
    for i in range(undo._undo.maxlen + 2):
        assert not _box(session, (i % 20, i // 20, 0), (i % 20, i // 20, 0), 1).error
    assert len(undo._undo) == undo._undo.maxlen
    reply, _ = _call(session, "rok_edit", {"do": "resize", "size": [200, 200, 200]})
    assert reply.error and "undone" in reply.text
    assert tuple(session.document.grid.size) == (24, 24, 24)


def test_an_unnamed_colour_is_refused_and_undone(session):
    _art(session)
    reply = _box(session, (0, 0, 0), (3, 3, 3), 7)
    assert reply.error and "no name" in reply.text and "do=palette" in reply.text
    assert int((session.document.grid.voxels != 0).sum()) == 0


def test_the_palette_names_ramps_lightest_first(session):
    _art(session)
    reply, _ = _call(session, "rok_edit", {"do": "palette", "colours": {"wood": "#8a5a2b", "leaf": "#4a7a3a"},
                                              "ramps": {"wood": 3}})
    assert not reply.error and "wood 1 = 1" in reply.text and "wood 3 = 3" in reply.text and "leaf = 4" in reply.text
    pal = session.document.scene.palette
    lum = [int(pal[k, :3].astype(int).sum()) for k in (1, 2, 3)]
    assert lum[0] > lum[1] > lum[2]
    reply, _ = _call(session, "rok_edit", {"do": "palette", "colours": {"wood": "#8a5a2b"}, "ramps": {"wood": 9}})
    assert reply.error and "2 to 6" in reply.text


def _two_boxes(session, colour=1):
    """Two loose boxes in one batch: one feedback, so the blockout cannot end between them."""
    reply, _ = _call(session, "rok_batch", {"steps": [
        {"tool": "rok_shape", "args": {"kind": "box", "low": [0, 0, 0], "high": [3, 3, 3], "colour": 1}},
        {"tool": "rok_shape", "args": {"kind": "box", "low": [10, 10, 0], "high": [13, 13, 3],
                                          "colour": colour}}]})
    return reply


def test_the_blockout_takes_one_colour_until_the_feedback_says_it_reads(session):
    _art(session, colours={"stone": "#8a8478", "moss": "#4a7a3a"})
    assert "Blockout not done" in _two_boxes(session).text
    reply = _box(session, (10, 10, 4), (13, 13, 4), 2)
    assert reply.error and "blockout" in reply.text and "rok_look" not in reply.text
    # the edit that joins the parts ends the blockout in its own feedback, which shows the shape
    reply = _box(session, (0, 0, 0), (13, 13, 0), 1)
    assert "Blockout done" in reply.text and "stage colour" in reply.text
    reply = _box(session, (10, 10, 4), (13, 13, 4), 2)
    assert not reply.error and "Painted" not in reply.text


def test_a_batch_cannot_block_out_and_colour_before_its_feedback_shows_the_shape(session):
    _art(session, colours={"stone": "#8a8478", "moss": "#4a7a3a"})
    reply, _ = _call(session, "rok_batch", {"steps": [
        {"tool": "rok_shape", "args": {"kind": "box", "low": [4, 4, 0], "high": [11, 11, 8], "colour": 1}},
        {"tool": "rok_shape", "args": {"kind": "box", "low": [4, 4, 8], "high": [11, 11, 8], "colour": 2,
                                          "mode": "paint"}}]})
    assert reply.error and "blockout" in reply.text
    reply = _box(session, (4, 4, 0), (11, 11, 8), 1)
    assert "Blockout done" in reply.text
    reply, _ = _call(session, "rok_shape", {"kind": "box", "low": [4, 4, 8], "high": [11, 11, 9], "colour": 2,
                                               "mode": "paint"})
    assert "Painted 64 filled cells of a box" in reply.text and "64 of its cells were empty" in reply.text


def test_a_blockout_of_loose_parts_is_not_done(session):
    _art(session, colours={"stone": "#8a8478"})
    _two_boxes(session)
    reply, _ = _call(session, "rok_look", {})
    assert "Blockout not done" in reply.text and "join the 2 parts" in reply.text


def test_the_budget_caps_voxels_and_colours(session):
    _art(session, budget={"voxels": 100, "colours": 2})
    reply, _ = _call(session, "rok_edit", {"do": "palette", "colours": {"a": "#ff0000", "b": "#00ff00",
                                                                            "c": "#0000ff"}})
    assert reply.error and "budget is 2" in reply.text
    _call(session, "rok_edit", {"do": "palette", "colours": {"a": "#ff0000"}})
    reply = _box(session, (0, 0, 0), (4, 4, 4), 1)  # 125 voxels
    assert reply.error and "over the budget of 100" in reply.text
    reply = _box(session, (0, 0, 0), (3, 3, 3), 1)
    assert not reply.error and "voxels 64/100" in reply.text


def test_voxels_set_places_a_few_cells_at_a_time(session):
    _art(session, colours={"stone": "#8a8478"})
    rows = [[x, y, 0, 1] for x in range(10) for y in range(7)]
    reply, _ = _call(session, "rok_voxels_set", {"voxels": rows})
    assert reply.error and "at most 64" in reply.text
    reply, _ = _call(session, "rok_voxels_set", {"voxels": rows[:60]})
    assert not reply.error


def test_an_edit_answers_with_flat_views_and_a_craft_line(session):
    _art(session, colours={"stone": "#8a8478"})
    reply, png = _call(session, "rok_shape", {"kind": "box", "low": [2, 2, 0], "high": [9, 5, 7], "colour": 1})
    assert png is not None and png[:4] == b"\x89PNG" and reply.picture is None
    assert "Flat views" in reply.text and "Craft: one part" in reply.text and "Voxel art: voxels 256" in reply.text
    assert reply.data["craft"]["parts"] == 1


def test_look_text_gives_the_views_as_characters(session):
    _art(session, colours={"stone": "#8a8478"})
    _box(session, (2, 2, 0), (4, 2, 1), 1)
    reply, _ = _call(session, "rok_look", {"text": True})
    front = reply.data["text_views"]["front"]
    assert front == ["###", "###"] and reply.data["text_views"]["legend"] == {"#": 1}


def test_a_render_waits_for_one_connected_part(session, tmp_path):
    _art(session, colours={"stone": "#8a8478"})
    _box(session, (0, 0, 0), (3, 3, 3), 1)
    _box(session, (10, 10, 0), (12, 12, 2), 1)
    reply, _ = _call(session, "rok_render", {"size": [64, 64], "samples": 1, "path": str(tmp_path / "r.png")})
    assert reply.error and "floating parts" in reply.text


def test_a_shader_names_the_colours_it_brings(session):
    _art(session, size=(32, 32, 24), colours={"grass": "#4a8a3a"})
    _box(session, (0, 0, 0), (31, 31, 1), 1)
    _call(session, "rok_look", {})
    reply, _ = _call(session, "rok_edit", {"do": "shader", "shader": "nature/trees",
                                             "volume": [8, 8, 2, 24, 24, 22]})
    assert not reply.error, reply.text
    assert "Named the shader's" in reply.text


def test_an_opened_file_is_free_and_gets_a_verdict(session):
    reply, _ = _call(session, "rok_scene", {"do": "open", "discard": True, "path": str(FIXTURES / "bird.vox")})
    assert "Verdict: voxel art" in reply.text and reply.picture is not None  # the quad, not flat views
    reply = _box(session, (0, 0, 0), (2, 2, 2), 200)  # an unnamed colour is fine here
    assert not reply.error


def test_the_verdict_tells_voxel_art_from_voxelized_3d():
    from rok.agent_art import verdict
    from rok.core.vox_read import read_scene

    for name in ("bird.vox", "car.vox", "robot.vox"):
        grids = [m.voxels for m in read_scene(FIXTURES / name).models]
        assert verdict(grids)[0] == "voxel art", name
    big = np.zeros((150, 60, 60), np.uint8)
    big[:, 10:50, 10:50] = 1
    kind, why = verdict([big])
    assert kind == "voxelized 3D" and any("voxels" in w for w in why) and any("side" in w for w in why)


def test_the_probe_reads_the_flat_views(session):
    _art(session, colours={"stone": "#8a8478"})
    _box(session, (0, 0, 0), (7, 3, 5), 1)
    _call(session, "rok_look", {"scale": 10})
    reply, _ = _call(session, "rok_probe", {"x": 35, "y": 5})  # front view: column 3, top row
    assert reply.data["cell"] == [3, 0, 5] and reply.data["face"] == "-y"


def test_look_lists_a_short_palette_unless_asked_for_all(session):
    _call(session, "rok_scene", {"do": "open", "discard": True, "path": str(FIXTURES / "agent" / "many_colours.vox")})
    short, _ = _call(session, "rok_look", {})
    full, _ = _call(session, "rok_look", {"detail": "full"})
    assert "and 12 more colours" in short.text and "more colours" not in full.text
    assert len(full.text) > len(short.text)


def test_the_shaders_stay_when_their_files_go(session, monkeypatch):
    from rok.app import xs_ops

    session.agent._shaders()  # read once
    monkeypatch.setattr(xs_ops, "shader_files", lambda: {})
    monkeypatch.setattr(xs_ops, "shader_by_name", lambda name: None)
    assert "nature/trees" in session.agent._shaders()
    assert session.agent._shader("nature/trees") is not None


def test_a_big_art_grid_takes_bigger_groups_of_cells(session):
    _art(session, size=(64, 48, 64), colours={"stone": "#8a8478"})
    rows = [[x, y, 0, 1] for x in range(16) for y in range(16)]  # 256 cells
    reply, _ = _call(session, "rok_voxels_set", {"voxels": rows})
    assert not reply.error, reply.text
    reply, _ = _call(session, "rok_voxels_set", {"voxels": rows + [[20, 20, 0, 1]]})
    assert reply.error and "at most 256" in reply.text


def test_the_budget_of_an_open_scene_can_be_raised_but_not_under_what_it_uses(session):
    _art(session, budget={"voxels": 100, "colours": 2}, colours={"a": "#ff0000"})
    assert not _box(session, (0, 0, 0), (3, 3, 3), 1).error  # 64 voxels
    reply, _ = _call(session, "rok_scene", {"do": "budget", "budget": {"voxels": 50}})
    assert reply.error and "64 voxels" in reply.text
    reply, _ = _call(session, "rok_scene", {"do": "budget", "budget": {"voxels": 400}})
    assert not reply.error and "400 voxels" in reply.text
    reply = _box(session, (0, 0, 0), (4, 4, 4), 1)  # 125 voxels, over the old cap
    assert not reply.error and "voxels 125/400" in reply.text


def test_a_pair_of_eyes_is_an_accent_and_one_stray_voxel_is_noise(session):
    _art(session, colours={"skin": "#e9c99a", "eye": "#1d1a22", "lamp": "#ffd27a"})
    _box(session, (4, 4, 0), (11, 11, 7), 1)
    _call(session, "rok_look", {})  # ends the blockout
    reply, _ = _call(session, "rok_voxels_set", {"voxels": [[6, 3, 5, 2], [9, 3, 5, 2]]})
    assert not reply.error, reply.text
    assert "single voxel" not in reply.text
    reply, _ = _call(session, "rok_voxels_set", {"voxels": [[7, 3, 2, 3]]})
    assert "lamp is used on a single voxel" in reply.text


def test_a_big_change_is_not_outlined_in_magenta(session):
    _art(session, colours={"stone": "#8a8478"})
    reply = _box(session, (0, 0, 0), (9, 9, 3), 1)
    assert "Most cells changed, so they are not outlined" in reply.text
    reply = _box(session, (0, 0, 4), (1, 1, 4), 1)
    assert "outlined in magenta" in reply.text


def test_a_ramp_keeps_the_hue_of_its_colour():
    import colorsys

    from rok.agent_art import ramp

    def hue(c):
        return colorsys.rgb_to_hsv(*(v / 255 for v in c))[0]

    red = ramp("d8402e", 3)  # a cap: its light step must not turn pink
    assert red[0][0] > red[0][2] * 2 and abs(hue(red[0]) - hue(red[1])) < 0.04
    skin = ramp("e9c99a", 2)  # its shadow must stay skin, not grey-brown
    assert skin[1][0] > 180 and abs(hue(skin[1]) - hue((0xe9, 0xc9, 0x9a))) < 0.04
    grey = ramp("7d7a86", 3)
    assert sum(grey[0]) > sum(grey[1]) > sum(grey[2])
