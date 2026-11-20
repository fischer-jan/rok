"""The agent tools' feedback (rok/agent.py) on models that are hard to read in a picture."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rok_native")

from rok.app import xs_ops  # noqa: E402
from rok.core import png  # noqa: E402

from helpers import FIXTURES  # noqa: E402

MARKER_SHADERS = FIXTURES / "marker_shaders"  # a personal folder with one marker shader, plume


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session()
    yield s
    s.close()


def _rgba(data: bytes) -> np.ndarray:
    return png.decode(data, max_pixels=1 << 22)[2]


def _room(session):
    session.run("rok.new", size=tuple([24, 24, 24]), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [2, 2, 0], "high": [21, 21, 15], "colour": 9, "hollow": True})


def test_resize_reports_no_change_for_voxels_that_stay_and_says_the_model_left_the_middle(session):
    session.run("rok.new", size=(9, 3, 8), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [2, 0, 0], "high": [6, 2, 3], "colour": 4})
    reply, _ = session.tool("rok_edit", {"do": "resize", "size": [13, 3, 8]})
    assert not reply.error, reply.text
    assert reply.data["added"] == 0 and reply.data["removed"] == 0
    assert "Resized the grid to 13 x 3 x 8" in reply.text
    assert "x 2..6 of 0..12" in reply.text and "offset [2, 0, 0]" in reply.text
    session.tool("rok_select", {"do": "all"})
    session.tool("rok_transform", {"do": "move", "offset": [2, 0, 0]})
    filled = np.argwhere(session.document.grid.voxels != 0)
    assert filled[:, 0].min() == 4 and filled[:, 0].max() == 8  # centred in 0..12
    # a grid that cannot hold it in the middle says so, without a move that does nothing
    reply, _ = session.tool("rok_edit", {"do": "resize", "size": [14, 3, 8]})
    assert "cannot sit exactly in the middle" in reply.text and "offset" not in reply.text
    # a model that was off the middle before gets no note
    reply, _ = session.tool("rok_edit", {"do": "resize", "size": [16, 3, 8]})
    assert "off the middle" not in reply.text


def test_hidden_finds_cells_closed_in_on_every_side():
    from rok.agent import Agent

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[:, :, :] = True
    mask[2, 2, 2] = False  # a hole in the middle
    cells = np.array([[2, 2, 2], [0, 2, 2], [2, 2, 1]])
    assert Agent.hidden(mask, cells).tolist() == [True, False, True]


def test_a_change_inside_a_closed_room_is_named_and_shows_through_the_walls(session):
    _room(session)
    reply, picture = session.tool("rok_shape", {"kind": "box", "low": [9, 9, 1], "high": [13, 13, 5], "colour": 1})
    assert "125 of the changed cells are hidden" in reply.text
    assert reply.data["warnings"][0].startswith("125 of the changed cells")
    rgba = _rgba(picture)
    magenta = (rgba[..., 0] > 180) & (rgba[..., 1] < 140) & (rgba[..., 2] > 160)
    assert magenta[500:, :500].sum() > 200  # the front view shows the hidden cube's outline


def test_a_cut_shows_the_inside_for_one_picture_and_changes_nothing(session):
    _room(session)
    session.tool("rok_shape", {"kind": "box", "low": [9, 9, 1], "high": [13, 13, 5], "colour": 1})
    before = session.document.grid.voxels.copy()
    closed, closed_png = session.tool("rok_look", {"views": ["front"], "size": [400, 400]})
    cut, cut_png = session.tool("rok_look", {"views": ["front"], "size": [400, 400], "cut_axis": "y", "cut_at": 11})
    assert "only cells with y >= 11 show" in cut.text
    assert np.array_equal(session.document.grid.voxels, before)
    a, b = _rgba(closed_png), _rgba(cut_png)
    # The front wall is cut away: the room's inside and the white cube show instead.
    assert (np.abs(a[..., :3].astype(int) - b[..., :3]).max(axis=2) > 40).sum() > 5000
    grey = lambda img: ((np.abs(img[40:360, 40:360, 0].astype(int) - img[40:360, 40:360, 2]) < 15)  # noqa: E731
                        & (img[40:360, 40:360, 0] > 150)).sum()
    assert grey(b) > grey(a) + 2000  # the white cube (neutral grey in the light) shows only in the cut
    again, again_png = session.tool("rok_look", {"views": ["front"], "size": [400, 400]})
    assert np.abs(_rgba(again_png).astype(int) - a).max() < 20  # the next picture is whole again


def test_a_dark_model_gets_a_light_background(session):
    session.run("rok.new", size=tuple([16, 16, 16]), discard=True)
    session.tool("rok_edit", {"do": "set_colour", "index": 200, "hex": "#26262b"})
    _r, dark = session.tool("rok_shape", {"kind": "sphere", "low": [3, 3, 3], "high": [12, 12, 12], "colour": 200})
    assert _rgba(dark)[3, 400, :3].mean() > 150  # the corner of the top view is light
    session.run("rok.new", size=tuple([16, 16, 16]), discard=True)
    _r, bright = session.tool("rok_shape", {"kind": "sphere", "low": [3, 3, 3], "high": [12, 12, 12], "colour": 9})
    assert _rgba(bright)[3, 400, :3].mean() < 80


def test_many_changes_get_one_box_and_keep_their_colour(session):
    session.run("rok.new", size=tuple([16, 16, 16]), discard=True)
    _r, pic = session.tool("rok_shape", {"kind": "box", "low": [2, 2, 0], "high": [13, 13, 3], "colour": 9})
    top = _rgba(pic)[:500, :500, :3].astype(int)
    background = top[5, 250]
    magenta = (top[..., 0] > 180) & (top[..., 1] < 140) & (top[..., 2] > 160)
    model = np.abs(top - background).max(axis=2) > 40
    # 144 changed cells: one box round them, so the box's own colour fills the view, not outlines.
    assert (model & ~magenta).mean() > 0.2 and magenta.mean() < 0.02


def test_two_views_sit_side_by_side():
    from rok.headless import tiles

    assert tiles(2, 1000, 500) == [(0.0, 0.0, 500.0, 500.0), (500.0, 0.0, 500.0, 500.0)]
    assert len(tiles(4, 1000, 1000)) == 4 and tiles(1, 10, 10) == [(0.0, 0.0, 10, 10)]


def test_a_small_change_on_a_big_model_gets_a_close_up_strip(session):
    from pathlib import Path

    session.open(Path(__file__).parent / "fixtures" / "bird.vox")
    reply, picture = session.tool("rok_shape", {"kind": "box", "low": [39, 30, 19], "high": [40, 31, 20], "colour": 170})
    assert "CLOSE-UP 3D x 0..333, y 1000..1333" in reply.text
    rgba = _rgba(picture)
    assert rgba.shape == (1333, 1000, 4)
    strip = rgba[1000:]
    magenta = (strip[..., 0] > 180) & (strip[..., 1] < 140) & (strip[..., 2] > 160)
    assert magenta.sum() > 500  # the change is big in the close-up
    hit, _ = session.tool("rok_probe", {"x": 500, "y": 1166})  # the middle of the close-up front view
    assert hit.data["view"] == "front" and hit.data["colour"] == 170


def test_a_change_on_a_small_model_needs_no_close_up(session):
    session.run("rok.new", size=tuple([24, 24, 24]), discard=True)
    reply, picture = session.tool("rok_shape", {"kind": "box", "low": [4, 4, 0], "high": [8, 8, 3], "colour": 9})
    assert "CLOSE-UP" not in reply.text and _rgba(picture).shape == (1000, 1000, 4)


def test_look_shows_a_close_up_of_a_box_with_what_is_in_front_cut_away(session):
    _room(session)
    session.tool("rok_shape", {"kind": "box", "low": [9, 9, 1], "high": [13, 13, 5], "colour": 1})
    reply, picture = session.tool("rok_look", {"views": ["front"], "low": [8, 8, 0], "high": [14, 14, 6], "size": [400, 400]})
    assert "Close-up of cells [8, 8, 0]..[14, 14, 6]" in reply.text
    hit, _ = session.tool("rok_probe", {"x": 200, "y": 250})
    assert hit.data["colour"] == 1  # the cube inside: the room's front wall is in front of the box, so it is cut away


# --- shaders, the current colour and fill_empty ("improve the mcp") -----------


def _hull(session):
    session.run("rok.new", size=tuple([56, 40, 24]), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [4, 6, 4], "high": [51, 33, 13], "colour": 9})


def _count(session) -> int:
    from rok.app.object import document

    return int((document.grid.voxels != 0).sum())


def test_a_shader_without_args_runs_at_its_own_defaults(session):
    """Arguments left out were 0, which clamped them to their lowest value:
    greebles at Density 0 and Seams 0 did nothing."""
    _hull(session)
    before = session.agent.doc.grid.voxels.copy()
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "hard_surface/greebles"})
    assert not reply.error, reply.text
    assert "Size = 12" in reply.text and "Density = 70" in reply.text
    assert not np.array_equal(session.agent.doc.grid.voxels, before)


def test_shader_args_by_name_and_the_rest_at_defaults(session):
    _hull(session)
    before = session.agent.doc.grid.voxels.copy()
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "hard_surface/greebles",
                                             "args": {"density": 0, "Seams": 0}})
    assert not reply.error, reply.text
    assert "Density = 0" in reply.text and "Size = 12" in reply.text
    assert np.array_equal(session.agent.doc.grid.voxels, before)  # nothing to do
    bad, _png = session.tool("rok_edit", {"do": "shader", "shader": "hard_surface/greebles", "args": {"Sparkle": 3}})
    assert bad.error and "Size = 12 (6 to 32)" in bad.text  # the arguments it has


def test_a_detail_shader_says_why_it_emptied_cells(session):
    """greebles answered "+257 added, -344 removed" with no word on why: it read as damage."""
    _hull(session)
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "hard_surface/greebles"})
    assert not reply.error, reply.text
    assert reply.data["removed"] > 0
    assert "on purpose: Seams cuts grooves" in reply.text and "Seams 0 keeps the hull whole" in reply.text
    _hull(session)
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "hard_surface/greebles",
                                             "args": {"Seams": 0, "Density": 0}, "only_colours": [9]})
    assert "on purpose" not in reply.text


def test_save_and_export_reply_without_a_picture(session, tmp_path):
    """After a save the model is as it was: the quad picture cost about 1,300 tokens for nothing."""
    _hull(session)
    reply, png = session.tool("rok_scene", {"do": "save", "path": str(tmp_path / "hull.vox")})
    assert not reply.error, reply.text
    assert png is None and "Saved" in reply.text and "voxels" in reply.text
    reply, png = session.tool("rok_scene", {"do": "export", "path": str(tmp_path / "hull.obj")})
    assert not reply.error, reply.text
    assert png is None and "Exported" in reply.text
    reply, png = session.tool("rok_scene", {"do": "undo"})
    assert png is not None  # a change of the model still shows the picture


def test_a_shader_can_keep_to_some_faces(session):
    from rok.core import shader_scope

    _hull(session)
    before = session.agent.doc.grid.voxels.copy()
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "hard_surface/battle_damage",
                                             "only_on": ["top"]})
    assert not reply.error, reply.text
    changed = session.agent.doc.grid.voxels != before
    assert changed.any() and not (changed & ~shader_scope.face_mask(before, "TOP")).any()
    bad, _png = session.tool("rok_edit", {"do": "shader", "shader": "hard_surface/battle_damage",
                                           "only_on": ["roof"]})
    assert bad.error and "top, sides" in bad.text


def test_the_shader_list_names_what_each_makes_and_its_arguments(session, monkeypatch):
    monkeypatch.setattr(xs_ops, "user_shader_folder", lambda: MARKER_SHADERS)
    reply, _png = session.tool("rok_edit", {"do": "shaders"})
    assert "hard_surface/greebles: Pipes, vents, tanks" in reply.text
    assert "Size = 12 (6 to 32)" in reply.text
    assert "plume: A short plume under voxels in a marker colour" in reply.text
    one, _png = session.tool("rok_edit", {"do": "shaders", "shader": "nature/rocks"})
    assert one.text.startswith("nature/rocks") and "hard_surface" not in one.text


def test_the_current_colour_is_set_and_used(session):
    _hull(session)
    reply, _png = session.tool("rok_edit", {"do": "current_colour", "index": 33})
    assert not reply.error and "current colour is 33" in reply.text
    session.tool("rok_shape", {"kind": "box", "low": [0, 0, 20], "high": [1, 1, 20]})
    assert session.agent.doc.grid.voxels[0, 0, 20] == 33


def test_fill_empty_keeps_what_is_there(session):
    _hull(session)
    reply, _png = session.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [55, 39, 15], "colour": 5,
                                              "mode": "fill_empty"})
    assert not reply.error and "empty cells" in reply.text
    v = session.agent.doc.grid.voxels
    assert v[10, 10, 8] == 9 and v[0, 0, 0] == 5  # the hull stays; around it is filled


def test_a_marker_shader_grows_from_the_marker_colour(session, monkeypatch):
    """The shader's colour is the marker: the plume grows under the painted voxels only."""
    monkeypatch.setattr(xs_ops, "user_shader_folder", lambda: MARKER_SHADERS)
    _hull(session)
    session.tool("rok_shape", {"kind": "box", "low": [10, 18, 4], "high": [13, 21, 4], "colour": 7, "mode": "paint"})
    before = _count(session)
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "plume", "colour": 7, "scope": "model"})
    assert not reply.error, reply.text
    v = session.agent.doc.grid.voxels
    assert _count(session) == before + 2 * 16  # two cells under each of the 16 marker voxels
    assert v[10:14, 18:22, 2:4].all() and not v[20:24, 18:22, 2:4].any()


def test_grow_keeps_only_what_grows_from_the_selection(session):
    """Grass on the selected rock only, not on the roof beside it."""
    session.run("rok.new", size=tuple([40, 30, 20]), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [2, 2, 0], "high": [17, 27, 3], "colour": 1})  # rock
    session.tool("rok_shape", {"kind": "box", "low": [22, 2, 0], "high": [37, 27, 3], "colour": 3})  # a roof
    session.tool("rok_select", {"do": "colour", "colour": 1})
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "weathering/cover",
                                             "args": {"Kind": 1, "Amount": 100}, "grow": True})
    assert not reply.error, reply.text
    assert "grows from the selection" in reply.text
    v = session.agent.doc.grid.voxels
    assert (v[2:18, :, 4:] != 0).any()  # grass on the rock
    assert not (v[22:38, :, 4:] != 0).any()  # none on the roof
    assert not session.agent.doc.selection.is_empty()  # the selection stays
    session.tool("rok_scene", {"do": "undo"})  # one undo step takes the grass away
    assert not (session.agent.doc.grid.voxels[:, :, 4:] != 0).any()
    session.tool("rok_select", {"do": "none"})
    bad, _png = session.tool("rok_edit", {"do": "shader", "shader": "weathering/cover", "grow": True})
    assert bad.error and "selection" in bad.text


def test_one_shaders_listing_has_its_own_notes(session):
    reply, _png = session.tool("rok_edit", {"do": "shaders", "shader": "weathering/cover"})
    assert "Kind 1 is grass" in reply.text and "xs_begin" not in reply.text and "arg :" not in reply.text


def test_rok_shape_makes_every_primitive(session):
    from rok.agent import SHAPE_KINDS

    session.run("rok.new", size=tuple([24, 24, 24]), discard=True)
    for kind in SHAPE_KINDS:
        if kind in ("line", "fill", "extrude", "curve"):
            continue  # they take cells and silhouettes, not a box (tests/test_voxel_draw.py)
        session.run("rok.new", size=tuple([24, 24, 24]), discard=True)
        reply, _png = session.tool("rok_shape", {"kind": kind, "low": [4, 4, 2], "high": [15, 11, 9], "colour": 4})
        assert not reply.error, (kind, reply.text)
        v = session.agent.doc.grid.voxels
        filled = np.argwhere(v)
        assert len(filled), kind
        assert filled.min(axis=0).tolist() >= [4, 4, 2] and filled.max(axis=0).tolist() <= [15, 11, 9], kind
    session.run("rok.new", size=tuple([24, 24, 24]), discard=True)
    session.tool("rok_shape", {"kind": "roof", "low": [2, 2, 2], "high": [13, 9, 6], "colour": 3})
    v = session.agent.doc.grid.voxels
    top = np.argwhere(v[:, :, 6])
    assert len(top) and set(top[:, 0].tolist()) >= {2, 13}  # the ridge runs along x, the longer side


def test_a_new_solid_is_not_warned_about_its_own_inside(session):
    session.run("rok.new", size=tuple([24, 24, 24]), discard=True)
    reply, _png = session.tool("rok_shape", {"kind": "box", "low": [2, 2, 2], "high": [12, 12, 12], "colour": 4})
    assert "hidden from every side" not in reply.text
    # Painting inside it is hidden by the box, which this change did not make: that warns.
    reply, _png = session.tool("rok_voxels_set", {"voxels": [[7, 7, 7, 9]]})
    assert "hidden from every side" in reply.text


def test_a_shaders_own_colours_can_be_set(session):
    """The shaders that use colours let you pick them (cover's grass)."""
    session.run("rok.new", size=tuple([24, 24, 12]), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [2, 2, 0], "high": [21, 21, 3], "colour": 9})
    listing, _png = session.tool("rok_edit", {"do": "shaders", "shader": "weathering/cover"})
    assert "Grass, Grass shade" in listing.text
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "weathering/cover",
                                             "args": {"Kind": 1, "Amount": 100},
                                             "colours": {"Grass": "#ff00ff", "Grass shade": "#aa00aa"}})
    assert not reply.error, reply.text
    doc = session.agent.doc
    v = doc.grid.voxels
    grown = {tuple(int(c) for c in doc.scene.palette[i, :3]) for i in np.unique(v[:, :, 4:]) if i}
    assert grown and grown <= {(255, 0, 255), (170, 0, 170)}  # only the colours you gave
    bad, _png = session.tool("rok_edit", {"do": "shader", "shader": "weathering/cover", "colours": {"Lava": "#ff0000"}})
    assert bad.error and "Snow" in bad.text


def test_a_shader_can_build_in_a_volume(session):
    session.run("rok.new", size=tuple([16, 16, 8]), discard=True)
    reply, _png = session.tool("rok_edit", {"do": "shader", "shader": "nature/trees", "volume": [0, 0, 0, 15, 15, 7]})
    assert not reply.error, reply.text
    bad, _png = session.tool("rok_edit", {"do": "shader", "shader": "nature/trees", "volume": [1, 2, 3]})
    assert bad.error and "two opposite corners" in bad.text


# --- rok_deform ---------------------------------------------------


def _slab(session):
    session.run("rok.new", size=tuple([32, 32, 16]), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [8, 8, 2], "high": [23, 23, 7], "colour": 5})


def test_deform_bends_the_whole_model_when_nothing_is_selected_and_undo_takes_it_back(session):
    _slab(session)
    before = session.document.grid.voxels.copy()
    reply, _png = session.tool("rok_deform", {"offsets": {"2,1,1": [4, 0, 0]}})  # the +x face centre
    assert not reply.error, reply.text
    after = session.document.grid.voxels.copy()
    assert "the whole model" in reply.text and (after != 0).sum() > (before != 0).sum()  # a bulge
    assert set(np.unique(after)) - {0} == {5}
    assert session.document.selection.is_empty()  # the whole model: the selection stays empty
    session.tool("rok_scene", {"do": "undo"})
    assert (session.document.grid.voxels == before).all()


def test_deform_bends_only_the_selection_and_selects_the_result(session):
    _slab(session)
    session.tool("rok_select", {"do": "box", "low": [16, 0, 0], "high": [31, 31, 15]})  # the +x half
    before = session.document.grid.voxels.copy()
    reply, _png = session.tool("rok_deform", {"counts": [2, 2, 2], "offsets": {"1,1,1": [0, 6, 0]}})
    assert not reply.error, reply.text
    after = session.document.grid.voxels
    assert "the selection" in reply.text
    assert (after[:16] == before[:16]).all() and (after != before).any()  # the other half is untouched
    assert session.document.selection.count() == int((after[16:] != 0).sum())


def test_deform_answers_a_bad_call_with_how_to_fix_it(session):
    _slab(session)
    for args in ({}, {"offsets": {"3,0,0": [1, 0, 0]}}, {"offsets": {"0,0,0": [1, 0, 0]}, "counts": 9}):
        reply, _png = session.tool("rok_deform", args)
        assert reply.error and ("offsets" in reply.text or "counts" in reply.text), (args, reply.text)
    session.run("rok.new", size=tuple([8, 8, 8]), discard=True)
    reply, _png = session.tool("rok_deform", {"offsets": {"0,0,0": [1, 0, 0]}})
    assert reply.error and "empty" in reply.text


def test_deform_is_a_batch_step(session):
    _slab(session)
    before = session.document.grid.voxels.copy()
    reply, _png = session.tool("rok_batch", {"steps": [
        {"tool": "rok_deform", "args": {"offsets": {"2,1,1": [3, 0, 0]}}},
        {"tool": "rok_deform", "args": {"offsets": {"1,2,1": [0, 3, 0]}}}]})
    assert not reply.error, reply.text
    session.tool("rok_scene", {"do": "undo"})
    assert (session.document.grid.voxels == before).all()  # the whole batch is one step


def test_deform_soft_pulls_the_whole_cage_at_five_points(session):
    _slab(session)
    args = {"counts": 5, "offsets": {"4,2,2": [6, 0, 0]}}  # the +x face centre of a 5-point cage, pulled out
    reply, _png = session.tool("rok_deform", args)
    assert not reply.error, reply.text
    default = session.document.grid.voxels.copy()
    session.tool("rok_scene", {"do": "undo"})
    assert (session.document.grid.voxels != 0).any()
    reply, _png = session.tool("rok_deform", {**args, "bernstein": True})
    assert not reply.error and ", soft cage" in reply.text, reply.text
    soft = session.document.grid.voxels.copy()
    assert (default != 0).sum() > (before_count := 16 * 16 * 6) and (soft != 0).sum() > before_count  # both bulge
    assert not (soft == default).all()  # the other basis reaches further from that point


def test_set_colour_glow_makes_a_light_and_undo_takes_it_back(session):
    session.run("rok.new", size=tuple([16, 16, 16]), discard=True)
    materials = session.agent.doc.scene.materials
    reply, _ = session.tool("rok_edit", {"do": "set_colour", "index": 30, "hex": "#ffd27a", "glow": 0.8})
    lamp = materials.get(30)
    assert not reply.error and lamp.type == "EMIT" and lamp.weight == 0.8 and lamp.flux == 1
    assert tuple(session.agent.doc.scene.palette[30, :3]) == (0xff, 0xd2, 0x7a)
    session.tool("rok_edit", {"do": "set_colour", "index": 30, "glow": 1, "power": 2})  # no hex: the colour stays
    assert materials.get(30).flux == 2 and tuple(session.agent.doc.scene.palette[30, :3]) == (0xff, 0xd2, 0x7a)
    session.tool("rok_edit", {"do": "set_colour", "index": 30, "glow": 0})
    assert materials.get(30).type == "DIFFUSE"
    session.tool("rok_scene", {"do": "undo"})
    assert materials.get(30).type == "EMIT" and materials.get(30).flux == 2


@pytest.mark.parametrize("args, says", [
    ({"glow": 1.5}, "glow is 0 to 1"),
    ({"glow": 1, "power": 7}, "power is a whole number 0 to 4"),
    ({"hex": "#ffffff", "power": 2}, "give glow"),
    ({}, "needs hex"),
])
def test_set_colour_glow_refuses_what_it_cannot_do(session, args, says):
    reply, _ = session.tool("rok_edit", {"do": "set_colour", "index": 30, **args})
    assert reply.error and says in reply.text


def test_clear_deletes_only_the_selection_and_says_how_many(session):
    _slab(session)
    total = _count(session)
    session.tool("rok_select", {"do": "box", "low": [16, 0, 0], "high": [31, 31, 15]})  # the +x half
    selected = session.document.selection.count()
    reply, _png = session.tool("rok_edit", {"do": "clear"})
    assert not reply.error, reply.text
    assert f"Deleted the {selected} selected voxels" in reply.text
    v = session.document.grid.voxels
    assert (v[16:] == 0).all() and _count(session) == total - selected  # the -x half stays
    session.tool("rok_scene", {"do": "undo"})
    assert _count(session) == total


def test_clear_empties_the_whole_model_only_when_nothing_is_selected(session):
    _slab(session)
    total = _count(session)
    session.tool("rok_select", {"do": "none"})
    reply, _png = session.tool("rok_edit", {"do": "clear"})
    assert not reply.error, reply.text
    assert f"Emptied the whole model ({total} voxels)" in reply.text
    assert _count(session) == 0


def test_voxels_set_fill_empty_leaves_what_is_there_and_paint_only_recolours(session):
    _slab(session)  # colour 5 in [8, 8, 2]..[23, 23, 7]
    rows = [[7, 8, 2, 9], [8, 8, 2, 9], [8, 8, 8, 9]]  # empty, filled, empty
    reply, _png = session.tool("rok_voxels_set", {"voxels": rows, "mode": "fill_empty"})
    assert not reply.error, reply.text
    v = session.document.grid.voxels
    assert v[7, 8, 2] == 9 and v[8, 8, 2] == 5 and v[8, 8, 8] == 9
    assert "1 filled cells were left as they are" in reply.text
    reply, _png = session.tool("rok_voxels_set", {"voxels": [[9, 9, 2, 3], [0, 0, 0, 3]], "mode": "paint"})
    assert not reply.error, reply.text
    assert v[9, 9, 2] == 3 and v[0, 0, 0] == 0
    assert "1 empty cells were left as they are" in reply.text
    # Colour 0 on a filled cell: paint does not erase, and the note does not call the cell empty.
    reply, _png = session.tool("rok_voxels_set", {"voxels": [[10, 9, 2, 0]], "mode": "paint"})
    assert not reply.error, reply.text
    assert v[10, 9, 2] == 5
    assert "1 filled cells were given colour 0" in reply.text and "empty cells" not in reply.text


def test_a_solid_curve_is_one_part_face_to_face(session):
    from rok.core import region

    session.run("rok.new", size=tuple([32, 32, 32]), discard=True)
    pts = [[2, 2, 2], [10, 14, 6], [20, 8, 18], [28, 26, 26]]
    reply, _png = session.tool("rok_shape", {"kind": "curve", "points": pts, "colour": 4, "solid": True})
    assert not reply.error, reply.text
    v = session.document.grid.voxels
    assert region.connected_colour(v, (2, 2, 2)).sum() == (v != 0).sum()  # one face-joined part


def test_a_curve_with_brush_sizes_thins_from_root_to_tip(session):
    from rok.core import region

    session.run("rok.new", size=(32, 32, 32), discard=True)
    reply, _png = session.tool("rok_shape", {"kind": "curve", "points": [[16, 16, 4], [16, 16, 24]],
                                                "size": [3, 1], "colour": 4})
    assert not reply.error, reply.text
    assert "brush sizes 3 to 1" in reply.text
    v = session.document.grid.voxels
    across = lambda z: int((v[:, 16, z] != 0).sum())  # noqa: E731
    assert across(4) == 5 and across(24) == 1  # the size 3 brush is 5 across, size 1 one voxel
    assert region.connected_colour(v, (16, 16, 4)).sum() == (v != 0).sum()


def test_a_curve_size_is_one_number_or_one_per_point_and_only_for_a_curve(session):
    session.run("rok.new", size=(16, 16, 16), discard=True)
    reply, _png = session.tool("rok_shape", {"kind": "curve", "points": [[2, 2, 2], [8, 8, 8], [12, 2, 2]],
                                                "size": [3, 1], "colour": 4})
    assert reply.error and "one per point: 3" in reply.text
    reply, _png = session.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [3, 3, 3], "size": 2})
    assert reply.error and "brush size of a curve" in reply.text
    reply, _png = session.tool("rok_shape", {"kind": "curve", "points": [[2, 2, 2], [8, 8, 8]], "size": 2,
                                                "colour": 4})
    assert not reply.error and "brush size 2" in reply.text


def test_turn_tilts_a_shape_round_the_middle_of_its_box(session):
    session.run("rok.new", size=(32, 32, 32), discard=True)
    reply, _png = session.tool("rok_shape", {"kind": "sphere", "low": [12, 12, 4], "high": [19, 19, 27],
                                                "turn": -40, "turn_axis": "x", "colour": 4})
    assert not reply.error, reply.text
    assert "tilted -40 degrees about x" in reply.text
    cells = np.argwhere(session.document.grid.voxels)
    top, bottom = cells[cells[:, 2] >= 20], cells[cells[:, 2] <= 11]
    assert top[:, 1].mean() > bottom[:, 1].mean() + 4  # the top leans back toward +y
    assert np.allclose(cells.mean(axis=0), (15.5, 15.5, 15.5), atol=1.0)  # round the box's middle
    assert np.ptp(cells[:, 0]) == 7  # a turn about x keeps the width in x


def test_describe_gives_commands_and_topics_in_full_and_run_takes_a_short_id(session):
    reply, _png = session.tool("rok_command", {"do": "describe", "id": ["scale", "voxel_art", "rok.repeat"]})
    assert not reply.error, reply.text
    assert "rok.scale (Scale)" in reply.text and "UP: x2 (Repeat every voxel into a 2 x 2 x 2 block)" in reply.text
    assert "voxel_art (topic)" in reply.text and "blockout" in reply.text
    assert "counts: 3 whole numbers, 1 to 8, default (1, 1, 1)" in reply.text
    assert '"id": "rok.repeat", "props": {"counts": [1, 1, 1]}' in reply.text and "Source" not in reply.text
    reply, _png = session.tool("rok_command", {"do": "describe", "id": "twist", "code": True})
    assert "Source (rok/app/model_ops.py, line" in reply.text and "class TwistAction" in reply.text
    reply, _png = session.tool("rok_command", {"do": "describe", "id": ["bevel", "nope"]})
    assert not reply.error and "No command 'nope'" in reply.text
    reply, _png = session.tool("rok_command", {"do": "describe", "id": "nope"})
    assert reply.error
    session.run("rok.new", size=(8, 8, 8), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [1, 0, 0], "colour": 4})
    reply, _png = session.tool("rok_command", {"do": "run", "id": "flip", "props": {"axis": "X"}})
    assert not reply.error and "Ran rok.flip" in reply.text
    assert session.document.grid.voxels[7, 0, 0] == 4


def test_the_catalogue_names_the_editing_commands_and_not_the_mouse_ones():
    from rok.agent_commands import catalogue

    text = catalogue()
    assert "\n  smooth: " in text and "\n  rotate_degrees: " in text and "\n  scene_array: " in text
    for hidden in ("scene_drag", "drop_voxels", "rok.cage", "theme", "nav_cube", "section"):
        assert f"  {hidden.removeprefix('rok.')}: " not in text


def test_a_brush_curve_that_thins_to_one_voxel_stays_one_part(session):
    from rok.core import region

    session.run("rok.new", size=(24, 24, 24), discard=True)
    reply, _png = session.tool("rok_shape", {"kind": "curve", "points": [[4, 4, 4], [10, 9, 12], [18, 12, 19]],
                                                "size": [2, 1, 1], "colour": 4})
    assert not reply.error, reply.text
    v = session.document.grid.voxels
    assert region.connected_colour(v, (4, 4, 4)).sum() == (v != 0).sum()  # the thin tip is face-joined


def test_shade_acts_on_the_selection_and_undo_names_the_step(session):
    session.run("rok.new", size=(16, 16, 16), discard=True)
    session.tool("rok_edit", {"do": "palette", "colours": {"stone": "#8a8478"}, "ramps": {"stone": 3}})
    session.tool("rok_shape", {"kind": "box", "low": [2, 2, 0], "high": [13, 13, 5], "colour": 2})
    session.tool("rok_select", {"do": "box", "low": [0, 0, 0], "high": [7, 15, 15]})
    reply, _png = session.tool("rok_edit", {"do": "shade", "ramp": "stone", "colour": 2})
    assert not reply.error and "of the selection" in reply.text, reply.text
    v = session.document.grid.voxels
    assert v[3, 5, 5] == 1 and v[12, 5, 5] == 2  # the top is lit inside the selection only
    reply, _png = session.tool("rok_scene", {"do": "undo"})
    assert reply.text.startswith("Undid MCP edit shade.") and "The next undo would take back" in reply.text
    reply, _png = session.tool("rok_scene", {"do": "redo"})
    assert reply.text.startswith("Redid MCP edit shade.")


def test_fill_empty_says_what_it_left_and_a_shape_off_the_grid_is_an_error(session):
    _slab(session)  # colour 5 in [8, 8, 2]..[23, 23, 7]
    reply, _png = session.tool("rok_shape", {"kind": "box", "low": [20, 8, 2], "high": [27, 8, 2], "colour": 9,
                                                "mode": "fill_empty"})
    assert not reply.error, reply.text
    assert "Filled 4 empty cells" in reply.text and "4 of its cells were filled already" in reply.text
    reply, _png = session.tool("rok_shape", {"kind": "sphere", "low": [40, 40, 2], "high": [44, 44, 6],
                                                "colour": 9, "mode": "fill_empty"})
    assert reply.error and "lies outside the grid" in reply.text


def test_delete_loose_keeps_the_largest_part(session):
    _slab(session)
    big = _count(session)
    session.tool("rok_voxels_set", {"voxels": [[1, 1, 1, 5], [30, 30, 14, 5], [30, 30, 13, 5]]})
    reply, _png = session.tool("rok_edit", {"do": "delete_loose"})
    assert not reply.error, reply.text
    assert "Deleted 2 loose parts (3 voxels)" in reply.text and _count(session) == big
    reply, _png = session.tool("rok_edit", {"do": "delete_loose"})
    assert "one part already" in reply.text


def test_check_says_how_many_parts_it_leaves_out(session):
    _slab(session)
    session.tool("rok_voxels_set", {"voxels": [[2 * k, 0, 0, 5] for k in range(14)]})  # 14 single cells
    reply, _png = session.tool("rok_check", {})
    assert "and 5 more" in reply.text  # 15 parts, 10 listed


def test_a_batch_reads_its_steps_from_a_json_file(session, tmp_path):
    import json

    session.run("rok.new", size=tuple([16, 16, 16]), discard=True)
    steps = [{"tool": "rok_shape", "args": {"kind": "box", "low": [k, 0, 0], "high": [k, 3, 3], "colour": 2}}
             for k in range(0, 12, 2)]
    file = tmp_path / "steps.json"
    file.write_text(json.dumps({"steps": steps}))
    reply, _png = session.tool("rok_batch", {"path": str(file)})
    assert not reply.error, reply.text
    assert _count(session) == 6 * 16 and reply.text.count("Added a box") == 6
    session.tool("rok_scene", {"do": "undo"})  # still one undo step
    assert _count(session) == 0
    bad, _png = session.tool("rok_batch", {"path": str(tmp_path / "missing.json")})
    assert bad.error and "no file" in bad.text
    both, _png = session.tool("rok_batch", {"path": str(file), "steps": steps})
    assert both.error and "not both" in both.text
