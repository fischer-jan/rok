"""MCP changes from a test run that built a scene through the tools only: a
model set on a cell of another, `model` on rok_scene, grow from only_colours,
face-joined lines, and rok_check's word on parts joined at a corner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.core import region

FIXTURES = Path(__file__).parent / "fixtures" / "agent"
LAND = FIXTURES / "landscape.vox"
CRATE = FIXTURES / "crate.vox"
GROUNDS = FIXTURES / "two_grounds.vox"  # colour 30 on x 0..15, colour 12 on x 16..31, z 0..2


@pytest.fixture(scope="module")
def session():
    pytest.importorskip("rok_native")  # Session needs the native module
    from rok.headless import Session

    s = Session()
    s.agent.feedback = "text"
    yield s
    s.close()


def _call(session, tool: str, args: dict):
    reply, _png = session.tool(tool, args)
    return reply


def _ok(session, tool: str, args: dict) -> str:
    reply = _call(session, tool, args)
    assert not reply.error, reply.text
    return reply.text


# --- setting a model on a cell -----------------------------------------------------


def _land_and_crate(session):
    _ok(session, "rok_scene", {"do": "open", "path": str(LAND), "discard": True})
    _ok(session, "rok_scene", {"do": "import", "path": str(CRATE)})
    scene = session.document.scene
    return scene.models[0], session.document.model


def test_move_to_sets_the_model_on_a_cell_of_the_target(session):
    land, crate = _land_and_crate(session)
    land_name = land.display_name(0)
    text = _ok(session, "rok_scene", {"do": "move", "to": [42, 12, 2], "target": land_name})
    assert "on cell [42, 12, 2]" in text
    cells = np.argwhere(crate.grid.voxels != 0) + np.asarray(crate.position) - np.asarray(land.position)
    assert int(cells[:, 2].min()) == 2
    middle = (cells[:, :2].min(axis=0) + cells[:, :2].max(axis=0) + 1) / 2.0
    assert np.all(np.abs(middle - (42.5, 12.5)) <= 0.5)


def test_move_to_counts_in_the_first_other_model_by_default(session):
    land, crate = _land_and_crate(session)
    _ok(session, "rok_scene", {"do": "move", "to": [10, 10, 5]})
    cells = np.argwhere(crate.grid.voxels != 0) + np.asarray(crate.position) - np.asarray(land.position)
    assert int(cells[:, 2].min()) == 5


@pytest.mark.parametrize("args, says", [
    ({"do": "move", "to": [1, 1, 1], "offset": [1, 0, 0]}, "not both"),
    ({"do": "move"}, "not both and not neither"),
    ({"do": "move", "to": [1, 1, 1], "target": "Nowhere"}, "no model named"),
    ({"do": "rename", "name": "X", "to": [1, 1, 1]}, "to and target are for do=move"),
    ({"do": "list", "model": "Anything"}, "acts on the whole scene"),
])
def test_move_arguments_are_checked(session, args, says):
    _land_and_crate(session)
    reply = _call(session, "rok_scene", args)
    assert reply.error and says in reply.text


def test_scene_actions_take_a_model(session):
    land, crate = _land_and_crate(session)
    before = tuple(int(p) for p in land.position)
    _ok(session, "rok_scene", {"do": "move", "model": land.display_name(0), "offset": [0, 3, 0]})
    assert tuple(int(p) for p in land.position) == (before[0], before[1] + 3, before[2])
    assert session.document.model is land  # it became the active model
    _ok(session, "rok_scene", {"do": "rename", "model": "Crate" if crate.name == "Crate" else crate.display_name(1),
                                "name": "Box"})
    assert crate.name == "Box"


# --- grow from only_colours ---------------------------------------------------------


def test_grow_without_a_selection_names_the_call_to_make(session):
    _ok(session, "rok_scene", {"do": "open", "path": str(GROUNDS), "discard": True})
    reply = _call(session, "rok_edit", {"do": "shader", "shader": "nature/meadow", "grow": True})
    assert reply.error and 'rok_select {"do": "colour"' in reply.text and "only_colours" in reply.text


def test_grow_starts_from_the_voxels_only_colours_names(session):
    _ok(session, "rok_scene", {"do": "open", "path": str(GROUNDS), "discard": True})
    before = session.document.grid.voxels.copy()
    text = _ok(session, "rok_edit", {"do": "shader", "shader": "nature/meadow", "grow": True,
                                      "only_colours": [12], "args": {"Flowers": 0}})
    assert "grows from the voxels of only_colours" in text
    v = session.document.grid.voxels
    assert np.array_equal(v[:, :, :3], before[:, :, :3])
    assert (v[16:, :, 3:] != 0).sum() > 100  # grass on the green half
    assert session.document.selection.is_empty()  # no selection was made or left behind


def test_grow_with_filters_that_match_nothing_says_so(session):
    _ok(session, "rok_scene", {"do": "open", "path": str(GROUNDS), "discard": True})
    reply = _call(session, "rok_edit", {"do": "shader", "shader": "nature/meadow", "grow": True,
                                         "only_colours": [200]})
    assert reply.error and "no voxels match" in reply.text


# --- face-joined lines and the check ------------------------------------------------


def _empty(session) -> None:
    session.run("rok.new", size=tuple([32, 32, 32]), discard=True)


def test_a_solid_line_joins_its_cells_face_to_face(session):
    _empty(session)
    _ok(session, "rok_shape", {"kind": "line", "start": [24, 16, 0], "end": [10, 10, 20], "colour": 8, "solid": True})
    mask = session.document.grid.voxels != 0
    assert len(region.components(mask)) == 1
    cells = np.argwhere(mask)
    assert {(24, 16, 0), (10, 10, 20)} <= {tuple(c) for c in cells.tolist()}


def test_check_says_when_parts_touch_only_at_an_edge_or_a_corner(session):
    _empty(session)
    _ok(session, "rok_shape", {"kind": "line", "start": [20, 16, 0], "end": [10, 16, 10], "colour": 8})
    # A crown that meets the line's top cell [10, 16, 10] at one corner only.
    _ok(session, "rok_shape", {"kind": "box", "low": [6, 17, 11], "high": [9, 20, 14], "colour": 9})
    reply = _call(session, "rok_check", {})
    assert "separate parts (face to face)" in reply.text
    assert "touch the rest only at an edge or a corner" in reply.text and "solid=true" in reply.text
    assert reply.data["parts_touching"] == 1


def test_check_after_a_solid_line_finds_one_part(session):
    _empty(session)
    _ok(session, "rok_shape", {"kind": "line", "start": [20, 16, 0], "end": [10, 16, 10], "colour": 8, "solid": True})
    _ok(session, "rok_shape", {"kind": "sphere", "low": [6, 12, 10], "high": [12, 20, 15], "colour": 9})
    assert "One connected part" in _call(session, "rok_check", {}).text


# --- saving and the meadow's colours ------------------------------------------------


def test_save_lists_the_models_it_wrote(session, tmp_path):
    _land_and_crate(session)
    text = _ok(session, "rok_scene", {"do": "save", "path": str(tmp_path / "both.vox")})
    scene = session.document.scene
    assert f"with {len(scene.models)} model(s)" in text
    for i, m in enumerate(scene.models):
        assert f"{m.display_name(i)} (grid {'x'.join(str(int(s)) for s in m.grid.size)}, " \
               f"{int((m.grid.voxels != 0).sum())} voxels)" in text


def _tops(session, clumps: int) -> np.ndarray:
    """The colour on top of each grass column of the green half (0: bare)."""
    _ok(session, "rok_scene", {"do": "open", "path": str(GROUNDS), "discard": True})
    _ok(session, "rok_edit", {"do": "shader", "shader": "nature/meadow", "only_colours": [12], "only_on": ["top"],
                               "room": 4, "args": {"Grass": 60, "Flowers": 0, "Clumps": clumps}})
    v = session.document.grid.voxels[16:, :, 3:]
    filled = v != 0
    top = np.where(filled.any(axis=2), filled.shape[2] - 1 - np.argmax(filled[:, :, ::-1], axis=2), -1)
    return np.where(top >= 0, np.take_along_axis(v, np.maximum(top, 0)[..., None], axis=2)[..., 0], 0)


def _speckle(tops: np.ndarray) -> float:
    """How often two neighbouring grass tops differ in colour."""
    pairs = [(tops[1:], tops[:-1]), (tops[:, 1:], tops[:, :-1])]
    both = sum(int(((a > 0) & (b > 0)).sum()) for a, b in pairs)
    differ = sum(int(((a > 0) & (b > 0) & (a != b)).sum()) for a, b in pairs)
    return differ / max(both, 1)


def test_clumped_meadow_tops_are_one_colour_not_speckled(session):
    loose, clumped = _speckle(_tops(session, 0)), _speckle(_tops(session, 100))
    assert clumped < 0.03 < loose


# --- shader colours and keeping grass off a pad ---------------------------------------


def test_only_the_grounds_colour_keeps_a_meadow_off_a_pad(session):
    _ok(session, "rok_scene", {"do": "open", "path": str(GROUNDS), "discard": True})
    _ok(session, "rok_shape", {"kind": "box", "low": [20, 10, 0], "high": [27, 20, 2], "colour": 40, "mode": "paint"})
    _ok(session, "rok_edit", {"do": "shader", "shader": "nature/meadow", "only_colours": [12], "only_on": ["top"],
                               "room": 4, "args": {"Flowers": 0}})
    v = session.document.grid.voxels
    assert not (v[20:28, 10:21, 3:] != 0).any()
    assert (v[16:, :, 3:] != 0).sum() > 300
