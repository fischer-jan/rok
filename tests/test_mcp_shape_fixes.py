"""Three fixes to the MCP tools: a line between two named cells in any
direction, a shape's run along a named axis (a half disc), and a meadow on the
tops of one colour, in solid patches if asked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app.object import document

FIXTURES = Path(__file__).parent / "fixtures" / "agent"
GROUNDS = FIXTURES / "two_grounds.vox"  # 32 x 32 x 16: colour 30 on x 0..15, colour 12 on x 16..31, z 0..2


@pytest.fixture(scope="module")
def session():
    pytest.importorskip("rok_native")  # Session needs the native module
    from rok.headless import Session

    s = Session()
    yield s
    s.close()


def _text(session, tool: str, args: dict) -> str:
    reply, _picture = session.tool(tool, args)
    return reply.text


def _empty(session) -> None:
    session.run("rok.new", size=tuple([32, 32, 32]), discard=True)


# --- a line in any direction ---------------------------------------------------------


def test_a_line_runs_from_start_to_end_toward_low_x(session):
    _empty(session)
    text = _text(session, "rok_shape", {"kind": "line", "start": [20, 5, 0], "end": [12, 5, 15], "colour": 9})
    assert "from [20, 5, 0] to [12, 5, 15]" in text
    cells = np.argwhere(document.grid.voxels == 9)
    assert len(cells) == 16  # one cell per step of the longest axis
    by_z = cells[np.argsort(cells[:, 2])]
    assert by_z[0].tolist() == [20, 5, 0] and by_z[-1].tolist() == [12, 5, 15]
    assert np.all(np.diff(by_z[:, 0]) <= 0)  # x never rises while z does


def test_low_and_high_still_draw_a_line_in_the_order_given(session):
    _empty(session)
    _text(session, "rok_shape", {"kind": "line", "low": [20, 5, 0], "high": [12, 5, 15], "colour": 9})
    cells = np.argwhere(document.grid.voxels == 9)
    assert {tuple(c) for c in cells} >= {(20, 5, 0), (12, 5, 15)}


@pytest.mark.parametrize("args", [
    {"kind": "line", "start": [0, 0, 0]},
    {"kind": "line", "start": [0, 0, 0], "end": [3, 3, 3], "low": [0, 0, 0]},
    {"kind": "box", "start": [0, 0, 0], "end": [3, 3, 3]},
])
def test_start_and_end_are_checked(session, args):
    _empty(session)
    reply, _ = session.tool("rok_shape", args)
    assert reply.error


# --- a run along a named axis -------------------------------------------------------


def test_a_half_cylinder_along_its_thin_side_is_a_half_disc(session):
    _empty(session)
    _text(session, "rok_shape", {"kind": "half_cylinder", "low": [0, 10, 0], "high": [20, 10, 10],
                                  "along": "y", "colour": 9})
    rows = [line for line in (FIXTURES / "half_disc_xz.txt").read_text().splitlines() if not line.startswith("//")]
    v = document.grid.voxels
    got = ["".join("#" if c else "." for c in (v[:21, 10, z] != 0)) for z in range(10, -1, -1)]
    assert got == rows
    assert not (v[:, :10] != 0).any() and not (v[:, 11:] != 0).any()


def test_without_along_a_thin_half_cylinder_runs_along_its_longer_side(session):
    """The old rule stays: a barrel along x, as wide as the box is deep (one cell): a plate."""
    _empty(session)
    _text(session, "rok_shape", {"kind": "half_cylinder", "low": [0, 10, 0], "high": [20, 10, 10], "colour": 9})
    assert int((document.grid.voxels != 0).sum()) == 21 * 11


@pytest.mark.parametrize("args, says", [
    ({"kind": "box", "along": "y"}, "a box has no run"),
    ({"kind": "roof", "along": "z"}, "along must lie across axis z"),
])
def test_along_is_checked(session, args, says):
    _empty(session)
    reply, _ = session.tool("rok_shape", {"low": [0, 0, 0], "high": [5, 5, 5], **args})
    assert reply.error and says in reply.text


def test_a_roof_runs_along_the_named_axis(session):
    _empty(session)
    _text(session, "rok_shape", {"kind": "roof", "low": [0, 0, 0], "high": [11, 5, 5], "along": "y", "colour": 9})
    ridge = document.grid.voxels[:12, :6, 5] != 0  # the top layer
    assert ridge.any(axis=0).all()  # it runs the whole way along y
    assert ridge.any(axis=1).sum() <= 2  # and is one or two cells wide in x


# --- the meadow ---------------------------------------------------------------------


def _grounds(session) -> None:
    _text(session, "rok_scene", {"do": "open", "path": str(GROUNDS), "discard": True})


def test_only_colours_alone_runs_again_with_room_above_the_tops(session):
    _grounds(session)
    before = document.grid.voxels.copy()
    reply, _ = session.tool("rok_edit", {"do": "shader", "shader": "nature/meadow", "only_colours": [12]})
    assert not reply.error and "Ran nature/meadow" in reply.text
    assert 'only_on ["top"] and room 4 added' in reply.text
    v = document.grid.voxels
    assert np.array_equal(v[:, :, :3], before[:, :, :3])
    assert not (v[:16, :, 3:] != 0).any() and (v[16:, :, 3:] != 0).any()


def test_a_run_that_still_changes_nothing_keeps_its_warning(session):
    _grounds(session)
    reply, _ = session.tool("rok_edit", {"do": "shader", "shader": "nature/meadow", "only_colours": [200]})
    assert "changed nothing" in reply.text and "room" in reply.text


def test_a_meadow_on_the_tops_of_one_colour_stays_on_that_colour(session):
    _grounds(session)
    before = document.grid.voxels.copy()
    _text(session, "rok_edit", {"do": "shader", "shader": "nature/meadow", "only_colours": [12],
                                 "only_on": ["top"], "room": 6})
    v = document.grid.voxels
    assert np.array_equal(v[:, :, :3], before[:, :, :3])  # the ground stays
    assert not (v[:16, :, 3:] != 0).any()  # nothing over colour 30
    assert (v[16:, :, 3:] != 0).sum() > 300  # grass over colour 12


def _grass_columns(session, clumps: int) -> np.ndarray:
    _grounds(session)
    _text(session, "rok_edit", {"do": "shader", "shader": "nature/meadow",
                                 "args": {"Grass": 50, "Flowers": 0, "Clumps": clumps}})
    return (document.grid.voxels[:, :, 3:] != 0).any(axis=2)


def _alike(g: np.ndarray) -> float:
    """How often a column's neighbour has grass the same as it."""
    return float((g[1:] == g[:-1]).mean() + (g[:, 1:] == g[:, :-1]).mean()) / 2.0


def test_clumps_make_solid_patches_instead_of_single_blades(session):
    loose, solid = _grass_columns(session, 0), _grass_columns(session, 100)
    assert 0.2 < loose.mean() < 0.8 and 0.2 < solid.mean() < 0.9
    assert _alike(loose) < 0.65 < 0.85 < _alike(solid)
    heights = (document.grid.voxels[:, :, 3:] != 0).sum(axis=2)
    assert len(set(heights[solid].tolist())) == 1  # one even height in the patches
