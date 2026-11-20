"""Hand tools for voxel art (rok/core/voxel_draw.py) and their MCP calls."""

from __future__ import annotations

import numpy as np
import pytest

from rok.core import voxel_draw, voxel_views


def test_an_extrude_along_y_reads_back_as_the_front_view():
    rows = [".#.", "###", "#.#"]
    cells, values = voxel_draw.extrude(rows, {"#": 4}, 1, (0, 0, 0), 2)
    v = np.zeros((3, 2, 3), np.uint8)
    v[tuple(cells.T)] = values
    front = voxel_views.text(voxel_views.project(v, "front")[0], {4: "#"})
    assert front == rows and (v[:, 1] == v[:, 0]).all()


def test_an_extrude_along_z_reads_back_as_the_top_view():
    rows = ["##.", "#.."]
    cells, values = voxel_draw.extrude(rows, {"#": 1}, 2, (0, 0, 0), 1)
    v = np.zeros((3, 2, 1), np.uint8)
    v[tuple(cells.T)] = values
    assert voxel_views.text(voxel_views.project(v, "top")[0], {1: "#"}) == rows


def test_clean_drops_l_shaped_corners():
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 1], [2, 0, 2]])
    assert voxel_draw.clean(path).tolist() == [[0, 0, 0], [1, 0, 1], [2, 0, 2]]


def test_a_curve_is_one_cell_wide_without_gaps_or_corners():
    cells = voxel_draw.curve([(0, 0, 0), (6, 0, 6), (14, 0, 8)])
    steps = np.abs(np.diff(cells, axis=0))
    assert steps.max() <= 1 and (steps.sum(axis=1) >= 1).all()
    assert cells[0].tolist() == [0, 0, 0] and cells[-1].tolist() == [14, 0, 8]
    for a, b, c in zip(cells, cells[1:], cells[2:]):  # no L: a and c never touch diagonally through b
        if np.abs(b - a).sum() == 1 and np.abs(c - b).sum() == 1:
            assert np.abs(c - a).max() == 2 or np.abs(c - a).sum() != 2


def test_mirrored_adds_the_other_half_once():
    cells, values = voxel_draw.mirrored(np.array([[1, 0, 0], [2, 0, 0]]), np.array([5, 6]), 0, (4, 1, 1))
    assert sorted(map(tuple, cells.tolist())) == [(1, 0, 0), (2, 0, 0)]  # 1 <-> 2: already both there
    cells, values = voxel_draw.mirrored(np.array([[0, 0, 0]]), np.array([5]), 0, (5, 1, 1))
    assert sorted(map(tuple, cells.tolist())) == [(0, 0, 0), (4, 0, 0)] and values.tolist() == [5, 5]


def test_shade_lights_tops_and_darkens_undersides():
    v = np.zeros((3, 3, 4), np.uint8)
    v[1, 1, 1:3] = 9  # a short column standing in the air
    cells, values = voxel_draw.shade(v, v == 9, [1, 2, 3])
    got = {tuple(c): int(k) for c, k in zip(cells.tolist(), values)}
    assert got[(1, 1, 2)] == 1 and got[(1, 1, 1)] == 3


@pytest.fixture(scope="module")
def session():
    pytest.importorskip("rok_native")
    from rok.headless import Session

    s = Session()
    yield s
    s.close()


def test_the_hand_tools_through_the_mcp(session):
    t = lambda tool, args: session.tool(tool, args)[0]  # noqa: E731
    assert not t("rok_scene", {"do": "new", "discard": True, "size": [6, 8, 16]}).error
    assert not t("rok_edit", {"do": "palette", "colours": {"stone": "#8a8478"}, "ramps": {"stone": 3}}).error
    r = t("rok_shape", {"kind": "extrude", "silhouette": ["##.", "###"], "low": [0, 2, 0], "axis": "y",
                           "depth": 2, "colour": 2, "mirror": "x"})
    assert not r.error and "mirrored across x" in r.text
    v = session.document.grid.voxels
    assert v[0, 2, 1] == 2 and v[5, 2, 1] == 2 and int((v != 0).sum()) == 20  # the halves meet in the middle
    assert not t("rok_look", {}).error  # the blockout reads: colours open
    r = t("rok_edit", {"do": "shade", "ramp": "stone"})
    assert not r.error and "tops stone 1" in r.text
    assert v[0, 2, 1] == 1  # a top face, the lightest step
    r = t("rok_shape", {"kind": "curve", "points": [[0, 4, 0], [2, 5, 4], [4, 6, 6]], "colour": 2})
    assert not r.error and "Drew a curve" in r.text
