"""Flat views drawn cell by cell (rok/core/voxel_views.py) and the craft checks
(rok/core/craft.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rok.core import craft, voxel_views
from rok.core.vox_read import read_scene

FIXTURES = Path(__file__).parent / "fixtures"


def _ramp():
    return read_scene(FIXTURES / "agent" / "jaggy_ramp.vox").models[0].voxels


def test_each_view_faces_the_way_the_quad_picture_shows_it():
    v = np.zeros((4, 5, 6), np.uint8)
    v[3, 0, 0] = 1  # x high, y low, z low
    v[0, 4, 0] = 2  # y high
    v[0, 0, 5] = 3  # z high
    front, _ = voxel_views.project(v, "front")
    assert front.shape == (6, 4) and front[5, 3] == 1 and front[0, 0] == 3  # x to the right, z up
    right, _ = voxel_views.project(v, "right")
    assert right.shape == (6, 5) and right[5, 4] == 2  # y to the right
    top, _ = voxel_views.project(v, "top")
    assert top.shape == (5, 4) and top[0, 0] == 2 and top[4, 3] == 1  # y high in the first row


def test_a_view_shows_the_nearest_cell_and_its_depth():
    v = np.zeros((1, 3, 1), np.uint8)
    v[0, 1, 0] = 4
    v[0, 2, 0] = 5
    colours, depth = voxel_views.project(v, "front")
    assert colours[0, 0] == 4 and depth[0, 0] == 1


def test_the_top_view_as_text_reads_back_through_voxels_set_layers():
    from rok.agent import Agent

    v = np.zeros((6, 4, 1), np.uint8)
    v[1:5, 0, 0] = 7
    v[2:4, 1:3, 0] = 9
    legend = voxel_views.legend_for(v)
    rows = voxel_views.text(voxel_views.project(v, "top")[0], legend)
    cells, values = Agent._layers([rows], (0, 0, 0), {ch: c for c, ch in legend.items()})
    back = np.zeros_like(v)
    back[tuple(cells.T)] = values
    assert np.array_equal(back, v)


def test_scaled_views_are_crisp_and_icons_keep_small_views_whole():
    colours = np.array([[1, 0], [0, 2]])
    pal = np.zeros((256, 4), np.uint8)
    pal[1] = (255, 0, 0, 255)
    pal[2] = (0, 0, 255, 255)
    img = voxel_views.rgb(colours, np.zeros_like(colours), pal)
    big = voxel_views.scaled(img, 4)
    assert big.shape == (8, 8, 3) and (big[:4, :4] == (255, 0, 0)).all()
    assert np.array_equal(voxel_views.icon(img, colours, 16), img)
    shrunk = voxel_views.icon(voxel_views.scaled(img, 8), voxel_views.scaled(colours, 8), 4)
    assert shrunk.shape == (4, 4, 3)


def test_the_craft_check_finds_the_faults_of_the_ramp():
    rep = craft.check(_ramp())
    assert [(j.side, j.edge) for j in rep.jags] == [("front", "top")]
    assert rep.jags[0].runs[:5] == [2, 1, 3, 1, 2]
    assert any(t["side"] == "front" and t["cells"] == 6 for t in rep.thin)
    assert len(rep.parts) == 3 and rep.parts[1]["colours"] == {3: 8} and rep.parts[2]["colours"] == {2: 6}
    line = craft.summary(rep)
    assert "2 floating parts" in line and "jagged slopes front 1" in line and "thin line" in line


def test_an_even_slope_and_a_solid_block_pass():
    v = np.zeros((12, 2, 8), np.uint8)
    for k in range(6):
        v[2 * k:2 * k + 2, :, :k + 1] = 1  # runs 2, 2, 2, 2, 2, 2
    rep = craft.check(v)
    assert rep.jags == [] and rep.thin == [] and len(rep.parts) == 1
    assert craft.jagged([1, 2, 1, 2, 1]) is False
    assert craft.jagged([1, 1, 3, 1, 2]) is True


def test_survival_counts_what_a_small_icon_still_shows():
    s = np.zeros((16, 16), bool)
    s[4:12, 4:12] = True
    assert craft.survival(s, 4) == 100.0
    s[0:16, 15] = True  # a one-cell line vanishes at a quarter
    assert craft.survival(s, 4) < 85.0
