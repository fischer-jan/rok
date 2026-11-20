"""The corner gizmo: its layout, its hit test, move and scale of a selection."""

import math

import numpy as np
import pytest

from rok.core import corner_gizmo as cg
from rok.core.select import Selection

CELL_PX = {0: (12.0, -4.0), 1: (-10.0, -5.0), 2: (0.0, 14.0)}


def block():
    voxels = np.zeros((20, 20, 20), dtype=np.uint8)
    voxels[4:8, 5:7, 2:5] = 3  # 4 x 2 x 3
    voxels[4, 5, 2] = 9  # one different colour in the low corner
    sel = Selection(voxels.shape)
    sel.replace(voxels != 0)
    return voxels, sel


def writer(voxels):
    def set_many(cells, values):
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return len(cells)
    return set_many


def test_the_widget_sits_bottom_right_and_hides_an_axis_at_the_eye():
    lay = cg.corner_layout(1000, 800, 2.0, {**CELL_PX, 2: (0.3, 0.2)})
    assert lay.centre == (1000 - cg.MARGIN_RIGHT * 2, cg.MARGIN_BOTTOM * 2)
    assert set(lay.tips) == {0, 1}  # Z points at the eye
    tip = lay.tips[0]
    assert np.hypot(tip[0] - lay.centre[0], tip[1] - lay.centre[1]) == pytest.approx(cg.ARM * 2)


def test_hit_test_finds_tips_boxes_and_the_centre():
    lay = cg.corner_layout(1000, 800, 1.0, CELL_PX)
    assert cg.hit_test(lay, *lay.tips[2]) == "CG_MOVE_Z"
    assert cg.hit_test(lay, *lay.boxes[0]) == "CG_SCALE_X"
    assert cg.hit_test(lay, *lay.centre) == "CG_SCALE_ALL"
    assert cg.hit_test(lay, 10.0, 10.0) is None
    assert cg.hit_test(None, 0, 0) is None


def test_an_arrow_drag_moves_by_whole_cells_along_its_axis():
    voxels, sel = block()
    g = cg.CornerDrag("CG_MOVE_X", voxels, sel, writer(voxels), CELL_PX)
    g.begin_screen(500, 400)
    assert g.extend_screen(500 + 12 * 3, 400 - 4 * 3)  # three cells along X on screen
    assert g.offset() == (3, 0, 0) and g.label() == "+3 X"
    g.finish()
    assert voxels[7, 5, 2] == 9 and voxels[4, 5, 2] == 0
    assert sel.count() == 4 * 2 * 3


def test_a_box_drag_scales_one_axis_around_the_centre():
    voxels, sel = block()
    g = cg.CornerDrag("CG_SCALE_X", voxels, sel, writer(voxels), CELL_PX)
    g.begin_screen(0, 0)
    g.extend_screen(12 * 4, -4 * 4)  # +4 cells: X 4 to 8
    assert g.new_lengths() == (8, 2, 3) and g.label() == "X 4 to 8"
    assert len(g.preview().cells) == 8 * 2 * 3
    g.finish()
    filled = np.argwhere(voxels != 0)
    # X 4..7 became 2..9: two cells more on each side; Y and Z stay put.
    assert filled.min(axis=0).tolist() == [2, 5, 2] and filled.max(axis=0).tolist() == [9, 6, 4]
    assert voxels[2, 5, 2] == 9 and voxels[3, 5, 2] == 9  # the corner colour doubled with the scale
    assert sel.count() == 8 * 2 * 3


def test_the_centre_box_scales_every_axis_and_never_below_one():
    voxels, sel = block()
    g = cg.CornerDrag("CG_SCALE_ALL", voxels, sel, writer(voxels), CELL_PX)
    g.begin_screen(0, 0)
    g.extend_screen(-1000, -1000)
    assert min(g.new_lengths()) >= 1
    g = cg.CornerDrag("CG_SCALE_ALL", voxels, sel, writer(voxels), CELL_PX)
    g.begin_screen(0, 0)
    unit = np.mean([np.hypot(*v) for v in CELL_PX.values()])
    g.extend_screen(4 * unit, 4 * unit)  # four steps: the long side 4 to 8
    assert g.new_lengths() == (8, 4, 6)


def test_three_rings_and_a_ring_hit():
    lay = cg.corner_layout(1000, 800, 1.0, CELL_PX)
    assert set(lay.rings) == {0, 1, 2}
    ring = lay.rings[2]
    x, y = ring[len(ring) // 3]
    assert cg.hit_test(lay, float(x), float(y)) == "CG_ROT_Z"


def test_a_ring_click_turns_once_and_a_sweep_turns_by_quarters():
    voxels, sel = block()
    lay = cg.corner_layout(1000, 800, 1.0, CELL_PX)
    cx, cy = lay.centre
    g = cg.CornerDrag("CG_ROT_Z", voxels, sel, writer(voxels), CELL_PX, centre=lay.centre, ring_sign=lay.ring_sign)
    g.begin_screen(cx + 40, cy)
    assert g.turns() == 1 and g.label() == "90° Z"
    # Half way round the centre: two quarter turns.
    for k in range(1, 19):
        a = math.pi * k / 18
        g.extend_screen(cx + 40 * math.cos(a), cy + 40 * math.sin(a))
    assert g.turns() == 2
    count = int((voxels != 0).sum())
    g.finish()
    assert int((voxels != 0).sum()) == count  # a half turn keeps every voxel
    filled = np.argwhere(voxels != 0)
    extent = filled.max(axis=0) - filled.min(axis=0) + 1
    assert extent.tolist() == [4, 2, 3]
    assert sel.count() == count


def _ring_drag(degrees_list, coarse=False):
    voxels, sel = block()
    lay = cg.corner_layout(1000, 800, 1.0, CELL_PX)
    cx, cy = lay.centre
    g = cg.CornerDrag("CG_ROT_Z", voxels, sel, writer(voxels), CELL_PX, centre=lay.centre, ring_sign=lay.ring_sign)
    g.coarse = coarse
    g.begin_screen(cx + 40, cy)
    for d in degrees_list:
        a = math.radians(d)
        g.extend_screen(cx + 40 * math.cos(a), cy + 40 * math.sin(a))
    return g, voxels, sel


def test_a_ring_drag_turns_the_real_voxels_by_any_angle():
    start = block()[0]
    g, voxels, sel = _ring_drag([10, 20, 30, 37])
    assert g.turns() == 0 and abs(g.turn_degrees()) == 37.0
    assert g.label().startswith(("37°", "-37°")) and g.preview().kind == "LABEL"
    assert not np.array_equal(voxels, start)  # turned during the drag, before the release
    live = voxels.copy()
    g.finish()
    assert np.array_equal(voxels, live) and sel.count() == int((voxels != 0).sum())


def test_turning_the_ring_back_gives_the_voxels_back():
    start = block()[0]
    g, voxels, sel = _ring_drag([20, 45, 20, 0])
    assert g.kind() == "NONE"
    assert g.finish() == 0 and np.array_equal(voxels, start)


def test_ctrl_turns_the_ring_in_15_degree_steps():
    g, _v, _s = _ring_drag([10, 22], coarse=True)
    assert abs(g.turn_degrees()) == 15.0


def test_a_shrink_stays_centred_and_an_odd_change_leans_low():
    voxels, sel = block()
    g = cg.CornerDrag("CG_SCALE_X", voxels, sel, writer(voxels), CELL_PX)
    g.begin_screen(0, 0)
    g.extend_screen(-12 * 1, 4 * 1)  # X 4 to 3
    g.finish()
    filled = np.argwhere(voxels != 0)
    assert filled[:, 0].min() == 4 and filled[:, 0].max() == 6  # (4 - 3) // 2 = 0: kept low, lost the top cell


def test_a_click_on_an_arrow_moves_one_cell_and_a_drag_back_moves_none():
    voxels, sel = block()
    g = cg.CornerDrag("CG_MOVE_Z", voxels, sel, writer(voxels), CELL_PX)
    g.begin_screen(300, 300)
    assert g.offset() == (0, 0, 1) and g.label() == "+1 Z"  # a click, no move yet
    g.extend_screen(301, 300)  # a jitter is still a click
    assert g.offset() == (0, 0, 1)
    g.finish()
    assert voxels[4, 5, 3] == 9 and voxels[4, 5, 2] == 0
    voxels, sel = block()
    g = cg.CornerDrag("CG_MOVE_Z", voxels, sel, writer(voxels), CELL_PX)
    g.begin_screen(300, 300)
    g.extend_screen(300, 300 + 14 * 2)  # dragged two cells up
    g.extend_screen(300, 300)  # and back to the start: a drag to nothing
    assert g.offset() == (0, 0, 0) and g.finish() == 0


def test_no_drag_changes_nothing():
    voxels, sel = block()
    before = voxels.copy()
    g = cg.CornerDrag("CG_SCALE_Y", voxels, sel, writer(voxels), CELL_PX)
    g.begin_screen(10, 10)
    assert not g.extend_screen(11, 10)
    assert g.finish() == 0 and np.array_equal(voxels, before)
