"""The scene's Align, Spread, Array and Carve maths (rok/core/arrange.py)."""

import numpy as np

from rok.core import arrange
from rok.core.arrange import CENTRE, MAX, MIN

BOXES = [((10, 0, 0), (4, 4, 4)), ((0, 5, 2), (2, 2, 2)), ((30, 9, 7), (6, 1, 1))]


def test_align_lines_the_boxes_up_on_the_lead():
    assert [p[0] for p in arrange.aligned(BOXES, 0, 0, MIN)] == [10, 10, 10]
    assert [p[0] for p in arrange.aligned(BOXES, 0, 0, MAX)] == [10, 12, 8]  # high sides at 14
    assert [p[0] for p in arrange.aligned(BOXES, 0, 0, CENTRE)] == [10, 11, 9]  # centres at 12
    assert [p[1:] for p in arrange.aligned(BOXES, 0, 0, MIN)] == [(0, 0), (5, 2), (9, 7)]  # other axes stay


def test_spread_leaves_equal_gaps_and_keeps_the_outer_boxes():
    boxes = [((0, 0, 0), (2, 1, 1)), ((3, 0, 0), (2, 1, 1)), ((20, 0, 0), (4, 1, 1)), ((5, 0, 0), (2, 1, 1))]
    out = arrange.spread(boxes, 0)
    assert out[0][0] == 0 and out[2][0] == 20  # the outer two stay
    # 2..20 holds two boxes of 2: 14 spare cells in three gaps: 5, 5, 4
    assert (out[1][0], out[3][0]) == (7, 14)


def test_array_offsets_step_by_size_and_gap():
    assert arrange.array_offsets((4, 2, 2), 3, 0, 1, 1) == [(5, 0, 0), (10, 0, 0)]
    assert arrange.array_offsets((4, 2, 2), 3, 1, -1, 0) == [(0, -2, 0), (0, -4, 0)]
    assert arrange.array_offsets((4, 2, 2), 2, 0, 1, -10) == [(1, 0, 0)]  # overlaps, never on the original
    assert len(arrange.array_offsets((1, 1, 1), 999, 0, 1, 0)) == arrange.ARRAY_MAX - 1


def test_carve_empties_shared_filled_cells_but_not_masked_ones():
    target = np.ones((4, 4, 4), np.uint8)
    cutter = np.zeros((2, 2, 2), np.uint8)
    cutter[0, 0, 0] = cutter[1, 1, 1] = 5
    mask = arrange.carve_mask(target, (0, 0, 0), cutter, (2, 2, 2))
    assert sorted(map(tuple, np.argwhere(mask))) == [(2, 2, 2), (3, 3, 3)]
    protected = np.zeros((4, 4, 4), bool)
    protected[3, 3, 3] = True
    mask = arrange.carve_mask(target, (0, 0, 0), cutter, (2, 2, 2), protected)
    assert sorted(map(tuple, np.argwhere(mask))) == [(2, 2, 2)]
    assert arrange.carve_mask(target, (0, 0, 0), cutter, (10, 0, 0)) is None  # no overlap
