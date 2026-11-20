"""The cage's chip row: where it sits, where it moves, what a click finds."""

import numpy as np
import pytest

from rok.core import cage_chips as cc

BOX = (400.0, 300.0, 600.0, 500.0)  # minx, miny, maxx, maxy of a projected cage; y grows upward


def test_the_row_is_points_three_locks_soft_reset_then_done_above_the_box():
    lay = cc.chip_layout(BOX, 1000, 800, 1.0)
    names = [n for n, _r in lay.chips]
    assert names == ["CAGE_POINTS", "CAGE_LOCK_0", "CAGE_LOCK_1", "CAGE_LOCK_2", "CAGE_SOFT", "CAGE_RESET", "CAGE_DONE"]
    assert lay.above
    x, y, w, h = lay.rect
    assert y == BOX[3] + cc.CHIP_GAP
    assert abs((x + w / 2) - 500) <= 1  # centred over the box
    rects = [r for _n, r in lay.chips]
    assert all(r[1] == y and r[3] == h for r in rects)
    xs = [r[0] for r in rects]
    assert xs == sorted(xs)
    assert all(rects[i][0] + rects[i][2] + cc.CHIP_GAP == rects[i + 1][0] for i in range(len(rects) - 1))


def test_the_row_moves_below_the_box_when_the_top_would_clip():
    box = (400.0, 700.0, 600.0, 780.0)
    lay = cc.chip_layout(box, 1000, 800, 1.0)
    assert not lay.above
    x, y, w, h = lay.rect
    assert y + h <= box[1]  # under the box
    assert y >= 0


def test_the_row_stays_inside_the_region_at_the_edges_and_scales():
    for box in ((0.0, 300.0, 40.0, 340.0), (960.0, 300.0, 1000.0, 340.0)):
        lay = cc.chip_layout(box, 1000, 800, 2.0)
        x, y, w, h = lay.rect
        assert x >= 0 and x + w <= 1000
        assert h == 60
    narrow = cc.chip_layout(BOX, 200, 800, 1.0)  # a region narrower than the row
    assert narrow.rect[0] >= 0


def test_the_row_stands_beyond_the_selection_pill_strip():
    strip = (350, 508, 300, 40)  # the pills' row above the box: x, y, w, h
    lay = cc.chip_layout(BOX, 1000, 800, 1.0, avoid=strip)
    assert lay.above and lay.rect[1] >= strip[1] + strip[3]
    low = (350, 252, 300, 40)  # the strip had to go below
    box = (400.0, 700.0, 600.0, 780.0)
    lay = cc.chip_layout(box, 1000, 800, 1.0, avoid=(350, 652, 300, 40))
    assert not lay.above and lay.rect[1] + lay.rect[3] <= 652
    assert low  # (kept for the picture of the case above)


def test_chip_hit_finds_each_chip_and_misses_the_gaps_and_the_rest():
    lay = cc.chip_layout(BOX, 1000, 800, 1.0)
    for name, (x, y, w, h) in lay.chips:
        assert cc.chip_hit(lay, x + w / 2, y + h / 2) == name
        assert cc.chip_contains(lay, x + w / 2, y + h / 2)
    reset = dict(lay.chips)["CAGE_RESET"]
    gap_x = reset[0] - cc.CHIP_GAP / 2
    assert cc.chip_hit(lay, gap_x, reset[1] + 5) is None and cc.chip_contains(lay, gap_x, reset[1] + 5)
    assert cc.chip_hit(lay, 5, 5) is None and not cc.chip_contains(lay, 5, 5)
    assert cc.chip_hit(None, 1, 1) is None and not cc.chip_contains(None, 1, 1)


def test_ids_round_trip_and_nothing_is_laid_out_without_a_box():
    assert [cc.parse_lock(cc.lock_id(a)) for a in range(3)] == [0, 1, 2]
    assert cc.parse_lock("CAGE_LOCK_3") is None and cc.parse_lock("CAGE_RESET") is None and cc.parse_lock(None) is None
    assert cc.is_chip("CAGE_DONE") and cc.is_chip("CAGE_POINTS") and cc.is_chip("CAGE_LOCK_2") and not cc.is_chip("PAINT")
    assert cc.chip_layout(None, 1000, 800) is None
    assert cc.chip_layout((np.nan, 0, 1, 1), 1000, 800) is None
    assert all(n in cc.CHIP_TITLES and n in cc.CHIP_TIPS for n, _r in cc.chip_layout(BOX, 1000, 800).chips)
    assert [cc.CHIP_TITLES[cc.lock_id(a)] for a in range(3)] == ["Lock X", "Lock Y", "Lock Z"]


def test_the_soft_chip_is_laid_out_and_found():
    lay = cc.chip_layout(BOX, 1000, 800, 1.0)
    rect = dict(lay.chips)[cc.SOFT]
    assert cc.is_chip(cc.SOFT) and cc.chip_hit(lay, rect[0] + 3, rect[1] + 3) == cc.SOFT
    assert cc.SOFT in cc.CHIP_TITLES and cc.SOFT in cc.CHIP_TIPS
