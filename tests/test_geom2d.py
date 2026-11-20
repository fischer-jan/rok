"""The 2D helpers the UI modules share."""

import math

import numpy as np

from rok.core.geom2d import fit_glyph, in_rect, loop_lines, polyline_distance, rect_tris


def test_in_rect_takes_low_edges_and_leaves_high_edges():
    rect = (10, 20, 30, 40)
    assert in_rect(rect, 10, 20)
    assert in_rect(rect, 39.9, 59.9)
    assert not in_rect(rect, 40, 30)
    assert not in_rect(rect, 20, 60)
    assert not in_rect(rect, 9.9, 30)


def test_in_rect_of_none_holds_nothing():
    assert in_rect(None, 0, 0) is False


def test_rect_tris_of_one_rect_and_of_a_batch():
    one = rect_tris((10, 20, 30, 40))
    assert one.shape == (6, 2) and one.dtype == np.float32
    assert one[0].tolist() == [10, 20] and one[2].tolist() == [40, 60]
    batch = rect_tris(np.array([(10, 20, 30, 40), (0, 0, 1, 1)]))
    assert batch.shape == (12, 2)
    assert np.array_equal(batch[:6], one)


def test_loop_lines_closes_the_polygon():
    lines = loop_lines([(0, 0), (4, 0), (4, 4)])
    assert lines.shape == (6, 2) and lines.dtype == np.float32
    assert lines[-1].tolist() == [0, 0] and lines[-2].tolist() == [4, 4]


def test_fit_glyph_centres_a_square_on_half_pixels():
    unit = np.array([[0.0, 0.0], [1.0, 1.0]])
    out = fit_glyph(unit, (0, 0, 40, 20))
    assert out.tolist() == [[10.5, 0.5], [30.5, 20.5]]
    assert np.all(out % 1.0 == 0.5)
    assert fit_glyph(unit, (0, 0, 40, 20), inset=5.0).tolist() == [[15.5, 5.5], [25.5, 15.5]]
    assert fit_glyph(unit, (0, 0, 40, 20), inset=10.0).shape == (0, 2)


def test_polyline_distance_to_a_closed_square():
    square = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    assert polyline_distance(square, 5.0, 0.0) == 0.0
    assert polyline_distance(square, 5.0, -3.0) == 3.0
    assert polyline_distance(square, 13.0, 14.0) == 5.0
    assert polyline_distance(square, 0.0, 5.0) == 0.0  # the closing edge counts


def test_polyline_distance_is_inf_for_nothing_to_measure():
    assert polyline_distance(np.zeros((1, 2)), 0.0, 0.0) == math.inf
    assert polyline_distance(np.array([(0.0, 0.0), (np.nan, 1.0)]), 0.0, 0.0) == math.inf
