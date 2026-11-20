"""Steady Hand: the brush trails the mouse on a string (rok/core/steady_hand.py)."""

import math

from rok.core import steady_hand


def test_length_zero_follows_the_mouse():
    assert steady_hand.follow((10.0, 10.0), (50.0, 3.0), 0) == (50.0, 3.0)


def test_the_brush_stays_while_the_mouse_is_inside_the_string():
    assert steady_hand.follow((10.0, 10.0), (14.0, 13.0), 20) == (10.0, 10.0)


def test_the_brush_trails_the_mouse_by_the_string_length():
    x, y = steady_hand.follow((0.0, 0.0), (100.0, 0.0), 30)
    assert (round(x, 6), round(y, 6)) == (70.0, 0.0)


def test_the_pull_goes_along_the_line_to_the_mouse():
    brush = steady_hand.follow((0.0, 0.0), (60.0, 80.0), 50)
    assert math.isclose(math.hypot(brush[0] - 60.0, brush[1] - 80.0), 50.0)
    assert math.isclose(brush[0] / brush[1], 60.0 / 80.0)


def test_a_shaking_hand_draws_nothing():
    brush = (100.0, 100.0)
    for dx, dy in [(3, 2), (-4, 1), (2, -5), (-3, -3)]:
        brush = steady_hand.follow(brush, (100.0 + dx, 100.0 + dy), 20)
    assert brush == (100.0, 100.0)
