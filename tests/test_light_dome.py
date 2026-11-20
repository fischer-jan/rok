"""The light dome's maths (rok/core/light_dome.py)."""

import math

import pytest

from rok.core import light_dome as kit


def test_a_light_goes_to_the_disc_and_back_from_any_view():
    dome = kit.dome_in((0, 0, 352, 176))
    for camera in (-120.0, 0.0, 45.0, 170.0):
        for azimuth, elevation in ((10.0, 30.0), (-150.0, 60.0), (95.0, 5.0)):
            x, y = kit.point(dome, azimuth, elevation, camera)
            back_az, back_el = kit.light_at(dome, x, y, camera)
            assert math.remainder(back_az - azimuth, 360.0) == pytest.approx(0.0, abs=1e-6)
            assert back_el == pytest.approx(elevation)


def test_the_camera_stands_at_the_bottom_and_left_is_left():
    dome = kit.dome_in((0, 0, 352, 176))
    camera = 30.0
    x, y = kit.point(dome, camera, 0.0, camera)  # a light behind the camera
    assert x == pytest.approx(dome.cx) and y == pytest.approx(dome.cy - dome.radius)
    x, _y = kit.point(dome, camera + 90.0, 0.0, camera)  # 90 degrees counter-clockwise: the view's right
    assert x == pytest.approx(dome.cx + dome.radius)
    x, y = kit.point(dome, 0.0, 90.0, camera)  # straight above: the middle
    assert (x, y) == pytest.approx((dome.cx, dome.cy))


def test_below_the_horizon_sits_on_the_rim_and_the_ring_is_apart_from_the_disc():
    dome = kit.dome_in((0, 0, 352, 176))
    x, y = kit.point(dome, 0.0, -10.0, 0.0)
    assert math.hypot(x - dome.cx, y - dome.cy) == pytest.approx(dome.radius)
    mid = (dome.ring_inner + dome.ring_outer) / 2
    assert kit.on_ring(dome, dome.cx + mid, dome.cy)
    assert not kit.on_ring(dome, dome.cx + dome.radius * 0.5, dome.cy)


def test_the_nearest_handle_prefers_the_one_on_top():
    assert kit.nearest([(0, 0), (1, 0)], 0.5, 0, 5) == 1
    assert kit.nearest([(0, 0)], 50, 50, 5) is None
