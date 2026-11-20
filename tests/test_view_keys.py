"""WASD + QE view keys: the pure part (rok/core/view_keys.py)."""

import pytest

from rok.core import view_keys as vk


def test_the_six_keys_and_their_directions():
    assert set(vk.VIEW_KEYS) == set("WASDQE")
    assert vk.VIEW_KEYS["A"] == (vk.ORBIT, -1) and vk.VIEW_KEYS["D"] == (vk.ORBIT, 1)
    assert vk.VIEW_KEYS["W"] == (vk.TILT, 1) and vk.VIEW_KEYS["S"] == (vk.TILT, -1)
    assert vk.VIEW_KEYS["Q"] == (vk.ZOOM, -1) and vk.VIEW_KEYS["E"] == (vk.ZOOM, 1)


def test_holding_ramps_up_then_moves_at_full_speed():
    assert vk.travelled(0.0) == 0.0
    early = vk.degrees_between(vk.ORBIT, 0.0, 0.05)
    late = vk.degrees_between(vk.ORBIT, 1.0, 1.05)
    assert 0 < early < late
    assert late == pytest.approx(vk.ORBIT_DEG_PER_S * 0.05)
    # Travel adds up however the time is cut.
    whole = vk.degrees_between(vk.TILT, 0.0, 0.7)
    parts = sum(vk.degrees_between(vk.TILT, a / 10, (a + 1) / 10) for a in range(7))
    assert whole == pytest.approx(parts)


def test_a_tap_always_moves_one_whole_step():
    for kind in (vk.ORBIT, vk.TILT):
        for held in (0.0, 0.05, 0.15):
            total = vk.degrees_between(kind, 0.0, held) + vk.tap_rest(kind, held)
            assert total == pytest.approx(vk.TAP_DEG)
    for held in (0.0, 0.1):
        total = vk.zoom_between(0.0, held) * vk.tap_rest(vk.ZOOM, held)
        assert total == pytest.approx(vk.TAP_ZOOM)
    assert vk.tap_rest(vk.ORBIT, 5.0) == 0.0  # a long hold owes nothing


def test_tilt_stops_short_of_the_poles():
    assert vk.clamp_tilt(0.0, 10.0) == 10.0
    assert vk.clamp_tilt(80.0, 10.0) == pytest.approx(vk.MAX_ELEVATION_DEG - 80.0)
    assert vk.clamp_tilt(-84.0, -10.0) == pytest.approx(-vk.MAX_ELEVATION_DEG + 84.0)
    assert vk.clamp_distance(0.0) == vk.MIN_DISTANCE
