"""The render presets and their camera maths (rok/core/render_presets.py)."""

import math

import numpy as np

from rok.core import render_presets as rp


def test_iso_turn_and_framing():
    assert rp.iso_turn(45, 130) == 90.0 and rp.iso_turn(45, None) == 0.0
    corners = rp.box_corners((0, 0, 0), (9, 9, 9))
    view = rp.direction(45, 35.264)
    loc, width = rp.frame_ortho(corners, view, 1.0, 0.1)
    assert width > 10 * math.sqrt(2) and np.dot(loc - corners.mean(axis=0), view) > 0
    loc, d = rp.frame_perspective(corners, view, rp.fov(35), rp.fov(35), 0.1)
    assert d > 10


def test_the_diorama_moods_float_the_tile_on_a_dark_backdrop():
    for pid in ("DIORAMA_DAY", "DIORAMA_GOLDEN", "DIORAMA_NIGHT"):
        p = rp.preset_by_id(pid)
        assert p.ortho and not p.ground and max(p.world.backdrop) < 0.05
    day, golden = rp.preset_by_id("DIORAMA_DAY").lights[0], rp.preset_by_id("DIORAMA_GOLDEN").lights[0]
    assert golden.elevation < day.elevation and golden.colour[2] < day.colour[2]  # lower and warmer
    night = rp.preset_by_id("DIORAMA_NIGHT")
    assert night.lights[0].power < day.power  # the moon is weak; the scene's own lights do the rest
    assert night.bloom > rp.preset_by_id("DIORAMA_DAY").bloom


def test_the_diorama_day_backdrop_stays_dark_after_its_exposure():
    day = rp.preset_by_id("DIORAMA_DAY")
    assert day.exposure >= 1.0  # muted, dark colours still read in daylight
    assert max(day.world.backdrop) * 2 ** day.exposure < 0.05
