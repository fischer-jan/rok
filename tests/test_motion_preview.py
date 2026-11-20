"""The preset cards' moving pictures (rok/core/motion_preview.py)."""

import math

import numpy as np
import pytest

from rok.core import anim, motion_preview as mp


@pytest.mark.parametrize("name", anim.PRESETS)
def test_every_preset_has_a_picture_that_fits_the_card(name):
    for t in (0.0, 0.3, 0.77, 1.5, mp.REST_T[name]):
        shapes = mp.frame(name, t)
        assert shapes
        x0, y0, x1, y1 = mp.bounds(shapes)
        assert -1.3 <= x0 and x1 <= 1.3 and -1.3 <= y0 and y1 <= 1.3, (name, t)
        for s in shapes:
            assert len(s.points) >= 3 and len(s.rgba) == 4
            assert all(0.0 <= c <= 1.0 for c in s.rgba)


def test_every_preset_has_a_rest_pose():
    assert set(mp.REST_T) == set(anim.PRESETS)


@pytest.mark.parametrize("name", ["SPIN", "SWING", "BOB", "SHAKE", "GLOW", "BLINK", "FLOW"])
def test_the_picture_moves(name):
    a = mp.frame(name, 0.05)
    b = mp.frame(name, 0.55)
    assert a != b


def test_a_cube_shows_three_faces_to_the_viewer():
    faces = mp.box((0, 0, 0), (1, 1, 1), (1, 1, 1))
    assert len(faces) == 3
    turned = mp.box((0, 0, 0), (1, 1, 1), (1, 1, 1), mp.rotation((0, 0, 1), math.radians(45)))
    assert len(turned) == 2  # corner-on: the top and one side
    assert all(len(shape.points) == 4 for _d, shape in faces)


def test_the_top_face_is_the_brightest():
    faces = [shape for _d, shape in mp.box((0, 0, 0), (1, 1, 1), (1, 1, 1))]
    tops = [s for s in faces if max(p[1] for p in s.points) == max(p[1] for f in faces for p in f.points)]
    assert max(sum(s.rgba[:3]) for s in faces) == max(sum(s.rgba[:3]) for s in tops)


def test_rotation_is_a_proper_turn():
    r = mp.rotation((1, 2, 3), 0.7)
    assert np.allclose(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)
