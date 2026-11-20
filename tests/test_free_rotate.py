"""Turning a model by any angle."""

import numpy as np
import pytest

from rok.core import free_rotate as fr
from rok.core import grid_ops


def _block(shape=(9, 5, 7), colour=3):
    v = np.zeros(shape, np.uint8)
    v[1:-1, 1:-1, 1:-1] = colour
    v[2, 2, 2] = 7  # one odd voxel
    return v


def test_snap_wraps_and_rounds_to_tenths():
    assert fr.snap(12.34) == 12.3
    assert fr.snap(190) == -170.0
    assert fr.snap(-180) == 180.0
    assert fr.snap(360) == 0.0
    assert fr.snap(0.04) == 0.0 and fr.snap(0.06) == 0.1


def test_quarter_turns_only_for_exact_multiples():
    assert fr.quarter_turns(90) == 1 and fr.quarter_turns(-90) == 3 and fr.quarter_turns(180) == 2
    assert fr.quarter_turns(0) == 0 and fr.quarter_turns(359.96) == 0
    assert fr.quarter_turns(89.9) is None and fr.quarter_turns(45) is None


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("degrees", [0, 90, 180, -90, 270])
def test_multiples_of_90_are_the_exact_quarter_turns(axis, degrees):
    v = _block()
    k = fr.quarter_turns(degrees)
    expected = v
    for _ in range(k):
        expected = grid_ops.rotate90(expected, axis)
    assert np.array_equal(fr.rotate_voxels(v, axis, degrees), expected)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_free_turn_agrees_with_the_matrix_at_90(axis):
    # The matrix is the quarter turn grid_ops uses: same handedness.
    assert np.allclose(fr.matrix(axis, 90), grid_ops._QUARTER[axis])


def test_a_turn_keeps_the_colours_and_makes_no_holes():
    v = np.zeros((21, 21, 3), np.uint8)
    v[:] = 5
    out = fr.rotate_voxels(v, 2, 30.0)
    assert set(np.unique(out)) <= {0, 5}
    # Inside the turned square every cell is filled: the middle has no holes.
    c = out.shape[0] // 2
    assert (out[c - 5:c + 6, c - 5:c + 6, :] == 5).all()


def test_the_volume_stays_about_the_same():
    v = np.zeros((30, 30, 4), np.uint8)
    v[5:25, 5:25, :] = 2
    for degrees in (7.3, 22.5, 45.0, 60.1):
        out = fr.rotate_voxels(v, 2, degrees)
        assert abs(int((out > 0).sum()) - int((v > 0).sum())) < 0.06 * (v > 0).sum(), degrees


def test_the_new_grid_holds_the_turned_box():
    assert fr.turned_size((10, 10, 4), 2, 45.0) == (15, 15, 4)
    assert fr.turned_size((10, 4, 6), 2, 90.0) == (4, 10, 6)
    assert fr.turned_size((10, 4, 6), 0, 30.0)[0] == 10  # the turning axis keeps its size


def test_turning_back_gets_close():
    v = np.zeros((25, 25, 3), np.uint8)
    v[4:21, 8:17, :] = 4
    there = fr.rotate_voxels(v, 2, 33.3)
    back = fr.rotate_voxels(there, 2, -33.3)
    a = (v > 0)
    b = np.zeros_like(a)
    dx = (back.shape[0] - a.shape[0]) // 2
    dy = (back.shape[1] - a.shape[1]) // 2
    b[...] = (back[dx:dx + a.shape[0], dy:dy + a.shape[1], :a.shape[2]] > 0)
    assert (a ^ b).sum() < 0.12 * a.sum()


def test_too_big_a_turn_says_so():
    with pytest.raises(ValueError):
        fr.rotate_voxels(np.ones((250, 250, 2), np.uint8), 2, 45.0)


def test_an_empty_model_stays_empty():
    assert not fr.rotate_voxels(np.zeros((6, 6, 6), np.uint8), 1, 12.3).any()


def test_one_model_turns_about_its_own_centre():
    v = _block((11, 11, 3))
    [(out, pos)] = fr.rotate_models([(v, (10, 20, 0))], 2, 45.0)
    centre_before = np.array([10, 20, 0]) + (np.array(v.shape) - 1) / 2
    centre_after = np.array(pos) + (np.array(out.shape) - 1) / 2
    assert np.abs(centre_before - centre_after).max() <= 0.5


def test_a_group_turns_about_its_common_centre():
    a = np.full((3, 3, 3), 1, np.uint8)
    items = [(a, (0, 0, 0)), (a, (10, 0, 0))]  # centres at x = 1 and 11; pivot 6
    out = fr.rotate_models(items, 2, 90.0)
    # A quarter turn about Z: the pair, along x, now lies along y.
    (_v1, p1), (_v2, p2) = out
    assert abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) == 10


@pytest.mark.parametrize("degrees", [12.3, -47.9, 90.0])
def test_the_preview_matrix_moves_a_centre_where_the_bake_puts_it(degrees):
    # The live preview turns the meshes with a 4x4; the bake resamples. Both
    # must put each model's centre at the same place (to the bake's rounding).
    a = _block((7, 5, 3))
    b = _block((4, 4, 4))
    items = [(a, (0, 0, 0)), (b, (12, 3, 1))]
    mats = fr.preview_matrices(items, 2, degrees)
    baked = fr.rotate_models(items, 2, degrees)
    for (v, p), m, (out, q) in zip(items, mats, baked):
        centre_mesh = np.asarray(p) + np.asarray(v.shape) / 2.0  # mesh units: cell i fills [i, i + 1)
        moved = m @ np.append(centre_mesh - np.asarray(p), 1.0)
        moved = moved[:3] + np.asarray(p)
        baked_centre = np.asarray(q) + np.asarray(out.shape) / 2.0
        assert np.abs(moved - baked_centre).max() <= 0.5 + 1e-9


def test_one_model_previews_about_its_own_centre():
    v = _block((11, 11, 3))
    [m] = fr.preview_matrices([(v, (10, 20, 0))], 2, 45.0)
    centre = np.array([5.5, 5.5, 1.5, 1.0])
    assert np.allclose(m @ centre, centre)
    assert np.allclose(m[:3, :3], fr.matrix(2, 45.0))
