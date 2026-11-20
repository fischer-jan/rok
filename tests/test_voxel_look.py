"""Voxel look (rok/core/voxel_look.py): a posed part as real voxels."""

import numpy as np
import pytest

from rok.core import anim, grid_ops, voxel_look


def bar() -> np.ndarray:
    v = np.zeros((8, 2, 2), np.uint8)
    v[:, :, :] = 3
    v[0] = 5  # one end has its own colour
    return v


def test_no_turn_keeps_the_voxels_and_moves_in_whole_cells():
    v = bar()
    out, origin = voxel_look.posed(v, np.eye(3), (4, 1, 1), (2.4, -1.6, 0))
    assert out is v and origin == (2, -2, 0)


def test_a_quarter_turn_about_the_middle_is_exact():
    v = bar()
    out, origin = voxel_look.posed(v, anim.matrix_from_euler((0, 0, 90)), (4, 1, 1), (0, 0, 0))
    expected, _s = grid_ops.apply_rotation(v, np.rint(anim.matrix_from_euler((0, 0, 90))).astype(int))
    assert np.array_equal(out, expected)
    assert origin == (3, -3, 0)  # the turned bar stands across the old middle
    assert np.count_nonzero(out) == np.count_nonzero(v)


def test_a_free_angle_resamples_from_the_rest_and_invents_no_colour():
    v = bar()
    out, _origin = voxel_look.posed(v, anim.matrix_from_euler((0, 0, 37)), (4, 1, 1), (0, 0, 0))
    assert set(np.unique(out)) <= {0, 3, 5}
    assert 5 in out and abs(np.count_nonzero(out) - np.count_nonzero(v)) <= 12
    # A full turn is the rest again.
    again, origin = voxel_look.posed(v, anim.matrix_from_euler((0, 0, 360)), (4, 1, 1), (0, 0, 0))
    assert np.array_equal(again, v) and origin == (0, 0, 0)


def test_double_size_doubles_each_side():
    v = np.ones((2, 2, 2), np.uint8)
    out, origin = voxel_look.posed(v, np.eye(3) * 2, (1, 1, 1), (0, 0, 0))
    assert out.shape == (4, 4, 4) and out.all() and origin == (-1, -1, -1)


def test_a_pose_too_big_raises():
    v = np.ones((200, 1, 1), np.uint8)
    with pytest.raises(ValueError):
        voxel_look.posed(v, np.eye(3) * 2, (100, 0.5, 0.5), (0, 0, 0))


def test_an_empty_border_costs_nothing_and_changes_nothing():
    v = bar()
    big = np.zeros((20, 20, 20), np.uint8)
    big[6:14, 9:11, 9:11] = v
    L = anim.matrix_from_euler((0, 0, 37))
    small, o_small = voxel_look.posed(v, L, (4, 1, 1), (0, 0, 0))
    out, origin = voxel_look.posed(big, L, (10, 10, 10), (0, 0, 0))
    assert np.array_equal(out, small)
    assert tuple(a - b for a, b in zip(origin, o_small)) == (6, 9, 9)


def test_a_quarter_turn_about_a_half_cell_pivot_stays_exact_and_snaps():
    """the .vox frames turn a part about its middle, which sits on half cells
    for an odd side: one voxel turned 90 degrees stayed one voxel, not two."""
    v = np.zeros((3, 4, 5), np.uint8)
    v[2, 0, 1] = 9
    pivot = (1.5, 2.0, 2.5)
    out, origin = voxel_look.posed(v, anim.matrix_from_euler((0, 0, 270)), pivot, (0, 0, 0))
    assert np.count_nonzero(out) == 1  # the filled box is cropped first: one cell
    assert all(isinstance(o, int) for o in origin)
    # The turned voxel sits within half a cell of where the free turn would put it.
    cell = np.argwhere(out != 0)[0] + np.array(origin) + 0.5
    free = (np.array([2.5, 0.5, 1.5]) - pivot) @ anim.matrix_from_euler((0, 0, 270)).T + pivot
    assert np.abs(cell - free).max() <= 0.5 + 1e-6
