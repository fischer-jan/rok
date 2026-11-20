"""Lathe, Twist, Remove Slice (rok/core/reshape.py) and heightmaps (rok/core/heightmap.py)."""

import numpy as np
import pytest

from rok.core.heightmap import colours, heightmap_voxels, heights, shrink
from rok.core.palette import default_palette
from rok.core.reshape import lathe, remove_slices, slice_axis, twist


def test_a_lathe_turns_a_profile_into_a_round_solid():
    v = np.zeros((12, 12, 6), np.uint8)
    v[6:10, 6, 0:3] = 5  # a flat profile, 4 out from the centre line, 3 high
    v[6:8, 6, 3:6] = 7  # narrower on top
    out = lathe(v)
    # Round: the same rings on every side of the centre line.
    assert np.array_equal(out, np.flip(out, axis=0)) and np.array_equal(out, np.flip(out, axis=1))
    assert np.array_equal(out, out.transpose(1, 0, 2))
    assert set(np.unique(out[:, :, :3])) == {0, 5} and set(np.unique(out[:, :, 3:])) == {0, 7}
    assert out[:, :, 3:].astype(bool).sum() < out[:, :, :3].astype(bool).sum()
    assert (lathe(out) == out).all()  # a round solid turns into itself


def test_a_lathe_around_x_keeps_the_x_layers():
    v = np.zeros((4, 10, 10), np.uint8)
    v[2, 5:8, 5] = 3
    out = lathe(v, axis=0)
    assert out[2].any() and not out[0].any() and not out[3].any()


def test_a_twist_turns_the_top_layer_and_keeps_the_bottom():
    v = np.zeros((9, 9, 5), np.uint8)
    v[4:9, 4, :] = 2  # an arm along +x in every layer
    out = twist(v, 90.0)
    assert np.array_equal(out[:, :, 0], v[:, :, 0])
    top = np.argwhere(out[:, :, 4])
    assert (top[:, 0] == 4).all() and top[:, 1].max() == 8  # the arm points along +y at the top
    assert np.array_equal(twist(v, 0.0), v)


def test_remove_slice_takes_the_selected_layers_out_and_closes_the_gap():
    v = np.zeros((4, 4, 6), np.uint8)
    for z in range(6):
        v[:, :, z] = z + 1
    mask = np.zeros(v.shape, bool)
    mask[1, 1, 2:4] = True  # thinnest along x and y (1), but Z comes first only when as thin
    assert slice_axis(mask) == (0, 1, 1)
    mask[:, :, 2:4] = True  # a whole 2-layer slab: thinnest along z
    axis, first, last = slice_axis(mask)
    assert (axis, first, last) == (2, 2, 3)
    out = remove_slices(v, axis, first, last)
    assert out.shape == (4, 4, 4) and list(out[0, 0]) == [1, 2, 5, 6]
    assert slice_axis(np.zeros((2, 2, 2), bool)) is None
    with pytest.raises(ValueError):
        remove_slices(v, 2, 0, 5)


def test_a_heightmap_rises_with_brightness_and_reads_from_above():
    img = np.zeros((2, 3, 4), np.uint8)
    img[..., 3] = 255
    img[0, 0, :3] = 255  # top-left white: the far left column is the tallest
    img[1, 2, 3] = 0  # bottom-right transparent: no column
    cols = heights(img, 8)
    assert cols.shape == (3, 2)
    assert cols[0, 1] == 8 and cols[1, 1] == 1 and cols[2, 0] == 0
    vox = heightmap_voxels(img, 8, 4)
    assert vox.shape == (3, 2, 8) and vox[0, 1].tolist() == [4] * 8 and not vox[2, 0].any()


def test_a_big_picture_shrinks_to_fit_a_model_and_keeps_its_colours():
    img = np.zeros((600, 300, 4), np.uint8)
    img[..., 0] = 255
    img[..., 3] = 255
    small = shrink(img)
    assert small.shape[:2] == (200, 100) and (small[..., 0] == 255).all()
    pal = default_palette()
    slots = colours(small[:2, :2], pal)
    red = tuple(int(c) for c in pal[slots[0, 0]][:3])
    assert red[0] > 200 and red[1] < 80 and red[2] < 80
