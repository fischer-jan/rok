import numpy as np
import pytest

from rok.core.grid_ops import MAX_SIZE
from rok.core.stretch import (
    HANDLES,
    Handle,
    cells_along,
    crop_or_grow,
    face_centre,
    facing_handles,
    handle_from_id,
    new_position,
    new_size,
    stretch,
)


def test_six_handles_with_ids_that_round_trip():
    assert len({h.id for h in HANDLES}) == 6
    assert all(handle_from_id(h.id) == h for h in HANDLES)
    assert handle_from_id("nope") is None and handle_from_id(None) is None


def test_a_handle_sits_in_the_middle_of_its_face():
    assert face_centre((10, 0, 0), (4, 6, 8), Handle(0, 1)) == (14, 3.0, 4.0)
    assert face_centre((10, 0, 0), (4, 6, 8), Handle(2, -1)) == (12.0, 3.0, 0)


def test_only_the_faces_toward_the_view_show_handles():
    shown = facing_handles((-0.6, -0.6, -0.5))  # looking from +x +y +z down toward the origin
    assert {h.id for h in shown} == {"SH_X+", "SH_Y+", "SH_Z+"}


def test_a_drag_counts_cells_along_the_face_axis_on_screen():
    assert cells_along((100, 100), (130, 110), (10, 0)) == 3
    assert cells_along((100, 100), (70, 100), (10, 0)) == -3  # dragged inward
    assert cells_along((0, 0), (5, 5), (0, 0)) == 0  # an axis seen end-on


def test_the_new_size_stays_in_range_and_uniform_scales_every_axis():
    assert new_size((10, 20, 30), Handle(0, 1), 5, False) == (15, 20, 30)
    assert new_size((10, 20, 30), Handle(0, 1), -50, False) == (1, 20, 30)
    assert new_size((10, 20, 30), Handle(2, 1), 1000, False)[2] == MAX_SIZE
    assert new_size((10, 20, 30), Handle(0, 1), 10, True) == (20, 40, 60)


def test_the_opposite_face_stays_put():
    # Dragging the high face keeps the position; the low face moves the start.
    assert new_position((5, 5, 5), (10, 10, 10), (14, 10, 10), Handle(0, 1), False) == (5, 5, 5)
    assert new_position((5, 5, 5), (10, 10, 10), (14, 10, 10), Handle(0, -1), False) == (1, 5, 5)
    # Uniform: the other axes keep their centre.
    assert new_position((5, 5, 5), (10, 10, 10), (20, 20, 20), Handle(2, 1), True) == (0, 0, 5)


def test_stretching_repeats_or_drops_cells_evenly():
    row = np.arange(1, 5, dtype=np.uint8).reshape(4, 1, 1)
    assert stretch(row, (8, 1, 1)).ravel().tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert stretch(row, (2, 1, 1)).ravel().tolist() == [2, 4]  # the cell under each new middle
    wall = np.ones((12, 2, 6), dtype=np.uint8)
    assert stretch(wall, (3, 2, 6)).shape == (3, 2, 6)  # a wide wall squeezed to a pillar


def test_crop_and_grow_keep_the_voxels_their_size():
    row = np.arange(1, 5, dtype=np.uint8).reshape(4, 1, 1)
    assert crop_or_grow(row, (6, 1, 1), Handle(0, 1), False).ravel().tolist() == [1, 2, 3, 4, 0, 0]
    assert crop_or_grow(row, (6, 1, 1), Handle(0, -1), False).ravel().tolist() == [0, 0, 1, 2, 3, 4]
    assert crop_or_grow(row, (2, 1, 1), Handle(0, 1), False).ravel().tolist() == [1, 2]
    assert crop_or_grow(row, (2, 1, 1), Handle(0, -1), False).ravel().tolist() == [3, 4]


@pytest.mark.parametrize("handle", HANDLES)
def test_crop_then_grow_back_on_the_same_face_loses_only_the_cut(handle):
    rng = np.random.default_rng(1)
    v = rng.integers(0, 3, size=(5, 6, 7)).astype(np.uint8)
    size = list(v.shape)
    size[handle.axis] -= 2
    small = crop_or_grow(v, size, handle, False)
    back = crop_or_grow(small, v.shape, handle, False)
    assert back.shape == v.shape and np.count_nonzero(back) <= np.count_nonzero(v)


def test_a_stretched_group_keeps_touching_models_touching():
    from rok.core.stretch import group_parts

    # Two 4-cell models side by side in x, a 2-cell one on top of the first.
    parts = [((10, 0, 0), (4, 2, 2)), ((14, 0, 0), (4, 2, 2)), ((10, 0, 2), (2, 2, 2))]
    out = group_parts(parts, (10, 0, 0), (8, 2, 4), (12, 2, 4), (10, 0, 0))
    (a_pos, a_size), (b_pos, b_size), (c_pos, c_size) = out
    assert a_pos[0] + a_size[0] == b_pos[0]  # still flush
    assert a_size[0] + b_size[0] == 12  # together they fill the new box
    assert c_pos[0] == a_pos[0] and c_size[0] == 3  # aligned faces stay aligned
    assert all(p[2] == q[2] for (p, _), (q, _) in zip(out, [(pp, s) for pp, s in parts]))  # z untouched


def test_a_squeezed_group_never_loses_a_model():
    from rok.core.stretch import group_parts

    parts = [((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (9, 1, 1))]
    out = group_parts(parts, (0, 0, 0), (10, 1, 1), (2, 1, 1), (0, 0, 0))
    assert all(size[0] >= 1 for _, size in out)


def _sphere_shell(n=40, radius=17.0):
    """A hollow ball one voxel thick, like a thin dish: every slant there is."""
    c = (n - 1) / 2.0
    x, y, z = np.indices((n, n, n))
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    return np.where(np.abs(d - radius) <= 0.5, 3, 0).astype(np.uint8)


def _outside(voxels):
    """The empty cells reachable from the grid border (6-connected)."""
    empty = voxels == 0
    seen = np.zeros_like(empty)
    for face in (np.s_[0], np.s_[-1]):
        for axis in range(3):
            index = [slice(None)] * 3
            index[axis] = face
            seen[tuple(index)] |= empty[tuple(index)]
    while True:
        grown = seen.copy()
        for axis in range(3):
            grown |= np.roll(seen, 1, axis) | np.roll(seen, -1, axis)
        grown &= empty
        if np.array_equal(grown, seen):
            return seen
        seen = grown


@pytest.mark.parametrize("new", [33, 27, 21, 14])
def test_squeezing_a_thin_shell_leaves_no_holes(new):
    shell = _sphere_shell()
    centre = (np.asarray(shell.shape) - 1) // 2
    assert not _outside(shell)[tuple(centre)]  # closed before
    out = stretch(shell, (new, new, new))
    assert out.shape == (new, new, new)
    assert not _outside(out)[tuple((np.asarray(out.shape) - 1) // 2)]  # and still closed
    assert set(np.unique(out)) == {0, 3}


def test_squeezing_takes_the_colour_nearest_the_middle():
    row = np.zeros((6, 1, 1), dtype=np.uint8)
    row[:, 0, 0] = (1, 2, 0, 4, 5, 0)
    out = stretch(row, (3, 1, 1))
    assert list(out[:, 0, 0]) == [2, 4, 5]  # middles 1, 3 and 5; cell 5 is empty, so its neighbour gives 5


def test_squeezing_fills_a_cell_whose_middle_is_empty():
    row = np.zeros((4, 1, 1), dtype=np.uint8)
    row[0, 0, 0] = 7  # only the first of the pair is filled
    assert stretch(row, (2, 1, 1))[0, 0, 0] == 7
