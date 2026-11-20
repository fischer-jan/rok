"""Flip, rotate90, resize, rotation bytes, whole-grid fills, loop, scale, repeat,
diagonal and model transforms on hand grids and the fixture models."""

import numpy as np
import pytest

from rok.core.gizmo import snap_pivot
from rok.core.grid_ops import (
    IDENTITY_BYTE,
    MAX_SIZE,
    apply_rotation,
    diagonal,
    fill_all,
    fill_volume,
    fit_bounds,
    flip,
    flip_model,
    inverse,
    loop,
    mirror_half,
    repeat,
    resize,
    rotate90,
    rotate_model,
    rotation_from_byte,
    rotation_to_byte,
    scale_down,
    scale_up,
    settle_cells,
)

from helpers import FIXTURE_META, model_voxels

MODELS = sorted(FIXTURE_META)


def _histogram(voxels: np.ndarray) -> np.ndarray:
    return np.bincount(voxels.ravel(), minlength=256)


# --- identities on the models ------------------------------------------------


@pytest.mark.parametrize("stem", MODELS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_twice_is_identity(stem, axis):
    voxels = model_voxels(stem)
    out = flip(flip(voxels, axis), axis)
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert out.shape == voxels.shape
    assert np.array_equal(out, voxels)


@pytest.mark.parametrize("stem", MODELS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_once_changes_only_when_asymmetric(stem, axis):
    voxels = model_voxels(stem)
    out = flip(voxels, axis)
    assert out.shape == voxels.shape
    assert np.array_equal(out, np.flip(voxels, axis=axis))
    assert np.array_equal(_histogram(out), _histogram(voxels))


@pytest.mark.parametrize("stem", MODELS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotate_four_times_is_identity(stem, axis):
    voxels = model_voxels(stem)
    out = voxels
    for _ in range(4):
        out = rotate90(out, axis)
        assert out.dtype == np.uint8
        assert out.flags.c_contiguous
    assert out.shape == voxels.shape
    assert np.array_equal(out, voxels)


def test_rotate_bird_about_z():
    voxels = model_voxels("bird")
    assert voxels.shape == (80, 68, 22)
    out = rotate90(voxels, 2)
    assert out.shape == (68, 80, 22)
    assert np.count_nonzero(out) == np.count_nonzero(voxels) == 9016
    assert np.array_equal(_histogram(out), _histogram(voxels))


# --- single voxel mappings ---------------------------------------------------


def test_rotate_z_documented_example():
    voxels = np.zeros((3, 2, 1), dtype=np.uint8)
    voxels[2, 0, 0] = 7
    out = rotate90(voxels, 2)
    assert out.shape == (2, 3, 1)
    assert out[1, 2, 0] == 7
    assert np.count_nonzero(out) == 1


def test_rotate_x_maps_y_z():
    # (y, z) -> (sz - 1 - z, y) on a (1, 3, 2) grid: (0, 2, 0) -> (0, 1, 2).
    sy, sz = 3, 2
    voxels = np.zeros((1, sy, sz), dtype=np.uint8)
    y, z = 2, 0
    voxels[0, y, z] = 9
    out = rotate90(voxels, 0)
    assert out.shape == (1, sz, sy)
    assert out[0, sz - 1 - z, y] == 9
    assert np.count_nonzero(out) == 1


def test_rotate_y_maps_z_x():
    # (z, x) -> (sx - 1 - x, z) on a (3, 1, 2) grid: x=2, z=0 -> z=0, x=0.
    sx, sz = 3, 2
    voxels = np.zeros((sx, 1, sz), dtype=np.uint8)
    x, z = 2, 0
    voxels[x, 0, z] = 5
    out = rotate90(voxels, 1)
    assert out.shape == (sz, 1, sx)
    new_z = sx - 1 - x
    new_x = z
    assert out[new_x, 0, new_z] == 5
    assert np.count_nonzero(out) == 1


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotate_keeps_every_colour(axis):
    rng = np.random.default_rng(3)
    voxels = rng.integers(0, 256, size=(5, 3, 4), dtype=np.uint8)
    out = rotate90(voxels, axis)
    assert out.size == voxels.size
    assert np.array_equal(_histogram(out), _histogram(voxels))


def test_flip_and_rotate_return_copies():
    # An axis of length 1 keeps the view contiguous; the result must still be a copy.
    voxels = np.ones((1, 3, 3), dtype=np.uint8)
    flipped = flip(voxels, 0)
    flipped[0, 0, 0] = 9
    assert voxels[0, 0, 0] == 1
    turned = rotate90(voxels, 0)
    turned[0, 0, 0] = 9
    assert voxels[0, 0, 0] == 1
    assert not np.shares_memory(flip(voxels, 0), voxels)
    assert not np.shares_memory(rotate90(voxels, 0), voxels)


def test_bad_axis_raises():
    voxels = np.zeros((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        flip(voxels, 3)
    with pytest.raises(ValueError):
        rotate90(voxels, -1)


# --- resize ------------------------------------------------------------------


@pytest.mark.parametrize("stem", MODELS)
def test_resize_grow_keeps_every_voxel_in_place(stem):
    voxels = model_voxels(stem)
    sx, sy, sz = voxels.shape
    out = resize(voxels, (sx + 5, sy + 1, sz + 10))
    assert out.shape == (sx + 5, sy + 1, sz + 10)
    assert out.dtype == np.uint8
    assert np.array_equal(out[:sx, :sy, :sz], voxels)
    assert np.count_nonzero(out) == np.count_nonzero(voxels)
    xs, ys, zs = np.nonzero(voxels)
    assert np.array_equal(out[xs, ys, zs], voxels[xs, ys, zs])


@pytest.mark.parametrize("stem", MODELS)
def test_resize_shrink_keeps_the_overlap(stem):
    voxels = model_voxels(stem)
    sx, sy, sz = voxels.shape
    new = (sx // 2, sy - 3, sz)
    out = resize(voxels, new)
    assert out.shape == new
    assert np.array_equal(out, voxels[: new[0], : new[1], : new[2]])


def test_resize_mixed_grow_and_shrink():
    voxels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4) + 1
    out = resize(voxels, (4, 2, 4))
    assert out.shape == (4, 2, 4)
    assert np.array_equal(out[:2, :2, :], voxels[:, :2, :])
    assert not out[2:].any()


def test_resize_same_size_is_a_copy():
    voxels = model_voxels("robot")
    out = resize(voxels, voxels.shape)
    assert np.array_equal(out, voxels)
    assert out is not voxels
    out[0, 0, 0] = 200
    assert voxels[0, 0, 0] != 200


@pytest.mark.parametrize(
    "size", [(0, 4, 4), (4, 257, 4), (4, 4, -1), (300, 300, 300), (4, 4)]
)
def test_resize_out_of_range_raises(size):
    voxels = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize(voxels, size)


def test_resize_limits_are_inclusive():
    voxels = np.zeros((4, 4, 4), dtype=np.uint8)
    voxels[0, 0, 0] = 1
    assert resize(voxels, (1, 1, 1)).shape == (1, 1, 1)
    assert resize(voxels, (1, 1, 1))[0, 0, 0] == 1
    assert resize(voxels, (256, 1, 1)).shape == (256, 1, 1)


# --- rotation bytes ----------------------------------------------------------

VALID_BYTES = [
    b for b in range(128) if (b & 3) != 3 and ((b >> 2) & 3) != 3 and (b & 3) != ((b >> 2) & 3)
]
INVALID_BYTES = [b for b in range(256) if b not in VALID_BYTES]

# One quarter turn, counter-clockwise seen from the positive axis: (x, y) -> (-y, x) about z.
QUARTER = {
    0: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    1: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    2: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
}


def _world_cells(voxels, position) -> dict[tuple[int, int, int], int]:
    cells = np.argwhere(voxels != 0)
    return {
        tuple(int(v) for v in c + np.asarray(position)): int(voxels[tuple(c)]) for c in cells
    }


def test_valid_bytes_are_48_and_identity_is_4():
    assert len(VALID_BYTES) == 48
    assert IDENTITY_BYTE == 4
    assert np.array_equal(rotation_from_byte(IDENTITY_BYTE), np.eye(3, dtype=np.int64))


def test_every_rotation_byte_round_trips_as_signed_permutation():
    for byte in VALID_BYTES:
        R = rotation_from_byte(byte)
        assert R.shape == (3, 3) and R.dtype == np.int64, byte
        assert np.array_equal(np.abs(R).sum(axis=0), [1, 1, 1]), byte
        assert np.array_equal(np.abs(R).sum(axis=1), [1, 1, 1]), byte
        assert rotation_to_byte(R) == byte


def test_rotation_bytes_are_distinct():
    matrices = {rotation_from_byte(b).tobytes() for b in VALID_BYTES}
    assert len(matrices) == 48


def test_rotation_byte_layout():
    # Byte 17: rows (0 -1 0), (1 0 0), (0 0 1); the apply matrix is the transpose.
    R = rotation_from_byte(17)
    assert R.tolist() == [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    # Bit 6 negates row 2.
    assert rotation_from_byte(4 | 64).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, -1]]


def test_every_invalid_rotation_byte_raises():
    assert len(INVALID_BYTES) == 256 - 48
    for byte in INVALID_BYTES:
        with pytest.raises(ValueError):
            rotation_from_byte(byte)


def test_rotation_to_byte_rejects_non_permutations():
    with pytest.raises(ValueError):
        rotation_to_byte(np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        rotation_to_byte(np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]))
    with pytest.raises(ValueError):
        rotation_to_byte(np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]]))


def test_apply_rotation_moves_every_cell_by_R_for_every_byte():
    voxels = model_voxels("robot")
    cells = np.argwhere(voxels != 0)
    for byte in VALID_BYTES:
        R = rotation_from_byte(byte)
        out, s = apply_rotation(voxels, R)
        assert out.dtype == np.uint8 and out.flags.c_contiguous, byte
        assert np.array_equal(_histogram(out), _histogram(voxels)), byte
        moved = cells @ R.T + s
        assert np.array_equal(out[moved[:, 0], moved[:, 1], moved[:, 2]], voxels[cells[:, 0], cells[:, 1], cells[:, 2]]), byte
        assert (s >= 0).all(), byte
        assert all(s[i] in (0, out.shape[i] - 1) for i in range(3)), byte


@pytest.mark.parametrize("pair", [(4, 17), (17, 17), (33, 66), (120, 9), (2, 50), (81, 100)])
def test_apply_rotation_composes(pair):
    b1, b2 = pair
    voxels = model_voxels("robot")
    R1, R2 = rotation_from_byte(b1), rotation_from_byte(b2)
    step, _ = apply_rotation(voxels, R1)
    twice, _ = apply_rotation(step, R2)
    once, _ = apply_rotation(voxels, R2 @ R1)
    assert np.array_equal(twice, once)


# --- rotate_model, flip_model ------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("turns", [1, 2, 3, 4])
def test_rotate_model_turns_world_cells_about_the_pivot(axis, turns):
    voxels = model_voxels("robot")
    position = (5, -7, 3)
    pivot2 = (2 * 5 + 24 + 4, 2 * -7 + 30 - 6, 2 * 3 + 20 + 2)  # off the box centre, even
    out, new_position = rotate_model(voxels, position, axis, turns, pivot2)
    R = np.linalg.matrix_power(QUARTER[axis], turns)
    piv = np.asarray(pivot2)
    expected = {}
    for cell, colour in _world_cells(voxels, position).items():
        c2 = 2 * np.asarray(cell) + 1
        w2 = R @ (c2 - piv) + piv
        expected[tuple(int(v) for v in (w2 - 1) // 2)] = colour
    assert _world_cells(out, new_position) == expected
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    if turns == 4:
        assert np.array_equal(out, voxels)
        assert new_position == position


def test_rotate_model_with_odd_pivot_of_matching_parity():
    voxels = np.zeros((3, 2, 1), dtype=np.uint8)
    voxels[2, 0, 0] = 7
    out, position = rotate_model(voxels, (0, 0, 0), 2, 1, (3, 3, 1))
    assert out.shape == (2, 3, 1)
    # Cell (2, 0) about pivot (1.5, 1.5): centre (2.5, 0.5) -> (2.5, 2.5) -> cell (2, 2).
    assert _world_cells(out, position) == {(2, 2, 0): 7}


def test_snapping_a_mixed_parity_box_centre_walks_the_model():
    """Why the gizmo keeps the box centre instead of snapping it.

    A 7 x 4 footprint has its box centre on a half cell in x. snap_pivot then adds 1
    to one plane axis each quarter turn, and the push adds up: four turns leave the
    array right but the model two cells over. Keeping t = position + size // 2 is
    exact for any size, which is the rule rok.rotate and the gizmo use.
    """
    voxels = np.zeros((7, 4, 3), dtype=np.uint8)
    voxels[0, 0, 0] = 5
    position = (-3, -2, -1)

    snapped = position
    array = voxels
    for _ in range(4):
        size = array.shape
        pivot2 = tuple(2 * snapped[i] + size[i] for i in range(3))
        array, snapped = rotate_model(array, snapped, 2, 1, snap_pivot(pivot2, 2))
    assert np.array_equal(array, voxels)
    assert snapped == (-1, 0, -1)

    kept = position
    array = voxels
    for _ in range(4):
        size = array.shape
        centre = tuple(kept[i] + size[i] // 2 for i in range(3))
        array, _ = rotate_model(array, kept, 2, 1, (0, 0, 0))
        kept = tuple(centre[i] - array.shape[i] // 2 for i in range(3))
    assert np.array_equal(array, voxels)
    assert kept == position


def test_rotate_model_zero_turns_is_a_copy():
    voxels = model_voxels("car")
    out, position = rotate_model(voxels, (1, 2, 3), 0, 0, (0, 0, 0))
    assert np.array_equal(out, voxels)
    assert not np.shares_memory(out, voxels)
    assert position == (1, 2, 3)


@pytest.mark.parametrize("stem", MODELS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_model_twice_is_identity_and_keeps_the_pivot_plane(stem, axis):
    voxels = model_voxels(stem)
    position = (-3, 8, 1)
    pivot2 = [2 * p + d for p, d in zip(position, voxels.shape)]
    pivot2[axis] += 5
    once, pos1 = flip_model(voxels, position, axis, pivot2)
    twice, pos2 = flip_model(once, pos1, axis, pivot2)
    assert np.array_equal(twice, voxels)
    assert pos2 == position
    before = _world_cells(voxels, position)
    after = _world_cells(once, pos1)
    expected = {}
    for cell, colour in before.items():
        c = list(cell)
        c[axis] = pivot2[axis] - cell[axis] - 1
        expected[tuple(c)] = colour
    assert after == expected
    for other in range(3):
        if other != axis:
            assert pos1[other] == position[other]


# --- mirror_half -------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("from_positive", [False, True])
def test_mirror_half_is_symmetric_and_keeps_the_source(axis, from_positive):
    voxels = model_voxels("bird")
    out = mirror_half(voxels, axis, from_positive)
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert out.shape == voxels.shape
    assert np.array_equal(out, np.flip(out, axis))
    n = voxels.shape[axis]
    k = n // 2
    source = [slice(None)] * 3
    source[axis] = slice(n - k, n) if from_positive else slice(0, k)
    assert np.array_equal(out[tuple(source)], voxels[tuple(source)])
    assert not np.shares_memory(out, voxels)


def test_mirror_half_keeps_the_middle_layer_of_an_odd_size():
    voxels = np.zeros((5, 1, 1), dtype=np.uint8)
    voxels[:, 0, 0] = [1, 2, 3, 4, 5]
    assert mirror_half(voxels, 0, False)[:, 0, 0].tolist() == [1, 2, 3, 2, 1]
    assert mirror_half(voxels, 0, True)[:, 0, 0].tolist() == [5, 4, 3, 4, 5]
    with pytest.raises(ValueError):
        mirror_half(voxels, 3, False)


# --- fill_all, fill_volume, inverse ------------------------------------------


@pytest.mark.parametrize("stem", MODELS)
def test_fill_all_recolours_only_the_filled_cells(stem):
    voxels = model_voxels(stem)
    out = fill_all(voxels, 7)
    assert out.dtype == np.uint8
    assert out.shape == voxels.shape
    assert np.array_equal(out != 0, voxels != 0)
    assert (out[voxels != 0] == 7).all()
    assert not np.shares_memory(out, voxels)


def test_fill_all_does_not_change_the_input():
    voxels = np.array([[[1, 0], [0, 2]]], dtype=np.uint8)
    fill_all(voxels, 9)
    assert voxels.tolist() == [[[1, 0], [0, 2]]]


def test_fill_volume_makes_every_cell_the_colour():
    out = fill_volume((2, 3, 4), 5)
    assert out.dtype == np.uint8
    assert out.shape == (2, 3, 4)
    assert (out == 5).all()


def test_fill_volume_bad_shape_raises():
    with pytest.raises(ValueError):
        fill_volume((2, 3), 5)


@pytest.mark.parametrize("stem", MODELS)
def test_inverse_swaps_filled_and_empty(stem):
    voxels = model_voxels(stem)
    out = inverse(voxels, 3)
    assert out.dtype == np.uint8
    assert out.shape == voxels.shape
    assert np.array_equal(out == 0, voxels != 0)
    assert (out[voxels == 0] == 3).all()
    assert not np.shares_memory(out, voxels)


def test_inverse_on_hand_values():
    voxels = np.array([[[1, 0, 2]]], dtype=np.uint8)
    assert inverse(voxels, 9).tolist() == [[[0, 9, 0]]]


# --- loop ----------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_loop_wraps_around(axis):
    voxels = np.zeros((4, 4, 4), dtype=np.uint8)
    at = [0, 0, 0]
    voxels[tuple(at)] = 5
    out = loop(voxels, axis, 1)
    moved = [0, 0, 0]
    moved[axis] = 1
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert out[tuple(moved)] == 5
    assert np.count_nonzero(out) == 1


def test_loop_negative_offset_wraps_the_other_way():
    voxels = np.zeros((4, 1, 1), dtype=np.uint8)
    voxels[0, 0, 0] = 5
    out = loop(voxels, 0, -1)
    assert out[3, 0, 0] == 5


@pytest.mark.parametrize("stem", MODELS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_loop_by_the_axis_size_is_identity(stem, axis):
    voxels = model_voxels(stem)
    out = loop(voxels, axis, voxels.shape[axis])
    assert np.array_equal(out, voxels)


@pytest.mark.parametrize("stem", MODELS)
def test_loop_keeps_every_colour(stem):
    voxels = model_voxels(stem)
    out = loop(voxels, 0, 7)
    assert np.array_equal(_histogram(out), _histogram(voxels))


def test_loop_bad_axis_raises():
    voxels = np.zeros((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        loop(voxels, 3, 1)


# --- fit_bounds ----------------------------------------------------------------


def test_fit_bounds_of_an_empty_grid_is_none():
    assert fit_bounds(np.zeros((5, 5, 5), dtype=np.uint8)) is None


def test_fit_bounds_of_one_voxel_is_a_single_cell():
    voxels = np.zeros((5, 5, 5), dtype=np.uint8)
    voxels[2, 3, 4] = 7
    assert fit_bounds(voxels) == ((2, 3, 4), (2, 3, 4))


@pytest.mark.parametrize("stem", MODELS)
def test_fit_bounds_matches_argwhere(stem):
    voxels = model_voxels(stem)
    cells = np.argwhere(voxels != 0)
    expected = (tuple(cells.min(axis=0).tolist()), tuple(cells.max(axis=0).tolist()))
    assert fit_bounds(voxels) == expected


def test_fit_bounds_is_a_tight_box():
    voxels = model_voxels("bird")
    lo, hi = fit_bounds(voxels)
    cropped = voxels[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
    assert np.count_nonzero(cropped) == np.count_nonzero(voxels)
    for axis in range(3):
        assert np.take(cropped, 0, axis=axis).any()
        assert np.take(cropped, -1, axis=axis).any()


# --- scale_up, scale_down ------------------------------------------------------


def test_scale_up_repeats_each_voxel_into_a_block():
    voxels = np.array([[[1, 0]]], dtype=np.uint8)  # shape (1, 1, 2)
    out = scale_up(voxels)
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert out.shape == (2, 2, 4)
    assert (out[:, :, :2] == 1).all()
    assert (out[:, :, 2:] == 0).all()


def test_scale_down_halves_each_axis_rounding_up():
    voxels = np.arange(1, 31, dtype=np.uint8).reshape(5, 2, 3)
    out = scale_down(voxels)
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert out.shape == (3, 1, 2)


def test_scale_down_takes_each_colour_from_its_own_block():
    voxels = np.arange(1, 9, dtype=np.uint8).reshape(4, 2, 1)
    out = scale_down(voxels)
    for i in range(2):
        assert out[i, 0, 0] in voxels[2 * i : 2 * i + 2, :, :]


def test_scale_down_keeps_a_thin_diagonal_whole():
    voxels = np.zeros((16, 16, 1), dtype=np.uint8)
    for k in range(15):
        voxels[k, k + 1, 0] = 4  # a slant between the cells "every second voxel" kept
    out = scale_down(voxels)
    assert all(out[k, k, 0] == 4 for k in range(8))


@pytest.mark.parametrize("stem", MODELS)
def test_scale_up_then_down_is_identity(stem):
    voxels = model_voxels(stem)
    out = scale_down(scale_up(voxels))
    assert out.shape == voxels.shape
    assert np.array_equal(out, voxels)


def test_scale_up_raises_when_the_result_would_be_too_big():
    voxels = np.zeros((200, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        scale_up(voxels)


def test_scale_up_at_the_size_limit_is_allowed():
    voxels = np.zeros((128, 4, 4), dtype=np.uint8)
    out = scale_up(voxels)
    assert out.shape == (MAX_SIZE, 8, 8)


# --- repeat ----------------------------------------------------------------------


def test_repeat_tiles_the_model():
    voxels = np.array([[[1, 2]]], dtype=np.uint8)  # shape (1, 1, 2)
    out = repeat(voxels, (2, 1, 3))
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert out.shape == (2, 1, 6)
    assert out[0, 0].tolist() == [1, 2, 1, 2, 1, 2]
    assert out[1, 0].tolist() == out[0, 0].tolist()


def test_repeat_by_one_is_a_copy():
    voxels = model_voxels("bird")
    out = repeat(voxels, (1, 1, 1))
    assert np.array_equal(out, voxels)
    assert not np.shares_memory(out, voxels)


def test_repeat_caps_at_the_grid_limit():
    voxels = np.zeros((100, 4, 4), dtype=np.uint8)
    out = repeat(voxels, (8, 1, 1))
    assert out.shape[0] == 200  # 256 // 100 == 2 whole copies fit; the rest of the 8 is dropped


def test_repeat_rejects_a_count_below_one():
    voxels = np.zeros((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        repeat(voxels, (0, 1, 1))


def test_repeat_bad_shape_raises():
    voxels = np.zeros((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        repeat(voxels, (1, 1))


# --- diagonal ----------------------------------------------------------------------


def test_diagonal_swaps_x_and_y():
    voxels = np.zeros((3, 2, 1), dtype=np.uint8)
    voxels[2, 0, 0] = 7
    out = diagonal(voxels)
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert out.shape == (2, 3, 1)
    assert out[0, 2, 0] == 7
    assert np.count_nonzero(out) == 1


@pytest.mark.parametrize("stem", MODELS)
def test_diagonal_twice_is_identity(stem):
    voxels = model_voxels(stem)
    out = diagonal(diagonal(voxels))
    assert out.shape == voxels.shape
    assert np.array_equal(out, voxels)


@pytest.mark.parametrize("stem", MODELS)
def test_diagonal_keeps_every_colour(stem):
    voxels = model_voxels(stem)
    out = diagonal(voxels)
    assert np.array_equal(_histogram(out), _histogram(voxels))


def test_drop_to_floor_moves_everything_down_together():
    from rok.core.grid_ops import drop_to_floor

    voxels = np.zeros((3, 3, 6), dtype=np.uint8)
    voxels[0, 0, 2] = 1
    voxels[2, 1, 4] = 7  # higher up: keeps its height above the first
    out, fall = drop_to_floor(voxels)
    assert fall == 2
    assert out[0, 0, 0] == 1 and out[2, 1, 2] == 7
    assert np.count_nonzero(out) == np.count_nonzero(voxels)
    assert out.shape == voxels.shape
    again, fall = drop_to_floor(out)
    assert fall == 0 and np.array_equal(again, out)
    empty, fall = drop_to_floor(np.zeros((2, 2, 2), dtype=np.uint8))
    assert fall == 0 and not empty.any()


# --- settle_cells: every selected voxel falls on its own ----------------------


def column(*cells, size=10):
    """A 1 x 1 grid column: each cell is (z, colour)."""
    voxels = np.zeros((1, 1, size), np.uint8)
    for z, colour in cells:
        voxels[0, 0, z] = colour
    return voxels


def test_a_selected_voxel_falls_to_the_floor_or_onto_a_voxel_that_stays():
    voxels = column((5, 3), (8, 4))
    mask = voxels == 3
    out, landed = settle_cells(voxels, mask)
    assert out[0, 0].tolist() == [3, 0, 0, 0, 0, 0, 0, 0, 4, 0]  # the floor; the unselected voxel stays up
    assert np.flatnonzero(landed).tolist() == [0]
    voxels = column((2, 9), (7, 3))
    out, _ = settle_cells(voxels, voxels == 3)
    assert np.flatnonzero(out[0, 0]).tolist() == [2, 3] and out[0, 0, 3] == 3  # on the voxel that stays


def test_a_pile_in_one_column_lands_bottom_first_with_its_colours_in_order():
    voxels = column((1, 9), (3, 1), (5, 2), (7, 3))
    mask = voxels != 9
    out, landed = settle_cells(voxels, mask)
    assert out[0, 0].tolist() == [0, 9, 1, 2, 3, 0, 0, 0, 0, 0]
    assert np.flatnonzero(landed).tolist() == [2, 3, 4]


def test_a_selected_voxel_below_an_unselected_one_stays_below_it():
    voxels = column((1, 3), (4, 9), (6, 3))
    out, _ = settle_cells(voxels, voxels == 3)
    assert out[0, 0].tolist() == [3, 0, 0, 0, 9, 3, 0, 0, 0, 0]  # the first to the floor, the second onto the 9


def test_each_column_falls_alone_and_a_resting_voxel_does_not_move():
    voxels = np.zeros((3, 1, 6), np.uint8)
    voxels[0, 0, 4] = 5
    voxels[1, 0, 0] = 6  # rests on the floor already
    voxels[2, 0, 3] = 7  # not selected
    mask = voxels != 7
    out, landed = settle_cells(voxels, mask)
    assert (out[0, 0] != 0).nonzero()[0].tolist() == [0] and out[0, 0, 0] == 5
    assert out[1, 0, 0] == 6 and out[2, 0, 3] == 7
    assert landed[0, 0, 0] and landed[1, 0, 0] and not landed[2].any()


def test_an_empty_mask_and_empty_cells_in_the_mask_change_nothing():
    voxels = column((3, 4))
    out, landed = settle_cells(voxels, np.zeros(voxels.shape, bool))
    assert np.array_equal(out, voxels) and not landed.any()
    out, landed = settle_cells(voxels, np.ones(voxels.shape, bool))  # the empty cells in the mask are not voxels
    assert out[0, 0, 0] == 4 and landed.sum() == 1


def test_settling_the_bird_keeps_every_voxel_and_every_colour_and_leaves_nothing_loose():
    voxels = model_voxels("bird")
    mask = np.zeros(voxels.shape, bool)
    mask[:, :, voxels.shape[2] // 2:] = True  # the whole upper half
    out, landed = settle_cells(voxels, mask)
    assert np.array_equal(np.bincount(out.ravel(), minlength=256)[1:], np.bincount(voxels.ravel(), minlength=256)[1:])
    assert np.array_equal(out[~mask & (voxels != 0)], voxels[~mask & (voxels != 0)])  # the rest did not move
    below = np.zeros_like(landed)
    below[:, :, 1:] = out[:, :, :-1] != 0
    assert np.all(~landed[:, :, 1:] | below[:, :, 1:])  # every settled voxel sits on the floor or on a voxel
    again, _ = settle_cells(out, landed)
    assert np.array_equal(again, out)  # settled once is settled
