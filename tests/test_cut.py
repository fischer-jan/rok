import numpy as np

from rok.core.cut import boundary, face_axes, fitted, pick_axis, plane_corners, split_at

# Screen directions of the three axes in a typical view from the front right.
SCREEN = {0: (0.8, -0.3), 1: (0.6, 0.35), 2: (0.0, 1.0)}


def test_a_face_has_the_two_axes_that_are_not_its_normal():
    assert face_axes((0, 0, 1)) == (0, 1)
    assert face_axes((-1, 0, 0)) == (1, 2)


def test_the_plane_stands_upright_on_screen_and_tab_turns_it():
    # On a side face (normal x) the in-face axes are y and z; y runs across the
    # screen, so the plane is across y: an upright slice.
    assert pick_axis((1, 0, 0), SCREEN) == 1
    assert pick_axis((1, 0, 0), SCREEN, other=True) == 2
    # On the top face both in-face axes run across; the more horizontal wins.
    assert pick_axis((0, 0, 1), SCREEN) == 0


def test_the_cut_snaps_to_the_nearest_boundary_inside_the_model():
    assert boundary((3.4, 0, 0), 0, (10, 4, 4)) == 3
    assert boundary((3.6, 0, 0), 0, (10, 4, 4)) == 4
    assert boundary((0.2, 0, 0), 0, (10, 4, 4)) is None  # at the edge: one side empty
    assert boundary((9.9, 0, 0), 0, (10, 4, 4)) is None


def test_a_split_keeps_every_voxel_once_in_two_boxes():
    rng = np.random.default_rng(3)
    v = rng.integers(0, 5, size=(6, 7, 8)).astype(np.uint8)
    low, high = split_at(v, 1, 3)
    assert low.shape == (6, 3, 8) and high.shape == (6, 4, 8)
    assert np.array_equal(np.concatenate([low, high], axis=1), v)


def test_the_plane_spans_the_box_at_the_boundary():
    corners = plane_corners((10, 20, 0), (4, 6, 8), 1, 2)
    assert all(c[1] == 22 for c in corners)
    assert {(c[0], c[2]) for c in corners} == {(10, 0), (14, 0), (14, 8), (10, 8)}


def test_a_part_fits_its_box_to_its_voxels():
    v = np.zeros((6, 6, 6), dtype=np.uint8)
    v[2:4, 1:5, 3] = 7
    box, offset = fitted(v)
    assert box.shape == (2, 4, 1) and offset == (2, 1, 3) and (box == 7).all()
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    assert fitted(empty)[0].shape == (2, 2, 2)
