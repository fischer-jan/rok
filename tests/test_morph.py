"""Grow, shrink and hollow: dilate and erode against plain-loop
references."""

import numpy as np
import pytest

import reference
from rok.core.morph import (
    bevel_mask,
    bbox_slices,
    changed_cells,
    dilate,
    erode,
    grow,
    hollow,
    keep_masked,
    shrink,
    smooth,
)
from rok.core.vox_read import read_vox

from helpers import FIXTURES

CONNECTIVITIES = [(6, False), (26, True)]


def single_cell(shape=(7, 7, 7), at=(3, 3, 3)):
    mask = np.zeros(shape, dtype=bool)
    mask[at] = True
    return mask


def robot_crop(lo=(4, 6, 2), size=12):
    voxels = read_vox(FIXTURES / "robot.vox").voxels
    x, y, z = lo
    return voxels[x : x + size, y : y + size, z : z + size]


def solid_cube(shape=(12, 12, 12), lo=3, hi=7, colour=5):
    voxels = np.zeros(shape, dtype=np.uint8)
    voxels[lo:hi, lo:hi, lo:hi] = colour
    return voxels


# --------------------------------------------------------------- dilate / erode


@pytest.mark.parametrize("connectivity,round_corners", CONNECTIVITIES)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_dilate_matches_reference_on_robot(connectivity, round_corners, n):
    mask = robot_crop() != 0
    assert mask.any() and not mask.all()
    fast = dilate(mask, n, round_corners)
    assert fast.shape == mask.shape
    assert fast.dtype == bool
    assert np.array_equal(fast, reference.dilate_ref(mask, n, connectivity))


@pytest.mark.parametrize("connectivity,round_corners", CONNECTIVITIES)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_erode_matches_reference_on_robot(connectivity, round_corners, n):
    mask = robot_crop() != 0
    fast = erode(mask, n, round_corners)
    assert fast.shape == mask.shape
    assert fast.dtype == bool
    assert np.array_equal(fast, reference.erode_ref(mask, n, connectivity))


@pytest.mark.parametrize("connectivity,round_corners", CONNECTIVITIES)
def test_dilate_matches_reference_on_the_whole_model(connectivity, round_corners):
    mask = read_vox(FIXTURES / "robot.vox").voxels != 0
    assert np.array_equal(dilate(mask, 1, round_corners), reference.dilate_ref(mask, 1, connectivity))
    assert np.array_equal(erode(mask, 1, round_corners), reference.erode_ref(mask, 1, connectivity))


def test_single_cell_grows_to_a_cross_then_a_25_cell_shape():
    mask = single_cell()
    one = dilate(mask, 1)
    assert one.sum() == 7
    assert one[3, 3, 3] and one[2, 3, 3] and one[4, 3, 3]
    assert one[3, 2, 3] and one[3, 4, 3] and one[3, 3, 2] and one[3, 3, 4]
    assert not one[2, 2, 3]
    two = dilate(mask, 2)
    assert two.sum() == 25
    assert np.array_equal(two, dilate(one, 1))
    assert np.array_equal(two, reference.dilate_ref(mask, 2, 6))


def test_single_cell_with_round_corners_grows_to_a_box():
    mask = single_cell()
    one = dilate(mask, 1, round_corners=True)
    assert one.sum() == 27
    assert one[2:5, 2:5, 2:5].all()
    two = dilate(mask, 2, round_corners=True)
    assert two.sum() == 125
    assert np.array_equal(two, reference.dilate_ref(mask, 2, 26))


@pytest.mark.parametrize("connectivity,round_corners", CONNECTIVITIES)
def test_dilate_stops_at_the_grid_border(connectivity, round_corners):
    mask = single_cell((4, 4, 4), (0, 0, 0))
    out = dilate(mask, 2, round_corners)
    assert out.shape == (4, 4, 4)
    assert np.array_equal(out, reference.dilate_ref(mask, 2, connectivity))
    assert out[0, 0, 0]


def test_erode_treats_the_grid_border_as_empty():
    mask = np.ones((4, 4, 4), dtype=bool)
    out = erode(mask, 1)
    assert out.sum() == 8
    assert out[1:3, 1:3, 1:3].all()
    assert not erode(mask, 2).any()
    # A slab on the wall loses its wall face too.
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[0:3, 1:5, 1:5] = True
    out = erode(mask, 1)
    assert not out[0].any()
    assert out.sum() == 1 * 2 * 2
    assert out[1, 2:4, 2:4].all()
    assert np.array_equal(out, reference.erode_ref(mask, 1, 6))


def test_dilate_and_erode_of_an_empty_mask_are_empty():
    mask = np.zeros((5, 5, 5), dtype=bool)
    assert not dilate(mask, 1).any()
    assert not erode(mask, 1).any()
    assert not dilate(mask, 1, round_corners=True).any()


def test_dilate_n_zero_is_a_copy():
    mask = single_cell()
    out = dilate(mask, 0)
    assert np.array_equal(out, mask)
    assert out is not mask
    out[0, 0, 0] = True
    assert not mask[0, 0, 0]


def test_dilate_does_not_change_the_input():
    mask = single_cell()
    before = mask.copy()
    dilate(mask, 2)
    erode(mask, 1)
    assert np.array_equal(mask, before)


# ---------------------------------------------------------------------- bbox


def test_bbox_slices_pads_and_clips():
    mask = single_cell((10, 10, 10), (5, 5, 5))
    assert bbox_slices(mask, 0) == (slice(5, 6), slice(5, 6), slice(5, 6))
    assert bbox_slices(mask, 2) == (slice(3, 8), slice(3, 8), slice(3, 8))
    assert bbox_slices(mask, 7) == (slice(0, 10), slice(0, 10), slice(0, 10))
    mask[1, 8, 0] = True
    assert bbox_slices(mask, 1) == (slice(0, 7), slice(4, 10), slice(0, 7))
    assert bbox_slices(np.zeros((3, 3, 3), dtype=bool), 1) is None


@pytest.mark.parametrize("connectivity,round_corners", CONNECTIVITIES)
def test_bbox_work_equals_full_grid_work(connectivity, round_corners):
    """A small mask far from the walls of a grid gives the full-grid result.

    The grid is 16 x 14 x 14: big enough that the bounding-box clip is real, small
    enough that the one-cell-per-step reference stays fast.
    """
    mask = np.zeros((16, 14, 14), dtype=bool)
    mask[6:10, 4:6, 9] = True
    mask[8, 5, 10:12] = True
    for n in (1, 3):
        assert np.array_equal(dilate(mask, n, round_corners), reference.dilate_ref(mask, n, connectivity))
        assert np.array_equal(erode(mask, n, round_corners), reference.erode_ref(mask, n, connectivity))
    # The same near a wall, so the clip matters.
    mask = np.zeros((16, 14, 14), dtype=bool)
    mask[0:3, 12:14, 12:14] = True
    assert np.array_equal(dilate(mask, 2, round_corners), reference.dilate_ref(mask, 2, connectivity))
    assert np.array_equal(erode(mask, 1, round_corners), reference.erode_ref(mask, 1, connectivity))


# ---------------------------------------------------------------------- grow


def test_grow_returns_a_uint8_copy():
    voxels = solid_cube()
    out = grow(voxels, 1)
    assert out.dtype == np.uint8
    assert out is not voxels
    assert (voxels != 0).sum() == 64
    assert (out != 0).sum() == dilate(voxels != 0, 1).sum()


def test_grow_colours_follow_the_neighbour_order():
    """The centre cell is empty; each neighbour has its own colour."""
    voxels = np.zeros((7, 7, 7), dtype=np.uint8)
    at = {
        (2, 3, 3): 1,  # -x
        (4, 3, 3): 2,  # +x
        (3, 2, 3): 3,  # -y
        (3, 4, 3): 4,  # +y
        (3, 3, 2): 5,  # -z
        (3, 3, 4): 6,  # +z
    }
    order = [(2, 3, 3), (4, 3, 3), (3, 2, 3), (3, 4, 3), (3, 3, 2), (3, 3, 4)]
    # Drop the neighbours one by one from the front: the centre takes the next one.
    for k in range(6):
        voxels[:] = 0
        for cell in order[k:]:
            voxels[cell] = at[cell]
        out = grow(voxels, 1)
        assert out[3, 3, 3] == at[order[k]]


def test_grow_colour_comes_from_the_source_cell():
    voxels = np.zeros((9, 9, 9), dtype=np.uint8)
    voxels[4, 4, 4] = 7
    out = grow(voxels, 2)
    filled = out != 0
    assert filled.sum() == 25
    assert set(np.unique(out[filled]).tolist()) == {7}


def test_grow_with_round_corners_fills_the_box_with_the_source_colour():
    voxels = np.zeros((9, 9, 9), dtype=np.uint8)
    voxels[4, 4, 4] = 7
    out = grow(voxels, 2, round_corners=True)
    assert np.array_equal(out != 0, dilate(voxels != 0, 2, round_corners=True))
    assert set(np.unique(out[out != 0]).tolist()) == {7}
    # Two colours side by side: a cell grown diagonally takes a colour from one of them.
    voxels = np.zeros((9, 9, 9), dtype=np.uint8)
    voxels[3, 4, 4] = 1
    voxels[5, 4, 4] = 2
    out = grow(voxels, 1, round_corners=True)
    assert np.array_equal(out != 0, dilate(voxels != 0, 1, round_corners=True))
    assert out[4, 4, 4] == 1
    assert out[2, 5, 5] == 1
    assert out[6, 3, 3] == 2


@pytest.mark.parametrize("round_corners", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_grow_then_shrink_of_a_convex_box_is_identity(round_corners, n):
    voxels = solid_cube((16, 16, 16), 5, 10)
    grown = grow(voxels, n, round_corners=round_corners)
    assert (grown != 0).sum() > (voxels != 0).sum()
    assert np.array_equal(shrink(grown, n, round_corners=round_corners), voxels)


def test_grow_matches_dilate_on_a_model():
    voxels = read_vox(FIXTURES / "robot.vox").voxels
    for n in (1, 2):
        out = grow(voxels, n)
        assert np.array_equal(out != 0, dilate(voxels != 0, n))
        # Existing voxels keep their colours.
        filled = voxels != 0
        assert np.array_equal(out[filled], voxels[filled])


def test_grow_stops_at_the_grid_border():
    voxels = np.zeros((4, 4, 4), dtype=np.uint8)
    voxels[0, 0, 0] = 3
    out = grow(voxels, 5)
    assert out.shape == (4, 4, 4)
    assert np.array_equal(out != 0, dilate(voxels != 0, 5))
    # The far corner is 9 steps away.
    out = grow(voxels, 9)
    assert (out == 3).all()


def test_limited_grow_adds_only_next_to_the_mask():
    voxels = np.zeros((12, 12, 12), dtype=np.uint8)
    voxels[2:6, 4:8, 4:8] = 1  # selected half
    voxels[6:10, 4:8, 4:8] = 2  # unselected half, touching
    limit = np.zeros(voxels.shape, dtype=bool)
    limit[2:6, 4:8, 4:8] = True
    out = grow(voxels, 1, limit=limit)
    added = (out != 0) & (voxels == 0)
    expected = dilate(limit, 1) & (voxels == 0)
    assert np.array_equal(added, expected)
    assert set(np.unique(out[added]).tolist()) == {1}
    # Nothing outside the selection changed.
    untouched = ~added
    assert np.array_equal(out[untouched], voxels[untouched])
    # The unselected half did not grow: its far face stays bare.
    assert not (out[10, 4:8, 4:8] != 0).any()


def test_limited_grow_never_overwrites_unselected_voxels():
    voxels = np.zeros((8, 8, 8), dtype=np.uint8)
    voxels[3, 3, 3] = 1
    voxels[4, 3, 3] = 2  # unselected neighbour
    limit = np.zeros(voxels.shape, dtype=bool)
    limit[3, 3, 3] = True
    out = grow(voxels, 2, limit=limit)
    assert out[4, 3, 3] == 2
    assert out[3, 3, 3] == 1
    # The second step spreads from the new cells only, not through the unselected voxel.
    assert out[5, 3, 3] == 0
    assert out[1, 3, 3] == 1
    assert out[2, 2, 3] == 1


def test_limited_grow_with_an_empty_limit_changes_nothing():
    voxels = solid_cube()
    out = grow(voxels, 3, limit=np.zeros(voxels.shape, dtype=bool))
    assert np.array_equal(out, voxels)


# -------------------------------------------------------------------- shrink


def test_shrink_removes_the_exposed_layer():
    voxels = solid_cube((12, 12, 12), 2, 8)  # 6^3
    out = shrink(voxels, 1)
    assert (out != 0).sum() == 4**3
    assert (out[3:7, 3:7, 3:7] == 5).all()
    assert np.array_equal(shrink(voxels, 3), np.zeros_like(voxels))


def test_shrink_matches_erode_on_a_model():
    voxels = read_vox(FIXTURES / "bird.vox").voxels
    out = shrink(voxels, 1)
    assert np.array_equal(out != 0, erode(voxels != 0, 1))
    kept = out != 0
    assert np.array_equal(out[kept], voxels[kept])


def test_shrink_on_the_grid_wall_removes_the_wall_cells():
    voxels = np.zeros((6, 6, 6), dtype=np.uint8)
    voxels[0:3, 0:6, 0:6] = 4
    out = shrink(voxels, 1)
    assert not out[0].any()
    assert not out[:, 0].any() and not out[:, 5].any()
    assert (out != 0).sum() == 1 * 4 * 4


def test_limited_shrink_keeps_interior_selected_cells():
    voxels = solid_cube((10, 10, 10), 2, 7)  # 5^3 at 2..6
    limit = np.zeros(voxels.shape, dtype=bool)
    limit[4, 4, 2:7] = True  # a column through the cube
    out = shrink(voxels, 1, limit=limit)
    # Only the two ends of the column touch empty space.
    assert out[4, 4, 2] == 0
    assert out[4, 4, 6] == 0
    assert (out[4, 4, 3:6] == 5).all()
    # Unselected cells stay, even on the surface.
    removed = (voxels != 0) & (out == 0)
    assert removed.sum() == 2
    assert np.array_equal(out[~limit], voxels[~limit])


def test_limited_shrink_with_round_corners():
    voxels = solid_cube((10, 10, 10), 2, 7)
    limit = np.zeros(voxels.shape, dtype=bool)
    limit[2:7, 2:7, 2] = True  # the bottom layer
    out = shrink(voxels, 1, limit=limit, round_corners=True)
    assert not out[2:7, 2:7, 2].any()
    assert (out[2:7, 2:7, 3:7] == 5).all()


def test_shrink_n_beyond_the_model_gives_an_empty_grid():
    voxels = read_vox(FIXTURES / "robot.vox").voxels
    assert not shrink(voxels, 16).any()


# -------------------------------------------------------------------- hollow


def test_hollow_of_a_solid_cube_leaves_the_wall():
    voxels = np.full((10, 10, 10), 3, dtype=np.uint8)
    out = hollow(voxels, 1)
    assert (out != 0).sum() == 10**3 - 8**3
    assert not out[1:9, 1:9, 1:9].any()
    assert (out[0] == 3).all() and (out[9] == 3).all()
    out = hollow(voxels, 2)
    assert (out != 0).sum() == 10**3 - 6**3
    assert np.array_equal(hollow(voxels, 5), voxels)


def test_hollow_with_a_limit_clears_only_inside_the_mask():
    voxels = np.full((10, 10, 10), 3, dtype=np.uint8)
    limit = np.zeros(voxels.shape, dtype=bool)
    limit[0:5] = True
    out = hollow(voxels, 1, limit=limit)
    cleared = (voxels != 0) & (out == 0)
    assert cleared.sum() == 4 * 8 * 8
    assert not cleared[5:].any()
    assert np.array_equal(out[~limit], voxels[~limit])


def test_hollow_keeps_a_thin_model():
    voxels = read_vox(FIXTURES / "robot.vox").voxels
    out = hollow(voxels, 1)
    assert np.array_equal(out != 0, (voxels != 0) & ~erode(voxels != 0, 1))
    kept = out != 0
    assert np.array_equal(out[kept], voxels[kept])


# ------------------------------------------------------------- changed_cells


def test_changed_cells_lists_the_difference_in_argwhere_order():
    before = np.zeros((4, 4, 4), dtype=np.uint8)
    before[1, 1, 1] = 2
    before[2, 2, 2] = 3
    after = before.copy()
    after[1, 1, 1] = 0
    after[0, 3, 2] = 9
    after[3, 0, 0] = 1
    cells, values = changed_cells(before, after)
    assert cells.dtype == np.int64
    assert values.dtype == np.uint8
    assert cells.shape == (3, 3)
    assert cells.tolist() == [[0, 3, 2], [1, 1, 1], [3, 0, 0]]
    assert values.tolist() == [9, 0, 1]


def test_changed_cells_of_equal_arrays_is_empty():
    voxels = solid_cube()
    cells, values = changed_cells(voxels, voxels.copy())
    assert cells.shape == (0, 3)
    assert values.shape == (0,)


def test_changed_cells_round_trip_through_grow():
    voxels = read_vox(FIXTURES / "robot.vox").voxels
    after = grow(voxels, 1)
    cells, values = changed_cells(voxels, after)
    assert cells.shape[0] == int((after != voxels).sum())
    rebuilt = voxels.copy()
    rebuilt[cells[:, 0], cells[:, 1], cells[:, 2]] = values
    assert np.array_equal(rebuilt, after)


# ------------------------------------------------- cut voxels look empty


def test_grow_on_visible_ignores_a_cut_voxel():
    """Grow of the visible array: a cut voxel neither grows nor is written.

    model_ops runs the morphs on grid.visible and diffs against it; set_voxels
    then refuses the cut cell itself.
    """
    from rok.core.grid import VoxelGrid
    from rok.core.undo import StrokeRecorder

    grid = VoxelGrid((12, 12, 12))
    grid.set_voxel(4, 4, 4, 3)
    grid.set_voxel(0, 0, 0, 2)
    grid.set_section(4)  # cuts z = 4 and above, so (4, 4, 4) is cut
    new = grow(grid.visible, 1)
    cells, values = changed_cells(grid.visible, new)
    rec = StrokeRecorder(grid)
    rec.set_many(cells, values)
    assert not grid.visible[3:6, 3:6, 3:6].any()  # nothing grew around the cut cell
    assert grid.voxels[4, 4, 4] == 3
    assert int((grid.visible != 0).sum()) == 4  # (0,0,0) and its three in-grid neighbours


def test_shrink_on_visible_does_not_let_a_cut_voxel_shield_its_neighbour():
    from rok.core.grid import VoxelGrid

    grid = VoxelGrid((12, 12, 12))
    grid.replace_voxels(solid_cube((12, 12, 12), 3, 7, 5))
    grid.set_section(4, axis=0, from_positive=False)  # cuts the x = 3 face layer
    new = shrink(grid.visible, 1)
    # The layer behind the cut one is exposed now and goes.
    assert not new[4, 3:7, 3:7].any()
    assert not new[3, 3:7, 3:7].any()
    assert new[5, 4:6, 4:6].all()


def test_bevel_takes_convex_edges_and_keeps_thin_parts():
    cube = np.zeros((9, 9, 9), dtype=bool)
    cube[2:7, 2:7, 2:7] = True  # a 5-cube in the air
    remove = bevel_mask(cube)
    assert int(remove.sum()) == 44  # 8 corners and 12 edges of 3
    assert not remove[4, 4, 6] and remove[2, 2, 4] and remove[2, 4, 6]
    small = np.zeros((5, 5, 5), dtype=bool)
    small[1:4, 1:4, 1:4] = True
    assert int((small & ~bevel_mask(small)).sum()) == 7  # the centre and the six face middles
    plate = np.zeros((7, 7, 3), dtype=bool)
    plate[1:6, 1:6, 1] = True  # one voxel thick: its edge stays
    assert not bevel_mask(plate).any()
    wing = np.zeros((9, 5, 5), dtype=bool)
    wing[1:4, 1:4, 1:4] = True  # a block
    wing[4:8, 2, 2] = True  # with a one-voxel wing
    remove = bevel_mask(wing)
    assert not remove[4:8, 2, 2].any() and remove[1, 1, 1]
    # On the floor the horizontal base edges stay (the wall counts as filled), while
    # the vertical edges run down to the floor; the selection limits the bevel.
    standing = np.zeros((7, 7, 5), dtype=bool)
    standing[1:6, 1:6, 0:4] = True
    remove = bevel_mask(standing)
    assert not remove[3, 1, 0] and remove[1, 1, 0] and remove[1, 1, 3] and remove[3, 1, 3]
    assert int(remove.sum()) == 16 + 4 * 3  # the top rim and the four corner columns below it
    limit = np.zeros(standing.shape, dtype=bool)
    limit[:, :, 3] = True
    assert bevel_mask(standing, limit=limit).sum() == 16 and not bevel_mask(standing, limit=limit)[1, 3, 2]



# ---------------------------------------------------------------- smooth


def test_smooth_takes_the_corners_off_a_cube_and_keeps_its_faces():
    voxels = solid_cube(lo=3, hi=9)
    out = smooth(voxels)
    for corner in ((3, 3, 3), (8, 8, 8), (3, 8, 3)):
        assert out[corner] == 0
    assert out[5, 5, 3] == 5 and out[3, 5, 5] == 5  # the middle of a face stays
    assert out[5, 5, 5] == 5


def test_smooth_fills_a_dent_with_the_colour_around_it():
    voxels = solid_cube(lo=2, hi=10, colour=7)
    voxels[5, 5, 9] = 0  # a one-voxel dent in the top face
    out = smooth(voxels)
    assert out[5, 5, 9] == 7


def test_smooth_removes_a_bump_and_a_lone_voxel():
    voxels = solid_cube(shape=(12, 12, 12), lo=2, hi=10)
    voxels[5, 5, 10] = 5  # a bump on the top face
    voxels[0, 0, 11] = 3  # a lone voxel
    out = smooth(voxels)
    assert out[5, 5, 10] == 0 and out[0, 0, 11] == 0


@pytest.mark.parametrize("thickness", [1, 2])
def test_smooth_keeps_a_thin_plate_whole(thickness):
    voxels = np.zeros((12, 12, 12), dtype=np.uint8)
    voxels[2:10, 2:10, 5 : 5 + thickness] = 4
    assert np.array_equal(smooth(voxels), voxels)


def test_smooth_keeps_a_slab_on_the_grid_floor_flat():
    voxels = np.zeros((10, 10, 10), dtype=np.uint8)
    voxels[:, :, 0:4] = 2  # wall to wall: no corner anywhere
    assert np.array_equal(smooth(voxels), voxels)


def test_smooth_with_a_selection_changes_only_the_selection_and_next_to_it():
    voxels = solid_cube(lo=2, hi=10)
    limit = np.zeros(voxels.shape, dtype=bool)
    limit[2:5, 2:5, 2:5] = voxels[2:5, 2:5, 2:5] != 0  # the corner at (2, 2, 2)
    out = smooth(voxels, limit=limit)
    assert out[2, 2, 2] == 0  # the selected corner went
    assert out[9, 9, 9] == 5  # an unselected corner stayed
    changed = np.argwhere(out != voxels)
    assert len(changed) and changed.max() <= 5


def test_smooth_on_a_real_model_changes_its_surface_but_keeps_most_of_it():
    voxels = read_vox(FIXTURES / "robot.vox").voxels
    out = smooth(voxels)
    before, after = np.count_nonzero(voxels), np.count_nonzero(out)
    assert not np.array_equal(out, voxels)
    assert after > 0.8 * before
    assert set(np.unique(out)) <= set(np.unique(voxels))  # no new colours


def test_keep_masked_keeps_masked_voxels_and_builds_nothing_against_them():
    before = np.zeros((5, 3, 3), np.uint8)
    before[1:4, 1, 1] = 4
    mask = np.zeros(before.shape, bool)
    mask[1, 1, 1] = True
    after = np.zeros_like(before)  # a shader that clears everything ...
    after[:, :, 2] = 7  # ... and builds a roof
    out = keep_masked(before, after, mask)
    assert out[1, 1, 1] == 4  # masked: kept
    assert out[2, 1, 1] == 0 and out[3, 1, 1] == 0  # not masked: cleared
    assert out[1, 1, 2] == 0  # would touch the masked voxel: not built
    assert out[2, 1, 2] == 7 and out[1, 0, 2] == 7  # touches it only by an edge: built
