"""Ray picking: six faces, six walls, blocked cells, edge cases and golden fixtures."""

import math

import numpy as np
import pytest

from rok.core.raycast import Hit, raycast, ray_to_rest
from rok.core.vox_read import read_vox

from helpers import FIXTURES


def one_voxel(size=(8, 8, 8), cell=(3, 3, 3)):
    voxels = np.zeros(size, dtype=np.uint8)
    voxels[cell] = 1
    return voxels


def empty_grid(size=(8, 8, 8)):
    return np.zeros(size, dtype=np.uint8)


# (origin, direction, expected normal, expected t) for the voxel at (3, 3, 3).
SIX_FACES = [
    ((10.0, 3.5, 3.5), (-1.0, 0.0, 0.0), (1, 0, 0), 6.0),
    ((-5.0, 3.5, 3.5), (1.0, 0.0, 0.0), (-1, 0, 0), 8.0),
    ((3.5, 10.0, 3.5), (0.0, -1.0, 0.0), (0, 1, 0), 6.0),
    ((3.5, -5.0, 3.5), (0.0, 1.0, 0.0), (0, -1, 0), 8.0),
    ((3.5, 3.5, 10.0), (0.0, 0.0, -1.0), (0, 0, 1), 6.0),
    ((3.5, 3.5, 0.5), (0.0, 0.0, 1.0), (0, 0, -1), 2.5),
]


@pytest.mark.parametrize("origin, direction, normal, t", SIX_FACES)
def test_six_faces(origin, direction, normal, t):
    hit = raycast(one_voxel(), origin, direction)
    assert hit is not None
    assert hit.voxel == (3, 3, 3)
    assert hit.normal == normal
    assert hit.t == pytest.approx(t)
    assert hit.target == tuple(3 + n for n in normal)
    assert not hit.wall


def test_hit_is_frozen():
    hit = Hit((1, 2, 3), (0, 0, 1), 1.0)
    with pytest.raises(AttributeError):
        hit.t = 2.0
    assert not hit.wall


def test_wall_hit_target_is_the_boundary_cell():
    floor = Hit((16, 16, -1), (0, 0, 1), 1.0, wall=True)
    assert floor.wall
    assert floor.target == (16, 16, 0)
    wall = Hit((8, 2, 3), (-1, 0, 0), 1.0, wall=True)
    assert wall.target == (7, 2, 3)


def test_diagonal_ray():
    # Enter the voxel through its +X face; the x-slab is crossed last.
    origin = (6.0, 5.4, 5.2)
    direction = (-1.0, -1.0, -1.0)
    hit = raycast(one_voxel(), origin, direction)
    assert hit is not None
    assert hit.voxel == (3, 3, 3)
    assert hit.normal == (1, 0, 0)
    assert hit.t == pytest.approx(2.0)


def test_miss_returns_none():
    hit = raycast(one_voxel(), (10.0, 6.5, 6.5), (-1.0, 0.0, 0.0), walls=False)
    assert hit is None


def test_miss_above_box_returns_none():
    hit = raycast(one_voxel(), (3.5, 3.5, 10.0), (0.0, 0.0, 1.0))
    assert hit is None


# ---------------------------------------------------------------- walls

# (axis, sign of the direction, origin) for an 8^3 grid: the ray runs along one
# axis through the column (2, 4) and leaves the box through the far wall.
WALLS = [
    (0, 1.0, (-5.0, 2.5, 4.5)),
    (0, -1.0, (13.0, 2.5, 4.5)),
    (1, 1.0, (2.5, -5.0, 4.5)),
    (1, -1.0, (2.5, 13.0, 4.5)),
    (2, 1.0, (2.5, 4.5, -5.0)),
    (2, -1.0, (2.5, 4.5, 13.0)),
]


@pytest.mark.parametrize("axis, sign, origin", WALLS)
def test_exit_through_each_wall(axis, sign, origin):
    direction = [0.0, 0.0, 0.0]
    direction[axis] = sign
    hit = raycast(empty_grid(), origin, direction)
    assert hit is not None
    assert hit.wall
    outside = list(origin)
    outside = [math.floor(v) for v in outside]
    outside[axis] = 8 if sign > 0 else -1
    assert hit.voxel == tuple(outside)
    normal = [0, 0, 0]
    normal[axis] = -int(sign)
    assert hit.normal == tuple(normal)
    boundary = list(outside)
    boundary[axis] = 7 if sign > 0 else 0
    assert hit.target == tuple(boundary)
    assert hit.t == pytest.approx(13.0)


def test_empty_grid_hits_floor():
    hit = raycast(empty_grid(), (2.5, 4.5, 10.0), (0.0, 0.0, -1.0))
    assert hit is not None
    assert hit == Hit((2, 4, -1), (0, 0, 1), 10.0, wall=True)
    assert hit.target == (2, 4, 0)


def test_empty_grid_without_walls_misses():
    assert raycast(empty_grid(), (2.5, 4.5, 10.0), (0.0, 0.0, -1.0), walls=False) is None
    assert raycast(empty_grid(), (-5.0, 4.5, 2.5), (1.0, 0.0, 0.0), walls=False) is None


def test_enter_minus_x_exit_plus_x():
    hit = raycast(empty_grid(), (-5.0, 2.5, 4.5), (1.0, 0.0, 0.0))
    assert hit is not None
    assert hit.wall
    assert hit.voxel == (8, 2, 4)
    assert hit.normal == (-1, 0, 0)
    assert hit.target == (7, 2, 4)


def test_ray_from_below_passes_the_floor_and_hits_the_top_wall():
    hit = raycast(empty_grid(), (2.5, 4.5, -10.0), (0.0, 0.0, 1.0))
    assert hit is not None
    assert hit.wall
    assert hit.voxel == (2, 4, 8)
    assert hit.normal == (0, 0, -1)
    assert hit.target == (2, 4, 7)
    assert hit.t == pytest.approx(18.0)


def test_near_wall_is_never_returned():
    # Sideways under the old floor level: the ray misses the box entirely.
    assert raycast(empty_grid(), (-5.0, 4.5, -0.5), (1.0, 0.0, 0.0)) is None
    # Through the box: the entry wall is transparent, only the exit counts.
    hit = raycast(empty_grid(), (-5.0, 4.5, 0.5), (1.0, 0.0, 0.0))
    assert hit is not None and hit.wall
    assert hit.voxel == (8, 4, 0)


def test_voxel_on_ground_beats_floor():
    voxels = empty_grid()
    voxels[2, 4, 0] = 7
    hit = raycast(voxels, (2.5, 4.5, 10.0), (0.0, 0.0, -1.0))
    assert hit is not None
    assert not hit.wall
    assert hit.voxel == (2, 4, 0)
    assert hit.normal == (0, 0, 1)
    assert hit.target == (2, 4, 1)


def test_voxel_next_to_far_wall_beats_the_wall():
    voxels = empty_grid()
    voxels[7, 2, 4] = 3
    hit = raycast(voxels, (-5.0, 2.5, 4.5), (1.0, 0.0, 0.0))
    assert hit is not None
    assert not hit.wall
    assert hit.voxel == (7, 2, 4)
    assert hit.normal == (-1, 0, 0)
    assert hit.t == pytest.approx(12.0)


def test_origin_inside_on_empty_cell_hits_far_wall():
    hit = raycast(empty_grid(), (1.5, 3.5, 3.5), (1.0, 0.0, 0.0))
    assert hit is not None
    assert hit.wall
    assert hit.voxel == (8, 3, 3)
    assert hit.normal == (-1, 0, 0)
    assert hit.t == pytest.approx(6.5)
    assert raycast(empty_grid(), (1.5, 3.5, 3.5), (1.0, 0.0, 0.0), walls=False) is None


def test_origin_inside_box_hits_first_filled_cell():
    voxels = empty_grid()
    voxels[5, 3, 3] = 1
    voxels[6, 3, 3] = 1
    hit = raycast(voxels, (1.5, 3.5, 3.5), (1.0, 0.0, 0.0))
    assert hit is not None
    assert hit.voxel == (5, 3, 3)
    assert hit.normal == (-1, 0, 0)
    assert hit.t == pytest.approx(3.5)


def test_origin_inside_filled_voxel_returns_none():
    assert raycast(one_voxel(), (3.5, 3.5, 3.5), (1.0, 0.0, 0.0)) is None
    assert raycast(one_voxel(), (3.5, 3.5, 3.5), (1.0, 0.0, 0.0), walls=False) is None


# ---------------------------------------------------------------- blocked


def test_blocked_cell_is_hit_as_if_solid():
    voxels = empty_grid()
    hit = raycast(voxels, (-5.0, 2.5, 4.5), (1.0, 0.0, 0.0), blocked={(3, 2, 4)})
    assert hit is not None
    assert not hit.wall
    assert hit.voxel == (3, 2, 4)
    assert hit.normal == (-1, 0, 0)
    assert hit.t == pytest.approx(8.0)
    assert voxels[3, 2, 4] == 0
    top = raycast(voxels, (3.5, 2.5, 20.0), (0.0, 0.0, -1.0), blocked={(3, 2, 4)})
    assert top is not None
    assert top.voxel == (3, 2, 4)
    assert top.normal == (0, 0, 1)


def test_blocked_entry_cell_gives_the_entry_normal():
    hit = raycast(empty_grid(), (-5.0, 2.5, 4.5), (1.0, 0.0, 0.0), blocked={(0, 2, 4)})
    assert hit is not None
    assert hit.voxel == (0, 2, 4)
    assert hit.normal == (-1, 0, 0)
    assert hit.t == pytest.approx(5.0)


def test_blocked_none_ignores_the_cell():
    hit = raycast(empty_grid(), (-5.0, 2.5, 4.5), (1.0, 0.0, 0.0), blocked=None)
    assert hit is not None
    assert hit.wall
    assert hit.voxel == (8, 2, 4)


def test_blocked_beyond_a_solid_voxel_does_not_matter():
    voxels = empty_grid()
    voxels[2, 2, 4] = 1
    hit = raycast(voxels, (-5.0, 2.5, 4.5), (1.0, 0.0, 0.0), blocked={(1, 2, 4)})
    assert hit is not None
    assert hit.voxel == (1, 2, 4)


# ---------------------------------------------------------------- edge cases


def test_hidden_interior_voxel_is_never_returned():
    voxels = np.ones((5, 5, 5), dtype=np.uint8)
    voxels[2, 2, 2] = 2
    for axis in range(3):
        for sign in (1.0, -1.0):
            origin = [2.5, 2.5, 2.5]
            origin[axis] = 2.5 - sign * 10.0
            direction = [0.0, 0.0, 0.0]
            direction[axis] = sign
            hit = raycast(voxels, origin, direction)
            assert hit is not None
            assert hit.voxel != (2, 2, 2)
            assert voxels[hit.voxel] == 1


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_axis_aligned_rays_with_zero_components(axis, sign):
    voxels = one_voxel()
    origin = [3.5, 3.5, 3.5]
    origin[axis] = 3.5 - sign * 20.0
    direction = [0.0, 0.0, 0.0]
    direction[axis] = sign
    hit = raycast(voxels, origin, direction, walls=False)
    assert hit is not None
    assert hit.voxel == (3, 3, 3)
    expected = [0, 0, 0]
    expected[axis] = -int(sign)
    assert hit.normal == tuple(expected)


def test_zero_direction_outside_slab_misses():
    voxels = one_voxel()
    assert raycast(voxels, (20.0, 3.5, 3.5), (0.0, 0.0, -1.0)) is None


def test_far_origin_hits_same_cell_as_near_origin():
    voxels = np.zeros((16, 16, 16), dtype=np.uint8)
    voxels[5, 7, 2] = 3
    direction = (-0.3, 0.2, -0.9)
    near_origin = (5.5 + 0.3 * 4.0, 7.5 - 0.2 * 4.0, 3.0 + 0.9 * 4.0)
    far_origin = tuple(near_origin[i] - direction[i] * 5000.0 for i in range(3))
    near = raycast(voxels, near_origin, direction)
    far = raycast(voxels, far_origin, direction)
    assert near is not None and far is not None
    assert near.voxel == far.voxel == (5, 7, 2)
    assert near.normal == far.normal == (0, 0, 1)
    assert far.t == pytest.approx(near.t + 5000.0)


def test_large_empty_grid_finishes_within_max_steps():
    voxels = np.zeros((256, 256, 256), dtype=np.uint8)
    # A long diagonal through the whole box: about 700 steps to the floor.
    origin = (-1.0, -1.0, 255.5)
    direction = (1.0, 1.0, -1.1)
    assert raycast(voxels, origin, direction, walls=False) is None
    hit = raycast(voxels, origin, direction, walls=True)
    assert hit is not None
    assert hit.wall
    assert hit.normal == (0, 0, 1)
    assert hit.voxel[2] == -1
    assert hit.voxel[0] > 200 and hit.voxel[1] > 200
    # Too few steps: the walk gives up before the wall.
    assert raycast(voxels, origin, direction, max_steps=10) is None


def test_unnormalised_direction_gives_scaled_t():
    hit = raycast(one_voxel(), (10.0, 3.5, 3.5), (-2.0, 0.0, 0.0))
    assert hit is not None
    assert hit.voxel == (3, 3, 3)
    assert hit.t == pytest.approx(3.0)


# ---------------------------------------------------------------- ray_to_rest


def test_ray_to_rest_undoes_a_90_degree_z_pose():
    # A 90 degree turn about Z about the grid's own origin: rest (x, y, z)
    # poses to (-y, x, z). Take one of SIX_FACES in rest space and turn its
    # ray by the same matrix, so ray_to_rest must turn it back.
    rest_origin, rest_direction, normal, t = SIX_FACES[0]
    matrix4 = ((0.0, -1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    posed_origin = (-rest_origin[1], rest_origin[0], rest_origin[2])
    posed_direction = (-rest_direction[1], rest_direction[0], rest_direction[2])

    origin, direction = ray_to_rest(posed_origin, posed_direction, matrix4)
    assert origin == pytest.approx(rest_origin)
    assert direction == pytest.approx(rest_direction)

    hit = raycast(one_voxel(), origin, direction)
    assert hit is not None
    assert hit.voxel == (3, 3, 3)
    assert hit.normal == normal
    assert hit.t == pytest.approx(t)
    assert not hit.wall


def test_ray_to_rest_identity_leaves_the_ray_alone():
    identity = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    origin = (10.0, 3.5, 3.5)
    direction = (-1.0, 0.0, 0.0)

    out_origin, out_direction = ray_to_rest(origin, direction, identity)
    assert out_origin == pytest.approx(origin)
    assert out_direction == pytest.approx(direction)


# ---------------------------------------------------------------- golden


def test_golden_bird_top_down():
    model = read_vox(FIXTURES / "bird.vox")
    voxels = model.voxels
    xs, ys, _ = np.nonzero(voxels)
    columns = sorted(set(zip(xs.tolist(), ys.tolist())))
    step = max(1, len(columns) // 20)
    picked = columns[::step][:20]
    assert len(picked) == 20
    for x, y in picked:
        top = int(np.nonzero(voxels[x, y, :])[0].max())
        hit = raycast(voxels, (x + 0.5, y + 0.5, 200.0), (0.0, 0.0, -1.0))
        assert hit is not None
        assert not hit.wall
        assert hit.voxel == (x, y, top)
        assert hit.normal == (0, 0, 1)
        assert hit.t == pytest.approx(200.0 - (top + 1))
        assert math.isfinite(hit.t)


def test_golden_robot_empty_columns_hit_the_floor():
    model = read_vox(FIXTURES / "robot.vox")
    voxels = model.voxels
    sx, sy, sz = voxels.shape
    filled = np.any(voxels != 0, axis=2)
    columns = [(x, y) for x in range(sx) for y in range(sy) if not filled[x, y]]
    step = max(1, len(columns) // 20)
    picked = columns[::step][:20]
    assert len(picked) == 20
    for x, y in picked:
        hit = raycast(voxels, (x + 0.5, y + 0.5, 200.0), (0.0, 0.0, -1.0))
        assert hit is not None
        assert hit.wall
        assert hit.voxel == (x, y, -1)
        assert hit.normal == (0, 0, 1)
        assert hit.target == (x, y, 0)
        assert hit.t == pytest.approx(200.0)
        origin = (x + 0.5, y + 0.5, 200.0)
        assert raycast(voxels, origin, (0.0, 0.0, -1.0), walls=False) is None


def test_box_enter_finds_the_box_of_an_empty_model():
    from rok.core.raycast import box_enter

    size = (4, 4, 4)
    assert box_enter((2.0, 2.0, 10.0), (0.0, 0.0, -1.0), size) == pytest.approx(6.0)
    assert box_enter((2.0, 2.0, 2.0), (0.0, 0.0, -1.0), size) == 0.0  # from inside
    assert box_enter((9.0, 9.0, 10.0), (0.0, 0.0, -1.0), size) is None  # beside it
    assert box_enter((2.0, 2.0, -3.0), (0.0, 0.0, -1.0), size) is None  # pointing away
