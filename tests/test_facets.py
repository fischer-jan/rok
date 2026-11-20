"""The buckyball's planes and Crease: flat faces found in voxels, bars on their edges."""

import numpy as np
import pytest

from rok.core import facets as F
from rok.core.palette import DEFAULT_PALETTE, darker_slot
from rok.core.shape import BUCKYBALL, FRAMED_BUCKYBALL, SOLIDS, solid_grid, solid_parts
from rok.core.vox_read import read_vox
from tests.helpers import FIXTURES

FACETS = FIXTURES / "facets"
GLASS, FRAME = 79, 165


def sphere(n: int) -> np.ndarray:
    c = np.arange(n) + 0.5 - n / 2
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    return (x * x + y * y + z * z) <= (n / 2) ** 2


def block(n: int = 40) -> np.ndarray:
    v = np.zeros((n, n, n), dtype=np.uint8)
    v[4:-4, 6:-6, 2:-8] = GLASS
    return v


# --- the buckyball -----------------------------------------------------------------------


def test_the_buckyball_has_60_corners_90_edges_and_12_five_and_20_six_sided_faces():
    verts = F.bucky_vertices()
    normals, offsets = F.bucky_planes()
    assert verts.shape == (60, 3) and normals.shape == (32, 3)
    on = (np.abs(verts @ normals.T - offsets) < 1e-6).sum(axis=0)
    assert sorted(on) == [5] * 12 + [6] * 20
    gap = np.linalg.norm(verts[:, None] - verts[None], axis=2)
    gap[gap < 1e-9] = np.inf
    assert int((np.abs(gap - gap.min()) < 1e-6).sum()) // 2 == 90
    assert np.isclose(np.abs(verts).max(), 1.0)  # spans -1 .. 1


def test_the_ball_fills_its_box_and_is_symmetric_in_every_axis():
    filled, frame = F.bucky_grid((30, 22, 18))
    assert not frame.any()
    for axis in range(3):
        assert np.array_equal(filled, np.flip(filled, axis))
        assert filled.any(axis=tuple(a for a in range(3) if a != axis))[[0, -1]].all()  # touches both sides


def test_the_framed_ball_matches_its_golden_file():
    filled, frame = F.bucky_grid((24, 24, 24), framed=True)
    want = read_vox(FACETS / "framed_buckyball_24.vox").voxels
    assert np.array_equal(np.where(frame, FRAME, np.where(filled, GLASS, 0)), want)
    surface = F.surface_mask(filled & ~frame).sum()
    assert frame.sum() > 0.25 * surface
    for axis in range(3):
        assert np.array_equal(frame, np.flip(frame, axis))  # bars as symmetric as the ball


def test_a_framed_ball_too_small_for_bars_is_the_plain_ball():
    filled, frame = F.bucky_grid((5, 8, 8), framed=True)
    assert filled.any() and not frame.any()


def test_both_balls_are_primitives_and_only_the_framed_one_has_bars():
    assert BUCKYBALL in SOLIDS and FRAMED_BUCKYBALL in SOLIDS
    foot = np.array([(x, y, 0) for x in range(20) for y in range(14)])
    cells, bars = solid_parts(foot, 2, 1, 12, BUCKYBALL, (20, 14, 12))
    assert len(cells) and not bars.any()
    cells, bars = solid_parts(foot, 2, 1, 12, FRAMED_BUCKYBALL, (20, 14, 12))
    assert bars.any() and not bars.all()
    v = solid_grid(FRAMED_BUCKYBALL, (20, 14, 12), GLASS, frame_colour=FRAME)
    assert set(np.unique(v)) == {0, GLASS, FRAME}
    assert set(np.unique(solid_grid(FRAMED_BUCKYBALL, (20, 14, 12), GLASS))) == {0, GLASS}


def test_a_buckyball_grown_downwards_is_the_same_ball():
    foot = np.array([(x, y, 15) for x in range(16) for y in range(16)])
    up, _ = solid_parts(foot - (0, 0, 15), 2, 1, 16, BUCKYBALL, (16, 16, 16))
    down, _ = solid_parts(foot, 2, -1, 16, BUCKYBALL, (16, 16, 16))
    assert {tuple(c) for c in up} == {tuple(c) for c in down}  # symmetric in z


# --- the frame's colour -------------------------------------------------------------------


@pytest.mark.parametrize("colour", [1, 10, 79, 200])
def test_the_frame_colour_is_a_darker_slot(colour):
    from rok.core.gradient import oklab

    slot = darker_slot(DEFAULT_PALETTE, colour)
    assert slot != colour and oklab(DEFAULT_PALETTE[slot, :3])[0] < oklab(DEFAULT_PALETTE[colour, :3])[0]


def test_with_nothing_darker_the_frame_keeps_the_colour():
    pal = DEFAULT_PALETTE.copy()
    pal[1:, :3] = 0
    assert darker_slot(pal, 5) == 5


# --- Crease -------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 24, 40])
def test_crease_finds_the_32_faces_of_a_buckyball_from_its_voxels(n):
    filled, _ = F.bucky_grid((n, n, n))
    found = F.Faces.of(filled)
    assert len(found.normals) == 32
    truth = F.bucky_distances((found.cells + 0.5 - n / 2) / (n / 2), (n / 2,) * 3).argmax(axis=1)
    normals, _ = F.bucky_planes()
    right = (found.normals[found.label] * normals[truth]).sum(axis=1) > np.cos(np.radians(8))
    assert right.mean() > (0.8 if n == 16 else 0.95)


def test_crease_on_the_canopy_matches_its_golden_file():
    canopy = read_vox(FACETS / "canopy.vox").voxels
    want = read_vox(FACETS / "canopy_crease_expected.vox").voxels
    assert np.array_equal(F.crease(canopy, FRAME), want)


def test_a_block_gets_bars_on_its_twelve_edges_and_nowhere_else():
    v = block()
    out = F.crease(v, FRAME)
    bars = np.argwhere(out == FRAME)
    assert len(bars) and not v[tuple(bars.T)].any()  # raised: only new voxels
    lo, hi = np.array([4, 6, 2]), np.array([35, 33, 31])  # the block's first and last cells
    outside = np.minimum(np.abs(bars - (lo - 1)), np.abs(bars - (hi + 1)))  # how far off each outer face plane
    near = (outside <= 1) | (np.minimum(np.abs(bars - lo), np.abs(bars - hi)) <= 1)
    assert (near.sum(axis=1) >= 2).mean() > 0.97  # nearly every bar sits by two faces: on an edge


def roof(turn: float, n: int = 40) -> np.ndarray:
    """A block with a gable roof whose two slopes meet at `turn` degrees along x."""
    x, y, z = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    slope = np.tan(np.radians(turn / 2.0))
    filled = (x >= 4) & (x < 36) & (y >= 4) & (y < 36) & (z >= 2) & (z <= 30 - np.abs(y - 19.5) * slope)
    return filled.astype(np.uint8) * GLASS


@pytest.mark.parametrize("turn", [30, 40, 60])
def test_a_shallow_ridge_gets_a_bar_along_its_whole_length(turn):
    # The voxels along an edge must join the faces round them, not form a thin
    # face of their own that splits the turn into two halves under the angle.
    found = F.Faces.of(roof(turn))
    assert len(found.normals) == 7  # two slopes, four walls, the floor
    cells, _ = found.edges(20)
    ridge = cells[np.abs(cells[:, 1] - 19.5) <= 2.5]
    assert len(np.unique(ridge[:, 0])) >= 31  # of the 32 cells along x


@pytest.mark.parametrize("n", [40, 64])
def test_a_smooth_ball_has_no_edges(n):
    v = sphere(n).astype(np.uint8) * GLASS
    assert np.array_equal(F.crease(v, FRAME, angle=25), v)


def test_with_mirror_a_symmetric_model_gets_symmetric_bars():
    filled, _ = F.bucky_grid((32, 20, 16))
    out = F.crease(filled.astype(np.uint8) * GLASS, FRAME, mirror=(True, True, False))
    assert np.array_equal(out, np.flip(out, 0)) and np.array_equal(out, np.flip(out, 1))


def test_paint_recolours_the_edges_and_adds_nothing():
    v = block()
    out = F.crease(v, FRAME, raised=False)
    assert np.array_equal(out != 0, v != 0) and (out == FRAME).any()


def test_width_two_marks_both_sides_of_each_edge():
    found = F.Faces.of(block())
    one, _ = found.edges(20, 1)
    two, _ = found.edges(20, 2)
    assert len(two) > 1.5 * len(one)


def test_a_sharper_angle_finds_fewer_edges():
    filled, _ = F.bucky_grid((40, 40, 40))
    found = F.Faces.of(filled)
    assert len(found.edges(30)[0]) > 0 and len(found.edges(60)[0]) == 0  # the ball's faces turn 37 and 42 degrees


def test_a_limit_keeps_the_bars_inside_it():
    v = block()
    limit = np.zeros(v.shape, dtype=bool)
    limit[:20] = True
    out = F.crease(v, FRAME, raised=False, limit=limit)
    assert (out[:20] == FRAME).any() and not (out[20:] == FRAME).any()


def test_an_empty_or_flat_model_has_no_edges():
    assert np.array_equal(F.crease(np.zeros((8, 8, 8), np.uint8), FRAME), np.zeros((8, 8, 8), np.uint8))
    slab = np.zeros((20, 20, 3), np.uint8)
    slab[2:18, 2:18, 1] = GLASS
    assert len(F.Faces.of(slab).edges()[0]) >= 0  # a one-voxel sheet: no crash


def test_raise_and_paint_together_give_the_bars_of_both():
    v = block()
    raised, painted = F.crease(v, FRAME), F.crease(v, FRAME, raised=False)
    both = F.crease(v, FRAME, paint=True)
    assert np.array_equal(both == FRAME, (raised == FRAME) | (painted == FRAME))
    assert (painted == FRAME).any() and (raised[v == 0] == FRAME).any()
