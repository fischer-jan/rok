"""Box, Line and Center footprints, their labels and the axis drag helper."""

import numpy as np
import pytest

from rok.core.edit import line_cells
from rok.core.shape import (
    BRIDGE_LINES,
    BRIDGE_SOLID,
    BOX,
    CENTER,
    LINE,
    axis_drag_param,
    bridge_cells,
    center_cells,
    line3d_cells,
    line_shape,
    rect_cells,
    shape_cells,
    shape_label,
)

SIZE = (8, 8, 8)


def cell_set(cells):
    return {tuple(int(c) for c in row) for row in np.asarray(cells).reshape(-1, 3)}


def free_axes(axis):
    return tuple(i for i in range(3) if i != axis)


# ---------------------------------------------------------------- rect_cells


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_rect_cells_inclusive_on_each_plane(axis):
    u, v = free_axes(axis)
    a = [3, 3, 3]
    b = [3, 3, 3]
    a[u], a[v] = 1, 2
    b[u], b[v] = 4, 5
    cells = rect_cells(tuple(a), tuple(b), axis)
    assert cells.shape == (4 * 4, 3)
    assert cells.dtype == np.int64
    assert set(cells[:, axis].tolist()) == {3}
    expected = set()
    for i in range(1, 5):
        for j in range(2, 6):
            c = [3, 3, 3]
            c[u], c[v] = i, j
            expected.add(tuple(c))
    assert cell_set(cells) == expected


def test_rect_cells_row_major():
    cells = rect_cells((0, 0, 0), (1, 2, 0), 2)
    assert cells.tolist() == [
        [0, 0, 0], [0, 1, 0], [0, 2, 0],
        [1, 0, 0], [1, 1, 0], [1, 2, 0],
    ]


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_rect_cells_corner_order_independent(axis):
    a, b = (1, 5, 2), (6, 0, 7)
    a = list(a)
    b = list(b)
    b[axis] = a[axis]
    a, b = tuple(a), tuple(b)
    ref = cell_set(rect_cells(a, b, axis))
    assert cell_set(rect_cells(b, a, axis)) == ref
    u, v = free_axes(axis)
    c, d = list(a), list(b)
    c[u], d[u] = d[u], c[u]  # swap one free coordinate: the other two corners
    assert cell_set(rect_cells(tuple(c), tuple(d), axis)) == ref


def test_rect_cells_single_cell():
    cells = rect_cells((2, 3, 4), (2, 3, 4), 1)
    assert cells.tolist() == [[2, 3, 4]]


# -------------------------------------------------------------- center_cells


def test_center_cells_symmetric():
    cells = center_cells((4, 4, 0), (6, 5, 0), 2, SIZE)
    assert cells.shape == (5 * 3, 3)
    xs, ys = cells[:, 0], cells[:, 1]
    assert xs.min() == 2 and xs.max() == 6
    assert ys.min() == 3 and ys.max() == 5
    # The same footprint from the mirrored corner.
    assert cell_set(center_cells((4, 4, 0), (2, 3, 0), 2, SIZE)) == cell_set(cells)


def test_center_cells_clipped_at_border():
    cells = center_cells((1, 6, 0), (4, 7, 0), 2, SIZE)
    xs, ys = cells[:, 0], cells[:, 1]
    assert xs.min() == 0 and xs.max() == 4
    assert ys.min() == 5 and ys.max() == 7
    assert cells.shape == (5 * 3, 3)


def test_center_cells_single_cell():
    assert center_cells((3, 3, 3), (3, 3, 3), 0, SIZE).tolist() == [[3, 3, 3]]


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_center_cells_stays_in_plane(axis):
    c = (3, 4, 5)
    b = list(c)
    for i in free_axes(axis):
        b[i] += 2
    cells = center_cells(c, tuple(b), axis, SIZE)
    assert set(cells[:, axis].tolist()) == {c[axis]}
    assert cells.shape == (25, 3)


# ---------------------------------------------------------------- line_shape


def test_line_shape_starts_with_a_and_ends_with_b():
    a, b = (0, 0, 0), (5, 2, 0)
    cells = line_shape(a, b, 2)
    assert tuple(cells[0]) == a
    assert tuple(cells[-1]) == b
    assert cells.tolist() == [list(a)] + [list(c) for c in line_cells(a, b, 2)]


@pytest.mark.parametrize("b", ((7, 1, 0), (1, 7, 0), (7, 7, 0), (0, 0, 0), (6, 3, 0)))
def test_line_shape_has_no_gaps(b):
    cells = line_shape((0, 0, 0), b, 2)
    for prev, cur in zip(cells[:-1], cells[1:]):
        assert np.abs(cur - prev).max() == 1


def test_line_shape_single_cell():
    cells = line_shape((3, 3, 3), (3, 3, 3), 0)
    assert cells.shape == (1, 3)
    assert cells.tolist() == [[3, 3, 3]]


# ---------------------------------------------------------------- line3d_cells


@pytest.mark.parametrize(
    "b", ((7, 0, 0), (0, 7, 0), (0, 0, 7), (7, 7, 7), (7, 3, 1), (2, 7, 5), (0, 0, 0))
)
def test_line3d_starts_with_a_ends_with_b_and_has_no_gaps(b):
    cells = line3d_cells((0, 0, 0), b)
    assert cells[0].tolist() == [0, 0, 0]
    assert cells[-1].tolist() == list(b)
    assert len(cells) == max(b) + 1
    for prev, cur in zip(cells[:-1], cells[1:]):
        assert np.abs(cur - prev).max() == 1


def test_line3d_runs_in_any_direction():
    down = line3d_cells((5, 5, 6), (5, 5, 1))
    assert down.tolist() == [[5, 5, z] for z in range(6, 0, -1)]
    back = line3d_cells((4, 6, 2), (1, 2, 2))
    assert back[0].tolist() == [4, 6, 2] and back[-1].tolist() == [1, 2, 2]
    assert all(z == 2 for z in back[:, 2].tolist())
    assert len(set(map(tuple, back.tolist()))) == len(back)


# --------------------------------------------------------------- shape_cells


def test_shape_cells_dispatch():
    a, b = (2, 2, 0), (4, 3, 0)
    assert np.array_equal(shape_cells(BOX, a, b, 2, SIZE), rect_cells(a, b, 2))
    assert np.array_equal(
        shape_cells(CENTER, a, b, 2, SIZE), center_cells(a, b, 2, SIZE)
    )
    assert np.array_equal(shape_cells(LINE, a, b, 2, SIZE), line3d_cells(a, b))
    with pytest.raises(ValueError):
        shape_cells("VOXEL", a, b, 2, SIZE)


# --------------------------------------------------------------- shape_label


def test_shape_label_texts():
    a, b = (2, 2, 0), (4, 5, 0)
    assert shape_label(BOX, a, b, 2) == "3 x 4"
    assert shape_label(BOX, b, a, 2) == "3 x 4"
    assert shape_label(CENTER, a, b, 2) == "5 x 7"
    assert shape_label(LINE, a, b, 2) == str(len(line3d_cells(a, b)))
    assert shape_label(BOX, a, a, 2) == "1 x 1"
    assert shape_label(CENTER, a, a, 2) == "1 x 1"
    assert shape_label(LINE, a, a, 2) == "1"


def test_shape_label_uses_the_free_axes():
    # Plane axis 0: the label is y by z.
    assert shape_label(BOX, (3, 1, 1), (3, 4, 2), 0) == "4 x 2"


# ----------------------------------------------------------- axis_drag_param


CENTRE = (2.5, 2.5, 3.0)
NORMAL = (0.0, 0.0, 1.0)


def test_axis_drag_param_perpendicular_through_centre():
    s = axis_drag_param(CENTRE, NORMAL, (10.0, 2.5, 3.0), (-1.0, 0.0, 0.0))
    assert s == pytest.approx(0.0)


def test_axis_drag_param_offset_along_normal():
    s = axis_drag_param(CENTRE, NORMAL, (10.0, 2.5, 5.0), (-1.0, 0.0, 0.0))
    assert s == pytest.approx(2.0)
    s = axis_drag_param(CENTRE, NORMAL, (10.0, 2.5, 1.0), (-1.0, 0.0, 0.0))
    assert s == pytest.approx(-2.0)


def test_axis_drag_param_parallel_is_none():
    assert axis_drag_param(CENTRE, NORMAL, (2.5, 2.5, 20.0), (0.0, 0.0, -1.0)) is None


def test_axis_drag_param_oblique_ray():
    # A ray that passes through centre + 3 * normal at an angle.
    point = (2.5, 2.5, 6.0)
    direction = (0.6, -0.3, -0.5)
    origin = tuple(point[i] - 4.0 * direction[i] for i in range(3))
    s = axis_drag_param(CENTRE, NORMAL, origin, direction)
    assert isinstance(s, float)
    assert s == pytest.approx(3.0)


def test_bridge_fills_the_beam_between_two_facing_faces():
    a = np.array([[0, y, z] for y in range(2, 4) for z in range(2, 4)])  # a 2 x 2 face at x 0
    b = a.copy()
    b[:, 0] = 5
    cells = bridge_cells(a, b, (8, 8, 8))
    got = cell_set(cells)
    beam = {(x, y, z) for x in range(1, 5) for y in range(2, 4) for z in range(2, 4)}
    assert got == beam
    assert not ({tuple(c) for c in a.tolist()} | {tuple(c) for c in b.tolist()}) & got
    # Touching parts have nothing between them; the grid clips the slab.
    assert len(bridge_cells(np.array([[0, 0, 0]]), np.array([[1, 0, 0]]), (8, 8, 8))) == 0
    assert len(bridge_cells(np.array([[0, 0, 0]]), np.array([[3, 0, 0]]), (2, 2, 2))) == 1


def test_bridge_default_is_solid_and_lines_is_the_old_fill():
    a = np.array([[0, y, z] for y in range(2, 4) for z in range(2, 4)])
    b = a.copy()
    b[:, 0] = 5
    solid = cell_set(bridge_cells(a, b, SIZE))
    assert solid == cell_set(bridge_cells(a, b, SIZE, kind=BRIDGE_SOLID))
    lines = cell_set(bridge_cells(a, b, SIZE, kind=BRIDGE_LINES))
    beam = {(x, y, z) for x in range(1, 5) for y in range(2, 4) for z in range(2, 4)}
    assert beam <= lines
    with pytest.raises(ValueError, match="not a bridge kind"):
        bridge_cells(a, b, SIZE, kind="CONE")


def test_bridge_slab_is_commutative_and_leaves_the_parts_out():
    a = np.array([[1, y, z] for y in range(1, 3) for z in range(0, 2)])
    b = np.array([[6, y, z] for y in range(1, 3) for z in range(0, 2)])
    assert cell_set(bridge_cells(a, b, SIZE)) == cell_set(bridge_cells(b, a, SIZE))
    got = cell_set(bridge_cells(a, b, SIZE))
    assert not cell_set(a) & got
    assert not cell_set(b) & got


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_bridge_slab_faces_along_each_axis(axis):
    a = np.zeros((4, 3), dtype=np.int64)
    b = np.zeros((4, 3), dtype=np.int64)
    u, v = free_axes(axis)
    k = 0
    for i in range(2):
        for j in range(2):
            a[k, axis], a[k, u], a[k, v] = 1, 2 + i, 3 + j
            b[k, axis], b[k, u], b[k, v] = 6, 2 + i, 3 + j
            k += 1
    got = cell_set(bridge_cells(a, b, SIZE))
    expected = set()
    for x in range(2, 6):
        for i in range(2):
            for j in range(2):
                c = [0, 0, 0]
                c[axis], c[u], c[v] = x, 2 + i, 3 + j
                expected.add(tuple(c))
    assert got == expected


def test_bridge_slab_steps_height_and_keeps_flat_side_walls():
    # Same Y extent, taller on the far side: side walls stay y=2..3; z steps.
    a = np.array([[0, y, z] for y in range(2, 4) for z in range(0, 2)])
    b = np.array([[6, y, z] for y in range(2, 4) for z in range(0, 5)])
    got = cell_set(bridge_cells(a, b, (10, 10, 10)))
    assert got
    assert all(2 <= y <= 3 for _, y, _ in got)
    assert all(1 <= x <= 5 for x, _, _ in got)
    xs = sorted({x for x, _, _ in got})
    assert xs == [1, 2, 3, 4, 5]
    assert {z for x, _, z in got if x == xs[0]} == {0, 1}
    assert {z for x, _, z in got if x == xs[-1]} == {0, 1, 2, 3, 4}
    # Height only grows toward the taller side; no holes in a slice's z run.
    prev_hi = 1
    for x in xs:
        zs = sorted({z for xx, _, z in got if xx == x})
        assert zs == list(range(zs[0], zs[-1] + 1))
        assert zs[-1] >= prev_hi
        prev_hi = zs[-1]


def test_bridge_slab_steps_when_the_sides_are_offset():
    a = np.array([[0, 0, z] for z in range(2)])
    b = np.array([[5, 3, z] for z in range(2)])
    got = cell_set(bridge_cells(a, b, SIZE))
    # Loft along X; Y steps 0→3; Z stays a 2-cell slab.
    assert all(z in (0, 1) for _, _, z in got)
    ys = [y for _, y, _ in sorted(got)]
    assert ys[0] == 0 and ys[-1] == 3
    lines = cell_set(bridge_cells(a, b, SIZE, kind=BRIDGE_LINES))
    assert got != lines


def test_bridge_empty_parts_and_kind_lines_clip():
    empty = np.zeros((0, 3), dtype=np.int64)
    assert len(bridge_cells(empty, np.array([[1, 0, 0]]), SIZE)) == 0
    assert len(bridge_cells(np.array([[0, 0, 0]]), empty, SIZE, kind=BRIDGE_LINES)) == 0
    assert len(bridge_cells(np.array([[0, 0, 0]]), np.array([[3, 0, 0]]), (2, 2, 2), kind=BRIDGE_LINES)) == 1



# --- the primitives -------------------------------------------------------------

from rok.core.shape import (  # noqa: E402
    CAPSULE,
    CYLINDER,
    DIAMOND,
    HALF_CYLINDER,
    HILL,
    OCTAGON_PRISM,
    ROCK,
    WALL_MAX,
    DOME,
    HEX_PRISM,
    ROOF,
    SOLIDS,
    SPHERE,
    STAIRS,
    TORUS,
    TUBE,
    WEDGE,
    solid_cells,
)

GRID = (40, 40, 40)


def primitive(kind, nx=9, ny=5, height=6, wall=0, seed=0):
    """The solid on a footprint nx by ny at z = 0, growing up: a bool block."""
    footprint = np.array([(x, y, 0) for x in range(nx) for y in range(ny)])
    cells = solid_cells(footprint, 2, 1, height, kind, GRID, wall=wall, seed=seed)
    block = np.zeros((nx, ny, height), dtype=bool)
    block[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    assert len(cells) == block.sum()  # no cell twice
    return block


def test_every_primitive_fills_part_of_its_box_and_touches_the_base():
    for kind in SOLIDS:
        block = primitive(kind)
        assert block.any() and block[:, :, 0].any()


def test_every_primitive_builds_something_on_the_smallest_bases():
    """A click is a 1 x 1 base: it must build at least a voxel, or the click
    does nothing (the torus did exactly that)."""
    for kind in SOLIDS:
        for nx in range(1, 5):
            for ny in range(1, 5):
                for height in (1, 2, 3, 6):
                    assert primitive(kind, nx, ny, height).any(), (kind, nx, ny, height)


def test_every_primitive_grows_down_as_well_as_up():
    for kind in SOLIDS:
        footprint = np.array([(x, y, 20) for x in range(9) for y in range(5)])
        up = solid_cells(footprint, 2, 1, 6, kind, GRID)
        down = solid_cells(footprint, 2, -1, 6, kind, GRID)
        flipped = up.copy()
        flipped[:, 2] = 40 - flipped[:, 2]
        assert sorted(map(tuple, down)) == sorted(map(tuple, flipped))


def test_every_primitive_is_clipped_to_the_grid():
    for kind in SOLIDS:
        footprint = np.array([(x, y, 38) for x in range(9) for y in range(5)])
        cells = solid_cells(footprint, 2, 1, 6, kind, GRID)
        assert (cells >= 0).all() and (cells < np.array(GRID)).all()


def test_a_sphere_is_symmetric_and_widest_in_the_middle():
    block = primitive(SPHERE, 9, 9, 9)
    assert np.array_equal(block, block[::-1, :, :]) and np.array_equal(block, block[:, :, ::-1])
    counts = block.sum(axis=(0, 1))
    assert counts[4] == counts.max() and counts[0] < counts[4]


def test_a_dome_is_widest_at_its_base_and_narrows_up():
    counts = primitive(DOME, 9, 9, 5).sum(axis=(0, 1))
    assert all(a >= b for a, b in zip(counts, counts[1:])) and counts[0] > counts[-1]


def test_a_wedge_rises_along_the_longer_side_to_full_height_at_the_back():
    block = primitive(WEDGE, 9, 5, 6)
    heights = block.sum(axis=2)[:, 0]
    assert all(a <= b for a, b in zip(heights, heights[1:]))
    assert heights[-1] == 6 and heights[0] < 6
    assert (block == block[:, :1, :]).all()  # the same across the shorter side


def test_a_roof_has_its_ridge_along_the_longer_side():
    block = primitive(ROOF, 9, 7, 4)
    top = block[:, :, -1]
    assert top.any(axis=1).all()  # the ridge runs the whole length
    assert top[:, 3].all() and not top[:, 0].any()  # in the middle of the shorter side
    assert np.array_equal(block, block[:, ::-1, :])


def test_a_tube_is_hollow_down_the_middle_and_full_at_its_wall():
    block = primitive(TUBE, 9, 9, 4)
    assert not block[4, 4, :].any()
    assert block[0, 4, :].all() and block[8, 4, :].all()
    assert np.array_equal(block[:, :, 0], block[:, :, -1])


def test_a_thin_tube_has_no_hole():
    assert primitive(TUBE, 2, 2, 3).all()


def test_a_hex_prism_cuts_the_corners_and_keeps_the_middle():
    layer = primitive(HEX_PRISM, 9, 7, 3)[:, :, 0]
    assert layer[4, :].all()  # across the middle
    assert not layer[0, 0] and not layer[-1, -1]  # the corners are cut
    assert layer[0, 3] and layer[-1, 3]  # the points on the longer axis


def test_a_flat_torus_has_a_hole_and_a_ring():
    block = primitive(TORUS, 13, 13, 3)
    assert not block[6, 6, :].any()  # the hole
    assert block[1, 6, 1] and block[11, 6, 1]  # the ring on both sides


@pytest.mark.parametrize("n,height", [(10, 3), (16, 2), (16, 6), (24, 8)])
def test_a_torus_hole_is_wide_enough_to_see(n, height):
    """Voxels round the hole inward; it must still be about a third of the base
    across, or the torus reads as a disc."""
    block = primitive(TORUS, n, n, height)
    middle = block[:, n // 2, :].any(axis=1)  # the row through the centre, seen from above
    hole = int((~middle).sum())
    assert hole >= n // 3
    assert middle[0] and middle[-1]  # the ring reaches the base's edge


def test_stairs_climb_in_steps_along_the_longer_side():
    block = primitive(STAIRS, 8, 3, 4)
    heights = block.sum(axis=2)[:, 0]
    assert all(a <= b for a, b in zip(heights, heights[1:]))
    assert heights[-1] == 4
    assert len(set(heights.tolist())) == 4  # four steps of two voxels each



def test_a_standing_capsule_is_round_at_both_ends_and_straight_between():
    block = primitive(CAPSULE, 7, 7, 15)
    counts = block.sum(axis=(0, 1))
    assert counts[0] < counts[7] and counts[-1] < counts[7]  # round ends
    assert counts[5] == counts[7] == counts[9]  # a straight middle
    assert np.array_equal(block, block[:, :, ::-1])


def test_a_long_capsule_lies_along_the_longer_side():
    block = primitive(CAPSULE, 15, 5, 5)
    per_x = block.sum(axis=(1, 2))
    assert per_x[0] < per_x[7] and per_x[-1] < per_x[7]


def test_a_half_cylinder_is_a_vault_along_the_longer_side():
    block = primitive(HALF_CYLINDER, 11, 7, 4)
    assert (block == block[:1]).all()  # the same all along x
    top = block[:, :, -1]
    assert top[:, 3].all() and not top[:, 0].any()  # the crown in the middle
    assert block[:, :, 0].all()  # flat on the footprint


def test_a_diamond_is_widest_in_the_middle_and_pointed_at_both_ends():
    counts = primitive(DIAMOND, 9, 9, 9).sum(axis=(0, 1))
    assert counts[4] == counts.max() and counts[0] < counts[4] and counts[-1] < counts[4]
    assert counts[0] == counts[-1]


def test_an_octagon_cuts_the_corners_and_keeps_the_sides():
    layer = primitive(OCTAGON_PRISM, 9, 9, 2)[:, :, 0]
    assert not layer[0, 0] and not layer[8, 8]
    assert layer[0, 4] and layer[4, 0] and layer[8, 4]
    assert layer[1, 1] or layer[2, 1]  # only the corner itself goes


@pytest.mark.parametrize("kind", [ROCK, HILL])
def test_a_lumpy_shape_repeats_with_its_seed_and_changes_with_another(kind):
    a = primitive(kind, 12, 12, 8, seed=3)
    assert np.array_equal(a, primitive(kind, 12, 12, 8, seed=3))
    assert not np.array_equal(a, primitive(kind, 12, 12, 8, seed=4))
    assert a[:, :, 0].any()


def test_hollow_keeps_a_shell_inside_the_solid():
    for kind in SOLIDS:
        solid = primitive(kind, 11, 11, 9, seed=1)
        thin = primitive(kind, 11, 11, 9, wall=1, seed=1)
        thick = primitive(kind, 11, 11, 9, wall=2, seed=1)
        assert np.array_equal(primitive(kind, 11, 11, 9, wall=0, seed=1), solid)
        assert not (thin & ~solid).any() and not (thin & ~thick).any()  # shells nest in the solid
        assert thin.any()


def test_a_hollow_sphere_is_empty_inside_and_whole_outside():
    solid = primitive(SPHERE, 11, 11, 11)
    shell = primitive(SPHERE, 11, 11, 11, wall=1)
    assert solid[5, 5, 5] and not shell[5, 5, 5]
    assert shell[0, 5, 5] and shell[5, 5, 0] and shell[5, 5, 10]  # closed top and bottom


def test_the_wall_stops_at_its_most():
    assert np.array_equal(primitive(SPHERE, 21, 21, 21, wall=WALL_MAX + 5), primitive(SPHERE, 21, 21, 21, wall=WALL_MAX))


@pytest.mark.parametrize("kind", [SPHERE, CYLINDER, DOME, CAPSULE, DIAMOND])
def test_a_hollow_wall_is_watertight(kind):
    """From outside, moving face to face through empty cells, the hollow inside
    is never reached: no two wall cells meet at an edge only."""
    shell = primitive(kind, 13, 13, 13, wall=1)
    if not (primitive(kind, 13, 13, 13) & ~shell).any():
        return  # nothing hollow to leak into
    empty = np.pad(~shell, 1, constant_values=True)
    reached = np.zeros(empty.shape, dtype=bool)
    reached[0] = reached[-1] = reached[:, 0] = reached[:, -1] = reached[:, :, 0] = reached[:, :, -1] = True
    reached &= empty
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown[:, :, 1:] |= reached[:, :, :-1]
        grown[:, :, :-1] |= reached[:, :, 1:]
        grown &= empty
        if np.array_equal(grown, reached):
            break
        reached = grown
    inside = primitive(kind, 13, 13, 13) & ~shell
    assert not reached[1:-1, 1:-1, 1:-1][inside].any()


def test_a_new_shape_model_fills_its_grid_and_stands_on_the_floor():
    for kind in SOLIDS:
        from rok.core.shape import solid_grid

        v = solid_grid(kind, (12, 10, 8), 5)
        assert v.shape == (12, 10, 8) and set(np.unique(v)) <= {0, 5}
        assert v[:, :, 0].any()  # it stands on its bottom layer
