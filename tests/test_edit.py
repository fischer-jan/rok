"""Tool rules, plane lock, mirror, line fill and strokes with an injected setter."""

import numpy as np
import pytest

from rok.core.edit import (
    ACTIVATE,
    ATTACH,
    ERASE,
    LASSO,
    MODE_ITEMS,
    MODES,
    PAINT,
    RECT,
    SCREEN_MODES,
    SELECT,
    TOOL_ITEMS,
    TOOLS,
    PlaneLock,
    Stroke,
    apply_cells,
    apply_tool,
    clamp_free,
    clamp_to_border,
    effective_tool,
    in_bounds,
    in_bounds_mask,
    line_cells,
    mirror_cells,
    mirror_images,
    modes_for,
    plane_cell,
    plane_crossing,
    select_extending,
    select_subtract,
    target_cell,
)
from rok.core.raycast import Hit, raycast
from rok.core.vox_read import read_vox

from helpers import FIXTURES, ArraySetMany, ArraySetter, filled, top_ray

SIZE = (8, 8, 8)


def make_stroke(voxels, tool, colour=5, mirror=(False, False, False)):
    setter = ArraySetter(voxels)
    return Stroke(voxels, tool, colour, setter, mirror=mirror), setter


# ---------------------------------------------------------------- tool rules


def test_tool_constants():
    assert len(set(TOOLS)) == len(TOOLS)
    assert [item[0] for item in TOOL_ITEMS] == list(TOOLS)
    for ident, label, tip in TOOL_ITEMS:
        assert label and tip
    assert len({label for _ident, label, _tip in TOOL_ITEMS}) == len(TOOL_ITEMS)
    assert {ident: label for ident, label, _tip in TOOL_ITEMS}[SELECT] == "Select"


def test_mode_constants():
    from rok.core.edit import CLONE, GRADIENT, SELECT_ONLY_MODES

    assert len(set(MODES)) == len(MODES)
    # The wheel's modes, Attach's Clone, then Select's own (their flyouts only).
    assert [item[0] for item in MODE_ITEMS] == list(MODES) + [CLONE, GRADIENT] + list(SELECT_ONLY_MODES)
    assert not set(SELECT_ONLY_MODES) & set(MODES) and CLONE not in MODES and GRADIENT not in MODES
    for ident, label, tip in MODE_ITEMS:
        assert label and tip
    assert len({label for _ident, label, _tip in MODE_ITEMS}) == len(MODE_ITEMS)


def test_screen_modes_and_activate():
    assert SCREEN_MODES == (RECT, LASSO)
    for mode in SCREEN_MODES:
        assert mode in MODES
    # Attach has no screen modes; the other actions have every mode. The groups cover all.
    assert modes_for(ATTACH) == tuple(m for m in MODES if m not in SCREEN_MODES)
    for tool in (ERASE, PAINT, SELECT):
        assert modes_for(tool) == MODES
    # A hover-only tool name: never a real tool.
    assert ACTIVATE == "ACTIVATE"
    assert ACTIVATE not in TOOLS


@pytest.mark.parametrize(
    "base, alt, shift, expected",
    [
        (ATTACH, False, False, ATTACH),
        (ATTACH, True, False, ERASE),
        (ATTACH, False, True, PAINT),
        (ATTACH, True, True, ERASE),
        (ERASE, False, False, ERASE),
        (ERASE, False, True, PAINT),
        (PAINT, False, False, PAINT),
        (PAINT, True, False, ERASE),
        (SELECT, False, False, SELECT),
        (SELECT, False, True, SELECT),
        (SELECT, True, False, SELECT),  # Option in Select deselects; it never erases
        (SELECT, True, True, SELECT),
    ],
)
def test_effective_tool(base, alt, shift, expected):
    assert effective_tool(base, alt, shift) == expected


def test_effective_tool_ctrl_is_not_a_parameter():
    import inspect

    params = inspect.signature(effective_tool).parameters
    assert "alt" in params and "shift" in params
    assert "ctrl" not in params
    params = inspect.signature(select_subtract).parameters
    assert "alt" in params and "ctrl" not in params


@pytest.mark.parametrize(
    "base, alt, shift, expected",
    [
        (SELECT, True, True, True),  # Option+Shift subtracts
        (SELECT, False, True, False),  # add
        (SELECT, True, False, True),  # Option alone deselects too; Select never erases
        (SELECT, False, False, False),
        (ATTACH, True, True, False),  # not Select: effective_tool is ERASE, never subtract
    ],
)
def test_select_subtract(base, alt, shift, expected):
    tool = effective_tool(base, alt, shift)
    assert select_subtract(tool, alt, shift) == expected


@pytest.mark.parametrize(
    "tool, shift, subtract, expected",
    [
        (SELECT, False, False, False),  # plain Select: cage handles stay live
        (SELECT, True, False, True),  # Shift add
        (SELECT, True, True, True),  # Option+Shift subtract
        (SELECT, False, True, True),
        (ATTACH, True, False, False),
        (ERASE, True, False, False),
        (PAINT, True, False, False),
    ],
)
def test_select_extending(tool, shift, subtract, expected):
    assert select_extending(tool, shift, subtract) is expected


def test_in_bounds():
    assert in_bounds((0, 0, 0), SIZE)
    assert in_bounds((7, 7, 7), SIZE)
    assert not in_bounds((8, 0, 0), SIZE)
    assert not in_bounds((0, -1, 0), SIZE)
    assert not in_bounds((0, 0, -1), SIZE)
    assert not in_bounds((3, 3, -2), SIZE)


def test_in_bounds_mask():
    cells = np.array(
        [(0, 0, 0), (7, 7, 7), (8, 0, 0), (0, -1, 0), (0, 0, -1), (3, 3, 8), (2, 5, 1)]
    )
    mask = in_bounds_mask(cells, SIZE)
    assert mask.dtype == bool
    assert mask.shape == (7,)
    assert mask.tolist() == [True, True, False, False, False, False, True]
    assert in_bounds_mask(np.zeros((0, 3), dtype=np.int64), SIZE).shape == (0,)


def test_target_cell_none_hit():
    for tool in TOOLS:
        assert target_cell(tool, None, SIZE) is None


def test_target_cell_attach_on_voxel():
    hit = Hit((3, 3, 3), (1, 0, 0), 1.0)
    assert target_cell(ATTACH, hit, SIZE) == (4, 3, 3)


def test_target_cell_attach_on_floor_places_at_z0():
    hit = Hit((3, 3, -1), (0, 0, 1), 1.0, wall=True)
    assert target_cell(ATTACH, hit, SIZE) == (3, 3, 0)


@pytest.mark.parametrize(
    "voxel, normal, expected",
    [
        ((8, 3, 3), (-1, 0, 0), (7, 3, 3)),
        ((-1, 3, 3), (1, 0, 0), (0, 3, 3)),
        ((3, 8, 3), (0, -1, 0), (3, 7, 3)),
        ((3, -1, 3), (0, 1, 0), (3, 0, 3)),
        ((3, 3, 8), (0, 0, -1), (3, 3, 7)),
    ],
)
def test_target_cell_attach_on_each_wall(voxel, normal, expected):
    hit = Hit(voxel, normal, 1.0, wall=True)
    assert target_cell(ATTACH, hit, SIZE) == expected


def test_target_cell_attach_out_of_grid():
    hit = Hit((7, 3, 3), (1, 0, 0), 1.0)
    assert target_cell(ATTACH, hit, SIZE) is None
    below = Hit((3, 3, -1), (0, 0, -1), 1.0, wall=True)
    assert target_cell(ATTACH, below, SIZE) is None


def test_target_cell_erase_and_paint():
    hit = Hit((3, 3, 3), (0, 0, 1), 1.0)
    assert target_cell(ERASE, hit, SIZE) == (3, 3, 3)
    assert target_cell(PAINT, hit, SIZE) == (3, 3, 3)
    floor = Hit((3, 3, -1), (0, 0, 1), 1.0, wall=True)
    assert target_cell(ERASE, floor, SIZE) is None
    assert target_cell(PAINT, floor, SIZE) is None
    wall = Hit((8, 3, 3), (-1, 0, 0), 1.0, wall=True)
    assert target_cell(ERASE, wall, SIZE) is None
    assert target_cell(PAINT, wall, SIZE) is None


def test_target_cell_select():
    hit = Hit((3, 3, 3), (0, 0, 1), 1.0)
    assert target_cell(SELECT, hit, SIZE) == (3, 3, 3)
    floor = Hit((3, 3, -1), (0, 0, 1), 1.0, wall=True)
    assert target_cell(SELECT, floor, SIZE) == (3, 3, 0)
    wall = Hit((8, 3, 3), (-1, 0, 0), 1.0, wall=True)
    assert target_cell(SELECT, wall, SIZE) == (7, 3, 3)


def test_clamp_free():
    assert clamp_free((100, -5, 4), 2, SIZE) == (7, 0, 4)
    assert clamp_free((3, 3, 4), 2, SIZE) == (3, 3, 4)
    assert clamp_free((-1, 99, 99), 0, SIZE) == (-1, 7, 7)
    assert clamp_free((8, -3, 8), 1, SIZE) == (7, -3, 7)


# ---------------------------------------------------------------- plane_cell


@pytest.mark.parametrize(
    "axis, plane, layer, origin, direction, expected",
    [
        # +X face of voxel (3, 3, 3): plane x == 4, layer 4.
        (0, 4.0, 4, (10.0, 3.5, 3.5), (-1.0, 0.0, 0.0), (4, 3, 3)),
        # -X face: plane x == 3, layer 2.
        (0, 3.0, 2, (-5.0, 3.5, 3.5), (1.0, 0.0, 0.0), (2, 3, 3)),
        # +Y face: plane y == 4, layer 4.
        (1, 4.0, 4, (3.5, 10.0, 3.5), (0.0, -1.0, 0.0), (3, 4, 3)),
        # -Y face: plane y == 3, layer 2.
        (1, 3.0, 2, (3.5, -5.0, 3.5), (0.0, 1.0, 0.0), (3, 2, 3)),
        # +Z face: plane z == 4, layer 4.
        (2, 4.0, 4, (3.5, 3.5, 10.0), (0.0, 0.0, -1.0), (3, 3, 4)),
        # -Z face: plane z == 3, layer 2.
        (2, 3.0, 2, (3.5, 3.5, 0.5), (0.0, 0.0, 1.0), (3, 3, 2)),
    ],
)
def test_plane_cell_six_faces(axis, plane, layer, origin, direction, expected):
    assert plane_cell(origin, direction, axis, plane, layer, SIZE) == expected


def test_plane_cell_oblique_ray():
    # Crosses z == 4 at t == 6: x == 3.5 + 6 * 0.25 == 5.0, y == 3.5 - 6 * 0.1 == 2.9.
    cell = plane_cell((3.5, 3.5, 10.0), (0.25, -0.1, -1.0), 2, 4.0, 4, SIZE)
    assert cell == (5, 2, 4)


def test_plane_cell_parallel_ray():
    assert plane_cell((3.5, 3.5, 10.0), (1.0, 0.0, 0.0), 2, 4.0, 4, SIZE) is None


def test_plane_cell_crossing_behind_origin():
    assert plane_cell((3.5, 3.5, 2.0), (0.0, 0.0, -1.0), 2, 4.0, 4, SIZE) is None
    assert plane_cell((3.5, 3.5, 4.0), (0.0, 0.0, -1.0), 2, 4.0, 4, SIZE) is None


def test_plane_cell_out_of_bounds():
    assert plane_cell((9.5, 3.5, 10.0), (0.0, 0.0, -1.0), 2, 4.0, 4, SIZE) is None
    assert plane_cell((-0.5, 3.5, 10.0), (0.0, 0.0, -1.0), 2, 4.0, 4, SIZE) is None
    assert plane_cell((3.5, 3.5, 10.0), (0.0, 0.0, -1.0), 2, 4.0, 8, SIZE) is None
    assert plane_cell((3.5, 3.5, -2.0), (0.0, 0.0, 1.0), 2, 0.0, -1, SIZE) is None


def test_plane_crossing_has_no_bounds_test():
    assert plane_crossing((9.5, 3.5, 10.0), (0.0, 0.0, -1.0), 2, 4.0, 4) == (9, 3, 4)
    assert plane_crossing((-0.5, 3.5, 10.0), (0.0, 0.0, -1.0), 2, 4.0, 4) == (-1, 3, 4)
    assert plane_crossing((3.5, 3.5, 10.0), (1.0, 0.0, 0.0), 2, 4.0, 4) is None
    assert plane_crossing((3.5, 3.5, 2.0), (0.0, 0.0, -1.0), 2, 4.0, 4) is None


def test_clamp_to_border_keeps_the_locked_axis():
    assert clamp_to_border((10000, -50, 4), 2, SIZE) == (8, -1, 4)
    assert clamp_to_border((3, 3, 4), 2, SIZE) == (3, 3, 4)
    assert clamp_to_border((4, 99, -99), 0, SIZE) == (4, 8, -1)


# ---------------------------------------------------------------- PlaneLock


@pytest.mark.parametrize(
    "normal, axis, plane, layer",
    [
        ((1, 0, 0), 0, 4.0, 4),
        ((-1, 0, 0), 0, 3.0, 2),
        ((0, 1, 0), 1, 4.0, 4),
        ((0, -1, 0), 1, 3.0, 2),
        ((0, 0, 1), 2, 4.0, 4),
        ((0, 0, -1), 2, 3.0, 2),
    ],
)
def test_plane_lock_from_hit_six_faces(normal, axis, plane, layer):
    hit = Hit((3, 3, 3), normal, 1.0)
    lock = PlaneLock.from_hit(hit, hit.target)
    assert lock == PlaneLock(axis, plane, layer)
    assert lock.axis == axis and lock.plane == plane and lock.layer == layer


def test_plane_lock_from_hit_floor_and_walls():
    floor = Hit((3, 3, -1), (0, 0, 1), 1.0, wall=True)
    lock = PlaneLock.from_hit(floor, target_cell(ATTACH, floor, SIZE))
    assert lock == PlaneLock(2, 0.0, 0)
    plus_x = Hit((8, 3, 3), (-1, 0, 0), 1.0, wall=True)
    lock = PlaneLock.from_hit(plus_x, target_cell(ATTACH, plus_x, SIZE))
    assert lock == PlaneLock(0, 8.0, 7)
    top = Hit((3, 3, 8), (0, 0, -1), 1.0, wall=True)
    lock = PlaneLock.from_hit(top, target_cell(ATTACH, top, SIZE))
    assert lock == PlaneLock(2, 8.0, 7)


def test_plane_lock_is_frozen_and_cell_at_uses_plane_crossing():
    lock = PlaneLock(2, 4.0, 4)
    with pytest.raises(AttributeError):
        lock.axis = 0
    assert lock.cell_at((3.5, 3.5, 10.0), (0.25, -0.1, -1.0)) == (5, 2, 4)
    assert lock.cell_at((3.5, 3.5, 10.0), (1.0, 0.0, 0.0)) is None
    assert lock.cell_at((9.5, 3.5, 10.0), (0.0, 0.0, -1.0)) == (9, 3, 4)


# ---------------------------------------------------------------- line_cells


def test_line_cells_empty_for_same_cell():
    assert line_cells((2, 3, 4), (2, 3, 4), 2) == []


def test_line_cells_excludes_a_includes_b():
    cells = line_cells((0, 0, 5), (3, 0, 5), 2)
    assert cells == [(1, 0, 5), (2, 0, 5), (3, 0, 5)]


def test_line_cells_keeps_the_locked_axis():
    cells = line_cells((4, 0, 0), (4, 3, 2), 0)
    assert cells[-1] == (4, 3, 2)
    assert all(c[0] == 4 for c in cells)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_line_cells_no_gaps_for_diagonal_jump(axis):
    a = [0, 0, 0]
    b = [40, 40, 40]
    a[axis] = 7
    b[axis] = 7
    b[(axis + 2) % 3] = 17
    a, b = tuple(a), tuple(b)
    cells = line_cells(a, b, axis)
    assert cells[0] != a
    assert cells[-1] == b
    assert len(cells) == 40
    prev = a
    for cell in cells:
        assert max(abs(cell[i] - prev[i]) for i in range(3)) == 1
        assert cell[axis] == 7
        prev = cell
    assert len(set(cells)) == len(cells)


def test_line_cells_backwards():
    cells = line_cells((5, 5, 0), (2, 1, 0), 2)
    assert cells[-1] == (2, 1, 0)
    prev = (5, 5, 0)
    for cell in cells:
        assert max(abs(cell[i] - prev[i]) for i in range(3)) == 1
        prev = cell


# ---------------------------------------------------------------- mirror


def test_mirror_images_no_axes():
    assert mirror_images((1, 2, 3), SIZE, (False, False, False)) == ((1, 2, 3),)


def test_mirror_images_even_size_source_first():
    assert mirror_images((1, 2, 3), SIZE, (True, False, False)) == ((1, 2, 3), (6, 2, 3))
    assert mirror_images((1, 2, 3), SIZE, (False, True, False)) == ((1, 2, 3), (1, 5, 3))
    assert mirror_images((1, 2, 3), SIZE, (False, False, True)) == ((1, 2, 3), (1, 2, 4))
    assert mirror_images((1, 2, 3), SIZE, (True, True, False)) == (
        (1, 2, 3),
        (6, 2, 3),
        (1, 5, 3),
        (6, 5, 3),
    )
    images = mirror_images((1, 2, 3), SIZE, (True, True, True))
    assert len(images) == 8
    assert len(set(images)) == 8
    assert images[0] == (1, 2, 3)
    assert set(images) == {(x, y, z) for x in (1, 6) for y in (2, 5) for z in (3, 4)}


def test_mirror_images_odd_size_centre_maps_to_itself():
    size = (7, 7, 7)
    assert mirror_images((3, 1, 1), size, (True, False, False)) == ((3, 1, 1),)
    assert mirror_images((1, 1, 1), size, (True, False, False)) == ((1, 1, 1), (5, 1, 1))
    assert mirror_images((3, 3, 3), size, (True, True, True)) == ((3, 3, 3),)
    images = mirror_images((3, 1, 0), size, (True, True, True))
    assert images == ((3, 1, 0), (3, 5, 0), (3, 1, 6), (3, 5, 6))


def test_mirror_images_outside_cell_is_returned_as_is():
    # No bounds test here; the caller drops out-of-grid images.
    assert mirror_images((9, 2, 3), SIZE, (True, False, False)) == ((9, 2, 3), (-2, 2, 3))


@pytest.mark.parametrize(
    "size, axes",
    [
        (SIZE, (True, False, False)),
        (SIZE, (True, True, True)),
        ((7, 7, 7), (True, True, True)),
        ((7, 8, 9), (False, True, True)),
        (SIZE, (False, False, False)),
    ],
)
def test_mirror_cells_equals_mirror_images(size, axes):
    cells = np.array([(1, 2, 3), (3, 3, 3), (0, 6, 4), (1, 2, 3)])
    got = mirror_cells(cells, size, axes)
    assert got.ndim == 2 and got.shape[1] == 3
    expected = set()
    for cell in map(tuple, cells.tolist()):
        expected |= set(mirror_images(cell, size, axes))
    assert {tuple(c) for c in got.tolist()} == expected
    assert len(got) == len(expected)
    # Sorted, unique output.
    assert got.tolist() == sorted(got.tolist())


def test_mirror_cells_empty_input():
    got = mirror_cells(np.zeros((0, 3), dtype=np.int64), SIZE, (True, True, True))
    assert got.shape == (0, 3)


# ---------------------------------------------------------------- apply_tool


def test_apply_tool_attach():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    setter = ArraySetter(voxels)
    assert apply_tool(voxels, ATTACH, 3, (1, 1, 1), setter)
    assert voxels[1, 1, 1] == 3
    # Attach into a filled cell writes nothing.
    assert not apply_tool(voxels, ATTACH, 4, (1, 1, 1), setter)
    assert voxels[1, 1, 1] == 3
    assert setter.calls == [(1, 1, 1, 3)]


def test_apply_tool_erase():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 2, 2] = 7
    setter = ArraySetter(voxels)
    assert not apply_tool(voxels, ERASE, 3, (1, 1, 1), setter)
    assert apply_tool(voxels, ERASE, 3, (2, 2, 2), setter)
    assert voxels[2, 2, 2] == 0
    assert setter.calls == [(2, 2, 2, 0)]


def test_apply_tool_paint():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 2, 2] = 7
    setter = ArraySetter(voxels)
    assert not apply_tool(voxels, PAINT, 3, (1, 1, 1), setter)
    # Paint with the same colour writes nothing.
    assert not apply_tool(voxels, PAINT, 7, (2, 2, 2), setter)
    assert apply_tool(voxels, PAINT, 3, (2, 2, 2), setter)
    assert voxels[2, 2, 2] == 3
    assert setter.calls == [(2, 2, 2, 3)]


def test_apply_tool_select_and_out_of_bounds_write_nothing():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 2, 2] = 7
    setter = ArraySetter(voxels)
    assert not apply_tool(voxels, SELECT, 3, (2, 2, 2), setter)
    assert not apply_tool(voxels, SELECT, 3, (1, 1, 1), setter)
    for tool in TOOLS:
        assert not apply_tool(voxels, tool, 3, (1, 1, -1), setter)
        assert not apply_tool(voxels, tool, 3, (8, 1, 1), setter)
    assert setter.calls == []


# ---------------------------------------------------------------- apply_cells


def test_apply_cells_attach_skips_filled_cells():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 0] = 9
    set_many = ArraySetMany(voxels)
    cells = np.array([(0, 0, 0), (1, 1, 0), (2, 2, 0), (2, 2, 0)])
    assert apply_cells(voxels, ATTACH, 3, cells, set_many) == 2
    assert len(set_many.calls) == 1
    assert set_many.cells == {(0, 0, 0), (2, 2, 0)}
    assert set_many.calls[0][1].tolist() == [3, 3]
    assert set_many.calls[0][1].dtype == np.uint8
    assert voxels[1, 1, 0] == 9
    assert filled(voxels, 3) == {(0, 0, 0), (2, 2, 0)}


def test_apply_cells_erase_only_filled():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 0] = 9
    voxels[3, 3, 0] = 2
    set_many = ArraySetMany(voxels)
    cells = np.array([(0, 0, 0), (1, 1, 0), (3, 3, 0)])
    assert apply_cells(voxels, ERASE, 3, cells, set_many) == 2
    assert set_many.cells == {(1, 1, 0), (3, 3, 0)}
    assert set_many.calls[0][1].tolist() == [0, 0]
    assert int(np.count_nonzero(voxels)) == 0


def test_apply_cells_paint_skips_empty_and_same_colour():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 0] = 9
    voxels[3, 3, 0] = 3
    voxels[4, 4, 0] = 2
    set_many = ArraySetMany(voxels)
    cells = np.array([(0, 0, 0), (1, 1, 0), (3, 3, 0), (4, 4, 0)])
    assert apply_cells(voxels, PAINT, 3, cells, set_many) == 2
    assert set_many.cells == {(1, 1, 0), (4, 4, 0)}
    assert filled(voxels, 3) == {(1, 1, 0), (3, 3, 0), (4, 4, 0)}


def test_apply_cells_select_and_no_change_call_nothing():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 0] = 9
    set_many = ArraySetMany(voxels)
    cells = np.array([(0, 0, 0), (1, 1, 0)])
    assert apply_cells(voxels, SELECT, 3, cells, set_many) == 0
    assert apply_cells(voxels, ATTACH, 3, np.array([(1, 1, 0)]), set_many) == 0
    assert apply_cells(voxels, ERASE, 3, np.array([(0, 0, 0)]), set_many) == 0
    assert apply_cells(voxels, PAINT, 9, np.array([(1, 1, 0)]), set_many) == 0
    assert apply_cells(voxels, ATTACH, 3, np.zeros((0, 3), dtype=np.int64), set_many) == 0
    assert set_many.calls == []


def test_apply_cells_drops_out_of_bounds():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    set_many = ArraySetMany(voxels)
    cells = np.array([(0, 0, -1), (8, 0, 0), (2, 2, 2)])
    assert apply_cells(voxels, ATTACH, 3, cells, set_many) == 1
    assert set_many.cells == {(2, 2, 2)}
    assert apply_cells(voxels, ERASE, 3, np.array([(0, 0, -1), (8, 0, 0)]), set_many) == 0


def test_apply_cells_with_mirror():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[6, 2, 0] = 9  # a filled image cell
    set_many = ArraySetMany(voxels)
    cells = np.array([(1, 2, 0), (1, 3, 0)])
    n = apply_cells(voxels, ATTACH, 3, cells, set_many, mirror=(True, False, False))
    assert n == 3
    assert set_many.cells == {(1, 2, 0), (1, 3, 0), (6, 3, 0)}
    assert voxels[6, 2, 0] == 9
    assert filled(voxels, 3) == {(1, 2, 0), (1, 3, 0), (6, 3, 0)}


def test_apply_cells_mirror_drops_out_of_bounds_images():
    voxels = np.zeros((8, 8, 8), dtype=np.uint8)
    set_many = ArraySetMany(voxels)
    # A source cell outside the grid: its image (-2, 2, 0) is outside too.
    cells = np.array([(9, 2, 0)])
    n = apply_cells(voxels, ATTACH, 3, cells, set_many, mirror=(True, False, False))
    assert n == 0
    assert set_many.calls == []
    # Mixed: only the in-bounds images are written.
    cells = np.array([(9, 2, 0), (0, 2, 0), (3, 3, 3)])
    n = apply_cells(voxels, ATTACH, 3, cells, set_many, mirror=(True, True, True))
    expected = set()
    for cell in [(0, 2, 0), (3, 3, 3)]:
        expected |= set(mirror_images(cell, SIZE, (True, True, True)))
    assert n == len(expected) == 16
    assert set_many.cells == expected


# ---------------------------------------------------------------- strokes


def test_attach_on_floor_places_at_z0():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, setter = make_stroke(voxels, ATTACH, colour=3)
    hit = raycast(voxels, *top_ray(2, 4))
    assert hit is not None and hit.wall
    assert stroke.begin(hit)
    assert voxels[2, 4, 0] == 3
    assert int(np.count_nonzero(voxels)) == 1
    assert stroke.last == (2, 4, 0)
    assert stroke.axis == 2
    assert stroke.plane == 0.0
    assert stroke.layer == 0
    assert stroke.lock == PlaneLock(2, 0.0, 0)
    assert setter.calls == [(2, 4, 0, 3)]


def test_attach_on_plus_x_wall_places_at_the_border():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, setter = make_stroke(voxels, ATTACH, colour=3)
    hit = raycast(voxels, (-5.0, 2.5, 4.5), (1.0, 0.0, 0.0))
    assert hit is not None and hit.wall and hit.voxel == (8, 2, 4)
    assert stroke.begin(hit)
    assert voxels[7, 2, 4] == 3
    assert stroke.lock == PlaneLock(0, 8.0, 7)
    # Drag up the wall: the new voxels stay on the border.
    assert stroke.extend((-5.0, 2.5, 6.5), (1.0, 0.0, 0.0))
    assert filled(voxels, 3) == {(7, 2, 4), (7, 2, 5), (7, 2, 6)}


def test_attach_drag_along_top_face_stays_in_one_layer():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0] = 1  # a slab at z == 0
    stroke, setter = make_stroke(voxels, ATTACH, colour=4)
    assert stroke.begin(raycast(voxels, *top_ray(1, 1)))
    assert voxels[1, 1, 1] == 4
    # Drag to (5, 1): the ray now passes over freshly placed voxels.
    assert stroke.extend(*top_ray(5, 1))
    assert stroke.last == (5, 1, 1)
    # Drag back over the same cells: nothing stacks, nothing changes.
    assert not stroke.extend(*top_ray(1, 1))
    placed = np.argwhere(voxels == 4)
    assert sorted(map(tuple, placed.tolist())) == [(x, 1, 1) for x in range(1, 6)]
    assert int(np.count_nonzero(voxels[:, :, 2:])) == 0
    assert len(setter.calls) == 5


def test_attach_drag_diagonal_lays_interpolated_cells():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, _ = make_stroke(voxels, ATTACH, colour=2)
    assert stroke.begin(raycast(voxels, *top_ray(0, 0)))
    assert stroke.extend(*top_ray(6, 3))
    expected = {(0, 0, 0)} | set(line_cells((0, 0, 0), (6, 3, 0), 2))
    got = {tuple(c) for c in np.argwhere(voxels == 2).tolist()}
    assert got == expected
    assert (6, 3, 0) in got


def _block(size=(8, 8, 8)):
    """A block at the back of the grid: x 2..5, y 4..7, z 0..2."""
    voxels = np.zeros(size, dtype=np.uint8)
    voxels[2:6, 4:8, 0:3] = 1
    return voxels


def test_attach_drag_from_a_side_face_onto_the_floor_follows_the_floor():
    """A case from use: the first hit is a side face at floor level; the drag
    down onto the floor must lay a row along y, not stay locked to the face plane."""
    voxels = _block()
    stroke, _ = make_stroke(voxels, ATTACH, colour=7)
    hit = raycast(voxels, (3.5, -5.0, 0.5), (0.0, 1.0, 0.0))
    assert hit.voxel == (3, 4, 0) and hit.normal == (0, -1, 0)
    assert stroke.begin(hit)
    assert stroke.lock == PlaneLock(1, 4.0, 3)
    assert voxels[3, 3, 0] == 7
    # An oblique ray from in front and above that hits the floor at (3, 1): the
    # crossing on the face plane is below the floor, the ray hits the floor wall.
    origin = (3.5, -10.0, 5.0)
    direction = (0.0, 11.5, -5.0)
    assert stroke.extend(origin, direction)
    assert stroke.lock == PlaneLock(2, 0.0, 0)
    assert filled(voxels, 7) == {(3, 3, 0), (3, 2, 0), (3, 1, 0)}
    assert stroke.last == (3, 1, 0)
    # Further along the floor in y with a plain top ray: still on the floor.
    assert stroke.extend(*top_ray(3, 0))
    assert (3, 0, 0) in filled(voxels, 7)
    assert stroke.lock == PlaneLock(2, 0.0, 0)


def test_attach_drag_along_the_floor_into_a_block_climbs_its_face():
    voxels = _block()
    stroke, _ = make_stroke(voxels, ATTACH, colour=7)
    assert stroke.begin(raycast(voxels, *top_ray(3, 1)))
    assert stroke.lock == PlaneLock(2, 0.0, 0)
    assert stroke.extend(*top_ray(3, 3))  # up to the foot of the block
    assert filled(voxels, 7) == {(3, 1, 0), (3, 2, 0), (3, 3, 0)}
    # The ray now hits the block's -y face two cells up: the drag moves onto the
    # face and, because the last cell lies at its foot, fills the column.
    hit = raycast(voxels, (3.5, -5.0, 2.5), (0.0, 1.0, 0.0))
    assert hit.voxel == (3, 4, 2)
    assert stroke.extend((3.5, -5.0, 2.5), (0.0, 1.0, 0.0))
    assert stroke.lock == PlaneLock(1, 4.0, 3)
    assert filled(voxels, 7) == {(3, 1, 0), (3, 2, 0), (3, 3, 0), (3, 3, 1), (3, 3, 2)}
    assert stroke.last == (3, 3, 2)


def test_attach_drag_jumps_to_a_face_the_last_cell_does_not_touch():
    voxels = _block()
    stroke, _ = make_stroke(voxels, ATTACH, colour=7)
    assert stroke.begin(raycast(voxels, *top_ray(3, 0)))
    # Straight from the far floor cell onto the face: no line across the air.
    assert stroke.extend((3.5, -5.0, 2.5), (0.0, 1.0, 0.0))
    assert filled(voxels, 7) == {(3, 0, 0), (3, 3, 2)}
    assert stroke.lock == PlaneLock(1, 4.0, 3)


def test_attach_drag_above_a_pillar_top_keeps_building_the_column():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[3, 3, 0:4] = 1  # a pillar
    stroke, _ = make_stroke(voxels, ATTACH, colour=6)
    assert stroke.begin(raycast(voxels, (10.0, 3.5, 1.5), (-1.0, 0.0, 0.0)))
    assert stroke.lock == PlaneLock(0, 4.0, 4)
    # Above the pillar the ray crosses the plane inside the grid and leaves
    # through the far wall: the plane wins, the column goes on into the air.
    assert stroke.extend((10.0, 3.5, 6.5), (-1.0, 0.0, 0.0))
    assert stroke.lock == PlaneLock(0, 4.0, 4)
    assert filled(voxels, 6) == {(4, 3, z) for z in range(1, 7)}


def test_attached_voxels_do_not_catch_the_ray():
    """A drag across a top face keeps hitting that face, not the fresh voxels, so
    nothing stacks toward the camera."""
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0] = 1
    stroke, _ = make_stroke(voxels, ATTACH, colour=4)
    assert stroke.begin(raycast(voxels, *top_ray(1, 1)))
    for x in range(2, 6):
        stroke.extend(*top_ray(x, 1))
    # An oblique ray that passes through the placed row before the slab.
    assert stroke.extend((6.5, 1.5, 3.0), (-1.0, 0.0, -1.0)) or True
    assert int(np.count_nonzero(voxels[:, :, 2:])) == 0
    assert raycast(voxels, *top_ray(3, 1), ignore=stroke.attached).voxel == (3, 1, 0)
    assert raycast(voxels, *top_ray(3, 1)).voxel == (3, 1, 1)


def test_attach_on_side_face_locks_to_that_plane():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[3, 3, 0:4] = 1  # a pillar
    stroke, _ = make_stroke(voxels, ATTACH, colour=6)
    hit = raycast(voxels, (10.0, 3.5, 1.5), (-1.0, 0.0, 0.0))
    assert hit.normal == (1, 0, 0)
    assert stroke.begin(hit)
    assert stroke.axis == 0 and stroke.plane == 4.0 and stroke.layer == 4
    assert voxels[4, 3, 1] == 6
    # Drag up the face: all new voxels sit at x == 4.
    assert stroke.extend(*((10.0, 3.5, 3.5), (-1.0, 0.0, 0.0)))
    got = {tuple(c) for c in np.argwhere(voxels == 6).tolist()}
    assert got == {(4, 3, 1), (4, 3, 2), (4, 3, 3)}


def test_erase_touches_only_filled_cells_and_never_the_floor():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2:5, 2, 0] = 1
    stroke, setter = make_stroke(voxels, ERASE)
    assert stroke.begin(raycast(voxels, *top_ray(2, 2)))
    assert voxels[2, 2, 0] == 0
    assert stroke.erased == {(2, 2, 0)}
    # Over empty ground: the re-raycast hits nothing, nothing changes.
    assert not stroke.extend(*top_ray(7, 2))
    assert stroke.last == (2, 2, 0)
    # Back over the run: each hovered voxel goes.
    assert stroke.extend(*top_ray(3, 2))
    assert stroke.extend(*top_ray(4, 2))
    assert int(np.count_nonzero(voxels)) == 0
    assert all(call[2] == 0 for call in setter.calls)
    assert len(setter.calls) == 3
    assert stroke.erased == {(2, 2, 0), (3, 2, 0), (4, 2, 0)}
    # A new erase stroke on bare floor does not start.
    stroke2, setter2 = make_stroke(voxels, ERASE)
    assert not stroke2.begin(raycast(voxels, *top_ray(1, 1)))
    assert setter2.calls == []


def test_paint_touches_only_filled_cells_and_never_the_floor():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 2, 0] = 1
    voxels[4, 2, 0] = 1
    stroke, setter = make_stroke(voxels, PAINT, colour=9)
    assert stroke.begin(raycast(voxels, *top_ray(2, 2)))
    assert voxels[2, 2, 0] == 9
    # A ray over empty ground hits nothing without the walls.
    assert not stroke.extend(*top_ray(3, 2))
    assert stroke.last == (2, 2, 0)
    assert stroke.extend(*top_ray(4, 2))
    assert voxels[4, 2, 0] == 9
    assert int(np.count_nonzero(voxels)) == 2
    assert len(setter.calls) == 2
    stroke2, setter2 = make_stroke(voxels, PAINT, colour=9)
    assert not stroke2.begin(raycast(voxels, *top_ray(6, 6)))
    assert setter2.calls == []


def test_select_stroke_writes_nothing():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 2, 0] = 1
    stroke, setter = make_stroke(voxels, SELECT)
    assert stroke.begin(raycast(voxels, *top_ray(2, 2)))
    assert stroke.last == (2, 2, 0)
    assert not stroke.extend(*top_ray(3, 2))
    assert setter.calls == []
    assert voxels[2, 2, 0] == 1
    # On the floor the target is the ground cell, and still nothing is written.
    stroke2, setter2 = make_stroke(voxels, SELECT)
    assert stroke2.begin(raycast(voxels, *top_ray(5, 5)))
    assert stroke2.last == (5, 5, 0)
    assert setter2.calls == []


def test_repeated_cell_is_applied_once():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, setter = make_stroke(voxels, ATTACH, colour=2)
    assert stroke.begin(raycast(voxels, *top_ray(1, 1)))
    assert not stroke.apply_cell((1, 1, 0))
    assert not stroke.extend(*top_ray(1, 1))
    assert stroke.visited == {(1, 1, 0)}
    assert len(setter.calls) == 1


def test_attach_on_filled_cell_returns_false_without_setter_call():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 0] = 1
    voxels[1, 1, 1] = 1
    stroke, setter = make_stroke(voxels, ATTACH, colour=2)
    assert not stroke.apply_cell((1, 1, 1))
    hit = Hit((1, 1, 0), (0, 0, 1), 1.0)  # target (1, 1, 1) is already filled
    assert stroke.begin(hit)  # the stroke starts; the cell itself is a no-op
    assert setter.calls == []
    assert voxels[1, 1, 1] == 1


def test_erase_on_empty_cell_returns_false_without_setter_call():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, setter = make_stroke(voxels, ERASE)
    assert not stroke.apply_cell((1, 1, 0))
    assert setter.calls == []
    assert stroke.erased == set()


def test_paint_same_colour_returns_false_without_setter_call():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 0] = 9
    stroke, setter = make_stroke(voxels, PAINT, colour=9)
    hit = raycast(voxels, *top_ray(1, 1))
    assert stroke.begin(hit)  # a same-colour paint still starts a stroke
    assert not stroke.extend(*top_ray(1, 1))
    assert setter.calls == []
    assert voxels[1, 1, 0] == 9


def test_out_of_bounds_cell_returns_false_without_setter_call():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    for tool in TOOLS:
        stroke, setter = make_stroke(voxels, tool)
        assert not stroke.apply_cell((1, 1, -1))
        assert not stroke.apply_cell((8, 1, 1))
        assert setter.calls == []


def test_paint_follows_re_raycast_on_stepped_surface():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[0:2, 1, 0] = 1  # low step
    voxels[2:4, 1, 0:3] = 1  # high step
    stroke, setter = make_stroke(voxels, PAINT, colour=7)
    assert stroke.begin(raycast(voxels, *top_ray(0, 1)))
    assert stroke.last == (0, 1, 0)
    assert stroke.extend(*top_ray(1, 1))
    assert stroke.last == (1, 1, 0)
    assert stroke.extend(*top_ray(2, 1))
    assert stroke.last == (2, 1, 2)
    assert stroke.extend(*top_ray(3, 1))
    assert stroke.last == (3, 1, 2)
    got = {tuple(c) for c in np.argwhere(voxels == 7).tolist()}
    assert got == {(0, 1, 0), (1, 1, 0), (2, 1, 2), (3, 1, 2)}
    assert len(setter.calls) == 4


def test_begin_returns_false_when_target_cell_is_none():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[7, 3, 3] = 1
    stroke, setter = make_stroke(voxels, ATTACH)
    hit = raycast(voxels, (20.0, 3.5, 3.5), (-1.0, 0.0, 0.0))
    assert hit.voxel == (7, 3, 3) and hit.normal == (1, 0, 0)
    assert not stroke.begin(hit)
    assert stroke.hit is None and stroke.last is None
    assert setter.calls == []
    floor = raycast(voxels, *top_ray(1, 1))
    assert floor.wall
    for tool in (ERASE, PAINT):
        stroke, setter = make_stroke(voxels, tool)
        assert not stroke.begin(floor)
        assert setter.calls == []


def test_extend_returns_false_when_ray_misses_plane():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, _ = make_stroke(voxels, ATTACH)
    assert stroke.begin(raycast(voxels, *top_ray(1, 1)))
    # Parallel to the plane, and crossing behind the origin.
    assert not stroke.extend((1.5, 1.5, 10.0), (1.0, 0.0, 0.0))
    assert not stroke.extend((1.5, 1.5, -5.0), (0.0, 0.0, -1.0))
    assert stroke.last == (1, 1, 0)


def test_attach_drag_off_the_edge_fills_to_the_border():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, setter = make_stroke(voxels, ATTACH, colour=3)
    assert stroke.begin(raycast(voxels, *top_ray(2, 2)))
    # One move whose ray crosses the plane just past the edge (x == 9.5).
    assert stroke.extend(*top_ray(9, 2))
    got = {tuple(c) for c in np.argwhere(voxels == 3).tolist()}
    assert got == {(x, 2, 0) for x in range(2, 8)}
    assert stroke.last == (7, 2, 0)
    assert len(setter.calls) == 6
    # Off the grid again: nothing more to do, the last cell stays at the border.
    assert not stroke.extend(*top_ray(12, 2))
    assert stroke.last == (7, 2, 0)


def test_erase_drag_off_the_edge_clears_the_border_cell():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[4:8, 5, 0] = 1
    stroke, setter = make_stroke(voxels, ERASE)
    assert stroke.begin(raycast(voxels, *top_ray(4, 5)))
    for x in range(5, 8):
        assert stroke.extend(*top_ray(x, 5))
    assert stroke.last == (7, 5, 0)
    # Off the grid: the ray misses, nothing changes, the last cell stays.
    assert not stroke.extend(*top_ray(20, 5))
    assert int(np.count_nonzero(voxels)) == 0
    assert stroke.last == (7, 5, 0)
    assert len(setter.calls) == 4


def test_extend_far_crossing_is_clamped_to_one_cell_past_the_edge():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, _ = make_stroke(voxels, ATTACH, colour=2)
    assert stroke.begin(raycast(voxels, *top_ray(3, 3)))
    # A grazing ray: it crosses z == 0 at x == 10003.5, y == 3.5.
    assert stroke.extend((3.5, 3.5, 10.0), (1000.0, 0.0, -1.0))
    got = {tuple(c) for c in np.argwhere(voxels == 2).tolist()}
    assert got == {(x, 3, 0) for x in range(3, 8)}
    # The walk stopped at the first out-of-bounds cell; nothing far away was visited.
    assert stroke.visited == got
    assert stroke.last == (7, 3, 0)


def test_extend_off_the_edge_and_back_continues_from_the_border():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, _ = make_stroke(voxels, ATTACH, colour=4)
    assert stroke.begin(raycast(voxels, *top_ray(5, 1)))
    assert stroke.extend(*top_ray(10, 1))
    assert stroke.last == (7, 1, 0)
    # Back inside on another row: the line starts at the border cell.
    assert stroke.extend(*top_ray(6, 3))
    got = {tuple(c) for c in np.argwhere(voxels == 4).tolist()}
    expected = {(5, 1, 0), (6, 1, 0), (7, 1, 0)} | set(
        line_cells((7, 1, 0), (6, 3, 0), 2)
    )
    assert got == expected
    assert stroke.last == (6, 3, 0)


def test_finish_and_preview():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    stroke, setter = make_stroke(voxels, ATTACH, colour=4)
    assert stroke.finish() == 0
    assert stroke.preview() is None
    assert stroke.begin(raycast(voxels, *top_ray(5, 1)))
    assert stroke.finish() == 0
    assert stroke.preview() is None
    assert len(setter.calls) == 1


# ---------------------------------------------------------------- erase follows the surface


def test_erase_follows_the_surface_on_a_step():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[0:2, 1, 0] = 1  # low step
    voxels[2:4, 1, 0:3] = 1  # high step
    stroke, setter = make_stroke(voxels, ERASE)
    assert stroke.begin(raycast(voxels, *top_ray(0, 1)))
    for x in range(1, 4):
        assert stroke.extend(*top_ray(x, 1))
        assert stroke.last[0] == x
    changed = {tuple(c) for c in np.argwhere(voxels == 0).tolist()} & {
        (0, 1, 0), (1, 1, 0), (2, 1, 2), (3, 1, 2), (2, 1, 1), (3, 1, 1)
    }
    assert changed == {(0, 1, 0), (1, 1, 0), (2, 1, 2), (3, 1, 2)}
    assert len(setter.calls) == 4


def test_erase_returns_to_one_column_three_times_and_removes_one_voxel():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[3, 3, 0:4] = 1  # a pillar of four
    stroke, setter = make_stroke(voxels, ERASE)
    assert stroke.begin(raycast(voxels, *top_ray(3, 3)))
    assert voxels[3, 3, 3] == 0
    # A jiggle over the same column: the phantom of the erased voxel is hit,
    # the voxel below it is never picked.
    assert not stroke.extend(*top_ray(3, 3))
    assert not stroke.extend((3.6, 3.4, 50.0), (0.0, 0.0, -1.0))
    assert not stroke.extend((3.2, 3.8, 50.0), (0.01, -0.01, -1.0))
    assert stroke.last == (3, 3, 3)
    assert voxels[3, 3, 0:3].tolist() == [1, 1, 1]
    assert len(setter.calls) == 1
    assert stroke.erased == {(3, 3, 3)}


def test_erase_oblique_step_does_not_dig_through_the_hole():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 3, 0] = 1  # low column
    voxels[3, 3, 0:3] = 1  # high column next to it
    stroke, setter = make_stroke(voxels, ERASE)
    # Look obliquely from -X over the low column onto the high voxel's -X face.
    origin = (-3.0, 3.5, 4.0)
    direction = (1.0, 0.0, -0.3)
    hit = raycast(voxels, origin, direction)
    assert hit.voxel == (3, 3, 2) and hit.normal == (-1, 0, 0)
    assert stroke.begin(hit)
    assert voxels[3, 3, 2] == 0
    # The same ray now sees (3, 3, 1) through the hole. It stays.
    assert raycast(voxels, origin, direction, walls=False).voxel == (3, 3, 1)
    assert not stroke.extend(origin, direction)
    assert voxels[3, 3, 1] == 1
    assert voxels[2, 3, 0] == 1
    assert stroke.last == (3, 3, 2)
    assert len(setter.calls) == 1


# ---------------------------------------------------------------- mirror strokes


def test_attach_stroke_with_mirror_x_writes_both_images():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[6, 2, 0] = 9  # a filled image cell
    stroke, setter = make_stroke(voxels, ATTACH, colour=3, mirror=(True, False, False))
    assert stroke.mirror == (True, False, False)
    assert stroke.begin(raycast(voxels, *top_ray(1, 4)))
    assert filled(voxels, 3) == {(1, 4, 0), (6, 4, 0)}
    assert stroke.extend(*top_ray(1, 2))
    assert filled(voxels, 3) == {(1, 4, 0), (6, 4, 0), (1, 3, 0), (6, 3, 0), (1, 2, 0)}
    # The filled image cell keeps its colour and gets no setter call.
    assert voxels[6, 2, 0] == 9
    assert (6, 2, 0, 3) not in setter.calls
    assert len(setter.calls) == 5
    assert stroke.visited == {(1, 4, 0), (1, 3, 0), (1, 2, 0)}
    assert stroke.last == (1, 2, 0)


def test_erase_stroke_with_mirror_records_both_images():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 4, 0:2] = 1
    voxels[6, 4, 0:2] = 1
    stroke, setter = make_stroke(voxels, ERASE, mirror=(True, False, False))
    assert stroke.begin(raycast(voxels, *top_ray(1, 4)))
    assert voxels[1, 4, 1] == 0 and voxels[6, 4, 1] == 0
    assert stroke.erased == {(1, 4, 1), (6, 4, 1)}
    # Hovering the image column hits its phantom, so nothing more goes.
    assert not stroke.extend(*top_ray(6, 4))
    assert voxels[6, 4, 0] == 1 and voxels[1, 4, 0] == 1
    assert len(setter.calls) == 2


def test_mirror_image_outside_the_grid_is_skipped():
    voxels = np.zeros((8, 8, 8), dtype=np.uint8)
    stroke, setter = make_stroke(voxels, ATTACH, colour=3, mirror=(False, False, True))
    # Attach on the top wall: the image of z == 7 is z == 0, inside. Then a
    # source cell whose image is outside: none exists on a symmetric grid, so
    # use apply_cell with an out-of-bounds source: nothing is written.
    assert not stroke.apply_cell((1, 1, 8))
    assert setter.calls == []
    assert stroke.begin(raycast(voxels, (1.5, 1.5, -5.0), (0.0, 0.0, 1.0)))
    assert filled(voxels, 3) == {(1, 1, 7), (1, 1, 0)}


# ---------------------------------------------------------------- golden


def test_golden_robot_erase_top_surface():
    model = read_vox(FIXTURES / "robot.vox")
    voxels = model.voxels.copy()
    before = voxels.copy()
    # Row y == 12, x == 7..11 is a flat run of the head; every column tops out at z == 18.
    y = 12
    xs = list(range(7, 12))
    tops = {x: int(np.nonzero(before[x, y, :])[0].max()) for x in xs}
    assert set(tops.values()) == {18}

    stroke, setter = make_stroke(voxels, ERASE)
    first = raycast(voxels, *top_ray(xs[0], y, z=100.0))
    assert first.voxel == (xs[0], y, 18) and first.normal == (0, 0, 1)
    assert stroke.begin(first)
    for x in xs[1:]:
        assert stroke.extend(*top_ray(x, y, z=100.0))

    expected = {(x, y, tops[x]) for x in xs}
    changed = {tuple(c) for c in np.argwhere(before != voxels).tolist()}
    assert changed == expected
    for x, yy, z in expected:
        assert voxels[x, yy, z] == 0
    assert int(np.count_nonzero(before)) - int(np.count_nonzero(voxels)) == len(xs)
    assert len(setter.calls) == len(xs)


def test_golden_robot_erase_stepped_run():
    model = read_vox(FIXTURES / "robot.vox")
    voxels = model.voxels.copy()
    before = voxels.copy()
    # Row y == 13, x == 2..11: the column tops step from z == 12 (an arm) to 13
    # (the shoulder) to 18 (the head).
    y = 13
    xs = list(range(2, 12))
    tops = {x: int(np.nonzero(before[x, y, :])[0].max()) for x in xs}
    assert len(set(tops.values())) >= 3

    stroke, setter = make_stroke(voxels, ERASE)
    first = raycast(voxels, *top_ray(xs[0], y, z=100.0))
    assert first.voxel == (xs[0], y, tops[xs[0]])
    assert stroke.begin(first)
    for x in xs[1:]:
        assert stroke.extend(*top_ray(x, y, z=100.0))
        assert stroke.last == (x, y, tops[x])

    expected = {(x, y, tops[x]) for x in xs}
    changed = {tuple(c) for c in np.argwhere(before != voxels).tolist()}
    assert changed == expected
    assert int(np.count_nonzero(before)) - int(np.count_nonzero(voxels)) == len(xs)
    assert len(setter.calls) == len(xs)
    assert stroke.erased == expected


def test_golden_robot_erase_same_column_three_times():
    model = read_vox(FIXTURES / "robot.vox")
    voxels = model.voxels.copy()
    before = voxels.copy()
    x, y = 11, 10
    top = int(np.nonzero(before[x, y, :])[0].max())
    assert before[x, y, top - 1] != 0  # there is a voxel below to protect

    stroke, setter = make_stroke(voxels, ERASE)
    assert stroke.begin(raycast(voxels, *top_ray(x, y, z=100.0)))
    for _ in range(3):
        assert not stroke.extend(*top_ray(x, y, z=100.0))
    changed = {tuple(c) for c in np.argwhere(before != voxels).tolist()}
    assert changed == {(x, y, top)}
    assert voxels[x, y, top - 1] == before[x, y, top - 1]
    assert len(setter.calls) == 1


# ---------------------------------------------------------------- the brush sphere

from rok.core.edit import BRUSH_MAX, sphere_offsets, sphere_shell  # noqa: E402


def test_brush_sizes_grow_and_stay_round():
    assert sphere_offsets(1).tolist() == [[0, 0, 0]]
    counts = [len(sphere_offsets(n)) for n in range(1, BRUSH_MAX + 1)]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    for n in range(2, BRUSH_MAX + 1):
        off = sphere_offsets(n)
        r = n - 1
        assert np.abs(off).max() == r  # reaches the radius on the axes
        assert not ((np.abs(off) == r).all(axis=1)).any()  # never the cube's corners
        assert len(sphere_shell(n)) < len(off)


def test_a_brush_stroke_writes_a_sphere_in_one_batch_and_mirrors_it():
    voxels = np.zeros((20, 20, 20), dtype=np.uint8)
    calls = []

    def set_many(cells, values):
        calls.append(len(cells))
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return len(cells)

    def setter(x, y, z, v):
        raise AssertionError("a brush stroke must not write cell by cell")

    stroke = Stroke(voxels, ATTACH, 5, setter, (True, False, False), brush=3, set_many=set_many)
    assert stroke.apply_cell((4, 10, 10))
    assert len(calls) == 1
    n = len(sphere_offsets(3))
    assert int((voxels != 0).sum()) == 2 * n  # the sphere and its X mirror
    assert voxels[4, 10, 10] == 5 and voxels[15, 10, 10] == 5
    # Erase takes the same sphere out; Paint recolours only filled cells.
    Stroke(voxels, PAINT, 7, setter, brush=3, set_many=set_many).apply_cell((4, 10, 10))
    assert int((voxels == 7).sum()) == n
    Stroke(voxels, ERASE, 0, setter, brush=3, set_many=set_many).apply_cell((4, 10, 10))
    assert voxels[4, 10, 10] == 0 and int((voxels != 0).sum()) == n


@pytest.mark.parametrize(
    "base, forced, alt, shift, expected",
    [
        (SELECT, "ERASE", False, False, (SELECT, True)),  # right-click in Select deselects
        (SELECT, None, True, False, (SELECT, True)),  # so does Option
        (SELECT, None, False, False, (SELECT, False)),
        (ATTACH, "ERASE", False, False, (ERASE, False)),  # right-click erases elsewhere
        (PAINT, None, True, False, (ERASE, False)),
    ],
)
def test_press_tool_never_erases_in_select(base, forced, alt, shift, expected):
    from rok.core.edit import press_tool

    assert press_tool(base, forced, alt, shift) == expected


def test_a_shader_stroke_writes_the_target_inside_the_brush_filled_or_empty():
    from rok.core.edit import ShaderStroke


    v = np.zeros((12, 12, 6), np.uint8)
    v[:, :, 0] = 5
    target = v.copy()
    target[:, :, 0] = 7  # repaints the floor ...
    target[:, :, 1] = 3  # ... and grows a layer on it
    written = []

    def set_many(cells, values):
        v[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        written.append(len(cells))
        return len(cells)

    stroke = ShaderStroke(v, target, None, brush=2, set_many=set_many)
    hit = Hit(voxel=(6, 6, 0), normal=(0, 0, 1), t=1.0)
    assert stroke.begin(hit)
    assert v[6, 6, 0] == 7 and v[6, 6, 1] == 3  # the hit voxel, and the empty cell above it
    assert v[0, 0, 0] == 5 and v[0, 0, 1] == 0  # outside the brush: as it was
    assert not stroke.apply_cell((6, 6, 0))  # once per stroke


def test_unique_cells_is_np_unique_without_the_sort():
    """The click that built a 500 000-cell box spent 200 ms
    in np.unique; a flag per grid cell gives the same rows in the same order."""
    from rok.core.edit import unique_cells

    rng = np.random.default_rng(2)
    size = (20, 17, 9)
    cells = rng.integers(0, [20, 17, 9], size=(3000, 3))
    cells = np.concatenate([cells, cells[:500], [[0, 0, 0], [19, 16, 8]]])  # doubles, both corners
    want = np.unique(cells, axis=0)
    assert np.array_equal(unique_cells(cells, size), want)
    outside = np.concatenate([cells, [[20, 0, 0], [-1, 3, 3]]])
    assert np.array_equal(unique_cells(outside, size), want)  # cells past the grid are dropped
    assert unique_cells(np.zeros((0, 3), dtype=np.int64), size).shape == (0, 3)


# --- the swept brush (capsule) ---------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_the_native_capsule_matches_the_numpy_one(seed, monkeypatch):
    from rok.core import edit as edit_mod

    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 30, 3), rng.integers(0, 30, 3)
    size = int(rng.integers(2, 9))
    lo = np.minimum(a, b) - size
    dims = np.abs(b - a) + 2 * size - 1
    native = edit_mod.capsule_mask(a, b, size, lo, dims)
    monkeypatch.setattr(edit_mod, "_CAPSULE_NATIVE", [None])
    assert np.array_equal(native, edit_mod.capsule_mask(a, b, size, lo, dims))


def test_a_capsule_of_no_length_is_the_brush_sphere():
    from rok.core.edit import capsule_mask, sphere_mask

    for size in (2, 3, 6, 12):
        r = size - 1
        c = (20, 20, 20)
        lo = [v - r for v in c]
        assert np.array_equal(capsule_mask(c, c, size, lo, (2 * r + 1,) * 3), sphere_mask(size))



def test_a_run_that_starts_on_a_cell_of_an_earlier_run_keeps_that_cell():
    """A designed stroke replays its runs. A run that starts on a cell the stroke passed
    before must still start there, or the replay begins one cell late."""
    voxels = np.zeros((32, 8, 8), dtype=np.uint8)
    stroke, _setter = make_stroke(voxels, ATTACH)
    for cell in ((2, 2, 0), (3, 2, 0), (4, 2, 0)):
        stroke._note(cell)
    stroke._note((3, 2, 0))  # back over a cell of the same run: noted once
    stroke._note((20, 2, 0))  # a jump: a new run
    stroke._note((3, 2, 0))  # a jump back onto the first run
    stroke._note((2, 2, 0))
    assert stroke.runs == [[(2, 2, 0), (3, 2, 0), (4, 2, 0)], [(20, 2, 0)], [(3, 2, 0)]]
