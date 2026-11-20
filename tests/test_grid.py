"""Dense grid edits and 32^3 chunk dirty tracking."""

import numpy as np
import pytest

from rok.core.grid import CHUNK_SIZE, VoxelGrid
from rok.core.undo import StrokeRecorder
from rok.core.vox_read import read_vox

from helpers import FIXTURES


def test_empty_32_cube():
    grid = VoxelGrid.empty()
    assert grid.size == (32, 32, 32)
    assert grid.voxels.shape == (32, 32, 32)
    assert grid.voxels.dtype == np.uint8
    assert int(grid.voxels.sum()) == 0
    assert grid.chunk_counts == (1, 1, 1)
    assert grid.dirty_chunks() == {(0, 0, 0)}


def test_set_and_get_voxel():
    grid = VoxelGrid((8, 8, 8))
    grid.set_voxel(1, 2, 3, 9)
    assert grid.get_voxel(1, 2, 3) == 9
    assert grid.voxels[1, 2, 3] == 9


def test_same_value_is_not_dirty():
    grid = VoxelGrid((32, 32, 32))
    grid.set_voxel(4, 4, 4, 3)
    grid.clear_dirty()
    grid.set_voxel(4, 4, 4, 3)
    assert grid.dirty_chunks() == frozenset()


def test_edit_marks_own_chunk():
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    grid.set_voxel(1, 1, 1, 2)
    assert grid.dirty_chunks() == {(0, 0, 0)}


def test_boundary_edit_marks_neighbour_chunk():
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    grid.set_voxel(CHUNK_SIZE, 0, 0, 5)
    assert (1, 0, 0) in grid.dirty_chunks()
    assert (0, 0, 0) in grid.dirty_chunks()

    grid.clear_dirty()
    grid.set_voxel(CHUNK_SIZE - 1, 0, 0, 6)
    assert (0, 0, 0) in grid.dirty_chunks()
    assert (1, 0, 0) in grid.dirty_chunks()


def test_many_cells_mark_their_chunks_and_the_chunks_beyond_their_faces():
    """The dirty marking of a big write ran a sort over every
    cell; now a flag per chunk. The same chunks as one cell at a time."""
    rng = np.random.default_rng(3)
    grid = VoxelGrid((70, 40, 33))
    cells = rng.integers(0, [70, 40, 33], size=(5000, 3))
    cells = np.concatenate([cells, [[0, 0, 0], [69, 39, 32], [32, 31, 0], [31, 32, 1]]])
    grid.clear_dirty()
    grid.set_voxels(cells, 7)
    many = set(grid.dirty_chunks())
    one = VoxelGrid((70, 40, 33))
    one.clear_dirty()
    for x, y, z in {tuple(int(v) for v in c) for c in cells}:
        one.set_voxel(x, y, z, 7)
    assert many == set(one.dirty_chunks())


def test_chunk_counts_for_uneven_size():
    grid = VoxelGrid((33, 1, 1))
    assert grid.chunk_counts == (2, 1, 1)
    bounds = grid.chunk_bounds((1, 0, 0))
    assert grid.voxels[bounds].shape == (1, 1, 1)


def test_rejects_too_large():
    with pytest.raises(ValueError):
        VoxelGrid((257, 1, 1))


def test_from_fixture_model():
    model = read_vox(FIXTURES / "robot.vox")
    grid = VoxelGrid.from_model(model)
    assert grid.size == model.size
    assert np.array_equal(grid.voxels, model.voxels)
    assert np.array_equal(grid.palette, model.palette)
    back = grid.to_model()
    assert back.size == model.size
    assert np.array_equal(back.voxels, model.voxels)
    assert np.array_equal(back.palette, model.palette)


def test_set_voxel_reports_change():
    grid = VoxelGrid((8, 8, 8))
    assert grid.set_voxel(1, 1, 1, 4) is True
    assert grid.set_voxel(1, 1, 1, 4) is False
    assert grid.set_voxel(1, 1, 1, 0) is True


def test_in_bounds():
    grid = VoxelGrid((4, 5, 6))
    assert grid.in_bounds(0, 0, 0)
    assert grid.in_bounds(3, 4, 5)
    assert not grid.in_bounds(4, 0, 0)
    assert not grid.in_bounds(0, 5, 0)
    assert not grid.in_bounds(0, 0, 6)
    assert not grid.in_bounds(-1, 0, 0)


def test_chunk_of():
    grid = VoxelGrid((64, 64, 64))
    assert grid.chunk_of(0, 0, 0) == (0, 0, 0)
    assert grid.chunk_of(31, 32, 63) == (0, 1, 1)
    assert grid.chunk_of(33, 1, 40) == (1, 0, 1)


def test_read_chunk_is_a_copy():
    grid = VoxelGrid((64, 32, 32))
    grid.set_voxel(3, 3, 3, 7)
    data = grid.read_chunk((0, 0, 0))
    assert data.shape == (32, 32, 32)
    assert data[3, 3, 3] == 7
    data[3, 3, 3] = 9
    data[0, 0, 0] = 1
    assert grid.get_voxel(3, 3, 3) == 7
    assert grid.get_voxel(0, 0, 0) == 0


def test_read_chunk_at_uneven_edge():
    grid = VoxelGrid((33, 1, 1))
    assert grid.read_chunk((1, 0, 0)).shape == (1, 1, 1)


def test_write_chunk_rejects_wrong_shape():
    grid = VoxelGrid((64, 32, 32))
    with pytest.raises(ValueError):
        grid.write_chunk((0, 0, 0), np.zeros((16, 32, 32), dtype=np.uint8))
    with pytest.raises(ValueError):
        grid.write_chunk((0, 0, 0), np.zeros((32, 32), dtype=np.uint8))


def test_write_chunk_assigns_and_marks_own_chunk():
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    data = np.zeros((32, 32, 32), dtype=np.uint8)
    data[5, 6, 7] = 3
    grid.write_chunk((0, 0, 0), data)
    assert grid.get_voxel(5, 6, 7) == 3
    assert np.array_equal(grid.read_chunk((0, 0, 0)), data)
    assert grid.dirty_chunks() == {(0, 0, 0)}


def test_write_chunk_interior_change_leaves_neighbour_clean():
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    data = grid.read_chunk((1, 0, 0))
    data[10, 10, 10] = 2
    grid.write_chunk((1, 0, 0), data)
    assert grid.dirty_chunks() == {(1, 0, 0)}


def test_write_chunk_boundary_change_marks_neighbour():
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    data = grid.read_chunk((1, 0, 0))
    data[0, 10, 10] = 2
    grid.write_chunk((1, 0, 0), data)
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}

    grid.clear_dirty()
    data = grid.read_chunk((0, 0, 0))
    data[CHUNK_SIZE - 1, 4, 4] = 5
    grid.write_chunk((0, 0, 0), data)
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}


def test_write_chunk_unchanged_boundary_leaves_neighbour_clean():
    grid = VoxelGrid((64, 32, 32))
    grid.set_voxel(CHUNK_SIZE, 1, 1, 9)
    grid.clear_dirty()
    data = grid.read_chunk((1, 0, 0))
    data[20, 20, 20] = 1
    grid.write_chunk((1, 0, 0), data)
    assert grid.dirty_chunks() == {(1, 0, 0)}


def test_write_chunk_at_grid_edge_marks_no_missing_neighbour():
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    data = grid.read_chunk((1, 0, 0))
    data[CHUNK_SIZE - 1, 0, 0] = 2
    data[0, CHUNK_SIZE - 1, CHUNK_SIZE - 1] = 2
    grid.write_chunk((1, 0, 0), data)
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}

    grid = VoxelGrid((33, 1, 1))
    grid.clear_dirty()
    grid.write_chunk((1, 0, 0), np.array([[[4]]], dtype=np.uint8))
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}


def test_set_voxels_returns_changed_count():
    grid = VoxelGrid((8, 8, 8))
    grid.set_voxel(1, 1, 1, 4)
    cells = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert grid.set_voxels(cells, 4) == 2
    assert grid.get_voxel(2, 2, 2) == 4
    assert grid.get_voxel(3, 3, 3) == 4


def test_set_voxels_writes_nothing_for_equal_values():
    grid = VoxelGrid((8, 8, 8))
    grid.set_voxel(1, 1, 1, 4)
    grid.clear_dirty()
    before = grid.voxels.copy()
    assert grid.set_voxels(np.array([[1, 1, 1]]), 4) == 0
    assert grid.set_voxels(np.array([[2, 2, 2]]), 0) == 0
    assert grid.set_voxels(np.zeros((0, 3), dtype=np.intp), 7) == 0
    assert np.array_equal(grid.voxels, before)
    assert grid.dirty_chunks() == frozenset()


def test_set_voxels_broadcasts_scalar():
    grid = VoxelGrid((8, 8, 8))
    cells = np.array([[0, 0, 0], [1, 2, 3], [7, 7, 7]])
    assert grid.set_voxels(cells, 9) == 3
    assert np.array_equal(grid.voxels[cells[:, 0], cells[:, 1], cells[:, 2]], [9, 9, 9])
    assert int(np.count_nonzero(grid.voxels)) == 3


def test_set_voxels_accepts_values_array():
    grid = VoxelGrid((8, 8, 8))
    cells = np.array([[0, 0, 0], [1, 2, 3], [7, 7, 7]])
    values = np.array([1, 0, 255], dtype=np.uint8)
    assert grid.set_voxels(cells, values) == 2
    assert grid.get_voxel(0, 0, 0) == 1
    assert grid.get_voxel(1, 2, 3) == 0
    assert grid.get_voxel(7, 7, 7) == 255
    with pytest.raises(ValueError):
        grid.set_voxels(cells, np.array([1, 2], dtype=np.uint8))


def test_set_voxels_marks_chunks_like_set_voxel():
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    grid.set_voxels(np.array([[10, 10, 10]]), 1)
    assert grid.dirty_chunks() == {(0, 0, 0)}

    grid.clear_dirty()
    grid.set_voxels(np.array([[CHUNK_SIZE, 5, 5]]), 2)
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}

    grid.clear_dirty()
    grid.set_voxels(np.array([[CHUNK_SIZE - 1, 5, 5]]), 3)
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}

    grid.clear_dirty()
    grid.set_voxels(np.array([[40, 0, 31]]), 4)
    assert grid.dirty_chunks() == {(1, 0, 0)}


def test_set_voxels_marks_every_changed_chunk_once():
    grid = VoxelGrid((96, 64, 32))
    grid.clear_dirty()
    cells = np.array([[1, 1, 1], [2, 2, 2], [40, 40, 5], [70, 1, 1], [CHUNK_SIZE, 40, 5]])
    assert grid.set_voxels(cells, 6) == 5
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 1, 0), (2, 0, 0), (0, 1, 0)}


def test_set_voxels_matches_set_voxel_dirty_marking_random():
    rng = np.random.default_rng(7)
    size = (70, 33, 64)
    cells = np.stack([rng.integers(0, d, size=200) for d in size], axis=1)
    values = rng.integers(0, 3, size=200).astype(np.uint8)

    a = VoxelGrid(size)
    a.clear_dirty()
    for (x, y, z), v in zip(cells, values):
        a.set_voxel(int(x), int(y), int(z), int(v))

    b = VoxelGrid(size)
    b.clear_dirty()
    b.set_voxels(cells, values)

    assert np.array_equal(a.voxels, b.voxels)
    assert a.dirty_chunks() == b.dirty_chunks()


def test_set_voxels_ignores_unchanged_cells_for_dirty_marking():
    grid = VoxelGrid((64, 32, 32))
    grid.set_voxel(40, 1, 1, 5)
    grid.clear_dirty()
    cells = np.array([[40, 1, 1], [1, 1, 1]])
    assert grid.set_voxels(cells, 5) == 1
    assert grid.dirty_chunks() == {(0, 0, 0)}


def test_replace_voxels_changes_size_and_marks_all_dirty():
    grid = VoxelGrid((8, 8, 8))
    grid.clear_dirty()
    new = np.zeros((40, 33, 3), dtype=np.uint8)
    new[39, 32, 2] = 7
    grid.replace_voxels(new)
    assert grid.size == (40, 33, 3)
    assert grid.voxels.shape == (40, 33, 3)
    assert grid.chunk_counts == (2, 2, 1)
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)}
    assert grid.get_voxel(39, 32, 2) == 7


def test_replace_voxels_copies_the_array():
    grid = VoxelGrid((8, 8, 8))
    new = np.zeros((4, 4, 4), dtype=np.int32)
    new[1, 1, 1] = 3
    grid.replace_voxels(new)
    assert grid.voxels.dtype == np.uint8
    new[1, 1, 1] = 9
    new[0, 0, 0] = 1
    assert grid.get_voxel(1, 1, 1) == 3
    assert grid.get_voxel(0, 0, 0) == 0


def test_replace_voxels_rejects_bad_sizes():
    grid = VoxelGrid((8, 8, 8))
    with pytest.raises(ValueError, match="out of range"):
        grid.replace_voxels(np.zeros((257, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="out of range"):
        grid.replace_voxels(np.zeros((1, 0, 1), dtype=np.uint8))
    with pytest.raises(ValueError):
        grid.replace_voxels(np.zeros((4, 4), dtype=np.uint8))
    assert grid.size == (8, 8, 8)
    assert grid.voxels.shape == (8, 8, 8)


def test_chunks_of_is_unique_and_sorted():
    grid = VoxelGrid((96, 64, 32))
    cells = np.array([[70, 1, 1], [1, 1, 1], [2, 2, 2], [40, 40, 5], [CHUNK_SIZE, 40, 5]])
    assert grid.chunks_of(cells) == [(0, 0, 0), (1, 1, 0), (2, 0, 0)]
    assert grid.chunks_of(np.zeros((0, 3), dtype=np.intp)) == []
    assert grid.chunks_of([(95, 63, 31)]) == [(2, 1, 0)]


def test_set_voxels_rejects_cells_outside_the_grid():
    grid = VoxelGrid((8, 8, 8))
    with pytest.raises(IndexError):
        grid.set_voxels(np.array([[1, 1, 1], [-1, 0, 0]]), 3)
    with pytest.raises(IndexError):
        grid.set_voxels(np.array([[8, 0, 0]]), 3)
    assert int(grid.voxels.sum()) == 0


# ------------------------------------------------------------ section plane


def cut_grid():
    """A 64 x 32 x 32 grid with a 2 x 2 x 2 block, the z = 5 layer of it cut."""
    grid = VoxelGrid((64, 32, 32))
    grid.set_voxels(np.array([[x, y, z] for x in (4, 5) for y in (4, 5) for z in (4, 5)]), 3)
    grid.clear_dirty()
    grid.set_section(5)
    return grid


def test_visible_is_voxels_while_nothing_is_cut():
    grid = VoxelGrid((8, 8, 8))
    assert grid.visible is grid.voxels
    grid.set_voxel(1, 1, 1, 2)
    assert grid.visible is grid.voxels


def test_set_section_zeroes_the_cut_cells_in_visible_only():
    grid = cut_grid()
    assert grid.visible is not grid.voxels
    assert grid.visible.dtype == np.uint8
    assert grid.voxels[4, 4, 5] == 3
    assert grid.visible[4, 4, 5] == 0
    assert grid.visible[5, 5, 4] == 3
    assert int((grid.visible != 0).sum()) == 4
    assert int((grid.voxels != 0).sum()) == 8


def test_set_section_marks_every_chunk_dirty():
    grid = cut_grid()
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}
    grid.clear_dirty()
    grid.set_section(None)
    assert grid.visible is grid.voxels
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}


def test_write_into_a_cut_cell_changes_nothing():
    grid = cut_grid()
    grid.clear_dirty()
    assert grid.set_voxel(4, 4, 5, 7) is False
    assert grid.set_voxel(4, 4, 5, 0) is False
    assert grid.set_voxels(np.array([[4, 4, 5]]), 7) == 0
    assert grid.set_voxels(np.array([[4, 4, 5]]), np.array([0], dtype=np.uint8)) == 0
    assert grid.voxels[4, 4, 5] == 3
    assert grid.visible[4, 4, 5] == 0
    assert grid.dirty_chunks() == frozenset()


def test_write_next_to_a_cut_cell_updates_both_arrays():
    grid = cut_grid()
    grid.clear_dirty()
    assert grid.set_voxel(3, 4, 4, 9) is True
    assert grid.voxels[3, 4, 4] == 9
    assert grid.visible[3, 4, 4] == 9
    assert grid.set_voxel(5, 4, 4, 0) is True
    assert grid.voxels[5, 4, 4] == 0
    assert grid.visible[5, 4, 4] == 0
    assert grid.dirty_chunks() == {(0, 0, 0)}


def test_set_voxels_drops_cut_cells_from_the_count():
    grid = cut_grid()
    grid.clear_dirty()
    cells = np.array([[4, 4, 5], [3, 4, 4], [5, 5, 4], [40, 1, 1]])
    values = np.array([7, 7, 3, 2], dtype=np.uint8)
    # (4, 4, 5) is cut and (5, 5, 4) already holds 3.
    assert grid.set_voxels(cells, values) == 2
    assert grid.voxels[4, 4, 5] == 3
    assert grid.visible[4, 4, 5] == 0
    assert grid.voxels[3, 4, 4] == 7 and grid.visible[3, 4, 4] == 7
    assert grid.voxels[40, 1, 1] == 2 and grid.visible[40, 1, 1] == 2
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}
    assert grid.set_voxels(np.array([[4, 4, 5], [4, 5, 5]]), 3) == 0


def test_set_voxels_ignore_cut_writes_the_cut_cell_too():
    # A palette swap must remap a cut voxel's index too, or it shows the wrong
    # colour once the plane moves.
    grid = cut_grid()
    grid.clear_dirty()
    cells = np.array([[4, 4, 5], [3, 4, 4]])
    values = np.array([9, 9], dtype=np.uint8)
    assert grid.set_voxels(cells, values, ignore_cut=True) == 2
    assert grid.voxels[4, 4, 5] == 9  # the cut cell's own index changed
    assert grid.voxels[3, 4, 4] == 9


def test_set_voxels_ignore_cut_keeps_the_cut_cell_cut():
    grid = cut_grid()
    grid.clear_dirty()
    grid.set_voxels(np.array([[4, 4, 5]]), 9, ignore_cut=True)
    assert grid.visible[4, 4, 5] == 0  # still cut, just a different colour underneath
    grid.set_section(None)
    assert grid.visible[4, 4, 5] == 9  # shown again: the remapped colour, not the old one


def test_write_chunk_keeps_cut_cells_cut_in_visible():
    grid = cut_grid()
    data = grid.read_chunk((0, 0, 0))
    data[4, 4, 5] = 8  # the cut cell
    data[10, 10, 10] = 1  # beyond the plane, too
    data[10, 10, 2] = 1
    grid.write_chunk((0, 0, 0), data)
    assert grid.voxels[4, 4, 5] == 8
    assert grid.visible[4, 4, 5] == 0
    assert grid.visible[10, 10, 10] == 0
    assert grid.visible[10, 10, 2] == 1
    expected = grid.voxels.copy()
    expected[:, :, 5:] = 0
    assert np.array_equal(grid.visible, expected)


def test_visible_stays_in_step_with_random_writes():
    rng = np.random.default_rng(3)
    size = (40, 33, 20)
    grid = VoxelGrid(size)
    voxels = (rng.integers(0, 3, size=size)).astype(np.uint8)
    grid.replace_voxels(voxels)
    grid.set_section(12)
    cells = np.stack([rng.integers(0, d, size=300) for d in size], axis=1)
    values = rng.integers(0, 4, size=300).astype(np.uint8)
    for (x, y, z), v in zip(cells[:100], values[:100]):
        grid.set_voxel(int(x), int(y), int(z), int(v))
    grid.set_voxels(cells[100:], values[100:])
    # Cut cells are untouched; visible is voxels with the cut cells zeroed.
    assert np.array_equal(grid.voxels[:, :, 12:], voxels[:, :, 12:])
    expected = grid.voxels.copy()
    expected[:, :, 12:] = 0
    assert np.array_equal(grid.visible, expected)


def test_section_cuts_the_view_but_not_the_voxels():
    grid = VoxelGrid((4, 4, 6))
    grid.voxels[...] = 3
    grid.mark_all_dirty()
    grid.clear_dirty()
    grid.set_section(4)  # z from above, the old call
    assert grid.sections == {2: (4, True)} and grid.section == 4 and grid.section_axis == 2
    assert grid.voxels.all()
    assert grid.visible[:, :, :4].all() and not grid.visible[:, :, 4:].any()
    assert grid.tool_voxels is grid.visible
    assert len(grid.dirty_chunks()) > 0
    # A write beyond the plane is refused; below it shows.
    assert not grid.set_voxel(1, 1, 5, 7) and grid.voxels[1, 1, 5] == 3
    assert grid.set_voxels(np.array([[0, 0, 1], [0, 0, 5]]), np.array([9, 9], dtype=np.uint8)) == 1
    assert grid.visible[0, 0, 1] == 9 and grid.voxels[0, 0, 5] == 3
    # None takes the plane away.
    grid.set_section(None)
    assert grid.sections == {} and grid.section is None
    assert grid.visible is grid.voxels and grid.tool_voxels is grid.voxels
    # Out-of-range coordinates mean no plane; replace_voxels drops every plane.
    grid.set_section(0)
    assert grid.sections == {}
    grid.set_section(6)
    assert grid.sections == {}
    grid.set_section(2)
    grid.replace_voxels(np.zeros((4, 4, 6), dtype=np.uint8))
    assert grid.sections == {} and grid._visible is None


def test_section_planes_on_every_axis_and_from_either_side():
    grid = VoxelGrid((6, 5, 4))
    grid.voxels[...] = 2
    # Across x from the negative side: the cells below x 2 go.
    grid.set_section(2, axis=0, from_positive=False)
    assert grid.sections == {0: (2, False)} and grid.section is None and grid.section_axis == 0
    assert not grid.visible[:2].any() and grid.visible[2:].all()
    assert grid.is_cut(np.array([[1, 0, 0], [2, 0, 0]])).tolist() == [True, False]
    assert not grid.set_voxel(0, 0, 0, 5) and grid.set_voxel(2, 0, 0, 5)
    # A second plane across y from the positive side: both cuts hold.
    grid.set_section(3, axis=1, from_positive=True)
    assert set(grid.sections) == {0, 1} and grid.section_axis == 1
    assert not grid.visible[:, 3:].any() and grid.visible[2:, :3].all()
    assert grid.cut_slices() == [(0, slice(0, 2)), (1, slice(3, 5))]
    # Chunk-local slices for the undo path: a chunk wholly beyond the plane is all cut.
    bounds = (slice(0, 6), slice(3, 5), slice(0, 4))
    assert grid.cut_slices(bounds) == [(0, slice(0, 2)), (1, slice(0, 2))]
    grid.set_section(None, axis=0)
    assert set(grid.sections) == {1} and grid.visible[:2, :3].all()
    grid.set_section(None, axis=1)
    assert grid.sections == {} and grid.visible is grid.voxels


def test_undo_writes_keep_the_visible_copy_in_step_under_a_section():
    """Undo under a section plane changed the voxels but not the
    mesh, so hollowed boxes looked intact and vanished when the plane went."""
    grid = VoxelGrid((8, 8, 40))
    grid.voxels[2:6, 2:6, 0:12] = 5  # a box up to z 11
    grid.mark_all_dirty()
    grid.set_section(10)
    recorder = StrokeRecorder(grid)
    inside = np.argwhere(grid.voxels[3:5, 3:5, 1:9] != 0) + np.array([3, 3, 1])
    recorder.set_many(inside, 0)  # hollow it below the plane
    record = recorder.finish()
    assert record is not None
    assert grid.visible[3, 3, 5] == 0 and grid.voxels[3, 3, 5] == 0
    record.undo(grid)
    assert grid.voxels[3, 3, 5] == 5
    assert grid.visible[3, 3, 5] == 5  # the mesh copy follows the undo
    assert not grid.visible[:, :, 10:].any()  # and the cut stays
    record.redo(grid)
    assert grid.visible[3, 3, 5] == 0
    # A chunk wholly beyond the plane stays out of view through undo and redo too:
    # the write happened before the cut (beyond a plane a write is refused).
    grid.set_section(None)
    high = StrokeRecorder(grid)
    high.set_many(np.array([[2, 2, 36]]), 9)
    rec = high.finish()
    assert rec is not None and grid.voxels[2, 2, 36] == 9
    grid.set_section(3)
    assert grid.visible[2, 2, 36] == 0
    rec.undo(grid)
    assert grid.voxels[2, 2, 36] == 0 and grid.visible[2, 2, 36] == 0
    rec.redo(grid)
    assert grid.voxels[2, 2, 36] == 9 and grid.visible[2, 2, 36] == 0
    refused = StrokeRecorder(grid)
    assert refused.set_many(np.array([[2, 2, 37]]), 9) == 0 and refused.finish() is None
    grid.set_section(None)
    assert grid.visible is grid.voxels and grid.voxels[3, 3, 5] == 0



def test_set_grid_writes_what_set_voxels_would_write():
    """The mask-based write of a live turn. The same voxels,
    the same dirty chunks, the same respect for the lock and a section plane."""
    rng = np.random.default_rng(9)
    size = (70, 40, 33)
    start = rng.integers(0, 4, size, dtype=np.uint8)
    want = rng.integers(0, 4, size, dtype=np.uint8)
    lock = rng.random(size) < 0.1
    a, b = VoxelGrid(size, start.copy()), VoxelGrid(size, start.copy())
    for g in (a, b):
        g.lock = lock
        g.set_section(20, axis=0)
        g.clear_dirty()
    cells = np.argwhere(want != start)
    n_a = a.set_voxels(cells, want[cells[:, 0], cells[:, 1], cells[:, 2]])
    n_b = b.set_grid(want)
    assert n_a == n_b > 0
    assert np.array_equal(a.voxels, b.voxels) and np.array_equal(a.visible, b.visible)
    assert a.dirty_chunks() == b.dirty_chunks()
    assert a.locked_writes == b.locked_writes > 0
    assert b.set_grid(b.voxels.copy()) == 0  # nothing to change
    # Through the recorder: the record's undo gives the voxels before the write back.
    before = b.voxels.copy()
    recorder = StrokeRecorder(b)
    assert recorder.set_grid(start) > 0
    record = recorder.finish()
    assert record is not None
    record.undo(b)
    assert np.array_equal(b.voxels, before)


def test_copy_keeps_the_voxels_the_mask_and_the_brick_table():
    from rok.core import brick

    grid = VoxelGrid((8, 8, 8))
    grid.set_voxel(5, 5, 5, 4)
    grid.lock = np.zeros(grid.size, dtype=bool)
    grid.lock[5, 5, 5] = True
    brick.put(grid, [(0, 0, 0)], 2, 2, 7)
    grid.lock = np.zeros(grid.size, dtype=bool)  # put cleared it; a mask set after the first brick
    copy = grid.copy()
    assert np.array_equal(copy.voxels, grid.voxels) and copy.voxels is not grid.voxels
    assert copy.bricks == grid.bricks and copy.bricks is not grid.bricks
    assert copy.lock is not grid.lock and copy.palette is grid.palette
    copy.bricks.add(brick.Brick(4, 4, 0, 1, 1, 7))
    assert len(grid.bricks) == 1
    assert VoxelGrid((2, 2, 2)).copy().bricks is None


def test_mark_dirty_cells_marks_the_chunks_and_their_face_neighbours():
    grid = VoxelGrid((64, 32, 32))
    grid.clear_dirty()
    grid.mark_dirty_cells(np.array([[31, 3, 3]]))
    assert grid.dirty_chunks() == {(0, 0, 0), (1, 0, 0)}
    grid.clear_dirty()
    grid.mark_dirty_cells(np.zeros((0, 3), dtype=np.int64))
    assert grid.dirty_chunks() == frozenset()
