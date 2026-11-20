"""The voxel clipboard: copy a selection, paste it back, into another grid and
into another file's palette. Golden data are the model fixtures."""

import numpy as np

from rok.core.clipboard import Clip, copy_selection, paste, paste_cells, paste_origin
from rok.core.material import MaterialTable, carry_materials
from rok.core.palette import merge_claimed
from rok.core.select import Selection
from rok.core.vox_read import read_vox

from helpers import FIXTURES, ArraySetMany, model_with_materials


def _middle_selection(voxels) -> Selection:
    """The filled cells of the middle third of the model's bounding box."""
    filled = np.argwhere(voxels != 0)
    lo = filled.min(axis=0)
    extent = filled.max(axis=0) - lo + 1
    a = lo + extent // 3
    b = lo + 2 * (extent // 3)
    selection = Selection(voxels.shape)
    mask = np.zeros(voxels.shape, dtype=bool)
    mask[a[0] : b[0] + 1, a[1] : b[1] + 1, a[2] : b[2] + 1] = True
    selection.replace(mask & (voxels != 0))
    assert 0 < selection.count() < len(filled)
    return selection


# --- copy --------------------------------------------------------------------


def test_copy_takes_the_selection_box_with_its_palette_and_materials():
    voxels, palette, materials = model_with_materials("bird")
    selection = _middle_selection(voxels)
    clip = copy_selection(voxels, selection, palette, materials)
    assert isinstance(clip, Clip)
    lo, hi = selection.bounds()
    assert clip.origin == lo
    assert clip.size == tuple(hi[i] - lo[i] + 1 for i in range(3))
    assert clip.count() == selection.count()
    # Every selected voxel is in the box at its offset, nothing else is.
    cells = selection.cells()
    inside = clip.voxels[cells[:, 0] - lo[0], cells[:, 1] - lo[1], cells[:, 2] - lo[2]]
    assert np.array_equal(inside, voxels[cells[:, 0], cells[:, 1], cells[:, 2]])
    assert np.count_nonzero(clip.voxels) == len(cells)
    assert clip.voxels.dtype == np.uint8
    # The palette and the table are copies.
    assert np.array_equal(clip.palette, palette) and clip.palette is not palette
    assert clip.materials == materials and clip.materials is not materials
    palette[5] = (1, 2, 3, 255)
    assert not np.array_equal(clip.palette, palette)
    used = clip.used_counts()
    assert used[0] == 0 and used.sum() == clip.count()


def test_copy_of_an_empty_selection_is_none():
    voxels, palette, materials = model_with_materials("bird")
    selection = Selection(voxels.shape)
    assert copy_selection(voxels, selection, palette, materials) is None
    # A selection over empty cells only is nothing to copy either.
    empty = np.zeros(voxels.shape, dtype=bool)
    empty[0, 0, voxels.shape[2] - 1] = voxels[0, 0, voxels.shape[2] - 1] == 0
    if empty.any():
        selection.replace(empty)
        assert copy_selection(voxels, selection, palette, materials) is None


# --- placement ---------------------------------------------------------------


def test_paste_origin_keeps_a_fitting_origin_and_moves_the_rest_inside():
    assert paste_origin((4, 4, 4), (3, 5, 0), (16, 16, 16)) == (3, 5, 0)
    assert paste_origin((4, 4, 4), (12, 12, 12), (16, 16, 16)) == (12, 12, 12)
    assert paste_origin((4, 4, 4), (13, 14, 20), (16, 16, 16)) == (12, 12, 12)
    assert paste_origin((4, 4, 4), (-2, 0, 0), (16, 16, 16)) == (0, 0, 0)
    # Larger than the grid on an axis: start at 0 there.
    assert paste_origin((20, 4, 4), (3, 3, 3), (16, 16, 16)) == (0, 3, 3)


def test_paste_cells_drops_cells_outside_and_remaps():
    clip = Clip(np.array([[[5, 0], [0, 7]]], dtype=np.uint8), (0, 0, 0), np.zeros((256, 4), np.uint8), MaterialTable())
    cells, values = paste_cells(clip, (0, 0, 0), (1, 2, 2))
    assert cells.tolist() == [[0, 0, 0], [0, 1, 1]] and values.tolist() == [5, 7]
    cells, values = paste_cells(clip, (0, 1, 1), (1, 2, 2))
    assert cells.tolist() == [[0, 1, 1]] and values.tolist() == [5]
    remap = np.arange(256, dtype=np.uint8)
    remap[5] = 9
    cells, values = paste_cells(clip, (0, 0, 0), (1, 2, 2), remap=remap)
    assert values.tolist() == [9, 7] and values.dtype == np.uint8


# --- paste -------------------------------------------------------------------


def test_cut_and_paste_in_place_restores_the_fixture():
    voxels, palette, materials = model_with_materials("bird")
    original = voxels.copy()
    selection = _middle_selection(voxels)
    clip = copy_selection(voxels, selection, palette, materials)
    cells = selection.cells()
    voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = 0  # the cut
    assert not np.array_equal(voxels, original)
    writer = ArraySetMany(voxels)
    mask = paste(voxels, clip, paste_origin(clip.size, clip.origin, voxels.shape), writer)
    assert np.array_equal(voxels, original)
    assert np.array_equal(mask, selection.mask)
    assert len(writer.calls) == 1


def test_paste_writes_filled_cells_only_and_overwrites():
    voxels = np.zeros((4, 4, 4), dtype=np.uint8)
    voxels[...] = 3
    clip = Clip(np.array([[[5, 0], [0, 7]]], dtype=np.uint8), (0, 0, 0), np.zeros((256, 4), np.uint8), MaterialTable())
    mask = paste(voxels, clip, (1, 1, 1), ArraySetMany(voxels))
    assert voxels[1, 1, 1] == 5 and voxels[1, 2, 2] == 7
    assert voxels[1, 1, 2] == 3 and voxels[1, 2, 1] == 3  # the clip's empty cells
    assert int(mask.sum()) == 2 and mask[1, 1, 1] and mask[1, 2, 2]


def test_paste_into_a_smaller_grid_is_moved_inside_and_cut_off():
    voxels, palette, materials = model_with_materials("bird")
    selection = _middle_selection(voxels)
    clip = copy_selection(voxels, selection, palette, materials)
    small = np.zeros((8, 8, 8), dtype=np.uint8)
    origin = paste_origin(clip.size, clip.origin, small.shape)
    assert all(0 <= origin[i] <= max(8 - clip.size[i], 0) for i in range(3))
    assert any(clip.size[i] > 8 for i in range(3))  # the box is really cut off
    mask = paste(small, clip, origin, ArraySetMany(small))
    expected = np.zeros((8, 8, 8), dtype=np.uint8)
    fit = tuple(min(clip.size[i], 8 - origin[i]) for i in range(3))
    expected[
        origin[0] : origin[0] + fit[0], origin[1] : origin[1] + fit[1], origin[2] : origin[2] + fit[2]
    ] = clip.voxels[: fit[0], : fit[1], : fit[2]]
    assert np.array_equal(small, expected)
    assert np.array_equal(mask, expected != 0)
    # Nothing at all when the box lands entirely outside.
    nothing = np.zeros((8, 8, 8), dtype=np.uint8)
    mask = paste(nothing, clip, (8, 8, 8), ArraySetMany(nothing))
    assert not mask.any() and not nothing.any()


def test_paste_into_another_file_keeps_the_colours_through_the_merge():
    src_voxels, src_palette, src_materials = model_with_materials("bird")
    dst_voxels, dst_palette, dst_materials = model_with_materials("robot")
    selection = _middle_selection(src_voxels)
    clip = copy_selection(src_voxels, selection, src_palette, src_materials)

    dst_used = np.bincount(dst_voxels.ravel(), minlength=256)
    merged, remap, claimed_by = merge_claimed(dst_palette, dst_used, clip.palette, clip.used_counts())
    materials = carry_materials(dst_materials, clip.materials, claimed_by)
    origin = paste_origin(clip.size, clip.origin, dst_voxels.shape)
    mask = paste(dst_voxels, clip, origin, ArraySetMany(dst_voxels), remap)
    assert mask.any()

    # Every pasted voxel shows the colour it had in the bird (no slot ran out).
    cells = np.argwhere(mask)
    local = cells - np.asarray(origin)
    src_index = clip.voxels[local[:, 0], local[:, 1], local[:, 2]]
    dst_index = dst_voxels[cells[:, 0], cells[:, 1], cells[:, 2]]
    assert np.array_equal(merged[dst_index], src_palette[src_index])
    # The merge changed the palette, and the robot's own colours stayed.
    assert not np.array_equal(merged, dst_palette)
    for k in np.nonzero(dst_used[1:])[0] + 1:
        assert np.array_equal(merged[k], dst_palette[k])
    # A claimed slot carries the bird's material.
    for k in np.nonzero(claimed_by)[0]:
        assert materials.get(int(k)) == src_materials.get(int(claimed_by[k]))


def test_paste_into_the_same_file_changes_nothing_in_the_palette():
    voxels, palette, materials = model_with_materials("bird")
    selection = _middle_selection(voxels)
    clip = copy_selection(voxels, selection, palette, materials)
    used = np.bincount(voxels.ravel(), minlength=256)
    merged, remap, claimed_by = merge_claimed(palette, used, clip.palette, clip.used_counts())
    assert np.array_equal(merged, palette)
    assert np.array_equal(remap, np.arange(256))
    assert carry_materials(materials, clip.materials, claimed_by) == materials
    # After a cut the copied colours may have no voxel left; the slots are free but
    # hold the same colour, so the indices still stay.
    cells = selection.cells()
    voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = 0
    used = np.bincount(voxels.ravel(), minlength=256)
    merged, remap, _ = merge_claimed(palette, used, clip.palette, clip.used_counts())
    assert np.array_equal(merged, palette)
    assert np.array_equal(remap, np.arange(256))


def test_read_vox_and_read_scene_agree_on_the_fixture():
    # The helper above reads through read_scene; the single-model reader gives the same array.
    model = read_vox(FIXTURES / "bird.vox")
    voxels, _, _ = model_with_materials("bird")
    assert np.array_equal(model.voxels, voxels)
