"""The pattern brush: orientation, placement on a face, the stamp rules per tool
and the PatternGesture. Small arrays plus the bird fixture as a stamp."""

import numpy as np

from rok.core.clipboard import copy_selection
from rok.core.edit import ATTACH, ERASE, PAINT, PATTERN, SELECT
from rok.core.gesture import PatternGesture, make_gesture
from rok.core.material import MaterialTable
from rok.core.pattern import (
    IDENTITY,
    anchor_cell,
    box_cells,
    mirror_pairs,
    oriented,
    placement,
    stamp,
    turned,
)
from rok.core.raycast import Hit, raycast
from rok.core.select import Selection
from rok.core.vox_read import read_vox

from helpers import FIXTURES, ArraySetMany

SIZE = (8, 8, 8)
NO_MIRROR = (False, False, False)


def _l_shape() -> np.ndarray:
    """3 x 2 x 1: an L of colours 1..4, so turns are visible."""
    box = np.zeros((3, 2, 1), dtype=np.uint8)
    box[0, 0, 0] = 1
    box[1, 0, 0] = 2
    box[2, 0, 0] = 3
    box[2, 1, 0] = 4
    return box


def filled(voxels) -> dict:
    return {tuple(c): int(voxels[tuple(c)]) for c in np.argwhere(voxels != 0).tolist()}


# --- orientation -------------------------------------------------------------


def test_turned_composes_quarter_turns_and_four_make_identity():
    r = IDENTITY
    for _ in range(4):
        r = turned(r, 2)
    assert r == IDENTITY
    r = turned(turned(IDENTITY, 2), 0)
    assert r != turned(turned(IDENTITY, 0), 2)  # order matters, as for real turns
    assert len(r) == 9 and all(isinstance(v, int) for v in r)


def test_oriented_turns_the_box_about_z():
    box = _l_shape()
    once = oriented(box, turned(IDENTITY, 2))
    assert once.shape == (2, 3, 1)
    # (x, y) -> (-y, x): the row along x becomes a row along y.
    assert once.dtype == np.uint8 and once.flags.c_contiguous
    assert sorted(once.ravel().tolist()) == sorted(box.ravel().tolist())
    assert once[1, 0, 0] == 1 and once[1, 1, 0] == 2 and once[1, 2, 0] == 3 and once[0, 2, 0] == 4
    assert np.array_equal(oriented(box, IDENTITY), box)
    twice = oriented(box, turned(turned(IDENTITY, 2), 2))
    assert np.array_equal(twice, box[::-1, ::-1, :])


def test_oriented_tips_the_box_about_x():
    box = _l_shape()
    tipped = oriented(box, turned(IDENTITY, 0))
    assert tipped.shape == (3, 1, 2)  # y and z swap
    assert sorted(tipped.ravel().tolist()) == sorted(box.ravel().tolist())


# --- placement ---------------------------------------------------------------


def test_anchor_cell_is_the_filled_cell_nearest_the_middle_of_the_near_layer():
    box = _l_shape()  # z layer 0 holds (0, 0), (1, 0), (2, 0), (2, 1); middle (1, 0.5)
    assert anchor_cell(box, 2, 0) == (1, 0, 0)
    # Along x: layer 0 has (0, 0) only, layer 2 has (0, 0) and (1, 0); middle (0.5, 0).
    assert anchor_cell(box, 0, 0) == (0, 0, 0)
    assert anchor_cell(box, 0, 2) == (2, 0, 0)
    # A stair: the bottom layer's only cell is the anchor, wherever the middle is.
    stair = np.zeros((4, 1, 4), dtype=np.uint8)
    for i in range(4):
        stair[i, 0, i] = 1
    assert anchor_cell(stair, 2, 0) == (0, 0, 0)
    assert anchor_cell(stair, 2, 3) == (3, 0, 3)


def test_placement_attach_puts_a_filled_cell_of_the_near_layer_under_the_mouse():
    box = _l_shape()
    # On the floor at (4, 4): the anchor cell (1, 0, 0) lands there.
    floor = Hit((4, 4, -1), (0, 0, 1), 1.0, wall=True)
    assert placement(box, floor, ATTACH) == (3, 4, 0)
    # On top of a voxel at z 2: the target is z 3.
    top = Hit((4, 4, 2), (0, 0, 1), 1.0)
    assert placement(box, top, ATTACH) == (3, 4, 3)
    # On the -y face of a voxel at y 5: the box's y layer 1 touches the face.
    # Layer y 1 holds (2, 1) only, so cell (2, 1, 0) lands in the target (4, 4, 2).
    front = Hit((4, 5, 2), (0, -1, 0), 1.0)
    assert placement(box, front, ATTACH) == (2, 3, 2)
    # On the +x face: x layer 0 holds (0, 0) only; it lands in the target (5, 4, 2).
    side = Hit((4, 4, 2), (1, 0, 0), 1.0)
    assert placement(box, side, ATTACH) == (5, 4, 2)


def test_placement_stacks_a_stair_flush_on_its_top_step():
    """A case from use: the stamp must sit on the voxels, not float."""
    stair = np.zeros((4, 1, 4), dtype=np.uint8)
    for i in range(4):
        stair[i, 0, i] = 1
    voxels = np.zeros((16, 8, 16), dtype=np.uint8)
    voxels[2:6, 3, 0:4] = stair[:, 0, :]  # the first stamp, cells (2+i, 3, i)
    top = raycast(voxels, (5.5, 3.5, 50.0), (0.0, 0.0, -1.0))
    assert top.voxel == (5, 3, 3) and top.normal == (0, 0, 1)
    origin = placement(stair, top, ATTACH)
    assert origin == (5, 3, 4)
    cells, _ = box_cells(stair, origin, voxels.shape)
    assert (5, 3, 4) in {tuple(c) for c in cells.tolist()}  # flush on the top step
    stamp(voxels, ATTACH, *box_cells(stair, origin, voxels.shape), ArraySetMany(voxels))
    assert int(np.count_nonzero(voxels)) == 8
    # From the side face of a step the stamp stands next to it, again flush.
    side = raycast(voxels, (50.0, 3.5, 1.5), (-1.0, 0.0, 0.0))
    assert side.voxel == (3, 3, 1) and side.normal == (1, 0, 0)
    origin = placement(stair, side, ATTACH)
    assert origin == (4, 3, 1)


def test_placement_erase_paint_select_reach_into_the_model():
    box = np.ones((2, 2, 3), dtype=np.uint8)
    top = Hit((4, 4, 2), (0, 0, 1), 1.0)
    # The z layer 2 touches the face; its anchor cell (0, 0, 2) lands in the hit voxel.
    assert placement(box, top, ERASE) == (4, 4, 0)
    assert placement(box, top, PAINT) == (4, 4, 0)
    assert placement(box, top, SELECT) == (4, 4, 0)
    front = Hit((4, 5, 2), (0, -1, 0), 1.0)
    assert placement(np.ones((2, 3, 2), dtype=np.uint8), front, ERASE) == (4, 5, 2)
    # A wall has nothing to erase, paint or select.
    floor = Hit((4, 4, -1), (0, 0, 1), 1.0, wall=True)
    assert placement(box, floor, ERASE) is None
    assert placement(box, floor, SELECT) is None


def test_box_cells_clips_and_remaps():
    box = _l_shape()
    cells, values = box_cells(box, (6, 7, 0), SIZE)
    assert cells.tolist() == [[6, 7, 0], [7, 7, 0]] and values.tolist() == [1, 2]
    remap = np.arange(256, dtype=np.uint8)
    remap[1] = 9
    cells, values = box_cells(box, (0, 0, 0), SIZE, remap)
    assert values.tolist() == [9, 2, 3, 4]


def test_mirror_pairs_keeps_values_and_the_first_of_a_shared_cell():
    cells = np.array([[0, 1, 1], [7, 1, 1]])
    values = np.array([5, 6], dtype=np.uint8)
    out_cells, out_values = mirror_pairs(cells, values, SIZE, (True, False, False))
    assert out_cells.tolist() == [[0, 1, 1], [7, 1, 1]]
    assert out_values.tolist() == [5, 6]  # each cell is the other's image; the first wins
    out_cells, out_values = mirror_pairs(np.array([[1, 1, 1]]), np.array([3]), SIZE, (True, True, False))
    assert sorted(out_cells.tolist()) == [[1, 1, 1], [1, 6, 1], [6, 1, 1], [6, 6, 1]]
    assert out_values.tolist() == [3, 3, 3, 3]


# --- stamp rules -------------------------------------------------------------


def test_stamp_attach_fills_empty_cells_only():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 2, 0] = 9
    cells, values = box_cells(_l_shape(), (2, 2, 0), SIZE)
    written = stamp(voxels, ATTACH, cells, values, ArraySetMany(voxels))
    assert filled(voxels) == {(2, 2, 0): 9, (3, 2, 0): 2, (4, 2, 0): 3, (4, 3, 0): 4}
    assert sorted(written.tolist()) == [[3, 2, 0], [4, 2, 0], [4, 3, 0]]


def test_stamp_erase_and_paint_act_on_filled_cells_only():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0] = 7
    cells, values = box_cells(_l_shape(), (2, 2, 0), SIZE)
    written = stamp(voxels, PAINT, cells, values, ArraySetMany(voxels))
    assert len(written) == 4
    assert voxels[2, 2, 0] == 1 and voxels[4, 3, 0] == 4 and voxels[0, 0, 0] == 7
    assert len(stamp(voxels, PAINT, cells, values, ArraySetMany(voxels))) == 0  # no change
    written = stamp(voxels, ERASE, cells, values, ArraySetMany(voxels))
    assert len(written) == 4 and int(np.count_nonzero(voxels)) == 64 - 4
    assert len(stamp(voxels, ERASE, cells, values, ArraySetMany(voxels))) == 0
    assert len(stamp(voxels, SELECT, cells, values, ArraySetMany(voxels))) == 0


def test_stamp_with_mirror_writes_the_images():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    cells, values = box_cells(_l_shape(), (0, 0, 0), SIZE)
    written = stamp(voxels, ATTACH, cells, values, ArraySetMany(voxels), (True, False, False))
    assert len(written) == 8
    assert voxels[0, 0, 0] == 1 and voxels[7, 0, 0] == 1 and voxels[5, 1, 0] == 4


# --- the gesture -------------------------------------------------------------


def test_pattern_gesture_attach_stamps_on_press_and_again_when_the_box_moves():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    writer = ArraySetMany(voxels)
    gesture = PatternGesture(voxels, ATTACH, _l_shape(), NO_MIRROR, writer, None, colour=6)
    floor = raycast(voxels, (2.5, 2.5, 50.0), (0.0, 0.0, -1.0))
    assert floor.wall
    assert gesture.begin(floor)
    assert gesture.origin == (1, 2, 0)
    # Attach stamps the shape in the current colour, not the stamp's own colours.
    assert filled(voxels) == {(1, 2, 0): 6, (2, 2, 0): 6, (3, 2, 0): 6, (3, 3, 0): 6}
    preview = gesture.preview()
    assert preview.kind == "CELLS" and preview.label == "3 x 2 x 1" and len(preview.cells) == 4
    # The same place again: nothing. The ray passes over the placed voxels and
    # still hits the floor, so the box does not climb onto itself.
    assert not gesture.extend((2.5, 2.5, 50.0), (0.0, 0.0, -1.0))
    # Two cells over: a second stamp, overlapping cells stay as they were.
    assert gesture.extend((4.5, 2.5, 50.0), (0.0, 0.0, -1.0))
    assert gesture.origin == (3, 2, 0)
    assert voxels[3, 2, 0] == 6 and voxels[4, 2, 0] == 6 and voxels[5, 2, 0] == 6 and voxels[5, 3, 0] == 6
    assert int(np.count_nonzero(voxels[:, :, 1:])) == 0
    assert gesture.finish() == 0


def test_pattern_gesture_erase_goes_one_stamp_deep():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0:4] = 7
    box = np.ones((2, 2, 2), dtype=np.uint8)
    gesture = PatternGesture(voxels, ERASE, box, NO_MIRROR, ArraySetMany(voxels), None)
    top = raycast(voxels, (4.5, 4.5, 50.0), (0.0, 0.0, -1.0))
    assert top.voxel == (4, 4, 3)
    assert gesture.begin(top)
    assert gesture.origin == (4, 4, 2)
    assert int(np.count_nonzero(voxels)) == 64 * 4 - 8
    # Hovering the hole again: the erased cells still stop the ray, so the box
    # stays where it was and nothing more goes.
    assert not gesture.extend((4.5, 4.5, 50.0), (0.0, 0.0, -1.0))
    assert int(np.count_nonzero(voxels)) == 64 * 4 - 8


def test_pattern_gesture_select_and_paint_and_wall_rules():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0] = 7
    selection = Selection(SIZE)
    box = np.ones((2, 2, 1), dtype=np.uint8) * 3
    gesture = PatternGesture(voxels, SELECT, box, NO_MIRROR, ArraySetMany(voxels), selection)
    assert gesture.begin(raycast(voxels, (4.5, 4.5, 50.0), (0.0, 0.0, -1.0)))
    assert selection.count() == 4 and selection.contains((4, 4, 0)) and selection.contains((5, 5, 0))
    assert int(np.count_nonzero(voxels == 3)) == 0
    paint = PatternGesture(voxels, PAINT, box, NO_MIRROR, ArraySetMany(voxels), None)
    assert paint.begin(raycast(voxels, (1.5, 1.5, 50.0), (0.0, 0.0, -1.0)))
    assert int(np.count_nonzero(voxels == 3)) == 4
    # A wall hit with Erase: nothing to do.
    empty = np.zeros(SIZE, dtype=np.uint8)
    erase = PatternGesture(empty, ERASE, box, NO_MIRROR, ArraySetMany(empty), None)
    assert not erase.begin(raycast(empty, (1.5, 1.5, 50.0), (0.0, 0.0, -1.0)))


def test_pattern_gesture_paint_keeps_the_stamp_colours_through_the_remap():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0] = 7
    remap = np.arange(256, dtype=np.uint8)
    remap[1:5] = [11, 12, 13, 14]
    gesture = PatternGesture(voxels, PAINT, _l_shape(), NO_MIRROR, ArraySetMany(voxels), None, remap, colour=6)
    assert gesture.begin(raycast(voxels, (2.5, 2.5, 50.0), (0.0, 0.0, -1.0)))
    assert sorted(set(voxels[(voxels != 0) & (voxels != 7)].tolist())) == [11, 12, 13, 14]
    # Attach with no colour given (0) falls back to the stamp's colours.
    empty = np.zeros(SIZE, dtype=np.uint8)
    plain = PatternGesture(empty, ATTACH, _l_shape(), NO_MIRROR, ArraySetMany(empty), None, remap)
    assert plain.begin(raycast(empty, (2.5, 2.5, 50.0), (0.0, 0.0, -1.0)))
    assert sorted(set(empty[empty != 0].tolist())) == [11, 12, 13, 14]


def test_make_gesture_pattern_needs_a_box():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    writer = ArraySetMany(voxels)
    assert make_gesture(PATTERN, ATTACH, voxels, 1, NO_MIRROR, None, writer, None) is None
    g = make_gesture(PATTERN, ATTACH, voxels, 1, NO_MIRROR, None, writer, None, pattern_box=_l_shape())
    assert isinstance(g, PatternGesture)


def test_a_fixture_part_stamps_onto_an_empty_grid_intact():
    model = read_vox(FIXTURES / "bird.vox")
    voxels = model.voxels.copy()
    # A 6-cell box in the middle of the bird's bounds.
    filled_cells = np.argwhere(voxels != 0)
    centre = (filled_cells.min(axis=0) + filled_cells.max(axis=0)) // 2
    lo = centre - 2
    hi = centre + 3
    selection = Selection(voxels.shape)
    mask = np.zeros(voxels.shape, dtype=bool)
    mask[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = True
    selection.replace(mask & (voxels != 0))
    clip = copy_selection(voxels, selection, model.palette, MaterialTable())
    assert clip is not None and clip.count() > 1
    target = np.zeros((32, 32, 32), dtype=np.uint8)
    gesture = PatternGesture(target, ATTACH, clip.voxels, NO_MIRROR, ArraySetMany(target), None, colour=9)
    assert gesture.begin(raycast(target, (16.5, 16.5, 50.0), (0.0, 0.0, -1.0)))
    origin = gesture.origin
    assert origin[2] == 0
    part = target[
        origin[0] : origin[0] + clip.size[0],
        origin[1] : origin[1] + clip.size[1],
        origin[2] : origin[2] + clip.size[2],
    ]
    assert np.array_equal(part != 0, clip.voxels != 0)  # the shape, whole
    assert (part[part != 0] == 9).all()  # in the current colour
    assert int(np.count_nonzero(target)) == clip.count()
