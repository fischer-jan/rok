import itertools

import numpy as np

from rok.core import grid_ops, link
from rok.core.grid import VoxelGrid
from rok.core.scene import VoxScene
from rok.core.undo import GridRecord, ModelRecord

# Every signed permutation a .vox file can store in an _r byte.
ORIENTS = []
for b in range(128):
    try:
        ORIENTS.append(link.as_orient(grid_ops.rotation_from_byte(b)))
    except ValueError:
        pass


def _shape():
    """An L of three colours in a 4 x 3 x 2 box: no turn or flip maps it onto itself."""
    v = np.zeros((4, 3, 2), dtype=np.uint8)
    v[:, 0, 0] = 1
    v[0, 1:, 0] = 2
    v[0, 0, 1] = 3
    return v


def _linked_pair(orient):
    """A scene of the shape and a linked copy turned by `orient`, at x 20."""
    source = _shape()
    scene = VoxScene.single(VoxelGrid(source.shape, source.copy()), (0, 0, 0))
    turned = link.from_source(source, orient)
    scene.add(VoxelGrid(turned.shape, turned), "copy", (20, 0, 0))
    a, b = scene.models
    a.link = b.link = 1
    b.orient = orient
    return scene, a, b


def test_there_are_48_orients_and_each_round_trips():
    assert len(set(ORIENTS)) == 48
    v = _shape()
    for orient in ORIENTS:
        assert np.array_equal(link.to_source(link.from_source(v, orient), orient), v)
        assert link.source_size(link.from_source(v, orient).shape, orient) == v.shape


def test_a_painted_voxel_shows_in_the_turned_copy_for_every_orient():
    for orient in ORIENTS:
        scene, a, b = _linked_pair(orient)
        a.grid.set_voxel(3, 2, 1, 9)
        assert link.sync(scene, a.uid) == [b.uid]
        assert np.array_equal(b.grid.voxels, link.from_source(a.grid.voxels, orient))
        assert tuple(b.position) == (20, 0, 0)  # the same size: nothing moves
        # And back the other way: an edit in the copy shows in the original.
        b.grid.set_voxels(np.argwhere(b.grid.voxels == 2), 7)
        link.sync(scene, b.uid)
        assert (a.grid.voxels == 7).sum() == 2 and not (a.grid.voxels == 2).any()


def test_a_box_that_grows_at_its_low_end_moves_each_copy_the_same_way_for_every_orient():
    """Grow the original by two cells at its low x end (its box moves -2 in x).
    In every copy the new cells sit on the side where the shape grew, so the
    copy's far side stays put."""
    for orient in ORIENTS:
        scene, a, b = _linked_pair(orient)
        grown = np.zeros((6, 3, 2), dtype=np.uint8)
        grown[2:] = a.grid.voxels
        grown[0, 0, 0] = 5  # a mark at the new low end
        a.grid.replace_voxels(grown)
        a.position = (-2, 0, 0)
        link.sync(scene, a.uid, (-2, 0, 0))
        assert np.array_equal(b.grid.voxels, link.from_source(grown, orient))
        # The shape's old cells stay where they were in the world in the copy too:
        # the old source cell (0, 0, 0) is now source cell (2, 0, 0).
        def world_of(model, source_cell, source_shape):
            R = link.matrix(model.orient)
            _, shift = grid_ops.apply_rotation(np.zeros(source_shape, dtype=np.uint8), R)
            return tuple(int(v) for v in R @ np.asarray(source_cell) + shift + np.asarray(model.position))

        old = _linked_pair(orient)[2]  # the copy as it was before the growth
        assert world_of(old, (0, 0, 0), (4, 3, 2)) == world_of(b, (2, 0, 0), (6, 3, 2))


def test_the_shift_helpers_undo_each_other():
    for orient, d in itertools.product(ORIENTS, [(0, 0, 0), (-2, 1, 3), (4, -1, 0)]):
        old, new = (4, 3, 2), (6, 5, 2)
        s = link.source_shift(d, orient, old, new)
        # (sizes in source cells; a member's world sizes follow its orient)
        assert link.placed_shift(s, orient, old, new) == tuple(d)


def test_a_turn_in_the_scene_is_not_passed_on_but_a_paint_is():
    scene, a, b = _linked_pair(link.IDENTITY)
    paint = ModelRecord(a.uid, GridRecord(a.grid.voxels.copy(), a.grid.voxels.copy()))
    turn = ModelRecord(a.uid, GridRecord(a.grid.voxels.copy(), a.grid.voxels.copy()),
                       position_before=(0, 0, 0), position_after=(1, 0, 0),
                       orient_before=link.IDENTITY, orient_after=ORIENTS[5])
    assert link.voxel_changes(paint, undone=False) == [(a.uid, (0, 0, 0))]
    assert link.voxel_changes(turn, undone=False) == []


def test_undo_brings_back_the_link_and_the_turn():
    scene, a, b = _linked_pair(ORIENTS[7])
    record = ModelRecord(b.uid, None, orient_before=b.orient, orient_after=link.IDENTITY, link_before=1, link_after=0)
    record.redo(scene)
    assert b.link == 0 and b.orient == link.IDENTITY and not link.is_linked(scene, a)
    record.undo(scene)
    assert b.link == 1 and b.orient == ORIENTS[7] and link.is_linked(scene, a)


def test_linked_copies_save_as_one_shape_shown_twice_and_come_back_linked():
    from rok.core.vox_read import loads_scene as read_scene_bytes
    from rok.core.vox_write import dumps_scene

    turn = ORIENTS[11]
    scene, a, b = _linked_pair(turn)
    scene.add(VoxelGrid((2, 2, 2)), "alone", (40, 0, 0))
    data = dumps_scene(scene.to_data())
    assert data.count(b"XYZI") == 2  # the shape once, the lone model once
    back = VoxScene.from_data(read_scene_bytes(data))
    a2, b2, c2 = back.models
    assert a2.link and a2.link == b2.link and not c2.link
    assert b2.orient == turn and a2.orient == link.IDENTITY
    for old, new in ((a, a2), (b, b2)):
        assert tuple(new.position) == tuple(old.position)
        assert np.array_equal(new.grid.voxels, old.grid.voxels)
