import numpy as np
import pytest

from rok.core.scene_move import (
    drag_offset,
    drop_distance,
    next_in_cycle,
    nudge_axes,
    ray_plane_facing,
    ray_plane_z,
)


def test_a_ray_meets_the_floor_plane_ahead_of_it_only():
    hit = ray_plane_z((0, 0, 10), (0.6, 0, -0.8), 2.0)
    assert np.allclose(hit, (6.0, 0.0, 2.0))
    assert ray_plane_z((0, 0, 10), (1, 0, 0), 2.0) is None  # parallel
    assert ray_plane_z((0, 0, 10), (0, 0, 1), 2.0) is None  # pointing away


def test_the_upright_plane_faces_the_view():
    hit = ray_plane_facing((0, -20, 5), (0, 1, 0.1), (3, 0, 0), (0, 1, -0.5))
    assert np.allclose(hit, (0.0, 0.0, 7.0))
    assert ray_plane_facing((0, 0, 20), (0, 0, -1), (0, 0, 0), (0, 0, -1)) is None  # looking straight down


def test_a_drag_moves_on_the_floor_or_straight_up_in_whole_cells():
    assert drag_offset((1.0, 1.0, 4.0), (4.6, -2.2, 9.0), vertical=False) == (4, -3, 0)
    assert drag_offset((1.0, 1.0, 4.0), (4.6, -2.2, 9.4), vertical=True) == (0, 0, 5)


@pytest.mark.parametrize(
    "right, forward, expect",
    [
        ((1, 0, 0), (0, 1, 0), ((1, 0, 0), (0, 1, 0))),
        ((0.7, -0.7, 0), (0.7, 0.72, -0.5), ((1, 0, 0), (0, 1, 0))),
        ((-1, 0.1, 0), (0.1, -1, 0), ((-1, 0, 0), (0, -1, 0))),
    ],
)
def test_the_arrow_keys_follow_the_view(right, forward, expect):
    assert nudge_axes(right, forward) == expect


def _box(shape):
    return np.ones(shape, dtype=bool)


def test_a_model_drops_to_the_floor():
    assert drop_distance(_box((2, 2, 2)), (0, 0, 7)) == 7
    assert drop_distance(_box((2, 2, 2)), (0, 0, 0)) == 0


def test_a_model_lands_on_the_voxels_below_it_not_on_their_box():
    table = np.zeros((6, 6, 4), dtype=bool)
    table[:, :, 0] = True  # a thin slab at the bottom of a tall box
    lamp = _box((2, 2, 3))
    assert drop_distance(lamp, (2, 2, 10), [(table, (0, 0, 0))]) == 9  # rests on the slab at z 0
    # Beside the slab it falls to the floor.
    assert drop_distance(lamp, (20, 2, 10), [(table, (0, 0, 0))]) == 10
    # Empty lower layers of the falling model count: its lowest voxel lands.
    hovering = np.zeros((2, 2, 5), dtype=bool)
    hovering[:, :, 3:] = True
    assert drop_distance(hovering, (2, 2, 4), [(table, (0, 0, 0))]) == 6


def test_a_model_already_resting_does_not_move():
    slab = _box((4, 4, 1))
    assert drop_distance(_box((2, 2, 2)), (0, 0, 1), [(slab, (0, 0, 0))]) == 0
    # An empty model is no exception: it falls by its box (test below).


def test_a_click_on_the_picked_model_cycles_to_the_one_behind():
    assert next_in_cycle([3, 7, 9], 3) == 7
    assert next_in_cycle([3, 7, 9], 9) == 3  # back to the front after the last
    assert next_in_cycle([3], 3) is None  # nothing behind it
    assert next_in_cycle([3, 7], 5) == 3  # the picked one is elsewhere: the nearest


def test_a_dragged_box_snaps_flush_against_a_neighbour():
    from rok.core.scene_move import snap_offset

    box = ((0, 0, 0), (4, 4, 4))
    wall = ((10, 0, 0), (12, 4, 4))
    # Dragged to x 5..9: one cell short of the wall at 10, so it snaps to touch it.
    offset, hit = snap_offset(box, (5, 0, 0), [wall], (0, 1))
    assert offset == (6, 0, 0) and hit == {0}
    # Far away: no snap.
    assert snap_offset(box, (1, 0, 0), [wall], (0, 1)) == ((1, 0, 0), set())


def test_faces_align_and_far_neighbours_do_not_pull():
    from rok.core.scene_move import snap_offset

    box = ((0, 0, 0), (4, 4, 4))
    beside = ((10, 20, 0), (16, 26, 4))  # across y: 16 cells away, too far to count
    assert snap_offset(box, (11, 0, 0), [beside], (0,))[1] == set()
    near = ((10, 6, 0), (16, 12, 4))  # 2 cells away in y: counts
    offset, hit = snap_offset(box, (11, 0, 0), [near], (0,))
    assert offset == (10, 0, 0) and hit == {0}  # its left face lines up with the neighbour's


def test_a_vertical_drag_stacks_on_top():
    from rok.core.scene_move import snap_offset

    box = ((0, 0, 0), (4, 4, 4))
    table = ((0, 0, 0), (6, 6, 3))
    offset, hit = snap_offset(box, (0, 0, 4), [table], (2,))
    assert offset == (0, 0, 3) and hit == {0}  # it rests on the table's top


def test_pick_order_puts_voxels_first_then_boxes():
    from rok.core.scene_move import pick_order

    voxels = [(5.0, 2), (3.0, 1)]
    boxes = [(1.0, 9), (4.0, 7), (2.0, 1)]  # 1 has a voxel hit too: it keeps that place
    assert pick_order(voxels, boxes) == [1, 2, 9, 7]
    assert pick_order([], [(2.0, 4)]) == [4]  # an empty model, by its box alone
    assert pick_order([], []) == []


def test_an_empty_model_falls_by_its_box():
    from rok.core.scene_move import drop_distance

    empty = np.zeros((4, 4, 4), dtype=bool)
    assert drop_distance(empty, (0, 0, 5)) == 5  # onto the floor
    below = np.zeros((4, 4, 2), dtype=bool)
    below[1, 1, 1] = True  # a model below, its top voxel at world z = 1
    assert drop_distance(empty, (0, 0, 5), [(below, (0, 0, 0))]) == 3  # onto that voxel
    assert drop_distance(empty, (0, 0, 0)) == 0


def test_floor_offset_puts_the_lowest_voxel_on_the_floor():
    from rok.core.scene_move import floor_offset

    lifted = np.zeros((2, 2, 6), dtype=bool)
    lifted[0, 0, 2] = True  # its lowest voxel two cells up its grid
    assert floor_offset([(lifted, (0, 0, 5))]) == -7
    assert floor_offset([(lifted, (0, 0, -4))]) == 2  # sunk into the floor: up onto it
    empty = np.zeros((2, 2, 2), dtype=bool)
    assert floor_offset([(empty, (0, 0, 3))]) == -3  # an empty model by its box
    # Together: the lowest of the pick lands on the floor, the rest keep their height above it.
    assert floor_offset([(lifted, (0, 0, 5)), (empty, (4, 0, 3))]) == -3
    assert floor_offset([(empty, (0, 0, 0))]) == 0


def test_a_picked_model_grabs_the_press_only_by_a_voxel():
    from rok.core.scene_move import choose_grab

    # B's voxel is under the mouse, the picked A only by its box: B is grabbed.
    assert choose_grab([2, 1], solid={2}, members={1}) == 2
    # A's voxel is there, behind B's: the picked A is grabbed (it can be dragged).
    assert choose_grab([2, 1], solid={2, 1}, members={1}) == 1
    # Nothing picked there: the first in click order.
    assert choose_grab([3, 4], solid={3}, members={9}) == 3
    # An empty picked model, by its box alone and nothing else there: it is first anyway.
    assert choose_grab([5], solid=set(), members={5}) == 5


def test_drop_stops_on_the_first_voxel_below():
    from rok.core.scene_move import drop_onto_offset

    block = np.ones((2, 2, 2), dtype=bool)
    slab = np.ones((6, 6, 1), dtype=bool)  # its top voxel at world z = 3 when placed at z 3
    assert drop_onto_offset([(block, (1, 1, 10))], [(slab, (0, 0, 3))]) == -6  # rests on the slab
    assert drop_onto_offset([(block, (1, 1, 10))]) == -10  # nothing below: onto the floor
    assert drop_onto_offset([(block, (20, 1, 10))], [(slab, (0, 0, 3))]) == -10  # the slab is not below
    assert drop_onto_offset([(block, (1, 1, 4))], [(slab, (0, 0, 3))]) == 0  # already resting
    # Together: the first of the pick to touch stops them all.
    assert drop_onto_offset([(block, (1, 1, 10)), (block, (20, 1, 8))], [(slab, (0, 0, 3))]) == -6
    assert drop_onto_offset([]) == 0
