"""The Marquee's rounded ends: scroll while dragging a Rect."""

import numpy as np
import pytest

from rok.core import gesture
from rok.core.edit import LASSO, RECT
from rok.core.gesture import ROUND_MAX, rounded_rect


def _inside(poly, x, y):
    """Even-odd point in polygon test."""
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def test_step_zero_is_the_plain_rectangle_whatever_the_drag_direction():
    a = rounded_rect((10, 40), (30, 10), 0)
    assert sorted(map(tuple, a.tolist())) == [(10, 10), (10, 40), (30, 10), (30, 40)]


def test_a_round_top_keeps_the_extent_and_cuts_the_top_corners():
    poly = rounded_rect((0, 0), (20, 40), 1)
    assert poly[:, 0].min() == 0 and poly[:, 0].max() == 20
    assert poly[:, 1].min() == 0 and poly[:, 1].max() == pytest.approx(40)
    assert _inside(poly, 10, 39) and not _inside(poly, 0.5, 39.5)  # the top corner is gone
    assert _inside(poly, 0.5, 0.5) and _inside(poly, 19.5, 0.5)  # the bottom corners stay square
    assert _inside(poly, 0.5, 15)  # the straight side below the arch


def test_the_arch_is_a_half_circle_when_there_is_room():
    poly = rounded_rect((0, 0), (20, 40), 1)
    # The arch starts at y = 40 - 10 (the radius is half the width).
    assert _inside(poly, 0.5, 32) and not _inside(poly, 0.5, 34)  # near the side the circle is almost upright
    assert _inside(poly, 10, 39.5) and not _inside(poly, 3, 38)  # and the shoulder is cut


def test_a_low_rectangle_gets_a_flatter_arch_inside_its_extent():
    poly = rounded_rect((0, 0), (40, 6), 1)
    assert poly[:, 1].max() == pytest.approx(6) and poly[:, 1].min() == 0


def test_round_ends_round_the_top_and_the_bottom():
    poly = rounded_rect((0, 0), (20, 60), 2)
    assert not _inside(poly, 0.5, 0.5) and not _inside(poly, 0.5, 59.5)
    assert _inside(poly, 10, 1) and _inside(poly, 10, 59)


def test_step_three_is_an_ellipse():
    poly = rounded_rect((0, 0), (40, 20), 3)
    assert _inside(poly, 20, 10) and not _inside(poly, 1, 1) and not _inside(poly, 39, 19)
    assert _inside(poly, 20, 0.5) and _inside(poly, 0.5, 10)


def test_a_square_round_ends_is_a_circle():
    poly = rounded_rect((0, 0), (20, 20), 2)
    centre = np.array([10.0, 10.0])
    radii = np.hypot(*(poly - centre).T)
    assert radii.min() > 9.8 and radii.max() < 10.2


def test_a_flat_or_empty_drag_stays_flat():
    assert len(rounded_rect((5, 5), (5, 40), 2)) == 4 and len(rounded_rect((5, 5), (30, 5), 3)) == 4


def test_the_gesture_rounds_only_a_rect_and_steps_clamp():
    voxels = np.zeros((4, 4, 4), np.uint8)
    g = gesture.ScreenGesture(voxels, RECT, "SELECT", 1, lambda *a, **k: None, None, lambda p: p)
    assert g.round == 0 and g.set_round(1) and not g.set_round(1)
    assert g.set_round(99) and g.round == ROUND_MAX and g.set_round(-5) and g.round == 0
    lasso = gesture.ScreenGesture(voxels, LASSO, "SELECT", 1, lambda *a, **k: None, None, lambda p: p)
    assert not lasso.set_round(1) and lasso.round == 0


def test_the_label_tells_the_step():
    voxels = np.zeros((4, 4, 4), np.uint8)
    g = gesture.ScreenGesture(voxels, RECT, "SELECT", 1, lambda *a, **k: None, None, lambda p: p)
    g.begin_screen(0, 0)
    g.extend_screen(20, 40)
    assert g.preview().label == "R to round"
    g.set_round(1)
    assert g.preview().label == "round top" and len(g.preview().points) > 20


def test_a_door_with_a_round_top_is_selected_through_the_whole_wall():
    from rok.core.select import Selection

    box = np.ones((40, 6, 40), np.uint8)  # a solid wall 6 thick, seen from the front (x right, z up)

    def project(points):  # an orthographic front view: pixel = (x, z)
        return np.stack([points[:, 0], points[:, 2]], axis=1)

    selection = Selection((40, 6, 40))
    g = gesture.ScreenGesture(box, RECT, "SELECT", 1, lambda *a, **k: None, selection, project)
    g.begin_screen(10, 0)        # the door: 20 wide, 30 tall, its top rounded
    g.extend_screen(30, 30)
    g.set_round(1)
    g.finish()
    mask = selection.mask
    front = mask.any(axis=1)
    assert front[20, 5] and front[11, 5] and front[29, 5]          # the straight part of the door
    assert front[20, 29]                                            # the top of the arch
    assert not front[11, 29] and not front[29, 29]                  # the shoulders are cut off
    assert not front[9, 5] and not front[31, 5] and not front[20, 31]  # nothing outside the drag
    assert mask.any(axis=(0, 2)).all()                              # through the whole wall, every layer
    # A plain rectangle takes the shoulders too.
    plain = Selection((40, 6, 40))
    g2 = gesture.ScreenGesture(box, RECT, "SELECT", 1, lambda *a, **k: None, plain, project)
    g2.begin_screen(10, 0)
    g2.extend_screen(30, 30)
    g2.finish()
    assert plain.mask.any(axis=1)[11, 29] and plain.mask.sum() > mask.sum()


def test_r_cycles_through_the_shapes_and_the_arrows_stop_at_the_ends():
    voxels = np.zeros((4, 4, 4), np.uint8)
    g = gesture.ScreenGesture(voxels, RECT, "SELECT", 1, lambda *a, **k: None, None, lambda p: p)
    seen = [g.round]
    for _ in range(4):  # R four times goes once round: 1, 2, 3, then back to flat
        assert g.step_round(1, wrap=True)
        seen.append(g.round)
    assert seen == [0, 1, 2, 3, 0]
    assert g.step_round(-1, wrap=True) and g.round == ROUND_MAX  # Shift+R from flat
    assert not g.step_round(1) and g.round == ROUND_MAX          # an arrow key stops at the end
    g.set_round(0)
    assert not g.step_round(-1) and g.round == 0


def _door_columns(x0, step, width=17.4):
    from rok.core.select import Selection

    box = np.ones((60, 4, 60), np.uint8)

    def project(points):
        return np.stack([points[:, 0], points[:, 2]], axis=1)

    sel = Selection(box.shape)
    g = gesture.ScreenGesture(box, RECT, "SELECT", 1, lambda *a, **k: None, sel, project)
    g.begin_screen(x0, 2.2)
    g.extend_screen(x0 + width, 30.6)
    g.set_round(step)
    g.finish()
    return sel.mask.any(axis=1)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_the_rounded_shape_is_symmetric_wherever_the_drag_lands(step):
    # Cells are picked by their centres; the drag's middle can sit anywhere against
    # them. Whatever the phase, every row is centred on the same column.
    for k in range(25):
        front = _door_columns(10 + k * 0.37, step)
        columns = np.flatnonzero(front.any(axis=1))
        middle = columns.min() + columns.max()
        for z in range(front.shape[1]):
            xs = np.flatnonzero(front[:, z])
            if len(xs):
                assert xs.min() + xs.max() == middle, (step, k, z)


def test_a_fitted_rounding_still_takes_the_same_columns_as_the_plain_drag():
    plain = _door_columns(10.3, 0)
    rounded = _door_columns(10.3, 1)
    assert np.array_equal(plain.any(axis=1), rounded.any(axis=1))
    assert rounded.sum() < plain.sum()


def test_in_perspective_the_drag_is_left_as_drawn():
    from rok.core.select import Selection

    box = np.ones((20, 10, 20), np.uint8)

    def project(points):  # cells further back are drawn smaller: not an even lattice
        depth = 1.0 + points[:, 1] * 0.05
        return np.stack([points[:, 0] / depth, points[:, 2] / depth], axis=1)

    sel = Selection(box.shape)
    g = gesture.ScreenGesture(box, RECT, "SELECT", 1, lambda *a, **k: None, sel, project)
    g.begin_screen(3.0, 2.0)
    g.extend_screen(12.0, 14.0)
    g.set_round(1)
    assert g._lattice_rect() is None
    assert g.finish() == 0 and sel.count() > 0
