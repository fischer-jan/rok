"""Endless drags: where the cursor jumps near a view edge."""

from rok.core.edge_wrap import EDGE_PX, INSET_PX, wrap_target

W, H = 1200, 800


def test_away_from_the_edges_nothing_jumps():
    assert wrap_target(600, 400, W, H) is None
    assert wrap_target(EDGE_PX + 1, H - 2 - EDGE_PX, W, H) is None


def test_the_right_edge_jumps_to_the_left_and_the_left_to_the_right():
    assert wrap_target(W - 1, 400, W, H) == (INSET_PX, 400)
    assert wrap_target(0, 400, W, H) == (W - INSET_PX, 400)


def test_the_top_and_bottom_edges_jump_across_too():
    assert wrap_target(600, H - 1, W, H) == (600, INSET_PX)
    assert wrap_target(600, 0, W, H) == (600, H - INSET_PX)


def test_a_corner_jumps_on_both_axes():
    assert wrap_target(W - 1, H - 1, W, H) == (INSET_PX, INSET_PX)


def test_the_landing_place_is_not_itself_an_edge():
    for target in (wrap_target(W - 1, 400, W, H), wrap_target(0, 400, W, H)):
        assert wrap_target(*target, W, H) is None


def test_a_tiny_region_never_wraps():
    assert wrap_target(0, 0, 10, 10) is None
