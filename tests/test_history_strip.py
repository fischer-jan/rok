"""The history strip: its geometry, going to a step, and the strip in the app
(View > Show History, a click, a drag)."""

from __future__ import annotations

import numpy as np
import pytest

from rok.app import history_strip, settings
from rok.app.object import document
from rok.core import history_strip as kit
from rok.core.undo import PaletteRecord, UndoStack
from test_app_tools import app, click, empty, on_floor  # noqa: F401 - fixtures


# --- geometry -----------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 7, 65])
def test_every_place_is_hit_at_its_own_x(count):
    layout = kit.strip_layout(1.0, 300, 1400, 10, count, count - 1)
    for place in range(count):
        assert layout.place_at(layout.x_of(place)) == place
    assert layout.place_at(-1000) == 0 and layout.place_at(layout.label[0] - 1) == count - 1


def test_few_steps_keep_a_short_spacing_and_many_fill_the_room():
    few = kit.strip_layout(1.0, 300, 1400, 10, 4, 3)
    assert few.spacing == kit.MAX_SPACING
    many = kit.strip_layout(1.0, 300, 1400, 10, 200, 3)
    assert many.track[1] < many.label[0] and many.spacing < kit.MAX_SPACING


def test_a_narrow_strip_draws_a_line_and_never_shrinks_below_its_name():
    narrow = kit.strip_layout(1.0, 300, 320, 10, 65, 0)
    assert narrow.rect[2] >= kit.LABEL_W
    assert narrow.as_line
    assert not kit.strip_layout(1.0, 300, 1400, 10, 65, 0).as_line


def test_the_name_is_not_on_the_track():
    layout = kit.strip_layout(1.0, 300, 1400, 10, 5, 2)
    lx, ly, lw, lh = layout.label
    assert layout.contains(lx + 5, ly + 5) and not layout.on_track(lx + 5, ly + 5)


# --- the undo stack -----------------------------------------------------------------


class _Target:
    """What the records write into: one palette."""

    def __init__(self) -> None:
        self.palette = np.zeros((256, 4), dtype=np.uint8)


def _stack_of(n: int):
    target, stack = _Target(), UndoStack()
    for k in range(1, n + 1):
        before = target.palette.copy()
        target.palette[k] = (k, k, k, 255)
        stack.push(PaletteRecord(None, before, target.palette.copy(), k))
    return target, stack


def test_go_to_equals_the_same_count_of_undo_and_redo_steps():
    target, stack = _stack_of(6)
    ref_target, ref = _stack_of(6)
    seen = []
    assert stack.go_to(2, target, seen.append) == 4
    for _ in range(4):
        ref.undo(ref_target)
    assert np.array_equal(target.palette, ref_target.palette) and seen == [True] * 4
    assert stack.position == 2 and len(stack.steps()) == 6
    assert stack.go_to(5, target) == 3
    for _ in range(3):
        ref.redo(ref_target)
    assert np.array_equal(target.palette, ref_target.palette)
    assert stack.go_to(99, target) == 1 and stack.position == 6


def test_steps_list_the_undone_ones_in_redo_order():
    target, stack = _stack_of(4)
    order = list(stack.steps())
    stack.go_to(1, target)
    assert stack.steps() == order


def test_dropped_steps_are_counted():
    stack = UndoStack(capacity=3)
    palette = np.zeros((256, 4), dtype=np.uint8)
    for k in range(5):
        stack.push(PaletteRecord(k + 1, palette[1], palette[1], k))
    assert stack.dropped == 2 and len(stack.steps()) == 3


# --- the app ------------------------------------------------------------------------


def _show(app) -> None:
    app.handle("menu", {"id": "rok.history_strip"})


def _strip_point(app, place: int) -> dict:
    layout = history_strip.layout_for(app.region)
    return {"x": float(layout.x_of(place)), "y": float(app.region.height - layout.centre_y)}


def _three_voxels(app) -> None:
    for x in (2, 5, 8):
        click(app, on_floor(app, x, 3))


def test_the_view_menu_shows_the_strip_with_cmd_l_and_keeps_the_choice(app):
    from test_app_navigation import navigation_menus

    assert history_strip.layout_for(app.region) is None
    _show(app)
    view = next(items for title, items in navigation_menus(app) if title == "View")
    assert ("rok.history_strip", "Show History", "CmdOrCtrl+L", 1) in view
    assert settings.get(history_strip.SETTING) == "on"
    assert history_strip.layout_for(app.region) is not None
    _show(app)
    assert settings.get(history_strip.SETTING) == "off"


def test_a_click_on_a_dot_goes_to_that_step_and_back(empty):
    _show(empty)
    _three_voxels(empty)
    done = int(np.count_nonzero(document.grid.voxels))
    assert document.undo.position == 3
    click(empty, _strip_point(empty, 1))
    assert document.undo.position == 1
    assert int(np.count_nonzero(document.grid.voxels)) == done - 2
    click(empty, _strip_point(empty, 3))
    assert document.undo.position == 3
    assert int(np.count_nonzero(document.grid.voxels)) == done


def test_a_drag_scrubs_through_the_steps(empty):
    _show(empty)
    _three_voxels(empty)
    seen = []
    empty.handle("mouse_move", _strip_point(empty, 3))
    empty.handle("mouse_down", {"button": "left"})
    for place in (2, 1, 0, 2):
        empty.handle("mouse_move", _strip_point(empty, place))
        seen.append(document.undo.position)
    empty.handle("mouse_up", {"button": "left"})
    assert seen == [2, 1, 0, 2]
    click(empty, on_floor(empty, 11, 3))  # a new edit drops the undone step
    assert document.undo.position == 3 and len(document.undo.steps()) == 3


def test_a_press_on_the_name_edits_nothing(empty):
    _show(empty)
    _three_voxels(empty)
    layout = history_strip.layout_for(empty.region)
    lx, ly, lw, lh = layout.label
    click(empty, {"x": float(lx + lw / 2), "y": float(empty.region.height - (ly + lh / 2))})
    assert document.undo.position == 3 and len(document.undo.steps()) == 3


def test_the_hint_line_sits_above_the_strip(empty):
    _show(empty)
    assert history_strip.top(empty.region) == pytest.approx(
        history_strip.layout_for(empty.region).rect[1] + kit.HEIGHT)


def test_the_first_dot_says_when_older_steps_were_dropped(empty):
    assert history_strip._name(0, [])[0] == "Start"
    document.undo.dropped = 1
    assert history_strip._name(0, [])[0] == "Older steps dropped"
