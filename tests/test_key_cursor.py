"""The key cursor: the arrows step a cell, Return clicks there
(rok/core/key_cursor.py, rok/app/key_cursor.py)."""

from __future__ import annotations

import numpy as np
import pytest

from rok.app import brick_mode, key_cursor
from rok.app.object import document
from rok.app.opkit import CONTEXT
from rok.app.state import state
from rok.core import key_cursor as core
from rok.core.chrome import LEVEL_BRICK, LEVEL_MODEL
from test_app_brick import _go, _start, config, run, undo  # noqa: F401
from test_foundation import key


def test_the_steps_follow_the_view():
    # Looking along +y: W is +y, D is +x.
    assert core.step("W", (0.1, 0.9, -0.4)) == (0, 1, 0)
    assert core.step("S", (0.1, 0.9, -0.4)) == (0, -1, 0)
    assert core.step("D", (0.1, 0.9, -0.4)) == (1, 0, 0)
    assert core.step("A", (0.1, 0.9, -0.4)) == (-1, 0, 0)
    # Looking along -x: W is -x, D is +y.
    assert core.step("W", (-0.8, 0.3, -0.5)) == (-1, 0, 0)
    assert core.step("D", (-0.8, 0.3, -0.5)) == (0, 1, 0)
    assert core.step("E", (1, 0, 0)) == (0, 0, 1) and core.step("Q", (1, 0, 0)) == (0, 0, -1)
    with pytest.raises(ValueError):
        core.step("X", (1, 0, 0))


def test_a_step_stays_inside_the_grid():
    assert core.moved((0, 0, 0), "Q", (0, 1, 0), (4, 4, 4)) == (0, 0, 0)
    assert core.moved((3, 3, 3), "E", (0, 1, 0), (4, 4, 4)) == (3, 3, 3)
    assert core.moved((1, 1, 1), "W", (0, 1, 0), (4, 4, 4)) == (1, 2, 1)


@pytest.fixture
def empty(config):  # noqa: F811
    a = _start(None)
    run("rok.new", size=(16, 16, 8), discard=True)
    document.colour = 9
    CONTEXT.prefs.rok_tool = "ATTACH"
    CONTEXT.prefs.rok_mode = "VOXEL"
    CONTEXT.prefs.rok_brush_size = 1
    yield a
    key_cursor.clear()


ARROWS = {"W": ("ArrowUp", False), "S": ("ArrowDown", False), "A": ("ArrowLeft", False),
          "D": ("ArrowRight", False), "E": ("ArrowUp", True), "Q": ("ArrowDown", True)}


def _step(app, name):
    arrow, shift = ARROWS[name]
    key(app, arrow, shift=shift)
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": False})


def test_option_keys_step_the_cursor_and_return_attaches_there(empty):
    app = empty
    view = app.view_state.view_direction()
    _step(app, "D")  # the first step starts the cursor
    start = key_cursor.cell()
    assert start is not None
    _step(app, "D")
    _step(app, "E")
    right = core.step("D", view)
    assert key_cursor.cell() == (start[0] + right[0], start[1] + right[1], start[2] + 1)
    steps = len(document.undo._undo)
    key(app, "Enter")
    c = key_cursor.cell()
    assert document.grid.voxels[c] == 9 and int((document.grid.voxels != 0).sum()) == 1
    assert len(document.undo._undo) == steps + 1 and document.undo.peek().label == "Attach Voxel"
    assert state.level == LEVEL_MODEL  # Enter clicked, it did not change the level
    key(app, "Enter", alt=True)  # Option+Return: Erase
    assert not document.grid.voxels.any()
    undo(app)
    assert document.grid.voxels[c] == 9


def test_the_cursor_shows_the_brush_ghost_and_a_real_mouse_move_puts_it_away(empty):
    app = empty
    app.handle("mouse_move", {"x": 700.0, "y": 450.0})
    _step(app, "W")
    c = key_cursor.cell()
    assert state.hover is not None and state.hover.target == c
    app.handle("mouse_move", {"x": 701.0, "y": 450.0})  # a tremble keeps it
    assert key_cursor.cell() == c
    app.handle("mouse_move", {"x": 760.0, "y": 450.0})
    assert key_cursor.cell() is None
    _step(app, "W")
    key(app, "Escape")
    assert key_cursor.cell() is None


def test_in_brick_mode_return_places_the_brick_at_the_cursor(empty):
    app = empty
    _go(LEVEL_BRICK)
    run("rok.brick_take", number=9)  # 2x4
    _step(app, "W")
    c = key_cursor.cell()
    key(app, "Enter")
    assert state.level == LEVEL_BRICK  # Enter placed; it did not go to Model level
    b = document.grid.bricks.bricks[0]
    assert (b.x, b.y) == (c[0], c[1]) and b.z == 0
    _step(app, "E")  # up onto the brick
    key(app, "Enter")
    assert len(document.grid.bricks) == 2 and document.grid.bricks.bricks[1].z == 1
    key(app, "Escape")  # puts the cursor away
    key(app, "Enter")  # with no cursor Enter goes to Model level, as before
    assert state.level == LEVEL_MODEL


def test_return_on_a_brick_model_in_model_level_says_why(empty):
    app = empty
    _go(LEVEL_BRICK)
    run("rok.brick_take", number=1)
    _step(app, "W")
    key(app, "Enter")
    key_cursor.clear()
    _go(LEVEL_MODEL)
    voxels = document.grid.voxels.copy()
    _step(app, "D")
    key(app, "Enter")
    assert np.array_equal(document.grid.voxels, voxels)
    assert "built from bricks" in app.message
    assert brick_mode.active() is False


def test_option_with_the_letters_does_nothing_to_the_cursor(empty):
    key(empty, "D", alt=True)
    assert key_cursor.cell() is None
