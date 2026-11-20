"""The shader settings panel's layout and sliders (rok/core/shader_panel.py)."""

import pytest

from rok.core import shader_panel as sp

SPECS = [
    sp.Spec("a0", "Tile X", 0, 64, 1, 0),
    sp.Spec("a1", "Roughness", 0.5, 4.0, 0.1, 1),
    sp.Spec("colour_count", "Colours", 1, 64, 1, 0, stepper=True),
    sp.Spec("iterations", "Iterations", 1, 64, 1, 0, stepper=True),
]


def test_a_slider_stays_inside_the_shaders_range_and_snaps_to_its_step():
    lay = sp.panel_layout(1.0, 1400, 900, SPECS)
    tile = lay.control("a0")
    x0, _y, w, _h = tile.track
    assert sp.value_at(tile, x0 - 500) == 0 and sp.value_at(tile, x0 + w + 500) == 64  # bracketed
    assert sp.value_at(tile, x0 + w / 2) == 32
    rough = lay.control("a1")
    rx, _y, rw, _h = rough.track
    v = sp.value_at(rough, rx + rw * 0.333)
    assert v == pytest.approx(round(v, 1)) and 0.5 <= v <= 4.0
    assert sp.x_of(tile, 64) == pytest.approx(x0 + w) and sp.x_of(tile, 1000) == pytest.approx(x0 + w)


def test_the_wheel_and_the_steppers_move_one_step_and_stop_at_the_ends():
    lay = sp.panel_layout(1.0, 1400, 900, SPECS)
    rough = lay.control("a1")
    assert sp.nudged(rough, 1.0, 1) == pytest.approx(1.1)
    assert sp.nudged(rough, 4.0, 3) == 4.0 and sp.nudged(rough, 0.5, -1) == 0.5
    it = lay.control("iterations")
    assert sp.nudged(it, 1, -1) == 1 and sp.nudged(it, 1, 1) == 2


def test_a_shift_drag_moves_a_tenth_as_far():
    lay = sp.panel_layout(1.0, 1400, 900, SPECS)
    tile = lay.control("a0")
    x0, _y, w, _h = tile.track
    assert sp.fine(tile, 10, x0, x0 + w) == pytest.approx(16.0, abs=1)


def test_every_part_of_the_panel_is_hit_where_it_is_drawn():
    lay = sp.panel_layout(2.0, 2800, 1800, SPECS)
    rx, ry, rw, rh = lay.run
    assert sp.hit(lay, rx + 2, ry + 2) == ("RUN", "")
    bx, by, _bw, _bh = lay.back
    assert sp.hit(lay, bx + 2, by + 2) == ("BACK", "")
    tile = lay.control("a0")
    assert sp.hit(lay, tile.track[0] + 5, tile.row[1] + 5) == ("TRACK", "a0")
    it = lay.control("iterations")
    assert sp.hit(lay, it.plus[0] + 2, it.plus[1] + 2) == ("PLUS", "iterations")
    assert sp.hit(lay, it.minus[0] + 2, it.minus[1] + 2) == ("MINUS", "iterations")
    assert sp.hit(lay, 1, 1) is None
    x, y, w, h = lay.rect
    for c in lay.controls:
        cx, cy, cw, ch = c.row
        assert x <= cx and cx + cw <= x + w and y <= cy and cy + ch <= y + h
    assert lay.thumb[1] > max(c.row[1] + c.row[3] for c in lay.controls)


def test_long_words_wrap_into_two_lines_with_an_ellipsis():
    lines = sp.wrap("one two three four five six seven eight", lambda t: len(t) * 10, 100, lines=2)
    assert len(lines) == 2 and lines[-1].endswith("…") and all(len(line) * 10 <= 110 for line in lines)
    assert sp.text_of(3.0, 0) == "3" and sp.text_of(0.25, 2) == "0.25"
