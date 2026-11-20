"""The colours a shader run puts in, and your changes to them (rok.core.recolour)."""

from __future__ import annotations

import numpy as np

from rok.core import recolour

GREY, DARK, RED = (136, 136, 136), (85, 85, 85), (255, 0, 0)


def _palette() -> np.ndarray:
    pal = np.zeros((256, 4), np.uint8)
    pal[:, 3] = 255
    pal[10, :3] = GREY
    pal[11, :3] = DARK
    return pal


def test_the_text_form_goes_both_ways_and_skips_broken_pairs():
    m = recolour.parse("888888:ff0000, #555555:00ff00, nothex:ffffff, 123456")
    assert m == {GREY: RED, DARK: (0, 255, 0)}
    assert recolour.parse(recolour.encode(m)) == m
    assert recolour.parse("888888:888888") == {}  # no change is no entry


def test_shown_lists_the_colours_of_the_changed_cells_its_own_first():
    before = np.zeros((4, 1, 1), np.uint8)
    before[0] = 10  # the model, grey
    after = before.copy()
    after[1:3] = 11  # two dark cells
    after[3] = 10  # one grey cell added
    pal = _palette()
    assert recolour.shown(before, after, pal) == [DARK, GREY]  # most cells first
    assert recolour.shown(before, after, pal, first=(GREY,)) == [GREY, DARK]
    assert recolour.shown(before, before, pal) == []  # nothing changed


def test_apply_changes_only_the_cells_the_run_changed():
    before = np.zeros((3, 1, 1), np.uint8)
    before[0] = 10  # the model's own grey stays grey
    after = before.copy()
    after[1:] = 10  # the run adds grey cells
    used = np.zeros(256, np.int64)
    used[10] = 1
    new, writes = recolour.apply(before, after, _palette(), {GREY: RED}, used)
    assert new[0, 0, 0] == 10
    slot = int(new[1, 0, 0])
    assert slot not in (0, 10, 11) and writes == {slot: RED} and new[2, 0, 0] == slot


def test_apply_takes_the_slot_that_already_holds_the_colour():
    before = np.zeros((2, 1, 1), np.uint8)
    after = before.copy()
    after[:] = 10
    new, writes = recolour.apply(before, after, _palette(), {GREY: DARK}, np.zeros(256, np.int64))
    assert (new == 11).all() and writes == {}


def test_apply_without_a_free_slot_says_so():
    before = np.zeros((1, 1, 1), np.uint8)
    after = before + 10
    used = np.ones(256, np.int64)
    try:
        recolour.apply(before, after, _palette(), {GREY: RED}, used)
    except ValueError:
        return
    raise AssertionError("no free slot, and no error")
