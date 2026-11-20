"""Each undo step has a name and a time: the action, tool or agent call that
made it, else its kind. The history strip shows them."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from rok.app import opkit
from rok.app.object import document
from rok.core.undo import PaletteRecord, UndoStack, kind_name, naming
from test_app_tools import app, click, empty, on_floor  # noqa: F401 - fixtures

BIRD = Path(__file__).parent / "fixtures" / "bird.vox"


def test_an_action_names_its_step(empty):
    click(empty, on_floor(empty, 3, 3))
    before = time.time()
    opkit.call("rok.flip", "EXEC", axis="X")
    record = document.undo.peek()
    assert record.label == "Flip"
    assert before <= record.when <= time.time()


def test_a_stroke_is_named_by_its_tool_and_mode(empty):
    click(empty, on_floor(empty, 3, 3))
    assert document.undo.peek().label == "Attach Voxel"


def test_a_record_pushed_outside_an_action_is_named_by_its_kind(app):
    palette = document.scene.palette
    document.undo.push(PaletteRecord(5, palette[5].copy(), palette[5].copy(), 0))
    assert document.undo.peek().label == "Colours"


def test_the_innermost_name_wins_and_the_outer_comes_back():
    stack = UndoStack()
    palette = np.zeros((256, 4), dtype=np.uint8)
    with naming("Outer"):
        with naming("Inner"):
            stack.push(PaletteRecord(1, palette[1], palette[1], 0))
        stack.push(PaletteRecord(2, palette[2], palette[2], 0))
    assert [r.label for r in stack._undo] == ["Inner", "Outer"]
    assert UndoStack.label == ""


def test_a_merged_colour_step_keeps_its_first_name():
    stack = UndoStack()
    palette = np.zeros((256, 4), dtype=np.uint8)
    with naming("Colour Popup"):
        stack.push(PaletteRecord(3, palette[3], palette[3], 7))
    with naming("Something Else"):
        stack.push(PaletteRecord(3, palette[3], palette[3], 7), merge=True)
    assert len(stack._undo) == 1 and stack.peek().label == "Colour Popup"


def test_an_agent_call_is_named_after_the_tool():
    pytest.importorskip("rok_native")  # Session needs the native module
    from rok.headless import Session

    session = Session()
    try:
        session.open(BIRD)
        session.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [1, 1, 1], "colour": 9})
        assert document.undo.peek().label == "MCP shape box"
    finally:
        session.close()


def test_a_compound_step_is_named_by_its_first_part_and_a_subclass_by_its_parent():
    from rok.app.scene_ops import ListRecord
    from rok.core.undo import CompoundRecord, MaskRecord, SceneRecord

    palette = np.zeros((256, 4), dtype=np.uint8)
    compound = CompoundRecord((PaletteRecord(1, palette[1], palette[1], 0),))
    assert kind_name(compound) == "Colours"
    assert kind_name(MaskRecord.__new__(MaskRecord)) == "Mask"
    assert issubclass(ListRecord, SceneRecord) and kind_name(ListRecord.__new__(ListRecord)) == "Scene"
