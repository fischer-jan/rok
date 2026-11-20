"""The MCP server (rok/mcp_server.py) over its real stdio transport, in a child process."""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rok_native")
pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402

from rok.core import png  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _run(script):
    env = {"ROK_IMAGE_SIZE": "400", "ROK_LIVE": "never", "PATH": "/usr/bin:/bin"}
    if "PYTHONPATH" in os.environ:  # the child loads the same native module as this test run
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    params = StdioServerParameters(command=sys.executable, args=["-m", "rok.mcp_server"], cwd=str(ROOT), env=env)

    async def main():
        async with Client(params, read_timeout_seconds=120) as c:
            return await script(c)

    return asyncio.run(main())


def _picture(result) -> np.ndarray:
    images = [c for c in result.content if c.type == "image"]
    assert len(images) == 1
    _w, _h, rgba = png.decode(base64.b64decode(images[0].data), max_pixels=1 << 22)
    return rgba


def test_fifteen_tools_with_hints_instructions_and_a_prompt():
    async def script(c):
        tools = (await c.list_tools()).tools
        prompts = (await c.list_prompts()).prompts
        return tools, prompts, c.instructions

    tools, prompts, instructions = _run(script)
    names = {t.name for t in tools}
    assert len(tools) == 15 and all(n.startswith("rok_") for n in names)
    assert {"rok_animate", "rok_render"} <= names
    hints = {t.name: t.annotations for t in tools}
    assert hints["rok_look"].read_only_hint and hints["rok_check"].read_only_hint
    assert hints["rok_scene"].destructive_hint and not hints["rok_shape"].read_only_hint
    assert all("Example" in (t.description or "") or t.name == "rok_check" for t in tools)
    # short: what Rok is, the topics describe explains, and the commands rok_command runs
    assert len(instructions) < 6000 and [p.name for p in prompts] == ["build_asset"]
    assert "voxel_art" in instructions and "\n  bevel: " in instructions and "scene_drag" not in instructions


def test_animate_and_render_over_the_transport():
    fixture = str(ROOT / "tests" / "fixtures" / "anim" / "propeller_spin.rok")

    async def script(c):
        await c.call_tool("rok_scene", {"do": "open", "discard": True, "path": fixture})
        status = await c.call_tool("rok_animate", {"do": "status"})
        strip = await c.call_tool("rok_animate", {"do": "look", "frames": [0, 6]})
        render = await c.call_tool("rok_render", {"preset": "studio", "size": [160, 120], "samples": 2})
        return status, strip, render

    status, strip, render = _run(script)
    assert not status.is_error and status.structured_content["parts"][1]["name"] == "Rotor"
    assert _picture(strip).shape == (200, 400, 4)  # two square tiles of the 400 pixel picture
    assert _picture(render).shape == (120, 160, 4)


PALETTE = {"do": "palette", "colours": {"stone": "#8a8478", "moss": "#4a7a3a"}}


def test_an_edit_returns_text_numbers_and_a_marked_picture():
    async def script(c):
        await c.call_tool("rok_scene", {"do": "new", "discard": True, "size": [16, 16, 16]})
        await c.call_tool("rok_edit", PALETTE)
        await c.call_tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [11, 11, 1], "colour": 1})
        plain = await c.call_tool("rok_look", {})
        edit = await c.call_tool("rok_shape", {"kind": "box", "low": [4, 4, 2], "high": [7, 7, 5], "colour": 1})
        return plain, edit

    plain, edit = _run(script)
    assert not edit.is_error
    assert "+64 added" in edit.content[0].text and "Craft:" in edit.content[0].text
    assert edit.structured_content["added"] == 64
    assert edit.structured_content["facts"]["bounds"] == {"low": [0, 0, 0], "high": [11, 11, 5]}
    before, after = _picture(plain), _picture(edit)
    # A voxel art scene draws its flat views cell by cell; the marks of a small change are magenta.
    magenta = lambda p: ((p[..., 0] > 200) & (p[..., 1] < 110) & (p[..., 2] > 170)).sum()  # noqa: E731
    assert magenta(before) == 0 and magenta(after) > 100


def test_a_batch_is_one_undo_step_and_a_bad_call_says_how_to_fix_it():
    async def script(c):
        await c.call_tool("rok_scene", {"do": "new", "discard": True, "size": [16, 16, 16]})
        await c.call_tool("rok_edit", PALETTE)
        batch = await c.call_tool("rok_batch", {"steps": [
            {"tool": "rok_shape", "args": {"kind": "box", "low": [0, 0, 0], "high": [15, 15, 0], "colour": 1}},
            {"tool": "rok_shape", "args": {"kind": "sphere", "low": [4, 4, 1], "high": [11, 11, 8], "colour": 1}},
        ]})
        undo = await c.call_tool("rok_scene", {"do": "undo"})
        bad = await c.call_tool("rok_shape", {"kind": "box", "low": [20, 0, 0], "high": [25, 1, 1]})
        # a second colour in the blockout: the whole batch is undone (a new scene, for the
        # first batch's feedback ended its blockout)
        await c.call_tool("rok_scene", {"do": "new", "discard": True, "size": [16, 16, 16]})
        await c.call_tool("rok_edit", PALETTE)
        mixed = await c.call_tool("rok_batch", {"steps": [
            {"tool": "rok_shape", "args": {"kind": "box", "low": [0, 0, 0], "high": [3, 3, 0], "colour": 1}},
            {"tool": "rok_shape", "args": {"kind": "box", "low": [0, 0, 1], "high": [3, 3, 1], "colour": 2}},
        ]})
        scene = await c.call_tool("rok_scene", {"do": "list"})
        return batch, undo, bad, mixed, scene

    batch, undo, bad, mixed, scene = _run(script)
    assert not batch.is_error and len([c for c in batch.content if c.type == "image"]) == 1
    added = batch.structured_content["added"]
    assert undo.structured_content["removed"] == added and undo.structured_content["facts"]["voxels"] == 0
    assert bad.is_error and "resize" in bad.content[0].text
    assert mixed.is_error and "blockout" in mixed.content[0].text
    assert scene.structured_content["models"][0]["voxels"] == 0


def test_probe_finds_the_cell_under_a_pixel_of_the_last_picture():
    async def script(c):
        await c.call_tool("rok_scene", {"do": "new", "discard": True, "size": [16, 16, 16]})
        await c.call_tool("rok_edit", PALETTE)
        await c.call_tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [15, 15, 0], "colour": 2})
        look = await c.call_tool("rok_look", {"scale": 10})
        # the top view is the third: 160 pixels wide each, 8 apart; its cells are 10 pixels
        return look, await c.call_tool("rok_probe", {"x": 2 * 168 + 55, "y": 5})

    look, probe = _run(script)
    assert "TOP x 336..496" in look.content[0].text
    assert probe.structured_content["colour"] == 2 and probe.structured_content["face"] == "top"
    assert probe.structured_content["cell"] == [5, 15, 0]


def test_an_argument_a_tool_does_not_have_is_an_error_and_the_text_rides_along():
    async def script(c):
        await c.call_tool("rok_scene", {"do": "new", "discard": True, "size": [8, 8, 8]})
        # "args" in place of "props": a tool must refuse it, or the command runs with its defaults.
        wrong = await c.call_tool("rok_command", {"do": "run", "id": "rok.terrain_new", "args": {"across": 1}})
        listed = await c.call_tool("rok_command", {"do": "list", "filter": "terrain_new"})
        steps = await c.call_tool("rok_batch", {"steps": "not json"})
        scene = await c.call_tool("rok_scene", {"do": "list"})
        tools = (await c.list_tools()).tools
        return wrong, listed, steps, scene, tools

    wrong, listed, steps, scene, tools = _run(script)
    assert wrong.is_error and "args" in wrong.content[0].text and "xtra" in wrong.content[0].text
    assert len(scene.structured_content["models"]) == 1  # no terrain was made
    # A client that shows only the structured result still sees the rows and the text.
    assert listed.structured_content["commands"][0].startswith("rok.terrain_new(")
    assert listed.structured_content["text"] == listed.content[0].text
    assert steps.is_error and "must be an array" in steps.content[0].text
    render = next(t for t in tools if t.name == "rok_render")
    assert {"eye", "target", "azimuth", "elevation", "fov", "low", "high", "ground"} <= set(render.input_schema["properties"])
    assert render.input_schema.get("additionalProperties") is False
