"""The MCP server driving an open Rok window (rok/app/live.py). It opens a
real window, so it runs only with ROK_GUI_TESTS=1."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rok_native")
pytest.importorskip("mcp")
if not os.environ.get("ROK_GUI_TESTS"):
    pytest.skip("opens a window: set ROK_GUI_TESTS=1", allow_module_level=True)

from mcp import Client, StdioServerParameters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_the_server_edits_the_open_window_and_undo_takes_it_back():
    import shutil
    import tempfile

    tmp_path = Path(tempfile.mkdtemp(prefix="rok-", dir="/tmp"))  # a socket path must stay short
    from rok.app import live
    from rok.core import png

    env = {**os.environ, "ROK_CONFIG": str(tmp_path), "ROK_NO_WELCOME": "1"}
    app = subprocess.Popen([sys.executable, "-m", "rok", str(FIXTURES / "bird.vox")], cwd=ROOT, env=env)
    try:
        assert live.wait_for_socket(tmp_path / live.SOCKET_NAME, 30.0), "the window did not open its socket"
        params = StdioServerParameters(command=sys.executable, args=["-m", "rok.mcp_server"], cwd=str(ROOT),
                                       env={"ROK_CONFIG": str(tmp_path), "ROK_LIVE": "only",
                                            "ROK_IMAGE_SIZE": "400", "PATH": "/usr/bin:/bin"})

        async def script(c):
            look = await c.call_tool("rok_look", {})
            paint = await c.call_tool("rok_shape", {"kind": "box", "low": [30, 20, 14], "high": [45, 35, 21],
                                                      "colour": 1, "mode": "paint"})
            undo = await c.call_tool("rok_scene", {"do": "undo"})
            return look, paint, undo

        async def main():
            async with Client(params, read_timeout_seconds=120) as c:
                return await script(c)

        look, paint, undo = asyncio.run(main())
        assert look.structured_content["facts"]["grid"] == [80, 68, 22]  # the window's document
        assert not paint.is_error and paint.structured_content["recoloured"] > 0
        image = next(c for c in paint.content if c.type == "image")
        _w, _h, rgba = png.decode(base64.b64decode(image.data), max_pixels=1 << 22)
        assert rgba.shape == (533, 400, 4)  # the quad and, for a small change on the bird, a close-up strip
        assert undo.structured_content["recoloured"] == paint.structured_content["recoloured"]
    finally:
        app.terminate()
        app.wait(timeout=10)
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_a_strip_of_frames_and_a_render_come_back_from_the_open_window():
    import shutil
    import tempfile

    tmp_path = Path(tempfile.mkdtemp(prefix="rok-", dir="/tmp"))
    from rok.app import live
    from rok.core import png

    env = {**os.environ, "ROK_CONFIG": str(tmp_path), "ROK_NO_WELCOME": "1"}
    app = subprocess.Popen([sys.executable, "-m", "rok", str(FIXTURES / "anim" / "propeller_spin.rok")], cwd=ROOT,
                           env=env)
    out = tmp_path / "rotor.png"
    try:
        assert live.wait_for_socket(tmp_path / live.SOCKET_NAME, 30.0), "the window did not open its socket"
        params = StdioServerParameters(command=sys.executable, args=["-m", "rok.mcp_server"], cwd=str(ROOT),
                                       env={"ROK_CONFIG": str(tmp_path), "ROK_LIVE": "only",
                                            "ROK_IMAGE_SIZE": "400", "PATH": "/usr/bin:/bin"})

        async def script(c):
            strip = await c.call_tool("rok_animate", {"do": "look", "frames": [0, 3, 6, 9, 12]})
            render = await c.call_tool("rok_render", {"size": [160, 120], "samples": 2, "path": str(out)})
            return strip, render

        async def main():
            async with Client(params, read_timeout_seconds=120) as c:
                return await script(c)

        strip, render = asyncio.run(main())
        for result in (strip, render):
            assert not result.is_error, result.content[0].text
        image = next(c for c in strip.content if c.type == "image")
        assert png.decode(base64.b64decode(image.data), max_pixels=1 << 22)[2].shape == (200, 400, 4)  # five frames: four square tiles in a row, one in the next
        assert not any(c.type == "image" for c in render.content)  # a picture saved to a path is not sent back
        assert png.decode(out.read_bytes(), max_pixels=1 << 22)[2].shape == (120, 160, 4)
    finally:
        app.terminate()
        app.wait(timeout=10)
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_the_server_finds_a_window_opened_again_after_the_first_closed():
    import shutil
    import tempfile

    tmp_path = Path(tempfile.mkdtemp(prefix="rok-", dir="/tmp"))  # a socket path must stay short
    from rok.app import live

    env = {**os.environ, "ROK_CONFIG": str(tmp_path), "ROK_NO_WELCOME": "1"}

    def open_window(path):
        app = subprocess.Popen([sys.executable, "-m", "rok", str(path)], cwd=ROOT, env=env)
        assert live.wait_for_socket(tmp_path / live.SOCKET_NAME, 30.0), "the window did not open its socket"
        return app

    def close_window(app):
        app.terminate()
        app.wait(timeout=10)

    app = open_window(FIXTURES / "bird.vox")
    try:
        params = StdioServerParameters(command=sys.executable, args=["-m", "rok.mcp_server"], cwd=str(ROOT),
                                       env={"ROK_CONFIG": str(tmp_path), "ROK_IMAGE_SIZE": "400",
                                            "PATH": "/usr/bin:/bin"})

        async def main():
            nonlocal app
            async with Client(params, read_timeout_seconds=120) as c:
                first = await c.call_tool("rok_look", {"facts": True})
                close_window(app)
                gone = await c.call_tool("rok_look", {})
                still_gone = await c.call_tool("rok_look", {})  # no headless engine in its place
                app = open_window(FIXTURES / "car.vox")
                again = await c.call_tool("rok_look", {})
                return first, gone, still_gone, again

        first, gone, still_gone, again = asyncio.run(main())
        assert first.structured_content["facts"]["grid"] == [80, 68, 22]
        assert gone.is_error and "closed" in gone.content[0].text
        assert still_gone.is_error and "No Rok window is open" in still_gone.content[0].text
        assert not again.is_error and "opened again" in again.content[0].text
        assert again.structured_content["facts"]["grid"] != [80, 68, 22]  # the new window's document
    finally:
        close_window(app)
        shutil.rmtree(tmp_path, ignore_errors=True)
