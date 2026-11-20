"""Drive rok/mcp_server.py over stdio: build a small model, look, render a PNG.

    .venv-rok/bin/python scripts/linux_mcp_smoke.py OUT.png

Needs no GPU and no display (software Vulkan is enough). Prints each call's time.
"""

import asyncio
import base64
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main(out: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "rok" / "mcp_server.py")])
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        names = [t.name for t in (await s.list_tools()).tools]
        print("tools:", len(names), names)

        async def call(tool: str, **args):
            t0 = time.perf_counter()
            res = await s.call_tool(tool, args)
            print(f"{tool:14s} {time.perf_counter() - t0:6.2f} s  error={res.is_error}")
            for c in res.content:
                if c.type == "text":
                    print("   ", c.text[:200].replace("\n", " | "))
            return res

        await call("rok_shape", kind="box", low=[2, 2, 0], high=[21, 21, 1], colour=12)
        await call("rok_shape", kind="box", low=[6, 6, 2], high=[17, 17, 9], colour=5)
        await call("rok_shape", kind="roof", low=[5, 5, 10], high=[18, 18, 14], colour=3)
        await call("rok_shape", kind="sphere", low=[9, 9, 4], high=[14, 14, 7], mode="erase")
        await call("rok_look", views="quad", size=[400, 400])
        res = await call("rok_render", preset="studio", size=[640, 480], samples=32, path=out)
        res = await call("rok_render", preset="studio", size=[640, 480], samples=32, path=out)
        assert Path(out).stat().st_size > 1000, "no picture written"
        print("wrote", out, Path(out).stat().st_size, "bytes")


asyncio.run(main(sys.argv[1]))
