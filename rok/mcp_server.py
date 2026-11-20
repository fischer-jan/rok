"""Rok as an MCP server: the tools of rok/agent.py for any MCP client.

    claude mcp add rok -- /path/to/rok/.venv-rok/bin/python /path/to/rok/rok/mcp_server.py

Over stdio. The tools run on one thread of their own: Rok's host must
stay on the thread that made it. The engine starts at the first call and
stays for the session.

Settings (environment):
  ROK_FEEDBACK    image (the default: a picture after every edit) or text
  ROK_IMAGE_SIZE  the picture's width and height in pixels (default 1000)
  ROK_FILE        a .vox to open at the start (headless only)
  ROK_LIVE        auto (the default): use the open Rok window when one runs, else
                   a headless engine; only: the open window or an error; never:
                   always headless
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations  # noqa: E402
from pydantic import Field  # noqa: E402

INSTRUCTIONS = """Rok is a voxel editor. Use its tools and the commands below.
rok_command do=describe tells more about any command, or about a topic: {topics}.
Read voxel_art before you start a scene."""

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rok")
_session = None
_live = None  # the socket to the open window, once chosen
_live_file = None
_lost = False  # the window closed: the next call connects to a new one, never headless


class LiveGone(RuntimeError):
    pass


def _drop_live() -> None:
    """Forget a dead window, so the next call looks for an open one again."""
    global _live, _live_file, _lost
    for f in (_live_file, _live):
        try:
            f.close()
        except (OSError, AttributeError):
            pass
    _live = _live_file = None
    _lost = True


def _engine():
    """The headless session, made on the tool thread at the first call."""
    global _session
    if _session is None:
        from rok.headless import Session

        _session = Session(os.environ.get("ROK_FILE") or None)
        agent = _session.agent
        agent.feedback = os.environ.get("ROK_FEEDBACK", "image")
        size = int(os.environ.get("ROK_IMAGE_SIZE", "1000"))
        agent.size = (size, size)
    return _session


def _live_call(tool: str, args: dict):
    """One call to the open window over its socket (rok/app/live.py)."""
    from rok.agent import Reply

    settings = {"feedback": os.environ.get("ROK_FEEDBACK", "image"), "size": int(os.environ.get("ROK_IMAGE_SIZE", "1000"))}
    line = json.dumps({"id": 1, "tool": tool, "args": args, "settings": settings}).encode() + b"\n"
    try:
        _live.sendall(line)
        answer = _live_file.readline()
    except OSError:
        answer = b""
    if not answer:
        _drop_live()
        raise LiveGone("The Rok window closed. Open Rok again, then call the tool again.")
    a = json.loads(answer)
    png = base64.b64decode(a["png"]) if a.get("png") else None
    return Reply(a.get("text", ""), a.get("data") or {}, None, bool(a.get("error"))), png


def _choose() -> str:
    """live or headless, chosen at the first call and kept for the session.

    After the window closed, the next call connects to a window opened again.
    It never falls back to a headless engine: that would edit another scene.
    """
    global _live, _live_file, _lost
    if _live is not None:
        return "live"
    if _session is not None:
        return "headless"
    mode = os.environ.get("ROK_LIVE", "auto")
    if mode != "never":
        from rok.app import live

        sock = live.connect()
        if sock is not None:
            _live, _live_file = sock, sock.makefile("rb")
            if _lost:
                _lost = False
                return "reopened"
            return "live"
        if mode == "only" or _lost:
            raise LiveGone("No Rok window is open. Start Rok, then call the tool again.")
    return "headless"


def _run(tool: str, args: dict) -> tuple[Any, bytes | None]:
    from rok.agent import Reply

    try:
        mode = _choose()
        if mode == "reopened":
            reply, png = _live_call(tool, args)
            reply.text = "Connected to a Rok window opened again: its scene is the one it shows now.\n" + reply.text
            return reply, png
        if mode == "live":
            return _live_call(tool, args)
    except LiveGone as exc:
        return Reply(str(exc), {}, None, True), None
    return _engine().tool(tool, args)


async def call(tool: str, args: dict) -> CallToolResult:
    """Run a tool on the Rok thread and turn its Reply into an MCP result."""
    args = {k: v for k, v in args.items() if v is not None}
    loop = asyncio.get_running_loop()
    reply, png = await loop.run_in_executor(_executor, _run, tool, args)
    content: list = [TextContent(type="text", text=reply.text)]
    if png is not None:
        content.append(ImageContent(type="image", data=base64.b64encode(png).decode("ascii"), mime_type="image/png"))
    # The text goes in the structured result too: a client that shows only that
    # (Claude Code does) saw a bare {"count": 172} from rok_command do=list.
    structured = {"text": reply.text, **reply.data} if reply.data else None
    return CallToolResult(content=content, structured_content=structured, is_error=bool(reply.error))


def _instructions() -> str:
    """The instructions with the catalogue of commands rok_command runs."""
    try:
        from rok.agent_commands import TOPICS, catalogue

        listed, topics = catalogue(), ", ".join(TOPICS)
    except Exception:  # noqa: BLE001 - the server starts without the list rather than not at all
        return INSTRUCTIONS.format(topics="voxel_art")
    return INSTRUCTIONS.format(topics=topics) + "\n\nCommands (rok_command do=run with the id):\n" + listed + "\n"


server = MCPServer("rok", instructions=_instructions())

READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
EDIT = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
RISKY = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

Cell = Annotated[list[int], Field(min_length=3, max_length=3, description="[x, y, z] in cells of the active model")]
Model = Annotated[str | None, Field(description="A model's name; it becomes the active model. Default: the active model.")]
Axis = Literal["x", "y", "z"]
Point = Annotated[list[float] | None, Field(min_length=3, max_length=3,
                                           description="[x, y, z] in cells of the active model; fractions allowed")]
Azimuth = Annotated[float | None, Field(description="a placed camera: the side it stands on, in degrees about z from +x "
                                                   "toward +y: 0 the +x side (right), 90 +y (back), 180 -x (left), "
                                                   "270 -y (front); the iso view is 315")]
Elevation = Annotated[float | None, Field(description="a placed camera: degrees above the ground, -90 to 90; 0 looks "
                                                     "level, 5 to 15 looks along a landscape")]
Eye = Annotated[list[float] | None, Field(description="a placed camera: where it stands, [x, y, z] in cells (fractions "
                                                     "allowed); it looks at `target`. Instead of azimuth and elevation")]
Target = Annotated[list[float] | None, Field(description="a placed camera: the point it looks at, [x, y, z] in cells; "
                                                        "default the middle of the scene (or of low..high)")]
Distance = Annotated[float | None, Field(description="with azimuth and elevation: cells from the target; default far "
                                                    "enough to frame the scene (or low..high)")]
Fov = Annotated[float | None, Field(description="a placed camera: the vertical field of view in degrees, 5 to 120 "
                                               "(default 35; 60 to 80 for a wide view along the ground)")]
Ortho = Annotated[bool | None, Field(description="a placed camera: true for a flat (orthographic) view without "
                                                "perspective")]


@server.tool(annotations=READ)
async def rok_look(
    views: Annotated[str | list[str], Field(description="quad (top, 3D, front, right; also as [\"quad\"]), or one to four of: top, bottom, front, back, left, right, iso")] = "quad",
    size: Annotated[list[int] | None, Field(description="[width, height] in pixels, 64 to 2000. Default 1000 x 1000 (about 1,300 tokens).")] = None,
    facts: bool = True,
    heightmap: Annotated[bool, Field(description="Add a text height map of the top view (grids up to 96 x 96).")] = False,
    cut_axis: Annotated[Axis | None, Field(description="Cut the model open for this picture: the axis of the cut plane")] = None,
    cut_at: Annotated[int | None, Field(description="The cut plane: a cell index along cut_axis")] = None,
    cut_keep: Annotated[Literal["low", "high"], Field(description="high keeps cells >= cut_at, low keeps cells < cut_at")] = "high",
    low: Annotated[list[int] | None, Field(description="With high: a close-up of this box of cells; what lies in front of it is cut away")] = None,
    high: Annotated[list[int] | None, Field(description="With low: the other corner of the close-up box")] = None,
    model: Model = None,
    azimuth: Azimuth = None,
    elevation: Elevation = None,
    eye: Eye = None,
    target: Target = None,
    distance: Distance = None,
    fov: Fov = None,
    ortho: Ortho = None,
    scale: Annotated[int | None, Field(description="pixels per cell: the front, right and top views drawn cell by "
                                                   "cell, crisp, with the front at 16 and 32 pixels below (a voxel art "
                                                   "scene always looks like this)")] = None,
    text: Annotated[bool, Field(description="add the front, right and top views as characters ('.' empty, one "
                                            "character per colour, with the legend); a TOP row reads like a row of a "
                                            "rok_voxels_set layer")] = False,
    detail: Annotated[Literal["short", "full"], Field(description="short lists the 12 most used colours, full every "
                                                                 "colour with where it is")] = "short",
) -> CallToolResult:
    """Look at the scene: a picture and the facts (grid size, voxel count, bounds, every colour in use). A cut
    shows the inside of closed shapes; nothing changes in the model. azimuth, elevation, eye, target, distance,
    fov or ortho place the camera by hand for one view (with low and high it frames that box).

    Example: {"views": "quad"} or {"views": ["iso"], "size": [600, 600], "heightmap": true}
    or {"eye": [2, 40, 3], "target": [60, 40, 2], "fov": 70} (from a low point at the edge along the ground)
    or {"views": ["front"], "cut_axis": "y", "cut_at": 12} (front view of the cells from y = 12 on)
    or {"views": ["iso", "front"], "low": [30, 20, 10], "high": [45, 35, 21]} (a close-up of that box)"""
    return await call("look", locals())


@server.tool(annotations=READ)
async def rok_voxels_get(
    low: Cell | None = None,
    high: Cell | None = None,
    axis: Annotated[Axis | None, Field(description="With index: one slice as a text grid instead of rows")] = None,
    index: int | None = None,
    model: Model = None,
) -> CallToolResult:
    """Read voxels: the filled cells of a box as [x, y, z, colour] rows (at most 4,096; has_more says
    when to ask for a smaller box), or one slice as a text grid.

    Example: {"low": [0, 0, 0], "high": [7, 7, 3]} or {"axis": "z", "index": 0}"""
    return await call("voxels_get", locals())


@server.tool(annotations=READ)
async def rok_probe(
    x: Annotated[float | None, Field(description="Pixel x in the last picture, from the left")] = None,
    y: Annotated[float | None, Field(description="Pixel y in the last picture, from the top")] = None,
    points: Annotated[list[list[float]] | None, Field(description="Many [x, y] pixels at once (at most 64)")] = None,
) -> CallToolResult:
    """Which cell is under a pixel of the last picture: the cell, its model, its colour and the face seen.
    Every picture's reply says which pixel ranges hold which view. Finding a part by colour is often
    quicker: rok_look lists where each colour is.

    Example: {"x": 740, "y": 260} or {"points": [[740, 260], [760, 260], [780, 260]]}"""
    return await call("probe", locals())


@server.tool(annotations=READ)
async def rok_check(model: Model = None) -> CallToolResult:
    """Check the model: separate parts (with where they are), enclosed empty cells, mirror symmetry per axis."""
    return await call("check", locals())


@server.tool(annotations=EDIT)
async def rok_voxels_set(
    voxels: Annotated[list[list[int]] | None, Field(description="Rows of [x, y, z, colour]; colour 0 empties the cell")] = None,
    layers: Annotated[list[list[str]] | None, Field(description="ASCII art: one list of strings per z layer, bottom layer first; the first string is the highest y row; each character is one x cell; '.' or space leaves a cell as it is; other characters map through legend")] = None,
    corner: Annotated[list[int] | None, Field(description="With layers: the cell of the lowest x, y and z")] = None,
    legend: Annotated[dict[str, int] | None, Field(description="With layers: character -> colour; '0' always empties")] = None,
    model: Model = None,
    mirror: Annotated[Axis | None, Field(description="also place the mirror image across the middle of the grid on "
                                                    "this axis (cell i and cell size - 1 - i): build one half of a "
                                                    "symmetric piece")] = None,
    mode: Annotated[Literal["set", "fill_empty", "paint"],
                    Field(description="set writes every cell; fill_empty writes only cells that are empty now (what "
                                      "is there stays: flames beside a claw); paint recolours only filled cells")] = "set",
) -> CallToolResult:
    """Set exact cells (at most 262,144 per call; a voxel art scene takes 64 per call on a grid up to 32 per side,
    256 on a bigger one). Cells outside the grid are skipped and counted.

    Examples: {"voxels": [[0, 0, 0, 12], [1, 0, 0, 12]]}
    {"layers": [["###", "#.#", "###"], ["###", "###", "###"]], "corner": [4, 4, 0], "legend": {"#": 9}}"""
    return await call("voxels_set", locals())


@server.tool(annotations=EDIT)
async def rok_shape(
    kind: Annotated[Literal["box", "sphere", "cylinder", "cone", "line", "fill", "dome", "capsule", "half_cylinder",
                            "pyramid", "diamond", "octagon_prism", "hex_prism", "tube", "wedge", "roof", "stairs",
                            "torus", "rock", "hill", "gear", "spiral_ramp", "tower", "buckyball", "framed_buckyball",
                            "extrude", "curve"],
                    Field(description="fill recolours the connected same-colour blob of `cell`; line runs from start to "
                                      "end; the shapes after fill are Rok's primitives: they fill the box and stand "
                                      "on its low side along `axis`. half_cylinder, wedge, roof, stairs and "
                                      "hex_prism run along the box's longer side across `axis`, or along `along`. buckyball is a ball of "
                                      "32 flat faces, its top ridge along the longer side (or `along`); "
                                      "framed_buckyball adds raised bars on its edges, as a canopy's frame "
                                      "(frame_colour). extrude pushes a silhouette (rows of characters, as rok_look "
                                      "text shows a view) along `axis` from `low`; curve draws a smooth line through "
                                      "`points` with clean steps")],
    low: Cell | None = None,
    high: Cell | None = None,
    colour: Annotated[int | None, Field(description="1 to 255; default: the current colour")] = None,
    mode: Annotated[Literal["add", "fill_empty", "erase", "paint"], Field(description="add fills every cell of the shape (over what is there), fill_empty only the empty ones (what is there stays), erase empties them, paint recolours only filled cells")] = "add",
    axis: Annotated[Axis, Field(description="Cylinder and cone: the long axis, a cone's tip at the high end; the primitives: their up axis")] = "z",
    hollow: Annotated[bool, Field(description="Box, sphere, cylinder, cone: only a one-cell shell")] = False,
    cell: Annotated[list[int] | None, Field(description="fill: a filled cell [x, y, z]")] = None,
    start: Annotated[list[int] | None, Field(description="line: its first cell [x, y, z]; any direction to end")] = None,
    end: Annotated[list[int] | None, Field(description="line: its last cell [x, y, z]")] = None,
    along: Annotated[Axis | None, Field(description="half_cylinder, wedge, roof, stairs, hex_prism, the buckyballs: the axis it runs "
                                                    "along (across `axis`); default the box's longer side. A "
                                                    "half_cylinder one cell thick along y with along=y is a half "
                                                    "disc standing in the x-z plane")] = None,
    solid: Annotated[bool, Field(description="line and curve: join its cells face to face (a cell fills each diagonal step), so "
                                             "what touches it counts as joined; default false: a thin line whose "
                                             "diagonal steps touch only at edges")] = False,
    frame_colour: Annotated[int | None, Field(description="framed_buckyball: the colour of its bars, 1 to 255; default: "
                                                          "the palette's nearest darker shade of the colour")] = None,
    model: Model = None,
    silhouette: Annotated[list[str] | None, Field(description="extrude: rows of characters, top row first ('.' leaves a "
                                                              "cell alone); along y it reads as the FRONT view (columns "
                                                              "x, rows z from high), along x as RIGHT (columns y), along "
                                                              "z as TOP (columns x, rows y from high)")] = None,
    legend: Annotated[dict[str, int] | None, Field(description="extrude: character -> colour; default {\"#\": colour}")] = None,
    depth: Annotated[int, Field(description="extrude: how many cells deep along axis")] = 1,
    points: Annotated[list[list[int]] | None, Field(description="curve: the cells [x, y, z] it passes through, in order")] = None,
    closed: Annotated[bool, Field(description="curve: join the last point back to the first")] = False,
    mirror: Annotated[Axis | None, Field(description="also place the mirror image across the middle of the grid on "
                                                    "this axis (cell i and cell size - 1 - i): build one half of a "
                                                    "symmetric piece")] = None,
    size: Annotated[int | list[int] | None, Field(description="curve: the brush size of Voxel mode, 1 to 50 (1 is one "
                                                              "voxel, 2 is 3 across, 3 is 5 across); one number, or one "
                                                              "per point so the curve thins or swells between them: "
                                                              "a tentacle [3, 2, 1]")] = None,
    turn: Annotated[float | None, Field(description="tilt the shape by this many degrees about turn_axis, round the "
                                                    "middle of its box, before it is placed (any shape but fill): "
                                                    "an egg leaning back, a horn at an angle. Positive turns "
                                                    "counter-clockwise seen from the axis' positive end")] = None,
    turn_axis: Annotated[Axis, Field(description="turn: the axis it tilts about")] = "x",
) -> CallToolResult:
    """Make a shape inside the box from low to high (both corners included), at most 256 x 256 x 64 cells; split a
    bigger one into several shapes in one rok_batch. A line runs from start to end, in any direction (low and high
    work too, taken in the order given).

    Examples: {"kind": "box", "low": [0, 0, 0], "high": [9, 9, 0], "colour": 12}
    {"kind": "extrude", "silhouette": [".##.", "####", "#..#"], "low": [4, 6, 0], "axis": "y", "depth": 3, "colour": 2}
    {"kind": "curve", "points": [[2, 8, 0], [6, 8, 6], [12, 8, 8]], "colour": 3, "mirror": "x"}
    {"kind": "curve", "points": [[10, 4, 12], [10, 3, 8], [12, 2, 5]], "size": [3, 2, 1], "colour": 3} (a tentacle:
    the brush sphere swept along the curve, 5 across at the root, one voxel at the tip)
    {"kind": "sphere", "low": [6, 4, 10], "high": [14, 12, 24], "turn": -40, "turn_axis": "x", "colour": 1} (a tall
    egg leaning back toward +y)
    {"kind": "sphere", "low": [2, 2, 2], "high": [9, 9, 9], "mode": "erase"}
    {"kind": "cone", "low": [4, 4, 10], "high": [11, 11, 20], "colour": 40}
    {"kind": "roof", "low": [2, 2, 8], "high": [13, 9, 12], "colour": 3} (a gable roof, ridge along x)
    {"kind": "line", "start": [20, 5, 0], "end": [12, 5, 15], "colour": 7, "solid": true} (a trunk leaning toward
    low x, its cells joined face to face)
    {"kind": "half_cylinder", "low": [0, 4, 0], "high": [20, 4, 10], "along": "y", "colour": 9} (a half disc, flat side down)
    {"kind": "box", "low": [0, 0, 0], "high": [9, 9, 3], "colour": 5, "mode": "fill_empty"} (fills around what is there)
    {"kind": "framed_buckyball", "low": [10, 4, 12], "high": [29, 15, 21], "colour": 79} (a faceted canopy: glass
    panes in 79, a darker frame; erase its lower half for a dome)"""
    return await call("shape", locals())


@server.tool(annotations=EDIT)
async def rok_select(
    do: Literal["all", "none", "box", "colour", "connected", "invert", "grow", "shrink", "mask", "unmask"],
    low: Cell | None = None,
    high: Cell | None = None,
    colour: int | None = None,
    cell: Annotated[list[int] | None, Field(description="connected: a filled cell; its whole connected part is selected")] = None,
    count: Annotated[int, Field(description="grow and shrink: how many cells")] = 1,
    model: Model = None,
) -> CallToolResult:
    """Choose which voxels rok_transform with target=selection and rok_edit (shaders too) act on. Only filled cells
    are selected. mask protects the selected voxels from every later change (it clears the selection); unmask lifts
    the mask.

    Example: {"do": "box", "low": [0, 0, 5], "high": [9, 9, 9]} or {"do": "colour", "colour": 40}
    {"do": "colour", "colour": 12} then {"do": "mask"} (colour 12 stays as it is through every command)"""
    return await call("select", locals())


@server.tool(annotations=EDIT)
async def rok_transform(
    do: Literal["move", "rotate", "flip", "mirror"],
    target: Literal["selection", "model"] = "selection",
    offset: Annotated[list[int] | None, Field(description="move: [dx, dy, dz] in cells")] = None,
    axis: Axis = "z",
    degrees: Annotated[float, Field(description="rotate: any angle in 0.1 degree steps; multiples of 90 are exact")] = 90.0,
    from_positive: Annotated[bool, Field(description="mirror: copy the + half of the axis onto the - half (false: the other way)")] = True,
    model: Model = None,
) -> CallToolResult:
    """Move, turn or flip the selection or the whole model; mirror copies one half of the model onto the other.

    Examples: {"do": "move", "offset": [0, 0, 3]} {"do": "rotate", "axis": "z", "degrees": 30}
    {"do": "mirror", "axis": "x"}"""
    return await call("transform", locals())


@server.tool(annotations=EDIT)
async def rok_deform(
    offsets: Annotated[dict[str, list[float]], Field(description="how far each cage point moves, in cells: {\"i,j,k\": [dx, dy, dz]}. i, j, k count points along x, y, z from 0; point 0 is on the box's low face and the last point on its high face")],
    counts: Annotated[int | list[int], Field(description="points per axis, 2 to 6: one number, or [nx, ny, nz]. The default 3 gives corners, edge middles, face centres and a centre point")] = 3,
    soft: Annotated[bool, Field(description="every point pulls the whole cage, not just its own part (full-degree Bernstein basis). Differs from the default only at 5 and 6 points")] = False,
    model: Model = None,
) -> CallToolResult:
    """Bend the selection (the whole model when nothing is selected) with a cage: a lattice of points on the bounding
    box of what is bent. Moving a point drags the voxels near it, so a wing sweeps back or a belly bulges. Points
    on the box's corners and edges bend its shape; face centres and the centre make a bulge. Voxels that leave the
    grid are lost. The selection becomes the bent voxels. One undo step.

    Examples: {"offsets": {"2,1,1": [0, -6, 0]}} (3 points per axis: pull the +x face centre back by 6 cells)
    {"counts": [4, 3, 3], "offsets": {"3,2,2": [0, 4, 0], "3,0,2": [0, -4, 0]}} (a flare at the +x end)"""
    args = locals()
    args["bernstein"] = args.pop("soft")
    return await call("deform", args)


@server.tool(annotations=EDIT)
async def rok_edit(
    do: Annotated[Literal["hollow", "smooth", "bevel", "grow", "shrink", "scale", "resize", "fit_grid", "clear",
                          "delete_loose", "inverse",
                          "drop_to_floor", "drop_voxels", "lathe", "twist", "shader", "shaders", "set_colour",
                          "palette", "shade", "current_colour"],
                  Field(description="clear deletes the selected voxels, or empties the whole model when nothing is "
                                    "selected; delete_loose keeps the largest connected part and deletes every "
                                    "floating bit; drop_to_floor moves the model (or selection) down as one block; "
                                    "drop_voxels lets each selected voxel fall on its own onto the floor or what is under it")],
    wall: Annotated[int, Field(description="hollow: wall thickness")] = 1,
    count: Annotated[int, Field(description="grow and shrink: cells")] = 1,
    size: Annotated[list[int] | None, Field(description="resize: the new grid [x, y, z]")] = None,
    axis: Axis = "z",
    angle: Annotated[int, Field(description="twist: degrees over the height")] = 90,
    shader: Annotated[str, Field(description="shader: a name such as weathering/erode; shaders: one shader's arguments (empty: every shader)")] = "",
    args: Annotated[list[float] | dict[str, float] | None, Field(description="shader: its arguments in order, or by name like {\"Size\": 12}; the rest keep the shader's defaults")] = None,
    colour: Annotated[int | None, Field(description="shader: the colour it works on or draws with (default: the current colour)")] = None,
    colour_count: Annotated[int | None, Field(description="shader: how many palette colours it gets from `colour` on (default as Rok's settings panel)")] = None,
    iterations: Annotated[int, Field(description="shader: run it this many times, each on the last result")] = 1,
    colours: Annotated[list[str] | dict[str, str] | None, Field(description="shader: its own colours instead of the defaults: hex in its order, or by name like {\"Grass\": \"#5a9e3a\"} (do=shaders names them). palette: the named colours to set, {\"wood\": \"#8a5a2b\", \"leaf\": \"#4a7a3a\"}")] = None,
    ramps: Annotated[dict[str, int] | None, Field(description="palette: shade steps per colour name, 2 to 6, lightest "
                                                             "first: {\"wood\": 4} makes wood 1 .. wood 4")] = None,
    ramp: Annotated[str | None, Field(description="shade: the ramp to shade with (a name from palette ramps): tops "
                                                  "take the lightest step, sides a middle one, undersides the darkest; "
                                                  "it shades the voxels of `colour`, or of only_colours, or of the "
                                                  "ramp itself")] = None,
    grow: Annotated[bool, Field(description="shader: run on the whole model but keep only what grows from the selection (grass on the selected rock, not on the roof); without it a selection limits the shader to the selected cells")] = False,
    scope: Annotated[Literal["model", "selection", "not_selection", "volume"] | None, Field(description="shader: where it runs: model, selection, not_selection (everything but the selection, to spare a part) or volume (set by `volume`); default: the selection when there is one, else the model")] = None,
    only_on: Annotated[list[Literal["top", "sides", "bottom", "edges", "hollows"]] | None, Field(description="shader: only voxels on these faces (hollows: next to narrow gaps), e.g. [\"top\"] for snow")] = None,
    only_colours: Annotated[list[int] | None, Field(description="shader: only voxels of these palette colours. They are "
                                                                "filled voxels: a shader that adds (grass, snow) also "
                                                                "needs room to reach the empty cells next to them; a "
                                                                "run that adds nothing without room runs again with "
                                                                "only_on top and room 4")] = None,
    room: Annotated[int, Field(description="shader: cells it may add around a limited scope (0-8), so a shader that adds voxels can grow out of the selection or the filtered faces; with only_on [\"top\"] alone it grows straight up only")] = 0,
    feather: Annotated[int, Field(description="shader: cells (0-8) over which it fades out toward the edge of a limited scope, for a soft border")] = 0,
    volume: Annotated[list[int] | None, Field(description="shader: run in this box of cells [x0, y0, z0, x1, y1, z1] (inclusive), empty cells too, so it can build there (trees on a patch of ground, city blocks in a lot)")] = None,
    volume_shape: Annotated[Literal["box", "sphere"], Field(description="shader with volume: the box, or the sphere (ellipsoid) inside it")] = "box",
    index: Annotated[int | None, Field(description="set_colour: the palette number; current_colour: the colour to make current")] = None,
    hex: Annotated[str | None, Field(description="set_colour: like #ff8800; may be left out when glow is given")] = None,
    glow: Annotated[float | None, Field(description="set_colour: 0 to 1, how strongly the colour glows and lights its "
                                                     "surroundings in rok_render (windows, lamps, pad lights); 0 makes it "
                                                     "a plain colour again")] = None,
    power: Annotated[int | None, Field(description="set_colour with glow: 0 to 4, each step ten times brighter (default 1: "
                                                    "a lamp that lights a few cells around it)")] = None,
    up: Annotated[bool, Field(description="scale: true doubles, false halves")] = True,
    model: Model = None,
) -> CallToolResult:
    """Whole-model operations, xs shaders, and palette colours. They act on the selection where one exists.
    The current colour is what shapes, Attach and shaders use when no colour is given.

    Examples: {"do": "hollow", "wall": 2} {"do": "resize", "size": [64, 64, 32]}
    {"do": "set_colour", "index": 12, "hex": "#3a7bd5"} {"do": "current_colour", "index": 12}
    {"do": "set_colour", "index": 30, "hex": "#ffd27a", "glow": 1} (a warm window light)
    {"do": "shaders"} (every shader, what it makes and its arguments)
    {"do": "shader", "shader": "hard_surface/greebles", "args": {"Size": 10, "Density": 80}}
    {"do": "shader", "shader": "...", "colour": 7} (a shader the list marks "works on the current colour (a
    marker)" grows from the voxels of `colour`, a marker you painted)
    {"do": "shader", "shader": "weathering/cover", "args": {"Kind": 1}, "grow": true} (grass only on the selection)
    {"do": "shader", "shader": "nature/meadow", "only_colours": [12], "only_on": ["top"], "room": 4, "args": {"Grass":
    50, "Clumps": 100}} (a meadow on the tops of colour 12 only, in solid patches with bare ground between; Clumps 0
    scatters single blades)
    {"do": "shader", "shader": "nature/trees", "only_colours": [12], "grow": true} (with nothing selected, grow starts
    from the voxels only_colours and only_on name)
    To keep grass or snow off an area (a landing pad, a path) in one run over the whole model, give only the ground's
    colour: {"do": "shader", "shader": "nature/meadow", "only_colours": [12], "only_on": ["top"], "room": 4} grows
    only straight above colour 12, never over a pad of another colour. A mask (rok_select do=mask) keeps voxels as
    they are, but a shader may still add into the empty cells above them.
    {"do": "shader", "shader": "hard_surface/battle_damage", "only_on": ["top", "edges"]}
    {"do": "shader", "shader": "weathering/patches", "feather": 6} (on the selection, fading out toward its edge)
    {"do": "shader", "shader": "nature/trees", "volume": [0, 0, 1, 20, 20, 30]} (one tree, built in that box)
    {"do": "shader", "shader": "nature/woodland", "volume": [0, 0, 1, 95, 95, 20], "args": {"Density": 100,
    "Clearings": 0, "Size": 10}} (a closed wood of small trees on the ground; Size is the tree height in cells)
    A shader run that changes nothing is an error that says why (a selection, no room, no ground)."""
    return await call("edit", locals())


@server.tool(annotations=RISKY)
async def rok_scene(
    do: Literal["list", "new", "open", "import", "save", "export", "undo", "redo", "add", "duplicate", "delete",
                "split", "merge_all", "rename", "activate", "move", "budget"],
    name: Annotated[str | None, Field(description="rename: the new name; activate: the model's name")] = None,
    offset: Annotated[list[int] | None, Field(description="move: [dx, dy, dz] for the model in the scene")] = None,
    to: Annotated[list[int] | None, Field(description="move: set the model on this cell [x, y, z] of `target`: the "
                                                      "middle of its voxels over the cell, its lowest voxels in the "
                                                      "cell's layer (in place of offset)")] = None,
    target: Annotated[str | None, Field(description="move with to: the model whose cells `to` counts in; default the "
                                                    "first other model")] = None,
    size: Annotated[list[int] | None, Field(description="new: the grid [x, y, z]; default 32 x 32 x 32")] = None,
    path: Annotated[str | None, Field(description="open (.vox or .rok), import (.vox), save (.vox, or .rok to keep parts, "
                                                  "clips and frames whole), export (.glb, .obj or .ply): a file path")] = None,
    solid: Annotated[str, Field(description="add: an empty model ('') or a shape name")] = "",
    merge: Annotated[Literal["auto", "on", "off"], Field(description="export: join neighbouring faces of one colour into "
                                                                     "rectangles (far fewer triangles); auto merges a .glb only")] = "auto",
    discard: Annotated[bool, Field(description="new and open: throw away changes that are not saved (they replace the "
                                               "whole scene and cannot be undone)")] = False,
    model: Annotated[str | None, Field(description="move, rename, duplicate, delete, split: the model to act on (it "
                                                   "becomes the active one); default the active model")] = None,
    budget: Annotated[dict[str, int] | None, Field(description="new: caps for the voxel art scene, {\"voxels\": 800, "
                                                              "\"colours\": 8} (colours at most 16, the default); an edit "
                                                              "past the budget is undone. budget: new caps for the "
                                                              "open scene, never under what it already uses")] = None,
) -> CallToolResult:
    """Models and files: list, new (a voxel art scene: each side at most 64), open, import, save, export, undo, redo, and add, duplicate, delete, split (the
    selection into its own model), merge_all, rename, activate or move a model. new and open REPLACE the whole scene
    and clear the undo history; with unsaved changes they refuse unless discard is true. import adds the models of a
    .vox file to this scene, beside the others (one undo step): use it to bring in a ship to check the fit. save
    overwrites without asking. A landscape of seamless tiles: rok_command do=run id=rok.terrain_new.

    Examples: {"do": "new", "size": [32, 32, 16], "budget": {"voxels": 1500, "colours": 8}} {"do": "save", "path": "/tmp/chair.vox"} {"do": "undo"}
    {"do": "import", "path": "/tmp/ship.vox"} then {"do": "move", "to": [42, 12, 2], "target": "Land"} (sets the ship,
    the active model, on cell [42, 12, 2] of Land: no offset to work out) {"do": "move", "model": "Tree", "offset": [0, 4, 0]}
    {"do": "save", "path": "/tmp/robot.rok"} (keeps the animation whole) {"do": "export", "path": "/tmp/chair.glb"}
    {"do": "budget", "budget": {"voxels": 4000}} (more room for the open scene; what is in it stays)"""
    return await call("scene", locals())


Frame = Annotated[int | None, Field(description="a frame of the clip, from 0 (default 0)")]


@server.tool(annotations=EDIT)
async def rok_animate(
    do: Annotated[Literal["status", "make_part", "empty_part", "parent", "pivot", "key", "delete_keys", "ease", "motion",
                          "colour_key", "voxel_frame_add", "voxel_frame_set", "voxel_frame_delete", "clip", "clip_add",
                          "clip_duplicate", "clip_delete", "marker", "marker_delete", "look", "export"],
                  Field(description="status lists clips, parts and keys; make_part cuts the selection into a part that "
                                    "follows its model; empty_part adds a part with no voxels (a root or a socket); "
                                    "parent sets the part it follows; pivot sets the point it turns about; key writes keys "
                                    "at `frame`; motion adds a ready-made motion; colour_key animates a palette colour; "
                                    "voxel_frame_* make, change and delete a part's extra voxel frames (frame-by-frame "
                                    "animation); clip* set and manage clips; marker names a frame for a game; look shows "
                                    "the posed parts; export writes the files")] = "status",
    part: Annotated[str | None, Field(description="The part: a model's name; it becomes the active model. Default: the active model.")] = None,
    frame: Frame = None,
    frames: Annotated[list[int] | None, Field(description="look: up to 8 frames side by side (default: 4 spread over the clip); "
                                                         "delete_keys and ease: the keys at these frames (default: all)")] = None,
    move: Annotated[list[float] | None, Field(description="key: [dx, dy, dz] in cells from the part's rest place")] = None,
    turn: Annotated[list[float] | None, Field(description="key: [x, y, z] degrees about the pivot, turned X, then Y, then Z; 720 turns twice")] = None,
    scale: Annotated[float | list[float] | None, Field(description="key: percent, one number or [x, y, z]; 100 is the part's own size")] = None,
    show: Annotated[bool | None, Field(description="key: false hides the part from this frame on")] = None,
    voxel_frame: Annotated[int | None, Field(description="key: which of the part's voxel frames shows from `frame` on (0 its "
                                                        "own voxels); voxel_frame_add: the one to copy (default 0); "
                                                        "voxel_frame_set and voxel_frame_delete: which one")] = None,
    ease: Annotated[Literal["hold", "linear", "smooth", "ease_in", "ease_out"] | None,
                    Field(description="key, colour_key, ease: how the value goes on to the next key (new keys: smooth)")] = None,
    channels: Annotated[list[str] | None, Field(description="delete_keys and ease: only these: move, turn, scale, show, voxel_frame "
                                                           "or one axis like move_z (default: all)")] = None,
    name: Annotated[str | None, Field(description="make_part, empty_part, clip_add, clip_duplicate, clip: the name; motion: spin, "
                                                 "swing, bob, shake (a part) or glow, blink, flow (a colour); marker: its name")] = None,
    parent: Annotated[str | None, Field(description="parent: the part to follow; empty: follow nothing")] = None,
    pivot: Annotated[list[float] | None, Field(description="pivot: [x, y, z] in the part's own cells; whole numbers are cell "
                                                          "edges, .5 a cell's middle; leave out for the middle of its grid")] = None,
    axis: Annotated[Axis | None, Field(description="motion: the axis to spin, swing or shake about (default: picked from "
                                                   "where the part touches its neighbour)")] = None,
    colour: Annotated[int | None, Field(description="motion glow, blink, flow and colour_key: the palette number (default: "
                                                    "the current colour); ease: a colour's keys instead of a part's")] = None,
    hex: Annotated[str | None, Field(description="colour_key: the colour at that frame, like #ff8800")] = None,
    glow: Annotated[float | None, Field(description="colour_key: brightness in percent at that frame (100: as it is)")] = None,
    seconds: Annotated[float | None, Field(description="clip: its length in seconds")] = None,
    fps: Annotated[int | None, Field(description="clip: frames per second (a new clip has 24); export .png: the sheet's rate")] = None,
    loop: Annotated[Literal["loop", "once", "ping_pong"] | None, Field(description="clip: how it plays")] = None,
    voxel_look: Annotated[bool | None, Field(description="clip: true shows a turned part as real voxels, false turns it smoothly")] = None,
    root_motion: Annotated[bool | None, Field(description="clip: true exports the root part's movement as the character's")] = None,
    clip: Annotated[str | int | None, Field(description="the clip to work on, by name or number from 0; it becomes the active clip")] = None,
    voxels: Annotated[list[list[int]] | None, Field(description="voxel_frame_set: rows of [x, y, z, colour] in the part's cells; 0 empties")] = None,
    layers: Annotated[list[list[str]] | None, Field(description="voxel_frame_set: ASCII layers as in rok_voxels_set")] = None,
    corner: Annotated[list[int] | None, Field(description="voxel_frame_set with layers: the lowest cell")] = None,
    legend: Annotated[dict[str, int] | None, Field(description="voxel_frame_set with layers: character -> colour")] = None,
    views: Annotated[str | list[str] | None, Field(description="look: quad, or one to four views (a strip of frames: one view)")] = None,
    size: Annotated[list[int] | None, Field(description="look: [width, height] in pixels, 64 to 2000")] = None,
    path: Annotated[str | None, Field(description="export: the file; its suffix picks the kind: .glb (glTF), .vox (with "
                                                 "frames), .png (sprite sheet and .json), .gif or .mp4")] = None,
    baked: Annotated[bool, Field(description="export .glb: every frame as voxels instead of parts that move and turn")] = False,
    merge: Annotated[bool, Field(description="export .glb: join neighbouring faces of one colour (far fewer triangles)")] = True,
    directions: Annotated[Literal[1, 4, 8, 16], Field(description="export .png: directions the sprite is drawn from")] = 8,
    cell_size: Annotated[int, Field(description="export .png: each sprite's width and height in pixels")] = 128,
    view: Annotated[Literal["top", "bottom", "front", "back", "left", "right", "iso"],
                    Field(description="export .png, .gif, .mp4: the camera's side")] = "iso",
) -> CallToolResult:
    """Animate the scene: parts, keys, motions, voxel frames, clips, and their export. Each change is one undo step;
    after a key or a motion the picture shows the parts posed. Frames count from 0; the clip plays frames 0 to its
    length - 1. A part's move, turn and scale are relative to its rest pose (the voxels as built); a part that follows
    another moves with it.

    Examples: {"do": "status"}
    {"do": "motion", "part": "propeller", "name": "spin"} (turns once over the clip, about the hub it touches)
    {"do": "key", "part": "arm", "frame": 0, "turn": [0, 0, 0]} then {"do": "key", "part": "arm", "frame": 12, "turn": [-40, 0, 0]}
    {"do": "pivot", "part": "arm", "pivot": [1, 1, 8]} (the shoulder: the top of the arm)
    {"do": "voxel_frame_add", "part": "face", "frame": 10} then {"do": "voxel_frame_set", "part": "face", "voxel_frame": 1,
    "voxels": [[3, 0, 5, 0]]} (an eye shut from frame 10) and {"do": "key", "part": "face", "frame": 12, "voxel_frame": 0}
    {"do": "look", "frames": [0, 6, 12, 18]} {"do": "export", "path": "/tmp/walk.glb"}"""
    return await call("animate", locals())


@server.tool(annotations=EDIT)
async def rok_render(
    preset: Annotated[Literal["standard", "studio", "sunset", "night", "iso", "diorama", "diorama_day",
                              "diorama_golden", "diorama_night"] | None,
                      Field(description="the light, sky and colour of the picture; iso and the diorama presets have their "
                                        "own isometric camera, iso a transparent background. diorama_day, diorama_golden "
                                        "and diorama_night put a tile on a near-black backdrop in a mood; at night the "
                                        "light comes from glowing colours (rok_edit set_colour glow). A named preset "
                                        "uses its own light. Default: the file's own light (set in Render Settings, kept "
                                        "with the file; studio for a file without one), with the camera of the preset "
                                        "it started from")] = None,
    view: Annotated[Literal["top", "bottom", "front", "back", "left", "right", "iso"],
                    Field(description="the side the camera is on")] = "iso",
    size: Annotated[list[int] | None, Field(description="[width, height] in pixels, 64 to 4096. Default: the preset's (1920 x 1080; "
                                                       "iso 2048 x 2048). A big picture costs many tokens to look at.")] = None,
    samples: Annotated[int | None, Field(description="rays per pixel: more is smoother and slower. Default: the preset's")] = None,
    path: Annotated[str | None, Field(description="save the PNG here as well (it overwrites)")] = None,
    frame: Annotated[int | None, Field(description="pose the parts as the active clip has them at this frame")] = None,
    model: Model = None,
    azimuth: Azimuth = None,
    elevation: Elevation = None,
    eye: Eye = None,
    target: Target = None,
    distance: Distance = None,
    fov: Fov = None,
    ortho: Ortho = None,
    low: Annotated[list[int] | None, Field(description="with high: frame this box of cells instead of the whole scene")] = None,
    high: Annotated[list[int] | None, Field(description="with low: the other corner of the box to frame")] = None,
    ground: Annotated[bool | None, Field(description="the preset's endless floor (standard, studio, sunset and night "
                                                    "have one): false leaves it out, so a landscape ends at its own "
                                                    "edge against the sky")] = None,
    return_image: Annotated[bool | None, Field(description="send the picture back; default: only when it is not saved "
                                                          "to `path` (a saved picture costs no tokens)")] = None,
    bloom: Annotated[float | None, Field(description="0 to 1: how far glowing colours bleed light into a soft halo; 0 "
                                                     "keeps a fire or a window a hard cube. Default: the preset's "
                                                     "(night 0.4)")] = None,
    exposure: Annotated[float | None, Field(description="-3 to 3 stops, +1 twice as bright: lifts a dark night scene "
                                                        "so a dark model still reads. Default: the preset's")] = None,
) -> CallToolResult:
    """A finished picture of the scene: Rok's path tracer with a render preset (soft shadows, glass, metal, glowing
    colours). For a check while building, rok_look is faster. Without camera arguments the camera frames the
    whole scene from `view`. Place it by hand with eye and target (a point to stand on and one to look at, in
    cells), or with azimuth, elevation and distance; fov sets the lens, ortho a flat view, low and high the box
    to frame.

    Example: {"preset": "studio", "size": [1200, 900], "path": "/tmp/ship.png"} {"preset": "iso", "path": "/tmp/icon.png"}
    {"preset": "sunset", "eye": [4, 48, 6], "target": [90, 48, 4], "fov": 60} (low at the edge, along the ground)
    {"azimuth": 300, "elevation": 12, "low": [30, 30, 0], "high": [50, 50, 10]} (low and close on one part)
    {"preset": "night", "bloom": 0.1, "exposure": 1} (fires stay hard cubes, a dark figure reads against the sky)"""
    return await call("render", locals())


@server.tool(annotations=EDIT)
async def rok_batch(
    steps: Annotated[list[dict] | str | None, Field(description="an array (not a string) of steps: [{\"tool\": \"rok_shape\", \"args\": {...}}, ...] in order; each step's arguments are the tool's own, in \"args\"; rok_look, rok_probe and rok_batch cannot be steps")] = None,
    path: Annotated[str | None, Field(description="instead of steps: a JSON file holding the array of steps (or "
                                                  "{\"steps\": [...]}), for a long batch a script wrote")] = None,
) -> CallToolResult:
    """Run several edits in order as one undo step and get one feedback at the end (at most 500 steps). A failing
    step stops the batch and says which one; the steps before it stay. A long batch made by a script goes in a JSON
    file: {"path": "/tmp/city.json"}.

    Example: {"steps": [{"tool": "rok_shape", "args": {"kind": "box", "low": [0, 0, 0], "high": [7, 7, 0], "colour": 3}},
    {"tool": "rok_shape", "args": {"kind": "box", "low": [0, 0, 1], "high": [0, 7, 7], "colour": 3}},
    {"tool": "rok_voxels_set", "args": {"voxels": [[3, 3, 1, 5]]}}]}"""
    return await call("batch", locals())


@server.tool(annotations=RISKY)
async def rok_command(
    do: Literal["list", "describe", "run"] = "list",
    id: Annotated[str | list[str] | None, Field(description="run: one command, such as lathe or rok.lathe; "
                                                            "describe: one or a list, [\"bevel\", \"smooth\"]")] = None,
    props: Annotated[dict | None, Field(description="run: the command's parameters by name, as describe shows them; an unknown name is an error")] = None,
    filter: Annotated[str, Field(description="list: only commands whose id or text holds this")] = "",
    code: Annotated[bool, Field(description="describe: add the command's source code, to see exactly what it does")] = False,
) -> CallToolResult:
    """Every other command of Rok. The instructions list the ones that change voxels and models; describe gives
    one command's full text (what it does, each parameter with its range and choices, a call to copy) and, with
    code=true, its source; list shows every command, the window's too. Prefer the other tools where they fit.

    Examples: {"do": "describe", "id": ["mirror_half", "lathe"]} then {"do": "run", "id": "mirror_half", "props":
    {"axis": "Y"}} {"do": "describe", "id": "bridge", "code": true}
    {"do": "run", "id": "terrain_new", "props": {"across": 2, "deep": 2, "size": 64, "height": 24, "seed": 7}}
    (a landscape of 2 x 2 tiles that meet without a seam, each 64 x 64 x 24, beside the scene)"""
    return await call("command", locals())


@server.prompt()
def build_asset(what: str) -> str:
    """Start building a voxel model with the right work cycle."""
    return (f"Build this voxel asset in Rok: {what}.\n"
            "1. Plan the parts and their cell ranges first; pick a grid size with rok_scene do=new.\n"
            "2. Block out the big forms with rok_shape, in one rok_batch where you can.\n"
            "3. After each change, read the feedback picture and summary: are the magenta cells where you meant?\n"
            "4. Add details with rok_voxels_set (ASCII layers work well) and paint with mode=paint.\n"
            "5. Run rok_check; fix separate parts that should touch.\n"
            "6. Show the result with rok_look and say what you made.")


def _strict_arguments() -> None:
    """Refuse an argument a tool does not have. The SDK drops it without a word,
    so rok_command with "args" in place of "props" ran with the defaults
   ."""
    for tool in server._tool_manager.list_tools():
        model = tool.fn_metadata.arg_model
        model.model_config["extra"] = "forbid"
        model.model_rebuild(force=True)
        tool.parameters = model.model_json_schema(by_alias=True)  # the published schema says so too


_strict_arguments()


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
