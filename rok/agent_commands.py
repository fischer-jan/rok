"""Rok's commands for the agent: the catalogue the MCP server announces at
startup, and the full description (or the source) of one command or several."""

from __future__ import annotations

import importlib
import inspect
import json

# The modules whose commands change voxels, models or the scene. The rest drive
# the window (panels, views, the dock), and the agent has its own tools for those.
CATALOGUE_MODULES = (
    ("Model", ("model_ops", "gizmo_ops")),
    ("Selection and clipboard", ("select_ops", "clip_ops")),
    ("Scene and models (the scene_ commands act on the active model)", ("scene_ops", "scene_arrange", "scene_tools")),
    ("Shaders and export", ("xs_ops", "live_shaders", "export_ops")),
)
# Commands of those modules that need the mouse or only change the view: the
# cage (rok_deform drives it), a fill under the mouse, the view's section,
# the shader folder in the file browser.
NOT_ANNOUNCED = frozenset({
    "rok.cage", "rok.cage_done", "rok.cage_points", "rok.fill", "rok.section",
    "rok.scene_frame", "rok.scene_tool", "rok.scene_carve_drag", "rok.xs_folder",
})
# How Rok's voxel art, space and tools work: rok_command do=describe returns a topic by name.
TOPICS = {
    "voxel_art": """Voxel art: every scene you start (rok_scene new) is a voxel art scene. Voxel art is a
small grid with each voxel and each colour placed on purpose, not a smooth 3D shape sampled
into many voxels. Work as a voxel artist: plan the grid (32 by default, 64 at most per side),
name a small palette with shade ramps (rok_edit do=palette; every colour you use needs a
name), block out the silhouette in one colour until the feedback reports the blockout done
(one connected mass that reads at icon size), then colour and shade by hand: the lighter
step of a ramp on faces turned to the light, darker steps below. Every edit answers with
flat views drawn cell by cell, the front at 16 and 32 pixels, and a craft line: floating
parts, symmetry, jagged slopes, thin lines and how much survives shown small. rok_look
with text=true gives the views as characters in the layer format of rok_voxels_set.
Hand tools: rok_shape extrude pushes a silhouette drawn as characters (edit a text view and
push it back), curve draws a smooth line with clean steps (with size, the Voxel-mode brush
swept along it, thinning from point to point: limbs, tentacles, horns), turn tilts any shape
before it is placed (an egg leaning back), mirror on a shape or on rok_voxels_set builds
both halves of a symmetric piece, and rok_edit do=shade lights the visible faces from a
ramp (tops light, sides middle, undersides dark). To reshape what is there: rok_transform
rotate turns a selection by any angle, rok_deform bends it with a cage, rok_edit smooth
and bevel round it off.
An edit that breaks a rule of the scene (grid size, budget, unnamed colour, a second
colour in the blockout) is undone and says why; rok_render waits for one connected part.
Opened files keep their own size and colours.""",
    "space": """Space: cells [x, y, z] of the active model, whole numbers from 0; Z is up. Cell i
spans i..i+1, so the ruler numbers in pictures mark cell edges. Each view's label names
the side its camera is on: FRONT is seen from -y (the low-y side), RIGHT from +x, TOP
from +z. Colours are palette numbers 1 to 255; 0 is empty.""",
    "work_cycle": """Work cycle: look (rok_look) -> change -> read the feedback -> check (rok_check).
Every edit is one undo step and returns a text summary and a 2 x 2 picture (top,
3D, front, right) in which the cells you just changed are outlined in magenta and
emptied cells in red. rok_probe tells you which cell is under a pixel of the last
picture. When a change is small next to the whole model, the picture also gets a
close-up strip of it (3D, front, top). rok_look with low and high shows a close-up of any
box. Use rok_batch for several edits: one round trip, one undo step, one
picture. Shapes are fastest for big forms; rok_voxels_set places exact cells,
also as ASCII layers. If a call fails, the message says how to fix it.""",
    "colours_and_shaders": """Colours: rok_edit set_colour gives a palette number its colour; current_colour picks
the colour that shapes, Attach and shaders use when you give none. rok_shape add
fills every cell of the shape, fill_empty only the empty ones around what is there.
Shaders (xs format): rok_edit do=shaders lists each with what it makes and its
arguments; run one with do=shader, its arguments by name ({"Size": 10}), the rest at
their defaults; do=shaders with a shader's name also prints its own notes (what Kind 1
means) and what each argument does and when it has an effect (some act only at some sizes). Some grow from a marker colour (voxels you painted for them): pass that
colour as `colour`. With a selection a shader changes only selected cells; with
grow=true it may add around them (grass on the selected rock only). rok_edit's
only_on, only_colours, scope, volume, room and feather limit where a shader acts (snow
on tops only, damage on edges with a soft border, trees in one box). A selection limits a
shader to the selected (filled) cells, so a shader that builds needs a volume, scope
model or grow=true; a run that changes nothing is an error that says why. Woods:
nature/woodland with Density 100 and Clearings 0 is a closed wood, and Size is the tree
height in cells (6 to 60). Masked voxels (rok_select do=mask) stay as they are through
every command.""",
    "scenes": """Scenes: rok_scene new and open replace the whole scene and cannot be undone; import adds
a .vox file's models to the scene (to check a fit). A seamless landscape of tiles:
rok_command do=run id=terrain_new (do=describe id=terrain_new shows its parameters).""",
    "animation": """Animation (rok_animate): parts are the models of the scene; Make Part cuts the selection
into a part that follows its model. The active clip holds keys: a part's move (cells),
turn (degrees about its pivot), scale (percent), show, and which of its voxel frames
shows; ready-made motions (spin, swing, bob, shake; glow, blink, flow on a colour) write
keys too. do=look shows the parts posed at any frames; do=export writes glTF, .vox, a
sprite sheet, GIF or MP4. Save with rok_scene save to a .rok path: a .vox keeps no
parents, pivots or eases. rok_render makes a finished, path-traced picture with a
render preset and saves it as a PNG. Its camera frames the whole scene from a side, or you
place it: eye and target (cells), or azimuth and elevation in degrees; fov is the lens, and
low and high frame a box. rok_look takes the same camera arguments for a check.""",
    "finishing": """Before you finish or save: call rok_look once with the quad view and check every part of
the request in all four views, also the top and 3D views (a big erase can open a roof or a
wall without a warning).""",
}
# The longest source `describe` returns for one command, in lines.
MAX_SOURCE_LINES = 300


def _registry() -> dict:
    from rok.app import opkit

    for _group, modules in CATALOGUE_MODULES:
        for name in modules:
            importlib.import_module(f"rok.app.{name}")
    return opkit.REGISTRY


def full_id(name: str) -> str:
    """`bevel` and `rok.bevel` both name rok.bevel."""
    name = str(name).strip()
    return name if "." in name else f"rok.{name}"


def _first_sentence(text: str, most: int = 80) -> str:
    text = " ".join(str(text).split())
    text = text.split(". ")[0]
    return text if len(text) <= most else text[: most - 1].rstrip() + "…"


def announced(cls) -> bool:
    """A command the catalogue names: in an editing module, not driven by the mouse."""
    module = cls.__module__.rsplit(".", 1)[-1]
    return (cls.command_id not in NOT_ANNOUNCED and "modal" not in vars(cls)
            and any(module in modules for _g, modules in CATALOGUE_MODULES))


def catalogue() -> str:
    """One line per command, grouped: its short id and what it does."""
    registry = _registry()
    out = []
    for group, modules in CATALOGUE_MODULES:
        rows = []
        for cid, cls in sorted(registry.items()):
            if cls.__module__.rsplit(".", 1)[-1] in modules and announced(cls):
                rows.append(f"  {cid.removeprefix('rok.')}: {_first_sentence(cls.tooltip or cls.title)}")
        out += [f"{group}:"] + rows
    return "\n".join(out)


def _param(name: str, p) -> str:
    """One parameter: its kind, range, default, what it does and its choices."""
    kind = getattr(p, "kind", "any")
    size = getattr(p, "size", None)
    what = {"bool": "true or false", "int": "a whole number", "float": "a number", "string": "text",
            "enum": "one of the choices"}.get(kind, kind)
    if kind in ("ivec", "fvec", "bvec"):
        what = f"{size} " + {"ivec": "whole numbers", "fvec": "numbers", "bvec": "true or false values"}[kind]
    lo, hi = getattr(p, "min", None), getattr(p, "max", None)
    if lo is not None or hi is not None:
        what += f", {lo if lo is not None else '…'} to {hi if hi is not None else '…'}"
    line = f"  {name}: {what}, default {p.default!r}"
    if p.description:
        line += f". {p.description}"
    items = getattr(p, "items", None)
    if items and not callable(items):
        line += "\n" + "\n".join(f"    {i[0]}: {i[1]}" + (f" ({i[2]})" if len(i) > 2 and i[2] else "")
                                 for i in items)
    elif callable(items):
        line += " (its choices depend on the scene)"
    return line


def describe(ids, code: bool = False) -> tuple[str, list[str]]:
    """The full text of each command in `ids`: what it does, every parameter and a
    call to run it; with `code`, the source of the command too. Returns the text
    and the ids that were not found."""
    from rok.app import opkit

    registry = _registry()
    blocks, missing = [], []
    for raw in ids:
        if str(raw).strip() in TOPICS:
            blocks.append(f"{raw} (topic)\n" + TOPICS[str(raw).strip()])
            continue
        cid = full_id(raw)
        cls = registry.get(cid)
        if cls is None:
            missing.append(str(raw))
            continue
        lines = [f"{cid} ({cls.title})"]
        if cls.tooltip:
            lines.append(cls.tooltip)
        doc = inspect.getdoc(cls)
        if doc and doc != inspect.getdoc(cls.__mro__[1]):
            lines.append(doc)
        if not announced(cls):
            lines.append("It is not in the catalogue: it needs the mouse, or only changes the view.")
        props = opkit.props_of(cls)
        if props:
            lines.append("Parameters:")
            lines += [_param(n, p) for n, p in props.items()]
            example = {n: (list(p.default) if isinstance(p.default, tuple) else p.default) for n, p in props.items()}
        else:
            lines.append("Parameters: none")
            example = {}
        lines.append("Run: " + json.dumps({"do": "run", "id": cid, "props": example}))
        if code:
            try:
                src, start = inspect.getsourcelines(cls)
                path = inspect.getsourcefile(cls) or "?"
            except (OSError, TypeError):
                lines.append("Its source is not available.")
            else:
                at = path[path.rfind("rok/"):] if "rok/" in path else path
                cut = len(src) > MAX_SOURCE_LINES
                lines.append(f"Source ({at}, line {start}" + (f", first {MAX_SOURCE_LINES} lines" if cut else "")
                             + "):\n" + "".join(src[:MAX_SOURCE_LINES]).rstrip())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), missing
