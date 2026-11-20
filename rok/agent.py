"""The tools an LLM uses to build in Rok, and the feedback it gets back.

`Agent.call(tool, args)`
runs one tool on the app's thread and returns a `Reply`: text, numbers for
`structuredContent`, and, when the tool changed something, a `Picture` to
render. The caller renders it (`Agent.queue_picture`, then the host's
"screenshot" event) and attaches the PNG. The same code serves the headless
`Session` (rok/headless.py) and the open window (step 3).

Coordinates are the active model's cells, Z up, from 0. Colours are palette
numbers 1 to 255; 0 is empty.
"""

from __future__ import annotations

import inspect
import os
import traceback
from dataclasses import dataclass, field

import numpy as np

from rok.agent_anim import AnimTools
from rok.agent_art import ART_DEFAULT, ART_MAX_SIDE, CRAFT_MAX_VOXELS, ArtTools, art_set_cells, verdict
from rok.core import craft, region, voxel_draw
from rok.core.brick import BrickLocked
from rok.core.shape import line3d_cells
from rok.core.undo import naming

MAX_SET_CELLS = 64 ** 3  # one rok_voxels_set call writes at most this many cells (they come as JSON rows)
# One rok_shape call fills at most this many cells: a 256 x 256 x 64 box. The
# shape is made here, not sent, so only the memory for its cells (about 50 bytes
# each while it is written) limits it. A 256 x 256 x 6 ground slab was refused at
# 64 cubed.
MAX_SHAPE_CELLS = 256 * 256 * 64
MAX_BATCH_STEPS = 500  # a scene's colour pass replays a few hundred shapes
MAX_GET_ROWS = 4096
SHORT_COLOURS = 12  # colours listed in a short answer; detail=full lists every one
CLOSEUP_VIEWS = ("iso", "front", "top")
CLOSEUP_PAD = 2  # cells round a change in its close-up
CLOSE_PPC = 10.0  # below this many pixels per voxel in the overview, a change gets a close-up
MAX_MARKS = 150  # changed cells outlined one by one; more get one box, so the colours stay readable
LIGHT_BG = 0.5  # linear grey of the background behind a dark model
DEFAULT_SIZE = (1000, 1000)
FEEDBACK_LEVELS = ("text", "image")
STRIP_COLUMNS = 4  # frames per row in a strip picture
RENDER_ROOM = 1.12  # a finished picture leaves this much more room round the model than a check

HIGHLIGHT = (1.0, 0.22, 0.85, 1.0)  # cells this call added or recoloured: magenta, rare in palettes
GHOST = (0.95, 0.30, 0.25, 0.9)  # cells this call emptied
RULER = (0.92, 0.90, 0.86, 0.95)
PLATE = (0.07, 0.07, 0.08, 0.78)
AXIS_RGB = ((0.94, 0.40, 0.33, 1.0), (0.55, 0.80, 0.30, 1.0), (0.35, 0.60, 0.95, 1.0))  # rok.core.gizmo.AXIS_COLOURS
LAYER = "agent:feedback"


class ToolError(Exception):
    """A call the model can fix: the message says how."""


@dataclass
class Picture:
    views: tuple[str, ...]
    size: tuple[int, int]
    marks: dict | None = None  # {"added": cells, "removed": cells} in world cells
    cut: tuple | None = None  # (axis, at, keep "low" or "high"): only the cells on the kept side show
    box: tuple | None = None  # (low, high) world box the views frame and clip to: a close-up
    closeup: tuple | None = None  # (low, high) world box of a close-up strip under the views
    frames: tuple | None = None  # clip frames: one poses the parts; more make a strip, one tile per frame
    render: tuple | None = None  # (render preset id, samples[, floor on]): one path-traced picture of views[0]
    save: str | None = None  # where the finished PNG goes as well
    camera: dict | None = None  # a camera placed by hand: headless.free_camera's arguments, in world units
    send: bool = True  # False: the picture is only saved (`save`), not sent back


@dataclass
class Reply:
    text: str
    data: dict = field(default_factory=dict)
    picture: Picture | None = None
    error: bool = False
    image: bytes | None = None  # a finished PNG (flat views drawn here, not by the renderer)


# --- the tools' descriptions (the MCP server publishes these) -------------------------

def xs_parse_hex(h: str):
    from rok.core import xs

    found = xs.parse_hex_list(h)
    return found[0] if len(found) == 1 else None


AXES = ("x", "y", "z")
# rok_shape's kinds: the four drawn here, line and fill, and the rest of the
# Primitives menu (rok.core.shape), which grow from the box's low side along `axis`.
SHAPE_KINDS = ("box", "sphere", "cylinder", "cone", "line", "fill", "dome", "capsule", "half_cylinder", "pyramid",
               "diamond", "octagon_prism", "hex_prism", "tube", "wedge", "roof", "stairs", "torus", "rock", "hill",
               "gear", "spiral_ramp", "tower", "buckyball", "framed_buckyball", "extrude", "curve")
# The room a shader run gets when its filters alone left it nothing to add to.
AUTO_ROOM = 4
# The rok_scene actions that take `model`: they act on one model.
MODEL_SCENE_ACTIONS = ("move", "rename", "duplicate", "delete", "split")
# The shapes with a run: `along` says which way it goes (else the box's longer side).
RUN_KINDS = ("half_cylinder", "wedge", "roof", "stairs", "hex_prism", "buckyball", "framed_buckyball")
VIEW_NAMES = ("top", "bottom", "front", "back", "left", "right", "iso")


def _undo_label(tool: str, args: dict) -> str:
    """The name of an agent call's undo step: "MCP shape box", "MCP edit shader"."""
    words = ["MCP", tool.removeprefix("rok_")]
    for key in ("do", "kind", "id"):
        if isinstance(args.get(key), str):
            words.append(args[key])
            break
    return " ".join(words)


def _touching_groups(mask: np.ndarray, parts: list) -> int:
    """How many groups the face-joined `parts` form when an edge or a corner
    also joins two cells."""
    label = np.full(mask.shape, -1, dtype=np.int64)
    for k, cells in enumerate(parts):
        label[tuple(cells.T)] = k
    root = list(range(len(parts)))

    def find(a: int) -> int:
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    nx, ny, nz = mask.shape
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue  # each pair of directions once, and not the cell itself
                a = label[max(0, -dx):nx - max(0, dx), max(0, -dy):ny - max(0, dy), max(0, -dz):nz - max(0, dz)]
                b = label[max(0, dx):nx - max(0, -dx), max(0, dy):ny - max(0, -dy), max(0, dz):nz - max(0, -dz)]
                both = (a >= 0) & (b >= 0) & (a != b)
                for i, j in set(zip(a[both].tolist(), b[both].tolist())):
                    root[find(i)] = find(j)
    return len({find(k) for k in range(len(parts))})


def _brush_sizes(size, count: int) -> list[int]:
    """A curve's brush size at each of its `count` points: one number for all, or one per point."""
    from rok.core.edit import BRUSH_MAX

    if size is None:
        return [1] * count
    sizes = list(size) if isinstance(size, (list, tuple)) else [size]
    if len(sizes) == 1:
        sizes = sizes * count
    if len(sizes) != count:
        raise ToolError(f"size takes one brush size, or one per point: {count} for these points, got {len(sizes)}")
    try:
        sizes = [int(v) for v in sizes]
    except (TypeError, ValueError):
        raise ToolError("size must be whole numbers, 1 to 50: the brush sizes of Voxel mode") from None
    if not all(1 <= v <= BRUSH_MAX for v in sizes):
        raise ToolError(f"size must be 1 to {BRUSH_MAX}: 1 is one voxel, 2 is 3 across, 3 is 5 across")
    return sizes


def _turned(cells: np.ndarray, values: np.ndarray, axis: int, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """The shape's cells turned by `degrees` about `axis` through the middle of
    their box, by the free turn of Model > Rotate (each new cell takes the colour
    of the cell it came from, so a turned shape keeps its edges clean)."""
    from rok.core import free_rotate

    if not len(cells):
        return cells, values
    lo = cells.min(axis=0)
    dims = cells.max(axis=0) - lo + 1
    kinds, index = np.unique(values, return_inverse=True)
    box = np.zeros(tuple(int(v) for v in dims), dtype=np.uint8)
    box[tuple((cells - lo).T)] = index.reshape(-1) + 1
    turned = free_rotate.rotate_voxels(box, axis, degrees, limit=None)
    new = np.argwhere(turned)
    # the same middle before and after the turn
    shift = np.floor(lo + (dims - 1) / 2 - (np.asarray(turned.shape) - 1) / 2 + 0.5).astype(np.int64)
    return new + shift, kinds[turned[tuple(new.T)].astype(np.int64) - 1]


def _face_joined(cells: np.ndarray) -> np.ndarray:
    """A line's cells with a cell added inside each diagonal step, so that
    neighbours share a face: z changes last, so a leaning trunk steps sideways
    first and then up."""
    out = [cells[0]]
    for nxt in cells[1:]:
        cur = out[-1].copy()
        for axis in (0, 1, 2):
            if cur[axis] != nxt[axis]:
                cur = cur.copy()
                cur[axis] = nxt[axis]
                out.append(cur)
    return np.asarray(out, dtype=np.int64)


def _xyz(v, what: str = "cell") -> tuple[int, int, int]:
    try:
        x, y, z = (int(round(float(c))) for c in v)
    except (TypeError, ValueError):
        raise ToolError(f"{what} must be [x, y, z], three whole numbers; got {v!r}") from None
    return x, y, z


def _off_centre(m) -> tuple[int, int] | None:
    """Per x and y, twice the shift that would centre the voxels in the grid (0: centred,
    so a mirror maps them onto themselves); None for an empty model."""
    filled = np.argwhere(m.grid.voxels != 0)
    if len(filled) == 0:
        return None
    lo, hi = filled.min(axis=0), filled.max(axis=0)
    return tuple(int(m.grid.size[i]) - 1 - int(lo[i]) - int(hi[i]) for i in (0, 1))


def _resize_note(m, was) -> str:
    """A note when a resize took a centred model off the middle: resize keeps the low corner."""
    now = _off_centre(m)
    if was is None or now is None:
        return ""
    moved = [i for i in (0, 1) if was[i] == 0 and now[i] != 0]
    if not moved:
        return ""
    filled = np.argwhere(m.grid.voxels != 0)
    lo, hi = filled.min(axis=0), filled.max(axis=0)
    where = ", ".join(f"{AXES[i]} {lo[i]}..{hi[i]} of 0..{m.grid.size[i] - 1}" for i in moved)
    offset = [int(now[0] / 2) if 0 in moved else 0, int(now[1] / 2) if 1 in moved else 0, 0]
    note = (f" The voxels keep their cells from the low corner, so the model is off the middle now ({where}): "
            "a mirror works across the middle of the grid.")
    if any(offset):
        note += f" To centre it: rok_select do=all, then rok_transform do=move offset {offset}."
    if any(now[i] % 2 for i in moved):
        note += (" It cannot sit exactly in the middle: the grid and the model differ by an odd number of cells; "
                 "a grid one cell wider or narrower fixes that.")
    return note


def _axis(a) -> int:
    s = str(a).lower()
    if s not in AXES:
        raise ToolError(f"axis must be x, y or z; got {a!r}")
    return AXES.index(s)


def _colour(c) -> int:
    try:
        c = int(c)
    except (TypeError, ValueError):
        raise ToolError(f"colour must be a palette number 1 to 255; got {c!r}") from None
    if not 0 <= c <= 255:
        raise ToolError(f"colour must be a palette number 1 to 255 (0 empties a cell); got {c}")
    return c


class Agent(AnimTools, ArtTools):
    def __init__(self, app, host) -> None:
        self.app = app
        self.host = host
        self.feedback = "image"
        self.size = DEFAULT_SIZE
        self.last_cameras: list = []  # (name, x, y, w, h, camera) of the last picture
        self._batch_depth = 0
        self._ink = RULER
        self._plate = (0.07, 0.07, 0.08, 0.65)
        self._cached_shaders()

    # --- the document ---------------------------------------------------------------

    @property
    def doc(self):
        from rok.app.object import document

        return document

    def _model(self, name=None):
        """The active model, or the model named `name`, made active."""
        doc = self.doc
        scene = doc.scene
        if scene is None or doc.model is None:
            raise ToolError("no document is open; use rok_scene do=new or do=open")
        if name in (None, ""):
            return doc.model
        for i, m in enumerate(scene.models):
            if m.display_name(i) == name or m.name == name:
                if m.uid != doc.active_uid:
                    self.app_run("rok.model_set_active", uid=m.uid)
                return m
        names = ", ".join(m.display_name(i) for i, m in enumerate(scene.models))
        raise ToolError(f"no model named {name!r}; the models are: {names}")

    def _name(self, model) -> str:
        scene = self.doc.scene
        return model.display_name(scene.models.index(model))

    def app_run(self, idname: str, **props) -> set:
        from rok.app import opkit

        if idname not in opkit.REGISTRY:
            raise ToolError(f"no command {idname!r}; rok_command do=list shows them")
        messages: list[str] = []
        say = self.app.say

        def capture(message: str, seconds: float = 3.0) -> None:
            messages.append(str(message))
            say(message, seconds)

        self.app.say = capture
        try:
            status = set(opkit.call(idname, "EXEC", **props) or ())
        except TypeError as exc:
            raise ToolError(f"{idname}: {exc}") from None
        finally:
            self.app.say = say
            self.app.after_event()
        if "FINISHED" not in status and "RUNNING" not in status:
            why = "; ".join(messages) or "it cannot run now (nothing to act on, or the wrong level)"
            raise ToolError(f"{idname} did not run: {why}")
        return status

    def _snapshot(self) -> dict:
        scene = self.doc.scene
        if scene is None:
            return {}
        return {m.uid: (m.grid.voxels.copy(), tuple(int(p) for p in m.position)) for m in scene.models}

    def _world_cells(self, voxels, position) -> tuple[np.ndarray, np.ndarray]:
        cells = np.argwhere(voxels != 0)
        return cells + np.asarray(position, dtype=np.int64), voxels[tuple(cells.T)]

    def _diff(self, before: dict) -> dict:
        """World cells this call added (or recoloured) and emptied, over every model."""
        scene = self.doc.scene
        added, removed, recoloured = [], [], []
        now = {m.uid: (m.grid.voxels, tuple(int(p) for p in m.position)) for m in scene.models} if scene else {}
        for uid in set(before) | set(now):
            b, a = before.get(uid), now.get(uid)
            if b is not None and a is not None and b[0].shape == a[0].shape and b[1] == a[1]:
                bv, av = b[0], a[0]
                pos = np.asarray(a[1], dtype=np.int64)
                added.append(np.argwhere((bv == 0) & (av != 0)) + pos)
                removed.append(np.argwhere((bv != 0) & (av == 0)) + pos)
                recoloured.append(np.argwhere((bv != 0) & (av != 0) & (bv != av)) + pos)
                continue
            # A new grid size or position: compare the world cells, so a resize that
            # leaves every voxel where it was reports no change.
            bc, bv = self._world_cells(*b) if b is not None else (np.zeros((0, 3), np.int64), np.zeros(0))
            ac, av = self._world_cells(*a) if a is not None else (np.zeros((0, 3), np.int64), np.zeros(0))
            if len(bc) and len(ac):
                lo = np.minimum(bc.min(axis=0), ac.min(axis=0))
                span = np.maximum(bc.max(axis=0), ac.max(axis=0)) - lo + 1
                weight = np.array([span[1] * span[2], span[2], 1], dtype=np.int64)
                _common, ib, ia = np.intersect1d((bc - lo) @ weight, (ac - lo) @ weight, return_indices=True)
                gone, new = np.ones(len(bc), bool), np.ones(len(ac), bool)
                gone[ib], new[ia] = False, False
                recoloured.append(ac[ia][bv[ib] != av[ia]])
                bc, ac = bc[gone], ac[new]
            removed.append(bc)
            added.append(ac)
        cat = lambda parts: np.concatenate(parts) if parts else np.zeros((0, 3), np.int64)  # noqa: E731
        return {"added": cat(added), "removed": cat(removed), "recoloured": cat(recoloured)}

    # --- running a tool ----------------------------------------------------------------

    def call(self, tool: str, args: dict | None = None) -> Reply:
        """Run one tool. Errors the model can fix come back as an error Reply."""
        args = dict(args or {})
        fn = getattr(self, "tool_" + tool.removeprefix("rok_"), None)
        if fn is None:
            return Reply(f"no tool {tool!r}", error=True)
        try:
            inspect.signature(fn).bind(**args)
        except TypeError as exc:
            return Reply(f"{tool}: {exc}. The tool description lists its parameters.", error=True)
        try:
            with naming(_undo_label(tool, args)):
                reply = fn(**args)
            if reply.picture is not None:
                reply.text += "\n" + self.layout(reply.picture)
            return reply
        except ToolError as exc:
            return Reply(str(exc), error=True)
        except BrickLocked as exc:
            return Reply(f"{exc} (rok_command do=run id=convert_to_voxels)", error=True)
        except Exception as exc:
            # A fault in a tool still answers the client: an unanswered live call
            # would leave it waiting for the whole timeout.
            return Reply(f"{tool} failed: {type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()},
                         error=True)

    def _edit(self, fn, *, look: bool = True) -> Reply:
        """Run `fn` (which changes the document) and build the feedback."""
        before = self._snapshot()
        before_box = self._voxel_box(self.doc.model) if self.doc.model is not None else None
        pushes = self.doc.undo.pushes
        note = fn() or ""
        diff = self._diff(before)
        # A step can break the rules and change no voxel: a resize that keeps every voxel in place.
        # The count of pushes tells it; the length of a full undo list does not change.
        stepped = self.doc.undo.pushes > pushes
        if self._batch_depth == 0 and (stepped or any(len(diff[k]) for k in diff)):
            self.enforce_art()
        reply = self._feedback(diff, note, before_box)
        if look and self.feedback == "image" and self._batch_depth == 0:
            self._attach_picture(reply, diff)
        return reply

    def _attach_picture(self, reply: Reply, diff: dict) -> None:
        """The feedback picture: flat views cell by cell in an art scene, else the quad."""
        if self.art is not None and self.doc.model is not None:
            m = self.doc.model
            changed = np.concatenate([diff["added"], diff["recoloured"]]) - np.asarray(m.position, dtype=np.int64)
            reply.image, line = self.flat_png(m, changed)
            reply.text += "\n" + line
            return
        reply.picture = Picture(("top", "iso", "front", "right"), self.size, diff, closeup=self._closeup(diff))

    def _closeup(self, diff: dict):
        """A close-up box for a change too small to judge in the overview, else None."""
        cells = [diff[k] for k in ("added", "removed", "recoloured") if len(diff[k])]
        if not cells:
            return None
        from rok.app.object import document

        cells = np.concatenate(cells)
        off = np.asarray(document.view_offset, dtype=np.float64)
        lo, hi = self._scene_box()
        overview = (self.size[0] / 2.0) / max(float(np.max(hi - lo)), 1.0)
        c_lo = cells.min(axis=0) + off - CLOSEUP_PAD
        c_hi = cells.max(axis=0) + off + 1 + CLOSEUP_PAD
        close = (self.size[0] / len(CLOSEUP_VIEWS)) / max(float(np.max(c_hi - c_lo)), 1.0)
        if overview >= CLOSE_PPC or close < 1.5 * overview:
            return None
        return (tuple(float(v) for v in c_lo), tuple(float(v) for v in c_hi))

    @staticmethod
    def tiles_of(picture: Picture) -> tuple[list, int, int]:
        """[(name, x, y, w, h, box, closeup)] for every view, and the whole picture's size.
        A strip of frames has one square tile per frame, STRIP_COLUMNS to a row."""
        from rok.headless import QUAD, tiles

        names = list(QUAD) if picture.views in ("quad", ("quad",)) else list(picture.views)
        w, h = int(picture.size[0]), int(picture.size[1])
        if picture.render is None and picture.frames is not None and len(picture.frames) > 1:
            n = len(picture.frames)
            cols = min(n, STRIP_COLUMNS)
            rows = -(-n // cols)
            tw = th = w // cols  # square tiles: the picture is as high as the rows need
            out = [(names[0], float(k % cols * tw), float(k // cols * th), float(tw), float(th), picture.box, False)
                   for k in range(n)]
            return out, tw * cols, th * rows
        out = [(n, x, y, tw, th, picture.box, picture.box is not None) for n, (x, y, tw, th) in zip(names, tiles(len(names), w, h))]
        if picture.closeup is not None:
            n_views = len(CLOSEUP_VIEWS)
            side = w // n_views
            for k, n in enumerate(CLOSEUP_VIEWS):
                tw = side if k < n_views - 1 else w - side * (n_views - 1)  # whole pixels, no gap
                out.append((n, float(k * side), float(h), float(tw), float(side), picture.closeup, True))
            h += side
        return out, w, h

    @classmethod
    def layout(cls, picture: Picture) -> str:
        """Where each view sits in the picture, in pixels: models aim rok_probe with it."""
        if picture.render is not None:
            return f"Picture {int(picture.size[0])} x {int(picture.size[1])} pixels."
        rows, w, h = cls.tiles_of(picture)
        frames = list(picture.frames or ())
        parts = []
        for k, (n, x, y, tw, th, _box, close) in enumerate(rows):
            name = ("CLOSE-UP " if close else "") + (n.upper() if n != "iso" else "3D")
            if len(frames) > 1:
                name += f" AT FRAME {frames[k]}"
            parts.append(f"{name} x {x:g}..{x + tw:g}, y {y:g}..{y + th:g}")
        posed = f" The parts show as the clip poses them at frame {frames[0]}." if len(frames) == 1 else ""
        return f"Picture {w} x {h} pixels, y from the top: " + "; ".join(parts) + "." + posed

    # --- feedback ----------------------------------------------------------------------

    @staticmethod
    def _voxel_box(model):
        filled = np.argwhere(model.grid.voxels != 0)
        if len(filled) == 0:
            return None
        return filled.min(axis=0), filled.max(axis=0)

    def facts(self, model=None) -> dict:
        model = model or self.doc.model
        scene = self.doc.scene
        v = model.grid.voxels
        values, counts = np.unique(v[v != 0], return_counts=True)
        order = np.argsort(-counts)
        box = self._voxel_box(model)
        sel = model.selection
        return {
            "model": self._name(model),
            "grid": [int(s) for s in model.grid.size],
            "position": [int(p) for p in model.position],
            "voxels": int(counts.sum()),
            "bounds": None if box is None else {"low": [int(c) for c in box[0]], "high": [int(c) for c in box[1]]},
            "colours": {int(values[i]): int(counts[i]) for i in order},
            "where": self._where(model, values[order]),
            "selected": int(sel.count()) if sel is not None else 0,
            "models": [self._name(m) for m in scene.models],
            "colour": int(self.doc.colour),
        }

    def _where(self, model, values) -> dict:
        """For each colour: its hex and the box its voxels fill, so a model finds
        a part by its colour without probing."""
        v = model.grid.voxels
        palette = self.doc.scene.palette
        cells = np.argwhere(v != 0)
        colours = v[tuple(cells.T)].astype(np.int64)
        low = np.full((256, 3), np.iinfo(np.int64).max, dtype=np.int64)
        high = np.full((256, 3), -1, dtype=np.int64)
        for a in range(3):  # one pass over the filled cells, not one per colour
            np.minimum.at(low[:, a], colours, cells[:, a])
            np.maximum.at(high[:, a], colours, cells[:, a])
        out = {}
        for c in values:
            c = int(c)
            out[c] = {"hex": "#" + "".join(f"{int(k):02x}" for k in palette[c, :3]),
                      "low": low[c].tolist(), "high": high[c].tolist()}
        return out

    def _feedback(self, diff: dict, note: str, before_box) -> Reply:
        model = self.doc.model
        f = self.facts(model)
        n_add, n_rem, n_rec = (len(diff[k]) for k in ("added", "removed", "recoloured"))
        lines = []
        if note:
            lines.append(note)
        lines.append(f"Changed: +{n_add} added, -{n_rem} removed, {n_rec} recoloured.")
        b = f["bounds"]
        was = "" if before_box is None else f" (was {list(map(int, before_box[0]))}..{list(map(int, before_box[1]))})"
        box = "empty" if b is None else f"cells {b['low']}..{b['high']}{was}"
        lines.append(f"Model \"{f['model']}\": grid {f['grid'][0]} x {f['grid'][1]} x {f['grid'][2]}, "
                     f"{f['voxels']} voxels, {box}.")
        if f["colours"]:
            items = list(f["colours"].items())
            more = f" and {len(items) - SHORT_COLOURS} more (rok_look detail=full lists all)" if len(items) > SHORT_COLOURS else ""
            lines.append("Colours (number: count): " + ", ".join(f"{c}: {n}" for c, n in items[:SHORT_COLOURS]) + more)
        if f["selected"]:
            lines.append(f"Selected: {f['selected']} voxels.")
        warnings = self._warnings(model)
        changed = np.concatenate([diff["added"], diff["recoloured"]]) - np.asarray(model.position, dtype=np.int64)
        inside = np.all((changed >= 0) & (changed < np.asarray(model.grid.size)), axis=1)
        # Hidden by voxels this change did not make: a table in a closed room, paint
        # inside a solid. A new solid's own inside is no news (test builds got this
        # warning on nearly every box).
        others = model.grid.voxels != 0
        c = changed[inside]
        others[tuple(c.T)] = False
        n_hidden = int(self.hidden(others, c).sum())
        if n_hidden:
            warnings.insert(0, f"{n_hidden} of the changed cells are hidden from every side (inside the model): "
                               "the picture shows them as faint outlines; rok_look with cut shows inside.")
        warnings += self.palette_warnings() if self.art is not None else []
        if warnings:
            lines.append("Warnings: " + " ".join(warnings))
        data = {"added": n_add, "removed": n_rem, "recoloured": n_rec, "facts": f, "warnings": warnings}
        if 0 < f["voxels"] <= CRAFT_MAX_VOXELS:
            rep = craft.check(model.grid.voxels)
            lines.append(craft.summary(rep, self.colour_names()))
            data["craft"] = {"parts": len(rep.parts), "symmetry": rep.symmetry,
                             "jagged": len(rep.jags), "thin": len(rep.thin), "survives": rep.survives}
        if self.art is not None:
            # Only the feedback that is sent, with its picture of the shape, ends the
            # blockout: never a step inside a batch.
            status = self.blockout_status() if self._batch_depth == 0 else ""
            if status:
                lines.append(status)
            lines.append(self.art_line())
        return Reply("\n".join(lines), data)

    def _warnings(self, model) -> list[str]:
        out = []
        mask = model.grid.voxels != 0
        count = int(mask.sum())
        if 0 < count <= 200_000:
            parts = region.components(mask)
            if len(parts) > 1:
                small = ", ".join(str(len(p)) for p in parts[1:6])
                more = "" if len(parts) <= 6 else f" and {len(parts) - 6} more"
                out.append(f"{len(parts)} separate parts: the largest has {len(parts[0])} voxels, "
                           f"the others {small}{more} (rok_check lists where).")
        return out

    # --- pictures ----------------------------------------------------------------------

    @staticmethod
    def hidden(mask: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """For each cell, True when filled cells hide it from all six sides."""
        if len(cells) == 0:
            return np.zeros(0, dtype=bool)
        c = tuple(cells.T)
        out = np.ones(len(cells), dtype=bool)
        m = mask.astype(np.int32)
        for axis in range(3):
            before = np.cumsum(m, axis=axis) - m  # filled cells on the low side
            after = np.flip(np.cumsum(np.flip(m, axis), axis=axis), axis) - m  # on the high side
            out &= (before[c] > 0) & (after[c] > 0)
        return out

    def _background(self) -> float:
        """A light background when the shown voxels are mostly dark, else the editor's."""
        scene = self.doc.scene
        lum, total = 0.0, 0
        for m in scene.models if scene is not None else ():
            v = m.grid.voxels
            values, counts = np.unique(v[v != 0], return_counts=True)
            rgb = scene.palette[values, :3].astype(np.float64) / 255.0
            lum += float((rgb @ np.array([0.2126, 0.7152, 0.0722]) * counts).sum())
            total += int(counts.sum())
        return LIGHT_BG if total and lum / total < 0.22 else 0.0

    def queue_picture(self, picture: Picture, token: str) -> None:
        """Send the agent layer and the render request; a "screenshot" event
        with `token` brings the picture back. A strip of frames sends one
        request per frame (`picture_tokens`); `assemble` joins them."""
        if picture.render is not None:
            self._queue_render(picture, token)
            return
        from rok.app import draw
        from rok.headless import camera_for, depth_range

        frames = tuple(picture.frames or ())
        tiles_, w, h = self.tiles_of(picture)
        lo, hi = self._voxel_scene_box(frames) if len(frames) > 1 else self._scene_box(frames)
        background = self._background()
        self._ink = (0.10, 0.10, 0.12, 0.95) if background else RULER  # rulers dark on a light background
        self._plate = (0.93, 0.93, 0.94, 0.75) if background else (0.07, 0.07, 0.08, 0.65)  # behind ruler numbers
        self.last_cameras = []
        if len(frames) > 1:
            for k, ((name, x, y, tw, th, _box, _close), t) in enumerate(zip(tiles_, frames)):
                self._pose(t)
                cam = camera_for(name, tw, th, lo, hi)
                near, depth = depth_range(cam, lo, hi)
                d = draw.DrawList()
                self._annotate(d, name, cam, (0.0, 0.0, tw, th))
                self._frame_label(d, t, tw)
                self.host.set_draw(LAYER, 900, **d.arrays(), icons=d.icons, texts=d.texts)
                row = (0.0, 0.0, tw, th, cam.view_proj(), tuple(float(c) for c in cam.eye()), float(cam.pixel), near, depth)
                self.host.render_views(f"{token}#{k}", int(tw), int(th), [row], background)
            self._unpose()
            return
        if frames:
            self._pose(frames[0])
        d = draw.DrawList()
        rows = []
        for name, x, y, tw, th, box, close in tiles_:
            blo, bhi = (lo, hi) if box is None else (np.asarray(box[0], float), np.asarray(box[1], float))
            if picture.camera is not None and not close:
                cam = self._free_camera(picture.camera, tw, th, blo, bhi)
            else:
                cam = camera_for(name, tw, th, blo, bhi, clip=close)
            near, depth = depth_range(cam, blo, bhi)  # farther surfaces darker: holes show in flat views
            rows.append((x, y, tw, th, cam.view_proj(), tuple(float(c) for c in cam.eye()), float(cam.pixel), near, depth))
            if not frames:
                self.last_cameras.append((name, x, y, tw, th, cam))  # rok_probe knows the rest pose only
            self._annotate(d, name, cam, (x, y, tw, th), close)
            if frames:
                self._frame_label(d, frames[0], tw, x, y)
        if picture.marks:
            self._marks(d, picture.marks)
        self.host.set_draw(LAYER, 900, **d.arrays(), icons=d.icons, texts=d.texts)
        restore = self._cut_models(picture.cut) if picture.cut else []
        self.host.render_views(token, w, h, rows, background)
        for i, voxels, position in restore:
            self.host.set_model(i, voxels, position)  # the cut was for this picture only
        if frames:
            self._unpose()

    def _queue_render(self, picture: Picture, token: str) -> None:
        """One path-traced picture with a render preset (rok_render)."""
        from dataclasses import replace

        from rok.app import render_picture, render_view
        from rok.core import light as light_core
        from rok.core import render_presets as rp
        from rok.headless import camera_for

        frames = tuple(picture.frames or ())
        if frames:
            self._pose(frames[0])
        lo, hi = self._voxel_scene_box(frames)
        preset_id, samples, *rest = picture.render
        w, h = int(picture.size[0]), int(picture.size[1])
        preset = replace(rp.preset_by_id(preset_id), resolution=(w, h))
        own = len(rest) > 1 and bool(rest[1])
        setup = render_view.scene_light() if own else light_core.from_preset(preset)
        if rest:
            setup = light_core.with_ground(setup, bool(rest[0]), preset)
        if len(rest) > 2 and rest[2]:
            setup = replace(setup, **rest[2])  # bloom and exposure for this picture only
        if picture.camera is not None:
            cam = self._free_camera(picture.camera, w, h, lo, hi)
        else:
            view = camera_for(picture.views[0], w, h, lo, hi)
            view.distance *= RENDER_ROOM
            cam = render_picture.picture_camera(view, preset, lo, hi)
        settings = render_view.settings_for(setup, lo, hi)
        settings["max_samples"] = int(samples)
        self.last_cameras = []
        self.host.render_picture(token, w, h, cam.view_proj(), tuple(float(v) for v in cam.eye()), **settings)
        if frames:
            self._unpose()

    @staticmethod
    def _free_camera(spec: dict, w: float, h: float, lo, hi):
        from rok.headless import SessionError, free_camera

        spec = dict(spec)
        frame = spec.pop("frame", None)  # a box to frame instead of the scene
        if frame is not None:
            lo, hi = np.asarray(frame[0], float), np.asarray(frame[1], float)
        try:
            return free_camera(w, h, lo, hi, **spec)
        except SessionError as exc:
            raise ToolError(str(exc)) from None

    def camera_spec(self, m, azimuth=None, elevation=None, eye=None, target=None, distance=None, fov=None,
                    ortho=None) -> tuple[dict | None, str]:
        """A hand-placed camera from a tool's arguments (cells of model m), as
        Picture.camera in world units, and a line that says where it stands.
        (None, "") when no argument places the camera."""
        if all(v is None for v in (azimuth, elevation, eye, target, distance, fov, ortho)):
            return None, ""
        from rok.app.object import model_origin

        o = np.asarray(model_origin(m), dtype=np.float64)

        def point(v, what):
            try:
                p = [float(c) for c in v]
            except (TypeError, ValueError):
                p = []
            if len(p) != 3:
                raise ToolError(f"{what} must be [x, y, z] in cells (fractions allowed); got {v!r}")
            return tuple(np.asarray(p) + o)

        if eye is not None and (azimuth is not None or elevation is not None or distance is not None):
            raise ToolError("place the camera with eye (and target), or with azimuth, elevation and distance; not both")
        if fov is not None and not 5 <= float(fov) <= 120:
            raise ToolError("fov is the vertical field of view in degrees, 5 to 120 (default 35)")
        if elevation is not None and not -90 <= float(elevation) <= 90:
            raise ToolError("elevation is degrees above the ground, -90 to 90")
        spec = {"fov": None if fov is None else float(fov), "ortho": bool(ortho)}
        if eye is not None:
            spec["eye"] = point(eye, "eye")
        if target is not None:
            spec["target"] = point(target, "target")
        if eye is not None and target is not None and np.allclose(spec["eye"], spec["target"]):
            raise ToolError("eye and target are the same point; the camera needs a direction")
        for k, v in (("azimuth", azimuth), ("elevation", elevation), ("distance", distance)):
            if v is not None:
                spec[k] = float(v)
        lens = "orthographic" if ortho else f"{35.0 if fov is None else float(fov):g}° field of view"
        if eye is not None:
            where = f"Camera at {[round(float(c), 2) for c in eye]} looking at " + (
                f"{[round(float(c), 2) for c in target]}" if target is not None else "the middle of the frame box")
        else:
            where = (f"Camera at azimuth {spec.get('azimuth', -45.0):g}°, elevation {spec.get('elevation', 35.0):g}°"
                     + (f", {spec['distance']:g} cells from " if "distance" in spec else ", framing ")
                     + (f"{[round(float(c), 2) for c in target]}" if target is not None else "the frame box"))
        return spec, f"{where}, {lens}."

    @staticmethod
    def picture_tokens(picture: Picture, token: str) -> list[str]:
        """The screenshot tokens `queue_picture` asks for, in order."""
        frames = picture.frames or ()
        if picture.render is None and len(frames) > 1:
            return [f"{token}#{k}" for k in range(len(frames))]
        return [token]

    def assemble(self, picture: Picture, shots: list) -> np.ndarray:
        """The (h, w, 4) picture from the screenshots of `picture_tokens`: each
        is (width, height, rgba bytes)."""
        images = [np.frombuffer(rgba, dtype=np.uint8).reshape(sh, sw, 4) for sw, sh, rgba in shots]
        if len(images) == 1:
            return images[0]
        tiles_, w, h = self.tiles_of(picture)
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[..., 3] = 255
        for (_n, x, y, tw, th, _b, _c), img in zip(tiles_, images):
            x, y = int(x), int(y)
            ih, iw = min(img.shape[0], h - y), min(img.shape[1], w - x)
            out[y:y + ih, x:x + iw] = img[:ih, :iw]
        return out

    @staticmethod
    def finish_picture(picture: Picture, png: bytes) -> None:
        """Save the finished PNG where the tool was asked to (rok_render path)."""
        if picture.save:
            from pathlib import Path

            path = Path(picture.save)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)

    def _pose(self, t: float) -> None:
        """Show the parts as the active clip poses them at frame t, for the next
        render request only; `_unpose` gives the view its own parts back."""
        from rok.app import anim_mode
        from rok.app import object as voxel_object

        shown = anim_mode.shown_at(anim_mode.clip(), float(t))
        voxel_object.anim_hidden = shown.hidden
        self.app.sync.flush(self.doc.scene, style_of=voxel_object.style_of, turns=shown.turns or None,
                            poses=shown.poses, palette=shown.palette)

    def _unpose(self) -> None:
        from rok.app import object as voxel_object

        voxel_object.anim_hidden = frozenset()
        self.app.flush()

    def _frame_label(self, d, t: int, tw: float, x0: float = 0.0, y0: float = 0.0) -> None:
        label = f"frame {int(t)}"
        lw, _lh = self.app.measure_text(label, 13)
        d.rect(x0 + tw - lw - 20, y0 + 6, lw + 12, 22, PLATE, radius=4)
        d.text(x0 + tw - lw - 14, y0 + 9, label, 13, HIGHLIGHT)

    def _cut_models(self, cut) -> list:
        """Send every model with the cells on the cut-away side removed; return
        what to send back after the picture."""
        axis, at, keep = cut
        model = self.doc.model
        at_world = int(model.position[axis]) + int(at)
        restore = []
        for i, m in enumerate(self.doc.scene.models):
            v = np.ascontiguousarray(m.grid.visible)
            coord = int(m.position[axis]) + np.arange(v.shape[axis])
            drop = coord < at_world if keep == "high" else coord >= at_world
            if not drop.any():
                continue
            shown = v.copy()
            index = [slice(None)] * 3
            index[axis] = drop
            shown[tuple(index)] = 0
            position = tuple(float(p) for p in m.position)
            self.host.set_model(i, shown, position)
            restore.append((i, v, position))
        return restore

    def _scene_box(self, frames=()):
        """The world box round every shown model, and round the posed parts at `frames`."""
        from rok.app.object import model_origin

        scene = self.doc.scene
        boxes = []
        for m in scene.models if scene is not None else ():
            if scene.is_hidden(m):
                continue
            o = np.asarray(model_origin(m), dtype=np.float64)
            boxes.append((o, o + np.asarray(m.grid.size, dtype=np.float64)))
        if frames:
            from rok.app import anim_mode, anim_pictures

            for t in frames:
                b = anim_pictures.shown_box(anim_mode.shown_at(anim_mode.clip(), float(t)))
                if b is not None:
                    boxes.append(b)
        if not boxes:
            return np.zeros(3), np.ones(3)
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)

    def _voxel_scene_box(self, frames=()):
        """The world box round the filled cells of every shown model, and round
        the posed parts at `frames`: a picture framed on what is there."""
        from rok.app import anim_mode, anim_pictures
        from rok.app.object import model_origin

        scene = self.doc.scene
        boxes = []
        for m in scene.models if scene is not None else ():
            filled = np.argwhere(m.grid.voxels != 0)
            if scene.is_hidden(m) or not len(filled):
                continue
            o = np.asarray(model_origin(m), dtype=np.float64)
            boxes.append((o + filled.min(axis=0), o + filled.max(axis=0) + 1.0))
        for t in frames:
            b = anim_pictures.shown_box(anim_mode.shown_at(anim_mode.clip(), float(t)))
            if b is not None:
                boxes.append(b)
        if not boxes:
            return self._scene_box(frames)
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)

    def _marks(self, d, marks: dict) -> None:
        """Outline the changed cells. Few: one outline each, and the hidden ones
        faintly through the voxels in front. Many: one box round them all."""
        from rok.app.draw import box_segments
        from rok.app.object import document

        off = np.asarray(document.view_offset, dtype=np.float64)
        model = document.model
        for key, colour in (("added", HIGHLIGHT), ("recoloured", HIGHLIGHT), ("removed", GHOST)):
            cells = marks.get(key)
            if cells is None or len(cells) == 0:
                continue
            if len(cells) > MAX_MARKS:
                d.box3(cells.min(axis=0) + off - 0.1, cells.max(axis=0) + off + 1.1, colour, 2.5, on_top=True)
                continue
            segs = np.concatenate([box_segments(c + off - 0.04, c + off + 1.04) for c in cells.astype(np.float64)])
            d.lines3(segs, colour, 1.5)
            if key != "removed" and model is not None:
                local = cells - np.asarray(model.position, dtype=np.int64)
                inside = np.all((local >= 0) & (local < np.asarray(model.grid.size)), axis=1)
                hid = np.zeros(len(cells), dtype=bool)
                hid[inside] = self.hidden(model.grid.voxels != 0, local[inside])
                if hid.any():
                    faint = (colour[0], colour[1], colour[2], 0.45)
                    xs = np.concatenate([box_segments(c + off - 0.04, c + off + 1.04) for c in cells[hid].astype(np.float64)])
                    d.lines3(xs, faint, 1.0, on_top=True)

    def _annotate(self, d, name: str, cam, tile, close: bool = False) -> None:
        """The view's name, and rulers in the active model's cells (orthographic
        views) or labelled axes (the 3D view)."""
        from rok.app.object import model_origin

        x0, y0, tw, th = tile
        # Name the side the camera is on: models mix up front and back without it.
        label = {"top": "TOP: from +z, looking down (x right, y up)", "bottom": "BOTTOM: from -z, looking up",
                 "front": "FRONT: from -y (x right, z up)", "back": "BACK: from +y (x left, z up)",
                 "right": "RIGHT: from +x (y right, z up)", "left": "LEFT: from -x (y left, z up)",
                 "iso": "3D: from +x, -y and above", "camera": "CAMERA: placed by hand"}.get(name, name)
        if close:
            label = "CLOSE-UP " + label
        lw, _lh = self.app.measure_text(label, 12)
        d.rect(x0 + 6, y0 + 6, lw + 12, 20, PLATE, radius=4)
        d.text(x0 + 12, y0 + 9, label, 12, RULER)
        model = self.doc.model
        if model is None:
            return
        origin = np.asarray(model_origin(model), dtype=np.float64)
        size = np.asarray(model.grid.size)
        if not cam.ortho:
            # Drawn flat into this tile only: 3D lines would show in every view.
            length = max(3.0, float(size.min()) * 0.6)
            base = cam.project([origin])[0]
            for a in range(3):
                end = origin.copy()
                end[a] += length
                px = cam.project([end])[0]
                if np.all(np.isfinite(px)) and np.all(np.isfinite(base)):
                    d.line2(x0 + base[0], y0 + base[1], x0 + px[0], y0 + px[1], AXIS_RGB[a], 2.5)
                    d.text(x0 + px[0] + 3, y0 + px[1] - 8, AXES[a], 13, AXIS_RGB[a])
            return
        v = cam.view()
        for row, edge in ((0, "bottom"), (1, "left")):
            screen_axis = v[row, :3]
            a = int(np.argmax(np.abs(screen_axis)))
            px_per_cell = 1.0 / max(cam.pixel, 1e-9)
            step = next((k for k in (1, 2, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100) if k * px_per_cell >= 26), 128)
            centre = (origin + size / 2.0)
            for i in range(0, int(size[a]) + 1, step):
                p = centre.copy()
                p[a] = origin[a] + i
                sx, sy = cam.project([p])[0]
                if not (np.isfinite(sx) and np.isfinite(sy)):
                    continue
                if edge == "bottom" and 0 <= sx <= tw:
                    d.line2(x0 + sx, y0 + th - 14, x0 + sx, y0 + th - 4, self._ink, 1.0)
                    d.rect(x0 + sx + 1, y0 + th - 31, 7 * len(str(i)) + 3, 15, self._plate, radius=3)
                    d.text(x0 + sx + 2, y0 + th - 30, str(i), 11, self._ink)
                elif edge == "left" and 34 <= sy <= th - 20:
                    d.line2(x0 + 4, y0 + sy, x0 + 14, y0 + sy, self._ink, 1.0)
                    d.rect(x0 + 15, y0 + sy - 8, 7 * len(str(i)) + 3, 15, self._plate, radius=3)
                    d.text(x0 + 16, y0 + sy - 7, str(i), 11, self._ink)
            name_axis = AXES[a] + ("" if screen_axis[a] > 0 else " (decreasing)")
            if edge == "bottom":
                d.text(x0 + tw - 90, y0 + th - 48, f"{name_axis} →", 12, self._ink)
            else:
                d.text(x0 + 30, y0 + 34, f"{name_axis} ↑", 12, self._ink)

    # --- the tools -----------------------------------------------------------------------

    def tool_look(self, views="quad", size=None, facts: bool = True, heightmap: bool = False, cut_axis=None,
                  cut_at=None, cut_keep: str = "high", low=None, high=None, model=None, azimuth=None,
                  elevation=None, eye=None, target=None, distance=None, fov=None, ortho=None,
                  scale: int | None = None, text: bool = False, detail: str = "short") -> Reply:
        """A picture now, and the facts. `scale` (pixels per cell) draws the flat views
        cell by cell, as an art scene always does; `text` adds them as characters."""
        m = self._model(model)
        if scale is not None and not 1 <= int(scale) <= 32:
            raise ToolError("scale is 1 to 32 pixels per cell")
        if detail not in ("short", "full"):
            raise ToolError("detail is short or full")
        camera, camera_line = self.camera_spec(m, azimuth, elevation, eye, target, distance, fov, ortho)
        if camera is not None:
            views = ("camera",)  # one view from the placed camera
        if isinstance(views, str):
            views = (views,)
        views = tuple(views)
        if views == ("quad",):
            views = ("top", "iso", "front", "right")  # "quad" and ["quad"] mean the same
        for v in views:
            if v not in VIEW_NAMES and camera is None:
                why = " (quad is the four views at once: give it alone)" if v == "quad" else ""
                raise ToolError(f"no view {v!r}{why}; the views are {', '.join(VIEW_NAMES)}, or quad alone")
        if not 1 <= len(views) <= 4:
            raise ToolError("views takes one to four names, or quad")
        size = tuple(int(s) for s in (size or self.size))
        if not all(64 <= s <= 2000 for s in size):
            raise ToolError("size must be [width, height], each 64 to 2000 pixels")
        f = self.facts(m)
        lines = []
        if facts:
            b = f["bounds"]
            lines.append(f"Model \"{f['model']}\" of {len(f['models'])} ({', '.join(f['models'])}): grid "
                         f"{f['grid'][0]} x {f['grid'][1]} x {f['grid'][2]}, {f['voxels']} voxels"
                         + ("" if b is None else f" in cells {b['low']}..{b['high']}") + ".")
            if f["colours"]:
                items = list(f["colours"].items())
                shown = items if detail == "full" else items[:SHORT_COLOURS]
                lines.append("Colours (number #hex: count, in cells low..high):")
                for c, n in shown:
                    w = f["where"][c]
                    lines.append(f"  {c} {w['hex']}: {n}, in {w['low']}..{w['high']}")
                if len(shown) < len(items):
                    lines.append(f"  and {len(items) - len(shown)} more colours (detail=full lists them)")
            lines.append(f"Current colour: {f['colour']}. Selected: {f['selected']} voxels.")
        data = {"facts": f}
        if heightmap:
            hm = self._heightmap(m)
            lines.append("Height map (top view; each number is the highest filled z of that column, "
                         "'.' an empty column; rows from y high to y low, columns x low to high):\n" + hm)
            data["heightmap"] = hm
        cut = None
        if cut_axis is not None:
            a = _axis(cut_axis)
            if cut_at is None or str(cut_keep) not in ("low", "high"):
                raise ToolError("a cut needs cut_at (a cell index) and cut_keep low or high")
            cut = (a, int(cut_at), str(cut_keep))
            lines.append(f"Cut along {AXES[a]} at {int(cut_at)}: only cells with {AXES[a]} "
                         + (f">= {int(cut_at)}" if cut_keep == "high" else f"< {int(cut_at)}") + " show.")
        box = None
        if (low is None) != (high is None):
            raise ToolError("a close-up needs both low and high, two corner cells [x, y, z]")
        if low is not None:
            from rok.app.object import model_origin

            o = np.asarray(model_origin(m), dtype=np.float64)
            a, b = np.asarray(_xyz(low, "low"), float), np.asarray(_xyz(high, "high"), float)
            box = (tuple(np.minimum(a, b) + o), tuple(np.maximum(a, b) + o + 1))
            if camera is None:
                lines.append(f"Close-up of cells {np.minimum(a, b).astype(int).tolist()}.."
                             f"{np.maximum(a, b).astype(int).tolist()}: what lies in front of the box is cut away.")
            else:
                lines.append(f"The camera frames cells {np.minimum(a, b).astype(int).tolist()}.."
                             f"{np.maximum(a, b).astype(int).tolist()}.")
        if camera_line:
            lines.append(camera_line)
        if text:
            t, tdata = self.text_views(m)
            lines.append(t)
            data["text_views"] = tdata
        if self.art is not None:
            status = self.blockout_status()
            if status:
                lines.append(status)
            lines.append(self.art_line())
        flat = (scale is not None or self.art is not None) and camera is None and cut is None and box is None \
            and views == ("top", "iso", "front", "right")
        if flat:
            image, line = self.flat_png(m, scale=None if scale is None else int(scale))
            lines.append(line)
            return Reply("\n".join(lines), data, image=image)
        picture = Picture(views, size, None, cut, None if camera is not None else box, camera=camera)
        if camera is not None and box is not None:
            picture.camera = dict(camera, frame=box)
        return Reply("\n".join(lines) or "Picture:", data, picture)

    @staticmethod
    def _heightmap(model) -> str:
        v = model.grid.voxels != 0
        sx, sy, sz = v.shape
        if sx * sy > 96 * 96:
            raise ToolError("the height map is for grids up to 96 x 96 columns; use rok_voxels_get for a box")
        any_z = v.any(axis=2)
        top = np.where(any_z, sz - 1 - np.argmax(v[:, :, ::-1], axis=2), -1)
        width = len(str(sz - 1))
        rows = []
        for y in range(sy - 1, -1, -1):
            rows.append(" ".join("." * width if top[x, y] < 0 else str(int(top[x, y])).rjust(width) for x in range(sx)))
        return "\n".join(rows)

    def tool_voxels_get(self, low=None, high=None, axis=None, index=None, model=None) -> Reply:
        """Voxels of a box as rows, or one slice as text."""
        m = self._model(model)
        v = m.grid.voxels
        if axis is not None:
            a = _axis(axis)
            if index is None or not 0 <= int(index) < v.shape[a]:
                raise ToolError(f"index must be 0 to {v.shape[a] - 1} for axis {AXES[a]}")
            layer = np.take(v, int(index), axis=a)
            u, w = [AXES[k] for k in range(3) if k != a]
            width = 3
            rows = [" ".join("." * width if c == 0 else str(int(c)).rjust(width) for c in layer[:, j])
                    for j in range(layer.shape[1] - 1, -1, -1)]
            text = (f"Slice {AXES[a]} = {int(index)}: columns are {u} 0..{layer.shape[0] - 1} (left to right), "
                    f"rows are {w} {layer.shape[1] - 1}..0 (top to bottom); numbers are colours, '.' empty.\n"
                    + "\n".join(rows))
            return Reply(text, {"slice": text})
        lo = np.asarray(_xyz(low if low is not None else (0, 0, 0), "low"))
        hi = np.asarray(_xyz(high if high is not None else np.asarray(v.shape) - 1, "high"))
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(v.shape) - 1)
        if np.any(hi < lo):
            return Reply("The box holds no cells of the grid.", {"rows": []})
        sub = v[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        cells = np.argwhere(sub != 0)
        more = len(cells) > MAX_GET_ROWS
        cells = cells[:MAX_GET_ROWS]
        rows = [[int(c[0] + lo[0]), int(c[1] + lo[1]), int(c[2] + lo[2]), int(sub[tuple(c)])] for c in cells]
        text = f"{len(rows)} filled cells in {lo.tolist()}..{hi.tolist()} as [x, y, z, colour]:\n{rows}"
        if more:
            text += f"\nMore cells follow: ask for a smaller box (at most {MAX_GET_ROWS} rows per call)."
        return Reply(text, {"rows": rows, "has_more": more})

    def tool_probe(self, x: float | None = None, y: float | None = None, points=None) -> Reply:
        """The cell under a pixel of the last picture; `points` probes many at once."""
        pts = [(x, y)] if points is None else [tuple(pt) for pt in points]
        if not pts or any(len(pt) != 2 or pt[0] is None or pt[1] is None for pt in pts):
            raise ToolError("give x and y, or points as a list of [x, y] pixels of the last picture")
        if len(pts) > 64:
            raise ToolError("at most 64 points per call")
        if not self.last_cameras:
            raise ToolError("no picture yet; look first (rok_look)")
        rows, lines = [], []
        for px, py in pts:
            hit = self._probe_one(float(px), float(py))
            rows.append(hit)
            lines.append(hit["text"])
        if len(rows) == 1:
            return Reply(rows[0]["text"], {k: v for k, v in rows[0].items() if k != "text"})
        return Reply("\n".join(lines), {"hits": [{k: v for k, v in r.items() if k != "text"} for r in rows]})

    def _probe_one(self, x: float, y: float) -> dict:
        from rok.app.object import model_origin
        from rok.core import raycast

        for name, tx, ty, tw, th, cam in self.last_cameras:
            if tx <= x < tx + tw and ty <= y < ty + th:
                break
        else:
            return {"x": x, "y": y, "hit": None, "text": f"({x:g}, {y:g}) is outside the last picture."}
        from rok.agent_art import FlatView

        if isinstance(cam, FlatView):
            found = cam.cell(x - tx, y - ty)
            if found is None:
                return {"x": x, "y": y, "view": name, "hit": None, "text": f"({x:g}, {y:g}): nothing, in the {name} view."}
            c, colour = found
            face = FlatView.FACE[name]
            text = (f"({x:g}, {y:g}): cell {c} of model \"{self._name(cam.model)}\", colour {colour}, its {face} face, "
                    f"in the {name} view.")
            return {"x": x, "y": y, "view": name, "cell": c, "model": self._name(cam.model), "colour": colour,
                    "face": face, "text": text}
        origin, direction = cam.ray(x - tx, y - ty)
        best = None
        scene = self.doc.scene
        for m in scene.models:
            if scene.is_hidden(m):
                continue
            o = np.asarray(model_origin(m), dtype=np.float64)
            hit = raycast.raycast(m.grid.voxels, origin - o, direction, walls=False)
            if hit is not None and not hit.wall and (best is None or hit.t < best[0]):
                best = (hit.t, m, hit)
        if best is None:
            return {"x": x, "y": y, "view": name, "hit": None, "text": f"({x:g}, {y:g}): nothing, in the {name} view."}
        _t, m, hit = best
        c = [int(v) for v in hit.voxel]
        colour = int(m.grid.voxels[tuple(c)])
        face = {(0, 0, 1): "top", (0, 0, -1): "bottom", (1, 0, 0): "+x", (-1, 0, 0): "-x",
                (0, 1, 0): "+y", (0, -1, 0): "-y"}.get(tuple(int(n) for n in hit.normal), "?")
        text = f"({x:g}, {y:g}): cell {c} of model \"{self._name(m)}\", colour {colour}, its {face} face, in the {name} view."
        return {"x": x, "y": y, "view": name, "cell": c, "model": self._name(m), "colour": colour, "face": face, "text": text}

    def tool_check(self, model=None) -> Reply:
        """Separate parts, enclosed holes, and mirror symmetry."""
        m = self._model(model)
        v = m.grid.voxels
        mask = v != 0
        lines = []
        data: dict = {}
        if not mask.any():
            return Reply("The model is empty.", {"empty": True})
        parts = region.components(mask)
        data["parts"] = [{"voxels": len(p), "low": p.min(axis=0).tolist(), "high": p.max(axis=0).tolist()} for p in parts[:20]]
        if len(parts) == 1:
            lines.append("One connected part (face to face).")
        else:
            lines.append(f"{len(parts)} separate parts (face to face):")
            for p in data["parts"][:10]:
                lines.append(f"  {p['voxels']} voxels in {p['low']}..{p['high']}")
            if len(parts) > 10:
                lines.append(f"  and {len(parts) - 10} more, {sum(len(p) for p in parts[10:])} voxels in all "
                             "(rok_edit do=delete_loose keeps only the largest part)")
            groups = _touching_groups(mask, parts)
            data["parts_touching"] = groups
            if groups < len(parts):
                lines.append(f"Counting edges and corners as joins too, they are {groups} part(s): "
                             f"{len(parts) - groups} of the parts above touch the rest only at an edge or a corner "
                             "(a diagonal line does that; rok_shape line with solid=true joins its cells face to "
                             "face).")
        # Holes: empty cells the outside cannot reach.
        outside = region.components(np.pad(~mask, 1, constant_values=True))
        reach = np.zeros(np.asarray(mask.shape) + 2, dtype=bool)
        for part in outside:
            if (part == 0).any():
                reach[tuple(part.T)] = True
        holes = (~mask) & ~reach[1:-1, 1:-1, 1:-1]
        n_holes = int(holes.sum())
        data["enclosed_empty"] = n_holes
        lines.append("No enclosed empty cells." if n_holes == 0 else f"{n_holes} empty cells are enclosed (hidden inside).")
        sym = {}
        lo, hi = (np.argwhere(mask).min(axis=0), np.argwhere(mask).max(axis=0) + 1)
        sub = v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        for a in range(3):
            flipped = np.flip(sub, axis=a)
            filled = (sub != 0) | (flipped != 0)
            same = ((sub != 0) == (flipped != 0)) & filled
            sym[AXES[a]] = round(100.0 * same.sum() / max(filled.sum(), 1), 1)
        data["symmetry_percent"] = sym
        lines.append("Mirror symmetry of the shape about the middle of its voxels: "
                     + ", ".join(f"{a} {p}%" for a, p in sym.items()) + ".")
        if len(parts) > 1:
            names = self.colour_names()
            for p in craft.parts(v, limit=6)[1:]:
                made = ", ".join(f"{names.get(c, 'colour ' + str(c))} ({n})" for c, n in p["colours"].items())
                lines.append(f"  the part in {p['low']}..{p['high']} is made of {made}")
        if int(mask.sum()) <= CRAFT_MAX_VOXELS:
            rep = craft.check(v, with_parts=False)
            for j in rep.jags[:8]:
                lines.append(f"Jagged slope on the {j.edge} edge of the {j.side} view (view columns or rows "
                             f"{j.cells[0]}..{j.cells[1]} of the filled box): flat runs {j.runs}; an even rhythm "
                             "(1,1,1 or 2,2,2 or 1,2,1,2) reads as a clean line.")
            for t in rep.thin[:8]:
                lines.append(f"Thin line on the {t['side']} view: {t['cells']} cells one cell wide, view rows and "
                             f"columns {t['low']}..{t['high']} of the filled box; it can vanish at icon size.")
            lines.append("Shown small, the silhouettes keep: " + "; ".join(
                f"{side} {r['half']}% at 1/2, {r['quarter']}% at 1/4" for side, r in rep.survives.items()) + ".")
            data["jagged"] = [{"side": j.side, "edge": j.edge, "runs": j.runs, "cells": list(j.cells)} for j in rep.jags]
            data["thin"] = rep.thin
            data["survives"] = rep.survives
        kind, why = verdict([mm.grid.voxels for mm in self.doc.scene.models])
        lines.append(f"Verdict: {kind}" + (f" ({'; '.join(why)})" if why else "") + ".")
        data["verdict"] = kind
        if self.art is not None:
            status = self.blockout_status()
            if status:
                lines.append(status)
        return Reply("\n".join(lines), data)

    # --- editing -------------------------------------------------------------------------

    _locked = 0

    def _locked_note(self) -> str:
        n = self._locked
        return f" {n} masked cell{'s' if n != 1 else ''} stayed as they are." if n else ""

    def _write(self, cells: np.ndarray, values: np.ndarray, model, limit: int = MAX_SET_CELLS) -> tuple[int, int]:
        """Write cells into `model` as one undo step. Returns (written, outside)."""
        from rok.app import object as voxel_object
        from rok.app.ops_common import commit
        from rok.core.undo import ModelRecord, StrokeRecorder

        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        values = np.asarray(values, dtype=np.uint8).reshape(-1)
        if len(cells) > limit:
            raise ToolError(f"{len(cells)} cells is more than {limit} in one call; split it into parts "
                            "(several calls in one rok_batch are still one undo step)")
        size = np.asarray(m_size := model.grid.size)
        inside = np.all((cells >= 0) & (cells < size), axis=1)
        outside = int((~inside).sum())
        cells, values = cells[inside], values[inside]
        if len(cells) == 0:
            if outside:
                raise ToolError(f"every cell is outside the grid, which is {m_size[0]} x {m_size[1]} x {m_size[2]} "
                                f"(cells 0..{m_size[0] - 1}, 0..{m_size[1] - 1}, 0..{m_size[2] - 1}); "
                                "resize with rok_edit do=resize")
            return 0, 0
        rec = StrokeRecorder(model.grid)
        model.grid.locked_writes = 0
        n = rec.set_many(cells, values, ignore_cut=True)
        self._locked = model.grid.locked_writes  # masked cells take no write
        model.grid.locked_writes = 0
        record = rec.finish()
        if record is not None:
            commit(ModelRecord(model.uid, record), serial=True)
        voxel_object.flush_dirty()
        self.app.after_event()
        return int(n), outside

    def tool_voxels_set(self, voxels=None, layers=None, corner=None, legend=None, model=None, mirror=None,
                        mode: str = "set") -> Reply:
        """The raw tool: rows [x, y, z, colour], or ASCII layers with a legend."""
        m = self._model(model)
        if (voxels is None) == (layers is None):
            raise ToolError("give either voxels or layers, not both and not neither: "
                            '{"voxels": [[0, 0, 0, 12], [1, 0, 0, 12]]} or '
                            '{"layers": [["##", "##"]], "corner": [0, 0, 0], "legend": {"#": 12}}. '
                            'In rok_batch the step is {"tool": "rok_voxels_set", "args": {"voxels": [...]}}')
        if voxels is not None:
            try:
                arr = np.asarray(voxels, dtype=np.int64).reshape(-1, 4)
            except (TypeError, ValueError):
                raise ToolError("voxels must be a list of [x, y, z, colour] rows of whole numbers") from None
            cells, values = arr[:, :3], arr[:, 3]
        else:
            cells, values = self._layers(layers, corner, legend)
        if len(values) and (values.min() < 0 or values.max() > 255):
            raise ToolError("colours must be 0 (empty) to 255")
        mode = str(mode).lower()
        if mode not in ("set", "fill_empty", "paint"):
            raise ToolError("mode must be set (write every cell), fill_empty (only empty cells) or paint "
                            "(only filled cells)")
        limit = art_set_cells(m.grid.size)
        if self.art is not None and len(cells) > limit:
            raise ToolError(f"{len(cells)} cells in one call; a voxel art scene of this grid places at most {limit} per "
                            "rok_voxels_set call. Place by hand in small groups, or use rok_shape (a box, a "
                            "line, an extrude of a silhouette) for a large form.")
        if mirror is not None:
            cells, values = voxel_draw.mirrored(cells, values, _axis(mirror), m.grid.size)
        skipped = 0
        no_colour = 0  # paint: filled cells that were given colour 0
        if mode != "set":
            v = m.grid.voxels
            inside = np.all((cells >= 0) & (cells < np.asarray(v.shape)), axis=1)
            keep = np.ones(len(cells), dtype=bool)
            filled = v[tuple(cells[inside].T)] != 0
            keep[np.flatnonzero(inside)] = filled if mode == "paint" else ~filled
            skipped = int((inside & ~keep).sum())
            if mode == "paint":
                no_colour = int((inside & keep & (values == 0)).sum())
                keep &= values != 0
            cells, values = cells[keep | ~inside], values[keep | ~inside]
        out = {}

        def write():
            n, outside = self._write(cells, values, m) if len(cells) else (0, 0)
            out["n"], out["outside"] = n, outside
            note = f"Wrote {len(cells) - outside} cells ({n} changed)."
            if skipped:
                note += (f" {skipped} {'filled' if mode == 'fill_empty' else 'empty'} cells were left as they are "
                         f"({mode}).")
            if no_colour:
                note += f" {no_colour} filled cells were given colour 0 and kept their colour (paint does not erase)."
            if outside:
                note += f" {outside} cells fell outside the grid and were skipped."
            return note + self._locked_note()

        return self._edit(write)

    @staticmethod
    def _layers(layers, corner, legend) -> tuple[np.ndarray, np.ndarray]:
        """ASCII art: layers[k][r][c] is the cell at x = corner.x + c,
        y = corner.y + (rows - 1 - r), z = corner.z + k. '.' or ' ' is
        empty-and-unchanged, '0' empties, other characters map through legend."""
        cx, cy, cz = _xyz(corner if corner is not None else (0, 0, 0), "corner")
        legend = {str(k): _colour(v) for k, v in (legend or {}).items()}
        legend.setdefault("0", 0)
        cells, values = [], []
        if not isinstance(layers, (list, tuple)):
            raise ToolError("layers must be a list (one per z, bottom first) of lists of strings (rows, top row first)")
        for k, layer in enumerate(layers):
            if isinstance(layer, str):
                layer = layer.splitlines()
            rows = len(layer)
            for r, row in enumerate(layer):
                for c, ch in enumerate(str(row)):
                    if ch in ". ":
                        continue
                    if ch not in legend:
                        raise ToolError(f"character {ch!r} (layer {k}, row {r}) is not in the legend {sorted(legend)}")
                    cells.append((cx + c, cy + rows - 1 - r, cz + k))
                    values.append(legend[ch])
        return np.asarray(cells, dtype=np.int64).reshape(-1, 3), np.asarray(values, dtype=np.int64)

    def tool_shape(self, kind: str, low=None, high=None, colour=None, mode: str = "add", axis: str = "z",
                   hollow: bool = False, cell=None, model=None, start=None, end=None, along=None,
                   solid: bool = False, frame_colour=None, silhouette=None, legend=None, depth: int = 1,
                   points=None, closed: bool = False, mirror=None, size=None, turn=None,
                   turn_axis: str = "x") -> Reply:
        """Box, sphere, cylinder, cone and the primitives from two corners; a
        line from start to end (or low to high, in the order given); an extrude of
        a silhouette; a curve through points, as thick as the brush `size`; or fill.
        `turn` tilts the shape by degrees about `turn_axis` before it is placed;
        `mirror` adds the mirror image."""
        m = self._model(model)
        kind = str(kind).lower()
        if kind not in SHAPE_KINDS:
            raise ToolError(f"kind must be one of {', '.join(SHAPE_KINDS)}; got {kind!r}")
        mode = str(mode).lower()
        if mode not in ("add", "fill_empty", "erase", "paint"):
            raise ToolError("mode must be add, fill_empty, erase or paint")
        c = _colour(colour if colour is not None else self.doc.colour)
        if size is not None and kind != "curve":
            raise ToolError(f"size is the brush size of a curve; a {kind} takes its size from low and high")
        if turn is not None and kind == "fill":
            raise ToolError("turn tilts a shape before it is placed; a fill recolours what is there")
        tilt = None if turn is None else (_axis(turn_axis), float(turn))
        if kind in ("extrude", "curve"):
            return self._drawn(m, kind, c, mode, low, axis, silhouette, legend, depth, points, closed, mirror, solid,
                               size, tilt)
        if kind == "fill":
            if cell is None:
                raise ToolError("fill needs cell [x, y, z]: a filled cell; its connected blob of one colour is recoloured")
            seed = _xyz(cell)
            v = m.grid.voxels
            if not all(0 <= seed[i] < v.shape[i] for i in range(3)) or v[seed] == 0:
                raise ToolError(f"cell {list(seed)} is empty or outside the grid; fill recolours a filled cell's "
                                "connected same-colour blob")
            cells = np.argwhere(region.connected_colour(v, seed))
            return self._edit(lambda: self._write(cells, np.full(len(cells), c), m, MAX_SHAPE_CELLS)
                              and f"Filled {len(cells)} cells with colour {c}.")
        if (start is not None or end is not None) and kind != "line":
            raise ToolError(f"start and end are for a line; a {kind} takes low and high, two corners of its box")
        if kind == "line" and (start is not None or end is not None):
            if start is None or end is None or low is not None or high is not None:
                raise ToolError("a line takes start and end, both cells [x, y, z], and no low or high")
            low, high = start, end
        if low is None or high is None:
            raise ToolError(f"{kind} needs low and high, two corner cells [x, y, z] (both inside the shape)")
        a, b = np.asarray(_xyz(low, "low")), np.asarray(_xyz(high, "high"))
        run = None
        if along is not None:
            if kind not in RUN_KINDS:
                raise ToolError(f"along is for {', '.join(RUN_KINDS)}: the way it runs; a {kind} has no run")
            run = _axis(along)
            if run == _axis(axis):
                raise ToolError(f"along must lie across axis {AXES[_axis(axis)]} (the up axis): pick one of "
                                f"{', '.join(AXES[k] for k in range(3) if k != _axis(axis))}")
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        if np.prod(hi - lo + 1) > MAX_SHAPE_CELLS:
            raise ToolError(f"the box {lo.tolist()}..{hi.tolist()} has {int(np.prod(hi - lo + 1))} cells, more than "
                            f"{MAX_SHAPE_CELLS} (256 x 256 x 64) in one shape; split it into several shapes in one "
                            "rok_batch (one undo step)")
        if frame_colour is not None and kind != "framed_buckyball":
            raise ToolError("frame_colour is for a framed_buckyball: the colour of its bars")
        bars = None
        if kind == "line":
            cells = np.asarray(line3d_cells(tuple(int(t) for t in a), tuple(int(t) for t in b)), dtype=np.int64)
            if solid:
                cells = _face_joined(cells)
        elif kind == "framed_buckyball":
            cells, bars = self._primitive(kind, lo, hi, _axis(axis), hollow, run, parts=True)
        else:
            cells = self._solid(kind, lo, hi, _axis(axis), hollow, run)
        value = 0 if mode == "erase" else c
        values = np.full(len(cells), value)
        if bars is not None and mode != "erase":
            from rok.core.palette import darker_slot

            fc = _colour(frame_colour) if frame_colour is not None else darker_slot(self.doc.scene.palette, c)
            values[bars] = fc
        if tilt is not None:
            cells, values = _turned(cells, values, *tilt)
        if mirror is not None:
            cells, values = voxel_draw.mirrored(cells, values, _axis(mirror), m.grid.size)
        in_grid = int(np.all((cells >= 0) & (cells < np.asarray(m.grid.size)), axis=1).sum())
        left = 0
        if mode in ("paint", "fill_empty"):
            v = m.grid.voxels
            inside = np.all((cells >= 0) & (cells < np.asarray(v.shape)), axis=1)
            cells, values = cells[inside], values[inside]
            filled = v[tuple(cells.T)] != 0
            pick = filled if mode == "paint" else ~filled
            left = int((~pick).sum())
            cells, values = cells[pick], values[pick]

        def write():
            if in_grid == 0:
                gx, gy, gz = m.grid.size
                raise ToolError(f"the whole {kind} lies outside the grid ({gx} x {gy} x {gz}, cells 0..{gx - 1}, "
                                f"0..{gy - 1}, 0..{gz - 1}); check its coordinates, or resize with rok_edit do=resize")
            n, outside = self._write(cells, values, m, MAX_SHAPE_CELLS) if len(cells) else (0, 0)
            where = (f"from {a.tolist()} to {b.tolist()}" if kind == "line"
                     else f"in {lo.tolist()}..{hi.tolist()}")
            if mode == "fill_empty":
                note = f"Filled {len(cells) - outside} empty cells of a {kind} {where}"
            elif mode == "paint":
                note = f"Painted {len(cells) - outside} filled cells of a {kind} {where}"
            else:
                verb = {"add": "Added", "erase": "Erased"}[mode]
                note = f"{verb} a {kind} of {len(cells) - outside} cells {where}"
            if tilt is not None:
                note += f", tilted {tilt[1]:g} degrees about {AXES[tilt[0]]}"
            note += "." if mode == "erase" else f" with colour {value}."
            if bars is not None and mode != "erase":
                note += f" Its bars have colour {fc}."
            if left:
                note += (f" {left} of its cells were filled already and stay as they are." if mode == "fill_empty"
                         else f" {left} of its cells were empty and stay empty.")
            if outside:
                note += f" {outside} cells fell outside the grid."
            return note + self._locked_note()

        return self._edit(write)

    def _drawn(self, m, kind, c, mode, low, axis, silhouette, legend, depth, points, closed, mirror,
               solid=False, size=None, tilt=None) -> Reply:
        """An extrude of a silhouette, or a curve through points."""
        if kind == "extrude":
            if not isinstance(silhouette, (list, tuple)) or not silhouette or low is None:
                raise ToolError('an extrude needs silhouette (rows of characters, top row first, as rok_look '
                                'text shows a view), low (the low corner it starts from) and axis (the way it is '
                                'pushed: y for a front view, x for a right view, z for a top view); legend maps '
                                'characters to colours, {"#": 1}; depth is how many cells deep')
            lg = {str(k): _colour(v) for k, v in (legend or {"#": c}).items()}
            try:
                cells, values = voxel_draw.extrude([str(r) for r in silhouette], lg, _axis(axis), _xyz(low, "low"),
                                                   int(depth))
            except KeyError as exc:
                raise ToolError(f"character {exc.args[0]!r} of the silhouette is not in the legend {sorted(lg)}") from None
            except ValueError as exc:
                raise ToolError(str(exc)) from None
            where = f"from {list(_xyz(low))}, {int(depth)} deep along {AXES[_axis(axis)]}"
        else:
            if not isinstance(points, (list, tuple)) or len(points) < 2:
                raise ToolError("a curve needs points: two or more cells [x, y, z] it passes through, in order")
            pts = [_xyz(p, "point") for p in points]
            sizes = _brush_sizes(size, len(pts))
            if max(sizes) > 1:
                # the centre line joined face to face: where the brush thins to one
                # voxel, its cells would touch only at their edges and come off
                line = _face_joined(voxel_draw.curve(pts, closed=bool(closed)))
                cells = np.unique(np.concatenate([voxel_draw.brush_curve(pts, sizes, closed=bool(closed)), line]),
                                  axis=0)
            else:
                cells = voxel_draw.curve(pts, closed=bool(closed))
                if solid:
                    cells = _face_joined(cells)
            values = np.full(len(cells), c)
            where = f"through {len(points)} points"
            if max(sizes) > 1:
                where += (f" with brush size {sizes[0]}" if len(set(sizes)) == 1
                          else f" with brush sizes {sizes[0]} to {sizes[-1]}")
        if mode == "erase":
            values = np.zeros(len(cells), dtype=np.int64)
        if tilt is not None:
            cells, values = _turned(cells, values, *tilt)
            where += f", tilted {tilt[1]:g} degrees about {AXES[tilt[0]]}"
        if mirror is not None:
            cells, values = voxel_draw.mirrored(cells, values, _axis(mirror), m.grid.size)
        if mode in ("paint", "fill_empty"):
            v = m.grid.voxels
            inside = np.all((cells >= 0) & (cells < np.asarray(v.shape)), axis=1)
            cells, values = cells[inside], values[inside]
            filled = v[tuple(cells.T)] != 0
            pick = filled if mode == "paint" else ~filled
            cells, values = cells[pick], values[pick]

        def write():
            n, outside = self._write(cells, values, m, MAX_SHAPE_CELLS)
            note = f"Drew {'a curve' if kind == 'curve' else 'an extrude'} of {len(cells) - outside} cells {where}"
            note += f", mirrored across {AXES[_axis(mirror)]}." if mirror is not None else "."
            if outside:
                note += f" {outside} cells fell outside the grid."
            return note + self._locked_note()

        return self._edit(write)

    @staticmethod
    def _solid(kind: str, lo, hi, axis: int, hollow: bool, along: int | None = None) -> np.ndarray:
        g = np.indices(tuple(int(v) for v in hi - lo + 1)).reshape(3, -1).T.astype(np.float64)
        size = (hi - lo + 1).astype(np.float64)
        centre = (size - 1) / 2.0
        r = np.maximum(size / 2.0, 0.5)
        q = (g - centre) / r  # -1..1 across the box, measured at cell centres
        if kind == "box":
            inside = np.ones(len(g), dtype=bool)
        elif kind == "sphere":
            inside = (q ** 2).sum(axis=1) <= 1.0 + 1e-9
        elif kind in ("cylinder", "cone"):
            u, w = [k for k in range(3) if k != axis]
            rad = q[:, u] ** 2 + q[:, w] ** 2
            if kind == "cylinder":
                inside = rad <= 1.0 + 1e-9
            else:
                t = g[:, axis] / max(size[axis] - 1, 1)  # 0 at low (the base), 1 at high (the tip)
                inside = rad <= (1.0 - t) ** 2 + 1e-9
        else:
            return Agent._primitive(kind, lo, hi, axis, hollow, along)
        cells = g[inside].astype(np.int64)
        if hollow and len(cells):
            from rok.core import morph

            m = np.zeros(tuple(int(v) for v in hi - lo + 1), dtype=bool)
            m[tuple(cells.T)] = True
            cells = np.argwhere(m & ~morph.erode(m, 1))
        return cells + lo

    @staticmethod
    def _primitive(kind: str, lo, hi, axis: int, hollow: bool, along: int | None = None, parts: bool = False):
        """One of the Primitives menu's shapes, filling the box lo..hi and
        standing on its low side along `axis` (a roof's ridge, a cone's tip and
        stairs rise toward high). Cells may lie outside the grid; the write
        skips them. With `parts`, (cells, which are bars of a frame)."""
        from rok.core import shape as shape_core

        u, v = [k for k in range(3) if k != axis]
        us, vs = np.arange(lo[u], hi[u] + 1), np.arange(lo[v], hi[v] + 1)
        gu, gv = np.meshgrid(us, vs, indexing="ij")
        footprint = np.zeros((gu.size, 3), dtype=np.int64)
        footprint[:, u], footprint[:, v] = gu.ravel(), gv.ravel()
        footprint[:, axis] = lo[axis]
        height = int(hi[axis] - lo[axis] + 1)
        big = np.asarray(hi, dtype=np.int64) + 1  # no clipping here: the write counts what falls outside
        cells, bars = shape_core.solid_parts(footprint - np.minimum(lo, 0), axis, 1, height, kind.upper(),
                                             big - np.minimum(lo, 0), wall=1 if hollow else 0, seed=7, along=along)
        cells = cells + np.minimum(lo, 0)
        return (cells, bars) if parts else cells

    def _set_selection(self, model, mask) -> None:
        from rok.app.ops_common import commit
        from rok.core.undo import selection_record

        before = model.selection.mask.copy()
        model.selection.replace(mask)
        commit(selection_record(model.uid, before, model.selection.mask), modified=False)
        self.app.after_event()

    def tool_select(self, do: str, low=None, high=None, colour=None, cell=None, count: int = 1, model=None) -> Reply:
        m = self._model(model)
        do = str(do).lower()
        if do in ("mask", "unmask"):
            return self._mask(m, do)
        v = m.grid.voxels
        if do == "all":
            mask = v != 0
        elif do == "none":
            mask = np.zeros(v.shape, dtype=bool)
        elif do == "box":
            if low is None or high is None:
                raise ToolError("box needs low and high, two corner cells [x, y, z]")
            a, b = np.asarray(_xyz(low, "low")), np.asarray(_xyz(high, "high"))
            lo, hi = np.maximum(np.minimum(a, b), 0), np.minimum(np.maximum(a, b), np.asarray(v.shape) - 1)
            mask = np.zeros(v.shape, dtype=bool)
            mask[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
            mask &= v != 0
        elif do == "colour":
            mask = v == _colour(colour)
        elif do == "connected":
            seed = _xyz(cell)
            mask = region.connected_filled(v, seed)
        elif do == "invert":
            mask = (v != 0) & ~m.selection.mask
        elif do in ("grow", "shrink"):
            from rok.core import morph

            fn = morph.dilate if do == "grow" else morph.erode
            mask = fn(m.selection.mask, int(count)) & (v != 0)
        else:
            raise ToolError("do must be all, none, box, colour, connected, invert, grow, shrink, mask or unmask")
        self._set_selection(m, mask)
        n = int(m.selection.count())
        return Reply(f"Selected {n} voxels.", {"selected": n}, None)

    def _mask(self, m, do: str) -> Reply:
        """Mask the selection (it clears the selection), or lift the mask."""
        if do == "mask":
            if m.selection.is_empty():
                raise ToolError("mask protects the selected voxels: select them first")
            self.app_run("rok.mask_selection")
        else:
            self.app_run("rok.mask_clear")
        lock = m.grid.lock
        n = 0 if lock is None else int(np.count_nonzero(lock))
        if do == "mask":
            text = f"Masked {n} voxels: every command leaves them as they are. do=unmask lifts the mask."
        else:
            text = "The mask is gone: every voxel can change again."
        return Reply(text, {"masked": n}, None)

    def tool_transform(self, do: str, target: str = "selection", offset=None, axis: str = "z", degrees: float = 90.0,
                       from_positive: bool = True, model=None) -> Reply:
        m = self._model(model)
        do, target = str(do).lower(), str(target).lower()
        if target not in ("selection", "model"):
            raise ToolError("target must be selection or model")
        if target == "selection" and m.selection.is_empty() and do != "mirror":
            raise ToolError("nothing is selected; select first (rok_select) or use target=model")
        a = _axis(axis)
        T = "SELECTION" if target == "selection" else "MODEL"

        def run():
            if do == "move":
                off = _xyz(offset if offset is not None else (0, 0, 0), "offset")
                if target == "model":
                    self.app_run("rok.model_move", offset=off)
                else:
                    self.app_run("rok.transform", target=T, kind="MOVE", offset=off)
                return f"Moved the {target} by {list(off)}."
            if do == "flip":
                self.app_run("rok.transform", target=T, kind="FLIP", axis=a)
                return f"Flipped the {target} along {AXES[a]}."
            if do == "mirror":
                self.app_run("rok.mirror_half", axis=AXES[a].upper(), from_positive=bool(from_positive))
                side = "+" if from_positive else "-"
                return f"Mirrored the {side}{AXES[a]} half of the model onto the other half."
            if do == "rotate":
                return self._rotate(m, target, a, float(degrees))
            raise ToolError("do must be move, rotate, flip or mirror")

        return self._edit(run)

    def tool_deform(self, offsets=None, counts=3, bernstein: bool = False, model=None) -> Reply:
        """Bend the selection (or the whole model) with a cage of control points."""
        from rok.app import object as voxel_object
        from rok.app.ops_common import commit
        from rok.core import lattice
        from rok.core.undo import ModelRecord, StrokeRecorder, selection_record, stacked_record

        m = self._model(model)
        v = m.grid.voxels
        whole = m.selection.is_empty()
        mask = (v != 0) if whole else m.selection.mask.copy()
        cells = np.argwhere(mask)
        if not len(cells):
            raise ToolError("nothing to bend: the model is empty")
        if not isinstance(offsets, dict) or not offsets:
            raise ToolError('offsets must be {"i,j,k": [dx, dy, dz], ...}: how far each cage point moves, in cells')
        try:
            counts_ = lattice._counts(counts if not isinstance(counts, list) else tuple(counts))
            rest = lattice.rest_lattice((cells.min(axis=0), cells.max(axis=0)), counts_, bool(bernstein))
            bent = lattice.offset_lattice(rest, offsets)
        except (ValueError, TypeError) as exc:
            raise ToolError(f"{exc}. Points are named i,j,k with i in 0..nx-1; counts is 2 to 6 per axis, one number or [nx, ny, nz].") from exc

        def run():
            before_mask = m.selection.mask.copy()
            rec = StrokeRecorder(m.grid)
            want, new_mask = lattice.live_deform_grid(v.copy(), mask, bent)
            changed = int(rec.set_grid(want))
            if not whole:
                m.selection.replace(new_mask)
            record = rec.finish()
            commit(stacked_record(None if record is None else ModelRecord(m.uid, record),
                                  selection_record(m.uid, before_mask, m.selection.mask)), serial=True)
            voxel_object.flush_dirty()
            self.app.after_event()
            where = "the whole model" if whole else "the selection"
            soft = ", soft" if bernstein else ""
            return (f"Bent {where} with a {counts_[0]} x {counts_[1]} x {counts_[2]}{soft} cage ({len(offsets)} points moved, "
                    f"{changed} voxels changed); cells outside the grid are lost.")

        return self._edit(run)

    def _rotate(self, m, target: str, a: int, degrees: float) -> str:
        from rok.core import free_rotate

        d = free_rotate.snap(degrees)
        q = free_rotate.quarter_turns(d)
        if d == 0.0:
            raise ToolError("degrees must not be 0")
        if q is not None:
            self.app_run("rok.transform", target="SELECTION" if target == "selection" else "MODEL", kind="ROTATE", axis=a, turns=q)
            return f"Turned the {target} {d}° about {AXES[a]} (exact quarter turns)."
        if target == "model":
            from rok.app import scene_tools

            rotation = scene_tools._Rotation([m], a)
            if not rotation.apply(d):
                raise ToolError(rotation.error)
            rotation.commit()
            self.app.after_event()
            return f"Turned the model {d}° about {AXES[a]}; its grid grew to fit."
        from rok.app import gizmo_ops
        from rok.app.ops_common import commit
        from rok.core import gizmo
        from rok.core.undo import ModelRecord, StrokeRecorder, selection_record, stacked_record

        pivot2 = gizmo_ops.selection_pivot2(m)
        before_mask = m.selection.mask.copy()
        rec = StrokeRecorder(m.grid)
        new = gizmo.free_rotate_cells(m.grid.voxels, m.selection, a, d, pivot2, rec.set_many)
        m.selection.replace(new)
        record = rec.finish()
        commit(stacked_record(None if record is None else ModelRecord(m.uid, record),
                              selection_record(m.uid, before_mask, m.selection.mask)), serial=True)
        from rok.app import object as voxel_object

        voxel_object.flush_dirty()
        self.app.after_event()
        return f"Turned the selection {d}° about {AXES[a]} around its centre; cells outside the grid are lost."

    def tool_edit(self, do: str, wall: int = 1, count: int = 1, size=None, axis: str = "z", angle: int = 90,
                  shader: str = "", args=None, index: int | None = None, hex: str | None = None,
                  glow: float | None = None, power: int | None = None,
                  up: bool = True, colour: int | None = None, colour_count: int | None = None,
                  iterations: int = 1, grow: bool = False, colours=None, scope: str | None = None,
                  only_on=None, only_colours=None, room: int = 0, feather: int = 0, volume=None,
                  volume_shape: str = "box", model=None, ramps=None, ramp: str | None = None) -> Reply:
        m = self._model(model)
        do = str(do).lower()
        if do == "palette":
            return Reply(self.palette(colours, ramps))
        if do == "shade":
            return self._shade(m, ramp, colour, only_colours)
        if do == "shaders":
            return Reply(self._shader_list(shader))
        if do == "current_colour":
            return Reply(self._current_colour(index))
        simple = {"smooth": "rok.smooth", "bevel": "rok.bevel", "fit_grid": "rok.fit_grid",
                  "inverse": "rok.inverse", "drop_to_floor": "rok.drop_to_floor", "drop_voxels": "rok.drop_voxels",
                  "hollow_inside": "rok.fill_volume"}

        where = self._shader_scope(scope, only_on, only_colours, room, feather, volume, volume_shape)

        def run():
            if do in simple:
                self.app_run(simple[do])
            elif do == "delete_loose":
                v = m.grid.voxels
                parts = region.components(v != 0)
                if len(parts) < 2:
                    return "The model is one part already; nothing was deleted."
                loose = np.concatenate(parts[1:])
                self._write(loose, np.zeros(len(loose)), m, MAX_SHAPE_CELLS)
                return (f"Deleted {len(parts) - 1} loose part{'s' if len(parts) > 2 else ''} ({len(loose)} voxels); "
                        f"the largest part ({len(parts[0])} voxels) stays.")
            elif do == "clear":
                # Like every edit, clear acts on the selection where one exists; only
                # without one does it empty the whole model.
                sel = m.selection
                selected = 0 if sel is None or sel.is_empty() else int(((m.grid.voxels != 0) & sel.mask).sum())
                if selected:
                    self.app_run("rok.selection_delete")
                    return f"Deleted the {selected} selected voxels; the rest of the model stays."
                filled = int((m.grid.voxels != 0).sum())
                self.app_run("rok.clear_all")
                return f"Emptied the whole model ({filled} voxels): nothing was selected."
            elif do == "hollow":
                self.app_run("rok.hollow", wall=int(wall))
            elif do in ("grow", "shrink"):
                self.app_run(f"rok.{do}", count=int(count), dialog=False)
            elif do == "scale":
                self.app_run("rok.scale", mode="UP" if up else "DOWN")
            elif do == "resize":
                was = _off_centre(m)
                self.app_run("rok.resize", size=_xyz(size, "size"))
                return f"Resized the grid to {' x '.join(map(str, m.grid.size))}." + _resize_note(m, was)
            elif do in ("lathe",):
                self.app_run("rok.lathe", axis=AXES[_axis(axis)].upper())
            elif do == "twist":
                self.app_run("rok.twist", angle=int(angle), axis=AXES[_axis(axis)].upper())
            elif do == "shader":
                named = set(self.colour_names())
                if grow:
                    note = self._grow_shader(shader, args, colour, colour_count, iterations, colours, where,
                                             only_on, only_colours)
                else:
                    note = self._run_shader(shader, args, colour, colour_count, iterations, colours, where)
                return note + self._name_shader_colours(shader, named)
            elif do == "set_colour":
                return self._set_colour(index, hex, glow, power)
            else:
                raise ToolError("do must be hollow, smooth, bevel, grow, shrink, scale, resize, fit_grid, clear, delete_loose, "
                                "inverse, drop_to_floor, drop_voxels, lathe, twist, shader, shaders, set_colour, "
                                "palette, shade or current_colour")
            return f"Ran {do}."

        return self._edit(run)

    def _shade(self, m, ramp, colour, only_colours) -> Reply:
        """Shade the visible faces from a named ramp: tops light, sides middle, undersides dark."""
        names = self.colour_names()
        if not ramp:
            raise ToolError('shade needs ramp, the name of a ramp from do=palette ({"ramp": "wood"})')
        steps = sorted(((n, s) for s, n in names.items() if n.startswith(f"{ramp} ") and n[len(ramp) + 1:].isdigit()),
                       key=lambda t: int(t[0][len(ramp) + 1:]))
        if len(steps) < 2:
            raise ToolError(f"no ramp named {ramp!r} with two or more steps; do=palette with ramps makes one")
        slots = [s for _n, s in steps]
        v = m.grid.voxels
        if colour is not None:
            target = v == _colour(colour)
        elif only_colours:
            target = np.isin(v, [_colour(k) for k in only_colours])
        else:
            target = np.isin(v, slots)
        sel = m.selection
        selected = sel is not None and not sel.is_empty()
        if selected:
            target = target & sel.mask  # like every edit, it acts on the selection where one exists
        if not target.any():
            raise ToolError("no voxels to shade: give colour (the voxels of one colour), or put ramp colours on the "
                            "model first" + (", or select voxels of that colour" if selected else ""))
        cells, values = voxel_draw.shade(v, target, slots)

        def write():
            n, _out = self._write(cells, values, m, MAX_SHAPE_CELLS)
            return (f"Shaded {len(cells)} visible voxels{' of the selection' if selected else ''} from the {ramp} "
                    f"ramp ({n} changed): tops {steps[0][0]}, "
                    f"sides {steps[len(steps) // 2 if len(steps) > 2 else 1][0]}, undersides {steps[-1][0]}.")

        return self._edit(write)

    def _name_shader_colours(self, shader: str, named_before: set) -> str:
        """In an art scene, name the colours a shader run brought in (after the shader and
        its own colour names), so they count as chosen colours."""
        if self.art is None:
            return ""
        scene = self.doc.scene
        store = scene.colours
        used = [c for c in self._used() if c not in named_before and c not in self.colour_names()]
        if not used:
            return ""
        label = str(shader).split("/")[-1].replace("_", " ")
        for k, slot in enumerate(sorted(used)):
            ident = store.id_of(slot)
            if ident is None:
                ident = store.claim(slot, tuple(int(v) for v in scene.palette[slot, :3]))
            store.edit(ident, name=f"{label} {k + 1}")
        self.doc.undo.push_colours()
        return f" Named the shader's {len(used)} colour{'s' if len(used) != 1 else ''} after it."

    _shader_cache: dict | None = None

    def _cached_shaders(self) -> dict:
        """Every shader read once, at the first call: a server whose checkout moved
        away while it runs still has them."""
        if Agent._shader_cache is None:
            from rok.app import xs_ops

            cache = {}
            for name in xs_ops.shader_files():
                try:
                    cache[name] = xs_ops.shader_by_name(name)
                except (OSError, ValueError):
                    continue
            Agent._shader_cache = cache
        return Agent._shader_cache

    def _shaders(self) -> list[str]:
        from rok.app import xs_ops

        return sorted(set(xs_ops.shader_files()) | set(self._cached_shaders()))

    def _shader(self, name: str):
        from rok.app import xs_ops

        try:
            found = xs_ops.shader_by_name(str(name))
        except OSError:
            found = None
        if found is None:
            found = self._cached_shaders().get(str(name))
        if found is None:
            raise ToolError(f"no shader {name!r}; rok_edit do=shaders lists them with their arguments")
        return found

    @staticmethod
    def _arg_line(arg) -> str:
        return f"{arg.name} = {arg.value:g} ({arg.low:g} to {arg.high:g})"

    @staticmethod
    def _shader_notes(source: str) -> str:
        """The shader's own explanation: its leading comment paragraphs, without
        the header (the xs line, xs_begin..xs_end, author, args, rok_ lines and
        what follows them). It says what an argument's numbers mean (Kind 0 is snow...)."""
        lines = source.splitlines()
        paragraphs, current = [], []
        for line in lines:
            t = line.strip()
            if not t.startswith("//"):
                if t:
                    break  # the code starts
                if current:
                    paragraphs.append(current)
                    current = []
                continue
            text = t[2:].strip()
            if not text:
                if current:
                    paragraphs.append(current)
                    current = []
                continue
            current.append(text)
        if current:
            paragraphs.append(current)
        header = ("xs ", "xs_begin", "xs_end", "arg :", "arg:", "author", "rok_")
        keep = [p for p in paragraphs if not any(line.startswith(header) for line in p)]
        return "\n".join(" ".join(p) for p in keep)

    def _shader_list(self, name: str = "") -> str:
        """Every shader with what it makes and its arguments; or one shader's."""
        from rok.core import xs

        names = [str(name)] if name else self._shaders()
        lines = []
        for n in names:
            s = self._shader(n)
            about = f": {s.about}" if s.about else ""
            lines.append(f"{n}{about}")
            if s.args and name:
                lines.append("  args:")
                lines += [f"  - {self._arg_line(a)}" + (f": {a.tip}" if a.tip else "") for a in s.args]
            elif s.args:
                lines.append("  args: " + "; ".join(self._arg_line(a) for a in s.args))
            if name:
                notes = self._shader_notes(s.source)
                if notes:
                    lines.append("  " + notes.replace("\n", "\n  "))
            if s.colours is not None:
                names = ", ".join(s.colours.name(k) for k in range(len(s.colours.rgb)))
                lines.append(f"  its own colours (set them with colours): {names}")
            elif xs.acts_on_selection(s):
                lines.append("  works on the current colour (a marker): set it with do=current_colour or pass colour")
        return "\n".join(lines)

    def _current_colour(self, index) -> str:
        i = _colour(index)
        if i == 0:
            raise ToolError("colour 0 is empty; the current colour is 1 to 255")
        from rok.app import object as voxel_object

        self.doc.colour = i
        voxel_object.redraw_3d_views()
        rgb = self.doc.scene.palette[i, :3] if self.doc.scene is not None else (0, 0, 0)
        return f"The current colour is {i} (#{''.join(f'{int(c):02x}' for c in rgb)})."

    def _grow_shader(self, name, args, colour, colour_count, iterations, colours=None, where=None,
                     only_on=None, only_colours=None) -> str:
        """Run a shader on the whole model and keep only what grows from the
        selection: changes to selected voxels, and new voxels joined to them face
        to face through other new ones. The rest goes back; one undo step; the
        selection stays. (Grass on the rock, not on the roof: a plain selection
        lets a shader change only selected cells, and new cells are empty, so
        never selected.) With nothing selected, the voxels that only_colours and
        only_on name are what it grows from, and the run itself is not limited."""
        from rok.app.ops_common import commit
        from rok.core import morph
        from rok.core.undo import stacked_record

        m = self.doc.model
        sel = self.doc.selection
        source = None
        if sel is None or sel.is_empty():
            if not (only_on or only_colours):
                raise ToolError("grow needs a selection, the voxels the shader grows from: select them first (for example "
                                'rok_select {"do": "colour", "colour": 12}), or give only_colours and/or only_on '
                                "(they name the voxels when nothing is selected)")
            from rok.core import shader_scope

            given = where or {}
            filters = shader_scope.Where(scope=shader_scope.MODEL,
                                         faces=shader_scope.parse_faces(given.get("only_on", "")),
                                         colours=shader_scope.parse_colours(given.get("only_colours", "")))
            source = shader_scope.allowed(m.grid.voxels, filters)
            if source is None or not source.any():
                raise ToolError("no voxels match only_colours and only_on, so nothing can grow from them")
            # The filters named the source; the run itself goes over the whole model.
            where = {k: v for k, v in (where or {}).items() if k not in ("only_on", "only_colours", "room")}
        mask = sel.mask.copy() if source is None else source
        before = m.grid.voxels.copy()
        undo = self.doc.undo
        depth = len(undo._undo)
        held = None if sel is None else sel.mask.copy()
        if sel is not None:
            sel.clear()
        try:
            note = self._run_shader(name, args, colour, colour_count, iterations, colours, where)
        finally:
            if sel is not None:
                sel.replace(held)
        after = m.grid.voxels.copy()
        changed = before != after
        added = (before == 0) & (after != 0)
        grown = added & morph.dilate(mask, 1)
        for _ in range(max(after.shape)):
            more = added & morph.dilate(grown, 1)
            if np.array_equal(more, grown):
                break
            grown = more
        keep = (changed & mask) | grown
        back = changed & ~keep
        if back.any():
            cells = np.argwhere(back)
            self._write(cells, before[back], m)
        new = []
        while len(undo._undo) > depth:
            new.append(undo._undo.pop())
        if new:
            commit(stacked_record(*reversed(new)), modified=True)
            self.app.after_event()
        what = "the selection" if source is None else "the voxels of only_colours / only_on"
        return note + f" Kept what grows from {what}; {int(back.sum())} other changed cells went back."

    def _shader_colours(self, s, colours) -> str:
        """rok.xs_run's colours_rgb from a list of hex in the shader's order, or
        {name: hex} by the names of its rok_colours line ('' for none)."""
        if not colours:
            return ""
        if s.colours is None:
            raise ToolError(f"{s.name} has no colours of its own; it uses the current colour (colour=...)")
        out = ["-"] * len(s.colours.rgb)
        if isinstance(colours, dict):
            by_name = {s.colours.name(k).lower(): k for k in range(len(s.colours.rgb))}
            for key, h in colours.items():
                k = by_name.get(str(key).lower())
                if k is None:
                    raise ToolError(f"{s.name} has no colour {key!r}; its colours: "
                                    + ", ".join(s.colours.name(k) for k in range(len(s.colours.rgb))))
                out[k] = str(h).lstrip("#")
        else:
            for k, h in enumerate(list(colours)[: len(out)]):
                out[k] = str(h).lstrip("#") if h else "-"
        for h in out:
            if h != "-" and xs_parse_hex(h) is None:
                raise ToolError(f"{h!r} is no colour; use hex like #ff8800")
        return ",".join(out)

    @staticmethod
    def _shader_scope(scope, only_on, only_colours, room, feather=0, volume=None, volume_shape="box") -> dict:
        """rok.xs_run's scope settings from the tool's arguments."""
        from rok.core import shader_scope

        out = {}
        if scope:
            key = str(scope).upper()
            if key not in shader_scope.SCOPES:
                raise ToolError("scope must be model, selection or not_selection")
            out["scope"] = key
        if only_on:
            faces = [str(f).upper() for f in ([only_on] if isinstance(only_on, str) else only_on)]
            bad = [f for f in faces if f not in shader_scope.FACES]
            if bad:
                raise ToolError(f"only_on takes top, sides, bottom, edges, hollows; not {', '.join(bad).lower()}")
            out["only_on"] = ",".join(faces)
        if only_colours:
            out["only_colours"] = ",".join(str(_colour(c)) for c in only_colours)
        if room:
            out["room"] = max(0, min(shader_scope.ROOM_MAX, int(room)))
        if feather:
            out["feather"] = max(0, min(shader_scope.FEATHER_MAX, int(feather)))
        if volume is not None:
            if len(volume) != 6:
                raise ToolError("volume is [x0, y0, z0, x1, y1, z1]: two opposite corners, inclusive")
            out["scope"] = shader_scope.VOLUME
            out["volume"] = ",".join(str(int(v)) for v in volume)
            shape = str(volume_shape or "box").upper()
            if shape not in shader_scope.SHAPES:
                raise ToolError("volume_shape must be box or sphere")
            out["volume_shape"] = shape
        elif out.get("scope") == shader_scope.VOLUME:
            raise ToolError("scope volume needs volume: [x0, y0, z0, x1, y1, z1]")
        return out

    def _run_shader(self, name, args, colour, colour_count, iterations, colours=None, where=None) -> str:
        """Run a shader as its settings panel does: arguments by position or by
        name, the rest at the shader's own defaults (an argument left out was 0
        before, which clamped it to its lowest value)."""
        from rok.app import xs_ops
        from rok.core import xs

        s = self._shader(name)
        values = [a.value for a in s.args]
        if isinstance(args, dict):
            by_name = {a.name.lower(): k for k, a in enumerate(s.args)}
            for key, v in args.items():
                k = by_name.get(str(key).lower())
                if k is None:
                    raise ToolError(f"{name} has no argument {key!r}; its arguments: "
                                    + "; ".join(self._arg_line(a) for a in s.args))
                values[k] = float(v)
        elif args:
            if len(args) > len(s.args):
                raise ToolError(f"{name} takes {len(s.args)} arguments: " + "; ".join(self._arg_line(a) for a in s.args))
            for k, v in enumerate(args):
                values[k] = float(v)
        props = {f"a{k}": float(v) for k, v in enumerate(values)}
        if colour_count is None:
            colour_count = 1 if xs.acts_on_selection(s) else xs_ops.COLOURS_DEFAULT
        rgb = self._shader_colours(s, colours)
        m = self.doc.model
        before = m.grid.voxels.copy()
        self.app_run("rok.xs_run", shader=str(name), colour=_colour(colour) if colour is not None else 0,
                     colour_count=int(colour_count), iterations=max(1, int(iterations)), colours_rgb=rgb,
                     **(where or {}), **props)
        used = "; ".join(f"{a.name} = {v:g}" for a, v in zip(s.args, values))
        if np.array_equal(before, m.grid.voxels):
            where = where or {}
            if (where.get("only_colours") or where.get("only_on")) and not where.get("room"):
                # The filters allow only filled voxels, and this shader only adds: every
                # agent stepped on this once. Run it again with room above the tops.
                roomy = {**where, "only_on": where.get("only_on") or "TOP", "room": AUTO_ROOM}
                again = self._run_shader(name, args, colour, colour_count, iterations, colours, roomy)
                if not again.startswith("Warning"):
                    added = "" if where.get("only_on") else 'only_on ["top"] and '
                    return (f"{again} It changed nothing at first: only_colours and only_on allow only voxels "
                            f"that are there, and it adds into empty cells; so it ran with {added}room "
                            f"{AUTO_ROOM} added.")
            # Said first, so a batch's one-line summary of the step shows it too.
            return f"Warning: {name} changed nothing (with {used or 'no arguments'}). " + self._why_unchanged(s, where)
        text = f"Ran {name}" + (f" with {used}." if used else ".")
        emptied = int(((before != 0) & (m.grid.voxels == 0)).sum())
        if emptied and s.removes:
            # "+257 added, -344 removed" from a detail shader read as damage.
            text += f" It emptied {emptied} cells on purpose: {s.removes}"
        return text

    def _why_unchanged(self, s, where) -> str:
        """Why a shader run left the model as it was, and how to fix the call.
        An agent's trees on a selected ground added 0 voxels with no word."""
        from rok.core import xs

        m = self.doc.model
        where = where or {}
        sel = m.selection
        reasons = []
        if not where.get("scope") and sel is not None and not sel.is_empty():
            reasons.append(f"A selection of {int(sel.count())} voxels is active, so the shader may change only those "
                           "voxels; new voxels go into empty cells, which are never selected. Give "
                           "volume=[x0, y0, z0, x1, y1, z1] (the box to build in), scope=\"model\", grow=true (keep "
                           "what grows from the selection), or rok_select do=none first.")
        elif where.get("scope") == "SELECTION":
            reasons.append("scope selection lets it change only selected voxels, and new voxels go into empty cells: "
                           "use a volume or scope model to build.")
        if (where.get("only_colours") or where.get("only_on")) and not where.get("room"):
            reasons.append("only_colours and only_on let it change only voxels that are already there, of those "
                           "colours or on those faces; a shader that adds (grass, flowers, snow, trees) writes into "
                           "the empty cells next to them, and room opens those. For grass on the tops of colour 12: "
                           "only_colours=[12], only_on=[\"top\"], room=6 (room above the tops only).")
        filled = np.argwhere(m.grid.voxels != 0)
        top = int(filled[:, 2].max()) if len(filled) else -1
        if "volume" not in where and not xs.is_brush(s) and m.grid.size[2] - 1 - top < 8:
            reasons.append(f"The model's voxels reach z = {top} of a grid {m.grid.size[2]} cells high, so a shader "
                           "that builds upward (trees, rocks) has little room: rok_edit do=resize makes the grid "
                           "taller.")
        if not reasons:
            reasons.append("With these arguments, this colour and this scope it has nothing to do: "
                           "rok_edit do=shaders with its name says what it needs (a colour to work on, ground to "
                           "stand on, a volume).")
        return " ".join(reasons)

    def _set_colour(self, index, hex_, glow=None, power=None) -> str:
        from rok.app import object as voxel_object
        from rok.app.ops_common import commit
        from rok.core.undo import PaletteRecord

        i = _colour(index)
        if i == 0:
            raise ToolError("colour 0 is empty and has no colour; use 1 to 255")
        h = str(hex_ or "").lstrip("#")
        if hex_ is None and glow is None:
            raise ToolError("set_colour needs hex (like #ff8800), glow (0 to 1), or both")
        if hex_ is not None and (len(h) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in h)):
            raise ToolError("hex must be a colour like #ff8800")
        if power is not None and glow is None:
            raise ToolError("power sets how bright a glowing colour is; give glow (0 to 1) with it")
        scene = self.doc.scene
        before = scene.palette.copy()
        materials_before = None
        lines = []
        if hex_ is not None:
            scene.palette[i] = [*(int(h[k:k + 2], 16) for k in (0, 2, 4)), 255]
            lines.append(f"Colour {i} is now #{h.lower()}.")
        if glow is not None:
            material = self._glow_material(scene.materials.get(i), glow, power)
            if material != scene.materials.get(i):
                materials_before = scene.materials.copy()
                scene.materials.set(i, material)
            lines.append(f"Colour {i} glows at {material.weight:g}, power {material.flux}." if material.type == "EMIT"
                         else f"Colour {i} no longer glows.")
        self.doc.made.add(i)  # kept while unused, as a colour set in the app
        voxel_object.update_palette()
        commit(PaletteRecord(i, before, scene.palette.copy(), self.doc.palette_session,
                             materials_before=materials_before,
                             materials_after=scene.materials.copy() if materials_before is not None else None))
        return " ".join(lines)

    @staticmethod
    def _glow_material(current, glow, power):
        """The slot's material with the glow set: a light of strength `glow` at `power`, or plain at glow 0."""
        try:
            g = float(glow)
        except (TypeError, ValueError):
            g = -1.0
        if not 0.0 <= g <= 1.0:
            raise ToolError("glow is 0 to 1 (0: no glow)")
        if g == 0.0:
            return current.with_type("DIFFUSE") if current.type == "EMIT" else current
        p = 1 if power is None else power
        if not isinstance(p, int) or isinstance(p, bool) or not 0 <= p <= 4:
            raise ToolError("power is a whole number 0 to 4; each step is ten times brighter")
        return current.with_type("EMIT").with_weight(round(g, 2)).with_value(b"_flux", p)

    def tool_scene(self, do: str, name=None, offset=None, size=None, path=None, solid: str = "",
                   merge: str = "auto", discard: bool = False, model=None, to=None, target=None,
                   budget=None) -> Reply:
        do = str(do).lower()
        if budget is not None and do not in ("new", "budget"):
            raise ToolError("budget is for do=new (the caps of a new scene) and do=budget (new caps for this one)")
        if do == "budget":
            return Reply(self.set_budget(budget))
        if do == "new":
            new_size = _xyz(size or (ART_DEFAULT,) * 3, "size")
            if max(new_size) > ART_MAX_SIDE or min(new_size) < 1:
                raise ToolError(f"a new scene is a voxel art scene: each side 1 to {ART_MAX_SIDE} cells (default "
                                f"{ART_DEFAULT}). Plan the piece small: voxel art places each voxel on purpose.")
        if model not in (None, ""):
            if do not in MODEL_SCENE_ACTIONS:
                raise ToolError(f"model is for do={', '.join(MODEL_SCENE_ACTIONS)}: the model they act on; "
                                f"do={do} acts on the whole scene")
            self._model(model)
        if (to is not None or target not in (None, "")) and do != "move":
            raise ToolError("to and target are for do=move")
        if do in ("new", "open") and self.doc.modified and not discard:
            # open and new replace the whole scene and clear the undo history: an
            # agent opened a ship to compare and lost its scene.
            raise ToolError(f"do={do} replaces the whole scene, and that cannot be undone; this scene has changes "
                            "that are not saved. Save it first (do=save), or add a file's models to this scene with "
                            "do=import, or pass discard=true to throw the changes away.")
        if do in ("open", "import", "save", "export") and not path:
            raise ToolError(f"do={do} needs path, a file path")
        if do == "list":
            scene = self.doc.scene
            rows = []
            for i, m in enumerate(scene.models):
                rows.append({"name": m.display_name(i), "grid": [int(s) for s in m.grid.size],
                             "position": [int(p) for p in m.position], "active": m.uid == self.doc.active_uid,
                             "voxels": int((m.grid.voxels != 0).sum())})
            text = "\n".join(f"{'*' if r['active'] else ' '} {r['name']}: grid {r['grid']}, at {r['position']}, "
                             f"{r['voxels']} voxels" for r in rows)
            return Reply("Models (* is the active one):\n" + text, {"models": rows})

        def run():
            if do == "new":
                self.app_run("rok.new", size=new_size, discard=True)
                return "A new, empty document. " + self.start_art(budget)
            if do == "open":
                self.app_run("rok.open", filepath=os.path.expanduser(str(path)), discard=True)
                return f"Opened {path}; it replaced the scene (no undo past this point)."
            if do == "import":
                self._exists(path)
                self.app_run("rok.model_add_file", filepath=os.path.expanduser(str(path)), merge=True)
                return (f"Added the models of {path} to the scene, beside the others; the first is the active "
                        "model. rok_scene do=undo takes them out again.")
            if do == "save":
                self.app_run("rok.save_as", filepath=os.path.expanduser(str(path)), check_existing=False)
                scene = self.doc.scene
                rows = [f"{m.display_name(i)} (grid {'x'.join(str(int(s)) for s in m.grid.size)}, "
                        f"{int((m.grid.voxels != 0).sum())} voxels)" for i, m in enumerate(scene.models)]
                return f"Saved {path} with {len(rows)} model(s): " + "; ".join(rows) + "."
            if do == "export":
                if str(merge).upper() not in ("AUTO", "ON", "OFF"):
                    raise ToolError("merge is auto, on or off")
                self.app_run("rok.export_mesh", filepath=os.path.expanduser(str(path)), check_existing=False,
                             merge=str(merge).upper())
                return f"Exported {path}."
            if do in ("undo", "redo"):
                stack = self.doc.undo
                record = stack.peek() if do == "undo" else stack.peek_redo()
                if record is None:
                    return f"Nothing to {do}."
                self.app_run(f"rok.{do}")
                # name the step, and the one the next undo would take back: undo three
                # times blind and a step that changed nothing hides which one went
                text = f"{'Undid' if do == 'undo' else 'Redid'} {getattr(record, 'label', '') or 'one step'}."
                after = stack.peek()
                return text + (f" The next undo would take back {after.label}." if after is not None else
                               " Nothing is left to undo.")
            if do == "add":
                self.app_run("rok.model_add", solid=str(solid))
                return "Added a model; it is the active one."
            if do in ("duplicate", "delete", "split", "merge_all"):
                self.app_run({"split": "rok.model_split", "merge_all": "rok.model_merge_all"}.get(do, f"rok.model_{do}"))
                return f"Ran {do}."
            if do == "rename":
                self.app_run("rok.model_rename", name=str(name))
                return f"Renamed the active model to {name}."
            if do == "activate":
                self._model(name)
                return f"\"{name}\" is the active model."
            if do == "move":
                if (offset is None) == (to is None):
                    raise ToolError("move takes offset [dx, dy, dz] or to [x, y, z] (with target), not both and not "
                                    "neither")
                if to is not None:
                    return self._move_to(to, target)
                self.app_run("rok.model_move", offset=_xyz(offset, "offset"))
                return f"Moved the model by {list(_xyz(offset))}."
            raise ToolError("do must be list, new, open, import, save, export, undo, redo, add, duplicate, delete, "
                            "split, merge_all, rename, activate or move")

        if do == "open":
            self._exists(path)
        if do in ("new", "open"):
            note = run()
            if do == "open":
                kind, why = verdict([m.grid.voxels for m in self.doc.scene.models])
                note += f" Verdict: {kind}" + (f" ({'; '.join(why)})" if why else "") + "."
            reply = self.tool_look()
            reply.text = note + "\n" + reply.text
            return reply
        # A save or an export leaves the model as it was: a picture would only cost tokens.
        return self._edit(run, look=do not in ("save", "export"))

    def _move_to(self, to, target) -> str:
        """Set the active model on cell `to` of `target`: the middle of its voxels
        over the cell in x and y, its lowest voxels in the cell's layer."""
        scene = self.doc.scene
        moved = self.doc.model
        others = [m for m in scene.models if m.uid != moved.uid]
        if target in (None, ""):
            if not others:
                raise ToolError("to needs another model to stand on; this scene has only one")
            ground = others[0]
        else:
            ground = next((m for i, m in enumerate(scene.models) if target in (m.display_name(i), m.name)), None)
            if ground is None:
                names = ", ".join(m.display_name(i) for i, m in enumerate(scene.models))
                raise ToolError(f"no model named {target!r}; the models are: {names}")
            if ground.uid == moved.uid:
                raise ToolError("target is the model being moved; name the model it should stand on")
        cells = np.argwhere(moved.grid.voxels != 0)
        if not len(cells):
            raise ToolError("the active model has no voxels to set down")
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        t = np.asarray(_xyz(to, "to"))
        low = np.array([t[0] - (hi[0] - lo[0]) // 2, t[1] - (hi[1] - lo[1]) // 2, t[2]])
        want = np.asarray(ground.position) + low - lo  # the moved model's new position
        offset = [int(v) for v in want - np.asarray(moved.position)]
        self.app_run("rok.model_move", offset=tuple(offset))
        name = self._name(ground)
        return (f"Set the model on cell {t.tolist()} of {name}: its voxels now fill {low.tolist()}.."
                f"{(low + hi - lo).tolist()} in {name}'s cells (moved by {offset}).")

    @staticmethod
    def _exists(path) -> None:
        from pathlib import Path

        if not Path(str(path)).expanduser().is_file():
            raise ToolError(f"there is no file {path}")

    def tool_command(self, do: str = "list", id=None, props: dict | None = None, filter: str = "",
                     code: bool = False) -> Reply:
        from rok import agent_commands
        from rok.app import opkit

        if do == "describe":
            ids = id if isinstance(id, (list, tuple)) else [id] if id else []
            if not ids:
                raise ToolError('describe needs id: one command or a list, {"do": "describe", "id": ["bevel", "lathe"]}')
            text, missing = agent_commands.describe(ids, code=bool(code))
            if missing:
                note = f"No command {', '.join(map(repr, missing))}; rok_command do=list shows them."
                if not text:
                    raise ToolError(note)
                text += "\n\n" + note
            return Reply(text, {"text": text, "missing": missing})
        if isinstance(id, (list, tuple)):
            raise ToolError("run takes one id; describe takes a list")

        if do == "list":
            rows = []
            for k, cls in sorted(opkit.REGISTRY.items()):
                desc = getattr(cls, "tooltip", "") or getattr(cls, "title", "")
                if filter and filter.lower() not in (k + " " + desc).lower():
                    continue
                params = []
                for pname, p in opkit.props_of(cls).items():
                    items = getattr(p, "items", None)
                    choice = "" if not items or callable(items) else "{" + ",".join(str(i[0]) for i in items) + "}"
                    params.append(f"{pname}={p.default!r}{choice}")
                rows.append(f"{k}({', '.join(params)}): {desc}")
            # The rows go in the structured result too: a client that shows only
            # that saw just the count.
            return Reply("\n".join(rows) or "No command matches.", {"count": len(rows), "commands": rows})
        if do != "run" or not id:
            raise ToolError("do must be list (optionally with filter), describe (with id: one command or a list) "
                            "or run (with id and props)")
        id = agent_commands.full_id(id)
        if isinstance(props, str):
            import json

            try:
                props = json.loads(props)
            except ValueError:
                raise ToolError('props must be an object like {"across": 2}, not a string') from None
        if props is not None and not isinstance(props, dict):
            raise ToolError('props must be an object like {"across": 2}')
        props = dict(props or {})
        if id == "rok.new":
            # a scene the LLM starts is a voxel art scene, by either road
            return self.tool_scene("new", size=props.get("size"), discard=bool(props.get("discard", False)))
        cls = opkit.REGISTRY.get(id)
        if cls is None:
            raise ToolError(f"no command {id!r}; rok_command do=list shows them")
        known = opkit.props_of(cls)
        unknown = sorted(k for k in props if k not in known)
        if unknown:
            # An operator drops a property it does not know without a word, so
            # the call ran with the defaults (terrain_new).
            raise ToolError(f"{id} has no parameter {', '.join(unknown)}; its parameters are: "
                            + (", ".join(f"{k}={p.default!r}" for k, p in known.items()) or "none"))
        used = ", ".join(f"{k}={v!r}" for k, v in props.items())
        return self._edit(lambda: (self.app_run(id, **props), f"Ran {id}" + (f" with {used}." if used else "."))[1])

    STEP_FORM = '{"tool": "rok_shape", "args": {"kind": "box", "low": [0, 0, 0], "high": [3, 3, 0]}}'

    def _steps(self, steps) -> list:
        """The batch's steps as a list. A JSON text of the list is read too:
        models sometimes send the array as a string."""
        if isinstance(steps, str):
            import json

            try:
                steps = json.loads(steps)
            except ValueError:
                raise ToolError("steps must be an array of step objects, not a string; each step is "
                                + self.STEP_FORM) from None
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ToolError("steps must be a non-empty array of step objects; each step is " + self.STEP_FORM)
        return list(steps)

    def _steps_file(self, path) -> list:
        """The steps of a JSON file: the array itself, or an object with "steps"."""
        import json
        from pathlib import Path

        file = Path(os.path.expanduser(str(path)))
        if not file.is_file():
            raise ToolError(f"no file {path}; path is a JSON file of batch steps")
        try:
            data = json.loads(file.read_text())
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise ToolError(f"{path} is not a JSON file of steps: {exc}") from None
        if isinstance(data, dict):
            data = data.get("steps")
        return self._steps(data)

    def _step(self, step) -> tuple[str, dict]:
        """(tool, args) of one batch step. The arguments go in "args"; a step
        with them next to "tool" ({"tool": "rok_voxels_set", "voxels": [...]})
        is read the same way."""
        if not isinstance(step, dict) or "tool" not in step:
            raise ToolError("each step must be an object like " + self.STEP_FORM)
        tool = str(step["tool"]).removeprefix("rok_")
        if tool in ("batch", "look", "probe"):
            raise ToolError(f"{tool} cannot run inside a batch")
        if not hasattr(self, "tool_" + tool):
            raise ToolError(f"no tool {step['tool']!r}")
        flat = {k: v for k, v in step.items() if k not in ("tool", "args")}
        args = step.get("args")
        if args is not None and flat:
            raise ToolError(f"put every argument in \"args\" (found {', '.join(sorted(flat))} next to it); "
                            "a step is " + self.STEP_FORM)
        if args is None:
            alias = [k for k in ("arguments", "params", "parameters", "input") if isinstance(flat.get(k), dict)]
            args = flat[alias[0]] if len(flat) == 1 and alias else flat
        if not isinstance(args, dict):
            raise ToolError("a step's \"args\" must be an object, like " + self.STEP_FORM)
        return tool, args

    def tool_batch(self, steps=None, path=None) -> Reply:
        """Run several tool calls in order as one undo step, with one picture at the end.
        The steps come inline, or from a JSON file at `path` (a script can write a
        few hundred shapes there instead of pasting them into the call)."""
        from rok.app.ops_common import commit
        from rok.core.undo import stacked_record

        if (steps is None) == (path is None):
            raise ToolError("give either steps (the array of steps) or path (a JSON file holding that array, or "
                            '{"steps": [...]}), not both and not neither')
        steps = self._steps(steps) if path is None else self._steps_file(path)
        if len(steps) > MAX_BATCH_STEPS:
            raise ToolError(f"at most {MAX_BATCH_STEPS} steps per batch; split it into two batches")
        undo = self.doc.undo
        depth = len(undo._undo)
        before = self._snapshot()
        before_box = self._voxel_box(self.doc.model) if self.doc.model is not None else None
        notes = []
        self._batch_depth += 1
        failed = None
        try:
            for i, step in enumerate(steps):
                try:
                    tool, args = self._step(step)
                except ToolError as exc:
                    failed = (i, str(exc))
                    break
                r = self.call(tool, args)
                if r.error:
                    failed = (i, r.text)
                    break
                notes.append(f"{i + 1}. {r.text.splitlines()[0]}")
        finally:
            self._batch_depth -= 1
        new = []
        while len(undo._undo) > depth:
            new.append(undo._undo.pop())
        if new:
            commit(stacked_record(*reversed(new)), modified=True)
            self.app.after_event()
        diff = self._diff(before)
        if new:
            try:
                self.enforce_art()
            except ToolError as exc:
                return Reply(f"{exc} (the whole batch)", error=True)
        reply = self._feedback(diff, "\n".join(notes), before_box)
        if failed is not None:
            i, why = failed
            reply.text = f"Step {i + 1} failed: {why}\nThe steps before it ran and stay (one undo step).\n" + reply.text
            reply.error = True
        if self.feedback == "image":
            self._attach_picture(reply, diff)
        return reply
