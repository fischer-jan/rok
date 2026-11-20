"""Run shaders on the active model (the shader gallery, Cmd+K).

The shaders are in the xs format. Rok lists its own set (rok/shaders, in
groups) and the files in your shader folder, where a file named like one of
Rok's replaces it. `rok.core.xs` parses and wraps them; this module runs them on the GPU and writes the result.

How a run works: the model is a 2D texture atlas of its z slices, laid out in
rows of tiles so that a 256-high model stays inside the GPU's texture limit; for each z
one fragment pass draws a region the size of the x-y slice into an offscreen,
the fragment calls the shader's map() for its cell and writes the index,
and the slice is read back. Several iterations feed each result into the next
pass, with `i_iter` counting. With a selection only the selected cells
change. One run is one undo step.
"""

from __future__ import annotations

from .opkit import CONTEXT, Action, ops, register

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from .opkit import BoolProp, FloatProp, IntProp, StringProp

from rok.core import morph, shader_palette, shader_scope, xs
from rok.core.material import DEFAULT
from rok.core import recolour as recolour_core
from . import object as voxel_object
from . import select_ops
from .object import document
from .ops_common import can_edit_model, refresh_views

ITER_MAX = 64
# Shaders that pick from the selected palette colours (xs color_sel) start with this
# many: with one, noise recolours a colour into itself and a gradient is flat.
COLOURS_DEFAULT = 4


def user_shader_folder() -> Path:
    """Your shader folder: your shaders and ones you download go here."""
    if sys.platform == "darwin":
        from . import recent

        return recent.APP_FOLDER / "shaders"
    return Path.home() / ".config" / "rok" / "shaders"


def bundled_folder() -> Path:
    """The shaders that come with Rok (rok/shaders)."""
    return Path(__file__).resolve().parent.parent / "shaders"


def shader_files() -> dict[str, Path]:
    """Every shader by name (nature/rocks for a file in a group folder): Rok's
    own set, then yours; yours replaces one of Rok's of the same name."""
    return xs.with_personal(xs.find_named((bundled_folder(),)), xs.find_named((user_shader_folder(),)))


def shader_by_name(name: str) -> xs.Shader | None:
    path = shader_files().get(name)
    return None if path is None else xs.load(path, name)


# --- the GPU pass ---------------------------------------------------------------------
#
# rok_native.ShaderRunner runs one fragment
# pass per z slice over an atlas of the model, read back slice by slice,
# with a loop guard, a time limit and a check for draws the GPU dropped.


class ShaderError(RuntimeError):
    """The shader does not compile, or the GPU run failed."""


class ModelTooLarge(ShaderError):
    """The model's texture atlas is larger than the GPU allows."""


class GpuGaveUp(ShaderError):
    """The GPU dropped a draw even at the smallest loop limit."""


class TooSlow(ShaderError):
    """The run took longer than xs.RUN_SECONDS."""


class NotRunnable(ShaderError):
    """Rok refuses the shader before the GPU sees it (xs.Unsupported)."""


_runner = None
_drops = 0  # dropped runs this session: after one, runs get a quarter of the budget


def _gpu():
    global _runner
    if _runner is None:
        import rok_native

        _runner = rok_native.ShaderRunner()
    return _runner


def run(
    shader: xs.Shader,
    voxels: np.ndarray,
    *,
    colour: int,
    palette: np.ndarray,
    args=(),
    iterations: int = 1,
    colours=None,
    mirror=(False, False, False),
    axis=(0.0, 0.0, 0.0),
    info: dict | None = None,
) -> np.ndarray:
    """The model after the shader, as a new uint8 array. Raises ShaderError.
    info, when given, gets "stopped": the cells whose loops the guard ended
    (see xs.LOOP_BUDGET), counted over all iterations. `axis` all zeros means
    every axis."""
    global _drops
    try:
        source = xs.fragment_source(shader)
    except xs.Unsupported as exc:
        raise NotRunnable(str(exc)) from exc
    voxels = np.ascontiguousarray(voxels, dtype=np.uint8)
    cols, _rows = xs.atlas_layout(voxels.shape)
    values = [float(v) for v in args][: xs.MAX_ARGS]
    values += [a.value for a in shader.args[len(values):]]
    values = (values + [0.0] * xs.MAX_ARGS)[: xs.MAX_ARGS]
    names = list(xs.arg_names(shader))
    by_var = {a.var: values[k] for k, a in enumerate(shader.args) if k < xs.MAX_ARGS}
    sels = [int(c) for c in (colours or [int(colour)])]
    budget = xs.LOOP_BUDGET // (4 if _drops else 1)
    try:
        out, stopped = _gpu().run(
            source, names, [float(by_var.get(n, 0.0)) for n in names], values, voxels, int(cols),
            np.ascontiguousarray(palette, dtype=np.uint8).reshape(256, 4), sels, float(colour),
            iterations=max(1, int(iterations)),
            mirror=tuple(float(bool(m)) for m in mirror), axis=tuple(float(a) for a in axis),
            loop_limit=int(budget), seconds=float(xs.RUN_SECONDS),
        )
    except RuntimeError as exc:
        text = str(exc)
        if "too large" in text:
            raise ModelTooLarge(text) from None
        if "dropped" in text:
            _drops += 1
            raise GpuGaveUp(text) from None
        if "seconds" in text:
            raise TooSlow(text) from None
        raise ShaderError(text) from None
    if info is not None:
        info["stopped"] = int(stopped)
    return np.asarray(out, dtype=np.uint8)


class PaletteFull(Exception):
    """A run's result needs more new colours than there are free slots."""


def shader_palette_for(shader: xs.Shader, voxels: np.ndarray, colour: int, palette: np.ndarray,
                       colour_count: int | None = None) -> shader_palette.ShaderPalette:
    """The palette shader `shader` sees on `voxels` (your slots), with the
    colour you paint with on slot `colour` (rok.core.shader_palette)."""
    if colour_count is None:
        colour_count = 1 if xs.acts_on_selection(shader) else COLOURS_DEFAULT
    need = shader.colours
    count = colour_count if xs.uses_colour_selection(shader) else 1
    unused = ()
    scene = document.scene
    if scene is not None and palette is scene.palette:
        store = scene.colours
        unused = [(store.entries[i].rgba, store.slot_of(i)) for i in store.unused_ids(scene.used_counts())]
    own_materials = () if need is None else tuple(need.material(k) for k in range(len(need.rgb)))
    return shader_palette.build(palette, voxels, colour, own=None if need is None else need.rgb,
                                marker=need is not None and need.start == 1, count=count, unused=unused,
                                own_materials=own_materials)


def palette_slots(palette: np.ndarray, used: np.ndarray, pinned=()):
    """A `free_slot` for ShaderPalette.back on a bare palette: the lowest slot
    that holds exactly the colour, else the lowest slot no voxel uses."""
    free = [s for s in range(1, 256) if not int(used[s]) and s not in set(pinned)]
    given: dict = {}

    def free_slot(colour, hint: int = 0, material=None) -> int:
        colour = tuple(int(c) for c in colour)
        if colour not in given:
            match = np.flatnonzero((np.asarray(palette)[1:] == np.asarray(colour, dtype=np.uint8)).all(axis=1))
            if hint in free and tuple(int(c) for c in palette[hint]) == colour:
                given[colour] = hint
                free.remove(hint)
            elif len(match):
                given[colour] = int(match[0]) + 1
            elif free:
                given[colour] = hint if hint in free else free[0]
                free.remove(given[colour])
            else:
                raise PaletteFull("the palette has no free slot for a new colour")
        return given[colour]

    return free_slot


def run_coloured(shader: xs.Shader, voxels: np.ndarray, *, colour: int, palette: np.ndarray,
                 colour_count: int | None = None, used=None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Run a shader the way the dialog does, on its own palette
    (shader_palette_for): (the result, the palette with the new colours the
    result uses). For scripts and checks; the dialog does the same with undo.

    New colours take slots no voxel uses. This sees only `voxels`. When other
    models share the palette, pass `used`, their voxel count per slot
    (Scene.used_counts()), or their colours are overwritten."""
    palette = np.array(palette, copy=True)
    sp = shader_palette_for(shader, voxels, colour, palette, colour_count)
    counts = np.bincount(np.asarray(voxels).ravel(), minlength=256)
    if used is not None:
        counts = counts + np.asarray(used)
    counts[colour] += 1  # the colour you paint with stays
    raw = run(shader, sp.to_run(voxels), colour=sp.colour, palette=sp.palette, colours=sp.run, **kwargs)
    new, writes = sp.back(raw, palette_slots(palette, counts))
    for slot, value in writes.items():
        palette[slot] = value
    return new, palette


# --- the operator -------------------------------------------------------------------


def _arg_props():
    """a{k} holds argument k for the run: the settings panel (rok.shader_panel)
    and scripts (a0=8) pass them."""
    return {f"a{k}": FloatProp(name=f"Arg {k + 1}", default=0.0, options={"SKIP_SAVE"})
            for k in range(xs.MAX_ARGS)}


def where_of(scope="", only_on="", only_colours="", room=0, feather=0, volume="", volume_shape="") -> shader_scope.Where:
    """A run's Where from rok.xs_run's settings."""
    return shader_scope.Where(scope, shader_scope.parse_faces(only_on), shader_scope.parse_colours(only_colours),
                              int(room), int(feather), shader_scope.parse_volume(volume),
                              volume_shape or shader_scope.BOX)


CURRENT = object()  # compute's default: the model's selection as it is now


def selection_mask() -> np.ndarray | None:
    """The active model's selection as a mask, or None when nothing is selected."""
    selection = document.selection
    return None if selection is None or selection.is_empty() else selection.mask


def limit_of(where: shader_scope.Where | None = None, voxels: np.ndarray | None = None,
             selection=CURRENT) -> np.ndarray | None:
    """The cells a run may change for a scope, or None for all of them. No
    scope: the selection when there is one. A live shader
    passes its own `voxels` and the `selection` it was made with."""
    mask = selection_mask() if selection is CURRENT else selection
    voxels = document.grid.visible if voxels is None else voxels
    return shader_scope.allowed(voxels, where or shader_scope.Where(), mask)


def _scope_voxels(limit: np.ndarray | None) -> np.ndarray:
    """The voxels a run may change."""
    visible = document.grid.visible
    return visible if limit is None else visible[limit]


def _scope_name(limit: np.ndarray | None) -> str:
    return "model" if limit is None else "scope"


def _empty_hint(shader: xs.Shader, colour: int, scope: np.ndarray) -> str:
    """Why a run would have emptied the model, as far as Rok can tell."""
    filled = scope[scope > 0]
    if xs.reads_model(shader) and filled.size and np.all(filled == colour):
        return (f"It works on colour {colour} against voxels of other colours, and every voxel "
                f"is colour {colour}: paint some voxels another colour, or pick another colour")
    return "Try other values, or another colour"


def _on_selection(where: shader_scope.Where | None, selection=CURRENT) -> bool:
    """True when a run is limited to the selection (by default or by choice)."""
    mask = selection_mask() if selection is CURRENT else selection
    has = mask is not None and bool(np.any(mask))
    return (where or shader_scope.Where()).resolved(has) == shader_scope.SELECTION


def _own_box(shader: xs.Shader, where: shader_scope.Where | None, shape) -> tuple | None:
    """The volume's box as slices when the shader builds one thing in its own
    grid (rok_flags own_grid) and runs on a volume, else None. A tree run on a
    volume of a big grid was a tree for the whole grid, cut to the volume:
    mostly nothing."""
    if where is None or xs.OWN_GRID not in shader.flags or where.scope != shader_scope.VOLUME or where.volume is None:
        return None
    lo = np.maximum(np.asarray(where.volume[:3]), 0)
    hi = np.minimum(np.asarray(where.volume[3:]), np.asarray(shape) - 1)
    if np.any(hi < lo):
        return None
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))


def _no_change_hint(shader: xs.Shader, colour: int, colour_count: int, limit: np.ndarray | None = None,
                    on_selection: bool = False) -> str:
    """Why a run changed nothing, as far as Rok can tell."""
    if on_selection and not xs.is_brush(shader):
        return ("It ran on the selection only, and a selection holds filled voxels: a shader that builds "
                "(trees, grass) needs empty cells. Run it on the Model or on a Volume, or select nothing first")
    brings_colours = shader.colours is not None and shader.colours.start == 0  # paints in its own colours
    if xs.reads_model(shader) and not brings_colours and not np.any(_scope_voxels(limit) == colour):
        return f"It works on voxels of colour {colour}, and the {_scope_name(limit)} has none: pick a colour it uses"
    if xs.uses_colour_selection(shader) and not xs.acts_on_selection(shader) and colour_count == 1:
        return "It picks from the Colours, and one colour leaves nothing to change to: raise Colours"
    return "With these values the shader leaves the model as it is; try other values"


def store_slots(merge: bool = False):
    """A `free_slot` for ShaderPalette.back on the scene's colours: the slot
    of a colour you have that is exactly it, else a free one (no voxel in any
    model or frame, not the colour you paint with). With no free slot left:
    PaletteFull, or with `merge` the nearest colour voxels use."""
    scene = document.scene
    store = scene.colours
    palette, materials = scene.palette, scene.materials
    counts = scene.used_counts()
    free = store.free_slots(counts, document.pinned())
    given: dict = {}  # one slot per new colour, however often it is asked for

    def free_slot(colour, hint: int = 0, material=DEFAULT) -> int:
        colour = tuple(int(c) for c in colour)
        key = (colour, material)
        if key in given:
            return given[key]
        ident = store.find(colour, material)
        slot = None if ident is None else store.slot_of(ident)
        if slot is None:
            # A slot that already holds exactly this colour and material: the
            # same colour, so the voxels may share it (the shader's own number first).
            same = [hint] if hint else []
            same += [k for k in range(1, 256) if k != hint]
            slot = next((k for k in same if tuple(int(c) for c in palette[k]) == colour
                         and materials.get(k) == material), None)
            if slot in free:
                free.remove(slot)  # no other new colour may take it now
        if slot is None and not free and merge:
            from rok.core.colour_store import nearest

            options = [(k, palette[k]) for k in range(1, 256) if counts[k] and materials.get(k) == DEFAULT]
            slot = nearest(colour, options or [(k, palette[k]) for k in range(1, 256) if counts[k]])
        if slot is None:
            if not free:
                raise PaletteFull("the palette has no free slot for its new colours")
            slot = hint if hint in free else free[0]
            free.remove(slot)
        given[key] = slot
        return slot

    return free_slot


def _colours_note(shader: xs.Shader, colour: int) -> str:
    """One line in the dialog: what the run does with the shader's own colours."""
    n = len(shader.colours.rgb)
    if shader.colours.start == 1 and not int(document.used_counts()[colour]):
        return f"Paint the nozzles in colour {colour} first; the flame grows from them"
    store = document.scene.colours
    new = sum(1 for c in shader.colours.rgb if store.find(c) is None)
    if not new:
        return f"Its {n} colours are already in the palette"
    return f"Brings {new} new colour{'s' if new != 1 else ''}; only the ones it uses stay"


def _plan_note(writes: dict) -> str:
    """The end of the report: how many colours the run added."""
    if not writes:
        return ""
    return f"; {len(writes)} new colour{'s' if len(writes) != 1 else ''} in the palette"


@register
class XsRunAction(Action):
    command_id = "rok.xs_run"
    title = "Run Shader"
    tooltip = "Run a shader on the active model, or on the selection"
    shader = StringProp(name="Shader", options={"SKIP_SAVE"})
    colour = IntProp(
        name="Colour",
        description="The palette colour the shader works on or draws with "
        "(i_color_index); 0 is the current colour",
        default=0, min=0, max=255, options={"SKIP_SAVE"},
    )
    colour_count = IntProp(
        name="Colours",
        description="How many palette colours the shader gets, from the current one on "
        "(the shader's selected palette colours)",
        default=1, min=1, max=64, options={"SKIP_SAVE"},
    )
    iterations = IntProp(
        name="Iterations", description="How many times in a row, each on the last result",
        default=1, min=1, max=ITER_MAX, options={"SKIP_SAVE"},
    )
    colours_rgb = StringProp(
        name="Its Colours",
        description="Hex colours instead of the shader's own, in its order ('-' keeps one)",
        default="", options={"SKIP_SAVE"},
    )
    recolour = StringProp(
        name="Your Colours",
        description="Your colours for the ones the shader puts in: from:to hex pairs, comma-separated",
        default="", options={"SKIP_SAVE"},
    )
    scope = StringProp(
        name="Runs On",
        description="MODEL, SELECTION, NOT_SELECTION or VOLUME; empty: the selection when there is one",
        default="", options={"SKIP_SAVE"},
    )
    only_on = StringProp(
        name="Only On",
        description="Only voxels on these faces, comma-separated: TOP, SIDES, BOTTOM, EDGES, HOLLOWS",
        default="", options={"SKIP_SAVE"},
    )
    only_colours = StringProp(
        name="Only Colours", description="Only voxels of these palette indices, comma-separated",
        default="", options={"SKIP_SAVE"},
    )
    room = IntProp(
        name="Room", description="Cells a shader may add around a limited scope",
        default=0, min=0, max=shader_scope.ROOM_MAX, options={"SKIP_SAVE"},
    )
    volume = StringProp(
        name="Volume", description="With VOLUME: the cells x0,y0,z0,x1,y1,z1 (inclusive), empty ones too",
        default="", options={"SKIP_SAVE"},
    )
    volume_shape = StringProp(
        name="Volume Shape", description="BOX, or SPHERE: the ellipsoid inside the box",
        default="BOX", options={"SKIP_SAVE"},
    )
    feather = IntProp(
        name="Feather", description="Cells over which the shader fades out toward the edge of a limited scope",
        default=0, min=0, max=shader_scope.FEATHER_MAX, options={"SKIP_SAVE"},
    )
    live = BoolProp(
        name="Live", description="Keep the shader on the model: it runs again after each edit, until you bake it",
        default=False, options={"SKIP_SAVE"},
    )
    ask_full = BoolProp(
        name="Ask When Full", description="When the palette has no room for new colours, ask to use the nearest ones",
        default=False, options={"SKIP_SAVE", "HIDDEN"},
    )
    merge = BoolProp(
        name="Use Nearest Colours", description="New colours without a free slot take the nearest colour you have",
        default=False, options={"SKIP_SAVE"},
    )
    locals().update(_arg_props())

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def settings(self) -> dict:
        """The run's settings, the colour resolved: what a live shader keeps."""
        out = {f"a{k}": float(getattr(self, f"a{k}")) for k in range(xs.MAX_ARGS)}
        out.update(colour=self._colour(), colour_count=int(self.colour_count), iterations=int(self.iterations),
                   colours_rgb=self.colours_rgb, recolour=self.recolour, scope=self.scope, only_on=self.only_on,
                   only_colours=self.only_colours, room=int(self.room), volume=self.volume,
                   volume_shape=self.volume_shape, feather=int(self.feather))
        return out

    def invoke(self, context, event):
        # Picked by hand (Cmd+K): the settings panel sets it up and runs it.
        if shader_by_name(self.shader) is None:
            self.report({"ERROR"}, f"No shader named {self.shader}")
            return {"CANCELLED"}
        ops.rok.shader_panel("INVOKE", shader=self.shader)
        return {"CANCELLED"}

    def _colour(self) -> int:
        return int(self.colour) if self.colour > 0 else int(document.colour)

    def execute(self, context):
        from .model_ops import _apply_cells

        raw = [float(getattr(self, f"a{k}")) for k in range(xs.MAX_ARGS)]
        where = where_of(self.scope, self.only_on, self.only_colours, self.room, self.feather, self.volume,
                         self.volume_shape)
        if self.live:
            from . import live_shaders

            return live_shaders.add(self, self.shader, self.settings())
        out = compute(self.shader, raw, colours_rgb=self.colours_rgb, colour=self._colour(),
                      colour_count=self.colour_count, iterations=self.iterations, where=where,
                      recolour=self.recolour, merge=self.merge)
        if out.full and self.ask_full and CONTEXT.app is not None:
            self.merge = True  # the answer runs this action again, merging
            return context.prefs.ask_confirm(
                self, None, title="The palette is full",
                message="The shader makes colours that have no free place in the palette. Use the nearest "
                        "colours you have instead?", confirm_text="Use Nearest")
        if out.error:
            self.report({"ERROR"}, out.error)
            return {"CANCELLED"}
        grid = document.grid
        palette_record = voxel_object.write_colours(out.writes, materials=out.materials)  # made ones from now on
        cells = _apply_cells(out.new, palette_record=palette_record)
        if document.selection is not None:
            document.selection.prune(grid.visible)
        refresh_views(serial=True, hover=True)
        self.report(*out.report(len(cells)))
        return {"FINISHED"}


@dataclass
class Outcome:
    """What a run would do, before anything is written: the new voxels and the
    palette slots its own colours go to, or the error that stops it."""

    name: str = ""
    new: np.ndarray | None = None
    writes: dict = field(default_factory=dict)
    error: str = ""
    stopped: int = 0
    mask_note: str = ""
    hint: str = ""
    colours: list = field(default_factory=list)  # the colours the run puts in, before your changes (recolour.shown)
    full: bool = False  # the error is a full palette: merging into the nearest colours would run
    materials: dict = field(default_factory=dict)  # slot -> the material of a new colour (a shader's glass)

    def report(self, changed: int) -> tuple[set, str]:
        """The status line once `changed` voxels were written."""
        if self.stopped:
            return ({"WARNING"}, f"{self.name}: {changed} voxels changed; its loops ran too long "
                                 f"and were stopped in {self.stopped} cells, so the result may be incomplete")
        if not changed:
            why = "Every voxel it would change is masked" if self.mask_note else self.hint
            return ({"INFO"}, f"{self.name}: 0 voxels changed. {why}")
        return ({"INFO"}, f"{self.name}: {changed} voxels changed{_plan_note(self.writes)}{self.mask_note}")


def compute(name: str, raw_args, *, colours_rgb: str = "", colour: int, colour_count: int = 1,
            iterations: int = 1, where: shader_scope.Where | None = None,
            voxels: np.ndarray | None = None, selection=CURRENT, mirror=None, recolour: str = "",
            merge: bool = False) -> Outcome:
    """Run shader `name` on the active model and say what it would change,
    writing nothing: rok.xs_run writes the outcome as one undo step, the
    settings panel shows it live. `where` limits it (by default the
    selection, when there is one); the mask holds. A live shader
    (live_shaders.py) passes the model's `voxels` without its output, and the
    `selection` mask and `mirror` it was made with. `recolour` is your
    change to the colours the run puts in (rok.core.recolour)."""
    found = shader_by_name(name)
    if found is None:
        return Outcome(name, error=f"No shader named {name}")
    label = xs.label(found.name)
    if colours_rgb and found.colours is not None:
        # Your colours instead of the shader's own (the panel's colour row).
        import dataclasses

        found = dataclasses.replace(found, colours=found.colours.with_rgb(xs.parse_hex_list(colours_rgb)))
    args = []
    for k, arg in enumerate(found.args):
        value = min(max(float(raw_args[k]), arg.low), arg.high)
        args.append(round(value, arg.precision) if arg.precision >= 0 else value)
    vis = document.grid.visible if voxels is None else np.asarray(voxels, dtype=np.uint8)
    wm = CONTEXT.prefs
    info = {}
    scene_palette = document.scene.palette
    # The shader runs on a palette of its own (rok.core.shader_palette); its
    # result comes back to your colours, new ones on free slots.
    try:
        sp = shader_palette_for(found, vis, colour, scene_palette, colour_count)
    except shader_palette.NoRoom as exc:
        return Outcome(label, error=f"{label}: {exc}; the model needs fewer colours first")
    base = sp.colour
    voxels = sp.to_run(vis)
    own = _own_box(found, where, vis.shape)
    try:
        new = run(
            found,
            voxels if own is None else np.ascontiguousarray(voxels[own]),
            colour=base,
            palette=sp.palette,
            args=args,
            iterations=iterations,
            colours=sp.run,
            mirror=tuple(getattr(wm, "rok_mirror", (False, False, False))) if mirror is None else tuple(mirror),
            info=info,
        )
    except (ModelTooLarge, NotRunnable, TooSlow, GpuGaveUp) as exc:
        return Outcome(label, error=f"{label}: {exc}")
    except ShaderError:
        return Outcome(label, error=f"{label} does not compile here (the console has the details)")
    if own is not None:
        whole = np.array(voxels, dtype=np.uint8, copy=True)
        whole[own] = new
        new = whole
    try:
        new, new_colours = sp.back(new, store_slots(merge))
    except PaletteFull as exc:
        return Outcome(label, error=f"{label}: {exc}; free some colours first", full=True)
    base = colour
    if xs.is_brush(found):
        new = np.where((vis > 0) & (new > 0), new, vis).astype(np.uint8)
    mask = limit_of(where, vis, selection)
    if mask is not None:
        new = np.where(mask, new, vis).astype(np.uint8)
        if where is not None and where.feather:
            walls = None
            if where.scope == shader_scope.VOLUME and where.volume is not None:
                walls = shader_scope.volume_mask(vis.shape, where.volume, where.shape)
            new = shader_scope.feathered(vis, new, mask, where.feather, walls)
    if xs.JOINED in found.flags:
        new = morph.join_loose(vis, new)
    masked = select_ops.mask_of()
    mask_note = ""
    if masked is not None:
        kept = morph.keep_masked(vis, new, masked)
        if (kept != new).any():
            mask_note = "; masked voxels stayed as they are"
        new = kept
    before_scope = vis if mask is None else vis[mask]
    after_scope = new if mask is None else new[mask]
    if before_scope.any() and not after_scope.any():
        # Nobody runs a shader for an empty grid (Clear All does that): keep the model.
        return Outcome(label, error=f"{label} would delete every voxel in the {_scope_name(mask)}, so Rok "
                                    f"kept it. {_empty_hint(found, base, before_scope)}")
    # The colours the run puts in, and your changes to them.
    after_palette = np.array(scene_palette, copy=True)
    for slot, value in new_colours.items():
        after_palette[slot] = value
    shown = recolour_core.shown(vis, new, after_palette, () if found.colours is None else found.colours.rgb)
    extra = {}
    mapping = recolour_core.parse(recolour)
    if mapping:
        try:
            new, extra = recolour_core.apply(vis, new, after_palette, mapping, document.scene.used_counts(),
                                             keep=document.pinned(), materials=document.scene.materials)
        except ValueError as exc:
            return Outcome(label, error=f"{label}: {exc}; free some colours first")
    # Only the colours the result uses reach the palette; the rest disappear.
    present = np.bincount(new.ravel(), minlength=256) > 0
    writes = {slot: value for slot, value in new_colours.items() if present[slot]}
    writes.update(extra)
    return Outcome(label, new, writes, stopped=int(info.get("stopped") or 0), mask_note=mask_note,
                   materials={s: m for s, m in sp.new_materials.items() if s in writes},
                   hint=_no_change_hint(found, base, colour_count, mask, _on_selection(where, selection)),
                   colours=shown)


@register
class XsFolderAction(Action):
    command_id = "rok.xs_folder"
    title = "Open Shader Folder"
    tooltip = "Open your shader folder: .txt shaders put here show in the shader gallery"
    def execute(self, context):
        folder = user_shader_folder()
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif os.name == "nt":
            os.startfile(str(folder))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        return {"FINISHED"}


def shader_commands():
    """Every shader as a Cmd+K entry."""
    from rok.core.chrome import Command

    return tuple(
        Command("xs_run", f"Shader: {xs.label(name)}", "", "shader xs", (("shader", name),))
        for name in shader_files()
    )


def shaders_menu(app) -> list:
    """The Shaders menu, which opens the gallery (Cmd+K opens it too)."""
    return [("Shaders", [
        ("rok.shader_gallery", "Shader Gallery…", "CmdOrCtrl+Shift+K", -1),
        None,
        ("rok.xs_folder", "Open Your Shader Folder", "", -1),
    ])]


def _install_menu() -> None:
    from .handlers import registry

    registry.menus.append(shaders_menu)


_install_menu()
