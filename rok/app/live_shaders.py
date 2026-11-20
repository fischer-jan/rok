"""Live shaders in the app: make, rerun, switch, bake and remove them.

Run with Live in the shader panel (rok.xs_run live=True) keeps the shader
on the model. After every edit of the model's voxels, the undo stack's
push hook (`after_push`) takes the model's live shaders back and runs them
again on the new voxels, in the same undo step as the edit. Undo and redo
do not rerun: the step already holds the voxels and the shaders as they
were. rok.core.live_shader has the rules.

The Layers flyout lists a model's live shaders under it: the eye switches
one off and on, a click opens it in the shader panel (Update, Bake, Remove).
"""

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np

from rok.core import link as link_core
from rok.core import live_shader as core
from rok.core import morph, xs
from rok.core.undo import CompoundRecord, ModelRecord, PaletteRecord, StrokeRecorder, UndoStack

from . import object as voxel_object
from .object import document
from .opkit import CONTEXT, IntProp, Action, register
from .ops_common import can_edit_model, refresh_views


def shaders_of(uid: int | None = None) -> list[core.LiveShader]:
    """The live shaders of a model (the active one by default), in order."""
    scene = document.scene
    if scene is None:
        return []
    return list(scene.live.get(document.active_uid if uid is None else uid, ()))


def find(uid: int, live_id: int) -> core.LiveShader | None:
    return next((s for s in shaders_of(uid) if s.id == live_id), None)


def owner(live_id: int) -> int | None:
    """The uid of the model that has live shader `live_id`."""
    scene = document.scene
    if scene is None:
        return None
    return next((uid for uid, shaders in scene.live.items() if any(s.id == live_id for s in shaders)), None)


def label(live: core.LiveShader) -> str:
    return xs.label(live.shader)


@contextlib.contextmanager
def _active(uid: int):
    """Run with `uid` as the active model: the shader code reads the active one."""
    was = document.active_uid
    document.active_uid = uid
    try:
        yield
    finally:
        document.active_uid = was


def _run(uid: int):
    """A core.Run for model `uid`: runs a live shader on given voxels and
    writes the colours it brings into the palette."""

    def run(live: core.LiveShader, voxels: np.ndarray):
        from . import xs_ops

        s = live.settings
        raw = [float(s.get(f"a{k}", 0.0)) for k in range(xs.MAX_ARGS)]
        where = xs_ops.where_of(s.get("scope", ""), s.get("only_on", ""), s.get("only_colours", ""),
                                s.get("room", 0), s.get("feather", 0), s.get("volume", ""), s.get("volume_shape", ""))
        with _active(uid):
            out = xs_ops.compute(live.shader, raw, colours_rgb=s.get("colours_rgb", ""),
                                 colour=int(s.get("colour", document.colour)),
                                 colour_count=int(s.get("colour_count", 1)), iterations=int(s.get("iterations", 1)),
                                 where=where, voxels=voxels, selection=live.selection, mirror=live.mirror,
                                 recolour=s.get("recolour", ""))
        if out.error:
            return out.error
        voxel_object.write_colours(out.writes, materials=out.materials)
        return out.new

    return run


def restack(uid: int, shaders: list[core.LiveShader], before: list[core.LiveShader] | None = None) -> list:
    """Take model `uid`'s shaders back and run `shaders` (the new list) on its
    voxels. Writes the voxels and the list; returns the undo parts. `before`
    is the list the undo step puts back (default: the list as it is)."""
    scene = document.scene
    model = scene.by_uid(uid)
    old = tuple(scene.live.get(uid, ()) if before is None else before)
    palette_before = scene.palette.copy()
    grid = model.grid
    new, done = core.restack(grid.visible, shaders, _run(uid))
    cells, values = morph.changed_cells(grid.visible, new)
    recorder = StrokeRecorder(grid)
    if len(cells):
        recorder.set_many(cells, values)
    stroke = recorder.finish()
    parts: list = []
    if not np.array_equal(palette_before, scene.palette):
        parts.append(PaletteRecord(None, palette_before, scene.palette.copy(), document.palette_session))
        voxel_object.update_palette()
    if stroke is not None:
        parts.append(ModelRecord(uid, stroke))
    core._put(scene, uid, done)
    parts.append(core.LiveRecord(uid, old, tuple(done)))
    return parts


def _has_live_record(record) -> bool:
    if isinstance(record, core.LiveRecord):
        return True
    return isinstance(record, CompoundRecord) and any(_has_live_record(p) for p in record.parts)


def after_push(record):
    """The undo stack's push hook: rerun the live shaders of every model the
    record changed, as part of the same step. Returns the amended record or None."""
    scene = document.scene
    if scene is None or not scene.live or _has_live_record(record):
        return None
    parts: list = []
    baked = []
    for uid, _d in link_core.voxel_changes(record, False):
        shaders = scene.live.get(uid)
        if not shaders:
            continue
        try:
            model = scene.by_uid(uid)
        except ValueError:
            continue
        if not all(core.fits(s, model.grid.size) for s in shaders):
            # A new size: the shaders' cells no longer line up. Their voxels stay.
            parts.append(core.LiveRecord(uid, tuple(shaders), ()))
            scene.live.pop(uid, None)
            baked.append(model.name or "the model")
            continue
        parts += restack(uid, shaders)
    if not parts:
        return None
    if baked:
        _say(f"Live shaders baked: {', '.join(baked)} changed size")
    voxel_object.flush_dirty()
    inner = record.parts if isinstance(record, CompoundRecord) else (record,)
    return CompoundRecord(tuple(inner) + tuple(parts))


UndoStack.after_push = after_push


def _say(text: str) -> None:
    app = CONTEXT.app
    if app is not None:
        app.say(text, 4.0)


def _push(parts: list) -> None:
    if not parts:
        return
    document.undo.push(CompoundRecord(tuple(parts)))
    document.modified = True
    voxel_object.flush_dirty()
    refresh_views(serial=True, hover=True)


def _next_id(scene) -> int:
    return 1 + max((s.id for shaders in scene.live.values() for s in shaders), default=0)


def add(op, shader: str, settings: dict) -> set:
    """rok.xs_run with Live: keep the shader on the active model and run it."""
    from . import xs_ops

    scene = document.scene
    uid = document.active_uid
    selection = xs_ops.selection_mask()
    mirror = tuple(bool(v) for v in getattr(CONTEXT.prefs, "rok_mirror", (False, False, False)))
    live = core.LiveShader(_next_id(scene), shader, dict(settings),
                           None if selection is None else selection.copy(), mirror)
    shaders = shaders_of(uid) + [live]
    parts = restack(uid, shaders)
    done = find(uid, live.id)
    if done is not None and done.error:
        # It cannot run: put everything back and say why.
        for part in reversed(parts):
            part.undo(scene)
        voxel_object.flush_dirty()
        op.report({"ERROR"}, done.error)
        return {"CANCELLED"}
    _push(parts)
    changed = 0 if done is None or done.diff is None else len(done.diff)
    op.report({"INFO"}, f"{label(live)} is live: {changed} voxels changed. It runs again after each edit; "
                        f"bake it in the Layers list")
    return {"FINISHED"}


def settings_from(kwargs: dict, colour: int) -> dict:
    """A live shader's settings from rok.xs_run's keyword arguments (the
    shader panel's), with every default filled in."""
    out = {f"a{k}": 0.0 for k in range(xs.MAX_ARGS)}
    out.update(colour=int(colour), colour_count=1, iterations=1, colours_rgb="", recolour="", scope="", only_on="",
               only_colours="", room=0, volume="", volume_shape="", feather=0)
    out.update(kwargs)
    return out


def update(uid: int, live_id: int, settings: dict) -> core.LiveShader | None:
    """New settings for a live shader (the shader panel's Update)."""
    shaders = [dataclasses.replace(s, settings=dict(settings)) if s.id == live_id else s for s in shaders_of(uid)]
    _push(restack(uid, shaders))
    return find(uid, live_id)


def set_on(uid: int, live_id: int, on: bool) -> None:
    shaders = [dataclasses.replace(s, on=on) if s.id == live_id else s for s in shaders_of(uid)]
    _push(restack(uid, shaders))


def remove(uid: int, live_id: int) -> None:
    """Take the shader's voxels back and forget it."""
    before = shaders_of(uid)
    off = [dataclasses.replace(s, on=False) if s.id == live_id else s for s in before]
    parts = restack(uid, off, before)
    rest = [s for s in shaders_of(uid) if s.id != live_id]
    core._put(document.scene, uid, rest)
    parts[-1] = core.LiveRecord(uid, tuple(before), tuple(rest))
    _push(parts)


def bake(uid: int, live_id: int | None = None) -> int:
    """Keep the shader's voxels as they are and forget it (None: every live
    shader of the model). Returns how many were baked."""
    before = shaders_of(uid)
    rest = [] if live_id is None else [s for s in before if s.id != live_id]
    if len(rest) == len(before):
        return 0
    core._put(document.scene, uid, rest)
    _push([core.LiveRecord(uid, tuple(before), tuple(rest))])
    return len(before) - len(rest)


# --- operators ------------------------------------------------------------------------


class _LiveOp(Action):
    uid = IntProp(name="Model", description="The model's uid; 0 is the active model", default=0)
    live = IntProp(name="Live Shader", description="The live shader's id", default=0)

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def _target(self):
        uid = int(self.uid) or document.active_uid
        live = find(uid, int(self.live))
        if live is None:
            self.report({"ERROR"}, "No such live shader")
        return uid, live


@register
class LiveToggleAction(_LiveOp):
    command_id = "rok.live_toggle"
    title = "Switch Live Shader"
    tooltip = "Switch a live shader off (its voxels go) or on again"

    def execute(self, context):
        uid, live = self._target()
        if live is None:
            return {"CANCELLED"}
        set_on(uid, live.id, not live.on)
        return {"FINISHED"}


@register
class LiveRemoveAction(_LiveOp):
    command_id = "rok.live_remove"
    title = "Remove Live Shader"
    tooltip = "Take a live shader's voxels back and remove it"

    def execute(self, context):
        uid, live = self._target()
        if live is None:
            return {"CANCELLED"}
        remove(uid, live.id)
        self.report({"INFO"}, f"{label(live)} removed")
        return {"FINISHED"}


@register
class LiveBakeAction(_LiveOp):
    command_id = "rok.live_bake"
    title = "Bake Live Shader"
    tooltip = "Keep a live shader's voxels as they are and stop running it (live 0: every one on the model)"

    def execute(self, context):
        uid = int(self.uid) or document.active_uid
        if int(self.live):
            uid, live = self._target()
            if live is None:
                return {"CANCELLED"}
        n = bake(uid, int(self.live) or None)
        if not n:
            self.report({"INFO"}, "No live shaders to bake")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Baked {n} live shader{'s' if n != 1 else ''}")
        return {"FINISHED"}
