"""Animate mode: the third cell of the rail's switch.

Building and posing are different jobs, so Animate mode is a place of its
own. Tab steps through Model, Scene and Animate, Shift+Tab back.
In Animate mode:

- every part of the scene shows, unfaded, posed as the clip has it at the
  play head. With Voxel look (on by default) a turned part shows as real
  voxels, turned from its rest voxels each frame (rok/core/voxel_look.py);
  off, it turns smoothly. Your voxels are never touched: the host shows a
  stand-in grid (GpuSync's `poses`).
- a click picks a part, as in the scene; the dock along the bottom shows
  the clip, and K keys the picked part's pose at the play head.
- out of Animate mode you see and edit the rest pose.

Undo covers every key edit (AnimRecord) and a pivot change (PivotRecord).
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from rok.core import anim, voxel_look
from rok.core.chrome import LEVEL_BRICK, LEVEL_MODEL, LEVEL_SCENE
from rok.core.undo import AnimRecord, ClipsRecord, CompoundRecord, FramesRecord

from . import object as voxel_object
from .object import document
from .opkit import CONTEXT, EnumProp, IntProp, Action, StringProp, ops, register
from .ops_common import _late, commit, redraw
from .state import state

ANIMATE = "ANIMATE"
ORDER = (LEVEL_MODEL, LEVEL_SCENE, ANIMATE)  # the rail's switch, top to bottom
PLAY_STEP = 1.0 / 60.0  # how often a playing clip looks at the clock


def clip() -> anim.Clip | None:
    scene = document.scene
    return None if scene is None else scene.timeline


def current() -> str:
    """The cell of the switch that is lit: Brick mode lights Model."""
    if state.animating:
        return ANIMATE
    return LEVEL_MODEL if state.level == LEVEL_BRICK else state.level


# --- going in and out -----------------------------------------------------------------


def enter() -> bool:
    """Go to Animate mode. True when it changed. It works on the scene: every
    part shows, and a click picks one."""
    if state.animating or document.scene is None or document.model is None:
        return False
    if state.level != LEVEL_SCENE:
        _late("level_ops", "set_level")(LEVEL_SCENE)
    state.animating = True
    state.anim_since = time.monotonic()
    state.chrome_open = None
    state.hover = None
    state.preview = None
    state.scene_tool = "PICK"
    _changed()
    return True


def leave() -> bool:
    """Back to building: the rest pose shows again. True when it changed."""
    if not state.animating:
        return False
    stop()
    state.animating = False
    state.anim_paint = None
    state.anim_since = time.monotonic()
    state.chrome_open = None
    _changed()
    return True


def go(target: str) -> None:
    """To Model, Scene, Animate or Brick, as the rail's switch does."""
    if target == ANIMATE:
        enter()
        return
    leave()
    if target == LEVEL_SCENE:
        _late("level_ops", "set_level")(LEVEL_SCENE)
    elif state.level == LEVEL_SCENE:
        _late("level_ops", "step_into_picked")()
        if target == LEVEL_BRICK:
            _late("level_ops", "set_level")(LEVEL_BRICK)
    else:
        _late("level_ops", "set_level")(target)
    voxel_object.apply_fade()
    redraw()


def cycle(step: int) -> str:
    """Tab (+1) and Shift+Tab (-1): the next place in the rail's order."""
    target = ORDER[(ORDER.index(current()) + step) % len(ORDER)]
    go(target)
    return target


def _changed() -> None:
    voxel_object.apply_fade()
    _late("gizmo_ops", "refresh")()
    _late("chrome", "start_level_fade")()
    voxel_object.redraw_3d_views()


# --- the play head --------------------------------------------------------------------


def frame() -> int:
    return int(state.anim_frame)


def set_frame(value: int) -> None:
    """Move the play head; it stays inside the clip."""
    c = clip()
    if c is None:
        return
    state.anim_frame = int(min(max(int(value), 0), max(0, c.length - 1)))
    _late("gizmo_ops", "refresh")()  # the gizmo stays on the posed part
    voxel_object.redraw_3d_views()


_play: dict = {}  # while playing: "t0" (clock), "f0" (frame at start), "timer"


def playing() -> bool:
    return bool(state.playing)


def play_frame(now: float) -> int:
    """The frame a playing clip shows at clock time `now`, by its loop mode."""
    c = clip()
    t = _play["f0"] + (now - _play["t0"]) * c.fps
    return int(c.time_at(t))


def stop() -> None:
    if not state.playing:
        return
    state.playing = False
    timer = _play.pop("timer", None)
    if timer is not None:
        CONTEXT.prefs.remove_timer(timer)
    _play.clear()
    voxel_object.redraw_3d_views()


# --- what the view shows ----------------------------------------------------------------

_POSED: OrderedDict = OrderedDict()  # pose key -> (voxels, origin): the Voxel look's cache
_POSED_MAX = 256


@dataclass(frozen=True)
class Shown:
    """What the view shows for the parts at the play head: GpuSync's poses,
    smooth turns, and the parts a Show key hides."""

    poses: dict
    turns: dict
    hidden: frozenset
    palette: object = None  # (256, 4) uint8 while a colour track changes the palette at the play head


_last_shown: Shown | None = None


def box_of(model):
    """(origin delta, size, matrix) of the box round a posed part as the view
    last showed it, or None at rest."""
    sh = _last_shown
    if sh is None or not state.animating:
        return None
    if model.uid in sh.poses:
        _key, voxels, place = sh.poses[model.uid]
        delta = tuple(int(p) - int(q) for p, q in zip(place, model.position))
        size = tuple(model.grid.size) if voxels is None else tuple(voxels.shape)
        return delta, size, sh.turns.get(model.uid)
    if model.uid in sh.turns:
        return (0, 0, 0), tuple(model.grid.size), sh.turns[model.uid]
    return None


def _posed(model, voxels: np.ndarray, pose: anim.Pose, source: int):
    # The scene's identity is part of the key: uids and the stroke serial start
    # again with a new file, and a grid of the old one showed for the new.
    key = (id(document.scene), model.uid, source, document.stroke_serial, tuple(np.round(pose.linear, 6).ravel()),
           tuple(np.round(pose.pivot, 6)), tuple(np.rint(pose.offset).astype(int)))
    hit = _POSED.get(key)
    if hit is None:
        try:
            hit = voxel_look.posed(voxels, pose.linear, pose.pivot, pose.offset)
        except ValueError:
            hit = None  # too big turned: it shows smooth
        _POSED[key] = hit
        while len(_POSED) > _POSED_MAX:
            _POSED.popitem(last=False)
    else:
        _POSED.move_to_end(key)
    return key, hit


def shown() -> Shown:
    """The parts at the play head, for the view. Empty out of Animate mode:
    building shows the rest pose."""
    scene = document.scene
    c = clip()
    global _last_shown
    if not state.animating or scene is None or c is None or (not c.tracks and _pose_drag is None):
        voxel_object.anim_hidden = frozenset()
        _last_shown = None
        return Shown({}, {}, frozenset())
    _last_shown = shown_at(c, frame(), dragging=_pose_drag is not None)
    voxel_object.anim_hidden = _last_shown.hidden
    return _last_shown


def shown_at(c: anim.Clip, t: float, dragging: bool = False) -> Shown:
    """The parts of clip `c` at time `t` (frames, any number), as GpuSync
    takes them; the exports of pictures step through a clip with it.
    `dragging` puts the gizmo drag's pose on the dragged part."""
    scene = document.scene
    fr = anim.evaluate(c, scene, t)
    poses: dict = {}
    turns: dict = {}
    for model in scene.models:
        source = fr.state.get(model.uid, 0)
        voxels = model.grid.visible
        if 0 < source <= len(model.states):
            voxels = model.states[source - 1]
        pose = fr.poses.get(model.uid)
        dragged = drag_pose(model, c.track_of(model.uid), c.time_at(t)) if dragging else None
        if dragged is not None:
            pose = dragged
        if pose is None or pose.is_rest():
            if source:
                poses[model.uid] = ((model.uid, "frame", source, document.stroke_serial), voxels, model.position)
            continue
        if c.voxel_look:
            key, hit = _posed(model, voxels, pose, source)
            if hit is not None:
                out, origin = hit
                place = tuple(int(p) + int(o) for p, o in zip(model.position, origin))
                if out is voxels and not source:
                    poses[model.uid] = (None, None, place)  # a move only: its own voxels
                else:
                    poses[model.uid] = (key, out, place)
                continue
        turns[model.uid] = pose.matrix()
        if source:
            poses[model.uid] = ((model.uid, "frame", source, document.stroke_serial), voxels, model.position)
    hidden = frozenset(uid for uid, on in fr.visible.items() if not on)
    return Shown(poses, turns, hidden, fr.palette)


# --- undo ---------------------------------------------------------------------------------


@dataclass
class PivotRecord:
    """A part's pivot before and after (doubled grid units, None: its middle)."""

    uid: int
    before: tuple | None
    after: tuple | None

    def undo(self, scene) -> None:
        scene.by_uid(self.uid).pivot2 = self.before

    def redo(self, scene) -> None:
        scene.by_uid(self.uid).pivot2 = self.after

    @property
    def nbytes(self) -> int:
        return 64


@dataclass
class ParentRecord:
    """A part's parent before and after (a uid, None: a root)."""

    uid: int
    before: int | None
    after: int | None

    def undo(self, scene) -> None:
        scene.by_uid(self.uid).parent = self.before

    def redo(self, scene) -> None:
        scene.by_uid(self.uid).parent = self.after

    @property
    def nbytes(self) -> int:
        return 32


def edit(change, label: str = "") -> bool:
    """Run `change(clip)` as one undo step. True when the clip changed."""
    c = clip()
    if c is None:
        return False
    before = copy.deepcopy(c)
    change(c)
    if before == c:
        return False
    global _clip_version
    _clip_version += 1
    commit(AnimRecord(before, c))
    voxel_object.redraw_3d_views()
    return True


_frame_cache: dict = {}  # (uid, play head, clip version, frames) -> the frame shown
_clip_version = 0  # bumps on every clip edit: shown_frame's cache key


def shown_frame(model) -> int:
    """Which of the part's frames the play head shows: 0 for its own grid."""
    c = clip()
    if c is None or not model.states or document.scene is None:
        return 0
    key = (model.uid, frame(), _clip_version, len(model.states), id(c))
    found = _frame_cache.get(key)
    if found is None:
        track = c.track_of(model.uid)
        found = 0
        if track is not None and track.keys(anim.STATE):
            found = int(anim.value_at(track.properties[anim.STATE], c.time_at(frame())))
        found = max(0, min(found, len(model.states)))
        _frame_cache.clear()
        _frame_cache[key] = found
    return found


def edit_clips(change) -> bool:
    """Run `change(scene)` on the list of clips as one undo step. True when it changed."""
    scene = document.scene
    if scene is None:
        return False
    before, before_active = copy.deepcopy(scene.clips), scene.active_clip
    change(scene)
    if before == scene.clips and before_active == scene.active_clip:
        return False
    commit(ClipsRecord(before, before_active, scene.clips, scene.active_clip))
    _clip_changed()
    return True


def _clip_changed() -> None:
    """Another clip shows: the play head stays inside it, no key stays picked."""
    stop()
    set_frame(frame())
    _late("dock", "clear_picked")()
    voxel_object.redraw_3d_views()


def picked_track(c: anim.Clip, create: bool = True) -> anim.Track | None:
    """The picked part's track, made on first use."""
    model = document.model
    if model is None:
        return None
    track = c.track_of(model.uid)
    if track is None and create:
        index = document.scene.models.index(model)
        track = anim.Track(anim.Target(anim.PART, model.uid), name=model.display_name(index))
        c.tracks.append(track)
    return track


def part_values(track: anim.Track | None, t: float) -> dict:
    """The picked part's channel values at t."""
    return {name: anim.channel(track, name, t) for name in anim.MOVE + anim.TURN}


# --- operators ------------------------------------------------------------------------------


class _AnimOp(Action):
    @classmethod
    def poll(cls, context):
        return state.animating and document.model is not None


@register
class AnimateAction(Action):
    command_id = "rok.animate"
    title = "Animate"
    tooltip = "Go to Animate mode, where the gizmo poses parts and writes keys, or back to building"
    action = EnumProp(items=(("ENTER", "Animate", ""), ("LEAVE", "Back to Building", ""),
                                 ("TOGGLE", "Animate or Build", "")), default="TOGGLE")

    @classmethod
    def poll(cls, context):
        return document.model is not None

    def execute(self, context):
        if self.action == "ENTER" or (self.action == "TOGGLE" and not state.animating):
            enter()
        else:
            go(LEVEL_MODEL)
        return {"FINISHED"}


@register
class LevelCycleAction(Action):
    command_id = "rok.level_cycle"
    title = "Model, Scene or Animate"
    tooltip = "Tab: the next of Model, Scene and Animate; Shift+Tab: the one before"
    step = IntProp(default=1, min=-1, max=1)

    @classmethod
    def poll(cls, context):
        return document.model is not None and not state.stroke_active

    def execute(self, context):
        target = ORDER[(ORDER.index(current()) + (1 if int(self.step) >= 0 else -1)) % len(ORDER)]
        if target == LEVEL_MODEL and state.level == LEVEL_SCENE and document.scene.is_hidden(document.model):
            self.report({"WARNING"}, "The picked model is hidden: show its layer to step in")
            return {"CANCELLED"}
        cycle(1 if int(self.step) >= 0 else -1)
        return {"FINISHED"}


@register
class PlayAction(_AnimOp):
    command_id = "rok.play"
    title = "Play or Stop"
    tooltip = "Play the clip from the play head, or stop it (Shift+Space)"

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        if state.playing:
            stop()
            return {"FINISHED"}
        c = clip()
        if c is None or c.length < 2:
            return {"CANCELLED"}
        start = frame()
        if c.loop_mode == anim.ONCE and start >= c.length - 1:
            start = 0
        state.playing = True
        _play.update(t0=time.monotonic(), f0=float(start), timer=context.prefs.add_timer(PLAY_STEP))
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        if not state.playing or not state.animating:
            stop()
            return {"FINISHED"}
        if event.type == "TIMER":
            c = clip()
            now = time.monotonic()
            f = play_frame(now)
            if c.loop_mode == anim.ONCE and _play["f0"] + (now - _play["t0"]) * c.fps >= c.length - 1:
                set_frame(c.length - 1)
                stop()
                return {"FINISHED"}
            if f != frame():
                set_frame(f)
            return {"PASS"}
        if event.type == "ESCAPE" and event.value == "PRESS":
            stop()
            return {"FINISHED"}
        return {"PASS"}


@register
class FrameGoAction(_AnimOp):
    command_id = "rok.frame_go"
    title = "Go to Frame"
    tooltip = "Move the play head: to a frame, a step on or back, or to the next or previous key"

    frame = IntProp(name="Frame", default=0)
    step = IntProp(name="Step", description="Frames on (or back); 0 goes to Frame", default=0)
    to_key = IntProp(name="Key", description="1: the next key of the picked part, -1: the one before", default=0)

    def execute(self, context):
        stop()
        c = clip()
        if self.to_key:
            track = picked_track(c, create=False)
            times = [] if track is None else track.key_times()
            here = frame()
            ahead = [t for t in times if (t > here if self.to_key > 0 else t < here)]
            if not ahead:
                return {"CANCELLED"}
            set_frame(ahead[0] if self.to_key > 0 else ahead[-1])
        elif self.step:
            set_frame((frame() + int(self.step)) % max(1, c.length))
        else:
            set_frame(int(self.frame))
        return {"FINISHED"}


@register
class KeyPoseAction(_AnimOp):
    command_id = "rok.key_pose"
    title = "Key the Pose"
    tooltip = "Key every move and turn of the picked part at the play head (K)"

    def execute(self, context):
        t = frame()

        def change(c):
            track = picked_track(c)
            for name, value in part_values(track, t).items():
                anim.key_change(track, name, t, float(value), anim.rest_of(name))

        if not edit(change):
            return {"CANCELLED"}
        self.report({"INFO"}, f"Keyed {document.model.name or 'the part'} at frame {t}")
        return {"FINISHED"}


@register
class KeyDeleteAction(_AnimOp):
    command_id = "rok.key_delete"
    title = "Delete Keys"
    tooltip = "Delete the picked keys, or the picked part's keys at the play head"

    def execute(self, context):
        from . import dock

        if dock.picked_marker() is not None:
            return ops.rok.marker_delete()
        picked = dock.picked_keys()
        t = frame()

        def change(c):
            if picked:
                for uid, name, time_ in picked:
                    track = c.track_of(uid)
                    if track is not None:
                        anim.remove_keys(track, name, [time_])
            else:
                track = picked_track(c, create=False)
                if track is not None:
                    for name in list(track.properties):
                        anim.remove_keys(track, name, [t])
            c.tracks = [tr for tr in c.tracks if tr.properties or tr.target.kind != anim.PART]

        if not edit(change):
            return {"CANCELLED"}
        dock.clear_picked()
        return {"FINISHED"}


@register
class KeyEaseAction(_AnimOp):
    command_id = "rok.key_ease"
    title = "Ease"
    tooltip = "How the picked keys go on to the next key"

    ease = EnumProp(items=tuple((e, anim.EASE_LABELS[e], "") for e in anim.EASES), default=anim.SMOOTH)

    def execute(self, context):
        from . import dock

        picked = dock.picked_keys()
        if not picked:
            self.report({"INFO"}, "Pick keys in the dock first")
            return {"CANCELLED"}

        def change(c):
            for uid, name, time_ in picked:
                track = c.track_of(uid)
                if track is not None:
                    anim.set_ease(track, name, [time_], self.ease)

        edit(change)
        return {"FINISHED"}


@register
class AnimPresetAction(_AnimOp):
    command_id = "rok.anim_preset"
    title = "Motion"
    tooltip = "A ready-made motion for the picked part: it writes ordinary keys"

    name = StringProp(name="Motion", default="SPIN")

    def execute(self, context):
        name = str(self.name).upper()
        if name in anim.COLOUR_PRESETS:
            return self._colour(name)
        if name not in anim.PART_PRESETS:
            return {"CANCELLED"}
        scene = document.scene
        model = document.model
        others = [m for m in scene.models if m.uid != model.uid and not scene.is_hidden(m)]
        pivot_before = model.pivot2
        anchor2, axis = anim.default_anchor2(model, others)
        if pivot_before is None and name in ("SPIN", "SWING"):
            # A part that touches another turns about the face it touches, like a
            # propeller on its hub; a loose one about its middle.
            lo, _hi = model.box_world()
            model.pivot2 = tuple(int(a) - 2 * int(p) for a, p in zip(anchor2, lo))
        c = clip()
        before = copy.deepcopy(c)
        track = picked_track(c)
        track.axis = axis if name in ("SPIN", "SWING") else 2
        for channel_name, keys in anim.preset_keys(name, track, c, scene):
            track.properties[channel_name] = anim.Property(channel_name)
            for key in keys:
                anim.set_key(track, channel_name, key)
        parts = [AnimRecord(before, c)]
        if model.pivot2 != pivot_before:
            parts.append(PivotRecord(model.uid, pivot_before, model.pivot2))
        commit(CompoundRecord(tuple(parts)) if len(parts) > 1 else parts[0])
        if not state.playing:
            ops.rok.play("INVOKE")
        voxel_object.redraw_3d_views()
        self.report({"INFO"}, f"{anim.PRESET_LABELS[name]}: keys on {model.name or 'the part'}")
        return {"FINISHED"}


@register
class PartParentAction(Action):
    command_id = "rok.part_parent"
    title = "Parent"
    tooltip = "The part the picked part follows (drag its name onto another in the dock); -1 for none"
    parent = IntProp(name="Parent", description="The parent's uid, or -1 for no parent", default=-1)
    uid = IntProp(name="Part", description="The part's uid, or -1 for the picked one", default=-1)

    @classmethod
    def poll(cls, context):
        return document.scene is not None and document.model is not None

    def execute(self, context):
        scene = document.scene
        uid = int(self.uid) if int(self.uid) >= 0 else document.active_uid
        if not scene.has_uid(uid):
            return {"CANCELLED"}
        parent = None if int(self.parent) < 0 else int(self.parent)
        model = scene.by_uid(uid)
        before = model.parent
        if parent == before:
            return {"CANCELLED"}
        if not scene.set_parent(uid, parent):
            self.report({"WARNING"}, "A part cannot follow itself or one of its own parts")
            return {"CANCELLED"}
        commit(ParentRecord(uid, before, parent))
        _late("gizmo_ops", "refresh")()
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class FrameAddAction(_AnimOp):
    command_id = "rok.frame_add"
    title = "Add Frame"
    tooltip = "A new frame of the picked part: a copy of the one shown, shown from the play head on"

    @classmethod
    def poll(cls, context):
        # A brick model's cells are locked: it poses, but takes no frame and no paint.
        return super().poll(context) and document.grid is not None and document.grid.bricks is None

    def execute(self, context):
        global _clip_version
        model = document.model
        c = clip()
        t = frame()
        shown_now = shown_frame(model)
        before_states = list(model.states)
        before_clip = copy.deepcopy(c)
        index = model.add_frame(model.frame_grid(shown_now).voxels)
        track = picked_track(c)
        anim.key_change(track, anim.STATE, t, index, 0)
        _clip_version += 1
        commit(CompoundRecord((FramesRecord(model.uid, before_states, model.states), AnimRecord(before_clip, c))))
        _frame_cache.clear()
        voxel_object.redraw_3d_views()
        self.report({"INFO"}, f"Frame {index + 1} of {model.name or 'the part'}: paint it")
        return {"FINISHED"}


@register
class FrameDeleteAction(_AnimOp):
    command_id = "rok.frame_delete"
    title = "Delete Frame"
    tooltip = "Delete the picked part's frame shown at the play head (not its first)"

    @classmethod
    def poll(cls, context):
        # A brick model's cells are locked: it poses, but takes no frame and no paint.
        return super().poll(context) and document.grid is not None and document.grid.bricks is None

    def execute(self, context):
        global _clip_version
        model = document.model
        c = clip()
        index = shown_frame(model)
        if index <= 0:
            self.report({"INFO"}, "The first frame stays; move the play head to another")
            return {"CANCELLED"}
        before_states = list(model.states)
        before_clip = copy.deepcopy(c)
        model.remove_frame(index)
        track = c.track_of(model.uid)
        if track is not None and anim.STATE in track.properties:
            prop = track.properties[anim.STATE]
            keys = []
            for k in prop.keys:
                v = int(k.value)
                if v == index:
                    continue  # the keys that showed this frame go
                keys.append(anim.Key(k.time, v - 1 if v > index else v, k.ease))
            prop.keys = keys
            if not any(int(k.value) for k in keys):
                del track.properties[anim.STATE]
        _clip_version += 1
        commit(CompoundRecord((FramesRecord(model.uid, before_states, model.states), AnimRecord(before_clip, c))))
        _frame_cache.clear()
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class AnimPaintAction(_AnimOp):
    command_id = "rok.anim_paint"
    title = "Paint in Animate"
    tooltip = "Attach, Erase or Paint on the picked part's frame at the play head; POSE goes back to the gizmo"
    tool = StringProp(name="Tool", default="POSE")

    @classmethod
    def poll(cls, context):
        # A brick model's cells are locked: it poses, but takes no frame and no paint.
        return super().poll(context) and document.grid is not None and document.grid.bricks is None

    def execute(self, context):
        tool = str(self.tool).upper()
        state.anim_paint = None if tool not in ("ATTACH", "ERASE", "PAINT") or tool == state.anim_paint else tool
        state.hover = None
        _late("gizmo_ops", "refresh")()
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class OnionSkinAction(_AnimOp):
    command_id = "rok.onion_skin"
    title = "Onion Skin"
    tooltip = "Faint copies of each part's pose at the key before and the key after the play head"

    def execute(self, context):
        state.onion_skin = not state.onion_skin
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class ClipAddAction(_AnimOp):
    command_id = "rok.clip_add"
    title = "Add Clip"
    tooltip = "A new clip: the same parts at rest, no keys, the length and frame rate of this one"

    def execute(self, context):
        def change(scene):
            c = scene.timeline
            scene.clips.append(anim.Clip(name=scene.new_clip_name(), fps=c.fps, length=c.length,
                                         loop_mode=c.loop_mode, voxel_look=c.voxel_look))
            scene.active_clip = len(scene.clips) - 1

        edit_clips(change)
        return {"FINISHED"}


@register
class ClipDuplicateAction(_AnimOp):
    command_id = "rok.clip_duplicate"
    title = "Duplicate Clip"
    tooltip = "A copy of this clip, keys and all, next to it"

    def execute(self, context):
        def change(scene):
            c = copy.deepcopy(scene.timeline)
            c.name = f"{c.name} copy"
            scene.clips.insert(scene.active_clip + 1, c)
            scene.active_clip += 1

        edit_clips(change)
        return {"FINISHED"}


@register
class ClipDeleteAction(_AnimOp):
    command_id = "rok.clip_delete"
    title = "Delete Clip"
    tooltip = "Delete this clip; the last clip stays"

    def execute(self, context):
        if len(document.scene.clips) < 2:
            self.report({"INFO"}, "The last clip stays")
            return {"CANCELLED"}

        def change(scene):
            del scene.clips[scene.active_clip]
            scene.active_clip = min(scene.active_clip, len(scene.clips) - 1)

        edit_clips(change)
        return {"FINISHED"}


@register
class ClipPickAction(_AnimOp):
    command_id = "rok.clip_pick"
    title = "Show Clip"
    tooltip = "Show another clip in the dock (its tab)"
    index = IntProp(name="Clip", default=0, min=0)

    def execute(self, context):
        scene = document.scene
        if not 0 <= int(self.index) < len(scene.clips) or int(self.index) == scene.active_clip:
            return {"CANCELLED"}
        scene.active_clip = int(self.index)
        _clip_changed()
        return {"FINISHED"}


@register
class ClipRenameAction(_AnimOp):
    command_id = "rok.clip_rename"
    title = "Rename Clip"
    tooltip = "The clip's name, as its tab and the exports show it"

    name = StringProp(name="Name", default="")

    def invoke(self, context, event):
        self.name = clip().name
        return context.prefs.open_dialog(self, title="Clip name", confirm_text="Rename")

    def draw(self, context):
        self.layout.prop(self, "name")

    def execute(self, context):
        new = str(self.name).strip()
        if not new:
            return {"CANCELLED"}

        def change(scene):
            scene.timeline.name = new

        edit_clips(change)
        return {"FINISHED"}


def flow_slots(scene, first: int, count: int = 6) -> tuple[int, ...]:
    """The slots a Flow cycles through: the current colour and the next ones the
    scene uses (or the next indices when it uses none), `count` in all."""
    counts = np.zeros(256, dtype=np.int64)
    for model in scene.models:
        counts += np.bincount(model.grid.voxels.ravel(), minlength=256)
    used = [i for i in range(first + 1, 256) if counts[i] > 0]
    slots = [first] + used[: count - 1]
    nxt = first + 1
    while len(slots) < count and nxt < 256:
        if nxt not in slots:
            slots.append(nxt)
        nxt += 1
    return tuple(slots)


def colour_track(c: anim.Clip, slots: tuple, create: bool = True) -> anim.Track | None:
    """The colour track of exactly these slots, made on first use."""
    for track in c.tracks:
        if track.target.kind == anim.COLOUR and tuple(track.target.slots) == tuple(slots):
            return track
    if not create:
        return None
    label = "Colour " + ", ".join(str(s) for s in slots[:3]) + ("…" if len(slots) > 3 else "")
    track = anim.Track(anim.Target(anim.COLOUR, 0, slots=tuple(int(s) for s in slots)), name=label)
    c.tracks.append(track)
    return track


def _colour_preset(self, name: str):
    """Glow, Blink or Flow on the current colour (Flow: on it and the next ones)."""
    scene = document.scene
    c = clip()
    slot = int(document.colour)
    slots = flow_slots(scene, slot) if name == "FLOW" else (slot,)
    before = copy.deepcopy(c)
    track = colour_track(c, slots)
    if name == "BLINK":
        track.settings["skin_colour"] = list(anim.neighbour_colour(scene, slots))
    for channel_name, keys in anim.preset_keys(name, track, c, scene):
        track.properties[channel_name] = anim.Property(channel_name)
        for key in keys:
            anim.set_key(track, channel_name, key)
    global _clip_version
    _clip_version += 1
    commit(AnimRecord(before, c))
    if not state.playing:
        ops.rok.play("INVOKE")
    voxel_object.redraw_3d_views()
    self.report({"INFO"}, f"{anim.PRESET_LABELS[name]}: colour {slot}" + (f" and {len(slots) - 1} more" if len(slots) > 1 else ""))
    return {"FINISHED"}


AnimPresetAction._colour = _colour_preset  # the colour half of the Motion operator, defined after its helpers


@register
class ClipSettingsAction(_AnimOp):
    command_id = "rok.clip_settings"
    title = "Clip"
    tooltip = "The clip's loop mode and Voxel look"

    loop_mode = StringProp(name="Loop", description="LOOP, ONCE or PING_PONG; empty: the next one", default="")
    voxel_look = IntProp(name="Voxel look", description="1 on, 0 off, -1 unchanged", default=-1)
    root_motion = IntProp(name="Root motion", description="1 on, 0 off, -1 unchanged, 2 the other way", default=-1)

    def execute(self, context):
        def change(c):
            if self.root_motion in (0, 1, 2):
                c.root_motion = (not c.root_motion) if self.root_motion == 2 else bool(self.root_motion)
            elif self.voxel_look in (0, 1):
                c.voxel_look = bool(self.voxel_look)
            elif self.loop_mode:
                c.loop_mode = self.loop_mode
            else:
                modes = anim.LOOP_MODES
                c.loop_mode = modes[(modes.index(c.loop_mode) + 1) % len(modes)]

        edit(change)
        return {"FINISHED"}


# --- posing with the gizmo ---------------------------------------------------------------
#
# In Animate mode the picked part's gizmo poses it. While a drag runs, the part
# shows at the new pose (in Voxel look); the release writes one key per channel
# that changed, at the play head, as one undo step. Shift on the centre square
# moves the part's pivot instead (it belongs to the part, it is not keyed).

_pose_drag: dict | None = None  # "uid", "values" (the channels at the press), "result", "gesture"


def gizmo_begin(gesture=None) -> None:
    global _pose_drag
    model = document.model
    c = clip()
    if model is None or c is None:
        _pose_drag = None
        return
    stop()
    track = c.track_of(model.uid)
    _pose_drag = {"uid": model.uid, "values": part_values(track, frame()), "result": None, "gesture": gesture,
                  "pivot2": None}


def _turn_degrees(result) -> float:
    """The ring drag's angle, unwrapped: 270 is not -90 for a key."""
    gesture = (_pose_drag or {}).get("gesture")
    unwrapped = getattr(gesture, "degrees", None)
    if isinstance(unwrapped, (int, float)) and unwrapped != 0.0:
        return float(unwrapped)
    if result.degrees:
        return float(result.degrees)
    return 90.0 * int(result.turns)


def _posed_values(result) -> dict:
    values = dict(_pose_drag["values"])
    if result is None:
        return values
    from rok.core import gizmo

    if result.kind == gizmo.MOVE:
        for name, d in zip(anim.MOVE, result.offset):
            values[name] = float(values[name]) + int(d)
    elif result.kind == gizmo.ROTATE:
        name = anim.TURN[int(result.axis)]
        values[name] = float(values[name]) + _turn_degrees(result)
    return values


def gizmo_preview(result) -> None:
    """The drag changed: show the part at its new pose."""
    from rok.core import gizmo

    if _pose_drag is None:
        return
    if result.kind == gizmo.PIVOT:
        _pose_drag["pivot2"] = tuple(int(v) for v in result.pivot2)
        if state.gizmo is not None:
            state.gizmo.pivot2 = _pose_drag["pivot2"]
    else:
        _pose_drag["result"] = result
        if result.kind == gizmo.MOVE:
            state.model_drag_offset = tuple(int(v) for v in result.offset)
    voxel_object.redraw_3d_views()


def gizmo_end() -> None:
    """The drag is over (kept or not): the clip shows as it is."""
    global _pose_drag
    _pose_drag = None
    state.model_drag_offset = (0, 0, 0)
    voxel_object.redraw_3d_views()


def gizmo_commit(result) -> bool:
    """The release: keys for the channels the drag changed, or the new pivot."""
    from rok.core import gizmo

    drag = _pose_drag
    if drag is None:
        return False
    scene = document.scene
    model = scene.by_uid(drag["uid"])
    if drag.get("pivot2") is not None and drag["pivot2"] != model.pivot2:
        before = model.pivot2
        model.pivot2 = drag["pivot2"]
        commit(PivotRecord(model.uid, before, model.pivot2))
        return True
    if result is None or result.kind not in (gizmo.MOVE, gizmo.ROTATE):
        return False
    t = frame()
    values = _posed_values(result)
    changed = {n: v for n, v in values.items() if abs(float(v) - float(drag["values"][n])) > 1e-9}
    if not changed:
        return False

    def change(c):
        track = c.track_of(model.uid)
        if track is None:
            index = scene.models.index(model)
            track = anim.Track(anim.Target(anim.PART, model.uid), name=model.display_name(index))
            c.tracks.append(track)
        for name, value in changed.items():
            anim.key_change(track, name, t, float(value), anim.rest_of(name))

    return edit(change)


def drag_pose(model, track, t) -> anim.Pose | None:
    """The picked part's pose while a gizmo drag runs, else None."""
    if _pose_drag is None or _pose_drag["uid"] != model.uid or _pose_drag.get("result") is None:
        return None
    values = _posed_values(_pose_drag["result"])
    degrees = tuple(float(values[n]) for n in anim.TURN)
    percent = anim.channel(track, anim.SCALE, t)
    linear = anim.matrix_from_euler(degrees) @ np.diag([p / 100.0 for p in percent])
    offset = np.array([float(values[n]) for n in anim.MOVE])
    return anim.Pose(linear, anim.pivot_of(track, model), offset)


def gizmo_place(model) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """(pivot2, origin) of the picked part's gizmo: its pivot, where the pose has it."""
    c = clip()
    pivot2 = model.pivot2 if model.pivot2 is not None else tuple(int(v) for v in model.grid.size)
    origin = voxel_object.model_origin(model)
    if c is not None and document.scene is not None:
        # Where the pose at the play head puts the pivot: the part's own move, and
        # its parents' turns and moves (anim.with_parents), in whole cells.
        pose = anim.evaluate(c, document.scene, frame()).poses.get(model.uid)
        if pose is not None:
            pivot = np.array(pivot2, dtype=np.float64) / 2.0
            moved = (pose.matrix() @ np.append(pivot, 1.0))[:3] - pivot
            origin = tuple(float(o) + round(float(d)) for o, d in zip(origin, moved))
    return tuple(int(v) for v in pivot2), origin
