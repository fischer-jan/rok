"""The agent's animation and render tools: rok_animate and rok_render.

`AnimTools` is a part of `rok.agent.Agent`. The animation tool works on the
same data as Animate mode (rok/app/anim_mode.py):
parts are models of the scene, the active clip holds the keys, and a part's
voxel frames are SceneModel.states. Every change is one undo step. The tool
never enters Animate mode: the window keeps the mode the user left it in, and a
picture poses the parts for itself only (Agent._pose).

Frames are clip time, whole numbers from 0. A part's voxel frames are
counted apart: 0 is the part's own voxels, 1.. its extra ones.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from rok.core import anim

ANIMATE_DOS = ("status", "make_part", "empty_part", "parent", "pivot", "key", "delete_keys", "ease", "motion",
               "colour_key", "voxel_frame_add", "voxel_frame_set", "voxel_frame_delete", "clip", "clip_add",
               "clip_duplicate", "clip_delete", "marker", "marker_delete", "look", "export")
# The channel groups a call can name; the exact channel names work too.
CHANNEL_GROUPS = {"move": anim.MOVE, "turn": anim.TURN, "scale": (anim.SCALE,), "show": (anim.VISIBLE,),
                  "voxel_frame": (anim.STATE,), "colour": (anim.COLOUR_RGB,), "glow": (anim.GLOW,),
                  "slot": (anim.SLOT,)}
CHANNEL_NAMES = {anim.STATE: "voxel_frame", anim.VISIBLE: "show", anim.SCALE: "scale", anim.COLOUR_RGB: "colour",
                 anim.GLOW: "glow", anim.SLOT: "slot"}  # the names the tool takes; move_x and the like stay
MAX_STRIP = 8  # frames in one strip picture
EXPORT_KINDS = {".glb": "glTF", ".vox": ".vox with frames", ".png": "sprite sheet and its JSON",
                ".gif": "GIF", ".mp4": "MP4"}


def _ease(value) -> str | None:
    if value in (None, ""):
        return None
    e = str(value).upper()
    if e not in anim.EASES:
        from rok.agent import ToolError

        raise ToolError(f"ease must be one of {', '.join(x.lower() for x in anim.EASES)}; got {value!r}")
    return e


def _channels(names) -> tuple[str, ...] | None:
    """Channel names from groups (move, turn, ...) or exact names; None: all."""
    if not names:
        return None
    from rok.agent import ToolError

    out: list[str] = []
    every = set(anim.PROPERTIES_OF[anim.PART]) | set(anim.PROPERTIES_OF[anim.COLOUR])
    for n in [names] if isinstance(names, str) else names:
        key = str(n).lower()
        if key in CHANNEL_GROUPS:
            out += CHANNEL_GROUPS[key]
        elif str(n).upper() in every:
            out.append(str(n).upper())
        else:
            raise ToolError(f"no channel {n!r}; the channels are {', '.join(CHANNEL_GROUPS)} or exact names like move_z")
    return tuple(dict.fromkeys(out))


def _value_text(value) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value) + ")"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class AnimTools:
    """rok_animate and rok_render; mixed into rok.agent.Agent."""

    # --- helpers ------------------------------------------------------------------------

    def _anim(self):
        from rok.app import anim_mode

        return anim_mode

    def _clip_named(self, clip) -> None:
        """Make the clip named `clip` (or numbered, from 0) the active one."""
        from rok.agent import ToolError

        if clip in (None, ""):
            return
        scene = self.doc.scene
        names = [c.name for c in scene.clips]
        if isinstance(clip, int) or str(clip).isdigit():
            index = int(clip)
        elif clip in names:
            index = names.index(clip)
        else:
            raise ToolError(f"no clip {clip!r}; the clips are: {', '.join(names)}")
        if not 0 <= index < len(scene.clips):
            raise ToolError(f"clip number {index} does not exist; there are {len(scene.clips)} (from 0)")
        if index != scene.active_clip:
            scene.active_clip = index
            self._anim()._clip_changed()

    def _frame(self, frame) -> int:
        from rok.agent import ToolError

        t = 0 if frame is None else int(frame)
        if t < 0:
            raise ToolError("frame counts from 0")
        return t

    def _length_note(self, t: int) -> str:
        c = self._anim().clip()
        return "" if t < c.length else (f" Frame {t} is past the clip's end ({c.length} frames, 0 to {c.length - 1}): "
                                        "it does not play; make the clip longer with do=clip seconds=...")

    def _track_text(self, track: anim.Track, scene) -> list[str]:
        if track.target.kind == anim.PART:
            head = f"  {self._part_name(track.target.uid)}:"
        else:
            head = f"  colour {', '.join(str(s) for s in track.target.slots)}:"
        lines = [head]
        for name, prop in track.properties.items():
            keys = ", ".join(f"{k.time}={_value_text(k.value)}" + ("" if k.ease == anim.HOLD and name in anim.HOLD_ONLY
                                                                    else f" {k.ease.lower()}") for k in prop.keys)
            lines.append(f"    {CHANNEL_NAMES.get(name, name.lower())}: {keys}")
        return lines

    def _part_name(self, uid: int) -> str:
        scene = self.doc.scene
        if not scene.has_uid(uid):
            return f"(gone {uid})"
        return scene.by_uid(uid).display_name(scene.index_of(uid))

    def _anim_picture(self, frames, views=None, size=None):
        from rok.agent import Picture

        if self.feedback != "image" or self._batch_depth:
            return None
        frames = tuple(int(f) for f in frames)
        views = views or (("iso",) if len(frames) > 1 else ("top", "iso", "front", "right"))
        return Picture(tuple(views), tuple(size or self.size), frames=frames)

    def _spread(self, n: int = 4) -> tuple[int, ...]:
        """n frames spread over the active clip, first to last."""
        c = self._anim().clip()
        last = max(0, c.length - 1)
        return tuple(dict.fromkeys(int(round(last * k / max(1, n - 1))) for k in range(n)))

    # --- rok_animate --------------------------------------------------------------------

    def tool_animate(self, do: str = "status", part=None, frame=None, frames=None, move=None, turn=None, scale=None,
                     show=None, voxel_frame=None, ease=None, channels=None, name=None, parent=None, pivot=None,
                     axis=None, colour=None, hex=None, glow=None, seconds=None, fps=None, loop=None,
                     voxel_look=None, root_motion=None, clip=None, voxels=None, layers=None, corner=None,
                     legend=None, views=None, size=None, path=None, baked: bool = False, merge: bool = True,
                     directions: int = 8, cell_size: int = 128, view: str = "iso"):
        from rok.agent import Reply, ToolError

        do = str(do).lower()
        if do not in ANIMATE_DOS:
            raise ToolError(f"do must be one of {', '.join(ANIMATE_DOS)}")
        if self.doc.scene is None or self.doc.model is None:
            raise ToolError("no document is open; use rok_scene do=new or do=open")
        self._clip_named(clip)
        handler = getattr(self, "_anim_" + do)
        takes = handler.__code__.co_varnames[1:handler.__code__.co_argcount]
        result = handler(**{k: v for k, v in dict(
            part=part, frame=frame, frames=frames, move=move, turn=turn, scale=scale, show=show,
            voxel_frame=voxel_frame, ease=ease, channels=channels, name=name, parent=parent, pivot=pivot,
            axis=axis, colour=colour, hex=hex, glow=glow, seconds=seconds, fps=fps, loop=loop,
            voxel_look=voxel_look, root_motion=root_motion, voxels=voxels, layers=layers, corner=corner,
            legend=legend, views=views, size=size, path=path, baked=baked, merge=merge, directions=directions,
            cell_size=cell_size, view=view).items() if k in takes})
        return result if isinstance(result, Reply) else Reply(result)

    def _anim_status(self):
        from rok.agent import Reply

        scene = self.doc.scene
        c = self._anim().clip()
        lines = ["Clips (* is the active one; the others' keys rest):"]
        for i, cl in enumerate(scene.clips):
            lines.append(f"{'*' if i == scene.active_clip else ' '} {i} \"{cl.name}\": {cl.length} frames (0 to "
                         f"{cl.length - 1}) at {cl.fps} fps = {cl.seconds:.2f} s, {cl.loop_mode.lower()}, voxel look "
                         f"{'on' if cl.voxel_look else 'off'}, root motion {'on' if cl.root_motion else 'off'}")
        lines.append("Parts (models of the scene; pivot in the part's own cells; voxel frames besides its own):")
        parts = []
        for i, m in enumerate(scene.models):
            pivot = anim.pivot_of(c.track_of(m.uid), m)
            row = {"name": m.display_name(i), "parent": None if m.parent is None else self._part_name(m.parent),
                   "pivot": [float(v) for v in pivot], "voxel_frames": len(m.states), "empty": bool(m.empty),
                   "active": m.uid == self.doc.active_uid}
            parts.append(row)
            lines.append(f"{'*' if row['active'] else ' '} {row['name']}" + (" (empty part)" if m.empty else "")
                         + (f", follows {row['parent']}" if row["parent"] else "")
                         + f", pivot {_value_text(tuple(row['pivot']))}"
                         + (f", {len(m.states)} voxel frames (1 to {len(m.states)})" if m.states else ""))
        tracks = [t for t in c.tracks if t.properties]
        lines.append(f"Keys in \"{c.name}\" (frame=value ease):" if tracks else f"\"{c.name}\" has no keys.")
        for track in tracks:
            lines += self._track_text(track, scene)
        if c.markers:
            lines.append("Markers: " + ", ".join(f"{m.time} \"{m.name}\"" for m in c.markers))
        data = {"clips": [{"name": cl.name, "length": cl.length, "fps": cl.fps, "loop": cl.loop_mode.lower()}
                          for cl in scene.clips], "active_clip": scene.active_clip, "parts": parts,
                "keys": anim.to_json(c)["tracks"] if tracks else []}
        return Reply("\n".join(lines), data)

    def _anim_make_part(self, part=None, name=None):
        from rok.agent import ToolError

        m = self._model(part)
        if m.selection.is_empty():
            raise ToolError("Make Part cuts the selected voxels into a part: select them first with rok_select")
        source = self._name(m)

        def run():
            self.app_run("rok.make_part")
            if name:
                self.app_run("rok.model_rename", name=str(name))
            return f"Made the part \"{self._name(self.doc.model)}\"; it follows \"{source}\" and is the active model."

        return self._edit(run)

    def _anim_empty_part(self, part=None, name=None):
        m = self._model(part)
        source = self._name(m)

        def run():
            self.app_run("rok.part_empty")
            if name:
                self.app_run("rok.model_rename", name=str(name))
            return (f"Made the empty part \"{self._name(self.doc.model)}\" in the middle of \"{source}\"; it follows "
                    f"\"{source}\". Move it with rok_scene do=move; parts can follow it (do=parent).")

        return self._edit(run, look=False)

    def _anim_parent(self, part=None, parent=None):
        from rok.agent import ToolError

        m = self._model(part)
        child = self._name(m)
        if parent in (None, ""):
            uid = -1
        else:
            uid = next((x.uid for i, x in enumerate(self.doc.scene.models)
                        if x.display_name(i) == parent or x.name == parent), None)
            if uid is None:
                raise ToolError(f"no part named {parent!r}")
        if (m.parent if m.parent is not None else -1) == uid:
            return f"\"{child}\" already " + ("follows nothing." if uid < 0 else f"follows \"{parent}\".")
        self.app_run("rok.part_parent", parent=uid, uid=m.uid)
        return f"\"{child}\" now " + ("follows nothing." if uid < 0 else f"follows \"{parent}\": it moves and turns with it.")

    def _anim_pivot(self, part=None, pivot=None):
        from rok.agent import ToolError
        from rok.app.anim_mode import PivotRecord
        from rok.app.ops_common import commit

        m = self._model(part)
        before = m.pivot2
        if pivot is None:
            after = None
        else:
            try:
                after = tuple(int(round(2.0 * float(v))) for v in pivot)
            except (TypeError, ValueError):
                raise ToolError("pivot is [x, y, z] in the part's own cells, like [4, 0, 2.5]") from None
            if len(after) != 3:
                raise ToolError("pivot is [x, y, z] in the part's own cells, like [4, 0, 2.5]")
        if after == before:
            return "The pivot is already there."
        m.pivot2 = after
        commit(PivotRecord(m.uid, before, after))
        self.app.after_event()
        where = _value_text(tuple(anim.pivot_of(None, m)))
        return f"\"{self._name(m)}\" turns and sizes about {where} (its own cells)" + (" (the middle)." if after is None else ".")

    def _anim_key(self, part=None, frame=None, move=None, turn=None, scale=None, show=None, voxel_frame=None, ease=None,
                  views=None, size=None):
        from rok.agent import ToolError

        m = self._model(part)
        am = self._anim()
        t = self._frame(frame)
        e = _ease(ease)
        values: dict = {}

        def triple(v, what):
            try:
                out = tuple(float(x) for x in v)
            except (TypeError, ValueError):
                raise ToolError(f"{what} is [x, y, z]") from None
            if len(out) != 3:
                raise ToolError(f"{what} is [x, y, z]")
            return out

        if move is not None:
            values.update(zip(anim.MOVE, triple(move, "move")))
        if turn is not None:
            values.update(zip(anim.TURN, triple(turn, "turn")))
        if scale is not None:
            s = (float(scale),) * 3 if isinstance(scale, (int, float)) else triple(scale, "scale")
            if min(s) <= 0:
                raise ToolError("scale is in percent and must be above 0 (100 is the part's own size)")
            values[anim.SCALE] = s
        if show is not None:
            values[anim.VISIBLE] = bool(show)
        if voxel_frame is not None:
            n = int(voxel_frame)
            if not 0 <= n <= len(m.states):
                raise ToolError(f"\"{self._name(m)}\" has voxel frames 0 to {len(m.states)}; do=voxel_frame_add makes one")
            values[anim.STATE] = n
        if not values:
            raise ToolError("key needs at least one of move, turn, scale, show or voxel_frame")

        def change(c):
            track = am.picked_track(c)
            for name_, value in values.items():
                anim.key_change(track, name_, t, value, anim.rest_of(name_))
                if e is not None:
                    anim.set_ease(track, name_, [t], e)

        am.edit(change)
        self.app.after_event()
        track = am.clip().track_of(m.uid)
        now = ", ".join(f"{CHANNEL_NAMES.get(n, n.lower())} {_value_text(anim.channel(track, n, t))}" for n in values)
        reply = self._anim_reply(f"Keyed \"{self._name(m)}\" at frame {t}: {now}." + self._length_note(t), (t,), views, size)
        return reply

    def _anim_reply(self, text: str, frames, views=None, size=None):
        from rok.agent import Reply

        return Reply(text, {}, self._anim_picture(frames, views, size))

    def _anim_delete_keys(self, part=None, frame=None, frames=None, channels=None):
        from rok.agent import ToolError

        m = self._model(part)
        am = self._anim()
        times = None if frames is None and frame is None else [int(x) for x in (frames if frames is not None else [frame])]
        names = _channels(channels)
        gone = [0]

        def change(c):
            track = c.track_of(m.uid)
            if track is None:
                return
            for name_ in list(track.properties):
                if names is not None and name_ not in names:
                    continue
                if times is None:
                    gone[0] += len(track.properties.pop(name_).keys)
                else:
                    gone[0] += anim.remove_keys(track, name_, times)
            c.tracks = [tr for tr in c.tracks if tr.properties or tr.target.kind != anim.PART]

        am.edit(change)
        self.app.after_event()
        if not gone[0]:
            raise ToolError(f"\"{self._name(m)}\" has no keys there; do=status lists them")
        return f"Deleted {gone[0]} keys of \"{self._name(m)}\"."

    def _anim_ease(self, part=None, frame=None, frames=None, channels=None, ease=None, colour=None):
        from rok.agent import ToolError

        e = _ease(ease)
        if e is None:
            raise ToolError("ease needs ease: hold, linear, smooth, ease_in or ease_out")
        am = self._anim()
        times = None if frames is None and frame is None else [int(x) for x in (frames if frames is not None else [frame])]
        names = _channels(channels)
        if colour is not None:
            slot = int(colour)
            track_of = lambda c: next((tr for tr in c.tracks if tr.target.kind == anim.COLOUR  # noqa: E731
                                       and slot in tr.target.slots), None)
            who = f"colour {slot}"
        else:
            m = self._model(part)
            track_of = lambda c: c.track_of(m.uid)  # noqa: E731
            who = f"\"{self._name(m)}\""
        count = [0]

        def change(c):
            track = track_of(c)
            if track is None:
                return
            for name_, prop in track.properties.items():
                if (names is not None and name_ not in names) or name_ in anim.HOLD_ONLY:
                    continue
                these = [k.time for k in prop.keys if times is None or k.time in times]
                anim.set_ease(track, name_, these, e)
                count[0] += len(these)

        am.edit(change)
        if not count[0]:
            raise ToolError(f"{who} has no keys there that ease (show, voxel_frame and slot always hold)")
        return f"{count[0]} keys of {who} go on {e.lower().replace('_', ' ')}."

    def _anim_motion(self, part=None, name=None, axis=None, colour=None, views=None, size=None):
        from rok.agent import ToolError, _axis
        from rok.app.anim_mode import PivotRecord, colour_track, flow_slots
        from rok.app.ops_common import commit
        from rok.core.undo import AnimRecord, CompoundRecord

        motion = str(name or "").upper()
        if motion not in anim.PRESETS:
            raise ToolError(f"name must be a motion: {', '.join(p.lower() for p in anim.PRESETS)} "
                            "(the last three work on a colour)")
        am = self._anim()
        scene = self.doc.scene
        c = am.clip()
        before = copy.deepcopy(c)
        parts = []
        if motion in anim.COLOUR_PRESETS:
            slot = int(colour) if colour is not None else int(self.doc.colour)
            slots = flow_slots(scene, slot) if motion == "FLOW" else (slot,)
            track = colour_track(c, slots)
            if motion == "BLINK":
                track.settings["skin_colour"] = list(anim.neighbour_colour(scene, slots))
            who = f"colour {slot}" + (f" and {len(slots) - 1} more ({', '.join(map(str, slots[1:]))})" if len(slots) > 1 else "")
        else:
            m = self._model(part)
            others = [x for x in scene.models if x.uid != m.uid and not scene.is_hidden(x)]
            pivot_before = m.pivot2
            anchor2, default_axis = anim.default_anchor2(m, others)
            if pivot_before is None and motion in ("SPIN", "SWING"):
                lo, _hi = m.box_world()
                m.pivot2 = tuple(int(a) - 2 * int(p) for a, p in zip(anchor2, lo))
            track = am.picked_track(c)
            track.axis = (_axis(axis) if axis is not None else default_axis) if motion in ("SPIN", "SWING") else (
                _axis(axis) if axis is not None else 2)
            if m.pivot2 != pivot_before:
                parts.append(PivotRecord(m.uid, pivot_before, m.pivot2))
            who = f"\"{self._name(m)}\"" + (f" about {'xyz'[track.axis]}" if motion in ("SPIN", "SWING") else "")
        for channel_name, keys in anim.preset_keys(motion, track, c, scene):
            track.properties[channel_name] = anim.Property(channel_name)
            for key in keys:
                anim.set_key(track, channel_name, key)
        am._clip_version += 1
        parts.insert(0, AnimRecord(before, c))
        commit(CompoundRecord(tuple(parts)) if len(parts) > 1 else parts[0])
        self.app.after_event()
        return self._anim_reply(f"{anim.PRESET_LABELS[motion]} on {who}: ordinary keys over the whole clip "
                                f"({c.length} frames); do=status lists them.", self._spread(), views, size)

    def _anim_colour_key(self, frame=None, colour=None, hex=None, glow=None, ease=None, views=None, size=None):
        from rok.agent import ToolError, _colour, xs_parse_hex
        from rok.app.anim_mode import colour_track

        if colour is None:
            raise ToolError("colour_key needs colour: the palette number to animate")
        slot = _colour(colour)
        if slot == 0:
            raise ToolError("colour 0 is empty and has no colour to animate")
        t = self._frame(frame)
        e = _ease(ease)
        scene = self.doc.scene
        values: dict = {}
        if hex is not None:
            rgb = xs_parse_hex(str(hex))
            if rgb is None:
                raise ToolError(f"hex must be a colour like #ff8800; got {hex!r}")
            values[anim.COLOUR_RGB] = tuple(int(v) for v in rgb[:3])
        if glow is not None:
            values[anim.GLOW] = float(glow)
        if not values:
            raise ToolError("colour_key needs hex (the colour at that frame) or glow (a percent of its brightness)")
        rest = {anim.COLOUR_RGB: tuple(int(v) for v in scene.palette[slot, :3]), anim.GLOW: anim.rest_of(anim.GLOW)}
        am = self._anim()

        def change(c):
            track = colour_track(c, (slot,))
            for name_, value in values.items():
                anim.key_change(track, name_, t, value, rest[name_])
                if e is not None:
                    anim.set_ease(track, name_, [t], e)

        am.edit(change)
        self.app.after_event()
        return self._anim_reply(f"Keyed colour {slot} at frame {t}." + self._length_note(t), (t,), views, size)

    def _anim_voxel_frame_add(self, part=None, frame=None, voxel_frame=None):
        from rok.agent import ToolError
        from rok.app.ops_common import commit
        from rok.core.undo import AnimRecord, CompoundRecord, FramesRecord

        m = self._model(part)
        am = self._anim()
        source = 0 if voxel_frame is None else int(voxel_frame)
        if not 0 <= source <= len(m.states):
            raise ToolError(f"\"{self._name(m)}\" has voxel frames 0 to {len(m.states)}")
        c = am.clip()
        before_states = list(m.states)
        before_clip = copy.deepcopy(c)
        index = m.add_frame(m.frame_grid(source).voxels)
        records = [FramesRecord(m.uid, before_states, m.states)]
        note = ""
        if frame is not None:
            t = self._frame(frame)
            anim.key_change(am.picked_track(c), anim.STATE, t, index, 0)
            records.append(AnimRecord(before_clip, c))
            note = f" It shows from frame {t} on." + self._length_note(t)
        am._clip_version += 1
        am._frame_cache.clear()
        commit(CompoundRecord(tuple(records)) if len(records) > 1 else records[0])
        self.app.after_event()
        return (f"Voxel frame {index} of \"{self._name(m)}\": a copy of voxel frame {source}.{note} Change its cells "
                f"with do=voxel_frame_set voxel_frame={index}; show it with do=key voxel_frame={index}.")

    def _anim_voxel_frame_set(self, part=None, voxel_frame=None, voxels=None, layers=None, corner=None, legend=None,
                              frame=None, views=None, size=None):
        from rok.agent import ToolError
        from rok.app.ops_common import commit
        from rok.core.undo import FramesRecord

        m = self._model(part)
        n = 0 if voxel_frame is None else int(voxel_frame)
        if n == 0:
            raise ToolError("voxel frame 0 is the part's own voxels: change them with rok_voxels_set or rok_shape")
        if not 1 <= n <= len(m.states):
            raise ToolError(f"\"{self._name(m)}\" has extra voxel frames 1 to {len(m.states)}; do=voxel_frame_add makes one")
        if voxels is not None:
            rows = np.asarray(voxels, dtype=np.int64).reshape(-1, 4)
            cells, values = rows[:, :3], rows[:, 3]
        elif layers is not None:
            cells, values = self._layers(layers, corner, legend)
        else:
            raise ToolError("voxel_frame_set needs voxels ([x, y, z, colour] rows) or layers")
        if len(values) and (values.min() < 0 or values.max() > 255):
            raise ToolError("colours are palette numbers 0 to 255")
        size_ = np.asarray(m.grid.size)
        inside = np.all((cells >= 0) & (cells < size_), axis=1)
        before = list(m.states)
        after = [np.array(v, copy=True) for v in m.states]
        target = after[n - 1]
        c_in = cells[inside]
        target[tuple(c_in.T)] = values[inside].astype(np.uint8)
        m.states = after
        m.frame_grids.clear()
        self.doc.stroke_serial += 1  # the posed voxels' cache keys on it
        self._anim()._frame_cache.clear()
        commit(FramesRecord(m.uid, before, after))
        self.app.after_event()
        skipped = int((~inside).sum())
        text = (f"Set {int(inside.sum())} cells of voxel frame {n} of \"{self._name(m)}\""
                + (f"; {skipped} outside the grid were skipped." if skipped else "."))
        shows = self._shows_voxel_frame(m, n)
        if shows is None:
            return text + f" No key shows it yet: do=key voxel_frame={n} frame=..."
        return self._anim_reply(text + f" It shows at frame {shows}.", (shows,), views, size)

    def _shows_voxel_frame(self, m, n: int):
        track = self._anim().clip().track_of(m.uid)
        return next((k.time for k in (track.keys(anim.STATE) if track else []) if int(k.value) == n), None)

    def _anim_voxel_frame_delete(self, part=None, voxel_frame=None):
        from rok.agent import ToolError
        from rok.app.ops_common import commit
        from rok.core.undo import AnimRecord, CompoundRecord, FramesRecord

        m = self._model(part)
        index = 0 if voxel_frame is None else int(voxel_frame)
        if not 1 <= index <= len(m.states):
            raise ToolError(f"voxel_frame must be 1 to {len(m.states)} (0, the part's own voxels, stays)")
        am = self._anim()
        c = am.clip()
        before_states = list(m.states)
        before_clip = copy.deepcopy(c)
        m.remove_frame(index)
        track = c.track_of(m.uid)
        if track is not None and anim.STATE in track.properties:
            prop = track.properties[anim.STATE]
            prop.keys = [anim.Key(k.time, int(k.value) - 1 if int(k.value) > index else int(k.value), k.ease)
                         for k in prop.keys if int(k.value) != index]
            if not any(int(k.value) for k in prop.keys):
                del track.properties[anim.STATE]
        am._clip_version += 1
        am._frame_cache.clear()
        commit(CompoundRecord((FramesRecord(m.uid, before_states, m.states), AnimRecord(before_clip, c))))
        self.app.after_event()
        return f"Deleted voxel frame {index} of \"{self._name(m)}\"; the ones after it moved down by one."

    def _anim_clip(self, name=None, seconds=None, fps=None, loop=None, voxel_look=None, root_motion=None):
        from rok.agent import ToolError

        am = self._anim()
        c = am.clip()
        new_fps = int(fps) if fps is not None else c.fps
        if not 1 <= new_fps <= 120:
            raise ToolError("fps is 1 to 120")
        secs = float(seconds) if seconds is not None else c.seconds
        if not 0.05 <= secs <= 600:
            raise ToolError("seconds is 0.05 to 600")
        mode = None
        if loop is not None:
            mode = str(loop).upper()
            if mode not in anim.LOOP_MODES:
                raise ToolError("loop is loop, once or ping_pong")

        def change(cl):
            if name:
                cl.name = str(name).strip()
            cl.fps = new_fps
            cl.length = max(1, int(round(secs * new_fps)))
            if mode:
                cl.loop_mode = mode
            if voxel_look is not None:
                cl.voxel_look = bool(voxel_look)
            if root_motion is not None:
                cl.root_motion = bool(root_motion)

        am.edit(change)
        am.set_frame(am.frame())
        self.app.after_event()
        c = am.clip()
        return (f"Clip \"{c.name}\": {c.length} frames (0 to {c.length - 1}) at {c.fps} fps = {c.seconds:.2f} s, "
                f"{c.loop_mode.lower()}, voxel look {'on' if c.voxel_look else 'off'}, root motion "
                f"{'on' if c.root_motion else 'off'}. Keys stay on their frames.")

    def _anim_clip_add(self, name=None):
        am = self._anim()

        def change(scene):
            c = scene.timeline
            scene.clips.append(anim.Clip(name=str(name).strip() if name else scene.new_clip_name(), fps=c.fps,
                                         length=c.length, loop_mode=c.loop_mode, voxel_look=c.voxel_look))
            scene.active_clip = len(scene.clips) - 1

        am.edit_clips(change)
        return f"Added the clip \"{am.clip().name}\" (no keys); it is the active clip."

    def _anim_clip_duplicate(self, name=None):
        am = self._anim()

        def change(scene):
            c = copy.deepcopy(scene.timeline)
            c.name = str(name).strip() if name else f"{c.name} copy"
            scene.clips.insert(scene.active_clip + 1, c)
            scene.active_clip += 1

        am.edit_clips(change)
        return f"Duplicated the clip as \"{am.clip().name}\"; it is the active clip."

    def _anim_clip_delete(self):
        from rok.agent import ToolError

        am = self._anim()
        scene = self.doc.scene
        if len(scene.clips) < 2:
            raise ToolError("the last clip stays")
        gone = am.clip().name

        def change(sc):
            del sc.clips[sc.active_clip]
            sc.active_clip = min(sc.active_clip, len(sc.clips) - 1)

        am.edit_clips(change)
        return f"Deleted the clip \"{gone}\"; \"{am.clip().name}\" is the active clip."

    def _anim_marker(self, frame=None, name=None):
        am = self._anim()
        t = self._frame(frame)
        label = str(name or "").strip() or "marker"

        def change(c):
            c.markers = sorted([m for m in c.markers if m.time != t] + [anim.Marker(t, label)], key=lambda m: m.time)

        am.edit(change)
        return f"Marker \"{label}\" at frame {t}: a game reads it as an event." + self._length_note(t)

    def _anim_marker_delete(self, frame=None):
        from rok.agent import ToolError

        am = self._anim()
        t = self._frame(frame)
        if not any(m.time == t for m in am.clip().markers):
            raise ToolError(f"no marker at frame {t}")

        def change(c):
            c.markers = [m for m in c.markers if m.time != t]

        am.edit(change)
        return f"Deleted the marker at frame {t}."

    def _anim_look(self, frame=None, frames=None, views=None, size=None):
        from rok.agent import VIEW_NAMES, Picture, Reply, ToolError

        c = self._anim().clip()
        if frames is not None:
            times = tuple(int(x) for x in frames)
        elif frame is not None:
            times = (int(frame),)
        else:
            times = self._spread()
        if not 1 <= len(times) <= MAX_STRIP:
            raise ToolError(f"frames takes 1 to {MAX_STRIP} frames")
        if min(times) < 0:
            raise ToolError("frames count from 0")
        if isinstance(views, str):
            views = ("top", "iso", "front", "right") if views == "quad" else (views,)
        views = tuple(views or (("iso",) if len(times) > 1 else ("top", "iso", "front", "right")))
        if views == ("quad",):
            views = ("top", "iso", "front", "right")  # "quad" and ["quad"] mean the same
        for v in views:
            if v not in VIEW_NAMES:
                raise ToolError(f"no view {v!r}; the views are {', '.join(VIEW_NAMES)} and quad")
        if len(times) > 1 and len(views) != 1:
            raise ToolError("a strip of frames shows one view; give views as one name, like [\"iso\"] or [\"front\"]")
        if not 1 <= len(views) <= 4:
            raise ToolError("views takes one to four names, or quad")
        size = tuple(int(s) for s in (size or self.size))
        if not all(64 <= s <= 2000 for s in size):
            raise ToolError("size must be [width, height], each 64 to 2000 pixels")
        lines = [f"Clip \"{c.name}\" at frame{'s' if len(times) > 1 else ''} {', '.join(map(str, times))} "
                 f"(of 0 to {c.length - 1}). rok_probe reads the rest pose only."]
        for t in times[:1] if len(times) == 1 else ():
            fr = anim.evaluate(c, self.doc.scene, c.time_at(t))
            for uid, pose in fr.poses.items():
                if pose is not None and not pose.is_rest():
                    track = c.track_of(uid)
                    lines.append(f"  {self._part_name(uid)}: move {_value_text(anim.triple(track, anim.MOVE, c.time_at(t)))}, "
                                 f"turn {_value_text(anim.triple(track, anim.TURN, c.time_at(t)))}")
            hidden = [self._part_name(u) for u, on in fr.visible.items() if not on]
            if hidden:
                lines.append("  hidden: " + ", ".join(hidden))
        return Reply("\n".join(lines), {"frames": list(times)}, Picture(views, size, frames=times))

    def _anim_export(self, path=None, baked: bool = False, merge: bool = True, directions: int = 8,
                     cell_size: int = 128, fps=None, view: str = "iso"):
        from rok.agent import VIEW_NAMES, ToolError
        from rok.app import anim_pictures, export_ops
        from rok.core import anim_export, video, vox_write

        if not path:
            raise ToolError("export needs path; its suffix picks the kind: " +
                            ", ".join(f"{k} ({v})" for k, v in EXPORT_KINDS.items()))
        p = Path(str(path)).expanduser()
        suffix = p.suffix.lower()
        if suffix not in EXPORT_KINDS:
            raise ToolError("the path's suffix picks the kind: " + ", ".join(f"{k} ({v})" for k, v in EXPORT_KINDS.items()))
        scene = self.doc.scene
        if suffix in (".glb", ".vox"):
            if not export_ops.has_animation(scene):
                raise ToolError("nothing to export: no clip has keys and no part has voxel frames")
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                if suffix == ".vox":
                    vox_write.write_scene(p, scene.to_vox())
                    return f"Exported {p}: every clip's moves, turns and voxel frames as .vox frames."
                anim_export.write(p, scene, anim_export.BAKED if baked else anim_export.SMOOTH, bool(merge))
            except (OSError, ValueError) as exc:
                raise ToolError(f"export: {exc}") from None
            return (f"Exported {p}: glTF, " + ("every frame baked as voxels." if baked else
                                                "parts that move and turn, one animation per clip."))
        if view not in VIEW_NAMES:
            raise ToolError(f"no view {view!r}; the views are {', '.join(VIEW_NAMES)}")
        if anim_pictures.running():
            raise ToolError("an export of pictures is still running; call again in a moment")
        if suffix == ".mp4" and video.find_ffmpeg() is None:
            raise ToolError(video.MISSING)
        from rok.headless import VIEWS, camera_for

        yaw, pitch, _ortho = VIEWS[view]
        try:
            if suffix == ".png":
                job = anim_pictures.sprite_job(self.app, int(directions), int(cell_size), int(fps or 0),
                                               angle=(yaw, pitch), path=p)
            else:
                c = self._anim().clip()
                boxes = [b for b in (anim_pictures.shown_box(self._anim().shown_at(c, t)) for t in range(c.length))
                         if b is not None]
                if not boxes:
                    raise ValueError("there are no voxels to draw")
                lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
                width = anim_pictures.GIF_WIDTH if suffix == ".gif" else anim_pictures.MP4_WIDTH
                cam = camera_for(view, width, width * 3 // 4, lo, hi)
                job = anim_pictures.video_job(self.app, anim_pictures.GIF if suffix == ".gif" else anim_pictures.MP4,
                                              camera=cam, path=p)
        except ValueError as exc:
            raise ToolError(f"export: {exc}") from None
        anim_pictures.start(self.app, job)
        made = f"{job.path} and {job.path.with_suffix('.json')}" if suffix == ".png" else str(job.path)
        return f"Exporting the {EXPORT_KINDS[suffix]} of \"{job.clip.name}\" ({len(job.times)} frames) to {made}."

    # --- rok_render ---------------------------------------------------------------------

    def tool_render(self, preset: str = "", view: str = "iso", size=None, samples=None, path=None, frame=None,
                    model=None, azimuth=None, elevation=None, eye=None, target=None, distance=None, fov=None,
                    ortho=None, low=None, high=None, ground=None, return_image=None, bloom=None, exposure=None):
        """A path-traced picture with a render preset, saved at `path` when given.
        azimuth ... ortho place the camera by hand (Agent.camera_spec); low and
        high frame a box of cells instead of the whole scene."""
        from rok.agent import VIEW_NAMES, Picture, Reply, ToolError, _xyz
        from rok.app import render_view
        from rok.app.object import model_origin
        from rok.core import render_presets as rp
        from rok.headless import VIEWS

        if self.art is not None:
            # a finished picture of a voxel art scene needs a finished piece
            from rok.agent_art import verdict
            from rok.core import craft

            why = []
            for m in self.doc.scene.models:
                n = len(craft.parts(m.grid.voxels, limit=2))
                if n > 1:
                    why.append(f"{self._name(m)} has floating parts (rok_check lists them)")
            kind, reasons = verdict([m.grid.voxels for m in self.doc.scene.models])
            if kind != "voxel art":
                why.append("the verdict is " + kind + " (" + "; ".join(reasons) + ")")
            if why:
                raise ToolError("No finished picture yet: " + "; ".join(why) + ". Fix that first; rok_look shows "
                                "the piece meanwhile.")
        m = self._model(model)
        ids = [p.id for p in rp.PRESETS]
        own = not preset  # no preset named: the file's own light, the camera of the preset it came from
        pid = str(preset or render_view.preset_id()).upper()
        if pid not in ids:
            raise ToolError(f"no preset {preset!r}; the presets are " +
                            ", ".join(f"{p.id.lower()} ({p.label})" for p in rp.PRESETS))
        p = rp.preset_by_id(pid)
        floor = (render_view.scene_light().ground is not None if own else p.ground) if ground is None else bool(ground)
        if view not in VIEW_NAMES:
            raise ToolError(f"no view {view!r}; the views are {', '.join(VIEW_NAMES)}")
        w, h = (int(s) for s in (size or p.resolution))
        if not (64 <= w <= 4096 and 64 <= h <= 4096):
            raise ToolError("size must be [width, height], each 64 to 4096 pixels")
        n = int(samples) if samples is not None else render_view.picture_samples(p)
        if not 1 <= n <= 4096:
            raise ToolError("samples is 1 to 4096")
        look = {}
        if bloom is not None:
            if not 0 <= float(bloom) <= 1:
                raise ToolError("bloom is 0 (none: glowing voxels stay hard cubes) to 1")
            look["bloom"] = float(bloom)
        if exposure is not None:
            if not -3 <= float(exposure) <= 3:
                raise ToolError("exposure is -3 to 3 stops (+1 is twice as bright)")
            look["exposure"] = float(exposure)
        frames = None
        if frame is not None:
            frames = (self._frame(frame),)
        save = None
        if path:
            save = str(Path(str(path)).expanduser().with_suffix(".png"))
        placed = any(v is not None for v in (azimuth, elevation, eye, target, distance, fov))
        if ortho is None and placed:
            ortho = p.ortho  # a placed camera keeps the preset's kind of lens unless told
        camera, camera_line = self.camera_spec(m, azimuth, elevation, eye, target, distance, fov, ortho)
        if (low is None) != (high is None):
            raise ToolError("low and high frame a box of cells together: give both, two corner cells [x, y, z]")
        if low is not None:
            o = np.asarray(model_origin(m), dtype=np.float64)
            a, b = np.asarray(_xyz(low, "low"), float), np.asarray(_xyz(high, "high"), float)
            box = (tuple(np.minimum(a, b) + o), tuple(np.maximum(a, b) + o + 1))
            if camera is None:  # the named side's angle, framed on the box
                yaw, pitch, _flat = VIEWS[view]
                camera = {"azimuth": float(np.degrees(yaw)), "elevation": float(np.degrees(pitch)),
                          "ortho": bool(p.ortho), "fov": None}
            camera["frame"] = box
            camera_line = (camera_line + " " if camera_line else "") + (
                f"It frames cells {np.minimum(a, b).astype(int).tolist()}..{np.maximum(a, b).astype(int).tolist()}.")
        posed = f", the parts posed at frame {frames[0]}" if frames else ""
        used = f"the file's light (the camera of {p.label})" if own else p.label
        if camera is None:
            iso = "" if not p.ortho else " (its own isometric camera, turned to the nearest side)"
            lines = [f"Rendered with {used}: {w} x {h} pixels, {n} samples, from the {view} side{iso}{posed}."]
        else:
            lines = [f"Rendered with {used}: {w} x {h} pixels, {n} samples{posed}. {camera_line}"]
        if look:
            lines.append("Look: " + ", ".join(f"{k} {v:g}" + (" stops" if k == "exposure" else "")
                                               for k, v in look.items()) + " (this picture only).")
        if floor != (render_view.scene_light().ground is not None if own else p.ground):
            lines.append("The preset's endless floor is " + ("on." if floor else "off: the scene ends at its own edge."))
        send = (save is None) if return_image is None else bool(return_image)
        if save:
            lines.append(f"Saved {save}." + ("" if send else " The picture is not sent back (return_image=true sends it)."))
        return Reply(" ".join(lines), {"path": save},
                     Picture((view,), (w, h), render=(pid, n, floor, own, look), frames=frames, save=save, camera=camera,
                             send=send))
