"""The timeline dock of Animate mode: drawing, presses and the two drags.

The geometry is rok/core/dock.py. The dock runs along the bottom of the
view, right of the rail, while Rok is in Animate mode. A press on the ruler
scrubs; a press on a key picks it (Shift adds) and a drag moves the picked
keys in whole frames; a press on a part's name picks the part; its triangle
opens its channels. The bar's buttons play, loop, set the clip's length and
frame rate, set the picked keys' ease and switch Voxel look.

One dark card in Rok's look; Animate's violet marks the dock, ember what is
picked (the play head, a picked key).
"""

from __future__ import annotations

import time

from rok.core import anim
from rok.core import chrome as chrome_kit
from rok.core import dock as kit

from . import anim_mode
from . import object as voxel_object
from .gfx import pen
from .handlers import registry
from .object import document
from .opkit import BoolProp, FloatProp, IntProp, Action, StringProp, ops, register
from .paint import clip_text, round_outline, round_rect, text
from .state import state


def _rgba(hex_rgb: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = hex_rgb.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, a)


DOCK_BG = _rgba("15121F", 0.96)
DOCK_EDGE = _rgba("2E2748")
NAME_BG = _rgba("1A1626")
ROW_STRIPE = (1.0, 1.0, 1.0, 0.025)
ROW_PICKED = _rgba("8B6CF2", 0.18)
VIOLET = _rgba("8B6CF2")
VIOLET_LIGHT = _rgba("A58DFF")
SPAN = _rgba("A58DFF", 0.45)
SPAN_HOLD = _rgba("A58DFF", 0.20)
EMBER = _rgba("E85D3A")
KEY_FILL = _rgba("EFEAFF")
KEY_RING = _rgba("0D0B14")
PAST_END = (0.0, 0.0, 0.0, 0.32)
BUTTON_BG = (1.0, 1.0, 1.0, 0.06)
BUTTON_HOVER = (1.0, 1.0, 1.0, 0.13)
BUTTON_ON = _rgba("8B6CF2", 0.55)
TEXT = (0.94, 0.92, 0.98, 1.0)
TEXT_DIM = (0.64, 0.61, 0.74, 1.0)
TEXT_FAINT = (0.45, 0.42, 0.54, 1.0)

_opened: set[int] = set()  # parts whose channel rows show
_picked: set[tuple[int, str | None, int]] = set()  # (uid, channel or None for the part's row, frame)
_hover: kit.Hit | None = None
_drag: dict | None = None  # a key drag: the frames moved so far, for the drawing
_tab_press: tuple[int, float] | None = None  # (tab index, when): the second press of a double-click renames
_parent_drag: dict | None = None  # a name dragged onto another part: "uid", "target" (a uid, None, or "KEEP")
_clipboard: list[tuple[int, str, int, object, str]] = []  # copied keys: (uid, channel, frames after the first, value, ease)
_box: dict | None = None  # a box pick in the lane: "x0", "y0", "x1", "y1", "shift"
_row_press: tuple[int, str, float] | None = None  # (uid, channel, when): a double-click on a value types it
_picked_marker: int | None = None  # a marker's frame
_marker_press: tuple[int, float] | None = None  # (frame, when): a double-click on the markers row adds or renames
_marker_drag: dict | None = None  # "frame", "offset"

MIRRORED = tuple(anim.TURN) + (anim.MOVE_X, anim.MOVE_Y)  # Paste mirrored negates these (the other leg's pose)


def active() -> bool:
    return state.animating and document.scene is not None and document.model is not None


def picked_keys() -> list[tuple[int, str, int]]:
    """The picked keys as (uid, channel, frame): a pick on a part's row is every
    channel's key at that frame."""
    c = anim_mode.clip()
    if c is None:
        return []
    out = []
    for uid, channel, frame in sorted(_picked, key=lambda p: (p[0], p[1] or "", p[2])):
        track = c.track_of(uid)
        if track is None:
            continue
        names = [channel] if channel is not None else list(track.properties)
        for name in names:
            if any(k.time == frame for k in track.keys(name)):
                out.append((uid, name, frame))
    return sorted(set(out))


def clear_picked() -> None:
    _picked.clear()


def _picked_ease() -> str | None:
    c = anim_mode.clip()
    for uid, name, frame in picked_keys():
        for k in c.track_of(uid).keys(name):
            if k.time == frame:
                return k.ease
    return None


def _region_left(region) -> float:
    from . import chrome

    rail = chrome.layout_for(region)
    return rail.rect[0] + rail.rect[2] + kit.MARGIN


def layout_for(region) -> kit.DockLayout | None:
    if not active() or region is None:
        return None
    from . import overlay

    scale = overlay.ui_scale()
    left = _region_left(region)
    width = max(200.0, float(region.width) - left - kit.MARGIN * scale)
    scene = document.scene
    index = {m.uid: k for k, m in enumerate(scene.models)}
    parts = [(m.uid, m.display_name(index[m.uid]), depth) for m, depth in scene.ordered_parts()]
    return kit.dock_layout(scale, left, width, scene.timeline, parts, document.active_uid, frozenset(_opened),
                           anim_mode.frame(), anim_mode.playing(), _picked_ease(),
                           clips=[c.name for c in scene.clips], active_clip=scene.active_clip,
                           onion=bool(state.onion_skin),
                           frames={m.uid: m.frame_count() for m in scene.models if m.states})


def contains(region, x: float, y: float) -> bool:
    layout = layout_for(region)
    return layout is not None and kit.hit(layout, x, y) is not None


def hover(region, x: float, y: float) -> bool:
    """Track the mouse over the dock. True when it is inside."""
    global _hover
    layout = layout_for(region)
    found = None if layout is None else kit.hit(layout, x, y)
    if found != _hover:
        _hover = found
        voxel_object.redraw_3d_views()
    return found is not None


# --- presses ------------------------------------------------------------------------------


def press(context, region, x: float, y: float, left: bool, shift: bool) -> bool:
    """Act on a press in the dock. True when the press was inside it."""
    layout = layout_for(region)
    found = None if layout is None else kit.hit(layout, x, y)
    if found is None:
        return False
    if not left:
        return True
    c = anim_mode.clip()
    global _picked_marker, _marker_press
    now = time.monotonic()
    if found.kind == kit.MARKER:
        _picked_marker = found.frame
        _picked.clear()
        anim_mode.set_frame(min(found.frame, c.length - 1))
        if chrome_kit.is_double_click(_marker_press, found.frame, now):
            _marker_press = None
            ops.rok.marker_rename("INVOKE")
        else:
            _marker_press = (found.frame, now)
            ops.rok.dock_marker_drag("INVOKE", frame=found.frame, grab_x=float(x))
        voxel_object.redraw_3d_views()
        return True
    if found.kind == kit.ROW and found.channel == kit.MARKER_ROW:
        _picked_marker = None
        if found.frame >= 0:
            anim_mode.set_frame(min(found.frame, c.length - 1))
            if chrome_kit.is_double_click(_marker_press, -1, now):
                _marker_press = None
                ops.rok.marker_add("INVOKE", frame=found.frame)
            else:
                _marker_press = (-1, now)
        voxel_object.redraw_3d_views()
        return True
    if found.kind != kit.MARKER:
        _picked_marker = None
    if found.kind == kit.BUTTON:
        _button(found.id)
    elif found.kind == kit.RULER:
        ops.rok.dock_scrub("INVOKE")
    elif found.kind == kit.OPENER:
        _opened.symmetric_difference_update({found.uid})
    elif found.kind == kit.KEY:
        key = (found.uid, found.channel, found.frame)
        if shift:
            _picked.symmetric_difference_update({key})
        elif key not in _picked:
            _picked.clear()
            _picked.add(key)
        _pick_part(found.uid)
        anim_mode.set_frame(min(found.frame, c.length - 1))
        if key in _picked:
            ops.rok.dock_key_drag("INVOKE", grab_x=float(x))
    elif found.kind == kit.ROW and found.channel == kit.COLOUR_ROW:
        if found.frame >= 0:
            anim_mode.set_frame(min(found.frame, c.length - 1))
    elif found.kind == kit.ROW:
        global _row_press
        _pick_part(found.uid)
        if not shift:
            _picked.clear()
        if found.frame >= 0:
            anim_mode.set_frame(min(found.frame, c.length - 1))
            ops.rok.dock_box_pick("INVOKE", shift=shift, x=float(x), y=float(y))  # a drag picks a box of keys
        elif found.channel is None:
            ops.rok.dock_parent_drag("INVOKE", uid=found.uid)  # a name: dragged onto another, it follows it
        else:
            now = time.monotonic()
            if _row_press is not None and _row_press[:2] == (found.uid, found.channel) \
                    and 0.0 <= now - _row_press[2] <= chrome_kit.DOUBLE_CLICK_S:
                _row_press = None
                names = anim.MOVE + anim.TURN
                if found.channel in names:
                    ops.rok.key_value("INVOKE", channel=names.index(found.channel))  # type the value
            else:
                _row_press = (found.uid, found.channel, now)
    voxel_object.redraw_3d_views()
    return True


def _pick_part(uid: int) -> None:
    if uid != document.active_uid:
        document.set_active(uid)
        from . import gizmo_ops

        gizmo_ops.refresh()


def _button(ident: str) -> None:
    global _tab_press
    if ident == kit.TAB_ADD:
        ops.rok.clip_add()
        return
    if ident.startswith("TAB_"):
        index = int(ident[len("TAB_"):])
        now = time.monotonic()
        if chrome_kit.is_double_click(_tab_press, index, now):
            _tab_press = None
            ops.rok.clip_rename("INVOKE")  # the second press: a name
            return
        _tab_press = (index, now)
        if index != document.scene.active_clip:
            ops.rok.clip_pick(index=index)
        return
    if ident == "START":
        anim_mode.stop()
        anim_mode.set_frame(0)
    elif ident == "PLAY":
        ops.rok.play("INVOKE")
    elif ident == "LOOP":
        ops.rok.clip_settings()
    elif ident == "VOXEL_LOOK":
        ops.rok.clip_settings(voxel_look=0 if anim_mode.clip().voxel_look else 1)
    elif ident == "ONION":
        ops.rok.onion_skin()
    elif ident in ("LENGTH", "FPS"):
        ops.rok.clip_length("INVOKE")
    elif ident.startswith("EASE_"):
        ops.rok.key_ease(ease=ident[len("EASE_"):])


# --- copy and paste ---------------------------------------------------------------------------


def copy_picked() -> int:
    """The picked keys into the clipboard, their frames counted from the first.
    Returns how many."""
    c = anim_mode.clip()
    keys = picked_keys()
    if c is None or not keys:
        return 0
    first = min(f for _u, _ch, f in keys)
    out = []
    for uid, name, frame in keys:
        for k in c.track_of(uid).keys(name):
            if k.time == frame:
                out.append((uid, name, frame - first, k.value, k.ease))
    _clipboard[:] = out
    return len(out)


def _mirrored(name: str, value):
    if name in MIRRORED and isinstance(value, (int, float)) and not isinstance(value, bool):
        return -value
    return value


def paste_clipboard(at: int, mirrored: bool = False) -> bool:
    """The clipboard's keys from frame `at` on, on the parts they came from.
    Mirrored negates the turns and the sideways moves: the other leg's pose."""
    if not _clipboard:
        return False
    scene = document.scene

    def change(c):
        for uid, name, rel, value, ease in _clipboard:
            if not scene.has_uid(uid):
                continue
            track = c.track_of(uid)
            if track is None:
                model = scene.by_uid(uid)
                track = anim.Track(anim.Target(anim.PART, uid), name=model.display_name(scene.index_of(uid)))
                c.tracks.append(track)
            anim.set_key(track, name, anim.Key(at + rel, _mirrored(name, value) if mirrored else value, ease))

    if not anim_mode.edit(change):
        return False
    _picked.clear()
    _picked.update((uid, name, at + rel) for uid, name, rel, _v, _e in _clipboard)
    return True


def picked_marker() -> int | None:
    return _picked_marker


def clear_picked_marker() -> None:
    global _picked_marker
    _picked_marker = None


@register
class MarkerAddAction(Action):
    command_id = "rok.marker_add"
    title = "Add Marker"
    tooltip = "A named moment in the clip at the play head (double-click the Markers row); games read them"
    name = StringProp(name="Name", default="marker")
    frame = IntProp(name="Frame", default=-1, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return active()

    def invoke(self, context, event):
        return context.prefs.open_dialog(self, title="Marker", confirm_text="Add")

    def draw(self, context):
        self.layout.prop(self, "name")

    def execute(self, context):
        global _picked_marker
        t = int(self.frame) if int(self.frame) >= 0 else anim_mode.frame()
        name = str(self.name).strip() or "marker"

        def change(c):
            c.markers = [m for m in c.markers if m.time != t] + [anim.Marker(t, name)]
            c.markers.sort(key=lambda m: m.time)

        if not anim_mode.edit(change):
            return {"CANCELLED"}
        _picked_marker = t
        return {"FINISHED"}


@register
class MarkerRenameAction(Action):
    command_id = "rok.marker_rename"
    title = "Rename Marker"
    tooltip = "The picked marker's name"
    name = StringProp(name="Name", default="")

    @classmethod
    def poll(cls, context):
        return active() and _picked_marker is not None

    def invoke(self, context, event):
        c = anim_mode.clip()
        self.name = next((m.name for m in c.markers if m.time == _picked_marker), "")
        return context.prefs.open_dialog(self, title="Marker", confirm_text="Rename")

    def draw(self, context):
        self.layout.prop(self, "name")

    def execute(self, context):
        t = _picked_marker
        name = str(self.name).strip()
        if t is None or not name:
            return {"CANCELLED"}

        def change(c):
            c.markers = [anim.Marker(m.time, name if m.time == t else m.name) for m in c.markers]

        return {"FINISHED"} if anim_mode.edit(change) else {"CANCELLED"}


@register
class MarkerDeleteAction(Action):
    command_id = "rok.marker_delete"
    title = "Delete Marker"
    tooltip = "Delete the picked marker"
    @classmethod
    def poll(cls, context):
        return active() and _picked_marker is not None

    def execute(self, context):
        global _picked_marker
        t = _picked_marker

        def change(c):
            c.markers = [m for m in c.markers if m.time != t]

        if not anim_mode.edit(change):
            return {"CANCELLED"}
        _picked_marker = None
        return {"FINISHED"}


@register
class DockMarkerDragAction(Action):
    command_id = "rok.dock_marker_drag"
    title = "Move Marker"
    tooltip = "Drag a marker along the Markers row"
    frame = IntProp(default=0, options={"SKIP_SAVE"})
    grab_x = FloatProp(default=0.0, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return active()

    def invoke(self, context, event):
        global _marker_drag
        self._layout = layout_for(context.region)
        self._start = self._layout.frame_at(self.grab_x)
        _marker_drag = {"frame": int(self.frame), "offset": 0}
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        global _marker_drag, _picked_marker
        if _marker_drag is None:
            return {"CANCELLED"}
        if event.type in ("MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"):
            offset = self._layout.frame_at(event.mouse_region_x) - self._start
            _marker_drag["offset"] = max(offset, -_marker_drag["frame"])
            voxel_object.redraw_3d_views()
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            drag, _marker_drag = _marker_drag, None
            if drag["offset"]:
                t0, t1 = drag["frame"], drag["frame"] + drag["offset"]

                def change(c):
                    c.markers = [anim.Marker(t1 if m.time == t0 else m.time, m.name) for m in c.markers if m.time != t1 or m.time == t0]
                    c.markers.sort(key=lambda m: m.time)

                if anim_mode.edit(change):
                    _picked_marker = t1
            voxel_object.redraw_3d_views()
            return {"FINISHED"}
        if event.type == "ESCAPE":
            _marker_drag = None
            voxel_object.redraw_3d_views()
            return {"CANCELLED"}
        return {"RUNNING"}


class _KeysOp(Action):
    @classmethod
    def poll(cls, context):
        return active()


@register
class KeyCopyAction(_KeysOp):
    command_id = "rok.key_copy"
    title = "Copy Keys"
    tooltip = "Copy the picked keys (Cmd+C in Animate mode)"

    @classmethod
    def poll(cls, context):
        return active() and bool(_picked)

    def execute(self, context):
        n = copy_picked()
        if not n:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Copied {n} key{'s' if n != 1 else ''}")
        return {"FINISHED"}


@register
class KeyPasteAction(_KeysOp):
    command_id = "rok.key_paste"
    title = "Paste Keys"
    tooltip = "Paste the copied keys at the play head; mirrored negates the turns and sideways moves"

    mirrored = BoolProp(name="Mirrored", default=False)

    @classmethod
    def poll(cls, context):
        return active() and bool(_clipboard)

    def execute(self, context):
        if not paste_clipboard(anim_mode.frame(), bool(self.mirrored)):
            return {"CANCELLED"}
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class KeyValueAction(_KeysOp):
    command_id = "rok.key_value"
    title = "Key Value"
    tooltip = "Type a channel's value at the play head: it becomes a key"

    channel = IntProp(default=0, options={"HIDDEN"})  # an index into MOVE + TURN
    value = FloatProp(name="Value", default=0.0, step=100, precision=1)

    def invoke(self, context, event):
        c = anim_mode.clip()
        name = self._name()
        track = c.track_of(document.active_uid) if c is not None else None
        self.value = float(anim.channel(track, name, c.time_at(anim_mode.frame()))) if c is not None else 0.0
        return context.prefs.open_dialog(self, title=anim.CHANNEL_LABELS.get(name, name), confirm_text="Key")

    def _name(self) -> str:
        names = anim.MOVE + anim.TURN
        return names[int(self.channel) % len(names)]

    def draw(self, context):
        self.layout.prop(self, "value")

    def execute(self, context):
        name = self._name()
        t = anim_mode.frame()

        def change(c):
            track = anim_mode.picked_track(c)
            anim.key_change(track, name, t, float(self.value), anim.rest_of(name))

        return {"FINISHED"} if anim_mode.edit(change) else {"CANCELLED"}


# --- the drags --------------------------------------------------------------------------------


@register
class DockBoxPickAction(Action):
    command_id = "rok.dock_box_pick"
    title = "Pick Keys"
    tooltip = "Drag a box over the dock: the keys inside are picked (Shift adds)"
    shift = BoolProp(default=False, options={"SKIP_SAVE"})
    x = FloatProp(default=0.0, options={"SKIP_SAVE"})  # the press, region pixels
    y = FloatProp(default=0.0, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return active()

    def invoke(self, context, event):
        global _box
        self._keep = set(_picked) if self.shift else set()
        _box = {"x0": float(self.x), "y0": float(self.y), "x1": float(self.x), "y1": float(self.y), "moved": False}
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        global _box
        if _box is None:
            return {"CANCELLED"}
        if event.type in ("MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"):
            _box["x1"], _box["y1"] = float(event.mouse_region_x), float(event.mouse_region_y)
            layout = layout_for(context.region)
            s = layout.scale if layout else 1.0
            if abs(_box["x1"] - _box["x0"]) > 4 * s or abs(_box["y1"] - _box["y0"]) > 4 * s:
                _box["moved"] = True
            if _box["moved"] and layout is not None:
                _picked.clear()
                _picked.update(self._keep)
                _picked.update(keys_in_box(layout, _box))
            voxel_object.redraw_3d_views()
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            _box = None
            voxel_object.redraw_3d_views()
            return {"FINISHED"}
        if event.type == "ESCAPE":
            _box = None
            voxel_object.redraw_3d_views()
            return {"CANCELLED"}
        return {"RUNNING"}


def keys_in_box(layout: kit.DockLayout, box: dict) -> set[tuple[int, str | None, int]]:
    """The keys whose row crosses the box's height and whose frame lies in its width."""
    x0, x1 = sorted((box["x0"], box["x1"]))
    y0, y1 = sorted((box["y0"], box["y1"]))
    r = kit.KEY_R * layout.scale
    out = set()
    for row in layout.rows:
        if row.rect[1] + row.rect[3] < y0 or row.rect[1] > y1:
            continue
        for t, _e in row.keys:
            x = layout.x_of(t)
            if x0 - r <= x <= x1 + r:
                out.add((row.uid, row.channel, t))
    return out


@register
class DockScrubAction(Action):
    command_id = "rok.dock_scrub"
    title = "Scrub"
    tooltip = "Drag along the dock's ruler to move the play head"
    @classmethod
    def poll(cls, context):
        return active()

    def invoke(self, context, event):
        anim_mode.stop()
        self._go(context, event)
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def _go(self, context, event) -> None:
        layout = layout_for(context.region)
        if layout is not None:
            anim_mode.set_frame(min(layout.frame_at(event.mouse_region_x), layout.length - 1))

    def modal(self, context, event):
        if event.type in ("MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"):
            self._go(context, event)
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            return {"FINISHED"}
        if event.type == "ESCAPE":
            return {"FINISHED"}
        return {"RUNNING"}


@register
class DockKeyDragAction(Action):
    command_id = "rok.dock_key_drag"
    title = "Move Keys"
    tooltip = "Drag the picked keys along the dock: they move in whole frames"
    grab_x = FloatProp(default=0.0, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return active() and bool(_picked)

    def invoke(self, context, event):
        global _drag
        self._layout = layout_for(context.region)
        self._start = self._layout.frame_at(self.grab_x)
        _drag = {"offset": 0}
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        global _drag
        if event.type in ("MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"):
            offset = self._layout.frame_at(event.mouse_region_x) - self._start
            lowest = min((f for _u, _c, f in _picked), default=0)
            _drag = {"offset": max(offset, -lowest)}
            voxel_object.redraw_3d_views()
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            offset = (_drag or {}).get("offset", 0)
            _drag = None
            if offset:
                self._move(offset)
            voxel_object.redraw_3d_views()
            return {"FINISHED"}
        if event.type == "ESCAPE":
            _drag = None
            voxel_object.redraw_3d_views()
            return {"CANCELLED"}
        return {"RUNNING"}

    @staticmethod
    def _move(offset: int) -> None:
        keys = picked_keys()

        def change(c):
            for uid, name, frame in keys:
                track = c.track_of(uid)
                if track is not None:
                    anim.move_keys(track, name, [frame], offset)

        if anim_mode.edit(change):
            moved = {(u, ch, f + offset) for u, ch, f in _picked}
            _picked.clear()
            _picked.update(moved)
            anim_mode.set_frame(min(anim_mode.frame() + offset, anim_mode.clip().length - 1))


@register
class DockParentDragAction(Action):
    command_id = "rok.dock_parent_drag"
    title = "Set Parent"
    tooltip = "Drag a part's name onto another part: it follows that part. Onto the ruler: it follows none"
    uid = IntProp(default=0, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return active()

    def invoke(self, context, event):
        global _parent_drag
        _parent_drag = {"uid": int(self.uid), "target": "KEEP", "moved": False, "y0": float(event.mouse_region_y)}
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        global _parent_drag
        if _parent_drag is None:
            return {"CANCELLED"}
        if event.type in ("MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"):
            layout = layout_for(context.region)
            x, y = float(event.mouse_region_x), float(event.mouse_region_y)
            if abs(y - _parent_drag["y0"]) > 6 * (layout.scale if layout else 1.0):
                _parent_drag["moved"] = True
            target = "KEEP"
            if layout is not None and _parent_drag["moved"]:
                found = kit.hit(layout, x, y)
                if found is not None and found.kind in (kit.ROW, kit.OPENER, kit.KEY) and found.uid != _parent_drag["uid"]:
                    target = found.uid
                elif found is not None and found.kind == kit.RULER or (found is not None and found.kind == kit.ROW and found.uid == _parent_drag["uid"]):
                    target = "KEEP" if found.kind == kit.ROW else None
            _parent_drag["target"] = target
            voxel_object.redraw_3d_views()
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            drag, _parent_drag = _parent_drag, None
            if drag["moved"] and drag["target"] != "KEEP":
                ops.rok.part_parent(uid=drag["uid"], parent=-1 if drag["target"] is None else int(drag["target"]))
            voxel_object.redraw_3d_views()
            return {"FINISHED"}
        if event.type == "ESCAPE":
            _parent_drag = None
            voxel_object.redraw_3d_views()
            return {"CANCELLED"}
        return {"RUNNING"}


@register
class ClipLengthAction(Action):
    command_id = "rok.clip_length"
    title = "Clip Length"
    tooltip = "The clip's length in seconds and its frame rate. Keys stay on their frames"
    seconds = FloatProp(name="Length (s)", default=1.0, min=0.05, max=600.0, step=10, precision=2)
    fps = IntProp(name="Frames per second", default=anim.DEFAULT_FPS, min=1, max=120)
    root_motion = BoolProp(name="Root motion",
                               description="On: the root part's movement goes into the export as the character's "
                                           "movement. Off: the clip plays in place (6.4)", default=False)

    @classmethod
    def poll(cls, context):
        return active()

    def invoke(self, context, event):
        c = anim_mode.clip()
        self.seconds = c.seconds
        self.fps = c.fps
        self.root_motion = bool(c.root_motion)
        return context.prefs.open_dialog(self, title="Clip", confirm_text="Set")

    def draw(self, context):
        self.layout.prop(self, "seconds")
        self.layout.prop(self, "fps")
        self.layout.prop(self, "root_motion")

    def execute(self, context):
        fps = int(self.fps)
        length = max(1, int(round(float(self.seconds) * fps)))
        root = bool(self.root_motion)

        def change(c):
            c.fps = fps
            c.length = length
            c.root_motion = root

        anim_mode.edit(change)
        anim_mode.set_frame(anim_mode.frame())
        return {"FINISHED"}


# --- drawing ------------------------------------------------------------------------------


def _draw_bg(context) -> None:
    layout = layout_for(context.region)
    if layout is None:
        return
    s = layout.scale
    round_rect(layout.rect, 12 * s, DOCK_BG)
    round_outline(layout.rect, 12 * s, DOCK_EDGE, 1.0 * s)
    target = (_parent_drag or {}).get("target", "KEEP")
    for k, row in enumerate(layout.rows):
        if row.kind == kit.PART_ROW and target not in ("KEEP", None) and row.uid == target:
            round_rect(row.rect, 4 * s, BUTTON_ON)  # where the dragged name would land
        elif row.picked:
            round_rect(row.rect, 4 * s, ROW_PICKED)
        elif k % 2 == 0:
            round_rect(row.rect, 4 * s, ROW_STRIPE)
    if _parent_drag is not None and target is None:
        round_rect(layout.ruler, 4 * s, BUTTON_ON)  # dropped on the ruler: no parent
    round_rect(layout.markers_row, 4 * s, NAME_BG)
    # Past the clip's end: keys there do not play.
    lx0, lx1 = layout.lane
    end = layout.x_of(layout.length)
    if end < lx1:
        top = layout.ruler[1] + layout.ruler[3]
        bottom = layout.markers_row[1]
        round_rect((end, bottom, lx1 - end + kit.KEY_R * 2 * s, top - bottom), 0, PAST_END)
    for b in layout.buttons:
        hovered = _hover is not None and _hover.kind == kit.BUTTON and _hover.id == b.id
        colour = BUTTON_ON if b.on else (BUTTON_HOVER if hovered else BUTTON_BG)
        round_rect(b.rect, b.rect[3] / 2, colour)


def _draw(context) -> None:
    layout = layout_for(context.region)
    if layout is None:
        return
    s = layout.scale
    c = anim_mode.clip()
    frame = anim_mode.frame()
    # The tabs and the bar.
    for b in layout.buttons:
        bx, by, bw, bh = b.rect
        label = clip_text(b.label, bw - 8 * s, 12 * s) if b.id.startswith("TAB_") else b.label
        text(label, bx + bw / 2, by + bh / 2, 12 * s, TEXT if b.on else TEXT_DIM, "centre")
    rx, ry, rw, rh = layout.readout
    text(f"{frame} / {c.length}", rx + rw / 2, ry + rh / 2, 13 * s, TEXT, "centre")
    # The ruler.
    gx, gy, gw, gh = layout.ruler
    text("PARTS", gx + 4 * s, gy + gh / 2, 10 * s, TEXT_FAINT)
    ticks = kit.ruler_ticks(layout, c.fps)
    pts = []
    for f, labelled in ticks:
        x = layout.x_of(f)
        pts += [(x, gy), (x, gy + (gh * 0.45 if labelled else gh * 0.25))]
        if labelled:
            text(str(f), x, gy + gh * 0.7, 10 * s, TEXT_FAINT, "centre")
    pen.lines(pen.to_3d(pts), TEXT_FAINT, 1.0 * s)
    # The rows.
    offset = (_drag or {}).get("offset", 0)
    picked = set(picked_keys())
    for row in layout.rows:
        bx, by, bw, bh = row.rect
        cy = by + bh / 2
        if row.kind == kit.PART_ROW:
            has_keys = bool(row.keys)
            indent = kit.INDENT * row.depth * s
            if has_keys:
                tri = (row.rect[0] + 8 * s + indent, cy)
                r = 3.5 * s
                points = ([(tri[0] - r, cy + r * 0.6), (tri[0] + r, cy + r * 0.6), (tri[0], cy - r * 0.8)]
                          if row.open else [(tri[0] - r * 0.6, cy - r), (tri[0] - r * 0.6, cy + r), (tri[0] + r * 0.8, cy)])
                pen.fan(points, TEXT_DIM)
            name = clip_text(row.label, kit.NAME_W * s - 26 * s - indent, 12 * s)
            text(name, bx + 18 * s + indent, cy, 12 * s, TEXT if row.picked else TEXT_DIM)
            if not has_keys and row.picked:
                text("K keys the pose · or pick a Motion", layout.lane[0], cy, 11 * s, TEXT_FAINT)
        elif row.kind == kit.COLOUR_ROW:
            text(row.label, bx + 18 * s, cy, 12 * s, TEXT_DIM)
        elif row.kind == kit.FRAMES_ROW:
            indent = kit.INDENT * row.depth * s
            text(row.label, bx + 30 * s + indent, cy, 11 * s, TEXT_DIM)
            text(f"{row.value} frames", bx + kit.NAME_W * s - 10 * s, cy, 11 * s, TEXT_FAINT, "right")
            # Blocks: from each key to the next (or the clip's end), labelled with the frame shown.
            starts = list(row.keys) + [(c.length, None)]
            shown_from_0 = row.keys and row.keys[0][0] == 0
            if not shown_from_0:
                starts.insert(0, (0, None))
            labels = list(row.values)
            if not shown_from_0:
                labels.insert(0, 1)
            for (t0, _e), (t1, _e1), label in zip(starts, starts[1:], labels):
                xa, xb = layout.x_of(t0), layout.x_of(t1)
                picked_block = _row_key_picked(row, t0, picked)
                round_rect((xa + 1 * s, cy - 8 * s, max(2 * s, xb - xa - 2 * s), 16 * s), 4 * s,
                           EMBER if picked_block else (SPAN if label > 1 else SPAN_HOLD))
                if xb - xa > 14 * s:
                    text(str(label), (xa + xb) / 2, cy, 10 * s, TEXT, "centre")
            continue
        else:
            text(row.label, bx + 30 * s, cy, 11 * s, TEXT_DIM)
            text(row.value, bx + kit.NAME_W * s - 10 * s, cy, 11 * s, TEXT_FAINT, "right")
        # Spans between keys, then the keys.
        keys = row.keys
        for (t0, e0), (t1, _e1) in zip(keys, keys[1:]):
            xa, xb = layout.x_of(t0), layout.x_of(t1)
            pen.rect_fill((xa, cy - 1.5 * s, xb - xa, 3 * s), SPAN_HOLD if e0 == anim.HOLD else SPAN)
        for t, ease in keys:
            is_picked = _row_key_picked(row, t, picked)
            shown_t = t + offset if is_picked else t
            x = layout.x_of(shown_t)
            r = kit.KEY_R * s
            pen.fan(kit.key_shape(ease, x, cy, r + 1.6 * s), KEY_RING)
            pen.fan(kit.key_shape(ease, x, cy, r), EMBER if is_picked else KEY_FILL)
    if layout.hint:
        text(layout.hint, layout.lane[0], layout.ruler[1] - kit.ROW_H * s / 2, 11 * s, TEXT_FAINT)
    # The Markers row: a triangle and a name per marker, the picked one in ember.
    mx, my, mw, mh = layout.markers_row
    mcy = my + mh / 2
    text("Markers", mx + 18 * s, mcy, 11 * s, TEXT_FAINT)
    if not layout.markers:
        text("Double-click here: a named moment games can read", layout.lane[0], mcy, 11 * s, TEXT_FAINT)
    drag = _marker_drag or {}
    for t, name in layout.markers:
        shown_t = t + drag.get("offset", 0) if drag.get("frame") == t else t
        x = layout.x_of(shown_t)
        w = kit.MARKER_W * s
        colour = EMBER if t == _picked_marker else VIOLET_LIGHT
        pen.fan([(x - w / 2, my + mh - 4 * s), (x + w / 2, my + mh - 4 * s), (x, my + mh - 4 * s - w * 0.9)], colour)
        text(clip_text(name, 90 * s, 10 * s), x + w, mcy - 2 * s, 10 * s, TEXT if t == _picked_marker else TEXT_DIM)
    if _box is not None and _box.get("moved"):
        x0, x1 = sorted((_box["x0"], _box["x1"]))
        y0, y1 = sorted((_box["y0"], _box["y1"]))
        pen.rect_fill((x0, y0, x1 - x0, y1 - y0), _rgba("E85D3A", 0.12))
        round_outline((x0, y0, x1 - x0, y1 - y0), 0, EMBER, 1.0 * s)
    # The play head, over every row.
    x = layout.x_of(frame)
    top = layout.ruler[1] + layout.ruler[3]
    bottom = layout.markers_row[1]
    pen.lines(pen.to_3d([(x, bottom), (x, top)]), EMBER, 1.5 * s)
    pen.fan([(x - 5 * s, top), (x + 5 * s, top), (x, top - 7 * s)], EMBER)


def _row_key_picked(row: kit.Row, t: int, picked) -> bool:
    if row.channel is None:
        return any(u == row.uid and f == t for u, _ch, f in picked)
    return (row.uid, row.channel, t) in picked


registry.add_draw("anim_dock_bg", 690, "PIXEL", _draw_bg)
registry.add_draw("anim_dock", 691, "PIXEL", _draw)
