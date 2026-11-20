"""The voxel tools: the tool, mode and mirror settings, their operators, and
rok.stroke.

`rok.stroke` runs from a mouse press
to the release. The keymap starts it on a left or right press. The operator
drives one gesture object from `rok.core` (Stroke, ShapeGesture,
FaceGesture, PickGesture, MoveGesture, ScreenGesture, GizmoDrag or
CornerDrag) and knows nothing else about the modes.

`invoke` tests, in this order, and the first hit wins: the rail, the file
label, the palette strip and the 256 grid, the selection pill strip, the
scene level, the corner gizmo, the gizmo handles, a nearer inactive model
under the press ray, the screen modes (Rect, Lasso), then the voxel hit of
that ray on the visible voxels.

Left out: the classic header rows (`draw_tool_rows`) and everything of
Animate mode (the
timeline dock, keys, the anchor dot, playback). The view keys and Ctrl-hold
orbit live in view_ops.py.
"""

from __future__ import annotations

import math
import time
import traceback

import numpy as np

from rok.core import brick, cage, edit, gizmo, hud, palette, scene_move, shape, steady_hand
from rok.core import chrome as chrome_kit
from rok.core.chrome import LEVEL_BRICK, LEVEL_MODEL, LEVEL_SCENE
from rok.core.cage_drag import CageDrag
from rok.core.edit import ATTACH, CLONE, GRADIENT, RECT, SCREEN_MODES, SELECT, STAMP_MODES, VOXEL, ShaderStroke
from rok.core.gesture import ShapeGesture, make_gesture
from rok.core.gizmo import GizmoDrag, skip_for
from rok.core.select import MoveGesture
from rok.core.undo import ModelRecord, StrokeRecorder, selection_record, stacked_record

from . import cage_ops, mode, opkit, overlay, pattern_ops
from . import object as voxel_object
from .app import RokPrefs
from .edge_wrap import MouseWrap
from .handlers import registry
from .object import document
from .opkit import (
    CONTEXT,
    BoolProp,
    BoolVectorProp,
    EnumProp,
    IntProp,
    Action,
    register,
)
from .ops_common import _late, refresh_views
from .state import Hover, cage_screen_for, gizmo_screen_for, state

SAMPLE_PX = 3.0  # a Voxel stroke casts a ray every this many pixels of the mouse's path
STROKE_SAMPLES = 64  # and this many between two events at most
GAP_SLOPE = 3.0  # a stroke fills between two hits up to this many times the screen step apart


class _WriteFilter:
    """A stroke's writes through a test: `allow(cells (N, 3)) -> (N,) bool`.
    `blocked` counts the writes it dropped, for the note when a stroke
    changed nothing."""

    def __init__(self, allow, set_, set_many) -> None:
        self.allow = allow
        self._set = set_
        self._set_many = set_many
        self.blocked = 0

    def set(self, x: int, y: int, z: int, value: int) -> bool:
        if not bool(self.allow(np.array([[x, y, z]]))[0]):
            self.blocked += 1
            return False
        return self._set(x, y, z, value)

    def set_many(self, cells, values, **kw) -> int:
        cells = np.asarray(cells).reshape(-1, 3)
        if len(cells) == 0:
            return 0
        keep = np.asarray(self.allow(cells), dtype=bool)
        self.blocked += int((~keep).sum())
        values = np.asarray(values)
        if values.ndim > 0 and len(values) == len(cells):
            values = values[keep]
        return self._set_many(cells[keep], values, **kw) if keep.any() else 0


def _shader_brush_on(tool: str, mode_name: str) -> bool:
    """True when a press paints with the armed shader brush: Paint in Voxel mode."""
    brush = _mod("shader_brush")
    return brush is not None and brush.armed is not None and tool == edit.PAINT and mode_name == edit.VOXEL


def _inside(mask: np.ndarray):
    """Paint or Erase while something is selected: only selected cells change."""
    return lambda cells: mask[cells[:, 0], cells[:, 1], cells[:, 2]]


def _unmasked(mask: np.ndarray):
    """Erase and Paint with a mask: masked voxels stay as they are."""
    return lambda cells: ~mask[cells[:, 0], cells[:, 1], cells[:, 2]]


_FACES6 = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int64)


def _not_onto(mask: np.ndarray, voxels: np.ndarray, normal_now):
    """Attach with a mask: no new voxel touches a masked one
    (checking only the voxel under it let a brush or a box reach a masked voxel
    from the side). A cell is not built when
    - one of its six faces touches a masked voxel, or
    - the first filled voxel behind it, against the press's face normal, is
      masked (a box standing on the mask), or
    - a cell between it and that voxel was not built for the rules above, so
      nothing is left floating over a gap."""
    size = np.asarray(voxels.shape, dtype=np.int64)

    def allow(cells):
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        # Touching the mask, by a face.
        near = cells[:, None, :] + _FACES6[None, :, :]
        inside = ((near >= 0) & (near < size)).all(axis=2)
        touch = np.zeros(near.shape[:2], dtype=bool)
        touch[inside] = mask[near[inside][:, 0], near[inside][:, 1], near[inside][:, 2]]
        ok = ~touch.any(axis=1)
        n = normal_now()
        if n is None or not any(n):
            return ok
        blocked = np.zeros(voxels.shape, dtype=bool)
        bad = cells[~ok]
        blocked[bad[:, 0], bad[:, 1], bad[:, 2]] = True
        step = np.asarray(n, dtype=np.int64)
        open_ = ok.copy()  # still walking back
        p = cells.copy()
        for _ in range(int(size.max())):
            if not open_.any():
                break
            p[open_] -= step
            open_ &= ((p >= 0) & (p < size)).all(axis=1)  # past the grid: nothing masked under it
            idx = np.flatnonzero(open_)
            if len(idx) == 0:
                break
            q = p[idx]
            gap = blocked[q[:, 0], q[:, 1], q[:, 2]]
            ok[idx[gap]] = False  # over a cell that is not built
            open_[idx[gap]] = False
            idx, q = idx[~gap], q[~gap]
            filled = voxels[q[:, 0], q[:, 1], q[:, 2]] != 0
            hit = idx[filled]
            ok[hit] = ~mask[p[hit, 0], p[hit, 1], p[hit, 2]]
            open_[hit] = False
        return ok

    return allow


INSIDE_NOTE = "Paint stays inside the selection · Shift+P paints all of it"
ERASE_INSIDE_NOTE = "Erase stays inside the selection · Delete erases all of it"
MASK_NOTE = "Masked voxels stay as they are · Option+M clears the mask"


# Pill ids invoke these operators. Keys and Cmd+K still reach them too.
# Each one finishes in one click.
_CAGE_GLYPH_OPS = {
    "PAINT": "paint_selection",
    "MASK": "mask_selection",
    "SEL_GROW": "select_grow",
    "SEL_SHRINK": "select_shrink",
    "INVERT": "select_invert",
    "BEVEL": "bevel",
    "SMOOTH": "smooth",
    "SPLIT": "model_split",
}

STROKE_BUDGET_MS = 16.0
ERROR_PRINT_INTERVAL_S = 60.0
TRACKPAD_STEP_PX = 24  # two-finger scroll this far on a trackpad for one hollow step
SCREEN_MODE_HINT = "Rect and Lasso work with Erase, Paint and Select"

_STROKE_TOOL_ITEMS = (
    ("ACTIVE", "Active", "Use the tool that the rail shows"),
    ("ATTACH", "Attach", "Add voxels, whatever the rail shows"),
    ("ERASE", "Erase", "Remove voxels, whatever the rail shows"),
    ("PAINT", "Paint", "Recolour voxels, whatever the rail shows"),
)

# Time of the last traceback printed by rok.stroke; one print per minute at most.
_last_error_time = 0.0

_LABELS = {ident: label for ident, label, _tip in (*edit.TOOL_ITEMS, *edit.MODE_ITEMS)}

# (text, monotonic time) of the last automatic switch; the status line shows it a while.
switch_note: tuple[str, float] | None = None
SWITCH_NOTE_S = 6.0


def _mod(name: str):
    """A sibling module, or None while it is not imported. Looked up at call time."""
    import importlib

    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ImportError:
        return None


def _note(text: str) -> None:
    global switch_note
    switch_note = (text, time.monotonic())
    app = CONTEXT.app
    if app is not None:
        app.say(text, SWITCH_NOTE_S)
    voxel_object.redraw_statusbar()


_last_tool = ATTACH  # the tool before the last change: a mirror change runs the same update


def _on_tool_change(self, context) -> None:
    """The action, mode or mirror changed: recompute the hover so the highlight follows.

    Attach has no screen modes: choosing Attach in Marquee or Lasso goes back
    to Voxel, and the status line says so. Switching to Select starts it in
    Marquee unless it is in Marquee or Lasso already: Select's other modes
    (Box, Face, ...) carried over from Attach, and nothing on the rail shows
    them for Select.
    """
    global _last_tool
    wm = context.prefs
    tool = getattr(wm, "rok_tool", None)
    switched = tool != _last_tool
    _last_tool = tool
    if switched and tool == SELECT and getattr(wm, "rok_mode", None) not in SCREEN_MODES:
        wm.rok_mode = RECT  # runs _on_mode_change
        return
    if tool != ATTACH and getattr(wm, "rok_mode", None) == CLONE:
        wm.rok_mode = edit.PATTERN  # Clone is Attach's; the others stamp the pattern
        return
    if tool != edit.PAINT and getattr(wm, "rok_mode", None) == GRADIENT:
        wm.rok_mode = VOXEL  # Gradient is Paint's
        return
    if tool == ATTACH and getattr(wm, "rok_mode", None) in SCREEN_MODES:
        _note(f"{_LABELS[wm.rok_mode]} works with Erase, Paint and Select: mode set to Voxel")
        wm.rok_mode = VOXEL  # runs _on_mode_change, which finds a legal pair
        return
    _remember_mode(wm)
    _late("gizmo_ops", "refresh")()
    mode.rehover()


def _remember_mode(wm) -> None:
    """Keep the tool's current mode (and Attach's solid) for the star's tool chips."""
    tool = getattr(wm, "rok_tool", None)
    if tool is not None:
        state.tool_modes[tool] = (getattr(wm, "rok_mode", VOXEL), getattr(wm, "rok_solid", "BOX"))


def _on_mode_change(self, context) -> None:
    """Marquee or Lasso under Attach: the action becomes Select, and
    the status line says so."""
    wm = context.prefs
    if wm.rok_mode in SCREEN_MODES and getattr(wm, "rok_tool", None) == ATTACH:
        _note(f"{_LABELS[wm.rok_mode]} works with Erase, Paint and Select: action set to Select")
        wm.rok_tool = SELECT  # runs _on_tool_change
        return
    _remember_mode(wm)
    _late("gizmo_ops", "refresh")()
    mode.rehover()


def fresh_note(now: float | None = None) -> str | None:
    """The last switch note while it is fresh, else None."""
    if switch_note is None:
        return None
    text, when = switch_note
    if (time.monotonic() if now is None else now) - when > SWITCH_NOTE_S:
        return None
    return text


def _on_brush_size_change(self, context) -> None:
    """The overlay draws the brush sphere from the size; nothing else redraws it."""
    voxel_object.redraw_3d_views()


def _print_error_throttled() -> None:
    global _last_error_time
    now = time.monotonic()
    if now - _last_error_time >= ERROR_PRINT_INTERVAL_S:
        _last_error_time = now
        traceback.print_exc()


def _run_op(name: str, **props) -> None:
    """Run rok.<name> with INVOKE. A missing operator is reported, not raised."""
    if f"rok.{name}" not in opkit.REGISTRY:
        print(f"Rok: operator rok.{name} is not registered")
        return
    op = getattr(opkit.ops.rok, name)
    if op.poll():
        op("INVOKE", **props)


def _set_viewer(gesture, tool, view_state) -> None:
    """With "Visible only" on, a Select gesture takes only what this view sees:
    the camera in grid coordinates (Rok's view is always in perspective)."""
    if gesture is None or tool != SELECT or view_state is None:
        return
    if not getattr(CONTEXT.prefs, "rok_select_visible", False):
        return
    origin = np.asarray(voxel_object.model_origin(document.model), dtype=np.float64)
    gesture.viewer = {"eye": tuple(float(v) for v in np.asarray(view_state.camera.eye(), dtype=np.float64) - origin)}


def _projector(region, view_state, origin):
    """(N, 3) grid points -> (N, 2) region pixels, for the screen gestures."""
    vp = np.asarray(view_state.screen_matrix, dtype=np.float64)
    shift = np.eye(4)
    shift[:3, 3] = origin
    matrix = vp @ shift
    width, height = int(region.width), int(region.height)

    def project(points):
        return gizmo.project(matrix, points, width, height)

    return project


@register
class SetToolAction(Action):
    command_id = "rok.set_tool"
    title = "Set Voxel Tool"
    tooltip = "Switch the tool that a click or drag runs"
    tool = EnumProp(name="Tool", description="The tool to switch to", items=edit.TOOL_ITEMS)

    @classmethod
    def poll(cls, context):
        return state.level == LEVEL_MODEL  # the scene's keys (G, R...) are its own, and Brick mode's

    def invoke(self, context, event):
        if event.type == "R" and opkit.ops.rok.pattern_turn.poll():
            return {"PASS"}  # R turns the pattern in Pattern mode
        return self.execute(context)

    def execute(self, context):
        context.prefs.rok_tool = self.tool
        return {"FINISHED"}


@register
class SetModeAction(Action):
    command_id = "rok.set_mode"
    title = "Set Voxel Mode"
    tooltip = "Switch the shape that a click or drag makes"
    mode = EnumProp(name="Mode", description="The shape to switch to", items=edit.MODE_ITEMS)

    @classmethod
    def poll(cls, context):
        return state.level == LEVEL_MODEL  # C is the Cut tool in the scene; Brick mode has its own keys

    def execute(self, context):
        context.prefs.rok_mode = self.mode
        return {"FINISHED"}


@register
class ToggleMirrorAction(Action):
    command_id = "rok.toggle_mirror"
    title = "Toggle Mirror Axis"
    tooltip = "Turn mirroring on or off for one axis"
    axis = IntProp(name="Axis", description="0 is X, 1 is Y, 2 is Z", min=0, max=2)

    def execute(self, context):
        wm = context.prefs
        flags = list(wm.rok_mirror)
        flags[self.axis] = not flags[self.axis]
        # A new tuple, not an item write: only a whole assignment runs the update.
        wm.rok_mirror = tuple(flags)
        app = getattr(context, "app", None)
        if app is not None and app._started:
            app.set_menu()  # the View menu's check marks
        return {"FINISHED"}


def _type_size() -> None:
    """The grid size label was clicked: open the field to type the size in
    (rok.grid_size_type, in model_ops.py)."""
    if "rok.grid_size_type" in opkit.REGISTRY and opkit.ops.rok.grid_size_type.poll():
        opkit.ops.rok.grid_size_type("INVOKE")


class _Press:
    """A mouse press in the view, in region pixels, with what the routing of
    the press looks up once."""

    def __init__(self, context, event) -> None:
        self.context = context
        self.event = event
        self.region = context.region
        self.view_state = context.view_state
        self.mx = event.mouse_region_x
        self.my = event.mouse_region_y
        self.left = event.type == "LEFT_MOUSE"
        self.grab = None  # (corner, handle) under the press, once looked up


def _designed_start(tool: str, mode_name: str, radius: int, colour: int):
    """The brush a designed stroke uses while the button is down: each curve's
    start value. Unchanged when the designer is off or this is not a voxel stroke."""
    if mode_name != VOXEL or tool not in (edit.ATTACH, edit.ERASE, edit.PAINT):
        return radius, colour, None
    from . import stroke_ops

    design = stroke_ops.design_of(CONTEXT.prefs)
    if design is None or not design.applies():
        return radius, colour, None
    spec = design.sample(0.0, radius, colour, tool)
    return spec.size, spec.colour, spec.exponent


# The name of a gizmo drag's undo step, by the first word of its handle.
GIZMO_LABELS = {"MOVE": "Move", "PLANE": "Move", "CENTER": "Move", "ROT": "Turn", "FLIP": "Flip", "SCALE": "Scale"}


@register
class StrokeAction(Action):
    command_id = "rok.stroke"
    title = "Voxel Stroke"
    tooltip = "Run the active tool where you click or drag"
    tool = EnumProp(name="Tool", description="The tool to run", items=_STROKE_TOOL_ITEMS, default="ACTIVE")

    @property
    def undo_label(self) -> str:
        """The name of this stroke's undo step: what a gizmo or cage drag did,
        else the tool and its mode ("Attach Box")."""
        gesture = self.__dict__.get("_gesture")
        if isinstance(gesture, CageDrag):
            return "Cage"
        if isinstance(gesture, GizmoDrag):
            word = gesture.handle.split("_")[0]
            if word == "FLIP" and state.level == LEVEL_SCENE:
                return "Stretch"  # in the scene these handles stretch the model
            return GIZMO_LABELS.get(word, "Transform")
        if isinstance(gesture, MoveGesture):
            return "Move"
        tool = self.__dict__.get("_tool")
        if not tool:
            return self.title
        mode_name = chrome_kit.MODE_NAMES.get(self.__dict__.get("_mode_name"), "")
        return f"{tool.title()} {mode_name}".strip()

    @classmethod
    def poll(cls, context):
        return document.grid is not None

    # Parts of a stroke that may not be set yet when a method reads them. Each
    # press makes a fresh operator, so these are the values before the press sets them.
    _copy_token = None
    _option_erase = False
    _sel_before = None
    _scroll = 0.0
    _gizmo_target = None
    _palette_record = None  # the colour store before this press added colours to it
    _saved_pivot = None

    def invoke(self, context, event):
        region = context.region
        if region is None or context.view_state is None:
            return {"PASS"}
        press = _Press(context, event)
        done = self._ui_press(press)
        if done is not None:
            return done
        if getattr(context.app, "cad", False):
            return {"PASS"}  # the CAD view shows the model; painting waits until it is off
        if event.ctrl and not event.shift and not event.alt:
            return {"PASS"}  # the Ctrl+LMB item is for the palette card's swap drag only
        return self._start_stroke(press)

    # -- the press: the UI drawn over the view

    def _ui_press(self, p):
        """Act on a press that belongs to something drawn over the view: the rail,
        the palette strip, the cage pills or the scene level. Returns the
        operator's result when the press was handled, None when it starts a stroke."""
        if self._panel_press(p):
            return {"FINISHED"}
        if state.level == LEVEL_BRICK and not state.animating:
            return _late("brick_mode", "press")(p)  # Brick mode: the bin and the bricks, never a stroke
        if state.level == LEVEL_SCENE:
            if state.animating and state.anim_paint and document.model is not None and p.left:
                _corner, handle = self._grab_at(p)
                if handle is None and _corner is None:
                    return None  # Animate mode's paint tool: a stroke on the picked part's frame
            return self._scene_level_press(p)
        if document.model is not None and document.scene.is_hidden(document.model):
            return {"FINISHED"}  # every model is on a hidden layer: nothing to edit or pick
        if document.grid is not None and document.grid.bricks is not None:
            _note(brick.LOCKED_MESSAGE)  # a brick model: Brick mode builds on it
            return {"CANCELLED"}
        return None

    def _panel_press(self, p) -> bool:
        """The left rail, the labels in the corner, the palette strip and the
        cage pills. True when one of them took the press: a press inside one
        that hits nothing is eaten too."""
        event = p.event
        chrome = _mod("chrome")
        if (chrome is not None and hasattr(chrome, "press")
                and chrome.press(p.context, p.region, p.mx, p.my, p.left, event.shift)):
            return True
        cad = _mod("cad_mode")
        if cad is not None and cad.press(p.context, p.mx, p.my, p.left):
            return True
        panel = _mod("stroke_panel")
        if panel is not None and panel.press(p.context, p.region, p.mx, p.my, p.left):
            return True
        dock = _mod("dock")
        if dock is not None and dock.press(p.context, p.region, p.mx, p.my, p.left, event.shift):
            return True
        strip = _mod("history_strip")
        if strip is not None and strip.press(p.context, p.region, p.mx, p.my, p.left, event.shift):
            return True
        part = mode.label_part_at(p.region, p.mx, p.my)
        if part is not None:
            self._label_press(part, p.left)
            return True
        return bool(
            (not state.animating  # Animate mode hides the palette strip
             and self._strip_press(p.context, p.region, p.mx, p.my, p.left, event.ctrl, event.shift))
            or self._cage_strip_press(p.region, p.mx, p.my, p.left)
            or self._cage_chip_press(p.region, p.mx, p.my, p.left)
        )

    @staticmethod
    def _label_press(part: str, left: bool) -> None:
        if left and part == "GRID_SIZE":
            _type_size()
        elif left and part == "LABEL_COLOUR":
            _run_op("colour_popup")  # the colour picker for the current colour

    @staticmethod
    def handle_at(region, view_state, mx, my) -> str | None:
        """The corner gizmo's handle or the 3D gizmo's handle under a region
        pixel, or None."""
        corner = overlay.corner_hit(region, mx, my)
        return corner if corner is not None else StrokeAction._gizmo_handle(region, view_state, mx, my)

    def _grab_at(self, p):
        """(corner, handle) under the press: the corner gizmo, else a handle of the
        3D gizmo, else None. `handle` is the corner's id when there is a corner.
        Looked up once per press."""
        if p.grab is None:
            corner = overlay.corner_hit(p.region, p.mx, p.my)
            handle = corner if corner is not None else self._gizmo_handle(p.region, p.view_state, p.mx, p.my)
            p.grab = (corner, handle)
        return p.grab

    def _scene_level_press(self, p):
        """A press at the scene level, with the Pick or the Cut tool. Returns the
        operator's result, or None for a press on the gizmo: the stroke moves the
        picked model."""
        _corner, handle = self._grab_at(p)
        if state.scene_tool == "CUT":
            scene_tools = _mod("scene_tools")
            aim = scene_tools.aim_cut(p.region, p.view_state, p.mx, p.my) if scene_tools is not None else None
            if p.left and aim is not None and "rok.scene_cut" in opkit.REGISTRY and opkit.ops.rok.scene_cut.poll():
                opkit.ops.rok.scene_cut(uid=aim[0], axis=aim[1], at=aim[2])
                mode.rehover()
            return {"FINISHED"}
        if handle is not None and handle.startswith(("ROT", "FLIP", "SCALE")) \
                and _late("brick_mode", "refuse_in_scene")():
            return {"FINISHED"}  # a brick model only moves, and turns by quarters with R
        stretch = "rok.scene_stretch" in opkit.REGISTRY and opkit.ops.rok.scene_stretch.poll()
        if handle == "SCALE_ALL":
            # The centre sphere: grow or shrink in all three dimensions, the
            # bottom on its floor; dragging up grows.
            if p.left and stretch:
                opkit.ops.rok.scene_stretch("INVOKE", handle="SH_Z+", uniform=True)
            return {"FINISHED"}
        if handle in gizmo.STRETCH_HANDLES:
            # The gizmo's cubes stretch the model in the scene: its face on
            # that axis's plus side moves, the opposite face stays put.
            if p.left and stretch:
                axis = gizmo.STRETCH_HANDLES.index(handle)
                opkit.ops.rok.scene_stretch("INVOKE", handle=f"SH_{'XYZ'[axis]}+")
            return {"FINISHED"}
        if handle is None:
            # The scene: a click picks a model, a double-click steps into it,
            # a drag moves it (Shift: leaves a copy behind; Option: up and
            # down). A click on nothing keeps the pick.
            if p.left:
                self._scene_press(p.region, p.view_state, p.mx, p.my, bool(p.event.shift))
            return {"FINISHED"}
        return None

    # -- the press: a stroke

    def _start_stroke(self, p):
        """A press that no UI took: run the action where the mouse is."""
        context, event = p.context, p.event
        wm = context.prefs
        forced = None if self.tool == "ACTIVE" else self.tool
        tool, subtract = edit.press_tool(wm.rok_tool, forced, event.alt, event.shift)
        holes = False
        mode_name = wm.rok_mode
        if state.animating and state.anim_paint and forced is None:
            # Animate mode's paint tool works on the picked part's frame, one voxel brush at a time.
            tool, subtract = edit.press_tool(state.anim_paint, None, event.alt, event.shift)
            mode_name = VOXEL
        elif wm.rok_mode in STAMP_MODES:
            tool, holes = pattern_ops.stamp_tool(tool)  # Carve stamps as Erase; Holes sinks in
        sel = document.selection
        sel_version = sel.version if sel is not None else None
        self._reset_stroke(tool, mode_name, event.alt)
        self._holes = holes
        try:
            gesture, done = self._press_gesture(p, tool, subtract, sel_version)
            if done is not None:
                return done
            self._arm(context, event, gesture)
        except Exception:
            self._abort_press()
            return {"CANCELLED"}
        return {"RUNNING"}

    def _reset_stroke(self, tool: str, mode_name: str, alt: bool) -> None:
        self._option_erase = bool(alt) and tool == edit.ERASE
        self._recorder = StrokeRecorder(document.grid)
        self._mode_name = mode_name
        self._tool = tool
        self._gizmo_target = None
        self._wrap = None  # a gizmo drag's mouse path; it goes on past the view's edge
        self._copy_token = None  # a Shift-drag's copy, shown since the press
        self._saved_pivot = None
        self._palette_record = None  # a Pattern stamp that fitted colours in
        sel = document.selection
        set_, set_many = self._recorder.set, self._recorder.set_many
        self._masked = None
        mask = _late("select_ops", "mask_of")()
        if mask is not None and tool in (edit.ATTACH, edit.ERASE, edit.PAINT):
            if tool == edit.ATTACH:
                allow = _not_onto(mask, document.grid.voxels, self._press_normal)
            else:
                allow = _unmasked(mask)
            self._masked = _WriteFilter(allow, set_, set_many)
            set_, set_many = self._masked.set, self._masked.set_many
        # The shader brush keeps to the selection itself (its scope), and may add
        # voxels next to it with Room: no inside-the-selection filter then.
        brushing = _shader_brush_on(tool, mode_name)
        self._inside = (
            _WriteFilter(_inside(sel.mask), set_, set_many)
            if tool in (edit.PAINT, edit.ERASE) and sel is not None and not sel.is_empty() and not brushing
            else None
        )
        self._filter_gesture = None
        # The mask before the stroke, for the SelectionRecord. Only a Select
        # stroke or a stroke with a selection can change it; an Attach click on
        # a big grid with nothing selected copies nothing.
        self._sel_before = (
            sel.mask.copy() if sel is not None and (tool == edit.SELECT or not sel.is_empty()) else None
        )

    def _press_gesture(self, p, tool, subtract, sel_version):
        """(gesture, None) for a press that starts a drag, (None, result) for one
        that ends here. A press on a gizmo handle is the gizmo's, whichever button;
        the cage's pills were handled before, so a click on Hollow is not a stroke."""
        point = overlay.cage_hit(p.region, p.mx, p.my)
        if point is not None:
            return self._cage_gesture(p, point)
        corner, handle = self._grab_at(p)
        if corner is not None:
            return self._corner_gesture(p, corner)
        if handle is not None:
            return self._gizmo_gesture(p, handle)
        return self._tool_gesture(p, tool, subtract, sel_version)

    def _cage_gesture(self, p, point: str):
        """A point of the lattice cage: pull it and the real voxels bend."""
        if not p.left:
            return None, {"FINISHED"}
        gesture = cage_ops.build_drag(point, p.region, p.view_state, self._recorder,
                                      tuple(p.context.prefs.rok_mirror))
        if gesture is None:
            return None, {"CANCELLED"}
        gesture.begin_screen(p.mx, p.my)
        self._wrap = MouseWrap()
        state.hover = None
        state.hover_handle = point
        return gesture, None

    def _corner_gesture(self, p, corner: str):
        """The corner gizmo: move or scale the selection from the bottom right."""
        if not p.left:
            return None, {"FINISHED"}
        gesture = self._build_corner_drag(corner, p.event.shift)
        if gesture is None:
            return None, {"CANCELLED"}
        gesture.begin_screen(p.mx, p.my)
        self._wrap = MouseWrap()
        state.hover = None
        state.hover_handle = corner
        return gesture, None

    def _gizmo_gesture(self, p, handle: str):
        """A handle of the 3D gizmo."""
        if not p.left:
            return None, {"FINISHED"}
        gizmo_ops = _mod("gizmo_ops")
        gesture = self._build_gizmo_drag(handle, p.region, p.view_state, p.event.shift)
        self._wrap = MouseWrap()
        gesture.set_mouse(p.mx, p.my)
        origin, direction = mode.gizmo_ray(p.region, p.view_state, p.mx, p.my)
        if not gesture.begin_ray(origin, direction):
            return None, {"CANCELLED"}
        state.hover = None
        state.hover_handle = handle
        if gesture.is_pivot and gizmo_ops is not None:
            self._saved_pivot = gizmo_ops.save_pivot()  # Esc puts it back
        if gesture.target == gizmo.MODEL and gizmo_ops is not None:
            # Shift on an arrow: the copy stands at the model's place from
            # the start of the drag.
            scene_tools = _mod("scene_tools")
            if gesture.copy and scene_tools is not None and not state.animating:
                self._copy_token = scene_tools.begin_copy(scene_tools.group_models())
            gizmo_ops.begin_model_drag(gesture)
            # The press already has a result: a flip, or one quarter turn on
            # a ring. Show its ghost now; extend() reports no change for them.
            # A pivot drag has nothing to show yet, and publishing its result
            # here would pin the pivot on a press that never moved.
            if not gesture.is_pivot:
                gizmo_ops.update_model_drag(gesture.result())
        return gesture, None

    def _tool_gesture(self, p, tool, subtract, sel_version):
        """The gesture of the active tool, for a press on no gizmo."""
        context, event = p.context, p.event
        mode_name = self._mode_name
        mirror = tuple(context.prefs.rok_mirror)
        # A press on a nearer, faded model: a double-click goes into it,
        # a single press is eaten, so nobody leaves a model by accident.
        # Decided from the press ray, not from the hover, which can be a
        # repick behind (a press right after a zoom).
        hit, other, other_hit = mode.pick(p.region, p.view_state, p.mx, p.my, tool == ATTACH)
        if other_hit is not None:
            if p.left:
                self._step_into_or_note(other)
            return None, {"FINISHED"}
        # Rect and Lasso start anywhere on screen.
        if mode_name in edit.SCREEN_MODES:
            gesture = self._build_screen_gesture(p.region, p.view_state, tool, mode_name, mirror, event.shift, subtract)
            if gesture is None:
                self.report({"INFO"}, SCREEN_MODE_HINT)
                return None, {"CANCELLED"}
            gesture.begin_screen(p.mx, p.my)
            return gesture, None
        return self._voxel_gesture(p, hit, tool, mirror, subtract, sel_version)

    @staticmethod
    def _step_into_or_note(other) -> None:
        """A press on a faded model: the second press of a double-click steps into it."""
        now = time.monotonic()
        if chrome_kit.is_double_click(state.last_pick, other.uid, now):
            state.last_pick = None
            _late("level_ops", "step_into")(other.uid)
            mode.rehover()
        else:
            state.last_pick = (other.uid, now)

    def _voxel_gesture(self, p, hit, tool, mirror, subtract, sel_version):
        """The voxel hit, on the visible voxels."""
        sel = document.selection
        shift = p.event.shift
        if hit is None:
            # Off the grid box: a plain Select click there deselects,
            # as a click on the floor inside it does.
            if p.left and tool == edit.SELECT and not shift and not subtract and sel is not None and not sel.is_empty():
                sel.clear()
                self._commit_selection()
                self._redraw_selection(p.context)
                _late("gizmo_ops", "refresh")()
            return None, {"CANCELLED"}
        gesture = self._build_gesture(hit, tool, self._mode_name, mirror, shift, subtract)
        if gesture is None or not gesture.begin(hit):
            self._revert_palette()
            if gesture is None and self._mode_name in STAMP_MODES:
                self.report({"INFO"}, pattern_ops.NO_PATTERN_HINT)
            if sel is not None and sel.version != sel_version:
                # A plain Select press on nothing deselected: one undo step.
                self._commit_selection()
                self._redraw_selection(p.context)
                _late("gizmo_ops", "refresh")()
            return None, {"CANCELLED"}
        return gesture, None

    def _arm(self, context, event, gesture) -> None:
        """The press has a gesture: show it and take the drag."""
        self._gesture = gesture
        self._button = event.type  # 'LEFT_MOUSE' or 'RIGHT_MOUSE'
        self._last_mouse = (float(event.mouse_region_x), float(event.mouse_region_y))  # for _sample_between
        self._steady = self._last_mouse  # where Steady Hand holds the brush
        self._stage = 1  # 2 while a Box or Center waits for its height
        self._moves = 0
        self._max_ms = 0.0
        self._changed = 0
        self._started = time.perf_counter()
        if self._gizmo_target is None:
            _late("gizmo_ops", "refresh")()  # a plain Select press may have cleared the selection
        # The handler last: nothing that can raise may follow it. If _flush
        # raises, the caller returns CANCELLED with no handler behind.
        self._flush(context)
        context.prefs.add_modal(self)
        state.stroke_active = True

    def _abort_press(self) -> None:
        """Something raised while the press was set up: undo what it did."""
        _print_error_throttled()
        self._recorder.revert()
        self._revert_palette()
        self._restore_selection()
        voxel_object.flush_dirty()
        state.preview = None
        state.stroke_active = False
        if self._gizmo_target == gizmo.MODEL:
            _late("gizmo_ops", "end_model_drag")(cancel=True)
            if self._copy_token is not None:
                _late("scene_tools", "cancel_copy")(self._copy_token)

    # -- the press: helpers

    def _strip_press(self, context, region, mx, my, left: bool, ctrl: bool = False, shift: bool = False) -> bool:
        """Act on a strip or grid click. True when the press was inside either.

        Ctrl or Shift on a swatch starts a palette drag (swap or gradient)
        instead of picking it.
        """
        wm = context.prefs
        if not overlay.is_main_region(region):
            return False  # the strip draws in the main view only
        strip = overlay.strip_layout(region)
        dock_item = overlay.dock_hit(region, mx, my)
        if dock_item is not None:
            if left:
                overlay.dock_click(region, dock_item)
            region.tag_redraw()
            return True
        item = hud.strip_hit_test(strip, mx, my)
        if item is not None:
            if item[0] == "FAV":
                # A favourite colour: a click uses it, a right-click names it.
                from . import palette_ui

                favs = palette_ui.favourites()
                if item[1] < len(favs):
                    if left:
                        if palette_ui.use_favourite(favs[item[1]]) is None:
                            _note(palette_ui.PALETTE_FULL)
                    else:
                        _run_op("favourite_name", hex=palette_ui._hex(favs[item[1]]))
                region.tag_redraw()
                return True
            if left:
                kind, index = item
                if kind == "SWATCH":
                    self._swatch_press(index, ctrl, shift)
                elif kind == "CURRENT":
                    _run_op("colour_popup")
                elif kind == "NEW":
                    _run_op("colour_new")
                elif kind == "EYEDROP":
                    _run_op("eyedrop")
                elif kind == "FAVOURITE":
                    from . import palette_ui

                    palette_ui.toggle_favourite()
                    overlay.invalidate_palette()
                elif kind == "GRID":
                    wm.rok_show_palette_grid = not wm.rok_show_palette_grid
                region.tag_redraw()
            return True
        if hud.strip_contains(strip, mx, my):
            return True
        if overlay.palette_grid_shown():
            layout = overlay.palette_layout(region)
            idx = hud.hit_test(layout, mx, my)
            if idx is not None:
                if left:
                    self._swatch_press(idx, ctrl, shift)
                    region.tag_redraw()
                return True
            if hud.contains(layout, mx, my):
                return True
        return False

    @staticmethod
    def _pick_colour(index: int) -> None:
        if index < 1:
            return
        if index >= overlay.EXTRA_KEY:  # an unused colour without a slot gets one now
            index = _late("palette_ui", "pick_entry")(overlay.entry_of_key(index))
            if index is None:
                return
        document.colour = index

    def _swatch_press(self, index: int, ctrl: bool, shift: bool) -> None:
        """A left click on a used swatch or a 256-grid cell. Plain: pick it.
        Ctrl or Shift: start rok.palette_drag (swap on Ctrl, gradient on
        Shift; Ctrl wins when both are held) instead."""
        if (ctrl or shift) and "rok.palette_drag" in opkit.REGISTRY and opkit.ops.rok.palette_drag.poll():
            opkit.ops.rok.palette_drag("INVOKE", start_index=index, kind="SWAP" if ctrl else "GRADIENT")
        else:
            self._pick_colour(index)

    @staticmethod
    def _gizmo_handle(region, view_state, mx, my) -> str | None:
        screen = gizmo_screen_for(region)
        if state.gizmo is None or screen is None:
            return None
        radius = mode.GIZMO_HIT_PX * overlay.ui_scale()
        skip = skip_for(state.level, state.animating)
        return gizmo.hit_test(screen, mx, my, radius, mode.view_dir(view_state), skip=skip)

    @staticmethod
    def _cage_strip_press(region, mx, my, left: bool) -> bool:
        """Act on a press in the selection bar. True when the press was on its plate."""
        screen = cage_screen_for(region)
        if screen is None or screen.strip is None:
            return False
        strip = screen.strip
        if not cage.strip_contains(strip, mx, my):
            return False
        if left:
            name = cage.strip_hit(strip, mx, my)
            if name is not None:
                StrokeAction._run_cage_glyph(name)
        return True

    @staticmethod
    def _cage_chip_press(region, mx, my, left: bool) -> bool:
        """Act on a press on the lattice cage's chip row. True when the press was inside it."""
        if not overlay.cage_chip_contains(region, mx, my):
            return False
        name = overlay.cage_chip_hit(region, mx, my)
        if left and name is not None:
            cage_ops.press_chip(name)
            mode.rehover()
        return True

    @staticmethod
    def _run_cage_glyph(handle: str) -> None:
        """Fire a selection-bar action at once.

        EXEC, not INVOKE: the bar is for actions that finish in one click.
        """
        name = _CAGE_GLYPH_OPS.get(handle)
        if not name or f"rok.{name}" not in opkit.REGISTRY:
            return
        op = getattr(opkit.ops.rok, name)
        if not op.poll():
            return
        try:
            op("EXEC")
        except Exception:
            _print_error_throttled()  # never silent: a glyph that does nothing is a bug

    def _build_corner_drag(self, handle: str, shift: bool):
        """The corner gizmo's drag, with the cell lengths it was drawn with."""
        from rok.core.corner_gizmo import CornerDrag

        found = state.corner
        sel = document.selection
        if found is None or sel is None or sel.is_empty():
            return None
        grid = document.grid
        return CornerDrag(
            handle,
            grid.visible,
            sel,
            self._recorder.set_many,
            found[2],
            copy=shift,
            set_grid=self._recorder.set_grid,
            centre=found[1].centre,
            ring_sign=found[1].ring_sign,
        )

    def _build_gizmo_drag(self, handle: str, region, view_state, shift: bool) -> GizmoDrag:
        gs = state.gizmo
        grid = document.grid
        self._gizmo_target = gs.target
        return GizmoDrag(
            handle,
            gs.target,
            gs.pivot2,
            grid.visible,
            document.selection,
            self._recorder.set_many,
            shift=shift,
            view_dir=mode.view_dir(view_state),
            pivot_px=gizmo_screen_for(region).pivot,
            size=grid.size,
            set_grid=self._recorder.set_grid,
        )

    @staticmethod
    def _scene_press(region, view_state, mx, my, shift: bool) -> None:
        """A left press in the scene with the Pick tool.

        A double-click on the picked model steps into it: only a click on the
        model that was already picked starts a double-click,
        so two quick clicks on another model just pick it. A press on the
        picked model, even behind another, grabs it: a drag moves it, a click
        without moving picks the next model along the line of sight
        (click-cycling). A press elsewhere picks the nearest model under the
        mouse, and a drag moves that one at once."""
        scene_tools = _mod("scene_tools")
        under, solid = mode.models_hit(region, view_state, mx, my)
        now = time.monotonic()
        last = state.last_pick
        clear_group = scene_tools.clear_group if scene_tools is not None else (lambda: None)
        members = scene_tools.group_uids() if scene_tools is not None else {document.active_uid}
        if not shift and last is not None and last[0] in under and chrome_kit.is_double_click(last, last[0], now):
            state.last_pick = None
            clear_group()
            _late("level_ops", "step_into")(last[0])
            return
        if not under:
            state.last_pick = None
            if not shift and len(members) > 1:
                clear_group()  # a click on nothing: back to the picked model alone
                voxel_object.redraw_3d_views()
            return
        picked_before = document.active_uid
        # The model the press grabs: a group member with a voxel under the
        # mouse first (so one behind another can still be dragged), else the
        # first in click order (scene_move.choose_grab).
        grabbed = scene_move.choose_grab(under, solid, members)
        drag = {"copy": shift, "grab_x": mx, "grab_y": my, "pressed": grabbed}
        if shift:
            drag["toggle"] = True  # a click adds or takes out; a drag copies
        elif grabbed in members and len(members) > 1:
            drag["single"] = True  # a drag moves the group; a click picks this one alone
        elif grabbed == document.active_uid:
            drag["cycle"] = True  # a click picks the next one behind
        else:
            clear_group()
            StrokeAction._activate(grabbed)
        # Only a click on the model that was picked already can start a double-click.
        state.last_pick = (grabbed, now) if grabbed == picked_before else None
        if state.level == LEVEL_SCENE and "rok.scene_drag" in opkit.REGISTRY and opkit.ops.rok.scene_drag.poll():
            opkit.ops.rok.scene_drag("INVOKE", **drag)

    @staticmethod
    def _activate(uid: int) -> None:
        """Make the model under the press the active one and show it."""
        document.set_active(uid)
        refresh_views(hover=True)

    def _build_screen_gesture(self, region, view_state, tool, mode_name, mirror, shift, subtract=False):
        """A ScreenGesture with a projection into this region, or None for Attach."""
        grid = document.grid
        sel = document.selection
        if tool == SELECT and sel is not None and not shift and not subtract:
            sel.clear()  # a plain press replaces; Option (deselect) keeps the rest
        gesture = make_gesture(
            mode_name,
            tool,
            grid.visible,
            document.colour,
            mirror,
            *self._setters(),
            sel,
            project=_projector(region, view_state, voxel_object.model_origin(document.model)),
            subtract=subtract,
        )
        _set_viewer(gesture, tool, view_state)
        return gesture

    def _build_gesture(self, hit, tool, mode_name, mirror, shift, subtract=False):
        """The gesture object for this press, or None."""
        grid = document.grid
        sel = document.selection
        rec = self._recorder
        wm = CONTEXT.prefs
        # The brush sphere: every tool in Voxel mode. A designed stroke uses the
        # curves' start values while the button is down.
        radius = int(getattr(wm, "rok_brush_size", 1))
        colour = document.colour
        radius, colour, shape_exp = _designed_start(tool, mode_name, radius, colour)
        if tool == SELECT and sel is not None:
            # Option+Shift (subtract) never moves the selection, even on a selected
            # voxel: it removes cells there instead.
            if not subtract and not hit.wall and sel.contains(hit.voxel):
                return MoveGesture(grid.visible, sel, copy=shift, set_many=rec.set_many)
            if not shift and not subtract:
                sel.clear()  # a plain press replaces; a press on nothing deselects
        if mode_name == GRADIENT:
            if tool != edit.PAINT:
                return None
            gradient_ops = _mod("gradient_ops")
            gesture, snapshot = gradient_ops.gesture(grid.tool_voxels, self._setters()[1])
            self._palette_record = snapshot if gesture is not None else None
            return gesture
        pattern_box = None
        remap = None
        if mode_name in STAMP_MODES:
            pattern_box = pattern_ops.current_pattern()
            if pattern_box is None:
                return None
            if tool == edit.PAINT or (tool == ATTACH and mode_name == CLONE):
                # Paint and Clone keep the clip's own colours; a clip from another
                # file gets them fitted in now, so the stamp shows them during the
                # drag. Pattern's Attach stamps the shape in the current colour.
                clip_ops = _mod("clip_ops")
                if clip_ops is not None:
                    snapshot = document.scene.colours.snapshot()
                    remap, changed = clip_ops.fit_clip_palette()
                    self._palette_record = snapshot if changed else None
        brush = _mod("shader_brush")
        if brush is not None and brush.armed is not None:
            if _shader_brush_on(tool, mode_name):
                return self._shader_stroke(brush, mirror, radius)
            brush.stop()  # another tool or mode: the brush is put away
        gesture = make_gesture(
            mode_name,
            tool,
            grid.tool_voxels,  # the visible voxels; a cut cell takes no write
            colour,
            mirror,
            *self._setters(),
            sel,
            radius=radius,
            pattern_box=pattern_box,
            remap=remap,
            subtract=subtract,
            holes=getattr(self, "_holes", False),
            palette=document.scene.palette if document.scene is not None else None,
            tolerance=int(getattr(wm, "rok_colour_tolerance", 0)),
        )
        self._filter_gesture = gesture  # Attach's mask test reads its hit
        _set_viewer(gesture, tool, CONTEXT.view_state)
        if isinstance(gesture, ShapeGesture):
            # The primitives are Attach's; Erase, Paint and Select take a plain box.
            if tool == ATTACH:
                gesture.solid = getattr(wm, "rok_solid", shape.BOX_SOLID)
                gesture.wall = int(getattr(wm, "rok_hollow", 0))
                if document.scene is not None:
                    gesture.frame_colour = palette.darker_slot(document.scene.palette, int(document.colour))
            gesture.square = bool(shift)
            # A plain box writes as a block; a write filter (the mask, inside the
            # selection) wants the cells, so the block path stands aside then.
            if self._inside is None and self._masked is None:
                gesture.set_grid = self._recorder.set_grid
        elif isinstance(gesture, edit.Stroke) and type(gesture) is edit.Stroke:
            gesture.shape_exp = shape_exp
            if self._inside is None and self._masked is None:
                # The brush writes as boxes, not cell lists (a brush of
                # 50 on a 256-cube took 28 ms a move); a write filter wants the cells.
                gesture.use_box_writes(self._recorder.write_brush)
        return gesture

    def _shader_stroke(self, brush, mirror, radius):
        """A Paint stroke with the armed shader: its result for the model now,
        revealed where the brush goes. Its own colours go into the palette
        first, as one step with the stroke (self._palette_record)."""
        out = brush.target()
        if out.error:
            self.report({"ERROR"}, out.error)
            return None
        if out.writes:
            self._palette_record = document.scene.colours.snapshot()
            voxel_object.write_colours(out.writes, materials=out.materials)
        set_, set_many = self._setters()
        gesture = ShaderStroke(document.grid.tool_voxels, out.new, set_, mirror, brush=radius, set_many=set_many)
        self._filter_gesture = gesture
        _set_viewer(gesture, edit.PAINT, CONTEXT.view_state)
        return gesture

    def _setters(self):
        """(set, set_many) for a gesture: the recorder's, through the mask and
        the inside-the-selection filter (Paint, Erase) when they apply."""
        for f in (getattr(self, "_inside", None), getattr(self, "_masked", None)):
            if f is not None:
                return f.set, f.set_many
        return self._recorder.set, self._recorder.set_many

    def _press_normal(self):
        """The face normal of the gesture's hit, for Attach's mask test."""
        hit = getattr(getattr(self, "_filter_gesture", None), "hit", None)
        return None if hit is None else tuple(int(v) for v in hit.normal)

    def _revert_palette(self) -> None:
        """Take back the palette fit of a Pattern press that came to nothing."""
        voxel_object.revert_colours(self._palette_record)
        self._palette_record = None

    # -- the drag

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception:
            _print_error_throttled()
            self._cancel_drag(context)
            return {"CANCELLED"}

    def _modal(self, context, event):
        # The stroke consumes every event, so the hover operator sees none of them.
        # Keep the shared state current: the rehover at the end picks at the release
        # point with the real modifier keys.
        state.last_mouse = (event.mouse_x, event.mouse_y)
        state.ctrl = event.ctrl
        state.shift = event.shift
        state.alt = event.alt
        if self._stage == 2:
            return self._modal_height(context, event)
        if event.value == "PRESS" and event.type in ("X", "Y", "Z") and self._gesture.locks_axis:
            # X, Y or Z locks a selection move to one axis; the same key unlocks.
            self._gesture.toggle_axis_lock("XYZ".index(event.type))
            self._flush(context)  # the label changes even when the offset does not
            return {"RUNNING"}
        if (event.value == "PRESS" and not event.is_repeat and event.type in ("R", "UP_ARROW", "DOWN_ARROW")
                and hasattr(self._gesture, "set_round")):
            # A Rect: R steps through flat / round top / round ends / ellipse (Shift+R back);
            # Up and Down step too.
            g = self._gesture
            if event.type == "R":
                changed = g.step_round(-1 if event.shift else 1, wrap=True)
            else:
                changed = g.step_round(1 if event.type == "UP_ARROW" else -1)
            if changed:
                self._flush(context)
            return {"RUNNING"}
        if event.type in ("WHEEL_UP", "WHEEL_DOWN") and hasattr(self._gesture, "set_round"):
            # A mouse wheel rounds too; a trackpad's two-finger scroll is left to the view.
            if self._gesture.step_round(1 if event.type == "WHEEL_UP" else -1):
                self._flush(context)
            return {"RUNNING"}
        if event.type == "MOUSE_MOVE":
            started = time.perf_counter()
            if self._wrap is not None:
                mx, my = self._wrap.pos(context, event)
            else:
                mx, my = event.mouse_region_x, event.mouse_region_y
            g = self._gesture
            if isinstance(g, edit.Stroke) and self._gizmo_target is None:
                # Steady Hand: the brush trails the mouse on a string.
                length = steady_hand.length_of(context.prefs)
                mouse = (float(mx), float(my))
                self._steady = steady_hand.follow(self._steady, mouse, length)
                mx, my = self._steady
                taut = length > 0 and math.hypot(mouse[0] - mx, mouse[1] - my) > 1.0
                line = (self._steady, mouse) if taut else None
                if line != state.steady_line:
                    state.steady_line = line
                    if context.area is not None:
                        context.area.tag_redraw()  # the string moves even when no voxel changes
            g.set_shift(event.shift)
            if hasattr(g, "coarse"):
                g.coarse = bool(event.ctrl)  # a ring drag: 15 degree steps
            changed = False  # a gizmo drag never samples between events, and a first move may change nothing
            if g.screen_drag:
                changed = g.extend_screen(mx, my)
            else:
                g.set_mouse(mx, my)
                if self._gizmo_target is not None:
                    # A gizmo drag: the gizmo is drawn at rest (see mode.gizmo_ray).
                    origin, direction = mode.gizmo_ray(context.region, context.view_state, mx, my)
                else:
                    changed = self._sample_between(context, g, mx, my)
                    origin, direction = mode.mouse_ray(context.region, context.view_state, mx, my)
                changed = g.extend(origin, direction) or changed
                if changed and self._gizmo_target is not None:
                    self._gizmo_changed(g.result())
            self._last_mouse = (float(mx), float(my))
            if changed:
                self._flush(context)
            if state.profile:
                self._profile_move(started)
            return {"RUNNING"}
        if (event.type == self._button and event.value == "RELEASE") or event.type == "FOCUS_LOST":
            if self._gizmo_target == gizmo.MODEL:
                self._finish_model_drag(context)
            elif event.type != "FOCUS_LOST" and self._begin_height(context):
                return {"RUNNING"}  # the footprint is set; the mouse now sets the height
            else:
                self._finish(context)
            return {"FINISHED"}
        if event.type == "ESCAPE" and event.value == "PRESS":
            # Deferred gestures have written nothing, so ESC only drops the ghost.
            self._cancel_drag(context)
            return {"CANCELLED"}
        # Everything else is consumed: no orbit or zoom while a button is down.
        return {"RUNNING"}

    def _begin_height(self, context) -> bool:
        """Enter the height stage of a Box or Center when the gesture has one."""
        if not self._gesture.begin_height():
            return False
        self._stage = 2
        self._flush(context)
        return True

    def _modal_height(self, context, event):
        """The height stage: the mouse sets the height, a click confirms, Esc or the
        other button drops the whole shape. With Attach, scrolling (a wheel or two
        fingers on a trackpad) sets the hollow wall, 0 for solid; Rok keeps it
        for the next drag."""
        if self._gesture.tool == ATTACH and event.type in ("WHEEL_UP", "WHEEL_DOWN", "TRACKPAD_PAN"):
            if event.type == "TRACKPAD_PAN":
                self._scroll += event.dy
                steps = int(self._scroll / TRACKPAD_STEP_PX)
                self._scroll -= steps * TRACKPAD_STEP_PX
            else:
                steps = 1 if event.type == "WHEEL_UP" else -1
            if steps:
                wm = context.prefs
                wall = max(0, min(shape.WALL_MAX, int(wm.rok_hollow) + steps))
                if wall != self._gesture.wall:
                    wm.rok_hollow = wall
                    self._gesture.wall = wall
                    self._flush(context)
            return {"RUNNING"}
        if event.type == "MOUSE_MOVE":
            mx, my = event.mouse_region_x, event.mouse_region_y
            origin, direction = mode.mouse_ray(context.region, context.view_state, mx, my)
            if self._gesture.extend_height(origin, direction):
                self._flush(context)
            return {"RUNNING"}
        if event.type == self._button and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        if event.type == "FOCUS_LOST":
            self._finish(context)
            return {"FINISHED"}
        if event.value == "PRESS" and event.type in ("ESCAPE", "LEFT_MOUSE", "RIGHT_MOUSE"):
            self._cancel_drag(context)
            return {"CANCELLED"}
        return {"RUNNING"}

    def _sample_between(self, context, g, mx: float, my: float) -> bool:
        """A Voxel stroke samples the mouse's path, not only its events (else a
        fast Erase or Paint stroke skips cells between two events). One ray every SAMPLE_PX along the
        way from the last event to this one, STROKE_SAMPLES at most; a ray
        costs about 15 microseconds. True when a voxel changed."""
        last = getattr(self, "_last_mouse", None)
        if last is None or not isinstance(g, edit.Stroke):
            return False
        dx, dy = mx - last[0], my - last[1]
        dist = math.hypot(dx, dy)
        cam = context.app.camera
        per_voxel = cam.height / max(2.0 * cam.half_height, 1e-6)  # pixels a voxel spans at the target
        # A big brush needs fewer samples: one every half radius on screen keeps
        # the stroke whole, and a brush of 50 covers a million cells a sample.
        step_px = SAMPLE_PX
        if g.brush > 1:
            step_px = max(SAMPLE_PX, 0.5 * (g.brush - 1) * per_voxel)
        steps = min(STROKE_SAMPLES, int(dist // step_px))
        # Between two hits the stroke fills the way in the voxels, so a fast
        # stroke never breaks. A surface that slopes away from the view puts
        # the hits farther apart than the screen step; a jump of more than
        # that is a step past an edge onto something behind, not filled.
        g.max_gap = GAP_SLOPE * (dist / (max(steps, 0) + 1)) / max(per_voxel, 1e-6) + 2.0
        if steps < 1:
            return False
        changed = False
        for k in range(1, steps + 1):
            f = k / (steps + 1)
            origin, direction = mode.mouse_ray(context.region, context.view_state, last[0] + dx * f, last[1] + dy * f)
            if g.extend(origin, direction):
                changed = True
        return changed

    def _gizmo_changed(self, result) -> None:
        """A gizmo drag's integer result changed: move the pivot or the model live."""
        gizmo_ops = _mod("gizmo_ops")
        if gizmo_ops is None:
            return
        if result.kind == gizmo.PIVOT:
            gizmo_ops.set_pivot(result.pivot2)
        elif self._gizmo_target == gizmo.MODEL:
            gizmo_ops.update_model_drag(result)
            if result.kind == gizmo.MOVE:
                _late("scene_tools", "show_group_offset")(result.offset)  # the rest of the group follows

    def _cancel_drag(self, context) -> None:
        self._recorder.revert()
        self._revert_palette()
        self._restore_selection()
        voxel_object.flush_dirty()
        state.preview = None
        if self._gizmo_target == gizmo.MODEL:
            _late("gizmo_ops", "end_model_drag")(cancel=True)
            if self._copy_token is not None:
                _late("scene_tools", "cancel_copy")(self._copy_token)
                self._copy_token = None
        if self._saved_pivot is not None:
            _late("gizmo_ops", "restore_pivot")(self._saved_pivot)
        if isinstance(getattr(self, "_gesture", None), CageDrag) and state.cage is not None:
            state.cage.live = None  # Esc: the points stay where they were
            cage_ops.adopt()
        self._end(context)

    def _flush(self, context) -> None:
        """Push the changed chunks to the GPU and show the gesture's state."""
        voxel_object.flush_dirty()
        g = self._gesture
        if g.hit is not None:
            state.hover = Hover(g.hit, g.tool, g.last, mode=self._mode_name)
        else:
            state.hover = None  # a screen or gizmo gesture has no voxel under it
        state.preview = g.preview()
        state.hover_view = context.view_state.screen_matrix.copy()
        if context.area is not None:
            context.area.tag_redraw()

    def _redesign(self, context) -> None:
        """Take the stroke back and draw it again with the curves over the whole path.

        The recorder's snapshots stay the voxels from before the press, so the
        undo step is the designed stroke alone.
        """
        gesture = getattr(self, "_gesture", None)
        if type(gesture) is not edit.Stroke or not any(gesture.runs) or self._mode_name != VOXEL:
            return
        from . import stroke_ops

        design = stroke_ops.design_of(context.prefs)
        if design is None or not design.applies():
            return
        from rok.core import stroke_design

        self._recorder.revert()
        write_brush = None
        if getattr(self, "_inside", None) is None and getattr(self, "_masked", None) is None:
            write_brush = self._recorder.write_brush
        stroke_design.replay(
            gesture.voxels, gesture.tool, [run for run in gesture.runs if run],
            int(context.prefs.rok_brush_size), design, int(document.colour),
            self._setters()[1], gesture.mirror, write_brush,
        )

    def _finish(self, context) -> None:
        self._changed = self._gesture.finish()
        self._redesign(context)
        if self._option_erase:
            state.option_erase_count, state.hud_tip_faded = hud.hud_after_option_erase(
                state.option_erase_count, state.hud_tip_faded
            )
        # The deferred gestures write their cells here, so upload them now.
        voxel_object.flush_dirty()
        if context.area is not None:
            context.area.tag_redraw()
        record = self._recorder.finish()
        inside = getattr(self, "_inside", None)
        masked = getattr(self, "_masked", None)
        locked = document.grid.locked_writes
        document.grid.locked_writes = 0  # the stroke's own; select_ops notes the commands'
        if record is None and ((masked is not None and masked.blocked) or locked):
            _note(MASK_NOTE)
        elif record is None and inside is not None and inside.blocked:
            _note(ERASE_INSIDE_NOTE if self._tool == edit.ERASE else INSIDE_NOTE)
        parts = []
        if self._palette_record is not None and record is None:
            self._revert_palette()  # colours fitted in for voxels that never came
        if record is not None:
            frame_index = 0
            if state.animating and document.model is not None and document.model.states:
                from . import anim_mode

                frame_index = anim_mode.shown_frame(document.model)
            parts.append(ModelRecord(document.active_uid, record, frame=frame_index))
            if self._tool in (edit.ATTACH, edit.PAINT):
                _late("palette_ui", "note_used")(document.colour)  # the star's recent colours
        voxel_or_palette = bool(parts)
        if document.selection is not None:
            document.selection.prune(document.grid.visible)
            self._redraw_selection(context)
        if self._sel_before is not None and document.selection is not None:
            sel_record = selection_record(document.active_uid, self._sel_before, document.selection.mask)
            if sel_record is not None:
                parts.append(sel_record)
        stacked = stacked_record(*parts)
        if stacked is not None:
            document.undo.push(stacked)
        if voxel_or_palette:
            document.modified = True
        self._palette_record = None
        document.stroke_serial += 1
        state.preview = None
        if isinstance(self._gesture, CageDrag):
            cage_ops.adopt()  # the cage's own bend is not a stale rest
        _late("gizmo_ops", "refresh")()
        self._end(context)

    def _commit_selection(self) -> None:
        """Push a selection-only record when the press changed the mask (click-deselect)."""
        if self._sel_before is None or document.selection is None:
            return
        record = selection_record(document.active_uid, self._sel_before, document.selection.mask)
        if record is not None:
            document.undo.push(record)

    def _restore_selection(self) -> None:
        """Put the mask from the press back; used when the drag is cancelled."""
        before = self._sel_before
        if before is not None and document.selection is not None:
            document.selection.replace(before)

    def _finish_model_drag(self, context) -> None:
        """Release on a MODEL gizmo drag: apply the result through rok.transform.
        In Animate mode the drag writes keys instead (anim_mode)."""
        gizmo_ops = _mod("gizmo_ops")
        scene_tools = _mod("scene_tools")
        res = self._gesture.result()
        state.preview = None
        if state.animating:
            from . import anim_mode

            anim_mode.gizmo_commit(res)
            gizmo_ops.end_model_drag(cancel=False)
            if context.area is not None:
                context.area.tag_redraw()
            _late("gizmo_ops", "refresh")()
            self._end(context)
            return
        token = self._copy_token
        self._copy_token = None
        if token is not None and not (res.kind == gizmo.MOVE and any(res.offset)):
            _late("scene_tools", "cancel_copy")(token)  # a Shift-click that moved nothing
            token = None
        group = list(scene_tools.group_models()) if scene_tools is not None else [document.model]
        if scene_tools is not None and (token is not None or (res.kind == gizmo.MOVE and len(group) > 1 and any(res.offset))):
            # An arrow moves the whole group; with Shift the copies that have
            # stood at their places since the press stay. One undo step.
            _late("gizmo_ops", "end_model_drag")(cancel=True)
            scene_tools.reset_group_empties()
            moves = []
            for model in group:
                before = tuple(int(p) for p in model.position)
                moves.append((model, before, tuple(b + int(o) for b, o in zip(before, res.offset))))
            if token is not None:
                scene_tools.finish_copy(token, moves)
            else:
                scene_tools._commit_moves(moves)
            if context.area is not None:
                context.area.tag_redraw()
            _late("gizmo_ops", "refresh")()
            self._end(context)
            return
        rotation = gizmo_ops.take_rotation() if gizmo_ops is not None else None
        if res.free and rotation is not None and rotation.applied == res.degrees:
            # The drag already turned the voxels: keep them, as one undo step.
            rotation.commit()
            gizmo_ops.end_model_drag(cancel=False)
            if context.area is not None:
                context.area.tag_redraw()
            _late("gizmo_ops", "refresh")()
            self._end(context)
            return
        if rotation is not None:
            # A quarter turn (or none): back to the press, then the exact turn, which
            # keeps linked copies linked.
            rotation.restore()
        if res.free and scene_tools is not None and gizmo_ops is not None:
            # Too big for the grid, or a model with animation states: no free angle.
            self.report({"WARNING"}, rotation.error if rotation is not None and rotation.error
                        else "A model with animation states cannot be turned by a free angle")
            gizmo_ops.end_model_drag(cancel=True)
            _late("gizmo_ops", "refresh")()
            self._end(context)
            return
        if res.kind in (gizmo.ROTATE, gizmo.FLIP) and len(group) > 1 and scene_tools is not None:
            if res.kind == gizmo.FLIP or int(res.turns) % 4:
                scene_tools.transform_group(res.kind, int(res.axis), int(res.turns))
        elif res.kind in (gizmo.MOVE, gizmo.ROTATE, gizmo.FLIP) and "rok.transform" in opkit.REGISTRY:
            opkit.ops.rok.transform(
                target=gizmo.MODEL,
                kind=res.kind,
                axis=int(res.axis),
                turns=int(res.turns),
                offset=tuple(int(v) for v in res.offset),
                copy=bool(res.copy),
            )
        if gizmo_ops is not None:
            gizmo_ops.end_model_drag(cancel=False)
        if context.area is not None:
            context.area.tag_redraw()
        _late("gizmo_ops", "refresh")()
        self._end(context)

    @staticmethod
    def _redraw_selection(context) -> None:
        """The selection changed: the overlay and the status line count."""
        if context.area is not None:
            context.area.tag_redraw()
        voxel_object.redraw_statusbar()

    def _end(self, context) -> None:
        state.stroke_active = False
        state.steady_line = None
        if state.profile:
            self._profile_summary()
        mode.rehover()

    def cancel(self, context):
        state.stroke_active = False
        state.steady_line = None
        state.preview = None
        state.gizmo_ghost = None
        state.model_drag_offset = (0, 0, 0)

    def _profile_move(self, started: float) -> None:
        ms = (time.perf_counter() - started) * 1000.0
        self._moves += 1
        self._max_ms = max(self._max_ms, ms)
        if ms > STROKE_BUDGET_MS:
            print(f"ROK stroke: extend+flush took {ms:.1f} ms")

    def _profile_summary(self) -> None:
        total_ms = (time.perf_counter() - self._started) * 1000.0
        g = self._gesture
        # Voxel mode counts its visited cells; the deferred gestures report finish().
        cells = len(getattr(g, "visited", ())) or self._changed
        print(
            f"ROK stroke: {self._mode_name} {g.tool} {cells} cells, "
            f"{self._moves} moves, max {self._max_ms:.1f} ms, {total_ms:.0f} ms total"
        )


# --- the settings -------------------------------------------------------------------------

RokPrefs.add_property("rok_tool", EnumProp(
    name="Tool",
    description="The tool a click or drag runs",
    items=edit.TOOL_ITEMS,
    default="ATTACH",
    update=_on_tool_change,
))
RokPrefs.add_property("rok_mode", EnumProp(
    name="Mode",
    description="The shape a click or drag makes",
    items=edit.MODE_ITEMS,
    default="VOXEL",
    update=_on_mode_change,
))
RokPrefs.add_property("rok_mirror", BoolVectorProp(
    name="Mirror",
    description="Repeat every change on the other side of the model",
    size=3,
    default=(False, False, False),
    update=_on_tool_change,
))
RokPrefs.add_property("rok_solid", EnumProp(
    name="Solid",
    description="The primitive an Attach in Box or Center mode builds once the height is set",
    items=shape.SOLID_ITEMS,
    default="BOX",
    update=lambda self, context: _remember_mode(context.prefs),
))
RokPrefs.add_property("rok_hollow", IntProp(
    name="Hollow",
    description="The wall of a hollow primitive in voxels, 0 for solid; scroll while setting the height",
    default=0, min=0, max=shape.WALL_MAX,
))
RokPrefs.add_property("rok_pattern_space", EnumProp(
    name="Pattern Stamp",
    description="What an Attach stamp in Pattern mode does with the pattern",
    items=(
        ("ADD", "Add", "Build the pattern's shape on the surface"),
        ("CARVE", "Carve", "Carve the pattern's shape out of the model"),
        ("HOLES", "Holes", "Sink into the surface, build the shape and clear the pattern's empty cells"),
    ),
    default="ADD",
))
RokPrefs.add_property("rok_select_visible", BoolProp(
    name="Visible Only",
    description="Select takes only the voxels you can see from this view, not the ones behind or inside",
    default=False,
))
RokPrefs.add_property("rok_colour_tolerance", IntProp(
    name="Tolerance",
    description="Select by colour: 0 takes the clicked colour only; higher also takes colours close to it",
    min=0,
    max=100,
    default=0,
    update=lambda self, context: mode.rehover(),
))
RokPrefs.add_property("rok_steady_hand", BoolProp(
    name="Steady Hand",
    description="Voxel strokes: the brush trails the mouse on a string",
    default=False,
))
RokPrefs.add_property("rok_steady_length", IntProp(
    name="String Length",
    description="Steady Hand: how far the mouse can move before it pulls the brush, in pixels",
    min=steady_hand.MIN_LENGTH,
    max=steady_hand.MAX_LENGTH,
    default=steady_hand.DEFAULT_LENGTH,
))
RokPrefs.add_property("rok_brush_size", IntProp(
    name="Size",
    description="The brush sphere in Voxel mode, for every tool: 1 is one voxel, 50 is 99 across",
    min=1,
    max=50,
    default=1,
    update=_on_brush_size_change,
))
if not hasattr(RokPrefs, "rok_show_palette_grid"):
    # palette_ui.py owns this setting; the strip's 256 button needs it
    # before that module is imported. Its own definition replaces this one.
    RokPrefs.add_property("rok_show_palette_grid", BoolProp(
        name="256", description="Show all 256 palette slots", default=False,
        update=lambda self, context: voxel_object.redraw_3d_views(),
    ))


def _start(context) -> None:
    """A fresh window manager: the switch rules start from its tool."""
    global _last_tool, switch_note
    _last_tool = context.prefs.rok_tool
    switch_note = None


registry.startup.insert(0, _start)
