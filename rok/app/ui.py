"""Rok's hint line and the File, Edit and View menus.

Rok has no status bar and no top bar: the hint is drawn at the bottom centre
of the view, and the menus are builders for the macOS menu bar
(`handlers.registry.menus`). An item's id is the action it runs; an item
whose action does not exist does nothing. The view is perspective until
CAD View, which is orthographic.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from rok.core import cage
from rok.core import chrome as chrome_kit
from rok.core.edit import ACTIVATE, ATTACH, CLONE, SCREEN_MODES, SELECT, STAMP_MODES, VOXEL

from . import command, marking, menus, nav_cube, palette_ui, render_view, theme, welcome
from . import object as voxel_object
from .gfx import fonts
from .handlers import registry
from .object import document
from .opkit import CONTEXT, EnumProp, Action, register
from .paint import round_rect
from .state import state

GIZMO_HINT = (
    "Gizmo: drag a cone or square to move, a ring to turn, click a cube to flip"
    "   Shift+drag the centre: move the pivot"
)
CAGE_HINT = cage.CAGE_IDLE_HINT
ACTIVATE_HINT = "Double-click: edit this model"
SCENE_HINT = ("Drag: move (snaps; ⌘ free) · ⇧ drag: linked copy · Ctrl drag onto a model: carve (⇧ keeps the cutter) · ⇧ click: add to the pick · ⌘J: merge · ⌥ drag: up and down · "
              "Arrows: nudge · ⌘↑↓: raise, lower · "
              "G: drop to the floor · ⇧G: drop onto a voxel · Gizmo: arrows move (⇧ copy), cubes stretch, the centre sphere scales all (⌥ crop) · "
              "R: turn · C: cut · F: frame · "
              "Double-click or Enter: step in")
CUT_HINT = "Click: cut the model in two along the plane · Tab: turn the plane · Esc: back to Pick"
SIZE_TYPE_HINT = "Type the grid size: 40 40 40, or 40 for a cube   Enter: resize   Esc: keep it"

HINT_FONT = 11.0
HINT_MIN_FONT = 9.0
HINT_TEXT = (0.905, 0.885, 0.855, 0.72)
HINT_PLATE = (0.060, 0.056, 0.053, 0.55)
HINT_BOTTOM = 6  # from the bottom of the view to the hint's plate
ANIMATE_HINT = "Drag a ring or an arrow: the release keys it · K keys the pose · Shift+Space plays · , and . step a frame"
HINT_ROOM = 440  # kept free on each side for the colour strip and the corner gizmo


def _module(name: str):
    """A sibling module that may not be ported yet, or None."""
    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ImportError:
        return None


def _hint(context) -> str:
    """What the hint line says now: the running modal's hint first, then the
    tool's, the mode's and what is under the mouse."""
    if state.size_edit is not None:
        return SIZE_TYPE_HINT
    from . import cad_mode

    if cad_mode.editing():
        return cad_mode.TYPE_HINT
    modal = palette_ui.status_hint()
    if modal is not None:
        return modal
    tab = marking.status_hint()
    if tab is not None:
        return tab
    search = command.status_hint()
    if search is not None:
        return search
    from . import overlay

    preview = overlay.gradient_preview()
    if preview is not None:
        return f"Blend {len(preview)} colours · release to apply (one undo step)"
    tip = overlay.dock_hover_tip()
    if tip is not None:  # the swatch the mouse rests on: its hex, material and gestures
        return tip
    if state.hover_strip is not None and state.hover_strip[0] == "GRID_SIZE":
        return "Click: type a new grid size"
    if state.hover_strip is not None and state.hover_strip[0] == "LABEL_COLOUR":
        return "Click: edit this colour"
    if state.animating:
        return ANIMATE_HINT
    if state.level == "SCENE":
        return CUT_HINT if state.scene_tool == "CUT" else SCENE_HINT
    if getattr(context.app, "cad", False):
        return cad_mode.status_text()
    if state.level == "BRICK":
        from . import brick_mode

        return brick_mode.status()
    if document.grid is not None and document.grid.bricks is not None:
        from rok.core.brick import LOCKED_MESSAGE

        return LOCKED_MESSAGE
    wm = context.prefs
    tool = getattr(wm, "rok_tool", ATTACH)
    mode_name = getattr(wm, "rok_mode", VOXEL)
    parts = [chrome_kit.IDLE_STATUS.get(tool, chrome_kit.IDLE_STATUS[ATTACH])]
    parts.append(chrome_kit.MODE_STATUS.get(mode_name, chrome_kit.MODE_STATUS[VOXEL]))
    selection = document.selection
    has_sel = selection is not None and not selection.is_empty()
    corner_tips = {
        "CG_MOVE_X": "Drag: move along X   Shift: copy", "CG_MOVE_Y": "Drag: move along Y   Shift: copy",
        "CG_MOVE_Z": "Drag: move along Z   Shift: copy", "CG_SCALE_X": "Drag: stretch or squash along X",
        "CG_SCALE_Y": "Drag: stretch or squash along Y", "CG_SCALE_Z": "Drag: stretch or squash along Z",
        "CG_SCALE_ALL": "Drag up and right: grow the selection   down and left: shrink it",
        "CG_ROT_X": "Drag round: turn about X in quarter turns   Click: one quarter turn",
        "CG_ROT_Y": "Drag round: turn about Y in quarter turns   Click: one quarter turn",
        "CG_ROT_Z": "Drag round: turn about Z in quarter turns   Click: one quarter turn",
    }
    if state.hover_handle in corner_tips:
        parts.append(corner_tips[state.hover_handle])
    elif has_sel and tool == SELECT:
        parts.append(cage.pill_hint(state.hover_pill) if state.hover_pill else CAGE_HINT)
    elif state.gizmo is not None:
        parts.append(GIZMO_HINT)
    if has_sel:
        parts.append(f"{selection.count()} selected")
    grid = document.grid
    if grid is not None and grid.sections:
        planes = ", ".join(
            f"{'XYZ'[axis]} at {coord} from {'+' if from_positive else '-'}{'XYZ'[axis]}"
            for axis, (coord, from_positive) in sorted(grid.sections.items())
        )
        parts.append(f"Section {planes}   Cmd+Up/Down: move {'XYZ'[grid.section_axis]}")
    hover = state.hover
    if hover is not None and hover.tool == ACTIVATE:
        parts.append(ACTIVATE_HINT)
    tools = _module("tools")
    note = tools.fresh_note() if tools is not None and hasattr(tools, "fresh_note") else None
    if note is not None:
        parts.append(note)
    elif tool == ATTACH and mode_name in SCREEN_MODES and tools is not None and hasattr(tools, "SCREEN_MODE_HINT"):
        parts.append(tools.SCREEN_MODE_HINT)
    if mode_name == "GRADIENT":
        gradient_ops = _module("gradient_ops")
        if gradient_ops is not None:
            parts.append(f"{gradient_ops.HINT} · {gradient_ops.steps()} colours")
    if mode_name in STAMP_MODES:
        pattern_ops = _module("pattern_ops")
        if pattern_ops is not None:
            has_clip = state.clipboard is not None
            hint = "CLONE_HINT" if mode_name == CLONE else "PATTERN_HINT"
            parts.append(getattr(pattern_ops, hint if has_clip else "NO_PATTERN_HINT", ""))
    return "   ".join(p for p in parts if p)


def overlay_mod():
    from . import overlay

    return overlay


def _draw_hint(context) -> None:
    """The hint line at the bottom centre of the view. Long hints shrink a little, then end in an ellipsis."""
    _watch_recent(context)
    if document.grid is None or welcome.active() or command.active():
        return
    text = _hint(context)
    if not text:
        return
    from . import marking

    if marking.ui_pose("HINT")[2]:
        return  # an arm of the star's play has pulled the text away
    region = context.region
    room = max(120.0, region.width - 2 * HINT_ROOM)
    left = None
    card = getattr(overlay_mod(), "_dock", None)
    if card is not None:
        # Beside the palette card: all the room right of it, not a fixed band on both sides.
        left = card.card[0] + card.card[2] + 16.0
        room = max(120.0, region.width - left - 16.0)
    size = HINT_FONT
    fonts.size(0, size)
    tw, _th = fonts.dimensions(0, text)
    while tw > room and size > HINT_MIN_FONT:
        size -= 0.5
        fonts.size(0, size)
        tw, _th = fonts.dimensions(0, text)
    if tw > room:
        from .paint import clip_text

        text = clip_text(text, room, size)
        tw, _th = fonts.dimensions(0, text)
    _w, cap = fonts.dimensions(0, "H")
    h = cap + 12
    x = (region.width - tw) / 2.0
    if left is not None:
        x = max(left, min(x, region.width - 16.0 - tw))
    bottom = HINT_BOTTOM
    from . import dock

    layout = dock.layout_for(region)
    if layout is not None:  # Animate mode: above the dock
        bottom = layout.rect[1] + layout.rect[3] + 8
    from . import history_strip

    strip_top = history_strip.top(region)
    if strip_top is not None:  # above the history strip
        bottom = strip_top + 8
    from . import overlay

    preview = overlay.gradient_preview()
    chips = 0.0
    if preview is not None:  # the Shift drag's in-between colours, in the HUD line
        chip = cap + 4
        chips = len(preview) * (chip + 2) + 6
        x = max(x - chips / 2.0, (left if left is not None else 0) + chips)
    round_rect((x - 10 - chips, bottom, tw + 20 + chips, h), h / 2.0, HINT_PLATE)
    if preview is not None:
        cx = x - chips
        for rgb in preview:
            round_rect((cx, bottom + (h - chip) / 2.0, chip, chip), 2.0, tuple(c / 255.0 for c in rgb) + (1.0,))
            cx += chip + 2
    fonts.color(0, *HINT_TEXT)
    fonts.position(0, float(round(x)), float(round(bottom + (h - cap) / 2.0)), 0)
    fonts.draw(0, text)


# --- operators ---------------------------------------------------------------------


@register
class ViewAction(Action):
    command_id = "rok.view"
    title = "View"
    tooltip = "Frame the scene or the active model, or turn the view to a side"
    action = EnumProp(
        name="Action",
        description="Which view change to run",
        items=(
            ("FRAME", "Frame All", "Zoom out until every model fits the view"),
            ("FRAME_MODEL", "Frame Model", "Zoom in until the active model fills the view"),
            ("FRONT", "Front", "Look at the front of the scene"),
            ("RIGHT", "Right", "Look at the right side of the scene"),
            ("TOP", "Top", "Look down on the scene from above"),
        ),
        default="FRAME",
    )

    # The view directions: the way the camera looks. Front looks
    # along +Y, Right along -X, Top straight down (a hair off, so the camera's
    # up stays defined).
    LOOK = {"FRONT": (0.0, 1.0, 0.0), "RIGHT": (-1.0, 0.0, 0.0), "TOP": (0.0, 0.0001, -1.0)}

    @classmethod
    def poll(cls, context):
        return document.grid is not None

    def execute(self, context):
        if self.action == "FRAME":
            voxel_object.frame_scene(context)
        elif self.action == "FRAME_MODEL":
            model = document.model
            if model is not None:
                voxel_object.frame_model(context, model)
        else:
            context.view_state.set_view_direction(np.asarray(self.LOOK[self.action]))
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


# --- menus --------------------------------------------------------------------------

MAX_RECENT_ITEMS = 5


def _recent_items() -> list:
    items = []
    for path in welcome.load_recent()[:MAX_RECENT_ITEMS]:
        p = Path(path)
        label = f"{p.name} — {p.parent.name}" if p.parent.name else p.name
        items.append((f"rok.open_recent(filepath={str(path)!r})", label, "", -1))
    return items


def file_menu(app) -> list:
    """io_ops's File menu, with the recent files under Open, Export Mesh and the
    Welcome Screen."""
    from . import io_ops

    title, items = io_ops.file_menu(app)[0]
    items = list(items)
    at = next((i + 1 for i, it in enumerate(items) if it is not None and it[0] == "rok.open"), len(items))
    recent = _recent_items()
    if recent:
        items[at:at] = [None] + recent
    items += [None, ("rok.welcome", "Welcome Screen", "", -1)]
    return [(title, items)]


def edit_menu(app) -> list:
    """Undo, Redo, the command search and the clipboard. The selection and the
    whole-model commands have their own menus (select_ops, model_ops)."""
    return [("Edit", [
        ("rok.undo", "Undo", "CmdOrCtrl+Z", -1),
        ("rok.redo", "Redo", "CmdOrCtrl+Shift+Z", -1),
        None,
        ("rok.command_search", "Search Commands…", "CmdOrCtrl+K", -1),
        None,
        ("rok.cut", "Cut", "", -1),
        ("rok.copy", "Copy", "", -1),
        ("rok.paste", "Paste", "", -1),
        ("rok.selection_delete", "Delete Selection", "", -1),
        None,
        ("rok.brick_mode", "Brick Mode", "", int(state.level == "BRICK" and not state.animating)),
        None,
        ("rok.palette_cleanup", "Clean Up Palette…", "", -1),
    ])]


def view_menu(app) -> list:
    """Few items. Framing, the sides, Section and Mirror, Quad
    and Rendered View are on the nav cube; looks are in View Settings, the light in Render Settings."""
    saved = ("Saved Views", [(f"rok.view_save(slot={s})", f"Save View {s}", "", -1) for s in (1, 2, 3, 4)]
             + [None] + [(f"rok.view_go(slot={s})", f"Go to View {s}", "", -1) for s in (1, 2, 3, 4)])
    click = marking.pick_mode() == marking.CLICK
    star = (f"rok.star_pick(pick={(marking.RELEASE if click else marking.CLICK)!r})", "Star: Pick on Click", "", int(click))
    return [("View", [
        ("rok.model_menu", "Customize Rail Menu and Keys…", "", -1),
        ("rok.animation_menu", "Show Animation Menu", "", int(getattr(app, "animation_menu", False))),
        ("rok.history_strip", "Show History", "CmdOrCtrl+L", int(getattr(app, "history_strip", False))),
        None,
        saved,
        None,
        ("rok.grid", "Voxel Grid", "CmdOrCtrl+G", int(getattr(app, "grid", False))),
        ("rok.cad_view", "CAD View", "", int(getattr(app, "cad", False))),
        star,
        ("Theme", theme.menu_items()),
        ("rok.cube_axis_letters", "Show Axis Letters on Nav Cube", "", int(nav_cube.letters_on())),
        None,
        ("rok.view_settings", "View Settings…", "", -1),
        ("rok.render_settings", "Render Settings…", "", -1),
    ])]


def install_menus() -> None:
    """Put this module's File, Edit and View menus where menus.install() put
    the foundation's."""
    from . import io_ops

    swap = {io_ops.file_menu: file_menu, menus.edit_menu: edit_menu, menus.view_menu: view_menu}
    registry.menus[:] = [swap.get(build, build) for build in registry.menus]


_recent_stamp: object = None  # the recent list's file time when the menu was last built


def _refresh_menu(context) -> None:
    """After New or Open: the check marks changed."""
    app = context.app
    if app is not None and app._started:
        app.set_menu()


def _watch_recent(context) -> None:
    """Build the menu again when the recent-files list changed. Open and Save
    write the list after their load handlers ran, so a redraw checks the
    list's file time (one stat call)."""
    global _recent_stamp
    from . import recent

    try:
        stamp = recent.FILE.stat().st_mtime_ns
    except OSError:
        stamp = None
    if stamp != _recent_stamp:
        _recent_stamp = stamp
        _refresh_menu(context)


registry.add_draw("hint", 450, "PIXEL", _draw_hint)
registry.load_post.append(_refresh_menu)
