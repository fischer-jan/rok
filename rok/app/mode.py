"""Voxel mode: the always-on hover operator, hover computation and the lifecycle.

`rok.voxel_mode` is a modal operator
that the app starts once the window is up (`handlers.registry.startup`). It
watches events, returns {'PASS'} for every one of them and consumes
nothing. Its only job is the hover highlight.

A view change (orbit, zoom, a view key) is found where it is visible: the
overlay compares the view's perspective matrix with the one the hover was
picked in and asks for a repick when they differ. No event guessing is needed
for that.

Rok has one view and no workspaces, so there is no region search, no Voxel
tool activation and no load handler. Animate mode is
parked: no timeline, no posed parts, no anchor dot.
"""

from __future__ import annotations

import os
import time
import traceback

import numpy as np

from rok.core import cage, edit, gizmo, hud, region
from rok.core.chrome import LEVEL_SCENE
from rok.core.edit import ACTIVATE, ATTACH, FACE, SELECT, VOXEL
from rok.core.gizmo import skip_for
from rok.core.raycast import raycast

from . import object as voxel_object
from . import opkit, overlay, pattern_ops
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, register
from .state import Hover, cage_screen_for, gizmo_screen_for, hover_key, state
from .view3d import ray_origin, ray_direction

HOVER_BUDGET_MS = 8.0
PROFILE_WINDOW = 200
ERROR_PRINT_INTERVAL_S = 60.0
GIZMO_HIT_PX = 10.0  # hit radius of the gizmo anchors, times the UI scale

_MODIFIER_KEYS = (
    "LEFT_CTRL",
    "RIGHT_CTRL",
    "LEFT_SHIFT",
    "RIGHT_SHIFT",
    "LEFT_ALT",
    "RIGHT_ALT",
)

# Time of the last traceback printed by modal(); one print per minute at most.
_last_error_time = 0.0

# Profiling: rolling max over PROFILE_WINDOW hover events.
_profile_count = 0
_profile_max_ms = 0.0

# Face mode: the last region flood, keyed by (voxel, normal, wall, attach, model uid,
# edit_serial). One entry is enough: the mouse stays on one face for many events.
_region_cache: tuple[tuple, object] | None = None


def _late(name: str):
    """A sibling module that may not be ported yet, or None."""
    import importlib

    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ImportError:
        return None


# --- the always-on operator -------------------------------------------------------------


@register
class VoxelModeAction(Action):
    command_id = "rok.voxel_mode"
    title = "Voxel Mode"
    tooltip = "Follow the mouse over the voxels"
    @classmethod
    def poll(cls, context):
        return context.window is not None

    def invoke(self, context, event):
        if state.running:
            return {"CANCELLED"}
        self._generation = state.generation
        state.running = True
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        if self._generation != state.generation:
            # A restart made this instance stale. PASS keeps the event
            # alive for the rest of the app; CANCELLED alone would eat it.
            return {"CANCELLED", "PASS"}
        try:
            _on_event(context, event)
        except Exception:
            _print_error_throttled()
            state.hover = None
        return {"PASS"}


def _print_error_throttled() -> None:
    global _last_error_time
    now = time.monotonic()
    if now - _last_error_time >= ERROR_PRINT_INTERVAL_S:
        _last_error_time = now
        traceback.print_exc()


def _on_event(context, event) -> None:
    """Update the hover from one event. Every other event type does nothing."""
    # Every event carries the pointer. Keep it current so a repick from a timer or a
    # modifier key picks at the real position, not at an old MOUSE_MOVE.
    mx, my = event.mouse_x, event.mouse_y
    state.last_mouse = (mx, my)
    kind = event.type
    if kind == "MOUSE_MOVE":
        if state.stroke_active:
            return
        from . import cad_mode

        cad_mode.pointer(mx, my)  # the edge under the cursor, while the CAD view is on
        state.ctrl = event.ctrl
        state.shift = event.shift
        state.alt = event.alt
        update_hover(context.window, mx, my)
    elif kind in _MODIFIER_KEYS:
        pressed = event.value == "PRESS"
        prev_alt = state.alt
        if kind in ("LEFT_CTRL", "RIGHT_CTRL"):
            state.ctrl = pressed
        elif kind in ("LEFT_ALT", "RIGHT_ALT"):
            state.alt = pressed
        else:
            state.shift = pressed
        update_hover(context.window, *state.last_mouse)
        # The HUD flips on Option even when the hover key does not (empty space,
        # or already on Erase). The hover redraw is skipped in those cases.
        if state.alt != prev_alt:
            voxel_object.redraw_3d_views()
    elif kind == "FOCUS_LOST":
        set_hover(None, None, None)


def _clear_hover_state() -> None:
    state.hover = None
    state.hover_strip = None
    state.hover_handle = None
    state.hover_pill = None
    state.pill_since = None
    state.hover_model = None
    state.hover_view = None


# --- hover computation ------------------------------------------------------------------


def find_view3d_window_region(window, mx: int, my: int):
    """(area, region, view_state) for the view under region pixel (mx, my), or None.
    Rok has one view that fills the window."""
    reg = CONTEXT.region
    if reg is None or window is None:
        return None
    if not (0 <= mx < reg.width and 0 <= my < reg.height):
        return None
    return CONTEXT.area, reg, CONTEXT.view_state


def world_ray(region, view_state, mx: float, my: float):
    """Ray in world space from region pixel (mx, my): (origin, direction) as 3-tuples."""
    coord = (mx, my)
    o = ray_origin(region, view_state, coord)
    d = ray_direction(region, view_state, coord)
    return (float(o[0]), float(o[1]), float(o[2])), (float(d[0]), float(d[1]), float(d[2]))


def mouse_ray(region, view_state, mx: float, my: float, model=None):
    """Ray in the grid space of `model` (default: the active model)."""
    origin, direction = world_ray(region, view_state, mx, my)
    return _model_ray(origin, direction, model or document.model)


def gizmo_ray(region, view_state, mx: float, my: float, model=None):
    """Ray in the grid space of `model`, at rest: a translation only.

    The gizmo is drawn at the model's rest place (`gizmo_state.origin` is
    `model_origin`). With animation, `mouse_ray` would go through a pose and
    the gizmo would take this one; without animation both are the same.
    """
    origin, direction = world_ray(region, view_state, mx, my)
    return _to_model_space(origin, model or document.model), direction


def _model_ray(origin, direction, model):
    """A world ray `(origin, direction)` in the grid space of `model`.

    None: world space, ray unchanged. Every model space is a translation of
    world space, so only the origin moves.
    """
    if model is None:
        return origin, direction
    return _to_model_space(origin, model), direction


def _to_model_space(point, model):
    """A world point in the grid space of `model`. None: world space."""
    if model is None:
        return (float(point[0]), float(point[1]), float(point[2]))
    ox, oy, oz = voxel_object.model_origin(model)
    return (point[0] - ox, point[1] - oy, point[2] - oz)


def view_dir(view_state) -> tuple[float, float, float]:
    """The direction the view looks along, in world space."""
    d = view_state.view_direction()
    return (float(d[0]), float(d[1]), float(d[2]))


def _same_view(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def update_hover(window, mx: int, my: int) -> None:
    """Recompute the hover for region pixel (mx, my) and redraw on change.

    In this order: the rail, the file label, the corner gizmo, the palette
    strip and the 256 grid, the cage pill strip, the gizmo handles, then the
    voxels of every model. The first hit wins.
    """
    started = time.perf_counter() if state.profile else 0.0
    found = find_view3d_window_region(window, mx, my)
    model = document.model
    scene_level = state.level == LEVEL_SCENE
    if found is None or model is None or (document.scene.is_hidden(model) and not scene_level):
        set_hover(None, None, None)
        return
    area, reg, view_state = found
    rx, ry = mx - reg.x, my - reg.y

    if _chrome_hover(reg, rx, ry, area):
        set_hover(None, None, area)
        return
    if _stroke_panel_hover(reg, rx, ry, area):
        set_hover(None, None, area)
        return

    part = label_part_at(reg, rx, ry)
    if part is not None:
        set_hover(None, (part, 0), area)
        return

    # The lattice cage works beside every tool: its chips and points come first.
    chip = overlay.cage_chip_hit(reg, rx, ry)
    if chip is not None or overlay.cage_chip_contains(reg, rx, ry):
        set_hover(None, None, area, pill=chip)
        return
    cage_point = overlay.cage_hit(reg, rx, ry)
    if cage_point is not None:
        set_hover(None, None, area, handle=cage_point)
        return

    corner = overlay.corner_hit(reg, rx, ry)
    if corner is not None:
        set_hover(None, None, area, handle=corner)
        return

    item = _strip_hover(reg, rx, ry)
    if item is not False:
        set_hover(None, item, area)
        return

    view = view_state.screen_matrix.copy()
    wm = CONTEXT.prefs
    tool = edit.effective_tool(wm.rok_tool, state.alt, state.shift)
    subtract = edit.select_subtract(tool, state.alt, state.shift)
    cage_screen = cage_screen_for(reg)
    if cage_screen is not None:
        strip = cage_screen.strip
        if strip is not None and cage.strip_contains(strip, rx, ry):
            set_hover(None, None, area, view, pill=cage.strip_hit(strip, rx, ry))
            return
    screen = gizmo_screen_for(reg)
    if state.gizmo is not None and screen is not None:
        radius = GIZMO_HIT_PX * overlay.ui_scale()
        skip = skip_for(state.level, state.animating)
        handle = gizmo.hit_test(screen, rx, ry, radius, view_dir(view_state), skip=skip)
        if handle is not None:
            set_hover(None, None, area, view, handle=handle)
            return

    key_cursor = _late("key_cursor")
    if key_cursor is not None and key_cursor.took_over(mx, my):
        return  # the key cursor leads until the mouse really moves
    if state.level == "BRICK" and not state.animating:
        # Brick mode: the brick in the hand is a ghost where it would land.
        brick_mode = _late("brick_mode")
        if brick_mode is not None:
            brick_mode.hover(reg, view_state, rx, ry)
        set_hover(None, None, area, view)
        return
    if not scene_level and document.grid is not None and document.grid.bricks is not None:
        set_hover(None, None, area, view)  # a brick model takes no voxel tool in Model level
        return

    if scene_level:
        # The scene: the model under the mouse, whole. A click picks it.
        scene_tools = _late("scene_tools")
        if state.scene_tool == "CUT" and scene_tools is not None:
            aim = scene_tools.aim_cut(reg, view_state, rx, ry)
            if aim != state.cut_aim:
                state.cut_aim = aim
                area.tag_redraw()
            set_hover(None, None, area, view, model_uid=aim[0] if aim else None)
            return
        set_hover(None, None, area, view, model_uid=model_at(reg, view_state, rx, ry))
        return

    grid = model.grid
    hit, other, other_hit = pick(reg, view_state, rx, ry, tool == ATTACH)
    if other_hit is not None:
        # Another model wins the ray: a click there makes it the active one.
        hover = Hover(other_hit, ACTIVATE, other_hit.voxel)
        set_hover(hover, None, area, view, model_uid=other.uid)
        return
    if hit is None:
        set_hover(None, None, area, view)
        return
    mode_name = getattr(wm, "rok_mode", VOXEL)
    target = edit.target_cell(tool, hit, grid.size)
    if tool == SELECT and hit.wall and mode_name in (VOXEL, FACE):
        target = None  # a click there deselects; no cube to show
    cells = None
    serial = 0
    if mode_name == FACE:
        cells = _face_hover_region(grid, hit, tool)
        serial = document.edit_serial
        if len(cells) == 0:
            target = None
    elif mode_name in edit.SELECT_ONLY_MODES and tool == SELECT and not hit.wall:
        cells = _connected_hover_region(grid, hit, mode_name)
        serial = document.edit_serial
    elif mode_name in edit.STAMP_MODES:
        # The ghost of the stamp. No pattern, or a wall with Erase: no click.
        cells = pattern_ops.hover_cells(grid, hit, tool)
        tool = pattern_ops.stamp_tool(tool)[0]  # Carve: the ghost in Erase's colour
        serial = state.pattern_serial
        if cells is None:
            target = None
    hover = Hover(hit, tool, target, mode_name, cells, serial, subtract)
    set_hover(hover, None, area, view)
    if state.profile:
        _profile_hover(started)


def _stroke_panel_hover(reg, rx: float, ry: float, area) -> bool:
    """The Stroke Designer, when it is open. True when the mouse is on it, so
    the voxels under the panel are not picked."""
    from . import stroke_panel

    if not stroke_panel.active():
        return False
    changed = stroke_panel.hover(reg, rx, ry)
    if changed and area is not None:
        area.tag_redraw()
    return stroke_panel.contains(reg, rx, ry)


def _chrome_hover(reg, rx: float, ry: float, area) -> bool:
    """Track the left rail (rok/app/chrome.py). True when the mouse is inside
    it, so no voxel is hovered behind it. False while the rail is not imported."""
    chrome = _late("chrome")
    if chrome is None or not hasattr(chrome, "hover"):
        return False
    dock = _late("dock")
    strip = _late("history_strip")
    in_dock = dock is not None and dock.hover(reg, rx, ry)
    in_strip = strip is not None and strip.hover(reg, rx, ry)  # both track, so each clears its hover
    if in_dock or in_strip:
        if chrome.set_hover(None, False) and area is not None:
            area.tag_redraw()
        return True  # nothing in the view is hovered behind the dock
    found = chrome.hover(reg, rx, ry)
    inside = found is not False
    marking = _late("marking")
    if marking is not None and getattr(marking, "active", lambda: False)():
        return inside  # the star is open over the rail: the rail opens nothing under it
    if chrome.set_hover(found if inside else None, inside) and area is not None:
        area.tag_redraw()
    return inside


def label_part_at(reg, rx: float, ry: float) -> str | None:
    """The clickable part of the file label under the pixel: "GRID_SIZE" (a
    click types a new size) or "LABEL_COLOUR" (a click opens the colour picker), or None."""
    found = state.label_parts
    if found is None or found[0] != reg.as_pointer():
        return None
    for part, (x, y, w, h) in found[1].items():
        if x <= rx < x + w and y <= ry < y + h:
            return part
    return None


def _strip_hover(reg, rx: float, ry: float):
    """The strip or grid item under the pixel, None inside their rectangles with no
    item, or False outside both."""
    if not overlay.is_main_region(reg):
        return False  # the strip draws in the main view only
    strip = overlay.strip_layout(reg)
    item = overlay.dock_hit(reg, rx, ry) or hud.strip_hit_test(strip, rx, ry)
    if item is not None or hud.strip_contains(strip, rx, ry):
        return item
    if overlay.palette_grid_shown():
        layout = overlay.palette_layout(reg)
        swatch = hud.hit_test(layout, rx, ry)
        if swatch is not None:
            return ("SWATCH", swatch)
        if hud.contains(layout, rx, ry):
            return None
    return False


def pick(reg, view_state, rx: float, ry: float, walls_count: bool = False):
    """What the ray through region pixel (rx, ry) hits: (hit, other, other_hit).

    `hit` is the active model's raycast (walls included). `other` and `other_hit` name
    a model whose voxel wins over it; both None when no other model is in the way.

    A wall hit is the grid box boundary, not geometry. Attach places its first voxel
    on a wall, so with `walls_count` a wall keeps its place and the other model has to
    be nearer. For every other action a wall has no target, so any voxel wins over it.
    The hover and the press pass the same flag, so a press acts on what the hover showed.
    """
    model = document.model
    origin, direction = world_ray(reg, view_state, rx, ry)
    model_origin, model_direction = _model_ray(origin, direction, model)
    hit = raycast(model.grid.tool_voxels, model_origin, model_direction, walls=True)
    other, other_hit = _nearest_other_model(origin, direction, model)
    dead_wall = hit is not None and hit.wall and not walls_count
    if other_hit is not None and (hit is None or dead_wall or other_hit.t < hit.t):
        return hit, other, other_hit
    return hit, None, None


def model_at(reg, view_state, rx: float, ry: float) -> int | None:
    """The uid of the model a click at region pixel (rx, ry) picks: the one
    first in `models_under`, or None. The scene level's pick: every model
    counts the same, and an empty one by its box."""
    under = models_under(reg, view_state, rx, ry)
    return under[0] if under else None


def models_under(reg, view_state, rx: float, ry: float) -> list[int]:
    """Every shown model under region pixel (rx, ry), in the scene's click and
    click-cycling order (`scene_move.pick_order`): the ones with a voxel
    there, nearest first, then the ones whose box alone is there, so an empty
    model can be picked too."""
    return models_hit(reg, view_state, rx, ry)[0]


def models_hit(reg, view_state, rx: float, ry: float) -> tuple[list[int], set[int]]:
    """(`models_under`'s list, the uids among them with a voxel there)."""
    from rok.core import scene_move
    from rok.core.raycast import box_enter

    scene = document.scene
    if scene is None:
        return [], set()
    origin, direction = world_ray(reg, view_state, rx, ry)
    voxel_hits, box_hits = [], []
    for m in scene.models:
        if scene.is_hidden(m):
            continue
        model_origin, model_direction = _model_ray(origin, direction, m)
        h = raycast(m.grid.tool_voxels, model_origin, model_direction, walls=False)
        if h is not None and not h.wall:
            voxel_hits.append((h.t, m.uid))
            continue
        t = box_enter(model_origin, model_direction, m.grid.size)
        if t is not None:
            box_hits.append((t, m.uid))
    return scene_move.pick_order(voxel_hits, box_hits), {uid for _t, uid in voxel_hits}


def _nearest_other_model(origin, direction, active):
    """(model, hit) of the nearest voxel hit in any model but `active`, or (None, None).

    `origin` and `direction` are world space. `hit.t` stays comparable between
    models: a translation does not change it.
    """
    scene = document.scene
    best = None
    best_hit = None
    if scene is None:
        return best, best_hit
    for m in scene.models:
        if m.uid == active.uid or scene.is_hidden(m):
            continue
        model_origin, model_direction = _model_ray(origin, direction, m)
        h = raycast(m.grid.tool_voxels, model_origin, model_direction, walls=False)
        if h is None or h.wall:
            continue
        if best_hit is None or h.t < best_hit.t:
            best, best_hit = m, h
    return best, best_hit


_connected_cache: tuple | None = None  # ((uid, edit_serial, mode), mask, cells)


def _connected_hover_region(grid, hit, mode_name):
    """(N, 3) cells a click would take in Select's own modes: joined to the
    voxel under the mouse (Connected, by faces or corners), or of its colour
    (By colour). Worked out once per region: a voxel inside the last one
    reuses it."""
    global _connected_cache
    tolerance = int(getattr(CONTEXT.prefs, "rok_colour_tolerance", 0)) if mode_name in edit.COLOUR_MODES else 0
    key = (document.active_uid, document.edit_serial, mode_name, tolerance)
    voxel = tuple(int(v) for v in hit.voxel)
    if _connected_cache is not None and _connected_cache[0] == key and _connected_cache[1][voxel]:
        return _connected_cache[2]
    voxels = grid.tool_voxels
    if mode_name in edit.COLOUR_MODES:
        from rok.core.select import ColourGesture

        gesture = ColourGesture(voxels, None, joined=mode_name != edit.BY_COLOUR,
                                corners=mode_name == edit.COLOUR_CONNECTED_CORNERS,
                                palette=document.scene.palette, tolerance=tolerance)
        mask = gesture.region(voxel)
    else:
        corners = mode_name == edit.CONNECTED_CORNERS
        mask = region.connected_blobs(voxels, np.asarray([voxel], dtype=np.int64), corners)
    cells = np.argwhere(mask)
    _connected_cache = (key, mask, cells)
    return cells


def _face_hover_region(grid, hit, tool):
    """(N, 3) cells of the face region under the mouse. Floods once per face."""
    global _region_cache
    # The uid is in the key: activating another model does not bump edit_serial.
    key = (
        hit.voxel,
        hit.normal,
        hit.wall,
        tool == ATTACH,
        document.active_uid,
        document.edit_serial,
    )
    if _region_cache is not None and _region_cache[0] == key:
        return _region_cache[1]
    if hit.wall:
        if tool == ATTACH:
            cells = region.wall_region(grid.tool_voxels, hit.target, hit.normal)
        else:
            cells = np.zeros((0, 3), dtype=np.int64)
    else:
        cells = region.face_region(grid.tool_voxels, hit.voxel, hit.normal)
    _region_cache = (key, cells)
    return cells


def set_hover(
    hover: Hover | None,
    strip: tuple[str, int] | None,
    area,
    view=None,
    *,
    handle: str | None = None,
    model_uid: int | None = None,
    pill: str | None = None,
) -> None:
    """Store the hover and the view it was picked in. Tag a redraw only on a change.

    `strip` is the palette strip or grid item, `handle` the gizmo handle,
    `pill` a selection-strip button, and `model_uid` the hovered inactive
    model. Two hovers on the same face with a different ray parameter count
    as equal, so moving the mouse across one face costs no redraw.
    """
    if (
        hover_key(hover) == hover_key(state.hover)
        and strip == state.hover_strip
        and handle == state.hover_handle
        and pill == state.hover_pill
        and model_uid == state.hover_model
        and _same_view(view, state.hover_view)
    ):
        return
    state.hover = hover
    state.hover_strip = strip
    state.hover_handle = handle
    state.hover_model = model_uid
    state.hover_view = view
    if pill != state.hover_pill:
        state.pill_since = time.monotonic() if pill is not None else None
        if pill is not None:
            _schedule_pill_tip()
    state.hover_pill = pill
    voxel_object.redraw_3d_views()


def _schedule_pill_tip() -> None:
    """Ask for the redraw that shows the pill HUD tip; a mouse that rests sends
    no event. A host tick redraws."""
    app = CONTEXT.app
    if app is not None:
        app.host.tick(cage.PILL_HOVER_S + 0.02)


def rehover() -> None:
    """Recompute the hover at the last known mouse position."""
    if CONTEXT.app is None:
        return
    update_hover(CONTEXT.window, *state.last_mouse)


def schedule_repick() -> None:
    """Recompute the hover now, after the view has moved.

    Rok's camera has moved by the time the overlay draws, so the repick runs
    at once (no wait for the view to settle).
    """
    if state.running and not state.stroke_active:
        rehover()


def _profile_hover(started: float) -> None:
    global _profile_count, _profile_max_ms
    ms = (time.perf_counter() - started) * 1000.0
    if ms > HOVER_BUDGET_MS:
        print(f"ROK hover: one event took {ms:.1f} ms")
    _profile_count += 1
    _profile_max_ms = max(_profile_max_ms, ms)
    if _profile_count >= PROFILE_WINDOW:
        print(f"ROK hover max={_profile_max_ms:.2f} ms over {PROFILE_WINDOW} events")
        _profile_count = 0
        _profile_max_ms = 0.0


# --- lifecycle --------------------------------------------------------------------------


def start(context) -> None:
    """Start the hover operator once the window is up. A second app in the
    same process (the tests) retires the old instance by the generation."""
    state.generation += 1
    state.running = False
    state.stroke_active = False
    _clear_hover_state()
    state.preview = None
    state.profile = os.environ.get("ROK_PROFILE") == "1"
    opkit.ops.rok.voxel_mode("INVOKE")


def _on_load_post(context) -> None:
    """New or Open: the old hover and the flood caches point into the old scene."""
    global _region_cache, _connected_cache
    _clear_hover_state()
    state.preview = None
    _region_cache = None
    _connected_cache = None


registry.startup.append(start)
registry.load_post.append(_on_load_post)
