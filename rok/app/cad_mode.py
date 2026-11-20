"""CAD view: an orthographic picture of the real voxels.

Each voxel face has its own grey, lines sit only on sharp corners, and the
side labels read width, depth and height. A click on an edge reads that
edge's length. Hidden edges can be shown as dotted lines, with the faces
washed out. One voxel length, in metres or feet, is kept on the scene.
Until it is set, the labels stay in cells. Painting waits until the view is
off again. The camera from before the view comes back.
"""

from __future__ import annotations

import zlib

import numpy as np

from rok.core import cad as cad_core
from rok.core.grid_ops import fit_bounds

from . import gfx
from .gfx import fonts, pen
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, register

HINT = "CAD view: each face has its own grey. Click an edge for its length, or the length field to set metres or feet."
TYPE_HINT = "Type the length of one voxel, in m or ft. Enter keeps it. Esc leaves it."

_INK = (0.10, 0.10, 0.14, 1.0)
_PLATE = (0.97, 0.97, 0.98, 1.0)
_EDGE = (0.22, 0.22, 0.26, 0.55)
_CHIP = (0.90, 0.90, 0.92, 1.0)
_CHIP_ON = (0.16, 0.16, 0.20, 1.0)
_CHIP_INK = (1.0, 1.0, 1.0, 1.0)

_BAR_X = 72.0  # clear of the rail
_BAR_H = 26.0
_FIELD_W = 78.0
_CHIP_W = 28.0
_HIDDEN_W = 70.0
_GAP = 4.0

_saved = None  # the camera from before the view
_cache: dict[int, tuple] = {}
_edges: dict[int, tuple] = {}  # uid -> (cache key, segments, idents, lookup)
_hits: dict[str, tuple[float, float, float, float]] = {}
_edit: str | None = None
_edit_fresh = False
_selected: tuple | None = None  # (uid, axis, fixed a, fixed b, start, end)
_hover: tuple | None = None
_hidden = False  # see-through: hidden edges are dotted lines
_ghost: dict[int, tuple] = {}  # uid -> (edge key, direction, scene key, runs)

_NUMPAD = {f"NUMPAD_{d}": str(d) for d in range(10)}
_NUMPAD["NUMPAD_PERIOD"] = "."


def editing() -> bool:
    """True while the voxel length is being typed."""
    return _edit is not None


def set_on(app, on: bool) -> None:
    """Turn the CAD view on or off. On replaces the rendered view and the quad
    view, and keeps the camera so leaving can put it back. Off does not bring
    those views back."""
    global _saved
    on = bool(on)
    if on == bool(getattr(app, "cad", False)):
        return
    if on:
        from . import render_view

        if render_view.rendered(app):
            render_view.set_rendered(app, False)
        if app.quad:
            app.set_quad(False)
        cam = app.main_camera
        _saved = (cam.ortho, cam.locked, cam.yaw, cam.pitch, cam.distance,
                  np.array(cam.target, dtype=np.float64, copy=True), cam.up_hint)
        cam.ortho = True
        cam.locked = False
        from .object import frame_scene

        frame_scene(CONTEXT)
        app.cad = True
        app.host.set_cad(True)
        _send_hidden(app)
        _cache.clear()
        _clear_pick()
        sync_fields(app)
    else:
        app.cad = False
        app.host.set_cad(False)
        _send_hidden(app)
        _cache.clear()
        _clear_pick()
        _cancel_edit()
        if _saved is not None:
            ortho, locked, yaw, pitch, distance, target, up_hint = _saved
            cam = app.main_camera
            cam.ortho, cam.locked = ortho, locked
            cam.yaw, cam.pitch, cam.distance = yaw, pitch, distance
            cam.target = np.array(target, dtype=np.float64, copy=True)
            cam.up_hint = up_hint
            _saved = None
    app.set_menu()
    app.camera_moved()


def sync_fields(app, dirty=None) -> None:
    """Upload a face field for each model the view is showing. A model whose
    voxels did not change keeps the field it already has."""
    global _cache
    if not getattr(app, "cad", False) or document.scene is None:
        _cache.clear()
        _edges.clear()
        _drop_lost()
        return
    from . import anim_mode

    scene = document.scene
    shown = anim_mode.shown()
    uids = {m.uid for m in scene.models}
    for uid in [u for u in _cache if u not in uids]:
        del _cache[uid]
    for uid in [u for u in _edges if u not in uids]:
        del _edges[uid]
    for i, model in enumerate(scene.models):
        buf = _shown_voxels(model, shown)
        pose = shown.poses.get(model.uid)
        key = (i, buf.shape, zlib.crc32(buf), None if pose is None else pose[0])
        cached = _edges.get(model.uid)
        if cached is None or cached[0] != key:
            segments, ekeys = cad_core.straight_edges(buf)
            idents = tuple(tuple(int(v) for v in row) for row in ekeys)
            _edges[model.uid] = (key, segments, idents, set(idents))
        # A whole-model upload builds a new GPU model with no field. A dirty
        # chunk clears the field that was fitted to the old voxels.
        replaced = getattr(app.sync, "replaced", ())
        force = i in replaced or bool(dirty and i < len(dirty) and dirty[i])
        if not force and _cache.get(model.uid) == key:
            continue
        app.host.set_cad_field(i, cad_core.field(buf))
        _cache[model.uid] = key
    _drop_lost()


def after_flush(app, dirty=None) -> None:
    """After the grids reached the GPU. A voxel edit clears the field there,
    so this sends it again when the voxels changed."""
    sync_fields(app, dirty)


def set_unit(scene, unit: str) -> None:
    """Show lengths in metres or feet. The stored length stays in metres."""
    if scene is None:
        return
    unit = "ft" if unit == "ft" else "m"
    if scene.cell_unit == unit:
        return
    scene.cell_unit = unit
    document.modified = True
    _redraw()


def set_length(scene, text: str) -> bool:
    """Keep `text` as the voxel length, in the scene's unit. Empty clears it
    back to cells. False when `text` is not a length."""
    if scene is None:
        return False
    try:
        value = cad_core.parse_length(text)
    except ValueError:
        return False
    metres = None if value is None else cad_core.to_metres(value, scene.cell_unit)
    if metres != scene.cell_metres:
        scene.cell_metres = metres
        document.modified = True
    _redraw()
    return True


def hidden() -> bool:
    """True when hidden edges are drawn as dotted lines."""
    return _hidden


def set_hidden(app, on: bool) -> None:
    """Show or hide the dotted hidden edges. The choice stays when the view
    goes off, and comes back with it. The editing view is never see-through."""
    global _hidden
    on = bool(on)
    if on == _hidden:
        return
    _hidden = on
    _send_hidden(app)
    _redraw()


def _send_hidden(app) -> None:
    if app is None:
        return
    app.host.set_cad_hidden(bool(_hidden) and bool(getattr(app, "cad", False)))


def press(context, x: float, y: float, left: bool) -> bool:
    """A click on the length bar or on an edge. True when the click is done.

    The bar wins, and it leaves the measured edge as it is. An edge click
    measures that edge, or clears it when it was already measured. A miss
    clears the edge and returns False, so the rest of the click still runs.
    The view does not paint either way."""
    global _selected
    if not left or not getattr(context.app, "cad", False):
        return False
    name = _hit(x, y)
    if name is not None:
        if name == "field":
            _open_field()
        elif name in ("m", "ft"):
            set_unit(document.scene, name)
        elif name == "hidden":
            set_hidden(context.app, not _hidden)
        return True
    region = context.region
    view = context.view_state
    if region is None or view is None:
        return False
    found = _edge_at(context.app, view.camera, float(region.height), float(x), float(y))
    if found is None:
        if _selected is not None:
            _selected = None
            _redraw()
        return False
    if found == _selected:
        _selected = None
    else:
        _selected = found
    _redraw()
    return True


def pointer(x: float, y: float) -> None:
    """The edge under the cursor, so the view can light it before a click."""
    global _hover
    app = CONTEXT.app
    if app is None or not getattr(app, "cad", False) or document.scene is None:
        if _hover is not None:
            _hover = None
            _redraw()
        return
    region = CONTEXT.region
    view = CONTEXT.view_state
    if region is None or view is None:
        return
    found = _edge_at(app, view.camera, float(region.height), float(x), float(y))
    if found == _hover:
        return
    _hover = found
    _redraw()


def selected_edge():
    """The measured edge as (model, axis, fixed a, fixed b, start, end), or None."""
    if _selected is None or not _known(_selected):
        return None
    return _selected


def edge_label(scene) -> str | None:
    """The measured edge in the scene's unit, or None when nothing is measured."""
    edge = selected_edge()
    if edge is None or scene is None:
        return None
    return cad_core.side_text(edge[1], edge[5] - edge[4], scene.cell_metres, scene.cell_unit)


def status_text() -> str:
    """The hint line: how to measure an edge, or how to clear the one shown."""
    label = edge_label(document.scene)
    if label is not None:
        return f"Edge {label}. Click empty space or press Esc to clear it."
    if _hidden:
        return "See-through: hidden edges are dotted lines. Click an edge for its length."
    return HINT


@register
class CadViewAction(Action):
    command_id = "rok.cad_view"
    title = "CAD View"
    tooltip = "Each face has its own grey. Click an edge for its length. Click again for the editing view."

    def execute(self, context):
        set_on(context.app, not getattr(context.app, "cad", False))
        return {"FINISHED"}


@register
class CadClearEdgeAction(Action):
    """Esc drops the measured edge. The next Esc deselects voxels."""

    command_id = "rok.cad_clear_edge"
    title = "Clear Measured Edge"
    tooltip = "Clear the measured edge"

    @classmethod
    def poll(cls, context):
        return getattr(context.app, "cad", False) and selected_edge() is not None

    def execute(self, context):
        _clear_edge()
        return {"FINISHED"}


@register
class CadLengthAction(Action):
    """Type the length of one voxel into the bar: the field opens with the
    number selected, the first key replaces it, Enter keeps it, Esc leaves it."""

    command_id = "rok.cad_length"
    title = "Type Voxel Length"
    tooltip = "Type the length of one voxel, in m or ft"

    @classmethod
    def poll(cls, context):
        return getattr(context.app, "cad", False) and document.scene is not None and _edit is None

    def invoke(self, context, event):
        global _edit, _edit_fresh
        scene = document.scene
        if scene.cell_metres is None or scene.cell_metres <= 0:
            _edit = ""
        else:
            _edit = cad_core.display_number(scene.cell_metres, scene.cell_unit)
        _edit_fresh = True
        context.prefs.add_modal(self)
        _redraw()
        return {"RUNNING"}

    def modal(self, context, event):
        global _edit, _edit_fresh
        if _edit is None or document.scene is None or not getattr(context.app, "cad", False):
            return self._end(cancelled=True)
        kind, value = event.type, event.value
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN", "TIMER", "TIMER_REPORT", "WHEEL_UP",
                    "WHEEL_DOWN", "TRACKPAD_PAN", "TRACKPAD_ZOOM", "FOCUS_LOST"}:
            return {"PASS"}
        if value != "PRESS":
            return {"RUNNING"}
        if kind in {"ENTER", "NUMPAD_ENTER"}:
            if not set_length(document.scene, _edit):
                self.report({"WARNING"}, "Type a length like 0.25, or leave it empty for cells")
                return {"RUNNING"}
            return self._end()
        if kind == "ESCAPE":
            return self._end(cancelled=True)
        if kind in {"LEFT_MOUSE", "RIGHT_MOUSE", "MIDDLE_MOUSE"}:
            self._end(cancelled=True)
            return {"CANCELLED", "PASS"}
        if kind == "BACKSPACE":
            if _edit_fresh:
                _edit, _edit_fresh = "", False
            else:
                _edit = _edit[:-1]
        else:
            char = _NUMPAD.get(kind) or event.unicode
            if char and (char.isdigit() or char == "."):
                if _edit_fresh:
                    _edit, _edit_fresh = "", False
                if char != "." or "." not in _edit:
                    if len(_edit) < 12:
                        _edit += char
        _redraw()
        return {"RUNNING"}

    def cancel(self, context):
        self._end(cancelled=True)

    def _end(self, cancelled: bool = False):
        global _edit, _edit_fresh
        _edit = None
        _edit_fresh = False
        _redraw()
        return {"CANCELLED"} if cancelled else {"FINISHED"}


def _open_field() -> None:
    from . import opkit

    if "rok.cad_length" in opkit.REGISTRY and opkit.ops.rok.cad_length.poll():
        opkit.ops.rok.cad_length("INVOKE")


def _cancel_edit() -> None:
    global _edit, _edit_fresh
    _edit = None
    _edit_fresh = False


def _clear_pick() -> None:
    """Drop the measured edge, the hover and the edge cache."""
    global _selected, _hover
    _selected = None
    _hover = None
    _edges.clear()
    _ghost.clear()


def _clear_edge() -> None:
    """Drop the measured edge and the hover. The edge cache stays."""
    global _selected, _hover
    _selected = None
    _hover = None
    _redraw()


def _drop_lost() -> None:
    """A measured edge that the voxels no longer have is no longer measured."""
    global _selected, _hover
    if _selected is not None and not _known(_selected):
        _selected = None
    if _hover is not None and not _known(_hover):
        _hover = None


def _known(item) -> bool:
    scene = document.scene
    if scene is None:
        return False
    model = next((m for m in scene.models if m.uid == item[0]), None)
    if model is None or scene.is_hidden(model):
        return False
    cached = _edges.get(item[0])
    return cached is not None and item[1:] in cached[3]


def _redraw() -> None:
    app = CONTEXT.app
    if app is not None:
        app.request_redraw()


def _shown_voxels(model, shown) -> np.ndarray:
    """The voxels the GPU is showing for `model`."""
    pose = shown.poses.get(model.uid)
    if pose is not None and pose[1] is not None:
        return np.ascontiguousarray(pose[1], dtype=np.uint8)
    return np.ascontiguousarray(model.grid.visible, dtype=np.uint8)


def _origin(model, shown) -> np.ndarray:
    pose = shown.poses.get(model.uid)
    place = pose[2] if pose is not None else model.position
    ox, oy, oz = document.view_offset
    return np.array([ox + place[0], oy + place[1], oz + place[2]], dtype=np.float64)


def _filled_box(scene, shown):
    """World lo (inclusive) and hi (exclusive) of the filled cells on show."""
    los, his = [], []
    for model in scene.models:
        if scene.is_hidden(model):
            continue
        bounds = fit_bounds(_shown_voxels(model, shown))
        if bounds is None:
            continue
        origin = _origin(model, shown)
        lo, hi = bounds
        los.append(origin + np.asarray(lo, dtype=np.float64))
        his.append(origin + np.asarray(hi, dtype=np.float64) + 1.0)
    if not los:
        return None
    return np.min(los, axis=0), np.max(his, axis=0)


def _listed(app):
    """Visible models as (uid, origin, segments, idents)."""
    if not getattr(app, "cad", False) or document.scene is None:
        return []
    from . import anim_mode

    scene = document.scene
    shown = anim_mode.shown()
    out = []
    for model in scene.models:
        if scene.is_hidden(model):
            continue
        cached = _edges.get(model.uid)
        if cached is None or len(cached[1]) == 0:
            continue
        out.append((model.uid, _origin(model, shown), cached[1], cached[2]))
    return out


def _pick_radius(cam) -> float:
    """Pixels. Wide enough to catch an edge, short of the next cell and of
    the corner line the picture draws inside the silhouette."""
    pixel = float(cam.pixel)
    if not np.isfinite(pixel) or pixel <= 0.0:
        return 8.0
    return max(8.0, 0.45 / pixel)


def _edge_at(app, cam, height: float, x: float, y: float):
    """The edge nearest the cursor, as (uid, axis, fixed a, fixed b, start, end)."""
    items = _listed(app)
    if not items:
        return None
    starts, ends, depth, ids = [], [], [], []
    eye = np.asarray(cam.eye(), dtype=np.float64)
    for uid, origin, segments, idents in items:
        world = np.asarray(segments, dtype=np.float64) + origin
        proj = cam.project(world.reshape(-1, 3)).reshape(-1, 2, 2)
        screen = np.empty_like(proj)
        screen[:, :, 0] = proj[:, :, 0]
        screen[:, :, 1] = height - proj[:, :, 1]
        starts.append(screen[:, 0])
        ends.append(screen[:, 1])
        depth.append(np.linalg.norm(world.mean(axis=1) - eye, axis=1))
        ids.extend((uid,) + ident for ident in idents)
    if not ids:
        return None
    hit = cad_core.nearest_segment(
        np.concatenate(starts), np.concatenate(ends),
        np.array([x, y], dtype=np.float64), _pick_radius(cam), np.concatenate(depth),
    )
    if hit < 0:
        return None
    return ids[hit]


def _segment_of(items, item):
    """The two world corners of `item`, or None."""
    want = item[1:]
    for uid, origin, segments, idents in items:
        if uid != item[0]:
            continue
        try:
            index = idents.index(want)
        except ValueError:
            return None
        return np.asarray(segments[index], dtype=np.float64) + origin
    return None


def _hit(x: float, y: float) -> str | None:
    for name in ("field", "m", "ft", "hidden", "bar"):
        rect = _hits.get(name)
        if rect is not None and rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]:
            return name
    return None


def _draw(context) -> None:
    """Side labels, the measured edge and the bar that sets the voxel length."""
    global _hits
    app = context.app
    if app is None or not getattr(app, "cad", False) or document.scene is None:
        _hits = {}
        return
    region = context.region
    view = context.view_state
    if region is None or view is None:
        _hits = {}
        return
    _draw_labels(app, region, view)
    _draw_edges(app, region, view)
    _draw_bar(region)


def _draw_labels(app, region, view) -> None:
    from . import anim_mode

    box = _filled_box(document.scene, anim_mode.shown())
    if box is None:
        return
    lo, hi = box
    cam = view.camera
    toward = cam.eye() - (lo + hi) * 0.5
    scene = document.scene
    height = float(region.height)
    fonts.size(0, 13.0)
    fonts.color(0, *_INK)
    for axis, (start, end) in enumerate(cad_core.dimension_ends(lo, hi, toward)):
        pts = cam.project([start, end])
        if not np.isfinite(pts).all():
            continue
        p0 = np.array([pts[0, 0], height - pts[0, 1]])
        p1 = np.array([pts[1, 0], height - pts[1, 1]])
        delta = p1 - p0
        span = float(np.linalg.norm(delta))
        if span < 1.0:
            continue
        side = np.array([-delta[1], delta[0]]) / span
        tick = side * 5.0
        pen.lines([p0, p1], _INK, 1.25)
        pen.lines([p0 - tick, p0 + tick, p1 - tick, p1 + tick], _INK, 1.25)
        cells = int(round(float(hi[axis] - lo[axis])))
        text = cad_core.side_text(axis, cells, scene.cell_metres, scene.cell_unit)
        w, th = fonts.dimensions(0, text)
        mid = (p0 + p1) * 0.5 + side * (8.0 + th * 0.5)
        fonts.position(0, float(mid[0] - w / 2.0), float(mid[1] - th / 2.0), 0)
        fonts.draw(0, text)


def _draw_bar(region) -> None:
    global _hits
    scene = document.scene
    y = float(region.height) - 16.0 - _BAR_H
    x = _BAR_X
    field = (x, y, _FIELD_W, _BAR_H)
    m = (x + _FIELD_W + _GAP, y, _CHIP_W, _BAR_H)
    ft = (m[0] + _CHIP_W + _GAP, y, _CHIP_W, _BAR_H)
    hidden = (ft[0] + ft[2] + _GAP, y, _HIDDEN_W, _BAR_H)
    bar = (x - 4.0, y - 4.0, hidden[0] + hidden[2] - x + 8.0, _BAR_H + 8.0)
    _hits = {"bar": bar, "field": field, "m": m, "ft": ft, "hidden": hidden}
    gfx.rrect(bar, _PLATE, 8.0)
    gfx.rrect(bar, _EDGE, 8.0, stroke=1.0)
    gfx.rrect(field, (1.0, 1.0, 1.0, 1.0), 6.0)
    gfx.rrect(field, _EDGE, 6.0, stroke=1.0)
    unit = scene.cell_unit if scene.cell_unit in ("m", "ft") else "m"
    for name, rect in (("m", m), ("ft", ft)):
        on = name == unit
        gfx.rrect(rect, _CHIP_ON if on else _CHIP, 6.0)
        _chip_text(name, rect, _CHIP_INK if on else _INK)
    gfx.rrect(hidden, _CHIP_ON if _hidden else _CHIP, 6.0)
    _chip_text("Hidden", hidden, _CHIP_INK if _hidden else _INK)
    if _edit is not None:
        text = _edit or " "
    elif scene.cell_metres is None or scene.cell_metres <= 0:
        text = "cells"
    else:
        text = cad_core.display_number(scene.cell_metres, unit)
    _chip_text(text, field, _INK)


def _hidden_runs(app, cam) -> np.ndarray:
    """Hidden edge pieces in world coordinates. Kept until the view turns."""
    empty = np.zeros((0, 2, 3), np.float64)
    if document.scene is None:
        return empty
    from . import anim_mode

    toward = np.asarray(cam.eye() - cam.target, dtype=np.float64)
    norm = float(np.linalg.norm(toward))
    if norm < 1e-8:
        return empty
    toward = toward / norm
    direction_key = tuple(round(float(v), 4) for v in toward)
    shown = anim_mode.shown()
    scene = document.scene
    prepared = []
    for model in scene.models:
        if scene.is_hidden(model):
            continue
        cached = _edges.get(model.uid)
        if cached is None or len(cached[1]) == 0:
            continue
        prepared.append((model.uid, _origin(model, shown), cached[0], cached[1]))
    scene_key = tuple((item[0], item[2]) for item in prepared)
    missing = []
    for uid, _at, key, _segments in prepared:
        memo = _ghost.get(uid)
        if memo is None or memo[0] != key or memo[1] != direction_key or memo[2] != scene_key:
            missing.append(uid)
    if missing:
        bufs = {}
        origins = {}
        for model in scene.models:
            if scene.is_hidden(model):
                continue
            bufs[model.uid] = _shown_voxels(model, shown)
            origins[model.uid] = _origin(model, shown)
        for uid, origin, key, segments in prepared:
            if uid not in missing:
                continue
            covers = [
                (origin - origins[other], bufs[other])
                for other in bufs
                if other != uid
            ]
            runs = cad_core.hidden_segments(bufs[uid], segments, toward, covers)
            _ghost[uid] = (key, direction_key, scene_key, runs)
    live = {item[0] for item in prepared}
    for uid in [uid for uid in _ghost if uid not in live]:
        del _ghost[uid]
    parts = [
        _ghost[uid][3] + at
        for uid, at, _key, _segments in prepared
        if len(_ghost[uid][3])
    ]
    if not parts:
        return empty
    return np.concatenate(parts)


def _draw_dotted(cam, height: float, runs: np.ndarray) -> None:
    """Hidden edges as short dashes, with a light halo so they read on a dark face."""
    if len(runs) == 0:
        return
    proj = cam.project(np.asarray(runs, dtype=np.float64).reshape(-1, 3)).reshape(-1, 2, 2)
    dashes: list[tuple[float, float]] = []
    for pair in proj:
        if not np.isfinite(pair).all():
            continue
        _dash(
            (float(pair[0, 0]), height - float(pair[0, 1])),
            (float(pair[1, 0]), height - float(pair[1, 1])),
            dashes,
        )
    if not dashes:
        return
    pen.lines(dashes, _PLATE, 3.2)
    pen.lines(dashes, _INK, 1.35)


def _dash(start: tuple[float, float], end: tuple[float, float], out: list) -> None:
    delta = (end[0] - start[0], end[1] - start[1])
    span = float(np.hypot(delta[0], delta[1]))
    if span < 2.0:
        return
    step = (delta[0] / span, delta[1] / span)
    dash, gap = 5.0, 4.0
    travelled = 0.0
    while travelled < span - 0.5:
        stop = min(travelled + dash, span)
        out.append((start[0] + step[0] * travelled, start[1] + step[1] * travelled))
        out.append((start[0] + step[0] * stop, start[1] + step[1] * stop))
        travelled = stop + gap


def _draw_edges(app, region, view) -> None:
    """The edge under the cursor, the measured edge, and dotted hidden edges."""
    from . import anim_mode

    cam = view.camera
    height = float(region.height)
    if _hidden:
        _draw_dotted(cam, height, _hidden_runs(app, cam))
    items = _listed(app)
    if not items:
        return
    centre = _centre_screen(cam, height, _filled_box(document.scene, anim_mode.shown()))
    if _hover is not None and _hover != _selected:
        _draw_one(cam, height, items, _hover, centre, False)
    if _selected is not None:
        _draw_one(cam, height, items, _selected, centre, True)


def _draw_one(cam, height, items, item, centre, measured: bool) -> None:
    world = _segment_of(items, item)
    if world is None:
        return
    pair = _screen_pair(cam, height, world)
    if pair is None:
        return
    p0, p1 = pair
    _trace(p0, p1)
    if not measured:
        return
    text = edge_label(document.scene)
    if text:
        _draw_measure(p0, p1, text, centre)


def _screen_pair(cam, height, world):
    proj = cam.project(world)
    if not np.isfinite(proj).all():
        return None
    p0 = np.array([proj[0, 0], height - proj[0, 1]])
    p1 = np.array([proj[1, 0], height - proj[1, 1]])
    return p0, p1


def _centre_screen(cam, height, box):
    if box is None:
        return None
    lo, hi = box
    proj = cam.project([(lo + hi) * 0.5])[0]
    if not np.isfinite(proj).all():
        return None
    return np.array([proj[0], height - proj[1]])


def _trace(p0, p1, width: float = 2.0) -> None:
    """A light halo, then the edge. The halo stays visible on a dark face."""
    pen.lines([p0, p1], _PLATE, width + 2.5)
    pen.lines([p0, p1], _INK, width)


def _draw_measure(p0, p1, text: str, centre) -> None:
    """A dimension beside the edge, with the length on a light plate.

    A short edge, or one that points at the camera, keeps the plate and
    skips the line."""
    delta = p1 - p0
    span = float(np.linalg.norm(delta))
    mid = (p0 + p1) * 0.5
    fonts.size(0, 13.0)
    th = fonts.dimensions(0, text)[1]
    if span >= 12.0:
        side = np.array([-delta[1], delta[0]]) / span
        if centre is not None and float(np.dot(side, mid - centre)) < 0.0:
            side = -side
        off = side * 16.0
        q0, q1 = p0 + off, p1 + off
        pen.lines([p0, q0, p1, q1], _INK, 1.0)
        pen.lines([q0, q1], _INK, 1.25)
        tick = side * 5.0
        pen.lines([q0 - tick, q0 + tick, q1 - tick, q1 + tick], _INK, 1.25)
        anchor = (q0 + q1) * 0.5 + side * (12.0 + th * 0.5)
    else:
        anchor = mid + _away(mid, centre) * 18.0
    _plate(anchor, text)


def _away(mid, centre) -> np.ndarray:
    """A unit step from the model centre toward `mid`, or straight up."""
    if centre is None:
        return np.array([0.0, 1.0])
    away = mid - centre
    length = float(np.linalg.norm(away))
    if length <= 1.0:
        return np.array([0.0, 1.0])
    return away / length


def _plate(anchor, text: str) -> None:
    fonts.size(0, 13.0)
    fonts.color(0, *_INK)
    w, th = fonts.dimensions(0, text)
    pad_x, pad_y = 6.0, 3.0
    rect = (
        float(anchor[0] - w / 2.0 - pad_x), float(anchor[1] - th / 2.0 - pad_y),
        float(w + 2.0 * pad_x), float(th + 2.0 * pad_y),
    )
    gfx.rrect(rect, _PLATE, 5.0)
    gfx.rrect(rect, _EDGE, 5.0, stroke=1.0)
    fonts.position(0, float(anchor[0] - w / 2.0), float(anchor[1] - th / 2.0), 0)
    fonts.draw(0, text)


def _chip_text(text: str, rect, colour) -> None:
    fonts.size(0, 12.0)
    fonts.color(0, *colour)
    w, h = fonts.dimensions(0, text)
    fonts.position(0, float(rect[0] + (rect[2] - w) / 2.0), float(rect[1] + (rect[3] - h) / 2.0), 0)
    fonts.draw(0, text)


registry.add_draw("cad", 520, "PIXEL", _draw)
