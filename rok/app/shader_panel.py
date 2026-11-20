"""The shader settings panel: drawn by Rok, at the right edge of the view.

Opened when a shader is picked in the gallery or in Cmd+K (rok.xs_run's
invoke). It shows the shader's picture, name and description, a slider per
argument bracketed to the header's range (rok/core/shader_panel.py), the
colour it runs with, steppers for Colours and Iterations, and Back / Run.
Drag a slider (Shift: finely), or turn the wheel over it for one step.
Enter runs, Esc or a click outside closes, Back returns to the gallery. The
model shows the result live while the panel is open; the wheel and the
trackpad beside the panel turn the view. The run itself is rok.xs_run with
the panel's values, so it is one undo step.
"""

from __future__ import annotations

from .opkit import IntProp, Action, StringProp, ops, register

import time

import numpy as np

from rok.core import chrome as kit
from rok.core import shader_panel as sp
from rok.core import recolour as recolour_core
from rok.core import shader_scope, xs
from rok.core.colour_store import rgba
from .gfx import pen
from . import object as voxel_object
from . import colour_picker, overlay
from .handlers import registry
from .opkit import CONTEXT
from .paint import clip_text as _clip_text
from .paint import round_outline as _round_outline
from .paint import round_rect as _round
from .paint import text as _text
from .paint import text_width as _text_width
from .object import document

PANEL_BG = (0.060, 0.056, 0.053, 0.97)
PANEL_EDGE = (0.228, 0.212, 0.196, 1.0)
STAGE_BG = (0.0, 0.0, 0.0, 0.22)
TRACK_BG = (1.0, 1.0, 1.0, 0.10)
BUTTON_BG = (1.0, 1.0, 1.0, 0.07)
BUTTON_HOVER = (0.208, 0.192, 0.180, 1.0)
EMBER = (0.910, 0.365, 0.227, 1.0)
EMBER_DARK = (0.102, 0.055, 0.031, 1.0)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)
TEXT_FAINT = (0.450, 0.430, 0.410, 1.0)

_session: "_Session | None" = None


class _Session:
    __slots__ = ("window", "region_ptr", "shader", "values", "hover", "drag", "notes",
                 "colours", "sources", "recolour", "picker", "pick_hsv", "pick_drag", "pick_hover",
                 "base", "base_palette", "base_uid", "shown", "pending", "ms", "status", "failed",
                 "scope", "faces", "only", "used", "where_open",
                 "volume", "vol_shape", "vol_hover", "vol_drag", "live", "orig", "orig_palette", "colours_before",
                 "hover_since")

    def __init__(self, window, region_ptr, shader: xs.Shader):
        self.window = window
        self.region_ptr = region_ptr
        self.shader = shader
        self.values: dict[str, float] = {}
        self.hover: tuple[str, str] | None = None
        self.hover_since = 0.0  # when the mouse came to rest on what `hover` names
        self.drag: tuple[str, float, float, bool] | None = None  # key, start value, start x, fine
        self.notes: list[str] = []
        # The colours the run puts in (from the last preview), as the swatches
        # show them: with your changes.
        self.sources: list[tuple[int, int, int]] = []
        self.colours: list[tuple[int, int, int]] = []
        self.recolour: dict = {}  # your changes: the run's colour -> yours
        self.picker: int | None = None  # the colour whose picker is open
        self.pick_hsv = (0.0, 0.0, 0.0)
        self.pick_drag: str | None = None  # "SV" or "HUE" while dragging in the picker
        self.pick_hover: str | None = None
        # The live preview: the model as it was when the panel opened.
        self.base: np.ndarray | None = None
        self.base_palette: np.ndarray | None = None
        self.base_uid: int | None = None
        self.shown = False  # the grid shows a preview, not the base
        self.pending = False  # a value changed since the last preview
        self.ms = 0.0  # how long the last preview took
        self.status = ""  # what Run would do, or why it cannot
        self.failed = False
        # Where it runs: the scope row.
        self.scope = ""  # "": the selection when there is one
        self.faces: set[str] = set()
        self.only: set[int] = set()  # palette indices
        self.used: list[int] = []  # the model's colours, most used first: the Only colours row
        self.where_open = False
        # The placed volume: [x0, y0, z0, x1, y1, z1], inclusive.
        self.volume: list[int] | None = None
        self.vol_shape = shader_scope.BOX
        self.vol_hover: tuple[int, int] | None = None  # (axis, 0 low face / 1 high face)
        self.vol_drag: tuple | None = None  # (axis, side, start value, start mouse, pixels per cell)
        # A live shader open for a change (live_shaders.py): Remove, Bake, Update.
        # The model shows without it and the ones after it while the panel is
        # open; `orig` is the model as it was, for Esc.
        self.live = None
        self.orig: np.ndarray | None = None
        self.orig_palette: np.ndarray | None = None
        self.colours_before = None  # the scene's colours when the panel opened: (snapshot, version)


# --- the colours a shader puts in: shown, changed, remembered per shader ----------------
#
# Show the colours a shader will use and let me change them.
# After each preview the panel shows a swatch for each colour the run puts
# into the model (rok.core.recolour.shown): the shader's own, the shades it
# makes of your model's colours, the palette colours it takes. A click opens
# a picker (saturation and value, hue, your favourites and this file's
# colours), a right-click puts a colour back. Your changes are a map by
# colour; they stay per shader (settings.txt, "shader_recolours") and go to
# the run as rok.xs_run's recolour.




def _hexes(colours) -> list[str]:
    return ["{:02x}{:02x}{:02x}".format(*c) for c in colours]


def saved_recolour(shader: xs.Shader) -> dict:
    """Your colour changes for this shader from last time. A shader whose own
    colours you picked before (the old "shader_colours") gets those as changes."""
    import json

    from . import settings

    try:
        saved = json.loads(settings.get("shader_recolours") or "{}").get(shader.name)
    except (ValueError, AttributeError):
        saved = None
    if isinstance(saved, str):
        return recolour_core.parse(saved)
    try:
        old = json.loads(settings.get("shader_colours") or "{}").get(shader.name)
    except (ValueError, AttributeError):
        old = None
    own = list(shader.colours.rgb) if shader.colours else []
    if not old or len(old) != len(own):
        return {}
    parsed = xs.parse_hex_list(",".join(old))
    return {o: p for o, p in zip(own, parsed) if p is not None and p != o}


def save_recolour(shader: xs.Shader, mapping: dict) -> None:
    import json

    from . import settings

    try:
        table = json.loads(settings.get("shader_recolours") or "{}")
    except ValueError:
        table = {}
    if mapping:
        table[shader.name] = recolour_core.encode(mapping)
    else:
        table.pop(shader.name, None)
    settings.put("shader_recolours", json.dumps(table))


def _sync(session: "_Session") -> None:
    """Your changes from the swatches: each one that differs from the colour
    the run puts in."""
    for src, now in zip(session.sources, session.colours):
        if now != src:
            session.recolour[src] = now
        else:
            session.recolour.pop(src, None)


def _colour_name(session: "_Session", k: int) -> str:
    """A swatch's name: the shader's name for one of its own colours, else its hex."""
    src = session.sources[k]
    own = session.shader.colours
    if own is not None and src in own.rgb:
        return own.name(list(own.rgb).index(src))
    return "#" + _hexes([src])[0]


_quick_colours = colour_picker.quick_colours
picker_layout = colour_picker.layout


def _open_picker(session: "_Session", k: int) -> None:
    session.picker = k
    session.pick_hsv = colour_picker.hsv_of(session.colours[k])
    session.pick_drag = None


def _set_from_hsv(session: "_Session") -> None:
    session.colours[session.picker] = colour_picker.rgb_of(session.pick_hsv)


def _pick_drag(session: "_Session", part: str, rect, px: float, py: float) -> None:
    session.pick_hsv = colour_picker.dragged(session.pick_hsv, part, rect, px, py)
    _set_from_hsv(session)


def _inside(rect, px: float, py: float) -> bool:
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


# --- the live preview -----------------------------------------------------------------
#
# While the panel is open, the model shows what Run would do (a
# live change shows the real voxels). Each change runs the shader on the
# model as it was when the panel opened and writes the result into the grid
# with no undo record. Run puts the model back and runs rok.xs_run, which is
# one undo step; Esc, Back and a click outside put it back.

PREVIEW_DRAG_MS = 60.0  # a slower preview waits for the end of a drag
PREVIEW_SETTING = "shader_preview"  # settings.txt: "off" turns the live preview off, for every shader
PREVIEW_OFF_NOTE = "Live preview off: Run shows the result"


def preview_on() -> bool:
    """The live preview: on unless you turned it off in the panel (kept)."""
    from . import settings

    return settings.get(PREVIEW_SETTING) != "off"


def _switch_preview(session: "_Session") -> None:
    """The header's Preview switch: off puts the model back as it was; on shows the result at once."""
    from . import settings

    settings.put(PREVIEW_SETTING, "off" if preview_on() else "on")
    if preview_on():
        session.status = ""
        _preview(session)
    else:
        _restore(session)
        session.sources, session.colours, session.picker = [], [], None
        session.status = PREVIEW_OFF_NOTE


def _run_kwargs(session: "_Session") -> dict:
    """rok.xs_run's settings for the panel's values and colours."""
    values = session.values
    kwargs = {f"a{k}": float(values[f"a{k}"]) for k in range(len(session.shader.args))}
    _sync(session)
    if session.recolour:
        kwargs["recolour"] = recolour_core.encode(session.recolour)
    kwargs["colour_count"] = int(values.get("colour_count", 1))
    kwargs["iterations"] = int(values.get("iterations", 1))
    where = _where(session)
    if where.scope:
        kwargs["scope"] = where.scope
    if where.faces:
        kwargs["only_on"] = ",".join(where.faces)
    if where.colours:
        kwargs["only_colours"] = ",".join(str(c) for c in where.colours)
    if where.room:
        kwargs["room"] = where.room
    if where.feather:
        kwargs["feather"] = where.feather
    if where.scope == shader_scope.VOLUME and where.volume is not None:
        kwargs["volume"] = ",".join(str(v) for v in where.volume)
        kwargs["volume_shape"] = where.shape
    return kwargs


def _where(session: "_Session") -> shader_scope.Where:
    return shader_scope.Where(session.scope, tuple(f for f in shader_scope.FACES if f in session.faces),
                              tuple(sorted(session.only)), int(session.values.get("room", 0)),
                              int(session.values.get("feather", 0)),
                              tuple(session.volume) if session.volume is not None else None, session.vol_shape)


def _from_live(session: "_Session", live) -> None:
    """The panel's values from a live shader's settings."""
    st = live.settings
    session.live = live
    for key in list(session.values):
        if key in st:
            session.values[key] = float(st[key])
    session.recolour = recolour_core.parse(st.get("recolour", ""))
    if st.get("colours_rgb") and session.shader.colours is not None:
        # One from before the change map: its picked own colours become changes.
        own = list(session.shader.colours.rgb)
        picked = session.shader.colours.with_rgb(xs.parse_hex_list(st["colours_rgb"])).rgb
        session.recolour.update({o: p for o, p in zip(own, picked) if o != p})
    session.scope = st.get("scope", "")
    session.faces = set(shader_scope.parse_faces(st.get("only_on", "")))
    session.only = set(shader_scope.parse_colours(st.get("only_colours", "")))
    volume = shader_scope.parse_volume(st.get("volume", ""))
    session.volume = None if volume is None else list(volume)
    session.vol_shape = st.get("volume_shape") or shader_scope.BOX


def _has_selection() -> bool:
    return document.selection is not None and not document.selection.is_empty()


ONLY_MAX = 24  # the Only colours row shows the most used colours


def _used_colours() -> list[int]:
    """The model's colours, most used first."""
    grid = document.grid
    if grid is None:
        return []
    counts = np.bincount(grid.visible.ravel(), minlength=256)[1:]
    used = np.flatnonzero(counts) + 1
    return [int(i) for i in used[np.argsort(-counts[used - 1], kind="stable")]][:ONLY_MAX]


def _snapshot(session: "_Session"):
    return (tuple(sorted(session.values.items())), tuple(session.colours), session.scope,
            tuple(sorted(session.faces)), tuple(sorted(session.only)),
            tuple(session.volume or ()), session.vol_shape)


def _same_model(session: "_Session") -> bool:
    grid = document.grid
    return (grid is not None and session.base is not None and document.active_uid == session.base_uid
            and grid.visible.shape == session.base.shape)


def _show(new: np.ndarray, palette: np.ndarray | None) -> None:
    """Write voxels and palette straight into the model: no undo, no unsaved mark."""
    from rok.core import morph

    from .ops_common import refresh_views

    if palette is not None and not np.array_equal(palette, document.scene.palette):
        document.scene.palette[:] = palette
        voxel_object.update_palette()
    grid = document.grid
    cells, values = morph.changed_cells(grid.visible, new)
    if len(cells):
        grid.set_voxels(cells, values)
        voxel_object.flush_dirty()
    refresh_views(serial=True)


def _restore(session: "_Session") -> None:
    """Put the model back as it was when the panel opened."""
    if session.shown and _same_model(session):
        _show(session.base, session.base_palette)
    # A preview's colours are not yours: the strip may have taken them on.
    voxel_object.colours_back(session.colours_before)
    session.shown = False


def _colour(session: "_Session") -> int:
    """The colour it runs with: a live shader keeps its own."""
    if session.live is not None:
        return int(session.live.settings.get("colour", document.colour))
    return int(document.colour)


def _start_preview(session: "_Session") -> None:
    grid = document.grid
    if grid is None or document.scene is None:
        return
    session.base = grid.visible.copy()
    if session.live is not None:
        # Without this shader and the ones after it: they run on its result.
        from rok.core import live_shader

        from . import live_shaders

        shaders = live_shaders.shaders_of()
        k = next(i for i, s in enumerate(shaders) if s.id == session.live.id)
        session.orig = session.base
        session.orig_palette = document.scene.palette.copy()
        session.base = live_shader.take_back(session.base, shaders[k:])
    session.base_palette = document.scene.palette.copy()
    session.colours_before = voxel_object.colours_now()
    session.base_uid = document.active_uid
    if preview_on():
        _preview(session)
    else:
        session.status = PREVIEW_OFF_NOTE


def _preview(session: "_Session") -> None:
    """Show the result of the panel's values on the model."""
    import time

    from rok.core import morph

    from . import xs_ops

    session.pending = False
    if session.base is None or not _same_model(session):
        return
    _restore(session)  # the shader reads the model as it was, and so does the colour plan
    kwargs = _run_kwargs(session)
    raw = [kwargs.get(f"a{k}", 0.0) for k in range(xs.MAX_ARGS)]
    started = time.perf_counter()
    live = session.live
    extra = {} if live is None else {"selection": live.selection, "mirror": live.mirror}
    out = xs_ops.compute(session.shader.name, raw, colour=_colour(session), colour_count=kwargs["colour_count"],
                         iterations=kwargs["iterations"], where=_where(session),
                         recolour=kwargs.get("recolour", ""), **extra)
    session.failed = bool(out.error)
    if out.error:
        session.status = out.error
    else:
        changed = len(morph.changed_cells(session.base, out.new)[0])
        palette = None
        if out.writes:
            palette = session.base_palette.copy()
            for slot, value in out.writes.items():
                palette[slot] = rgba(value)
        _show(out.new, palette)
        session.shown = True
        session.sources = list(out.colours)
        session.colours = [session.recolour.get(c, c) for c in session.sources]
        if session.picker is not None and session.picker >= len(session.colours):
            session.picker = None
        session.status = out.report(changed)[1].split(": ", 1)[-1]
    session.ms = (time.perf_counter() - started) * 1000.0


def _dragging(session: "_Session") -> bool:
    return session.drag is not None or session.pick_drag is not None or session.vol_drag is not None


# --- the placed volume ---------------------------------------------------
#
# "Runs on: Volume" keeps the shader to a box or a sphere of cells, empty ones
# too, so a shader can build in it. The volume shows in the view while the
# panel is open, with a handle on each face: a drag moves that face, and the
# live preview follows.

HANDLE_PX = 7.0  # a face handle's radius
MIN_CELL_PX = 4.0  # a drag moves at least this far per cell


def _default_volume(session: "_Session") -> list[int]:
    """The selection's box, else the voxels' box, else the whole grid. The
    voxels as they were: the grid shows the preview now."""
    from rok.core import grid_ops

    grid = document.grid
    sel = document.selection
    if sel is not None and not sel.is_empty():
        lo, hi = sel.bounds()
        return [*lo, *hi]
    found = grid_ops.fit_bounds(session.base if session.base is not None else grid.voxels)
    if found is not None:
        lo, hi = found
        return [int(v) for v in (*lo, *hi)]
    sx, sy, sz = grid.size
    return [0, 0, 0, sx - 1, sy - 1, sz - 1]


def _to_region(region, point):
    """A point in the active model's cells on the screen, or None behind the view."""
    from .view3d import world_to_region

    view_state = CONTEXT.view_state
    if view_state is None or document.model is None:
        return None
    origin = np.asarray(voxel_object.model_origin(document.model), dtype=np.float64)
    p = world_to_region(region, view_state, tuple(origin + np.asarray(point, dtype=np.float64)))
    return None if p is None else (float(p[0]), float(p[1]))


def _face_centre(volume, axis: int, side: int) -> np.ndarray:
    """The middle of a face of the volume, in cells (the box's corners are lo and hi + 1)."""
    lo = np.asarray(volume[:3], dtype=np.float64)
    hi = np.asarray(volume[3:], dtype=np.float64) + 1.0
    c = (lo + hi) / 2.0
    c[axis] = hi[axis] if side else lo[axis]
    return c


def _volume_handle_at(session: "_Session", region, mx: float, my: float):
    """(axis, side) of the face handle under the pointer, or None."""
    best, found = (HANDLE_PX + 3.0) ** 2, None
    for axis in range(3):
        for side in (0, 1):
            p = _to_region(region, _face_centre(session.volume, axis, side))
            if p is not None:
                d = (p[0] - mx) ** 2 + (p[1] - my) ** 2
                if d < best:
                    best, found = d, (axis, side)
    return found


def _cell_step(session: "_Session", region, axis: int, side: int) -> tuple[float, float]:
    """Region pixels for one cell along `axis` at the face."""
    c = _face_centre(session.volume, axis, side)
    unit = np.zeros(3)
    unit[axis] = 1.0
    a, b = _to_region(region, c), _to_region(region, c + unit)
    if a is None or b is None:
        return (0.0, MIN_CELL_PX)
    v = np.array([b[0] - a[0], b[1] - a[1]])
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return (0.0, MIN_CELL_PX)  # the axis points at the eye: up the screen
    return tuple(v / n * max(n, MIN_CELL_PX))


def _volume_drag(session: "_Session", mx: float, my: float) -> None:
    """Move the dragged face to the cell under the pointer's way along its axis."""
    axis, side, start, (sx, sy), step = session.vol_drag
    step = np.asarray(step, dtype=np.float64)
    cells = int(round(float(np.dot((mx - sx, my - sy), step) / max(float(np.dot(step, step)), 1e-9))))
    size = document.grid.size
    v = session.volume
    if side:
        v[axis + 3] = int(min(max(start + cells, v[axis]), size[axis] - 1))
    else:
        v[axis] = int(min(max(start + cells, 0), v[axis + 3]))


def _draw_volume(context) -> None:
    """The volume and its face handles, under the panel."""
    session = _session
    region = context.region
    if (session is None or region is None or region.as_pointer() != session.region_ptr
            or session.scope != shader_scope.VOLUME or session.volume is None):
        return
    s = overlay.ui_scale()
    v = session.volume
    lo = np.asarray(v[:3], dtype=np.float64)
    hi = np.asarray(v[3:], dtype=np.float64) + 1.0
    segments = []
    if session.vol_shape == shader_scope.SPHERE:
        centre, radius = (lo + hi) / 2.0, (hi - lo) / 2.0
        t = np.linspace(0.0, 2.0 * np.pi, 49)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            ring = np.tile(centre, (len(t), 1))
            ring[:, a] += radius[a] * np.cos(t)
            ring[:, b] += radius[b] * np.sin(t)
            pts = [_to_region(region, p) for p in ring]
            segments += [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    corners = [np.array([hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]])
               for i in range(8)]
    box = [_to_region(region, c) for c in corners]
    for i in range(8):
        for bit in (1, 2, 4):
            if not i & bit:
                segments.append((box[i], box[i | bit]))
    flat = [p for seg in segments if None not in seg for p in seg]
    if flat:
        colour = EMBER if session.vol_shape == shader_scope.BOX else (EMBER[0], EMBER[1], EMBER[2], 0.45)
        pen.lines(pen.to_3d(np.array(flat, np.float32)), colour, 1.5 * s)
        if session.vol_shape == shader_scope.SPHERE:
            rings = [p for seg in segments[:-12] if None not in seg for p in seg]
            pen.lines(pen.to_3d(np.array(rings, np.float32)), EMBER, 1.5 * s)
    for axis in range(3):
        for side in (0, 1):
            p = _to_region(region, _face_centre(v, axis, side))
            if p is None:
                continue
            on = session.vol_hover == (axis, side) or (session.vol_drag is not None
                                                       and session.vol_drag[:2] == (axis, side))
            r = (HANDLE_PX + (1.5 if on else 0.0)) * s
            pen.fan(pen.to_3d(kit.circle_tris(p[0], p[1], r + 1.5 * s)), EMBER_DARK)
            pen.fan(pen.to_3d(kit.circle_tris(p[0], p[1], r)), TEXT if on else EMBER)


def active() -> bool:
    return _session is not None


COLOURS_TIP = ("How many palette colours, counting from the current colour on, the shader works with. "
               "A shader that paints picks among them (noise, stripes); a shader that acts on a marker "
               "colour (flames, cuts) treats every one of them as the marker, so keep it at 1 there.")
ITERATIONS_TIP = ("How many times the shader runs in a row, each run on the result of the one before. "
                  "It matters for shaders that grow or wear step by step (erode, gravity, cover); "
                  "a shader that builds the same thing each time gives the same result.")


def specs_of(shader: xs.Shader) -> list[sp.Spec]:
    """The panel's rows for a shader: its arguments, then the steppers."""
    out = []
    for k, arg in enumerate(shader.args):
        step = arg.step if arg.step > 0 else 10.0 ** -max(0, arg.precision)
        out.append(sp.Spec(f"a{k}", arg.name, arg.low, arg.high, step, max(0, arg.precision), tip=arg.tip))
    if xs.uses_colour_selection(shader):
        out.append(sp.Spec("colour_count", "Colours", 1, 64, 1, 0, stepper=True, tip=COLOURS_TIP))
    out.append(sp.Spec("iterations", "Iterations", 1, 64, 1, 0, stepper=True, tip=ITERATIONS_TIP))
    return out


def start_values(shader: xs.Shader) -> dict[str, float]:
    """The values the panel opens with: the header's defaults; an argument
    that names the colour to work on starts at your current colour."""
    from .xs_ops import COLOURS_DEFAULT

    values = {}
    for k, arg in enumerate(shader.args):
        value = float(document.colour) if xs.is_colour_arg(arg) else arg.value
        values[f"a{k}"] = sp.snap(value, arg.low, arg.high, 0.0, max(0, arg.precision))
    values["colour_count"] = 1.0 if xs.acts_on_selection(shader) else float(COLOURS_DEFAULT)
    values["iterations"] = 1.0
    values["room"] = 0.0
    values["feather"] = 0.0
    return values


def _notes(shader: xs.Shader) -> list[str]:
    from .xs_ops import _colours_note

    notes = []
    if shader.colours is not None:
        notes.append(_colours_note(shader, int(document.colour)))
    if xs.is_brush(shader):
        notes.append("Paints the voxels that are there; adds and erases none")
    return notes


def _view_region(window):
    """The view's region: Rok has one view."""
    return CONTEXT.region


def _side_top(region) -> float:
    """The panel stands at the right edge, from the top: the nav cube hides
    while it is open (nav_cube._hidden)."""
    return float(region.height)


def _layout(session: _Session, region) -> sp.PanelLayout:
    # One more note line: what Run would do, from the live preview.
    return sp.panel_layout(overlay.ui_scale(), region.width, region.height, specs_of(session.shader),
                           notes=len(session.notes) + 1, colours=len(session.colours),
                           side_top=_side_top(region), where_open=session.where_open,
                           where_colours=len(session.used),
                           shape_row=session.scope == shader_scope.VOLUME,
                           live_button=session.live is None)


def _picker(session: _Session, region, layout: sp.PanelLayout) -> dict | None:
    if session.picker is None or session.picker >= len(layout.swatches):
        return None
    return picker_layout(layout.swatches[session.picker], region, layout.scale, len(_quick_colours()))


@register
class ShaderPanelAction(Action):
    command_id = "rok.shader_panel"
    title = "Shader Settings"
    tooltip = "Set a shader up and run it"
    shader = StringProp(name="Shader", options={"SKIP_SAVE"})
    live = IntProp(name="Live Shader", description="A live shader of the active model to change; 0: a new run",
                       default=0, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def invoke(self, context, event):
        global _session
        from .xs_ops import shader_by_name

        live = None
        if self.live:
            from . import live_shaders

            live = live_shaders.find(document.active_uid, int(self.live))
            if live is None:
                self.report({"ERROR"}, "No such live shader")
                return {"CANCELLED"}
            self.shader = live.shader
        found = shader_by_name(self.shader)
        region = _view_region(context.window)
        if found is None or region is None:
            self.report({"ERROR"}, f"No shader named {self.shader}")
            return {"CANCELLED"}
        _session = _Session(context.window, region.as_pointer(), found)
        _session.values = start_values(found)
        _session.recolour = saved_recolour(found)
        _session.notes = _notes(found)
        _session.used = _used_colours()
        from . import settings

        _session.where_open = settings.get("shader_where_open") == "1"
        if live is not None:
            _from_live(_session, live)
        from . import shader_gallery

        shader_gallery._thumb(found.name)  # load it now, not while drawing
        _start_preview(_session)
        context.prefs.add_modal(self)
        voxel_object.redraw_3d_views()
        return {"RUNNING"}

    def modal(self, context, event):
        session = _session
        before = _snapshot(session) if session is not None else None
        result = self._modal(context, event)
        if session is not None and _session is session:
            if _snapshot(session) != before:
                session.pending = True
            if session.pending and preview_on() and (not _dragging(session) or session.ms < PREVIEW_DRAG_MS):
                _preview(session)
        return result

    def _modal(self, context, event):
        session = _session
        if session is None:
            return {"CANCELLED"}
        region = _view_region(session.window)
        if region is None:
            _close()
            return {"CANCELLED"}
        kind, value = event.type, event.value
        mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
        layout = _layout(session, region)
        picker = _picker(session, region, layout)
        if picker is not None:
            done = self._picker_event(session, picker, kind, value, mx, my)
            if done:
                region.tag_redraw()
                return {"RUNNING"}
        if self._volume_event(session, layout, region, event, mx, my):
            region.tag_redraw()
            return {"RUNNING"}
        if kind == "RIGHT_MOUSE" and value == "PRESS":
            found = sp.hit(layout, mx, my)
            if found is not None and found[0] == "COLOUR" and int(found[1]) < len(session.sources):
                k = int(found[1])
                session.colours[k] = session.sources[k]  # back to the colour the run puts in
                region.tag_redraw()
                return {"RUNNING"}
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            if session.drag is not None:
                key, start_value, start_x, _fine = session.drag
                control = layout.control(key)
                if event.shift:
                    session.values[key] = sp.fine(control, start_value, start_x, mx)
                else:
                    session.values[key] = sp.value_at(control, mx)
                    session.drag = (key, session.values[key], mx, False)
            else:
                found = sp.hit(layout, mx, my)
                if found != session.hover:
                    session.hover_since = time.monotonic()
                    if found is not None and found[0] == "LABEL":
                        _schedule_tip()
                session.hover = found
            region.tag_redraw()
            return {"RUNNING"}
        if kind == "LEFT_MOUSE" and value == "RELEASE":
            session.drag = None
            region.tag_redraw()
            return {"RUNNING"}
        if kind == "LEFT_MOUSE" and value == "PRESS":
            found = sp.hit(layout, mx, my)
            if found is None:
                _close()
                return {"CANCELLED"}
            what, key = found
            if what == "PREVIEW":
                _switch_preview(session)
            elif what == "COLOUR":
                k = int(key)
                if session.picker == k:
                    session.picker = None
                else:
                    _open_picker(session, k)
            elif what == "WHERE":
                from . import settings

                session.where_open = not session.where_open
                settings.put("shader_where_open", "1" if session.where_open else "0")
            elif what == "SCOPE":
                if key == shader_scope.VOLUME:
                    if session.volume is None:
                        session.volume = _default_volume(session)
                    session.scope = key
                elif key == shader_scope.MODEL or _has_selection():
                    session.scope = key
            elif what == "SHAPE":
                session.vol_shape = key
            elif what == "FACE":
                session.faces ^= {key}
            elif what == "ONLY":
                k = int(key)
                if k < len(session.used):
                    session.only ^= {session.used[k]}
            elif what == "TRACK":
                control = layout.control(key)
                if not event.shift:
                    session.values[key] = sp.value_at(control, mx)
                session.drag = (key, session.values[key], mx, bool(event.shift))
            elif what in ("MINUS", "PLUS"):
                control = layout.control(key)
                session.values[key] = sp.nudged(control, session.values[key], -1 if what == "MINUS" else 1)
            elif session.live is not None and what in ("RUN", "BRUSH", "BACK"):
                return self._live_change(context, what)
            elif what == "RUN":
                return self._run(context)
            elif what == "LIVE":
                return self._run(context, live=True)
            elif what == "BRUSH":
                return self._brush(context)
            elif what == "BACK":
                _close()
                ops.rok.shader_gallery("INVOKE")
                return {"FINISHED"}
            region.tag_redraw()
            return {"RUNNING"}
        if kind in {"TRACKPAD_PAN", "TRACKPAD_ZOOM", "WHEEL_UP", "WHEEL_DOWN"} \
                and not _inside(layout.rect, mx, my) and CONTEXT.app is not None:
            CONTEXT.app.navigate(event)  # turn the view to look at the preview
            return {"RUNNING"}
        if kind in {"WHEEL_UP", "WHEEL_DOWN"}:
            found = sp.hit(layout, mx, my)
            if found is not None and found[0] in ("TRACK", "MINUS", "PLUS"):
                control = layout.control(found[1])
                session.values[control.key] = sp.nudged(control, session.values[control.key],
                                                         1 if kind == "WHEEL_UP" else -1)
                region.tag_redraw()
            return {"RUNNING"}
        if value == "PRESS" and kind in {"ENTER", "NUMPAD_ENTER"}:
            return self._run(context)
        if value == "PRESS" and kind in {"ESCAPE", "RIGHT_MOUSE"}:
            _close()
            return {"CANCELLED"}
        if kind == "FOCUS_LOST":
            _close()
            return {"CANCELLED"}
        return {"RUNNING"}

    @staticmethod
    def _volume_event(session: _Session, layout, region, event, mx, my) -> bool:
        """The volume's face handles in the view. True when the event was theirs."""
        if session.scope != shader_scope.VOLUME or session.volume is None:
            return False
        kind, value = event.type, event.value
        if session.vol_drag is not None:
            if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
                _volume_drag(session, mx, my)
                return True
            if kind == "LEFT_MOUSE" and value == "RELEASE":
                session.vol_drag = None
                return True
            return kind not in {"ESCAPE", "FOCUS_LOST"}
        if _inside(layout.rect, mx, my):
            session.vol_hover = None
            return False
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            session.vol_hover = _volume_handle_at(session, region, mx, my)
            return session.vol_hover is not None
        if kind == "LEFT_MOUSE" and value == "PRESS" and session.vol_hover is not None:
            axis, side = session.vol_hover
            step = _cell_step(session, region, axis, side)
            session.vol_drag = (axis, side, session.volume[axis + 3 * side], (mx, my), step)
            return True
        return False

    def _picker_event(self, session: _Session, picker: dict, kind, value, mx, my) -> bool:
        """An event while a colour's picker is open. True when the picker took it."""
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            if session.pick_drag is not None:
                _pick_drag(session, session.pick_drag, picker[session.pick_drag], mx, my)
                return True
            session.pick_hover = next((n for n, r in picker.items() if n != "PANEL" and _inside(r, mx, my)), None)
            return _inside(picker["PANEL"], mx, my)
        if kind == "LEFT_MOUSE" and value == "RELEASE" and session.pick_drag is not None:
            session.pick_drag = None
            return True
        if kind == "LEFT_MOUSE" and value == "PRESS":
            if not _inside(picker["PANEL"], mx, my):
                hit = sp.hit(_layout(session, _view_region(session.window)), mx, my)
                if hit is not None and hit[0] == "COLOUR":
                    return False  # another swatch: the panel opens that one
                session.picker = None  # a click outside the picker closes it
                return True
            part = next((n for n, r in picker.items() if n != "PANEL" and _inside(r, mx, my)), None)
            if part in ("SV", "HUE"):
                session.pick_drag = part
                _pick_drag(session, part, picker[part], mx, my)
            elif part is not None and part.startswith("Q:"):
                quick = _quick_colours()
                q = int(part[2:])
                if q < len(quick):
                    import colorsys

                    session.colours[session.picker] = quick[q]
                    r, g, b = quick[q]
                    session.pick_hsv = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            elif part == "RESET":
                import colorsys

                own = session.sources[session.picker]
                session.colours[session.picker] = own
                session.pick_hsv = colorsys.rgb_to_hsv(*(c / 255.0 for c in own))
            return True
        if value == "PRESS" and kind in {"ESCAPE", "ENTER", "NUMPAD_ENTER"}:
            session.picker = None  # Esc or Enter closes the picker first
            return True
        return False

    def cancel(self, context):
        _close()

    def _brush(self, context):
        """Arm the shader brush with the panel's settings and go to Paint."""
        from . import shader_brush, tools

        session = _session
        _close()  # puts the model back: the brush paints from there
        if session is None:
            return {"CANCELLED"}
        _sync(session)
        save_recolour(session.shader, session.recolour)
        shader_brush.arm(session.shader.name, _run_kwargs(session))
        ops.rok.set_tool(tool="PAINT")
        ops.rok.set_mode(mode="VOXEL")
        tools._note(f"Shader brush: {shader_brush.label()} · paint it on · Esc puts it away")
        voxel_object.redraw_3d_views()
        return {"FINISHED"}

    def _run(self, context, live: bool = False):
        session = _session
        _close()  # puts the model back: the run is one undo step from there
        if session is None or not ops.rok.xs_run.poll():
            self.report({"INFO"}, "Open a model first: a shader runs on the model")
            return {"CANCELLED"}
        _sync(session)
        save_recolour(session.shader, session.recolour)
        ops.rok.xs_run("EXEC", shader=session.shader.name, colour=0, live=live, ask_full=True,
                        **_run_kwargs(session))
        return {"FINISHED"}

    def _live_change(self, context, what: str):
        """A live shader's buttons: Remove (Back's place), Bake (Brush's), Update (Run's)."""
        from . import live_shaders

        session = _session
        _close()  # the model as it was, with every live shader
        live = session.live
        uid = document.active_uid
        if what == "BACK":
            live_shaders.remove(uid, live.id)
            self.report({"INFO"}, f"{live_shaders.label(live)} removed")
        elif what == "BRUSH":
            live_shaders.bake(uid, live.id)
            self.report({"INFO"}, f"{live_shaders.label(live)} baked: its voxels are yours now")
        else:
            _sync(session)
            save_recolour(session.shader, session.recolour)
            settings = live_shaders.settings_from(_run_kwargs(session), _colour(session))
            done = live_shaders.update(uid, live.id, settings)
            if done is not None and done.error:
                self.report({"ERROR"}, done.error)
        return {"FINISHED"}


def _close() -> None:
    global _session
    if _session is not None:
        _restore(_session)
        if _session.orig is not None and _same_model(_session):
            _show(_session.orig, _session.orig_palette)  # the live shaders as they were
            voxel_object.colours_back(_session.colours_before)
    _session = None
    voxel_object.redraw_3d_views()


# --- drawing -------------------------------------------------------------------------


def draw(region, scale: float) -> None:
    """The panel, over everything else in the view. Reads `_session`."""
    session = _session
    if session is None or region.as_pointer() != session.region_ptr:
        return
    from . import shader_gallery

    layout = _layout(session, region)
    s = layout.scale
    shader = session.shader
    hover = session.hover or ("", "")
    if True:
        _round(layout.rect, 14 * s, PANEL_BG)
        _round_outline(layout.rect, 14 * s, PANEL_EDGE, 1.0 * s)
        # The header: the picture, the name, what it makes, who wrote it.
        _round(layout.thumb, 10 * s, STAGE_BG)
        tex = shader_gallery._thumb(shader.name)
        if tex is not None:
            shader_gallery._draw_image(tex, layout.thumb)
        tx, ty, tw, th = layout.title
        _text(_clip_text(xs.label(shader.name.rpartition("/")[2]), tw, 20 * s), tx, ty + th / 2, 20 * s, TEXT)
        if layout.preview_switch is not None:
            # The live preview's switch: ember while on.
            rect = layout.preview_switch
            on = preview_on()
            lit = hover[0] == "PREVIEW"
            _round(rect, rect[3] / 2, EMBER if on else (BUTTON_HOVER if lit else BUTTON_BG))
            _text("Preview" if on else "Preview off", rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, 11 * s,
                  EMBER_DARK if on else (TEXT if lit else TEXT_DIM), "centre")
        group = shader.name.rpartition("/")[0]
        ax, ay, aw, ah = layout.about
        line_y = ay + ah - 12 * s
        if group:
            _text(group.replace("_", " ").upper(), tx, ty - 4 * s, 10 * s, TEXT_FAINT)
            line_y -= 12 * s
        for line in sp.wrap(shader.about or "Your own shader", lambda t: _text_width(t, 13 * s), aw, lines=3):
            _text(line, ax, line_y, 13 * s, TEXT_DIM)
            line_y -= 18 * s
        if shader.author and shader.author != "Rok":
            _text(f"by {shader.author}", ax, line_y, 11 * s, TEXT_FAINT)
        left, right = layout.rect[0] + 20 * s, layout.rect[0] + layout.rect[2] - 20 * s
        pen.lines(pen.to_3d(np.array([(left, layout.divider), (right, layout.divider)], np.float32)),
                            PANEL_EDGE, 1.0 * s)
        # The rows.
        dragging = session.drag[0] if session.drag is not None else None
        for c in layout.controls:
            value = session.values.get(c.key, c.low)
            lit = c.key == dragging or hover[1] == c.key
            lx, ly, lw, lh = c.label_rect
            _text(_clip_text(c.label, lw - 8 * s, 12 * s), lx, ly + lh / 2, 12 * s, TEXT if lit else TEXT_DIM)
            vx, vy, vw, vh = c.value_rect
            if c.stepper:
                for rect, sign, part in ((c.minus, "−", "MINUS"), (c.plus, "+", "PLUS")):
                    on = hover == (part, c.key)
                    _round(rect, rect[3] / 2, BUTTON_HOVER if on else BUTTON_BG)
                    _text(sign, rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, 14 * s, TEXT if on else TEXT_DIM, "centre")
                bx, by, bw, bh = c.track
                _text(sp.text_of(value, 0), bx + bw / 2, by + bh / 2, 13 * s, TEXT, "centre")
                continue
            bx, by, bw, bh = c.track
            _round(c.track, bh / 2, TRACK_BG)
            knob_x = sp.x_of(c, value)
            if knob_x > bx:
                _round((bx, by, knob_x - bx, bh), bh / 2, EMBER)
            steps = sp.span_steps(c.low, c.high, c.step)
            if 0 < steps <= 16:
                # A tick per step when there are few: the slider snaps to them.
                for k in range(steps + 1):
                    x = bx + bw * k / steps
                    pen.fan(pen.to_3d(kit.circle_tris(x, by - 5 * s, 1.3 * s)), TEXT_FAINT)
            # The range, faint, under the ends of the bar.
            _text(sp.text_of(c.low, c.precision), bx, by - 9 * s, 9 * s, TEXT_FAINT)
            _text(sp.text_of(c.high, c.precision), bx + bw, by - 9 * s, 9 * s, TEXT_FAINT, "right")
            r = (8.0 if lit else 7.0) * s
            pen.fan(pen.to_3d(kit.circle_tris(knob_x, by + bh / 2, r)), TEXT)
            pen.fan(pen.to_3d(kit.circle_tris(knob_x, by + bh / 2, r - 2.5 * s)), EMBER if lit else PANEL_BG)
            if c.key.startswith("a") and xs.is_colour_arg(shader.args[int(c.key[1:])]):
                # An argument that names a colour shows that colour.
                pal = document.scene.palette if document.scene is not None else None
                if pal is not None:
                    idx = max(0, min(255, int(value)))
                    rgb = tuple(float(v) / 255.0 for v in pal[idx][:3])
                    _round((vx, vy + vh / 2 - 8 * s, 16 * s, 16 * s), 3 * s, rgb + (1.0,))
            _text(sp.text_of(value, c.precision), vx + vw, vy + vh / 2, 13 * s, TEXT if lit else TEXT_DIM, "right")
        # The colours the run puts in: a swatch each, the open one ringed.
        if layout.colours_label is not None:
            lx, ly, lw, lh = layout.colours_label
            _text("Its colours", lx, ly + lh / 2, 12 * s, TEXT_DIM)
            for k, rect in enumerate(layout.swatches):
                rgb = tuple(c / 255.0 for c in session.colours[k])
                _round(rect, 5 * s, rgb + (1.0,))
                changed = session.colours[k] != session.sources[k]
                ring = EMBER if session.picker == k else (TEXT if hover == ("COLOUR", str(k)) else PANEL_EDGE)
                _round_outline(rect, 5 * s, ring, (2.0 if session.picker == k or hover == ("COLOUR", str(k)) else 1.0) * s)
                if changed:  # a dot: your colour, not the one the run puts in
                    pen.fan(pen.to_3d(kit.circle_tris(rect[0] + rect[2] - 4 * s, rect[1] + rect[3] - 4 * s,
                                                                2.5 * s)), TEXT)
        _draw_where(session, layout, hover)
        # The notes, the colour it runs with first.
        for rect, note in zip(layout.notes, session.notes):
            nx, ny, nw, nh = rect
            _text(_clip_text(note, nw, 11 * s), nx, ny + nh / 2, 11 * s, TEXT_FAINT)
        if session.status and len(layout.notes) > len(session.notes):
            nx, ny, nw, nh = layout.notes[len(session.notes)]
            _text(_clip_text(session.status, nw, 11 * s), nx, ny + nh / 2, 11 * s,
                  EMBER if session.failed else TEXT_DIM)
        # Back, Brush and Run.
        editing = session.live is not None
        buttons = [(layout.back, "Remove" if editing else "Back", "BACK"),
                   (layout.run, "Update" if editing else "Run", "RUN")]
        if layout.brush is not None:
            buttons.insert(1, (layout.brush, "Bake" if editing else "Brush", "BRUSH"))
        if layout.live is not None:
            buttons.insert(2, (layout.live, "Live", "LIVE"))
        for rect, word, part in buttons:
            on = hover[0] == part
            if part == "RUN":
                _round(rect, rect[3] / 2, EMBER)
                if on:
                    _round_outline(rect, rect[3] / 2, TEXT, 1.5 * s)
                _text(word, rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, 13 * s, EMBER_DARK, "centre")
            else:
                _round(rect, rect[3] / 2, BUTTON_HOVER if on else BUTTON_BG)
                _text(word, rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, 13 * s, TEXT if on else TEXT_DIM, "centre")
        # The colour it runs with, beside the buttons.
        if document.scene is not None:
            idx = _colour(session)
            rgb = tuple(float(v) / 255.0 for v in document.scene.palette[idx][:3])
            sx = layout.rect[0] + 20 * s
            sy = layout.run[1] + (layout.run[3] - 20 * s) / 2
            _round((sx, sy, 20 * s, 20 * s), 4 * s, rgb + (1.0,))
            _round_outline((sx, sy, 20 * s, 20 * s), 4 * s, PANEL_EDGE, 1.0 * s)
            if layout.live is None:  # four buttons leave no room for the words
                _text(f"Colour {idx}", sx + 28 * s, sy + 10 * s, 11 * s, TEXT_DIM)
        _draw_tip(session, layout, region.width)


def _schedule_tip() -> None:
    """Ask for the redraw that shows a row's tooltip: a mouse that rests sends no event."""
    app = CONTEXT.app
    if app is not None:
        app.host.tick(kit.TOOLTIP_DELAY_S + 0.02)


def _draw_tip(session: _Session, layout: sp.PanelLayout, region_width: float) -> None:
    """The tooltip of the row whose name the mouse rests on: what the argument
    does and when it has an effect, wrapped in a box left of the panel."""
    hover = session.hover
    if hover is None or hover[0] != "LABEL" or session.drag is not None:
        return
    if not kit.tooltip_due(session.hover_since, time.monotonic()):
        return
    control = layout.control(hover[1])
    if control is None or not control.tip:
        return
    s = layout.scale
    lines = sp.wrap(control.tip, lambda t: _text_width(t, 12 * s), (sp.TIP_W - 2 * sp.TIP_PAD) * s,
                    lines=sp.TIP_LINES)
    rect = sp.tip_rect(layout, control, len(lines) + 1, region_width)
    _round(rect, 8 * s, PANEL_BG)
    _round_outline(rect, 8 * s, PANEL_EDGE, 1.0 * s)
    x, y, w, h = rect
    line_y = y + h - (sp.TIP_PAD + sp.TIP_LINE / 2) * s
    _text(control.label, x + sp.TIP_PAD * s, line_y, 12 * s, TEXT)
    for line in lines:
        line_y -= sp.TIP_LINE * s
        _text(line, x + sp.TIP_PAD * s, line_y, 12 * s, TEXT_DIM)


def _draw_where(session: _Session, layout: sp.PanelLayout, hover) -> None:
    """The Where row, and when it is open: Runs on, Only on, Only colours."""
    if layout.where_row is None:
        return
    s = layout.scale
    wx, wy, ww, wh = layout.where_row
    on = hover[0] == "WHERE"
    has_selection = _has_selection()
    where = _where(session)
    _text("Where", wx, wy + wh / 2, 12 * s, TEXT if on else TEXT_DIM)
    arrow = "▾" if session.where_open else "▸"
    summary = _clip_text(f"{where.summary(has_selection)}  {arrow}", ww - 80 * s, 12 * s)
    _text(summary, wx + ww, wy + wh / 2, 12 * s, TEXT if on or where.limits(has_selection) else TEXT_DIM, "right")
    for label, (lx, ly, lw, lh) in layout.where_labels:
        _text(_clip_text(label, lw - 6 * s, 12 * s), lx, ly + lh / 2, 12 * s, TEXT_DIM)
    current = where.resolved(has_selection)
    for kind, chips, labels in (("SCOPE", layout.scope_chips, shader_scope.SCOPE_LABELS),
                                ("SHAPE", layout.shape_chips, shader_scope.SHAPE_LABELS),
                                ("FACE", layout.face_chips, shader_scope.FACE_LABELS)):
        for key, rect in chips:
            lit = {"SCOPE": key == current, "SHAPE": key == session.vol_shape}.get(kind, key in session.faces)
            usable = kind != "SCOPE" or key in (shader_scope.MODEL, shader_scope.VOLUME) or has_selection
            over = hover == (kind, key) and usable
            _round(rect, rect[3] / 2, EMBER if lit else (BUTTON_HOVER if over else BUTTON_BG))
            ink = EMBER_DARK if lit else (TEXT if over else (TEXT_DIM if usable else TEXT_FAINT))
            word = _clip_text(labels[key], rect[2] - 6 * s, 11 * s)
            _text(word, rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, 11 * s, ink, "centre")
    palette = document.scene.palette if document.scene is not None else None
    for k, rect in enumerate(layout.only_swatches):
        if palette is None or k >= len(session.used):
            break
        idx = session.used[k]
        rgb = tuple(float(v) / 255.0 for v in palette[idx][:3])
        _round(rect, 4 * s, rgb + (1.0,))
        picked = idx in session.only
        over = hover == ("ONLY", str(k))
        ring = EMBER if picked else (TEXT if over else PANEL_EDGE)
        _round_outline(rect, 4 * s, ring, (2.0 if picked or over else 1.0) * s)
    if session.where_open and not layout.only_swatches and layout.where_labels:
        lx, ly, lw, lh = layout.where_labels[-1][1]
        _text("The model has no voxels", lx + lw, ly + lh / 2, 11 * s, TEXT_FAINT)


def _draw_handler(context) -> None:
    if _session is not None and context.region is not None:
        draw(context.region, overlay.ui_scale())


def _draw_picker(context) -> None:
    """The open colour's picker, or the name of the swatch under the mouse: a
    layer above the panel, so the panel's text does not show through."""
    session = _session
    region = context.region
    if session is None or region is None or region.as_pointer() != session.region_ptr:
        return
    layout = _layout(session, region)
    s = layout.scale
    picker = _picker(session, region, layout)
    if picker is None:
        hover = session.hover or ("", "")
        if hover[0] == "COLOUR" and int(hover[1]) < len(session.sources):
            k = int(hover[1])
            rect = layout.swatches[k]
            name = _colour_name(session, k)
            text = f"{name} · click to pick, right-click puts it back"
            w = _text_width(text, 11 * s) + 16 * s
            x = min(rect[0], region.width - w - 4 * s)
            y = rect[1] + rect[3] + 6 * s
            _round((x, y, w, 22 * s), 6 * s, PANEL_BG)
            _round_outline((x, y, w, 22 * s), 6 * s, PANEL_EDGE, 1.0 * s)
            _text(text, x + 8 * s, y + 11 * s, 11 * s, TEXT)
        return
    name = _colour_name(session, session.picker)
    colour_picker.draw(picker, session.pick_hsv, _quick_colours(), session.colours[session.picker],
                       session.pick_hover, f"Back to {name}", s)


registry.add_draw("shader_volume", 755, "PIXEL", _draw_volume)
registry.add_draw("shader_panel", 760, "PIXEL", _draw_handler)
registry.add_draw("shader_panel_picker", 761, "PIXEL", _draw_picker)
