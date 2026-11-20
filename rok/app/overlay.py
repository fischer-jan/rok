"""The 3D overlay and the palette strip, drawn over the voxels.

Two draw handlers.
`_draw_view` (the world pass) draws the model boxes, the mirror planes, the
section planes, the selection, the hover highlight, the drag preview, the
model ghost, the selection cage and the transform gizmo in world space.
`_draw_pixel` (the pixel pass) draws the palette strip, the file label with the
typed grid size, the tool HUD, the selection pill strip, the corner gizmo,
the 256 grid on demand, the screen polygon of a Rect or Lasso drag, the
preview label and the palette drag's label, in region pixels (origin at the
bottom left). Both read the context and pass values to the pure draw
functions `draw_scene`, `draw_strip` and `draw_palette`.

The gizmo or the selection cage is drawn last with the depth test off. Every draw
projects gizmo anchors into `state.gizmo_screen` (or cage handles into
`state.cage_screen`) and names its region, so the hit test in mode.py and
tools.py matches the last drawn frame. A selection with Select draws the cage
instead of the gizmo cones.

Notes:
- Colours are sRGB, as the host takes them; the palette goes in as it is.
- Rok's layers skip unchanged content, so the arrays are built on each draw.
- `_Space` pulls the points toward the eye (DEPTH_BIAS).
- Animate mode is parked: no timeline dock, no anchor dot, no posed pill.
- The rail, the star, the command palette and the welcome screen belong to
  their own modules. When such a module registers no draw handler of its
  own, this pass calls its `draw`.
"""

from __future__ import annotations

import sys

import contextlib
import time
import traceback
from dataclasses import replace

import numpy as np

from rok.core import cage, cage_chips, cage_drag, corner_gizmo, edit, gizmo, hud, lattice
from rok.core import chrome as kit
from rok.core import palette_dock
from rok.core import palette_anim
from rok.core.edit import ACTIVATE, ATTACH, ERASE, PAINT, SELECT, VOXEL
from rok.core.hud import PaletteLayout, StripLayout

from . import object as voxel_object
from . import gfx
from .draw import DrawList as _DrawList
from .gfx import fonts, pen
from .gfx import on_top as _on_top
from .handlers import registry
from .opkit import CONTEXT
from .paint import clip_text, text_width
from .state import Hover, hover_key, state

# After the first exception in a callback both callbacks stop drawing until
# the next New or Open, and the traceback prints once.
_disabled = False

_strip_layout_cache: tuple[object, StripLayout] | None = None

TOOL_COLOURS = {
    "ATTACH": (0.4, 1.0, 0.4),
    "ERASE": (1.0, 0.3, 0.3),
    "SELECT": (1.0, 0.75, 0.2),
    "ACTIVATE": (1.0, 1.0, 1.0),
}
BOX_COLOUR = (1.0, 1.0, 1.0, 0.25)
BOX_ALPHAS = {"ACTIVE": 0.25, "OTHER": 0.08, "HOVER": 0.6}
EMBER_RGB = (0.91, 0.365, 0.227)  # the scene's picked model
PICKED_ALPHA = 0.8
FRAME_WHITE = (1.0, 1.0, 1.0, 1.0)
FRAME_BLACK = (0.0, 0.0, 0.0, 1.0)
FRAME_GREY = (0.8, 0.8, 0.8, 1.0)
TEXT_COLOUR = (1.0, 1.0, 1.0, 1.0)
LINE_RGB = (1.0, 1.0, 1.0)  # the guide lines in the view: box edges, planes, the gizmo centre
AMBER = (1.0, 0.75, 0.2)
FACE_ALPHA = 0.4
AXIS_COLOURS = gizmo.AXIS_COLOURS  # x, y, z guides
SECTION_FILL_ALPHA = 0.06
PLANE_PREVIEW_FILL = 0.16  # the plane under a nav cube X, Y or Z button
MASK_RGB = (0.36, 0.62, 1.0)  # the mask: a cool blue, apart from the amber selection
MASK_FILL_ALPHA = 0.30
MASK_EDGE_ALPHA = 0.55
SECTION_EDGE_ALPHA = 0.55
LEVEL_FILL_ALPHA = 0.10
LEVEL_EDGE_ALPHA = 0.7
EDGE_ALPHA = 0.9
MIRROR_FILL_ALPHA = 0.08
MIRROR_EDGE_ALPHA = 0.35
MIRROR_HOVER_ALPHA = 0.45
SELECTION_FILL_ALPHA = 0.28
SELECTION_MOVE_ALPHA = 0.35
SELECTION_EDGE_ALPHA = 0.9
PREVIEW_FILL_ALPHA = 0.35
PREVIEW_EDGE_ALPHA = 0.6
PREVIEW_BOX_ALPHA = 0.9
PREVIEW_GHOST_ALPHA = 0.2
# The Voxel hover shows the result, not an outline. Attach and Paint put the voxels a click would add or paint into the
# renderer as ghosts (host.set_ghost): the renderer shades them as it shades
# every voxel, so the preview is the real thing in every look. Erase blanks the
# voxels it would take the same way (an erase ghost). Without a ghost-capable
# host the result is drawn as shaded faces instead (FACE_TONES: a bright top,
# two sides, a dark underside) and Erase as a red tint.
VOXEL_PREVIEW_ALPHA = 0.92
VOXEL_PREVIEW_MIRROR_ALPHA = 0.5
ERASE_PREVIEW_ALPHA = 0.6
ERASE_CAGE_ALPHA = 0.9  # the Erase hover's cage over the voxels a click would take
ERASE_CAGE_LIMIT = 8_000  # erased cells with a full cage (brush 12, under 0.5 ms); more get the box around them
FACE_TONES = {(0, 0, 1): 1.0, (0, 0, -1): 0.55, (1, 0, 0): 0.82, (-1, 0, 0): 0.82, (0, 1, 0): 0.68, (0, -1, 0): 0.68}
_TONE_TABLE = np.array([[FACE_TONES[tuple(-1 if i == a else 0 for i in range(3))],
                         FACE_TONES[tuple(1 if i == a else 0 for i in range(3))]] for a in range(3)], np.float32)
PREVIEW_EDGE_LIMIT = 512  # cells_edges only up to this many preview cells
SCREEN_ALPHA = 0.9
SCREEN_DASH_PX = 6.0
SCREEN_GAP_PX = 4.0
BUTTON_BG = (1.0, 1.0, 1.0, 0.08)
BUTTON_BG_ON = (0.91, 0.46, 0.16, 0.85)  # ember: the pressed 256 button
EMBER = (0.91, 0.46, 0.16, 1.0)  # the accent
CAGE_LINE_PX = 1.0
CAGE_HOVER_PX = 2.0
CAGE_FILL_ALPHA = 0.14
CAGE_GLYPH_BG = (0.082, 0.076, 0.072, 0.92)
BAR_BG = (0.060, 0.056, 0.053, 1.0)  # the plate of the selection bar
BAR_EDGE = (0.30, 0.285, 0.272, 1.0)  # its outline and the lines between groups
CAGE_GLYPH_ON = (0.070, 0.060, 0.055, 1.0)
CAGE_GLYPH_HOVER = (0.14, 0.128, 0.120, 0.94)
# The palette strip in Rok's look: the plate of the tool panel, rounded tiles.
STRIP_PLATE_BG = (0.11, 0.10, 0.10, 0.86)
STRIP_PLATE_EDGE = (1.0, 1.0, 1.0, 0.07)
TIP_BG = (0.06, 0.055, 0.05, 0.97)  # a tooltip over the strip
STRIP_PLATE_RADIUS = 12
STRIP_PLATE_PAD = 10
STRIP_SHIFT = 12  # px the strip and the grid sit away from the edge and the rail
STRIP_LABEL_H = 13  # the file label's line on the plate
STRIP_CAPTION_INK = (0.53, 0.51, 0.49, 1.0)  # "USED", "UNUSED", "FAV" upright left of the strip's sections
STRIP_LABEL_INK = (0.905, 0.885, 0.855, 0.78)
STRIP_SWATCH_GAP = 1.0  # px of plate between two swatches
STRIP_SWATCH_RADIUS = 3
RING_CLIP_ROOM = 3  # px a scrolled section's clip grows left, right and top for the swatch rings
STRIP_CURRENT_RADIUS = 8
STRIP_TILE_EDGE = (1.0, 1.0, 1.0, 0.30)
STRIP_HOVER_RING = (1.0, 1.0, 1.0, 0.85)
STRIP_DIVIDER = (1.0, 1.0, 1.0, 0.10)
STRIP_BUTTON_BG = (0.22, 0.21, 0.20, 1.0)
STRIP_BUTTON_HOVER = (0.30, 0.29, 0.28, 1.0)
STRIP_BUTTON_RADIUS = 6
PANEL_BG = (0.055, 0.050, 0.048, 0.78)  # the rounded panel behind the palette strip (classic UI)
PILL_BG = (0.10, 0.092, 0.088, 0.88)  # behind a label, a step lighter than the panel
PANEL_FLAT_BG = (0.082, 0.076, 0.072, 0.92)  # square, the colour of the rail
PANEL_FLAT_EDGE = (1.0, 1.0, 1.0, 0.18)  # its one-pixel outline
PANEL_PAD = 6  # px, times scale
PANEL_RADIUS = 8  # px, times scale
PILL_PAD = 5  # px around a label, times scale
GLYPH_INK = (0.780, 0.760, 0.735, 1.0)  # chrome.GLYPH: the grey of the rail's icons
GLYPH_HOVER_INK = (1.0, 0.985, 0.960, 1.0)
# The Size field while typing: its plate, its text, the text on the ember mark, a frame that is not a size yet.
SIZE_FIELD_BG = (0.06, 0.056, 0.053, 1.0)
SIZE_FIELD_INK = (0.95, 0.93, 0.9, 1.0)
SIZE_FIELD_INK_LIT = (0.07, 0.06, 0.055, 1.0)
SIZE_FIELD_OFF = (0.5, 0.48, 0.46, 1.0)
# The gizmo.
GIZMO_PLANE_ALPHA = 0.5
GIZMO_FADE_ALPHA = 0.25  # edge-on rings, and the other handles during a drag
GIZMO_LINE_PX = 1.5
GIZMO_RING_PX = 2.0
# Depth bias for everything drawn on the voxel faces. The quads and lines sit
# 0.003 units off the faces, less than the depth buffer resolves far away, so
# they would fight the voxels for depth. Scaling about the eye by (1 - bias)
# pulls the overlay closer by a fixed fraction of its distance: no screen
# position moves in perspective, and the gap grows with the distance exactly
# as the buffer's precision shrinks.
DEPTH_BIAS = 1e-3

# Layer order of the two passes: other modules draw above (the rail, menus).
VIEW_Z = 10
PIXEL_Z = 100

_QUAD_OUTLINE = np.array([0, 1, 1, 2, 2, 3, 3, 0])
_MOVE = ("MOVE_X", "MOVE_Y", "MOVE_Z")
_PLANE = ("PLANE_YZ", "PLANE_XZ", "PLANE_XY")
_ROT = ("ROT_X", "ROT_Y", "ROT_Z")
_FLIP = ("FLIP_X", "FLIP_Y", "FLIP_Z")

# Palette colours go to the host as they are (sRGB). True would turn them
# linear first.
PALETTE_LINEARIZE = False


def _mod(name: str):
    """A sibling module, or None while it is not imported."""
    import importlib

    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ImportError:
        return None


# --- the view space ---------------------------------------------------------------------


class _Space:
    """Where the view pass draws: grid points of one model go through its
    `matrix` (a turn preview, or None), then to `origin`, then toward the eye
    by DEPTH_BIAS."""

    eye = np.zeros(3)
    origin = np.zeros(3)
    matrix = None


@contextlib.contextmanager
def _space(origin, matrix=None, eye=None):
    saved = (_Space.origin, _Space.matrix, _Space.eye)
    _Space.origin = np.asarray(origin, dtype=np.float64)
    _Space.matrix = None if matrix is None else np.asarray(matrix, dtype=np.float64)
    if eye is not None:
        _Space.eye = np.asarray(eye, dtype=np.float64)
    try:
        yield
    finally:
        _Space.origin, _Space.matrix, _Space.eye = saved


def _world(points, bias: bool = True) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if _Space.matrix is not None:
        p = p @ _Space.matrix[:3, :3].T + _Space.matrix[:3, 3]
    p = p + _Space.origin
    if bias:
        p = _Space.eye + (p - _Space.eye) * (1.0 - DEPTH_BIAS)
    return p.astype(np.float32)


def _tris(points, colour, bias: bool = True) -> None:
    p = np.asarray(points).reshape(-1, 3)
    if len(p):
        pen.tris(_world(p, bias), colour)


def _lines(points, colour, width: float, bias: bool = True) -> None:
    p = np.asarray(points).reshape(-1, 3)
    if len(p) >= 2:
        pen.lines(_world(p, bias), colour, width)


def _fan(points, colour) -> None:
    pen.fan(_world(points), colour)


# --- small queries -----------------------------------------------------------------------


def invalidate_palette() -> None:
    """Drop the strip layout. The next draw builds it again."""
    global _strip_layout_cache
    _strip_layout_cache = None


def ui_scale() -> float:
    """Pixels per point: 1.0 in Rok, which works in logical pixels (the host
    doubles them on a Retina screen)."""
    return 1.0


def palette_grid_shown() -> bool:
    """The 256 toggle."""
    wm = CONTEXT.prefs
    return bool(getattr(wm, "rok_show_palette_grid", False))


def _palette_rail_width() -> float:
    """Left-rail inset in points for the colour strip and 256 grid."""
    return float(hud.HUD_RAIL_PX)


def is_main_region(region) -> bool:
    """Rok has one view: it carries the rail, the palette and the readout."""
    return True


# The corner gizmo's axis colours: Rok's own (rok.core.gizmo.AXIS_COLOURS).
CORNER_AXIS_RGBA = tuple((*c, 1.0) for c in gizmo.AXIS_COLOURS)
CORNER_CENTRE_RGBA = (0.78, 0.76, 0.735, 1.0)


def corner_cell_px(region, view_state):
    """One cell along each world axis at the selection's centre, in region
    pixels, or None without a selection."""
    from .view3d import world_to_region

    selection = voxel_object.document.selection
    model = voxel_object.document.model
    if selection is None or model is None or selection.is_empty() or view_state is None:
        return None
    lo, hi = selection.bounds()
    origin = voxel_object.model_origin(model)
    centre = np.array([origin[i] + (lo[i] + hi[i] + 1) / 2.0 for i in range(3)])
    p0 = world_to_region(region, view_state, centre)
    if p0 is None:
        return None
    out = {}
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1.0
        p1 = world_to_region(region, view_state, centre + step)
        if p1 is not None:
            out[axis] = (float(p1[0] - p0[0]), float(p1[1] - p0[1]))
    return out


def corner_hit(region, mx: float, my: float) -> str | None:
    """The corner gizmo handle under a region pixel, as last drawn there."""
    found = state.corner
    if found is None or found[0] != region.as_pointer():
        return None
    return corner_gizmo.hit_test(found[1], mx, my)


def _draw_corner_gizmo(region, scale: float) -> None:
    """The move arrows and scale boxes at the bottom right, while something is selected."""
    cell_px = corner_cell_px(region, CONTEXT.view_state)
    if cell_px is None:
        if state.corner is not None and state.corner[0] == region.as_pointer():
            state.corner = None
        return
    layout = corner_gizmo.corner_layout(region.width, region.height, scale, cell_px)
    state.corner = (region.as_pointer(), layout, cell_px)
    hovered = state.hover_handle if corner_gizmo.is_handle(state.hover_handle) else None
    cx, cy = layout.centre
    white = (1.0, 1.0, 1.0, 1.0)
    for axis, (ux, uy) in layout.dirs.items():
        colour = CORNER_AXIS_RGBA[axis]
        tx, ty = layout.tips[axis]
        pen.lines(np.array([[cx, cy], [tx, ty]], dtype=np.float32), colour, 2.0 * scale)
        # The arrow head: a triangle at the tip.
        head = 9.0 * scale
        px, py = -uy, ux
        tri = np.array(
            [[tx + ux * head, ty + uy * head], [tx + px * head * 0.6, ty + py * head * 0.6],
             [tx - px * head * 0.6, ty - py * head * 0.6]],
            dtype=np.float32,
        )
        on = hovered == corner_gizmo.MOVE_IDS[axis]
        pen.tris(tri, white if on else colour)
        bx, by = layout.boxes[axis]
        half = 5.0 * scale
        on = hovered == corner_gizmo.SCALE_IDS[axis]
        pen.fan(hud.rounded_rect(bx - half, by - half, 2 * half, 2 * half, 1.0 * scale), white if on else colour)
    for axis, ring in (layout.rings or {}).items():
        on = hovered == corner_gizmo.ROT_IDS[axis]
        r, g, b, _a = CORNER_AXIS_RGBA[axis]
        pairs = np.stack([ring, np.roll(ring, -1, axis=0)], axis=1).reshape(-1, 2)
        pen.lines(pairs, white if on else (r, g, b, 0.75), (2.0 if on else 1.5) * scale)
    half = 6.5 * scale
    on = hovered == corner_gizmo.SCALE_ALL
    pen.fan(hud.rounded_rect(cx - half, cy - half, 2 * half, 2 * half, 1.5 * scale), white if on else CORNER_CENTRE_RGBA)


# --- the lattice cage ---------------------------------------

CAGE_HIT_PX = 10.0  # hit radius of a cage point, times the UI scale
CAGE_EDGE_POINT_PX = 7.0  # corner and edge-middle points
CAGE_FACE_POINT_PX = 5.0  # face centres and the centre, fainter
CAGE_FACE_POINT_ALPHA = 0.6
CAGE_LINE_ALPHA = 0.55
CAGE_MIRROR_LINE_ALPHA = 0.3  # the lines of Mirror's copy of the cage
CAGE_CHIP_LIT_INK = (0.10, 0.09, 0.085, 1.0)


def cage_hit(region, mx: float, my: float) -> str | None:
    """The id of the cage point under a region pixel, as last drawn there: the
    nearest to the eye when two overlap."""
    found = state.cage.layout if state.cage is not None else None
    if found is None or found[0] != region.as_pointer():
        return None
    _ptr, ids, pts, depth = found
    d = np.hypot(pts[:, 0] - mx, pts[:, 1] - my)
    near = np.flatnonzero(d <= CAGE_HIT_PX * ui_scale())  # nan compares False
    if not len(near):
        return None
    return ids[int(near[np.argmin(depth[near])])]


def cage_chip_hit(region, mx: float, my: float) -> str | None:
    """The cage's chip under a region pixel, as last drawn there."""
    found = state.cage.chips if state.cage is not None else None
    if found is None or found[0] != region.as_pointer():
        return None
    return cage_chips.chip_hit(found[1], mx, my)


def cage_chip_contains(region, mx: float, my: float) -> bool:
    """True inside the chip row's box, gaps included (a press there is the row's)."""
    found = state.cage.chips if state.cage is not None else None
    if found is None or found[0] != region.as_pointer():
        return False
    return cage_chips.chip_contains(found[1], mx, my)


def _square_tris(centres: np.ndarray, side: float) -> np.ndarray:
    """Two triangles per square, (6 N, 2) float32."""
    h = side / 2.0
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 1, 2)
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, -h], [h, h], [-h, h]])
    return (c + corners[None]).reshape(-1, 2).astype(np.float32)


def _draw_cage_lattice(region, scale: float) -> None:
    """The cage's lines and points, in region pixels over everything, and the chip row.
    Writes `state.cage.layout` and `.chips` for the hit tests."""
    cage_state = state.cage
    if cage_state is None:
        return
    view_state = CONTEXT.view_state
    model = voxel_object.document.model
    if view_state is None or model is None:
        cage_state.layout = cage_state.chips = None
        return
    lat = cage_state.live or cage_state.lattice
    origin = np.asarray(voxel_object.model_origin(model), dtype=np.float64)
    vp = view_state.screen_matrix
    width, height = int(region.width), int(region.height)
    # The lines: built once per set of points, projected every frame.
    if cage_state.lines is None or cage_state.lines[0] is not lat:
        polys = lattice.edge_polylines(lat, 8)
        pairs = np.concatenate([np.stack([p[:-1], p[1:]], axis=1).reshape(-1, 3) for p in polys])
        cage_state.lines = (lat, pairs)
    seg = gizmo.project(vp, cage_state.lines[1] + origin, width, height).reshape(-1, 2, 2)
    ok = np.isfinite(seg).all(axis=(1, 2))
    r, g, b, _a = EMBER
    if ok.any():
        pen.lines(seg[ok].reshape(-1, 2).astype(np.float32), (r, g, b, CAGE_LINE_ALPHA), 1.5 * scale)
    # Mirror's copies on the other side of a plane: their lines only, fainter; they have no points to drag
    shape = cage_state.rest_voxels.shape
    mirror = tuple(bool(v) for v in getattr(CONTEXT.prefs, "rok_mirror", (False, False, False)))
    for axes in cage_drag.image_axes(cage_drag.mirror_copies(lat, mirror, shape)):
        pairs = cage_state.lines[1].copy()
        for a in axes:
            pairs[:, a] = shape[a] - pairs[:, a]
        seg = gizmo.project(vp, pairs + origin, width, height).reshape(-1, 2, 2)
        ok = np.isfinite(seg).all(axis=(1, 2))
        if ok.any():
            pen.lines(seg[ok].reshape(-1, 2).astype(np.float32), (r, g, b, CAGE_MIRROR_LINE_ALPHA), 1.0 * scale)
    # The points.
    world = lat.points.reshape(-1, 3) + origin
    pts = gizmo.project(vp, world, width, height)
    eye = np.asarray(view_state.camera.eye(), dtype=np.float64)
    depth = (world - eye) @ np.asarray(view_state.view_direction(), dtype=np.float64)
    ids = lattice.point_ids(lat.counts)
    cage_state.layout = (region.as_pointer(), ids, pts, depth)
    edge = lattice.edge_points(lat.counts).reshape(-1)
    hot = state.hover_handle if lattice.parse_point_id(state.hover_handle) is not None else None
    hot_i = ids.index(hot) if hot in ids else -1
    shown = np.isfinite(pts).all(axis=1)
    hot_mask = np.zeros(len(ids), dtype=bool)
    if hot_i >= 0:
        hot_mask[hot_i] = True
    # Far points first, so a near one is on top of an overlapping one.
    order = np.argsort(-np.where(shown, depth, -np.inf))
    for sel, side, colour in (
        (edge & ~hot_mask, CAGE_EDGE_POINT_PX, (r, g, b, 1.0)),
        (~edge & ~hot_mask, CAGE_FACE_POINT_PX, (r, g, b, CAGE_FACE_POINT_ALPHA)),
        (hot_mask, CAGE_EDGE_POINT_PX + 2.0, FRAME_WHITE),
    ):
        pick = [i for i in order if sel[i] and shown[i]]
        if pick:
            pen.tris(_square_tris(pts[pick], side * scale), colour)
    _draw_cage_chips(region, scale, pts[shown], cage_state, lat)


def _draw_cage_chips(region, scale: float, projected: np.ndarray, cage_state, lat) -> None:
    from .state import cage_screen_for

    if not len(projected):
        cage_state.chips = None
        return
    aabb = (float(projected[:, 0].min()), float(projected[:, 1].min()),
            float(projected[:, 0].max()), float(projected[:, 1].max()))
    screen = cage_screen_for(region)
    avoid = screen.strip.rect if screen is not None and screen.strip is not None else None
    layout = cage_chips.chip_layout(aabb, region.width, region.height, scale, avoid=avoid)
    if layout is None:
        cage_state.chips = None
        return
    cage_state.chips = (region.as_pointer(), layout)
    hover = state.hover_pill
    radius = round(8.0 * float(scale))
    font = 0
    fonts.size(font, 12 * scale)
    for name, (x, y, w, h) in layout.chips:
        on = hover == name
        lock = cage_chips.parse_lock(name)
        is_lit = (name == cage_chips.SOFT and lat.bernstein) or (lock is not None and cage_state.locks[lock])
        pen.fan(hud.rounded_rect(x, y, w, h, radius), CAGE_GLYPH_HOVER if on else CAGE_GLYPH_BG)
        if is_lit:
            pen.fan(hud.rounded_rect(x + 2 * scale, y + 2 * scale, w - 4 * scale, h - 4 * scale, radius - 2 * scale), EMBER)
        if on:
            _draw_rect_outline((x, y, w, h), EMBER, 0.0, 1.5 * scale)
        text = cage_chips.CHIP_TITLES[name]
        tw, th = fonts.dimensions(font, text)
        fonts.color(font, *(CAGE_CHIP_LIT_INK if is_lit else (GLYPH_HOVER_INK if on else GLYPH_INK)))
        fonts.position(font, float(round(x + (w - tw) / 2.0)), float(round(y + (h - th) / 2.0)), 0)
        fonts.draw(font, text)


EXTRA_KEY = 256  # strip keys from here on: colours that hold no slot (key - EXTRA_KEY: place in _strip_extras)
_strip_extras: list[int] = []  # the colour ids of those swatches


def strip_entries() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The strip's swatch keys: the colours voxels use, then the unused ones,
    each in your order (the colour store's). A colour on a slot is keyed by
    its slot; one without a slot by EXTRA_KEY + its place in _strip_extras.
    A favourite colour shows in the favourites row only."""
    global _strip_extras
    document = voxel_object.document
    if document.scene is None:
        _strip_extras = []
        return (), ()
    store = document.scene.colours
    counts = document.used_counts()
    store.adopt_used(counts)  # a colour painted before the store knew it
    favs = set(_favourites())
    extras: list[int] = []

    def key(ident: int) -> int:
        slot = store.slot_of(ident)
        if slot is not None:
            return slot
        extras.append(ident)
        return EXTRA_KEY + len(extras) - 1

    def shown(ident: int) -> bool:
        return store.entries[ident].rgba[:3] not in favs

    used = tuple(store.slot_of(i) for i in store.used_ids(counts) if shown(i))
    unused = tuple(key(i) for i in store.unused_ids(counts) if shown(i))
    _strip_extras = extras
    return used, unused


def entry_of_key(key: int) -> int | None:
    """The colour id behind a strip swatch key."""
    if key >= EXTRA_KEY:
        k = key - EXTRA_KEY
        return _strip_extras[k] if k < len(_strip_extras) else None
    scene = voxel_object.document.scene
    return None if scene is None else scene.colours.id_of(key)


def strip_palette(palette: np.ndarray) -> np.ndarray:
    """The palette with a row for every strip colour that holds no slot, so
    a swatch key indexes its colour."""
    scene = voxel_object.document.scene
    if not _strip_extras or scene is None:
        return palette
    rows = [scene.colours.entries[i].rgba for i in _strip_extras]
    return np.concatenate([np.asarray(palette), np.asarray(rows, dtype=np.uint8)])


def strip_type_codes(codes: np.ndarray) -> np.ndarray:
    """The material codes, with a code for every strip colour that holds no slot."""
    scene = voxel_object.document.scene
    if not _strip_extras or scene is None:
        return codes
    from rok.core.material import MaterialTable

    extra = []
    for ident in _strip_extras:
        table = MaterialTable()
        table.set(1, scene.colours.entries[ident].material)
        extra.append(table.type_codes()[1])
    return np.concatenate([np.asarray(codes), np.asarray(extra, dtype=np.uint8)])


def strip_layout(region) -> StripLayout:
    """Strip layout for a region, cached by the colours, the counts and the size.

    The region is unused: the strip sits at the bottom left whatever the size,
    inset past the left rail so the plate does not cover the swatches.
    """
    global _strip_layout_cache
    document = voxel_object.document
    show_grid = palette_grid_shown()
    scale = ui_scale()
    rail = _palette_rail_width()
    favs = tuple(_favourites())
    palette_key = document.scene.palette.tobytes() if document.scene is not None else b""
    version = document.scene.colours.version if document.scene is not None else 0
    key = (document.stroke_serial, show_grid, scale, rail, favs, palette_key, version, _dock_key(region),
           int(document.colour))
    if _strip_layout_cache is None or _strip_layout_cache[0] != key:
        used, unused = strip_entries()
        layout = _dock_strip(region, scale, used, unused, len(favs), show_grid, rail)
        _strip_layout_cache = (key, layout, list(_strip_extras))
    _strip_extras[:] = _strip_layout_cache[2]
    return _strip_layout_cache[1]


# --- palette v2: the stacked dock card ------------------------------------------------
# The card's geometry comes from core.palette_dock; it is turned into a
# hud.StripLayout so the swatch hit tests, the drags (Ctrl swap, Shift
# gradient) and the material dots keep working unchanged. The headings, the
# chevron and the scroll clip live in `_dock`, the DockLayout last built.
_dock = None
_FLIGHTS = palette_anim.Flights()  # colours flying between In use and Unused
_TW = palette_anim.Tweens()  # the card's height, chevrons and chip colour
_HOVER = palette_anim.Hover()  # how long the mouse rests on a swatch
_dock_dy = 0.0  # shown card height - final height, while the card animates
_last_editor_h = [0]
_fly_back = None  # (key, (x, y) released at, start time) of a cancelled drag's ghost
_swap_armed = [False]  # the next layout change moves swatches over SWAP seconds
_dock_seen = [None]  # the current colour the card last revealed
_dock_codes = None  # the material codes as last drawn, for the dot tip
MATERIAL_DOT_NAMES = {1: "Metal", 2: "Glass", 3: "Light", 4: "Plastic", 5: "Other"}
_dock_scroll: dict[str, int] = {}  # section name -> px scrolled, when that section is taller than the card allows
DOCK_BUTTON_NAMES = {"NEW": "NEW", "EYEDROP": "EYEDROP", "FAVOURITE": "FAVOURITE"}


def _dock_key(region):
    palette_ui = _mod("palette_ui")
    if palette_ui is None:
        return None
    editor = _dock_editor_height()
    recent = tuple(palette_ui.recent_colours(8)) if palette_ui.dock_collapsed() else ()
    return (palette_ui.dock_collapsed(), palette_dock.fold_text(palette_ui.dock_folds()), recent,
            int(getattr(region, "height", 0)), tuple(sorted(_dock_scroll.items())), editor)


def _dock_editor_height() -> int:
    palette_ui = _mod("palette_ui")
    return int(palette_ui.docked_editor_height()) if hasattr(palette_ui, "docked_editor_height") else 0


def dock_layout(region=None):
    """The DockLayout of the palette card as last laid out (None before the first)."""
    if region is not None:
        strip_layout(region)
    return _dock


def _dock_strip(region, scale, used, unused, favs, show_grid, rail) -> StripLayout:
    global _dock
    palette_ui = _mod("palette_ui")
    collapsed = palette_ui.dock_collapsed() if palette_ui is not None else False
    folds = palette_ui.dock_folds() if palette_ui is not None else {}
    recent = palette_ui.recent_colours(8) if palette_ui is not None else []
    x0 = hud.palette_x0(scale, rail + STRIP_SHIFT)
    y0 = round((hud.STRIP_MARGIN + STRIP_SHIFT) * scale)
    vh = float(getattr(region, "height", 1000) or 1000)
    used = list(dict.fromkeys(int(i) for i in used if 0 < int(i)))
    unused = list(unused)  # no count cap: a long list scrolls inside its section
    colour = int(voxel_object.document.colour)
    scene = voxel_object.document.scene
    rgb = tuple(int(c) for c in scene.palette[colour, :3]) if scene is not None else None
    if colour not in used and colour not in unused and rgb not in set(_favourites()):
        unused.insert(0, colour)  # the current colour is always on the card, ringed; a favourite is in its row
    dock = palette_dock.dock_layout(vh, used, unused, favs, x0=x0, y0=y0, scale=scale, folds=folds,
                                    collapsed=collapsed, recent=recent, scroll_offsets=dict(_dock_scroll),
                                    editor_height=_dock_editor_height(),
                                    reveal=colour if colour != _dock_seen[0] else None)
    _dock_seen[0] = colour  # a new current colour scrolls into view once; the wheel can scroll away
    _dock = dock
    _dock_scale[0] = scale
    for sec in dock.sections:
        if sec.scroll:
            _dock_scroll[sec.name] = sec.offset
        else:
            _dock_scroll.pop(sec.name, None)
    return dock_to_strip(dock, scale)


def dock_to_strip(dock, scale: float) -> StripLayout:
    """A hud.StripLayout with the dock card's rects: no captions, no dividers
    (the headings draw their own), the 256 grid above the card."""
    sec = {s.name: s for s in dock.sections}
    pick = lambda n: tuple(sec[n].swatches) if n in sec else ()  # noqa: E731
    used = pick("USED") if not dock.collapsed else tuple(dock.recent)
    hits = lambda n: tuple(sec[n].hits) if n in sec else ()  # noqa: E731
    cells = (tuple(("SWATCH", k, r) for k, r in dock.recent_hits) if dock.collapsed
             else tuple(("SWATCH", k, r) for k, r in hits("USED") + hits("UNUSED"))
             + tuple(("FAV", k, r) for k, r in hits("FAV")))
    x, y, w, h = dock.card
    return StripLayout(
        x0=x, y0=y, swatch=dock.swatch, current=dock.chip,
        buttons=tuple((DOCK_BUTTON_NAMES[n], r) for n, r in dock.buttons),
        used=used, width=w, height=h, grid_y0=y + h + round(hud.STRIP_TEXT_BAND * scale),
        made=pick("UNUSED"), divider=None, favourites=pick("FAV"), fav_divider=None, captions=(),
        cells=cells,
    )


_dock_scale = [1.0]  # the UI scale the card was last laid out at


def _colour_count(colour: int) -> int:
    """How many voxels use palette index `colour` in the current scene."""
    try:
        return int(voxel_object.document.used_counts()[colour]) if colour < 256 else 0
    except Exception:
        return 0


def _header_count(dock) -> tuple[str, str]:
    """(shown, exact) voxel count text of the current colour: the short form shows
    when the exact one does not fit the count rect."""
    count = _colour_count(int(voxel_object.document.colour))
    cw, size = dock.count[2], 10 * _dock_scale[0]
    return palette_dock.count_text(count, lambda t: text_width(t, size) <= cw), palette_dock.count_text(count)


def _count_exact(dock) -> str | None:
    """The exact "12,345 voxels" while the header shows the short form, else None.
    Worked out from the scene and the count rect as they are now."""
    if dock is None or voxel_object.document.scene is None:
        return None
    shown, exact = _header_count(dock)
    return None if shown == exact else exact


def dock_hit(region, mx: float, my: float):
    """('HEADING', i) for a section heading, ('CHEVRON', 0) for the card's
    collapse chevron, ('COUNT', 0) for a shortened voxel count, or None. The swatches and buttons go through hud.strip_hit_test."""
    dock = dock_layout(region)
    if dock is None:
        return None
    inside = lambda r: r[0] <= mx < r[0] + r[2] and r[1] <= my < r[1] + r[3]  # noqa: E731
    if inside(dock.chevron):
        return ("CHEVRON", 0)
    if inside(dock.count) and _count_exact(dock) is not None:
        return ("COUNT", 0)
    for i, s in enumerate(dock.sections):
        if inside(s.heading):
            return ("HEADING", i)
    return None


def dock_click(region, item) -> bool:
    """Act on a heading or chevron click. True when it was one."""
    palette_ui = _mod("palette_ui")
    if item is None or palette_ui is None:
        return False
    if item[0] == "CHEVRON":
        if getattr(getattr(palette_ui, "_popup", None), "docked", False):
            palette_ui.close_popup()  # with the editor open the chevron closes it
        else:
            palette_ui.set_dock_collapsed(not palette_ui.dock_collapsed())
    elif item[0] == "HEADING" and _dock is not None and item[1] < len(_dock.sections):
        palette_ui.toggle_dock_fold(_dock.sections[item[1]].name)
    else:
        return False
    invalidate_palette()
    return True


def dock_scroll(region, mx: float, my: float, steps: float) -> bool:
    """Scroll the section under the pointer by wheel `steps`. True when that section scrolls."""
    dock = dock_layout(region)
    if dock is None:
        return False
    card = dock.card
    if not (card[0] <= mx < card[0] + card[2] and card[1] <= my < card[1] + card[3]):
        return False
    scrolling = [s for s in dock.sections if s.scroll and s.clip is not None]
    if not scrolling:
        return False

    def inside(r) -> bool:
        return r[0] <= mx < r[0] + r[2] and r[1] <= my < r[1] + r[3]

    target = next((s for s in scrolling if inside(s.clip)), None)
    if target is None:  # header, heading or a gap: the top section that scrolls
        target = scrolling[0]
    c = target.clip
    step = dock.swatch + dock.gap
    cur = int(_dock_scroll.get(target.name, target.offset))
    _dock_scroll[target.name] = max(0, min(target.content_height - c[3], cur - int(round(steps)) * step))
    invalidate_palette()
    return True


def dock_tip(hover) -> str | None:
    """The hover HUD tip for the card item under the mouse (no status-bar text)."""
    if hover is None:
        return None
    kind = hover[0]
    if kind == "SWATCH" and _dock_codes is not None and 0 <= hover[1] < len(_dock_codes):
        name = MATERIAL_DOT_NAMES.get(int(_dock_codes[hover[1]]))
        if name:  # the white dot: a material other than Plain
            return f"Dot: {name} material · " + palette_dock.HOVER_TIP
    if kind in ("SWATCH", "FAV"):
        return palette_dock.HOVER_TIP
    return {
        "CURRENT": "Click to edit the colour",
        "NEW": "New colour",
        "EYEDROP": "Pick a colour from the model",
        "FAVOURITE": "Favourite the current colour, or take it off the favourites",
        "CHEVRON": "Collapse or expand the palette",
        "HEADING": "Fold or unfold this section",
    }.get(kind)


def draw_dock(dock, layout: StripLayout, palette_linear, colour: int, hover, scale: float,
              type_codes=None) -> None:
    """The palette dock card: charcoal plate, current-colour header with hex,
    name and count, the three buttons and the chevron, then the stacked
    sections with their headings, rings on the current and hovered swatch."""
    global _dock_dy
    pal = np.asarray(palette_linear, dtype=np.float32)
    final_h = dock.card[3]
    eh = dock.editor[3] if dock.editor is not None else 0
    duration = palette_anim.EDITOR if eh != _last_editor_h[0] else palette_anim.FOLD
    _last_editor_h[0] = eh
    shown_h = _TW.value("card_h", float(final_h), duration)
    _dock_dy = shown_h - final_h
    if _dock_dy:
        dock = palette_dock.shifted(dock, _dock_dy)
        layout = _shifted(layout, 0, _dock_dy)
    card = dock.card
    radius = palette_dock.CARD_RADIUS * scale
    _draw_plate(card, radius, scale)
    hover_kind = hover[0] if hover is not None else None
    hover_index = hover[1] if hover is not None else None
    # Header: the chip, its hex + name + voxel count, the buttons, the chevron.
    cur_r = STRIP_CURRENT_RADIUS * scale
    chip_rgb = _TW.value("chip", tuple(float(c) for c in pal[colour][:3]), palette_anim.CHIP_FADE, eased=False)
    gfx.rrect(dock.chip, tuple(chip_rgb) + (1.0,), cur_r)
    gfx.rrect(dock.chip, palette_dock.SWATCH_OUTLINE, cur_r, stroke=1.0)
    if hover_kind == "CURRENT":
        gfx.rrect(_inset(dock.chip, -2.0 * scale), palette_dock.RING_HOVER, cur_r + 2.0 * scale, stroke=1.0 * scale)
    _draw_dock_label(dock, colour, scale)
    favs = _favourites()
    current_rgb = None
    if voxel_object.document.scene is not None:
        current_rgb = tuple(int(c) for c in voxel_object.document.scene.palette[colour, :3])
    palette_ui = _mod("palette_ui")
    star_on = palette_ui.STAR_ON if palette_ui is not None and current_rgb in set(favs) else None
    for name, rect in layout.buttons:
        _draw_button(name, rect, False, scale, hovered=hover_kind == name,
                     on_colour=star_on if name == "FAVOURITE" else None)
    _draw_chevron(dock.chevron, turn=_TW.value("chev:card", 1.0 if dock.collapsed else 0.0, palette_anim.FOLD),
                  scale=scale, hovered=hover_kind == "CHEVRON")
    gap = STRIP_SWATCH_GAP * scale
    sr = STRIP_SWATCH_RADIUS * scale
    fav_rgb = _palette_linear(np.array([(*c, 255) for c in favs], dtype=np.uint8)) if favs else None
    global _dock_codes
    _dock_codes = type_codes
    moving = _swap_rects(dock)
    anchors = {sec.name: sec.chevron for sec in dock.sections}
    drawn = {k: r for sec in dock.sections if sec.name != "FAV" for k, r in sec.swatches}
    _FLIGHTS.update({sec.name: sec.keys for sec in dock.sections if sec.name != "FAV"}, drawn, anchors)
    flying = _FLIGHTS.rects(drawn, anchors)
    moving.update(flying)
    # The sections live below the header: while the card moves they are cut to
    # the part of the card under the header, so nothing slides over it.
    header_bottom = dock.chip[1] - palette_dock.PAD * scale
    content = (card[0], card[1], card[2], max(0.0, header_bottom - card[1]))
    top_shown = card[1] + card[3]

    def swatch(sec_name, key, rect, alpha):
        if sec_name == "FAV":
            if fav_rgb is None or key >= len(favs):
                return
            rgb = tuple(fav_rgb[key][:3])
            is_current = favs[key] == current_rgb
            hovered = hover_kind == "FAV" and hover_index == key
            code = 0
        else:
            if key >= len(pal):
                return
            rgb = tuple(pal[key][:3])
            is_current = key == colour
            hovered = hover_kind == "SWATCH" and hover_index == key
            rect = moving.get(key, rect)
            code = int(type_codes[key]) if type_codes is not None and key < len(type_codes) else 0
        with gfx.faded(alpha):
            gfx.rrect(_inset(rect, gap), rgb + (1.0,), sr)
            gfx.rrect(_inset(rect, gap), palette_dock.SWATCH_OUTLINE, sr, stroke=1.0)
            _draw_rings(rect, is_current, hovered, sr, scale)
            if code:
                _draw_dot(rect, scale)

    with gfx.clipped(content):
        for i, sec in enumerate(dock.sections):
            hx, hy, hw, hh = sec.heading
            hovered_head = hover_kind == "HEADING" and hover_index == i
            fonts.size(0, palette_dock.HEADING_SIZE * scale)
            fonts.color(0, *(TEXT_COLOUR if hovered_head else palette_dock.HEADING_INK))
            th = fonts.dimensions(0, sec.title)[1]
            fonts.position(0, float(hx), float(hy + (hh - th) / 2.0), 0)
            fonts.draw(0, sec.title)
            count = str(sec.count)
            cw = fonts.dimensions(0, count)[0]
            fonts.color(0, *palette_dock.COUNT_INK)
            fonts.position(0, float(sec.chevron[0] - 4 * scale - cw), float(hy + (hh - th) / 2.0), 0)
            fonts.draw(0, count)
            _draw_chevron(sec.chevron, turn=_TW.value("chev:" + sec.name, 0.0 if sec.folded else 1.0, palette_anim.FOLD),
                          scale=scale, small=True)
            gfx.rrect(sec.divider, STRIP_DIVIDER, 0.0)
            # Content fades in and out over CONTENT_FADE as the section opens and folds.
            fade = _TW.value("fade:" + sec.name, 0.0 if sec.folded else 1.0, palette_anim.CONTENT_FADE, eased=False)
            clip = sec.clip if sec.clip is not None else content
            if sec.scroll and sec.clip is not None:
                # Room for the rings, which sit outside the swatch: left, right
                # and top only; the bottom edge stays cut by the fade.
                m = RING_CLIP_ROOM * scale
                clip = (clip[0] - m, clip[1], clip[2] + 2.0 * m, clip[3] + m)
            with gfx.clipped(clip):
                if not sec.folded:
                    _sec_cache[sec.name] = [(k, (r[0], r[1] - top_shown, r[2], r[3])) for k, r in sec.swatches]
                    for key, rect in sec.swatches:
                        swatch(sec.name, key, rect, fade)
                elif fade > 0.0:  # just folded: the old rows fade out where they were, cut by the card
                    for key, rel in _sec_cache.get(sec.name, ()):
                        swatch(sec.name, key, (rel[0], rel[1] + top_shown, rel[2], rel[3]), fade)
            if sec.scroll and sec.clip is not None:
                _draw_scroll_fades(sec, scale)
        for key, rect in flying.items():
            if key not in drawn:  # flying into or out of a folded section: drawn over the headings
                swatch("", key, rect, 1.0)
        for key, rect in dock.recent:
            gfx.rrect(_inset(rect, gap), tuple(pal[key][:3]) + (1.0,), sr)
            gfx.rrect(_inset(rect, gap), palette_dock.SWATCH_OUTLINE, sr, stroke=1.0)
            _draw_rings(rect, key == colour, hover_kind == "SWATCH" and hover_index == key, sr, scale)
    if type_codes is not None and 0 <= colour < len(type_codes) and type_codes[colour]:
        _draw_dot(dock.chip, scale)
    _draw_drag(dock, pal, hover, sr, scale)
    # The tip goes in the HUD line (ui._hint) after HOVER_TIP_DELAY, not in a box here.
    rest = _HOVER.update(hover if hover_kind in ("SWATCH", "FAV", "COUNT") else None)
    waiting = hover_kind in ("SWATCH", "FAV", "COUNT") and rest < palette_anim.HOVER_TIP_DELAY
    if (_TW.running() or waiting or moving or _FLIGHTS.running() or _fly_back is not None) and CONTEXT.app is not None:
        CONTEXT.app.host.tick(1.0 / 60.0)  # redraw while something moves or a tip is due


def dock_hover_tip() -> str | None:
    """The HUD-line tip for the swatch the mouse has rested on for HOVER_TIP_DELAY:
    its hex, its material and the gestures. None before that."""
    item = _HOVER.item
    if item is None or palette_anim.now() - _HOVER.since < palette_anim.HOVER_TIP_DELAY:
        return None
    if item[0] == "COUNT":
        return _count_exact(_dock)
    scene = voxel_object.document.scene
    parts = []
    if item[0] == "SWATCH" and scene is not None:
        rgba = strip_palette(scene.palette)[item[1]] if item[1] < len(strip_palette(scene.palette)) else None
        if rgba is not None:
            parts.append("#{:02X}{:02X}{:02X}".format(*(int(c) for c in rgba[:3])))
        code = int(_dock_codes[item[1]]) if _dock_codes is not None and item[1] < len(_dock_codes) else 0
        parts.append(MATERIAL_DOT_NAMES.get(code, "Plain"))
    elif item[0] == "FAV":
        favs = _favourites()
        if item[1] < len(favs):
            parts.append("#{:02X}{:02X}{:02X}".format(*favs[item[1]]))
        parts.append("Favourite")
    parts.append("Ctrl drag to move · Shift drag for a gradient")
    return " · ".join(parts)


def arm_swap_motion() -> None:
    """The next layout change (a Ctrl drag's move) slides the swatches over SWAP seconds."""
    _swap_armed[0] = True
    _swap_from.clear()
    if _dock is not None:
        for s in _dock.sections:
            if s.name != "FAV":
                _swap_from.update(dict(s.swatches))


_swap_from: dict = {}
_swap_flights: dict = {}  # key -> (from rect, start time)


def _swap_rects(dock) -> dict:
    """{key: rect} of swatches sliding to their new place after a swap."""
    t_now = palette_anim.now()
    if _swap_armed[0] and _swap_from:
        _swap_armed[0] = False
        for s in dock.sections:
            for k, r in s.swatches:
                if s.name != "FAV" and k in _swap_from and tuple(_swap_from[k]) != tuple(r):
                    _swap_flights[k] = (_swap_from[k], t_now)
        _swap_from.clear()
    out = {}
    target = {k: r for s in dock.sections if s.name != "FAV" for k, r in s.swatches}
    for k, (start, t0) in list(_swap_flights.items()):
        t = (t_now - t0) / palette_anim.SWAP
        if t >= 1.0 or k not in target:
            del _swap_flights[k]
            continue
        out[k] = palette_anim.lerp(tuple(start), tuple(target[k]), palette_anim.ease_out(t))
    return out


def start_fly_back(key: int, at) -> None:
    """A Ctrl drag released off any swatch: the ghost flies from `at` back home."""
    global _fly_back
    _fly_back = (int(key), (float(at[0]), float(at[1])), palette_anim.now())


def _swatch_rect_of(dock, key):
    for s in dock.sections:
        if s.name != "FAV":
            for k, r in s.swatches:
                if k == key:
                    return r
    return None


def insertion_line(dock, start: int, target: int, scale: float = 1.0):
    """The 2 px ember line, a swatch high, between the two swatches where a
    moved colour lands: after the target when it moves forward, before it when
    it moves back."""
    for s in dock.sections:
        if s.name == "FAV":
            continue
        keys = [k for k, _r in s.swatches]
        if target not in keys:
            continue
        r = dict(s.swatches)[target]
        forward = start in keys and keys.index(start) < keys.index(target)
        g = palette_dock.GAP * scale
        x = r[0] + r[2] + g / 2.0 - 1.0 * scale if forward else r[0] - g / 2.0 - 1.0 * scale
        return (x, r[1], 2.0 * scale, r[3])
    return None


def gradient_preview():
    """[(r, g, b)] 0..255 of a running Shift drag over a swatch, or None."""
    palette_ui = _mod("palette_ui")
    drag = getattr(palette_ui, "_drag", None) if palette_ui is not None else None
    hover = state.hover_strip
    if drag is None or drag[0] != "GRADIENT" or hover is None or hover[0] != "SWATCH" or hover[1] == drag[1]:
        return None
    found = palette_ui._entries_between(drag[1], hover[1])
    if found is None:
        return None
    _ia, _ib, span = found
    store = voxel_object.document.scene.colours
    first, last = store.entries[span[0]].rgba[:3], store.entries[span[-1]].rgba[:3]
    return palette_dock.gradient_preview(first, last, len(span) - 1)


def _draw_drag(dock, pal, hover, sr: float, scale: float) -> None:
    """The Ctrl-drag ghost at the mouse and the ember outline on its target; the
    Shift drag's start and end marks; a cancelled drag's ghost flying home."""
    global _fly_back
    palette_ui = _mod("palette_ui")
    drag = getattr(palette_ui, "_drag", None) if palette_ui is not None else None
    region = CONTEXT.region
    size = dock.swatch
    if drag is not None:
        kind, start = drag
        target = hover[1] if hover is not None and hover[0] == "SWATCH" and hover[1] != start else None
        home = _swatch_rect_of(dock, start)
        if kind == "GRADIENT" and home is not None:
            gfx.rrect(_inset(home, -2.0 * scale), palette_dock.RING_EMBER, sr + 2.0 * scale, stroke=2.0 * scale)
        if target is not None:
            r = _swatch_rect_of(dock, target)
            if r is not None and kind == "GRADIENT":
                gfx.rrect(_inset(r, -2.0 * scale), palette_dock.RING_EMBER, sr + 2.0 * scale, stroke=2.0 * scale)
            elif r is not None:
                line = insertion_line(dock, start, target, scale)
                if line is not None:
                    gfx.rrect(line, palette_dock.RING_EMBER, 1.0 * scale)
        if kind == "SWAP" and region is not None and 0 <= start < len(pal):
            mx = state.last_mouse[0] - region.x
            my = state.last_mouse[1] - region.y
            ghost = (mx - size / 2.0, my - size / 2.0, size, size)
            gfx.rrect(ghost, tuple(pal[start][:3]) + (0.5,), sr)
    if _fly_back is not None:
        key, (fx, fy), t0 = _fly_back
        t = (palette_anim.now() - t0) / palette_anim.FLY_BACK
        home = _swatch_rect_of(dock, key)
        if t >= 1.0 or home is None or key >= len(pal):
            _fly_back = None
            return
        start = (fx - size / 2.0, fy - size / 2.0, size, size)
        r = palette_anim.lerp(start, tuple(home), palette_anim.ease_out(t))
        gfx.rrect(r, tuple(pal[key][:3]) + (0.5,), sr)


def _draw_scroll_fades(sec, scale: float) -> None:
    """8 px fades at the top and bottom of a scrolling grid where more rows hide."""
    x, y, w, h = sec.clip
    n = max(1, int(round(8 * scale)))
    bg = palette_dock.CARD_BG
    more_above = sec.offset > 0
    more_below = sec.offset + h < sec.content_height
    for i in range(n):
        a = bg[3] * (1.0 - i / n) * 0.9
        if more_above:
            gfx.rrect((x - 2, y + h - 1 - i, w + 4, 1), bg[:3] + (a,), 0.0)
        if more_below:
            gfx.rrect((x - 2, y + i, w + 4, 1), bg[:3] + (a,), 0.0)


_sec_cache: dict = {}  # each section's swatches when last open, relative to the card's top


def _draw_plate(card, radius: float, scale: float) -> None:
    """The card's plate: a soft drop shadow (0/4 px, blur 12, 35 % black), the
    charcoal fill, a 1 px edge and a 1 px highlight along the top at 6 % white."""
    x, y, w, h = card
    blur, steps = 12.0 * scale, 6
    for k in range(steps, 0, -1):
        grow = blur * k / steps / 2.0
        a = palette_dock.SHADOW_ALPHA / steps
        gfx.rrect((x - grow, y - 4.0 * scale - grow, w + 2 * grow, h + 2 * grow), (0.0, 0.0, 0.0, a), radius + grow)
    gfx.rrect(card, palette_dock.CARD_BG, radius)
    gfx.rrect(card, STRIP_PLATE_EDGE, radius, stroke=1.0)
    gfx.rrect((x + radius, y + h - 1.0 - 1.0 * scale, w - 2 * radius, 1.0 * scale), palette_dock.TOP_HIGHLIGHT, 0.0)


def _draw_dot(rect, scale: float) -> None:
    """A material dot: 4 px, 2 px in from the top-right corner, a 1 px dark outline."""
    x, y, w, h = rect
    side, inset = palette_dock.DOT * scale, palette_dock.DOT_INSET * scale
    dot = (x + w - inset - side, y + h - inset - side, side, side)
    ring = _inset(dot, -1.0 * scale)
    gfx.rrect(ring, FRAME_BLACK, ring[2] / 2.0)
    gfx.rrect(dot, FRAME_WHITE, side / 2.0)


def _draw_rings(rect, current: bool, hovered: bool, radius: float, scale: float) -> None:
    """Current: 2 px ember outside a 1 px dark ring, so it reads on light and dark
    swatches. Hover: a soft white ring."""
    if current:
        gfx.rrect(_inset(rect, -1.0 * scale), palette_dock.RING_EMBER, radius + 1.0 * scale,
                  stroke=palette_dock.RING_EMBER_W * scale)
        gfx.rrect(_inset(rect, 1.0 * scale), palette_dock.RING_INNER, max(0.0, radius - 1.0 * scale),
                  stroke=palette_dock.RING_INNER_W * scale)
    elif hovered:
        gfx.rrect(_inset(rect, -1.0 * scale), palette_dock.RING_HOVER, radius + 1.0 * scale, stroke=1.0 * scale)


def _draw_dock_label(dock, colour: int, scale: float) -> None:
    """Hex and name of the current colour next to the chip, its voxel count below."""
    document = voxel_object.document
    scene = document.scene
    if scene is None:
        return
    r, g, b = (int(c) for c in scene.palette[colour, :3])
    hex_text = f"#{r:02X}{g:02X}{b:02X}"
    palette_ui = _mod("palette_ui")
    name = ""
    if palette_ui is not None:
        name = palette_ui.favourite_name((r, g, b)) or ""
    x, y, w, h = dock.label
    line = hex_text + (f"  {name}" if name else "")
    size = 12.0 * scale
    fonts.size(0, size)
    while fonts.dimensions(0, line)[0] > w and size > 10.0 * scale:
        size -= 0.5 * scale
        fonts.size(0, size)
    if fonts.dimensions(0, line)[0] > w:  # only a long name is cut; the hex always fits the line
        line = clip_text(line, w, size)
    editing = palette_ui is not None and getattr(getattr(palette_ui, "_popup", None), "docked", False)
    if not editing:  # with the editor open the editor draws the hex field on line 1
        fonts.size(0, size)
        fonts.color(0, *TEXT_COLOUR)
        th = fonts.dimensions(0, "#")[1]
        fonts.position(0, float(x), float(y + (h - th) / 2.0), 0)
        fonts.draw(0, line)
    # Line 2: the count, left, level with the buttons.
    # It has its own rect, which ends before the first button. The exact count shows when
    # it fits; otherwise the short form, with the exact number in the hover tip.
    cx, cy, cw, ch = dock.count
    fonts.size(0, 10 * scale)
    fonts.color(0, *palette_dock.HEADING_COLOUR)
    th = fonts.dimensions(0, "0")[1]
    fonts.position(0, float(cx), float(cy + (ch - th) / 2.0), 0)
    shown = palette_dock.count_text(_colour_count(colour), lambda t: text_width(t, 10 * scale) <= cw)
    fonts.draw(0, clip_text(shown, cw, 10 * scale))


def _draw_chevron(rect, turn: float = 0.0, scale: float = 1.0, hovered: bool = False, small: bool = False,
                  up: bool | None = None) -> None:
    """A chevron pointing down at `turn` 0 and up at 1, rotated in between."""
    import math

    if up is not None:
        turn = 1.0 if up else 0.0
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    d = (3.0 if small else 4.5) * scale
    a = math.pi * float(turn)
    ca, sa = math.cos(a), math.sin(a)
    pts = [(-d, d / 2), (0.0, -d / 2), (d, d / 2)]  # pointing down (y up)
    p = [(cx + px_ * ca - py_ * sa, cy + px_ * sa + py_ * ca) for px_, py_ in pts]
    ink = GLYPH_HOVER_INK if hovered else GLYPH_INK
    pen.lines(np.array([p[0], p[1], p[1], p[2]], dtype=np.float32), ink, 1.5 * scale)


def _caption_length(text: str) -> float:
    """A strip caption's length in points, measured by the host when there is one."""
    if fonts.measure is None:
        return hud._caption_estimate(text)
    return float(fonts.measure(text, hud.STRIP_CAPTION_SIZE)[0])


def _draw_favourite_tip(context) -> None:
    """A favourite's name above its swatch in the strip, at once. Its own layer above the strip: text draws over the plates of
    its layer, so in the strip's layer the file label showed through it."""
    hover = state.hover_strip
    region = context.region
    if hover is None or hover[0] != "FAV" or region is None or not is_main_region(region):
        return
    palette_ui = _mod("palette_ui")
    favs = _favourites()
    layout = strip_layout(region)
    k = hover[1]
    rect = dict(layout.favourites).get(k)
    if palette_ui is None or rect is None or k >= len(favs):
        return
    scale = ui_scale()
    name = palette_ui.favourite_name(favs[k])
    text = name if name else "#{:02x}{:02x}{:02x} · right-click to name".format(*favs[k])
    # Above the strip's plate, where nothing lies under it, over the swatch.
    fonts.size(0, 11 * scale)
    tw, th = fonts.dimensions(0, text)
    pad = 6 * scale
    plate = strip_plate(layout, scale)
    w, h = tw + 2 * pad, th + 2 * pad
    x = max(4 * scale, rect[0] + rect[2] / 2.0 - w / 2.0)
    y = plate[1] + plate[3] + 6 * scale
    gfx.rrect((x, y, w, h), TIP_BG, 6 * scale)
    gfx.rrect((x, y, w, h), STRIP_PLATE_EDGE, 6 * scale, stroke=1.0)
    fonts.color(0, *TEXT_COLOUR)
    fonts.position(0, float(x + pad), float(y + pad), 0)
    fonts.draw(0, text)


def _favourites() -> list:
    palette_ui = _mod("palette_ui")
    return palette_ui.favourites() if palette_ui is not None else []


def _raised(layout: StripLayout, dy: int) -> StripLayout:
    """The layout moved up by `dy`: the strip layout puts it 8 px from the edge;
    the rounded plate needs room around it. Drawing and hit tests share this."""
    import dataclasses

    up = lambda r: (r[0], r[1] + dy, r[2], r[3])  # noqa: E731
    return dataclasses.replace(
        layout,
        y0=layout.y0 + dy,
        current=up(layout.current),
        buttons=tuple((n, up(r)) for n, r in layout.buttons),
        used=tuple((i, up(r)) for i, r in layout.used),
        made=tuple((i, up(r)) for i, r in layout.made),
        grid_y0=layout.grid_y0 + dy,
        divider=None if layout.divider is None else layout.divider + dy,
        favourites=tuple((k, up(r)) for k, r in layout.favourites),
        fav_divider=None if layout.fav_divider is None else layout.fav_divider + dy,
        captions=tuple((t, up(r)) for t, r in layout.captions),
        cells=tuple((kind, k, up(r)) for kind, k, r in layout.cells),
    )


def _shifted(layout: StripLayout, dx: float, dy: float) -> StripLayout:
    """The layout moved by (dx, dy), for drawing only (the star's play shakes it)."""
    import dataclasses

    move = lambda r: (r[0] + dx, r[1] + dy, r[2], r[3])  # noqa: E731
    return dataclasses.replace(
        _raised(layout, 0),
        x0=layout.x0 + dx,
        y0=layout.y0 + dy,
        current=move(layout.current),
        buttons=tuple((n, move(r)) for n, r in layout.buttons),
        used=tuple((i, move(r)) for i, r in layout.used),
        made=tuple((i, move(r)) for i, r in layout.made),
        grid_y0=layout.grid_y0 + dy,
        divider=None if layout.divider is None else layout.divider + dy,
        favourites=tuple((k, move(r)) for k, r in layout.favourites),
        fav_divider=None if layout.fav_divider is None else layout.fav_divider + dy,
        captions=tuple((t, move(r)) for t, r in layout.captions),
        cells=tuple((kind, k, move(r)) for kind, k, r in layout.cells),
    )


def _clone_colours(hover, cells):
    """[(rgb 0..1, cells)] of a Clone ghost: each cell in the colour the copied
    voxel there has. None when the stamp is not Attach's or has no colours."""
    pattern_ops = _mod("pattern_ops")
    if pattern_ops is None or hover.tool != ATTACH or cells is None or len(cells) == 0:
        return None
    found = pattern_ops.hover_colours(cells)
    if found is None:
        return None
    return [(tuple(float(c) / 255.0 for c in rgb), group) for rgb, group in found]


# The strip as last drawn ({index: (section, rect)}), and the swatches in flight
# ({index: (from rect, start time)}): a colour that becomes used or unused flies
# from one section to the other.
_swatch_last: dict = {}
_swatch_flights: dict = {}


def _swatch_rects(layout: StripLayout) -> dict:
    """{index: rect} for the swatches in flight now; starts new flights."""
    now = time.monotonic()
    for index, start in hud.swatch_moves(_swatch_last, layout).items():
        _swatch_flights[index] = (start, now)
    _swatch_last.clear()
    for section, items in (("USED", layout.used), ("MADE", layout.made)):
        for index, rect in items:
            _swatch_last[index] = (section, rect)
    out = {}
    targets = dict(layout.swatches)
    for index, (start, t0) in list(_swatch_flights.items()):
        t = (now - t0) / hud.SWATCH_FLY_S
        if t >= 1.0 or index not in targets:
            del _swatch_flights[index]
            continue
        out[index] = hud.flying_rect(start, targets[index], t)
    if _swatch_flights and CONTEXT.app is not None:
        CONTEXT.app.host.tick(1.0 / 60.0)  # redraw while a swatch flies
    return out


def palette_layout(region) -> PaletteLayout:
    """Layout of the on-demand 256 grid: above the strip and the status text."""
    scale = ui_scale()
    strip = strip_layout(region)
    plate = strip_plate(strip, scale)
    # Above the strip's plate, with room for the grid's own plate between them.
    y0 = max(strip.grid_y0, round(plate[1] + plate[3] + (STRIP_PLATE_PAD + 8) * scale))
    return hud.make_layout(scale, y0, rail_width=_palette_rail_width() + STRIP_SHIFT)


def _palette_linear(palette: np.ndarray) -> np.ndarray:
    """(256, 4) float colours of the palette for the swatches (sRGB in Rok)."""
    if PALETTE_LINEARIZE:
        return hud.srgb_to_linear(palette)
    return np.asarray(palette, dtype=np.float32) / 255.0


# --- callbacks (read the context) -----------------------------------------------------


def _draw_view(context) -> None:
    if _disabled:
        return
    try:
        document = voxel_object.document
        model = document.model
        if model is None:
            return
        region = context.region
        view_state = context.view_state
        if region is None or view_state is None:
            return
        grid = model.grid
        hover = state.hover
        if hover is not None and _view_moved(view_state):
            # The view changed since the pick (orbit, zoom, a view key). The old
            # highlight would sit on the wrong pixels: pick again first.
            _schedule_repick()
            hover = state.hover if not _view_moved(view_state) else None
        preview = state.preview
        preview_colour = None
        if preview is not None:
            preview_colour = _colour_of(preview.tool, grid, getattr(preview, "subtract", False))
        wm = context.prefs
        mirror = getattr(wm, "rok_mirror", (False, False, False))
        mirror = tuple(bool(v) for v in mirror)
        in_scene = state.level == "SCENE"
        whole = in_scene  # whole models: the scene
        if state.level == "BRICK" or (grid is not None and grid.bricks is not None):
            hover = preview = None  # Brick mode draws its own ghost; a brick model takes no voxel tool
        hovered_uid = state.hover_model if (hover is not None or whole) else None
        if whole:
            # The scene shows whole models: no voxel hover, preview, selection or mirror.
            hover = preview = None
            mirror = (False, False, False)
        boxes = []
        group = set()
        if in_scene:
            scene_tools = _mod("scene_tools")
            if scene_tools is not None:
                group = set(scene_tools.group_uids())
        # Animate mode draws the posed box itself (below), drag included.
        drag = state.model_drag_offset if state.stroke_active and not state.animating else (0, 0, 0)
        for m in document.scene.models:
            if m.uid == model.uid or (in_scene and document.scene.is_hidden(m)):
                continue
            origin_m = voxel_object.model_origin(m)
            if m.uid in group:
                # A member of the picked group: an ember box that moves with a drag.
                # A live move shows it through its preview matrix; adding the
                # offset as well moved the box twice as fast.
                kind = "PICKED"
                if m.uid not in state.turn_preview:
                    origin_m = tuple(o + d for o, d in zip(origin_m, drag))
            elif m.uid in state.snap_uids:
                kind = "SNAP"  # a drag snapped to it: a bright outline while it holds
            else:
                kind = "HOVER" if m.uid == hovered_uid else "OTHER"
            size_m, matrix_m = m.grid.size, state.turn_preview.get(m.uid)
            posed = _late_anim_box(m)
            if posed is not None:  # Animate mode: the box goes round the posed part
                delta, size_m, matrix_m = posed
                origin_m = tuple(o + d for o, d in zip(origin_m, delta))
            boxes.append((origin_m, size_m, kind, matrix_m))
        radius = 1
        if hover is not None and hover.mode == VOXEL:
            radius = int(getattr(wm, "rok_brush_size", 1))
        ghost = state.gizmo_ghost[1] if state.gizmo_ghost is not None else None
        origin = voxel_object.model_origin(model)
        box_size = grid.size
        picked_matrix = state.turn_preview.get(model.uid)
        posed = _late_anim_box(model)
        if posed is not None:  # Animate mode: the picked part's box at its pose
            delta, box_size, picked_matrix = posed
            origin = tuple(o + d for o, d in zip(origin, delta))
        stretching = state.gizmo_ghost is not None and state.gizmo_ghost[0][0] == "STRETCH"
        if stretching:
            # A stretch drag: the box shows the size and place it will have.
            box_size = state.gizmo_ghost[0][1][0]
            origin = state.gizmo_ghost[0][2]
        tool_name = getattr(wm, "rok_tool", "")
        selection = document.selection
        use_cage = (
            not whole
            and tool_name == SELECT
            and selection is not None
            and not selection.is_empty()
        )
        cage_bounds = selection.bounds() if use_cage else None
        gizmo_state = state.gizmo if not use_cage else None
        hide_selection = _shader_panel_open()
        if hide_selection:
            cage_bounds = gizmo_state = None
        offset = state.model_drag_offset
        if any(offset) and state.stroke_active:
            # A MODEL move drag: the box, the mirror planes and the gizmo follow
            # the offset. model.position changes on release. The model shows at
            # its new place through state.turn_preview, which the box is drawn
            # with too: then only the gizmo, drawn outside it, takes the offset.
            if posed is not None:
                # Animate mode: the box above is the posed part and already has the
                # drag in it (anim_mode.shown_at). Adding the offset again put the
                # box and the gizmo twice as far as the part. The
                # gizmo stands where it did at the press, so it takes the offset.
                if gizmo_state is not None:
                    g = gizmo_state.origin
                    gizmo_state = replace(gizmo_state, origin=(g[0] + offset[0], g[1] + offset[1], g[2] + offset[2]))
            else:
                moved = (origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2])
                if model.uid not in state.turn_preview:
                    origin = moved
                if gizmo_state is not None:
                    gizmo_state = replace(gizmo_state, origin=moved)

        draw_scene(
            box_size,
            origin,
            hover,
            _tool_colour(hover, grid),
            preview=preview,
            preview_colour=preview_colour,
            sections=None if whole else grid.sections,
            selection=None if in_scene or hide_selection else document.selection,
            mirror=mirror,
            plane=None if whole else _plane_preview(),
            scale=ui_scale(),
            boxes=boxes,
            brush_radius=radius,
            gizmo_state=gizmo_state,
            hover_handle=state.hover_handle,
            dragging=state.stroke_active,
            ghost=ghost,
            view_projection=view_state.screen_matrix,
            view_matrix=view_state.view_matrix,
            eye=view_state.camera.eye(),
            region_size=(region.width, region.height),
            region_id=region.as_pointer(),
            cage_bounds=cage_bounds,
            picked=whole and not document.scene.is_hidden(model),
            picked_hover=whole and hovered_uid == model.uid,
            matrix=picked_matrix,
        )
    except Exception:
        _disable()


def _late_anim_box(model):
    """(origin delta, size, matrix) of a part posed in Animate mode, or None."""
    if not state.animating:
        return None
    from . import anim_mode

    return anim_mode.box_of(model)


def _view_moved(view_state) -> bool:
    """True when the view being drawn is not the one the hover was picked in."""
    picked = state.hover_view
    return picked is not None and not np.array_equal(np.asarray(view_state.screen_matrix), np.asarray(picked))


def _schedule_repick() -> None:
    from . import mode  # late import: mode imports this module

    mode.schedule_repick()


def _draws_itself(module) -> bool:
    """True when `module` registered a draw handler of its own."""
    name = module.__name__
    return any(getattr(h.fn, "__module__", None) == name for h in registry.draw)


_foreign_errors: set = set()


def _foreign(name: str, fn_name: str, *args, **kwargs):
    """Call `fn_name` of sibling `name` when it exists and draws nowhere else.
    An error there prints once and does not stop this pass."""
    module = _mod(name)
    if module is None or _draws_itself(module):
        return None
    fn = getattr(module, fn_name, None)
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        if (name, fn_name) not in _foreign_errors:
            _foreign_errors.add((name, fn_name))
            print(f"Rok overlay: {name}.{fn_name} failed")
            traceback.print_exc()
        return None


def _draw_pixel(context) -> None:
    if _disabled:
        return
    try:
        document = voxel_object.document
        grid = document.grid
        if grid is None:
            return
        region = context.region
        if region is None:
            return
        scale = ui_scale()
        show_grid = palette_grid_shown()
        layout = strip_layout(region)
        palette_linear = _palette_linear(strip_palette(grid.palette))
        type_codes = strip_type_codes(document.type_codes())
        main = is_main_region(region) and not state.animating  # Animate mode: the dock stands there
        marking = _mod("marking")
        if marking is not None:
            dx, dy, _gone = marking.ui_pose("PALETTE")
            if dx or dy:  # an arm of the star's play shakes the card
                layout = _shifted(layout, dx, dy)
        if main:
            sx, sy, sz = (int(v) for v in grid.size)
            prefix = ""  # the label starts with the grid size; no file name
            size_text = f"{sx} x {sy} x {sz}"
            typing = state.size_edit is not None
            if typing:
                size_text = state.size_edit or " "  # the field keeps a width while empty
            colour_text = ""  # the card's header shows the colour; no slot number
            label_x = layout.x0
            label_y = layout.y0 + layout.height + round(6 * scale)
            if _dock is not None:  # the file line sits at the top, the title area, not on the card
                label_y = region.height - round(26 * scale)
            text, parts = _file_label(document, prefix, size_text, colour_text, label_x, label_y, scale)
            global _label_width
            fonts.size(0, 11 * scale)
            _label_width = fonts.dimensions(0, text)[0]
            if _dock is not None:
                draw_dock(_dock, layout, palette_linear, document.colour, state.hover_strip, scale,
                          type_codes=type_codes)
            else:
                draw_strip(layout, palette_linear, document.colour, state.hover_strip, scale,
                           grid_on=show_grid, type_codes=type_codes)
            draw_label(text, label_x, label_y, scale, pill=False, colour=STRIP_LABEL_INK)
            state.label_parts = (region.as_pointer(), {k: rect for k, (_t, rect) in parts.items()})
            hovered = state.hover_strip[0] if state.hover_strip is not None else None
            if typing:
                _draw_size_field(size_text, parts["GRID_SIZE"][1], label_y, scale)
            elif hovered in parts:
                part_text, rect = parts[hovered]
                _draw_size_hover(part_text, rect, rect[0], label_y, scale)
            # 12px above the in-view hint band (the file-name row above the palette).
            # With the 256 grid open, above the grid: in Rok text always draws
            # over the shapes of its layer, so the grid could not cover it.
            _draw_tool_hud(scale, _hud_top(region, layout, scale, show_grid), region)
            if show_grid:
                hover_swatch = None
                if state.hover_strip is not None and state.hover_strip[0] == "SWATCH":
                    hover_swatch = state.hover_strip[1]
                draw_palette(palette_layout(region), palette_linear, document.colour, hover_swatch, None, scale,
                             type_codes=type_codes)
        if state.steady_line is not None:
            draw_steady_hand(*state.steady_line, scale)
        preview = state.preview
        if preview is not None:
            if getattr(preview, "kind", None) == "SCREEN":
                draw_screen_polygon(preview.points, scale)
            if preview.label:
                draw_label(
                    preview.label,
                    state.last_mouse[0] - region.x + 16 * scale,
                    state.last_mouse[1] - region.y + 16 * scale,
                    scale,
                )
        chrome = _mod("chrome")
        if main and chrome is not None and not _draws_itself(chrome) and hasattr(chrome, "layout_for"):
            _foreign(
                "chrome", "draw_chrome",
                chrome.layout_for(region),
                hover_id=state.hover_chrome,
                open_id=state.chrome_open,
                focus_id=state.chrome_focus,
                tooltip_for=chrome.tooltip_id() if hasattr(chrome, "tooltip_id") else None,
                region_width=region.width,
                size_hover=getattr(chrome, "hover_stop", None),
                size_dragging=getattr(chrome, "slider_dragging", False),
            )
        if not _shader_panel_open():  # the panel stands there; its Where row names the scope
            _draw_cage_strip(region, scale)
        if main and state.level == "MODEL" and not _shader_panel_open():
            _draw_corner_gizmo(region, scale)
            _draw_cage_lattice(region, scale)
        elif state.corner is not None and state.corner[0] == region.as_pointer():
            state.corner = None  # the scene has no corner gizmo: nothing to hit
        if main and state.level == "SCENE":
            _foreign("scene_tools", "draw_scene_extras", region, scale)
        for name in ("marking", "command", "welcome"):
            _foreign(name, "draw", region, scale)
        palette_ui = _mod("palette_ui")
        drag_label = None
        if palette_ui is not None and hasattr(palette_ui, "drag_label"):
            drag_label = palette_ui.drag_label(state.hover_strip)
        if drag_label:
            draw_label(
                drag_label,
                state.last_mouse[0] - region.x + 16 * scale,
                state.last_mouse[1] - region.y + 16 * scale,
                scale,
            )
    except Exception:
        _disable()


def _disable() -> None:
    """Print the traceback once and stop both callbacks until the next New or Open."""
    global _disabled
    _disabled = True
    print("Rok overlay: draw error, overlay off until the next New or Open")
    traceback.print_exc()


def _enable(_context=None) -> None:
    global _disabled
    _disabled = False
    invalidate_palette()


def _tool_colour(hover: Hover | None, grid) -> tuple[float, float, float]:
    """Highlight colour of the effective tool under the mouse.

    Select with subtract (Option+Shift) draws in the Erase colour, so the target
    outline shows it will remove cells from the selection, not add them.
    """
    tool = hover.tool if hover is not None else "ATTACH"
    subtract = hover.subtract if hover is not None else False
    return _colour_of(tool, grid, subtract)


def _colour_of(tool: str, grid, subtract: bool = False) -> tuple[float, float, float]:
    """Colour of a tool. PAINT uses the current palette colour. Select with
    subtract (Option+Shift) uses the Erase colour instead of its own."""
    if tool == SELECT and subtract:
        return TOOL_COLOURS["ERASE"]
    if tool in TOOL_COLOURS:
        return TOOL_COLOURS[tool]
    rgb = _palette_linear(grid.palette)[voxel_object.document.colour]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


# --- pure draw functions ----------------------------------------------------------------


def draw_scene(
    size,
    origin,
    hover: Hover | None,
    tool_colour,
    *,
    preview=None,
    preview_colour=None,
    sections=None,
    selection=None,
    mirror=(False, False, False),
    scale: float = 1.0,
    boxes=(),
    brush_radius: int = 1,
    gizmo_state=None,
    hover_handle: str | None = None,
    dragging: bool = False,
    ghost=None,
    view_projection=None,
    view_matrix=None,
    eye=None,
    region_size=None,
    region_id=None,
    cage_bounds=None,
    picked: bool = False,
    picked_hover: bool = False,
    matrix=None,
    plane=None,
) -> None:
    """The active model's box, mirror planes, selection, hover and preview, the boxes
    of the other models, the model ghost, the selection cage and the gizmo, in
    world space.

    `size` and `origin` are the active model's; everything of the active model
    goes through `matrix` (a scene Rotate preview, or None) and then to `origin`.
    `boxes` lists the other models as (origin, size, kind, matrix) with kind
    'OTHER', 'HOVER', 'PICKED' or 'SNAP'. `ghost` is (origin, tris, lines) of a
    transformed model. `cage_bounds` is inclusive (lo, hi) of the selection;
    when set, the ember cage is drawn and the gizmo is not. `gizmo_state` is a
    state.GizmoState; the gizmo is drawn last with the depth test off, and its
    projected anchors go to state.gizmo_screen (None when no gizmo is drawn).
    `view_projection`, `view_matrix`, `eye` and `region_size` describe the
    view; `region_id` names the region those anchors belong to.
    """
    if size is None:
        return
    eye = np.zeros(3) if eye is None else np.asarray(eye, dtype=np.float64)
    with _space(origin, matrix, eye):
        if picked:
            # The scene's picked model: an ember box, brighter under the mouse.
            alpha = 1.0 if picked_hover else PICKED_ALPHA
            _lines(hud.box_edges(size), EMBER_RGB + (alpha,), 2.0 * scale)
        elif state.model_box:
            _lines(hud.box_edges(size), LINE_RGB + (BOX_ALPHAS["ACTIVE"],), 1.0 * scale)
        _draw_mirror_planes(size, mirror, scale)
        if sections:
            for axis, (coord, _from_positive) in sections.items():
                _draw_section(size, axis, coord, scale)
        if plane is not None:
            _draw_plane_preview(size, plane, sections or {}, scale)
        if hover is not None:
            _draw_hover(hover, tool_colour, size, mirror, selection, brush_radius, scale)
        if preview is not None:
            _draw_preview(preview, preview_colour, size, mirror, scale)
    for box_origin, box_size, kind, *box_matrix in boxes:
        with _space(box_origin, box_matrix[0] if box_matrix else None):
            if kind == "PICKED":
                _lines(hud.box_edges(box_size), EMBER_RGB + (PICKED_ALPHA,), 2.0 * scale)
            elif kind == "SNAP":
                _lines(hud.box_edges(box_size), LINE_RGB + (1.0,), 2.0 * scale)
            elif state.model_box or kind not in ("OTHER", "ACTIVE"):
                alpha = BOX_ALPHAS.get(kind, BOX_ALPHAS["OTHER"])
                _lines(hud.box_edges(box_size), LINE_RGB + (alpha,), 1.0 * scale)
    if ghost is not None:
        with _space(ghost[0]):
            _draw_ghost(ghost, scale)
    state.gizmo_screen = None
    state.gizmo_region = None
    state.cage_screen = None
    state.cage_region = None
    state.gizmo_screens.pop(region_id, None)
    state.cage_screens.pop(region_id, None)
    with _on_top():
        if cage_bounds is not None:
            _draw_cage(origin, cage_bounds, scale, view_projection, region_size, region_id)
        elif gizmo_state is not None:
            _draw_gizmo(gizmo_state, hover_handle, dragging, scale, view_matrix, view_projection,
                        region_size, region_id)


def _inset(rect, d: float):
    x, y, w, h = rect
    return (x + d, y + d, w - 2 * d, h - 2 * d)


_label_width = 0.0  # the file label's width as last drawn: the plate holds it


def strip_plate(layout: StripLayout, scale: float):
    """The rounded plate behind the strip and its file label, region pixels.
    With the palette card: the card itself, its editor included."""
    if _dock is not None and (layout.x0, layout.y0) == (_dock.card[0], _dock.card[1]):
        x, y, w, h = _dock.card
        return (float(x), float(y), float(w), float(h + _dock_dy))  # the HUD follows the card's motion
    pad = STRIP_PLATE_PAD * scale
    top = layout.y0 + layout.height + round(6 * scale) + STRIP_LABEL_H * scale
    width = max(layout.width, _label_width)
    return (layout.x0 - pad, layout.y0 - pad, width + 2 * pad, top + pad * 0.7 - (layout.y0 - pad))


def draw_strip(
    layout: StripLayout,
    palette_linear: np.ndarray,
    colour: int,
    hover: tuple[str, int] | None,
    scale: float,
    *,
    grid_on: bool = False,
    type_codes: np.ndarray | None = None,
    new_look: bool = True,
) -> None:
    """The palette strip in Rok's look: one dark rounded plate (the file label
    sits on it too), rounded swatches with a hairline gap, the current colour
    as a large rounded tile, and rounded buttons. The layout and the hit tests
    come from hud.strip_layout; only the drawing changed.
    `palette_linear` holds the swatch colours (sRGB in Rok)."""
    plate = strip_plate(layout, scale)
    radius = STRIP_PLATE_RADIUS * scale
    gfx.rrect(plate, STRIP_PLATE_BG, radius)
    gfx.rrect(plate, STRIP_PLATE_EDGE, radius, stroke=1.0)
    flying = _swatch_rects(layout)
    pal = np.asarray(palette_linear, dtype=np.float32)
    if layout.divider is not None:
        y = float(layout.divider)
        gfx.rrect((layout.x0, y - 0.5, layout.width, 1.0), STRIP_DIVIDER, 0.0)
    gap = STRIP_SWATCH_GAP * scale
    for index, rect in layout.swatches:
        r = flying.get(index, rect)
        gfx.rrect(_inset(r, gap), tuple(pal[index][:3]) + (1.0,), STRIP_SWATCH_RADIUS * scale)
    favs = _favourites()
    if layout.favourites and favs:
        if layout.fav_divider is not None:
            gfx.rrect((layout.x0, float(layout.fav_divider) - 0.5, layout.width, 1.0), STRIP_DIVIDER, 0.0)
        fav_rgb = _palette_linear(np.array([(*c, 255) for c in favs], dtype=np.uint8))
        current_rgb = tuple(int(c) for c in voxel_object.document.scene.palette[colour, :3]) \
            if voxel_object.document.scene is not None else None
        for k, rect in layout.favourites:
            if k >= len(favs):
                break
            gfx.rrect(_inset(rect, gap), tuple(fav_rgb[k][:3]) + (1.0,), STRIP_SWATCH_RADIUS * scale)
            if favs[k] == current_rgb:
                gfx.rrect(_inset(rect, -1.0 * scale), EMBER, (STRIP_SWATCH_RADIUS + 1.0) * scale, stroke=1.5 * scale)
            elif hover is not None and hover[0] == "FAV" and hover[1] == k:
                gfx.rrect(_inset(rect, -1.0 * scale), STRIP_HOVER_RING, (STRIP_SWATCH_RADIUS + 1.0) * scale,
                          stroke=1.0 * scale)
    # Each section's name upright left of its rows, so used and unused colours read apart.
    app = CONTEXT.app
    for caption, rect in layout.captions:
        if app is not None:
            app.icons.upright_text(caption, hud.STRIP_CAPTION_SIZE * scale, rect[0], rect[1], STRIP_CAPTION_INK,
                                   height=rect[3])
    cur_r = STRIP_CURRENT_RADIUS * scale
    gfx.rrect(layout.current, tuple(pal[colour][:3]) + (1.0,), cur_r)
    gfx.rrect(layout.current, STRIP_TILE_EDGE, cur_r, stroke=1.0)
    hover_kind = hover[0] if hover is not None else None
    hover_index = hover[1] if hover is not None else None
    if hover_kind == "CURRENT":
        gfx.rrect(_inset(layout.current, -2.5 * scale), STRIP_HOVER_RING, cur_r + 2.5 * scale, stroke=1.5 * scale)
    # The material dots: a white dot with a dark ring, top right of a swatch.
    global _dock_codes
    _dock_codes = type_codes
    if type_codes is not None:
        for rect in hud.strip_marker_rects(layout, type_codes, colour):
            ring = _inset(rect, -_marker_outline(scale))
            gfx.rrect(ring, FRAME_BLACK, ring[2] / 2.0)
            gfx.rrect(rect, FRAME_WHITE, rect[2] / 2.0)
    for name, rect in layout.buttons:
        pressed = name == "GRID" and grid_on
        _draw_button(name, rect, pressed, scale, hovered=hover_kind == name)
    for index, rect in layout.swatches:
        r = flying.get(index, rect)
        if index == colour and index not in flying:
            gfx.rrect(_inset(r, -1.0 * scale), EMBER, (STRIP_SWATCH_RADIUS + 1.0) * scale, stroke=1.5 * scale)
        elif hover_kind == "SWATCH" and index == hover_index:
            gfx.rrect(_inset(r, -1.0 * scale), STRIP_HOVER_RING, (STRIP_SWATCH_RADIUS + 1.0) * scale, stroke=1.0 * scale)


def draw_palette(
    layout: PaletteLayout,
    palette_linear: np.ndarray,
    colour: int,
    hover_swatch: int | None,
    text: str | None = None,
    scale: float = 1.0,
    *,
    type_codes: np.ndarray | None = None,
) -> None:
    """The 256 swatches on their own rounded plate, as rounded cells with a
    hairline gap; rings on the current and the hovered swatch; material dots."""
    pad = STRIP_PLATE_PAD * scale
    w, h = layout.width, layout.height
    plate = (layout.x0 - pad, layout.y0 - pad, w + 2 * pad, h + 2 * pad)
    radius = STRIP_PLATE_RADIUS * scale
    gfx.rrect(plate, STRIP_PLATE_BG, radius)
    gfx.rrect(plate, STRIP_PLATE_EDGE, radius, stroke=1.0)
    pal = np.asarray(palette_linear, dtype=np.float32)
    gap = STRIP_SWATCH_GAP * scale
    r = STRIP_SWATCH_RADIUS * scale * 0.7
    for index in range(1, 256):
        rect = hud.cell_rect(layout, index)
        gfx.rrect(_inset(rect, gap), tuple(pal[index][:3]) + (1.0,), r)
    empty = hud.cell_rect(layout, 0)
    gfx.rrect(_inset(empty, gap), STRIP_BUTTON_BG, r)  # slot 0 is "no voxel"
    if type_codes is not None:
        for rect in hud.palette_marker_rects(layout, type_codes):
            ring = _inset(rect, -_marker_outline(scale))
            gfx.rrect(ring, FRAME_BLACK, ring[2] / 2.0)
            gfx.rrect(rect, FRAME_WHITE, rect[2] / 2.0)
    cur = hud.cell_rect(layout, colour)
    gfx.rrect(_inset(cur, -1.0 * scale), EMBER, r + 1.0 * scale, stroke=1.5 * scale)
    if hover_swatch is not None and hover_swatch != colour:
        gfx.rrect(_inset(hud.cell_rect(layout, hover_swatch), -1.0 * scale), STRIP_HOVER_RING, r + 1.0 * scale,
                  stroke=1.0 * scale)
    if text:
        _draw_text(layout, text, scale)


def draw_screen_polygon(points, scale: float) -> None:
    """A Rect or Lasso drag: the points as a closed, dashed white polyline."""
    dashes = dash_segments(points, True, SCREEN_DASH_PX * scale, SCREEN_GAP_PX * scale)
    if dashes.shape[0] == 0:
        return
    pen.lines(dashes, LINE_RGB + (SCREEN_ALPHA,), 1.5 * scale)


STEADY_RING_PX = 4.0  # the radius of the ring where Steady Hand holds the brush


def draw_steady_hand(brush, mouse, scale: float) -> None:
    """The string of Steady Hand: a line from the brush to the mouse, and a ring at the brush."""
    pen.lines(np.array([brush, mouse], dtype=np.float32), LINE_RGB + (SCREEN_ALPHA,), 1.5 * scale)
    r = STEADY_RING_PX * scale
    gfx.rrect((brush[0] - r, brush[1] - r, 2 * r, 2 * r), EMBER, r, stroke=1.5 * scale)


def _draw_icon(name: str, rect, ink, side: float | None = None) -> bool:
    """The icon PNG `name` tinted with `ink`, centred in `rect` (a square of
    `side`, or 82 % of the rect). True when the file was there."""
    app = CONTEXT.app
    if app is None or not name:
        return False
    x, y, w, h = (float(v) for v in rect)
    s = float(side) if side is not None else min(w, h) * 0.82
    return app.icons.draw(name, (x + (w - s) / 2.0, y + (h - s) / 2.0, s, s), ink)


def _hud_top(region, layout, scale: float, show_grid: bool) -> float:
    """The `hint_top` of the tool readout: above the palette card, or above the 256 grid."""
    plate = strip_plate(layout, scale)
    hint_top = max(layout.grid_y0, plate[1] + plate[3] + 10 * scale)
    if show_grid:
        grid_layout = palette_layout(region)
        hint_top = grid_layout.y0 + grid_layout.height + (STRIP_PLATE_PAD + 10) * scale
    return hint_top


def readout_top(region) -> int:
    """The top edge of the tool readout (its text and a small margin), in region pixels from the bottom."""
    scale = ui_scale()
    hint_top = _hud_top(region, strip_layout(region), scale, palette_grid_shown())
    block = hud.HUD_TIP_PX + hud.HUD_TITLE_TIP_GAP_PX + hud.HUD_TITLE_PX
    return int(round(hud.hud_anchor(scale, hint_top)[1] + (block + 8) * scale))


def _draw_tool_hud(scale: float, hint_top: float, region=None) -> None:
    """Large tool readout. Draw-only: no hit testing, no click steal.

    The strip ember stays on the rail's tool. Option-held erase flips only this HUD.
    While the mouse is on a flyout item of the rail, the readout names that
    item instead (its icon, its name, what it does): the flyouts have no
    floating tooltip.
    """
    wm = CONTEXT.prefs
    base = getattr(wm, "rok_tool", "ATTACH")
    mode_name = getattr(wm, "rok_mode", VOXEL)
    state.hud_last_tool, state.hud_tip_faded = hud.hud_on_tool_change(state.hud_last_tool, base, state.hud_tip_faded)
    point = state.hover_handle if lattice.parse_point_id(state.hover_handle) is not None else None
    if point is not None:
        readout = hud.HudReadout(title="Cage point", title_rgba=hud.HUD_TITLE_RGBA, tip=cage_drag.DRAG_TIP,
                                 tip_rgba=hud.HUD_TIP_RGBA, erase_held=False)
    elif cage.pill_tip_due(state.pill_since) and state.hover_pill:
        if cage_chips.is_chip(state.hover_pill):
            title, tip = cage_chips.CHIP_TITLES[state.hover_pill], cage_chips.CHIP_TIPS[state.hover_pill]
        else:
            title, tip = cage.pill_hud(state.hover_pill)
        readout = hud.HudReadout(
            title=title,
            title_rgba=hud.HUD_TITLE_RGBA,
            tip=tip or None,
            tip_rgba=hud.HUD_TIP_RGBA,
            erase_held=False,
        )
    else:
        readout = hud.hud_readout(base, state.alt, state.shift, tip_faded=state.hud_tip_faded, mode=mode_name)
        brush = sys.modules.get("rok.app.shader_brush")
        if (brush is not None and brush.armed is not None and base == "PAINT" and mode_name == VOXEL
                and not readout.erase_held):
            readout = hud.HudReadout(title="Shader brush", title_rgba=hud.HUD_TITLE_RGBA,
                                     tip=f"{brush.label()} · Esc puts it away", tip_rgba=hud.HUD_TIP_RGBA,
                                     erase_held=False)
    glyph = kit.RING_GLYPHS.get(mode_name, "")
    model = voxel_object.document.model
    scene = voxel_object.document.scene
    if model is not None and scene is not None and not readout.erase_held:
        name = model.display_name(scene.models.index(model))
        from rok.core import link as link_core

        copies = len(link_core.members(scene, model.link))
        if state.level == "SCENE":
            readout = hud.scene_readout(name, copies)
            glyph = "PICK"
        elif state.level == "BRICK":
            from . import brick_mode

            title, tip, glyph = brick_mode.readout()
            readout = hud.HudReadout(title=title, title_rgba=hud.HUD_TITLE_RGBA, tip=tip,
                                     tip_rgba=hud.HUD_TIP_RGBA, erase_held=False)
        elif len(scene.models) > 1:
            readout = hud.inside_readout(readout, name, copies)
    item = _hovered_flyout_item(region)
    if item is not None:
        name, what = kit.split_tooltip(item.tooltip)
        readout = hud.HudReadout(
            title=name, title_rgba=hud.HUD_TITLE_RGBA, tip=what or None,
            tip_rgba=hud.HUD_TIP_RGBA, erase_held=False,
        )
        glyph = item.glyph
    _x, y_tip = hud.hud_anchor(scale, hint_top)
    _x, y_title = hud.hud_title_origin(scale, hint_top)
    x = hud.hud_text_x(scale, hint_top)
    icon_rect = hud.hud_icon_rect(scale, hint_top)
    if glyph:
        _draw_icon(glyph, icon_rect, GLYPH_INK, side=float(icon_rect[2]))
    _draw_hud_text(readout.title, x, y_title, hud.HUD_TITLE_PX * scale, readout.title_rgba)
    if readout.tip:
        _draw_hud_text(readout.tip, x, y_tip, hud.HUD_TIP_PX * scale, readout.tip_rgba, bold=False)


def _hovered_flyout_item(region):
    """The rail flyout item under the mouse, or None."""
    if region is None or state.hover_chrome is None or state.chrome_open is None:
        return None
    chrome = _mod("chrome")
    if chrome is None or not hasattr(chrome, "layout_for"):
        return None
    if not kit.is_flyout_item(state.hover_chrome) or kit.layer_part(state.hover_chrome) is not None:
        return None
    for widget in kit.flyout_widgets(chrome.layout_for(region), state.chrome_open):
        if widget.id == state.hover_chrome:
            return widget
    return None


def _draw_hud_text(text: str, x: float, y: float, size: float, colour, *, bold: bool = False) -> None:
    """Text, pixel-snapped, with a 1px charcoal halo. Fake-bold is a 1px echo."""
    font = 0
    fonts.size(font, size)
    x = float(round(x))
    y = float(round(y))
    fonts.color(font, *hud.HUD_HALO_RGBA)
    for dx, dy in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
        fonts.position(font, x + dx, y + dy, 0)
        fonts.draw(font, text)
    fonts.color(font, *colour)
    if bold:
        fonts.position(font, x + 1.0, y, 0)
        fonts.draw(font, text)
    fonts.position(font, x, y, 0)
    fonts.draw(font, text)


def _file_label(document, prefix: str, size_text: str, colour_text: str, label_x: float, label_y: float, scale: float):
    """The file label's full text and its named, clickable parts: GRID_SIZE and
    LABEL_COLOUR. (A third part, what a .vox loses of an animation, is
    parked with the animation.)"""
    parts = {
        "GRID_SIZE": (size_text, _label_part_rect(prefix, size_text, label_x, label_y, scale)),
        "LABEL_COLOUR": (
            colour_text,
            _label_part_rect(f"{prefix}{size_text}   ", colour_text, label_x, label_y, scale),
        ),
    }
    if not colour_text:
        del parts["LABEL_COLOUR"]
        return f"{prefix}{size_text}", parts
    text = f"{prefix}{size_text}   {colour_text}"
    return text, parts


def _label_part_rect(prefix: str, part: str, x: float, y: float, scale: float):
    """The region-pixel rectangle of `part` inside a draw_label text that starts
    with `prefix`: the grid size's click target."""
    font = 0
    fonts.size(font, 11 * scale)
    # A space at the end is not measured on its own: measure prefix + "x" - "x".
    lead = fonts.dimensions(font, prefix + "x")[0] - fonts.dimensions(font, "x")[0]
    width, height = fonts.dimensions(font, part)
    _w, full_h = fonts.dimensions(font, "Xy")
    pad = PILL_PAD * scale
    return (
        int(round(x + lead - pad * 0.5)),
        int(round(y - pad)),
        int(round(width + pad)),
        int(round(max(height, full_h) + 2 * pad)),
    )


def _draw_size_field(text: str, rect, y: float, scale: float) -> None:
    """The grid size while it is typed: a field with an ember frame. Opened, the
    whole size is selected (ember behind dark text); then a cursor follows the
    text. A frame in grey says the text is not a size yet."""
    from rok.core.size_entry import parse_size

    font = 0
    fonts.size(font, 11 * scale)
    x = rect[0] + PILL_PAD * scale * 0.5
    text_w = fonts.dimensions(font, text.rstrip() or "")[0]
    fx, fy, fw, fh = rect
    fw = max(fw, int(text_w + PILL_PAD * scale * 1.5))
    pen.rect_fill((fx, fy, fw, fh), SIZE_FIELD_BG)
    if state.size_edit_fresh and text.strip():
        pen.rect_fill((x - 1, fy + 2, text_w + 2, fh - 4), EMBER)
        ink = SIZE_FIELD_INK_LIT
    else:
        ink = SIZE_FIELD_INK
        cursor = np.array([[x + text_w + 2, fy + 3], [x + text_w + 2, fy + fh - 3]], dtype=np.float32)
        pen.lines(cursor, EMBER, 1.0 * scale)
    frame_colour = EMBER if parse_size(text) is not None else SIZE_FIELD_OFF
    pen.lines(hud.rect_outline(fx, fy, fw, fh), frame_colour, 1.0 * scale)
    fonts.color(font, *ink)
    fonts.position(font, float(x), float(y), 0)
    fonts.draw(font, text)


def _draw_size_hover(text: str, rect, text_x: float, y: float, scale: float) -> None:
    """A clickable label part under the mouse: ember text with a hairline under it."""
    font = 0
    fonts.size(font, 11 * scale)
    x = rect[0] + PILL_PAD * scale * 0.5
    fonts.color(font, *EMBER)
    fonts.position(font, float(x), float(y), 0)
    fonts.draw(font, text)
    underline = np.array([[x, y - 2 * scale], [x + fonts.dimensions(font, text)[0], y - 2 * scale]], dtype=np.float32)
    pen.lines(underline, EMBER, 1.0 * scale)


def draw_label(text: str, x: float, y: float, scale: float, *, pill: bool = True, colour=None) -> None:
    """A short label at region pixel (x, y) on a dark rounded pill, e.g. the drag
    size at the mouse. Without `pill` it sits on a plate already (the file label)."""
    font = 0
    fonts.size(font, 11 * scale)
    tw, th = fonts.dimensions(font, text)
    if pill:
        pad = PILL_PAD * scale
        gfx.rrect((x - pad, y - pad, tw + 2 * pad, th + 2 * pad), STRIP_PLATE_BG, (th + 2 * pad) / 2.0)
    fonts.color(font, *(colour or TEXT_COLOUR))
    fonts.position(font, float(x), float(y), 0)
    fonts.draw(font, text)


def dash_segments(points, closed: bool, dash: float, gap: float) -> np.ndarray:
    """Cut a polyline into dashes: LINES pairs (2 K, 2) float32.

    A dash that would cross a vertex is cut at the vertex. Pure numpy.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    if closed and pts.shape[0] >= 3:
        a = pts
        b = np.roll(pts, -1, axis=0)
    else:
        a = pts[:-1]
        b = pts[1:]
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(length)))
    total = float(cum[-1])
    period = float(dash) + float(gap)
    if total <= 0.0 or period <= 0.0:
        return np.zeros((0, 2), dtype=np.float32)
    starts = np.arange(0.0, total, period)
    seg = np.searchsorted(cum, starts, side="right") - 1
    seg = np.clip(seg, 0, len(length) - 1)
    ends = np.minimum(starts + float(dash), cum[seg + 1])
    ok = (length[seg] > 0.0) & (ends > starts)
    starts, ends, seg = starts[ok], ends[ok], seg[ok]
    if starts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    unit = d[seg] / length[seg][:, None]
    p0 = a[seg] + unit * (starts - cum[seg])[:, None]
    p1 = a[seg] + unit * (ends - cum[seg])[:, None]
    return np.stack([p0, p1], axis=1).reshape(-1, 2).astype(np.float32)


# --- helpers --------------------------------------------------------------------------


def _draw_mirror_planes(size, mirror, scale: float) -> None:
    for axis in range(3):
        if not mirror[axis]:
            continue
        quad = hud.mirror_plane_quad(size, axis)
        _fan(quad, LINE_RGB + (MIRROR_FILL_ALPHA,))
        _lines(quad[_QUAD_OUTLINE], LINE_RGB + (MIRROR_EDGE_ALPHA,), 1.0 * scale)


_selection_cache: tuple[int, np.ndarray, np.ndarray] | None = None


def _selection_surface(selection):
    """(tris, lines) of the selection surface, cached by its version."""
    global _selection_cache
    version = selection.version
    if _selection_cache is None or _selection_cache[0] != version:
        lo, hi = selection.bounds()
        tris, lines = hud.mask_surface(selection.mask, lo, hi)
        _selection_cache = (version, tris, lines)
    return _selection_cache[1], _selection_cache[2]


def _draw_hover(hover: Hover, tool_colour, size, mirror, selection, brush_radius: int, scale: float) -> None:
    """What a click would touch: the face under the pointer, the target cell,
    a Face or Pattern preview. No mode-shaped ghost; the mode shows as an icon
    in the bottom-left readout, see `_draw_tool_hud`."""
    if hover.tool == ACTIVATE:
        return  # the hit is in another model's space; its box shows the hover
    rgb = tuple(float(c) for c in tool_colour[:3])
    mode_name = hover.mode
    cells = hover.cells
    # A faint face under the pointer, so the hit is obvious; but not on the grid's
    # wall where a click would do nothing (Erase or Paint over empty space: the
    # face showed on every wall, as if the pointer caught everywhere).
    result_shown = mode_name == VOXEL and hover.tool in (ATTACH, ERASE, PAINT)  # _HoverLayer draws the result
    if mode_name not in (edit.RECT, edit.LASSO) and not result_shown and shows_face(hover):
        _fan(hud.face_quad(hover.hit.voxel, hover.hit.normal), rgb + (FACE_ALPHA,))
    # The region a click would take (Pattern's stamp, a Face, a Connected or
    # By-colour blob, the brush's shell) is a keyed layer of its own
    # (_HoverLayer, sent from _draw_marks): built when the hover changes, not
    # every frame. Here only the cheap marks.
    if mode_name in (edit.PATTERN, edit.CLONE, edit.FACE, edit.RECT, edit.LASSO):
        return
    target = hover.target
    if target is None:
        return
    if result_shown:
        return  # the result itself is in the layer: no outline
    edge_rgb = rgb
    if hover.tool == SELECT and selection is not None and selection.contains(target):
        edge_rgb = LINE_RGB  # a drag here moves the selection
    _lines(hud.cube_edges(target), edge_rgb + (EDGE_ALPHA,), 2.0 * scale)
    if mode_name == VOXEL and brush_radius > 1:
        return  # the brush shell is in the layer; one outline is enough
    if any(mirror):
        for image in edit.mirror_images(target, size, mirror)[1:]:
            _lines(hud.cube_edges(image), rgb + (MIRROR_HOVER_ALPHA,), 2.0 * scale)


def shows_face(hover) -> bool:
    """The faint face under the pointer: on a voxel, or on a wall where a click acts."""
    if not hover.hit.wall:
        return True
    cells = hover.cells
    return hover.target is not None or (cells is not None and len(cells) > 0)


def _draw_preview(preview, preview_colour, size, mirror, scale: float) -> None:
    kind = preview.kind
    if kind not in ("PLANE", "CELLS"):
        return  # MOVE is drawn with the selection, SCREEN in the pixel pass
    if kind == "CELLS":
        # The cells themselves are a keyed layer of their own (_PreviewLayer,
        # sent from _draw_marks); the level plane is cheap and drawn here.
        level = getattr(preview, "level", None)
        if level is not None:
            _draw_level(size, level, scale)
        return
    cells = np.asarray(preview.cells).reshape(-1, 3)
    if cells.shape[0] == 0:
        return
    colour = preview_colour if preview_colour is not None else TOOL_COLOURS["ATTACH"]
    rgb = tuple(float(c) for c in colour[:3])
    fill = rgb + (PREVIEW_FILL_ALPHA,)
    # A footprint that fills its box (a Box base always does) is one quad, and
    # its mirror images are boxes: a 200 x 200 base on a 256-cube cost 15 to
    # 30 ms a move as one quad per cell (a ROK_PROFILE run).
    full = preview.box is not None and _fills_box(cells, preview.box)
    if kind == "PLANE":
        if full:
            _tris(hud.box_face(*preview.box, preview.normal), fill)
        else:
            _tris(hud.cells_face_quads(cells, preview.normal), fill)
        if preview.box is not None:
            lo, hi = preview.box
            _lines(hud.slab_edges(lo, hi), rgb + (PREVIEW_BOX_ALPHA,), 1.5 * scale)
    if any(mirror) and preview.tool != SELECT:
        boxes = _mirrored_boxes(preview.box, size, mirror) if full else None
        if boxes is not None and _disjoint(boxes):
            for lo, hi in boxes[1:]:
                _tris(hud.box_faces(lo, hi), rgb + (PREVIEW_GHOST_ALPHA,))
        else:
            ghosts = _mirror_ghosts(cells, size, mirror)
            if ghosts.shape[0] > 0:
                _tris(hud.cells_faces(ghosts), rgb + (PREVIEW_GHOST_ALPHA,))
    level = getattr(preview, "level", None)
    if level is not None:
        _draw_level(size, level, scale)


# A CELLS preview (a box, a line, a solid during its height stage) is a keyed
# layer of its own, as the selection's skin is (raising a
# big box's height stuttered; a 490 000-cell box cost 14 ms a frame in
# cells_faces and 7 ms more to hash and upload). The gesture makes a new
# Preview on every change, so the Preview and its cells are the key: a frame
# without a change sends nothing. The depth bias is the GPU's (bias=False).


class _PreviewLayer:
    name = "overlay.preview"

    def __init__(self) -> None:
        self.key = None
        self.sent = False
        self.serial = 0

    def clear(self, layers) -> None:
        if self.sent:
            layers.clear(self.name)
        self.sent = False
        self.key = None

    def update(self, layers, z: int, preview, colour, size, mirror, place, origin, matrix, scale: float) -> None:
        if preview is None or getattr(preview, "kind", None) != "CELLS":
            self.clear(layers)
            return
        cells = np.asarray(preview.cells).reshape(-1, 3)
        box = getattr(preview, "box", None)
        if cells.shape[0] == 0 and box is None:
            self.clear(layers)
            return
        mirror_key = tuple(bool(m) for m in mirror)
        key = (id(preview), id(preview.cells), box, place, mirror_key, tuple(int(s) for s in size), colour, scale)
        if self.sent and key == self.key:
            return
        rgb = tuple(float(c) for c in colour[:3])
        d = _DrawList()
        ghosts = None
        if box is not None:
            # A plain box: its six faces and twelve edges; no cell listed
            # (a 200 x 200 base felt slow).
            lo, hi = box
            faces = hud.box_faces(lo, hi)
            edges = hud.slab_edges(lo, hi)
            if any(mirror) and preview.tool != SELECT:
                images = [b for b in _mirrored_boxes(box, size, mirror) if b != (tuple(lo), tuple(hi))]
                if images:
                    ghosts = np.concatenate([hud.box_faces(blo, bhi) for blo, bhi in images])
        else:
            faces = hud.cells_faces(cells)
            edges = hud.cells_edges(cells) if cells.shape[0] <= PREVIEW_EDGE_LIMIT else None
            if any(mirror) and preview.tool != SELECT:
                g = _mirror_ghosts(cells, size, mirror)
                ghosts = hud.cells_faces(g) if g.shape[0] > 0 else None
        if matrix is None:
            _direct(d, origin, faces, rgb + (PREVIEW_FILL_ALPHA,), edges, rgb + (PREVIEW_EDGE_ALPHA,), 1.0 * scale)
            if ghosts is not None:
                d.tris3(ghosts, gfx._srgb(rgb + (PREVIEW_GHOST_ALPHA,)))
        else:
            with gfx.view(d), _space(origin, matrix, np.zeros(3)):
                _tris(faces, rgb + (PREVIEW_FILL_ALPHA,), bias=False)
                if edges is not None:
                    _lines(edges, rgb + (PREVIEW_EDGE_ALPHA,), 1.0 * scale, bias=False)
                if ghosts is not None:
                    _tris(ghosts, rgb + (PREVIEW_GHOST_ALPHA,), bias=False)
        self.serial += 1
        d.key = (self.name, self.serial)
        layers.send(self.name, z, d)
        self.sent = True
        self.key = key
        self._preview = preview  # keeps id(preview) ours while the key holds


_PREVIEW_LAYER = _PreviewLayer()


class _HoverLayer:
    """The region under the mouse that a click would take, as a keyed layer
    (a By-colour hover over a big model rebuilt the whole
    blob's faces on every frame while the mouse rested). The key is the
    hover's (state.hover_key) and its region's identity: a frame without a
    change sends nothing."""

    name = "overlay.hover"

    def __init__(self) -> None:
        self.key = None
        self.sent = False
        self.serial = 0
        self._hover = None
        self.ghost_idx = None  # the host model that carries ghost voxels, or None

    def clear(self, layers) -> None:
        if self.sent:
            layers.clear(self.name)
        self.sent = False
        self.key = None
        self._hover = None
        self.clear_ghost(layers)

    def clear_ghost(self, layers) -> None:
        if self.ghost_idx is not None:
            layers.host.set_ghost(self.ghost_idx, [])
            self.ghost_idx = None

    def update(self, layers, z: int, hover, colour, size, mirror, brush_radius: int, edge_white: bool,
               place, origin, matrix, scale: float, paint_rgb=None, voxels=None, ghost=None) -> None:
        """`ghost` is (host model index, palette index) when the host can show
        ghost voxels (host.set_ghost) for this model; else None."""
        voxel_result = (hover is not None and hover.mode == VOXEL and hover.tool in (ATTACH, ERASE, PAINT)
                        and hover.target is not None and voxels is not None and not state.stroke_active)
        as_ghost = voxel_result and ghost is not None and hasattr(layers.host, "set_ghost")
        if not as_ghost:
            self.clear_ghost(layers)
        heavy = hover is not None and hover.tool != ACTIVATE and (
            (hover.mode in (edit.PATTERN, edit.CLONE, edit.FACE) or hover.mode in edit.SELECT_ONLY_MODES)
            and hover.cells is not None and len(hover.cells) > 0
            or voxel_result
            or (hover.mode == VOXEL and brush_radius > 1 and hover.target is not None and not state.stroke_active))
        if not heavy:
            self.clear(layers)
            return
        mirror_key = tuple(bool(m) for m in mirror)
        key = (hover_key(hover), id(hover.cells), mirror_key, tuple(int(s) for s in size), colour, int(brush_radius),
               bool(edge_white), place, scale, paint_rgb, None if voxels is None else voxel_object.document.edit_serial,
               ghost if as_ghost else None)
        if self.sent and key == self.key:
            return
        rgb = tuple(float(c) for c in colour[:3])
        d = _DrawList()
        with gfx.view(d), _space(origin, matrix, np.zeros(3)):
            cells = None if hover.cells is None else np.asarray(hover.cells).reshape(-1, 3)
            clone_colours = _clone_colours(hover, hover.cells) if hover.mode == edit.CLONE else None
            if clone_colours is not None:
                # Clone: the ghost in the copied voxels' own colours, one batch per colour.
                for c_rgb, group in clone_colours:
                    _tris(hud.cells_faces(group), c_rgb + (min(1.0, PREVIEW_FILL_ALPHA * 2.0),), bias=False)
                if len(cells) <= PREVIEW_EDGE_LIMIT:  # faint edges: the colours speak
                    _lines(hud.cells_edges(cells), rgb + (PREVIEW_EDGE_ALPHA * 0.35,), 1.0 * scale, bias=False)
            elif hover.mode in edit.STAMP_MODES:
                _tris(hud.cells_faces(cells), rgb + (PREVIEW_FILL_ALPHA,), bias=False)
                if len(cells) <= PREVIEW_EDGE_LIMIT:
                    _lines(hud.cells_edges(cells), rgb + (PREVIEW_EDGE_ALPHA,), 1.0 * scale, bias=False)
                if any(mirror) and hover.tool != SELECT:
                    ghosts = _mirror_ghosts(cells, size, mirror)
                    if ghosts.shape[0] > 0:
                        _tris(hud.cells_faces(ghosts), rgb + (PREVIEW_GHOST_ALPHA,), bias=False)
            elif hover.mode == edit.FACE:
                normal = hover.hit.normal
                if hover.hit.wall:
                    normal = (-normal[0], -normal[1], -normal[2])
                _tris(hud.cells_face_quads(cells, normal), rgb + (FACE_ALPHA,), bias=False)
            elif hover.mode in edit.SELECT_ONLY_MODES:
                # Connected: a light skin over every voxel the click would take.
                _tris(hud.cells_faces(cells), rgb + (PREVIEW_GHOST_ALPHA,), bias=False)
            elif as_ghost:
                # The renderer shows the result itself: real voxels, really shaded;
                # an erase ghost blanks the voxels a click would take.
                idx, palette_index = ghost
                erase = hover.tool == ERASE
                boxes = _ghost_boxes(hover.target, brush_radius, size, mirror, voxels, 1 if erase else palette_index,
                                     hover.tool == ATTACH)
                layers.host.set_ghost(idx, boxes, erase=erase)
                self.ghost_idx = idx
                if erase:
                    _erase_cage(boxes, scale)
            elif voxel_result:
                # No ghosts on this host: the result as shaded faces, the erased
                # voxels in the Erase colour.
                sphere = edit.sphere_offsets(brush_radius) + np.asarray(hover.target, dtype=np.int64)
                sphere = sphere[edit.in_bounds_mask(sphere, size)]
                filled = voxels[sphere[:, 0], sphere[:, 1], sphere[:, 2]] != 0
                if hover.tool == ATTACH:
                    cells, result_rgb, alpha = sphere[~filled], paint_rgb, VOXEL_PREVIEW_ALPHA
                elif hover.tool == PAINT:
                    cells, result_rgb, alpha = sphere[filled], paint_rgb, VOXEL_PREVIEW_ALPHA
                else:
                    cells, result_rgb, alpha = sphere[filled], TOOL_COLOURS["ERASE"], ERASE_PREVIEW_ALPHA
                tris, colours = _shaded_cells(cells, size, result_rgb, alpha)
                if len(tris):
                    d.tris3(_world(tris, bias=False), colours)  # a colour per vertex: past pen
                if any(mirror) and len(cells):
                    images = _mirror_ghosts(cells, size, mirror)
                    tris, colours = _shaded_cells(images, size, result_rgb, min(alpha, VOXEL_PREVIEW_MIRROR_ALPHA))
                    if len(tris):
                        d.tris3(_world(tris, bias=False), colours)
            else:
                # The brush sphere around the cell a click would touch: its outer cells as a light skin.
                edge_rgb = LINE_RGB if edge_white else rgb
                shell = edit.sphere_shell(brush_radius) + np.asarray(hover.target, dtype=np.int64)
                shell = shell[edit.in_bounds_mask(shell, size)]
                if len(shell):
                    _tris(hud.cells_faces(shell), edge_rgb + (PREVIEW_GHOST_ALPHA,), bias=False)
        self.serial += 1
        d.key = (self.name, self.serial)
        layers.send(self.name, z, d)
        self.sent = True
        self.key = key
        self._hover = hover  # keeps id(hover.cells) ours while the key holds


_HOVER_LAYER = _HoverLayer()


def _draw_level(size, level, scale: float) -> None:
    """The height stage's level plane: across the grid at the top of the block, in the
    axis colour, so the height can be lined up with other things."""
    axis, coord = level
    quad = hud.mirror_plane_quad(size, int(axis))
    quad[:, int(axis)] = float(coord)
    colour = AXIS_COLOURS[int(axis)]
    _fan(quad, colour + (LEVEL_FILL_ALPHA,))
    _lines(quad[_QUAD_OUTLINE], colour + (LEVEL_EDGE_ALPHA,), 1.5 * scale)


def _draw_section(size, axis: int, coord: int, scale: float) -> None:
    """A section plane: a faint quad with an outline across `axis` at `coord`."""
    quad = hud.mirror_plane_quad(size, int(axis))
    quad[:, int(axis)] = float(coord)
    _fan(quad, LINE_RGB + (SECTION_FILL_ALPHA,))
    _lines(quad[_QUAD_OUTLINE], LINE_RGB + (SECTION_EDGE_ALPHA,), 1.5 * scale)


def _mask_now():
    """(key, mask) of the active model's mask, or None."""
    select_ops = _mod("select_ops")
    mask = select_ops.mask_of() if select_ops is not None else None
    if mask is None:
        return None
    doc = voxel_object.document
    return (doc.active_uid, doc.mask_serial, mask.shape), mask


# The selection's and the mask's skins can be very big (100 000+ voxels). Each
# is cut into blocks of BLOCK cells a side, and every block is a draw layer of
# its own, keyed, never hashed: a change rebuilds and sends only the blocks it
# touched, and a frame without a change sends nothing (a big
# selection slowed every frame). A block looks across its edges, so the skin
# is the same as one built whole. No CPU depth bias: the GPU pulls overlay
# geometry toward the eye (overlay.wgsl); a bias by the eye would make every
# block change with every camera move.
BLOCK = 32  # for grids up to 96 a side; a bigger grid takes bigger blocks (block_size)


def block_size(shape) -> int:
    """A skin's block side: BLOCK for every grid. Bigger blocks for big grids
    were tried and lost: the native builder gets fewer threads,
    and a block's cost is its geometry, not its count."""
    return BLOCK
_NEAR = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int64)


def _dirty_blocks(changed: np.ndarray, block: int = BLOCK) -> set[tuple[int, int, int]]:
    """The blocks of `block` cells a side that hold a changed cell or one of its six neighbours."""
    if not changed.any():
        return set()
    grown = changed.copy()
    for axis in range(3):
        lo = tuple(slice(1, None) if i == axis else slice(None) for i in range(3))
        hi = tuple(slice(None, -1) if i == axis else slice(None) for i in range(3))
        grown[lo] |= changed[hi]
        grown[hi] |= changed[lo]
    shape = np.asarray(grown.shape, dtype=np.int64)
    counts = -(-shape // block)
    pad = counts * block - shape
    if pad.any():
        grown = np.pad(grown, [(0, int(p)) for p in pad], constant_values=False)
    blocks = grown.reshape(counts[0], block, counts[1], block, counts[2], block).any(axis=(1, 3, 5))
    return set(map(tuple, np.argwhere(blocks).tolist()))


def _direct(d, origin, tris, fill, lines=None, edge=None, width: float = 1.0) -> None:
    """Grid-space geometry straight into the draw list, the model's origin as
    the layer's GPU offset: no Python pass over the points (a
    210-cube's skin is 264 000 quads a step; transforming their points cost
    15 ms). The colours go through gfx's sRGB step, as gfx.tris applies it."""
    d.offset = tuple(float(v) for v in origin)
    if tris is not None and len(tris):
        d.tris3(tris, gfx._srgb(fill))
    if lines is not None and len(lines):
        d.lines3(lines, gfx._srgb(edge), width)


class _Skin:
    """One mask's skin, as block layers named `name`/i.j.k."""

    def __init__(self, name: str, rgb, fill_alpha: float, edge_alpha: float, width: float) -> None:
        self.name = name
        self.rgb = tuple(rgb)
        self.fill_alpha = fill_alpha
        self.edge_alpha = edge_alpha
        self.width = width
        self.mask: np.ndarray | None = None  # what the blocks show
        self.version = None
        self.place = None
        self.blocks: set[tuple[int, int, int]] = set()  # the block layers that are sent
        self.serial = 0

    def _layer(self, b) -> str:
        return f"{self.name}/{b[0]}.{b[1]}.{b[2]}"

    def clear(self, layers) -> None:
        for b in self.blocks:
            layers.clear(self._layer(b))
        self.blocks = set()
        self.mask = None
        self.version = None
        self.place = None

    def update(self, layers, z: int, mask, version, place, origin, matrix, scale: float) -> None:
        """Show `mask` (None: nothing). `version` names its content, `place`
        the model, its place and the layers' generation."""
        if mask is None:
            if self.blocks or self.mask is not None:
                self.clear(layers)
            return
        if place == self.place and version == self.version:
            return
        shape = np.asarray(mask.shape, dtype=np.int64)
        block = block_size(mask.shape)
        if place != self.place or self.mask is None or self.mask.shape != mask.shape:
            for b in self.blocks:
                layers.clear(self._layer(b))
            self.blocks = set()
            counts = -(-shape // block)
            dirty = {(i, j, k) for i in range(counts[0]) for j in range(counts[1]) for k in range(counts[2])}
            dirty = {b for b in dirty if mask[tuple(slice(c * block, (c + 1) * block) for c in b)].any()}
        else:
            # The blocks with a changed cell or a face neighbour of one, by mask
            # arithmetic: no cell list, no sort (a live turn
            # of a 729 000-cell selection spent 130 ms a frame in np.unique here).
            dirty = _dirty_blocks(mask != self.mask, block)
        self.mask = mask.copy()
        self.version = version
        self.place = place
        dirty = sorted(dirty)
        boxes = []
        for b in dirty:
            lo = np.asarray(b, dtype=np.int64) * block
            boxes.append((lo, np.minimum(lo + block - 1, shape - 1)))
        built = hud.mask_surface_blocks(mask, boxes, neighbours=True)  # every block at once, in parallel
        for b, (tris, lines) in zip(dirty, built):
            name = self._layer(b)
            if len(tris) == 0:
                if b in self.blocks:
                    layers.clear(name)
                    self.blocks.discard(b)
                continue
            d = _DrawList()
            if matrix is None:
                # Grid-space points as the builder made them; the GPU adds the
                # model's origin (the layer's offset). No Python pass over the
                # geometry (a 210-cube's skin is 264 000 quads
                # a step, and the transform of their points cost 15 ms).
                _direct(d, origin, tris, self.rgb + (self.fill_alpha,), lines, self.rgb + (self.edge_alpha,),
                        self.width * scale)
            else:
                with gfx.view(d), _space(origin, matrix, np.zeros(3)):
                    _tris(tris, self.rgb + (self.fill_alpha,), bias=False)
                    _lines(lines, self.rgb + (self.edge_alpha,), self.width * scale, bias=False)
            self.serial += 1
            d.key = (name, self.serial)
            layers.send(name, z, d)
            self.blocks.add(b)


_SELECTION_SKIN = _Skin("overlay.selection", AMBER, SELECTION_FILL_ALPHA, SELECTION_EDGE_ALPHA, 1.5)
_MASK_SKIN = _Skin("overlay.mask", MASK_RGB, MASK_FILL_ALPHA, MASK_EDGE_ALPHA, 1.0)


def _marks_place(view_state):
    """(uid, origin, matrix, key of both) of the active model in a model, or None."""
    doc = voxel_object.document
    model = doc.model
    if model is None or doc.grid is None or state.level == "SCENE" or view_state is None:
        return None
    origin = voxel_object.model_origin(model)
    offset = state.model_drag_offset if state.stroke_active else (0, 0, 0)
    matrix = state.turn_preview.get(model.uid)
    if matrix is None and any(offset):
        origin = tuple(o + d for o, d in zip(origin, offset))
    mkey = None if matrix is None else np.asarray(matrix, dtype=np.float64).tobytes()
    return model.uid, tuple(origin), matrix, (model.uid, tuple(origin), mkey, ui_scale())


def _shader_panel_open() -> bool:
    """While the shader panel is open the model shows its live result: the
    selection's skin, cage and gizmo would cover it (its Where row names the
    scope instead)."""
    panel = sys.modules.get("rok.app.shader_panel")
    return panel is not None and panel.active()


def _draw_marks(context) -> None:
    """The selection's and the mask's block layers (they send themselves)."""
    gfx.current().key = "marks"  # this handler's own layer stays empty
    app = context.app
    layers = getattr(app, "layers", None)
    if layers is None:
        return
    found = _marks_place(context.view_state)
    if found is None:
        _SELECTION_SKIN.update(layers, VIEW_Z - 1, None, None, None, None, None, 1.0)
        _MASK_SKIN.update(layers, VIEW_Z - 1, None, None, None, None, None, 1.0)
        _PREVIEW_LAYER.clear(layers)
        _HOVER_LAYER.clear(layers)
        return
    uid, origin, matrix, place = found
    place = place + (layers.generation,)
    scale = ui_scale()
    grid_now = voxel_object.document.grid
    hover = state.hover if not _view_moved(context.view_state) else None  # draw_scene picks again
    if hover is not None and grid_now is not None:
        wm = context.prefs
        mirror = tuple(bool(v) for v in getattr(wm, "rok_mirror", (False, False, False)))
        radius = int(getattr(wm, "rok_brush_size", 1)) if hover.mode == VOXEL else 1
        sel = voxel_object.document.selection
        edge_white = (hover.tool == SELECT and hover.target is not None and sel is not None
                      and not _shader_panel_open() and sel.contains(hover.target))
        paint = _palette_linear(grid_now.palette)[voxel_object.document.colour]
        sync = getattr(app, "sync", None)
        ghost = None
        if sync is not None and uid in sync.uids and matrix is None and not state.stroke_active:
            ghost = (sync.uids.index(uid), int(voxel_object.document.colour))
        _HOVER_LAYER.update(layers, VIEW_Z - 1, hover, _tool_colour(hover, grid_now), grid_now.size, mirror, radius,
                            edge_white, place, origin, matrix, scale,
                            paint_rgb=(float(paint[0]), float(paint[1]), float(paint[2])), voxels=grid_now.visible,
                            ghost=ghost)
    else:
        _HOVER_LAYER.clear(layers)
    preview = state.preview
    grid = voxel_object.document.grid
    if preview is not None and grid is not None:
        colour = _colour_of(preview.tool, grid, getattr(preview, "subtract", False))
        mirror = getattr(context.prefs, "rok_mirror", (False, False, False))
        _PREVIEW_LAYER.update(layers, VIEW_Z - 1, preview, colour, grid.size, mirror, place, origin, matrix, scale)
    else:
        _PREVIEW_LAYER.clear(layers)
    selection = voxel_object.document.selection
    if selection is not None and not selection.is_empty() and not _shader_panel_open():
        _SELECTION_SKIN.update(layers, VIEW_Z - 1, selection.mask, selection.version, place, origin, matrix, scale)
    else:
        _SELECTION_SKIN.update(layers, VIEW_Z - 1, None, None, None, None, None, scale)
    mask_now = _mask_now()
    if mask_now is not None:
        _MASK_SKIN.update(layers, VIEW_Z - 1, mask_now[1], mask_now[0], place, origin, matrix, scale)
    else:
        _MASK_SKIN.update(layers, VIEW_Z - 1, None, None, None, None, None, scale)


_move_built: tuple[object, list] | None = None  # (key, [(part, array)])


def _draw_move(context) -> None:
    """The selection's moving copy during a move drag: its skin, built once per
    selection and place, shifted on the GPU by the drag's offset (no upload
    per step: 23 MB for a 250 000-voxel selection)."""
    global _move_built
    d = gfx.current()
    preview = state.preview
    found = _marks_place(context.view_state)
    selection = voxel_object.document.selection
    if (found is None or selection is None or selection.is_empty() or preview is None
            or preview.kind != "MOVE" or not any(preview.offset)):
        d.key = None
        return
    uid, origin, matrix, place = found
    scale = ui_scale()
    key = (place, selection.version)
    d.key = key
    offset = np.asarray(preview.offset, dtype=np.float64)
    if matrix is not None:
        offset = matrix[:3, :3] @ offset  # the grid's axes, in the world
    d.offset = tuple(float(v) for v in offset)
    if _move_built is None or _move_built[0] != key:
        built = _DrawList()
        with gfx.view(built), _space(origin, matrix, np.zeros(3)):
            fill, edges = _selection_surface(voxel_object.document.selection)
            _tris(fill, AMBER + (SELECTION_MOVE_ALPHA,), bias=False)
            _lines(edges, AMBER + (SELECTION_EDGE_ALPHA,), 1.5 * scale, bias=False)
        _move_built = (key, [(k, a) for k, parts in built.parts.items() for a in parts])
    for k, a in _move_built[1]:
        d.parts[k].append(a)


def _plane_preview():
    """("SECTION" | "MIRROR", axis) while the mouse is on a nav cube X, Y or Z button,
    or while Mirror Half shows its result on the model."""
    mirror_preview = _mod("mirror_preview")
    axis = mirror_preview.shown_axis() if mirror_preview is not None else None
    if axis is not None:
        return "MIRROR", axis
    nav_cube = _mod("nav_cube")
    return nav_cube.plane_preview() if nav_cube is not None else None


def _draw_plane_preview(size, plane, sections, scale: float) -> None:
    """The plane a nav cube X, Y or Z button turns on or off:
    the mirror plane through the middle, or the section plane where it is or
    where a click puts it (the middle). Ember, brighter than a plane that is on."""
    kind, axis = plane
    quad = hud.mirror_plane_quad(size, int(axis))
    if kind == "SECTION":
        current = sections.get(int(axis))
        quad[:, int(axis)] = float(current[0] if current is not None else max(1, int(size[int(axis)]) // 2))
    _fan(quad, EMBER_RGB + (PLANE_PREVIEW_FILL,))
    _lines(quad[_QUAD_OUTLINE], EMBER_RGB + (0.95,), 2.0 * scale)


def _draw_ghost(ghost, scale: float) -> None:
    """The exposed surface of a transformed model at its new place: (origin, tris[, lines])."""
    _tris(np.asarray(ghost[1], dtype=np.float32).reshape(-1, 3), AMBER + (SELECTION_MOVE_ALPHA,))
    if len(ghost) > 2 and ghost[2] is not None:
        _lines(np.asarray(ghost[2], dtype=np.float32).reshape(-1, 3), AMBER + (PREVIEW_EDGE_ALPHA,), 1.0 * scale)


def _erase_cage(boxes, scale: float) -> None:
    """A red cage where the voxels a click would take stand:
    the edges of every exposed face. A big brush gets the box around them
    instead, so the hover stays as fast as it was (Erase took a lot of
    speed work; a cage of 50 costs 4 ms and 5 MB a move, a box nothing)."""
    rgba = TOOL_COLOURS["ERASE"] + (ERASE_CAGE_ALPHA,)
    for at, box in boxes:
        filled = box != 0
        origin = np.asarray(at, dtype=np.float32)
        if np.count_nonzero(filled) <= ERASE_CAGE_LIMIT:
            _tris_, lines = hud.mask_surface(filled, (0, 0, 0), np.asarray(box.shape) - 1)
        else:
            lo = [int(np.argmax(np.any(filled, axis=tuple(j for j in range(3) if j != k)))) for k in range(3)]
            hi = [box.shape[k] - int(np.argmax(np.any(filled, axis=tuple(j for j in range(3) if j != k))[::-1]))
                  for k in range(3)]
            lines = hud.box_edges(np.asarray(hi) - np.asarray(lo)) + np.asarray(lo, dtype=np.float32)
        if len(lines):
            _lines(lines + origin, rgba, 1.0 * scale, bias=False)


def _ghost_boxes(target, brush: int, size, mirror, voxels, colour: int, attach: bool) -> list:
    """The ghost boxes of a click's result: the brush sphere at `target` and
    its mirror images, each a dense (origin, uint8 box) holding `colour`. An
    Attach ghost covers empty cells only, a Paint or Erase ghost filled ones,
    as the click would. Box arithmetic, no cell list (a brush of 50 covers a
    million cells)."""
    full = edit.sphere_mask(brush)
    r = brush - 1
    boxes = []
    for image in edit.mirror_images(tuple(int(v) for v in target), size, mirror):
        lo = [int(image[k]) - r for k in range(3)]
        clip_lo = [max(lo[k], 0) for k in range(3)]
        clip_hi = [min(int(image[k]) + r, int(size[k]) - 1) for k in range(3)]
        if any(clip_lo[k] > clip_hi[k] for k in range(3)):
            continue
        mask = full[tuple(slice(clip_lo[k] - lo[k], clip_hi[k] - lo[k] + 1) for k in range(3))]
        sub = voxels[tuple(slice(clip_lo[k], clip_hi[k] + 1) for k in range(3))]
        hit = mask & ((sub == 0) if attach else (sub != 0))
        if not hit.any():
            continue
        box = np.where(hit, np.uint8(colour), np.uint8(0))
        boxes.append(((clip_lo[0], clip_lo[1], clip_lo[2]), box))
    return boxes


def _shaded_cells(cells, size, rgb, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """The exposed faces of `cells` as TRIS with a colour per vertex: `rgb`
    shaded by the face's direction (FACE_TONES), so the preview reads as the
    voxels it stands for. (tris (N, 3), colours (N, 4)) float32."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if cells.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
    lo = cells.min(axis=0)
    hi = cells.max(axis=0)
    mask = np.zeros(tuple(int(v) for v in hi - lo + 1), dtype=bool)
    local = cells - lo
    mask[local[:, 0], local[:, 1], local[:, 2]] = True
    tris, _lines = hud.mask_surface(mask, (0, 0, 0), hi - lo)
    tris = (tris + lo.astype(np.float32)).astype(np.float32)
    if tris.shape[0] == 0:
        return tris, np.zeros((0, 4), np.float32)
    t = tris.reshape(-1, 3, 3)
    normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    axis = np.argmax(np.abs(normal), axis=1)
    sign = np.sign(normal[np.arange(len(normal)), axis]).astype(np.int64)
    tone = _TONE_TABLE[axis, (sign > 0).astype(np.int64)]
    base = np.asarray(rgb, dtype=np.float32)[:3]
    colours = np.empty((len(normal), 3, 4), np.float32)
    colours[..., :3] = (base[None, :] * tone[:, None])[:, None, :]
    colours[..., 3] = alpha
    return tris, colours.reshape(-1, 4)


def _mirrored_boxes(box, size, mirror) -> list:
    """A box (lo, hi) and its mirror images across the mirror axes, each once."""
    lo, hi = box
    boxes = [(tuple(int(v) for v in lo), tuple(int(v) for v in hi))]
    for axis in range(3):
        if not mirror[axis]:
            continue
        n = int(size[axis]) - 1
        for blo, bhi in list(boxes):
            mlo, mhi = list(blo), list(bhi)
            mlo[axis], mhi[axis] = n - bhi[axis], n - blo[axis]
            if (tuple(mlo), tuple(mhi)) not in boxes:
                boxes.append((tuple(mlo), tuple(mhi)))
    return boxes


def _fills_box(cells: np.ndarray, box) -> bool:
    """True when the cells are every cell of the inclusive box, once each."""
    lo, hi = box
    volume = int(np.prod(np.asarray(hi, dtype=np.int64) - np.asarray(lo, dtype=np.int64) + 1))
    return len(cells) == volume and len(cells) > 0


def _disjoint(boxes) -> bool:
    """True when no two inclusive boxes share a cell."""
    for i, (alo, ahi) in enumerate(boxes):
        for blo, bhi in boxes[i + 1:]:
            if all(alo[k] <= bhi[k] and blo[k] <= ahi[k] for k in range(3)):
                return False
    return True


def _mirror_ghosts(cells: np.ndarray, size, mirror) -> np.ndarray:
    """The mirror images of the preview cells without the cells themselves."""
    images = edit.mirror_cells(cells, size, mirror)
    stride = np.asarray([int(size[1]) * int(size[2]), int(size[2]), 1], dtype=np.int64)
    own = np.asarray(cells, dtype=np.int64) @ stride
    keys = images @ stride
    return images[~np.isin(keys, own)]


# --- the gizmo ------------------------------------------------------------------------


def _draw_gizmo(
    gizmo_state, hover_handle, dragging: bool, scale: float, view_matrix,
    view_projection, region_size, region_id=None,
) -> None:
    """Draw the handles around the pivot and write state.gizmo_screen.

    The arrow length is `length_px` converted to world units at the pivot depth.
    Drawn on top: no depth bias.
    """
    width, height = int(region_size[0]), int(region_size[1])
    vp = np.asarray(view_projection, dtype=np.float64)
    view = np.asarray(view_matrix, dtype=np.float64)
    right = view[0, :3]
    view_dir = -view[2, :3]
    origin = np.asarray(gizmo_state.origin, dtype=np.float64)
    pivot = np.asarray(gizmo_state.pivot2, dtype=np.float64) / 2.0
    pivot_world = origin + pivot
    px = gizmo.project(vp, np.stack([pivot_world, pivot_world + right]), width, height)
    if not np.all(np.isfinite(px)):
        return  # the pivot is behind the eye
    px_per_unit = float(np.hypot(*(px[1] - px[0])))
    if px_per_unit <= 1e-6:
        return
    length = float(gizmo_state.length_px) / px_per_unit
    geom = gizmo.geometry(pivot, length, view_dir)
    shift = np.eye(4)
    shift[:3, 3] = origin
    state.gizmo_screen = gizmo.screen_geometry(geom, vp @ shift, width, height)
    state.gizmo_region = region_id
    state.gizmo_screens[region_id] = state.gizmo_screen

    def rgba(handle: str, base, alpha: float):
        if handle == hover_handle:
            return LINE_RGB + (1.0,)
        if dragging and hover_handle is not None:
            alpha *= GIZMO_FADE_ALPHA
        return tuple(float(c) for c in base) + (alpha,)

    colours = gizmo.AXIS_COLOURS
    with _space(origin):
        for axis in range(3):
            _lines(geom.lines[_MOVE[axis]], rgba(_MOVE[axis], colours[axis], 1.0), GIZMO_LINE_PX * scale, bias=False)
        for axis in range(3):
            _tris(geom.tris[_PLANE[axis]], rgba(_PLANE[axis], colours[axis], GIZMO_PLANE_ALPHA), bias=False)
        for axis in range(3):
            alpha = GIZMO_FADE_ALPHA if gizmo.ring_faded(view_dir, axis) else 1.0
            ring = geom.rings[_ROT[axis]]
            pairs = np.stack([ring, np.roll(ring, -1, axis=0)], axis=1).reshape(-1, 3)
            _lines(pairs, rgba(_ROT[axis], colours[axis], alpha), GIZMO_RING_PX * scale, bias=False)
        if not state.animating:  # a part is posed, not flipped or stretched
            for axis in range(3):
                _tris(geom.tris[_FLIP[axis]], rgba(_FLIP[axis], colours[axis], 1.0), bias=False)
        for axis in range(3):
            _tris(geom.tris[_MOVE[axis]], rgba(_MOVE[axis], colours[axis], 1.0), bias=False)
        if state.level != "SCENE" or state.animating:  # the scene's gizmo has no centre square...
            _tris(geom.tris["CENTER"], rgba("CENTER", LINE_RGB, 1.0), bias=False)
        else:  # ...but a sphere that scales the model in all three dimensions
            _tris(geom.tris["SCALE_ALL"], rgba("SCALE_ALL", LINE_RGB, 1.0), bias=False)


def _draw_cage(origin, bounds, scale: float, view_projection, region_size, region_id=None) -> None:
    """Draw the ember box around the selection and write state.cage_screen."""
    width, height = int(region_size[0]), int(region_size[1])
    vp = np.asarray(view_projection, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    geom = cage.geometry(bounds)
    shift = np.eye(4)
    shift[:3, 3] = origin
    state.cage_screen = cage.screen_geometry(geom, vp @ shift, width, height, scale)
    state.cage_region = region_id
    state.cage_screens[region_id] = state.cage_screen
    with _space(origin):
        _lines(geom.lines, EMBER, CAGE_LINE_PX * scale, bias=False)


PILL_ICONS = {"PAINT": "BRUSH", "INVERT": "INVERSE", "SEL_GROW": "GROW", "SEL_SHRINK": "SHRINK"}  # a pill whose icon file has another name


def _draw_cage_strip(region, scale: float) -> None:
    """The selection bar, in region pixels. Nothing when the cage is not drawn.

    Each group is a rounded well. Glyph PNG masks when present, line lists
    otherwise, in the rail's grey ink. Hover is a lighter fill, white ink
    and an ember ring; the HUD tip is drawn by `_draw_tool_hud`.
    """
    from .state import cage_screen_for

    screen = cage_screen_for(region)
    if screen is None or screen.strip is None:
        return
    strip = screen.strip
    hover = state.hover_pill
    s = float(scale)
    radius = round(8.0 * s)
    px, py, pw, ph = strip.rect  # the plate under the groups, on a slightly larger one that makes its edge
    edge = max(1, round(1.0 * s))
    pen.fan(hud.rounded_rect(px - edge, py - edge, pw + 2 * edge, ph + 2 * edge, round(13.0 * s)), BAR_EDGE)
    pen.fan(hud.rounded_rect(px, py, pw, ph, round(12.0 * s)), BAR_BG)
    for x, y, w, h in strip.wells:
        pen.fan(hud.rounded_rect(x - edge, y - edge, w + 2 * edge, h + 2 * edge, radius + edge), BAR_EDGE)
        pen.fan(hud.rounded_rect(x, y, w, h, radius), CAGE_GLYPH_BG)
    masked = _mask_now() is not None
    for name, rect in strip.pills:
        on = hover == name
        x, y, w, h = rect
        if on:
            pen.fan(hud.rounded_rect(x, y, w, h, radius), CAGE_GLYPH_HOVER)
            _draw_rect_outline(rect, EMBER, 0.0, 1.5 * s)
        if name == "PAINT":
            # The current colour, as a dot in the corner: what the button paints.
            grid = voxel_object.document.grid
            if grid is not None:
                rgb = _palette_linear(grid.palette)[int(voxel_object.document.colour)]
                _corner_dot(x, y, w, s, tuple(float(c) for c in rgb[:3]) + (1.0,))
        elif name == "MASK" and masked:
            # The same blue as the mask in the view: the protection is on.
            _corner_dot(x, y, w, s, MASK_RGB + (1.0,), 8.0)
        icon = PILL_ICONS.get(name, name)
        if _draw_icon(icon, rect, GLYPH_HOVER_INK if on else GLYPH_INK, side=min(w, h) * (0.88 if on else 0.82)):
            continue
        lines = cage.glyph_lines(name, rect)
        colour = GLYPH_HOVER_INK if on else GLYPH_INK
        pen.lines(lines, colour, 1.5 * s)


def _corner_dot(x: float, y: float, w: float, scale: float, colour, diameter: float = 11.0) -> None:
    """A small disc in the button's lower-right corner."""
    d = round(diameter * scale)
    dx, dy = x + w - d - round(4 * scale), y + round(4 * scale)
    ring = round(1.5 * scale)
    pen.fan(hud.rounded_rect(dx - ring, dy - ring, d + 2 * ring, d + 2 * ring, d / 2.0 + ring),
            (0.0, 0.0, 0.0, 0.7))
    pen.fan(hud.rounded_rect(dx, dy, d, d, d / 2.0), colour)


# --- the strip ------------------------------------------------------------------------


def _draw_button(name: str, rect, pressed: bool, scale: float, hovered: bool = False, on_colour=None) -> None:
    """A strip button: a rounded chip with a light glyph. `+`, the eyedropper, `256`.
    `on_colour` fills the star in that colour: the current colour is a favourite."""
    x, y, w, h = rect
    bg = BUTTON_BG_ON if pressed else (STRIP_BUTTON_HOVER if hovered else STRIP_BUTTON_BG)
    gfx.rrect(rect, bg, STRIP_BUTTON_RADIUS * scale)
    ink = GLYPH_HOVER_INK if hovered or pressed else GLYPH_INK
    inset = round(5.5 * scale)
    x0, y0, x1, y1 = x + inset, y + inset, x + w - inset, y + h - inset
    cx, cy = x + w / 2.0, y + h / 2.0
    if name == "NEW":
        lines = np.array([[x0, cy], [x1, cy], [cx, y0], [cx, y1]], dtype=np.float32)
        pen.lines(lines, ink, 1.5 * scale)
    elif name == "EYEDROP":
        # A pipette: a diagonal tube and a short tip at the bottom left.
        mid = np.array([cx - (cx - x0) * 0.35, cy - (cy - y0) * 0.35], dtype=np.float32)
        pen.lines(np.array([mid, [x1, y1]], dtype=np.float32), ink, 2.5 * scale)
        pen.lines(np.array([[x0, y0], mid], dtype=np.float32), ink, 1.2 * scale)
    elif name == "FAVOURITE":
        import math

        r_out, r_in = (x1 - x0) / 2.0, (x1 - x0) / 5.0
        pts = []
        for k in range(10):
            r = r_out if k % 2 == 0 else r_in
            a = math.pi / 2 + k * math.pi / 5
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
        if on_colour is not None:
            # Filled with lines from the centre to its edge: a layer draws
            # triangles before rounded rects, so a triangle fill would sit
            # under the card's plate.
            fan = []
            for k in range(10):
                a, b = np.asarray(pts[k]), np.asarray(pts[(k + 1) % 10])
                n = max(2, int(np.hypot(*(b - a)) / (0.75 * scale)) + 1)
                for t in np.linspace(0.0, 1.0, n):
                    fan += [[cx, cy], a + (b - a) * t]
            pen.lines(np.array(fan, dtype=np.float32), on_colour, 1.5 * scale)
            ink = on_colour
        segs = []
        for k in range(10):
            segs += [pts[k], pts[(k + 1) % 10]]
        pen.lines(np.array(segs, dtype=np.float32), ink, 1.2 * scale)
    else:
        font = 0
        fonts.size(font, 7 * scale)
        fonts.color(font, *ink)
        tw, th = fonts.dimensions(font, "256")
        fonts.position(font, float(cx - tw / 2.0), float(cy - th / 2.0), 0)
        fonts.draw(font, "256")


def _draw_rect_outline(rect, colour, inset: float, width: float) -> None:
    x, y, w, h = rect
    pen.lines(hud.rect_outline(x, y, w, h, inset), colour, width)


def _draw_frame(layout: PaletteLayout, index: int, colour, inset: float, width: float) -> None:
    _draw_rect_outline(hud.cell_rect(layout, index), colour, inset, width)


def _draw_text(layout: PaletteLayout, text: str, scale: float) -> None:
    draw_label(text, layout.x0, layout.y0 + layout.height + round(6 * scale), scale)


def _marker_outline(scale: float) -> int:
    """The dark ring of a material dot in pixels: one pixel, times scale."""
    return max(1, round(hud.MARK_OUTLINE * scale))


registry.add_draw("overlay.view", VIEW_Z, "VIEW", _draw_view)
registry.add_draw("overlay.marks", VIEW_Z - 1, "VIEW", _draw_marks)
registry.add_draw("overlay.favourite_tip", PIXEL_Z + 1, "PIXEL", _draw_favourite_tip)
registry.add_draw("overlay.move", VIEW_Z - 1, "VIEW", _draw_move)
registry.add_draw("overlay.pixel", PIXEL_Z, "PIXEL", _draw_pixel)
registry.load_post.append(_enable)
