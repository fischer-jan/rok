"""The star: hold Space for a quick menu at the cursor (rok/core/star.py).

Arms go out from the cursor, one per category, each a row of chips. Closed,
an arm shows two chips; move onto it and it reaches out to its full length
while the others fade. The rays unroll when the star opens and sway very
slowly while it is open; at the edge of the view they bend and run along it.

Picking follows View > Star: On Release (hold Space, let go over a chip) or
On Click (Space opens the star, a click picks, Space or Esc closes). A chip
does what the rail's button of the same id does (chrome._run); the star's own
chips (Model, Shaders) are handled here. Nothing on the star wipes the model
without asking first.

The name of the chip under the mouse is written in a pill at the centre.

Hold Space long enough with the pointer still and no arm out, and the arms
start to play (rok/core/star_play.py): now and then, one thing at a time,
they write ROK, make a shape, shake the palette card, eat the nav cube,
fight, or (rarely) call up the deep. A move sends them back to the star; a
release while they play picks nothing.
"""

from __future__ import annotations

import math
import random
import time
from time import localtime

import numpy as np

from rok.core import chrome as kit
from rok.core import star, star_play

from . import chrome, opkit
from .app import RokPrefs
from .gfx import pen
from .handlers import registry
from .object import document, redraw_3d_views
from .opkit import CONTEXT, EnumProp, Action, register
from .ops_common import redraw, rehover
from .paint import round_outline, round_rect, text, text_width
from .state import state

# The live star, or None. The draw handler and the status hint read it.
_session: "_Session | None" = None

RELEASE, CLICK = "RELEASE", "CLICK"
REACH_S = 0.16  # seconds an arm takes to reach out or pull back

CHIP_BG = (0.074, 0.069, 0.064, 0.94)
PILL = (0.055, 0.050, 0.048, 0.92)
# One colour per family: every arm belongs to a tool.
FAMILY = {
    "ATTACH": (0.310, 0.780, 0.500, 1.0),  # green
    "ERASE": (0.930, 0.330, 0.300, 1.0),  # red
    "PAINT": (0.960, 0.700, 0.240, 1.0),  # amber
    "SELECT": (0.380, 0.620, 0.980, 1.0),  # blue
    star.MODEL: (0.690, 0.470, 0.960, 1.0),  # violet: the whole-model tools
    star.SCENE: (0.560, 0.640, 0.800, 1.0),  # the scene's slate
    star.LAST: (0.820, 0.800, 0.770, 1.0),  # the last command: warm white
}
TEXT = chrome.TEXT
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)
DOT_IDLE = (0.300, 0.285, 0.272, 1.0)  # the centre dot with no arm under the pointer


class _Session:
    __slots__ = ("origin", "pointer", "rays", "starts", "since",
                 "extents", "curl_keys", "out_since", "open_ray", "last", "timer", "mode",
                 "still_since", "seed", "pose", "exit_from", "exit_at", "exit_thing")

    def __init__(self, origin, rays, starts, mode):
        self.origin = origin  # region pixels
        self.pointer = origin
        self.rays = rays
        self.starts = starts
        self.since = time.monotonic()
        self.extents = [0.0] * len(rays)
        self.curl_keys = [0] * len(rays)  # each arm's own rolls while it is out (star.curl_amount)
        self.out_since = [0.0] * len(rays)
        self.open_ray: int | None = None
        self.last = self.since
        self.timer = None
        self.mode = mode
        # The play (rok/core/star_play.py): the stillness it waits for, its
        # pose this frame, and the pose it goes back from after a move.
        self.still_since = self.since
        self.seed = random.randrange(1 << 30)
        self.pose: star_play.Pose | None = None
        self.exit_from: star_play.Pose | None = None
        self.exit_at = 0.0
        self.exit_thing: str | None = None  # what the arms were doing when the pointer moved


def active() -> bool:
    return _session is not None


def ui_pose(name: str) -> tuple[float, float, bool]:
    """How far the star's play has pulled a part of the UI (star_play.NAV_CUBE,
    PALETTE) and whether an arm has swallowed it: (dx, dy, hidden)."""
    session = _session
    if session is None or session.pose is None:
        return (0.0, 0.0, False)
    return session.pose.ui.get(name, (0.0, 0.0, False))


def playing(session: "_Session | None" = None) -> bool:
    """True while the arms play or go back from it: a release then picks nothing."""
    session = _session if session is None else session
    return session is not None and session.pose is not None


# --- the pick setting (View > Star), kept in Rok's config folder -------------------------


def _read_pick() -> str:
    from . import settings

    value = settings.get("star_pick")
    return value if value in (RELEASE, CLICK) else RELEASE


def _write_pick(self, _context) -> None:
    from . import settings

    settings.put("star_pick", self.rok_star_pick)
    app = CONTEXT.app
    if app is not None and app._started:
        app.set_menu()  # the View menu's check marks


def pick_mode() -> str:
    return getattr(CONTEXT.prefs, "rok_star_pick", RELEASE)


def status_hint() -> str | None:
    """The hint line while the star is open, or None."""
    session = _session
    if session is None:
        return None
    if session.mode == CLICK:
        return "Star: click a chip · move onto an arm to see all of it · Space or Esc closes"
    return "Star: let go of Space over a chip · move onto an arm to see all of it · let go in the middle to cancel"


# --- the operator -------------------------------------------------------------------------


def _rail_edge(region) -> float:
    """Where the free view starts: the rail's right edge."""
    rect = chrome.layout_for(region).rect
    return float(rect[0] + rect[2])


def _layout(session: _Session, region) -> star.StarLayout:
    now = time.monotonic()
    return star.star_layout(
        session.rays, session.origin, region.width, region.height, 1.0,
        opened=now - session.since, seconds=now - session.since, ray_starts=session.starts,
        extents=session.extents, left=_rail_edge(region),
        wiggles=[star.curl_amount(now - since, key) for since, key in zip(session.out_since, session.curl_keys)],
    )


@register
class MarkingAction(Action):
    command_id = "rok.marking"
    title = "Star"
    tooltip = "Hold Space: the quick menu at the cursor. Move onto an arm to see all of it"
    @classmethod
    def poll(cls, context):
        from rok.app.object import document

        return context.region is not None and document.grid is not None

    def invoke(self, context, event):
        global _session
        if _session is not None:
            self._close(context)
            return {"CANCELLED"}
        from . import last_command, model_menu

        rays = star.rays_for(state.level, False, model_menu.items(), last_command.current(), _recent_chips())  # animation is parked
        if not rays:
            return {"FINISHED"}
        # A rail list the pointer had open closes: the star is on top now.
        state.chrome_open = None
        state.hover_chrome = None
        starts = star.starts(star.closed(rays), 1.0)
        origin = (float(event.mouse_region_x), float(event.mouse_region_y))
        _session = _Session(origin, rays, starts, pick_mode())
        wm = context.prefs
        _session.timer = wm.add_timer(1.0 / 60.0)
        wm.add_modal(self)
        redraw()
        return {"RUNNING"}

    def modal(self, context, event):
        session = _session
        if session is None:
            return {"CANCELLED"}
        region = context.region
        kind, value = event.type, event.value
        if kind == "TIMER":
            self._animate(session)
            _play(session, region)
            region.tag_redraw()
            return {"RUNNING"}
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            pointer = (float(event.mouse_region_x), float(event.mouse_region_y))
            if math.hypot(pointer[0] - session.pointer[0], pointer[1] - session.pointer[1]) > 2.0:
                _wake(session)
            session.pointer = pointer
            session.open_ray = star.arm_under(_layout(session, region), *session.pointer, session.open_ray)
            region.tag_redraw()
            return {"RUNNING"}
        if kind == "SPACE" and value == "RELEASE" and session.mode == RELEASE:
            self._pick(context, region, session)
            return {"FINISHED"}
        if kind == "SPACE" and value == "PRESS" and not event.is_repeat and session.mode == CLICK:
            self._close(context)  # Space again closes the clicked-open star
            return {"CANCELLED"}
        if kind == "LEFT_MOUSE" and value == "PRESS":
            self._pick(context, region, session)
            return {"FINISHED"}
        if value == "PRESS" and kind in {"ESCAPE", "RIGHT_MOUSE"}:
            self._close(context)
            return {"CANCELLED"}
        if kind == "FOCUS_LOST":
            self._close(context)
            return {"CANCELLED"}
        return {"RUNNING"}

    def cancel(self, context):
        self._close(context)

    @staticmethod
    def _animate(session: _Session) -> None:
        """Move each arm's reach a step toward out (the open arm) or in."""
        now = time.monotonic()
        dt = max(0.0, now - session.last)
        session.last = now
        step = dt / REACH_S
        for i in range(len(session.extents)):
            target = 1.0 if i == session.open_ray else 0.0
            e = session.extents[i]
            if e == 0.0 and target > 0.0:
                # Each time an arm reaches out, its rolls are new ones.
                session.curl_keys[i] = random.randrange(1 << 30)
                session.out_since[i] = now
            session.extents[i] = min(target, e + step) if e < target else max(target, e - step)

    def _pick(self, context, region, session: _Session) -> None:
        # While the arms play, their chips wander under the pointer: a release picks nothing.
        chip = None if playing(session) else star.chip_at(_layout(session, region), *session.pointer, session.open_ray)
        self._close(context)
        if chip is not None:
            _apply(context, region, chip.chip.id)

    def _close(self, context) -> None:
        global _session
        session = _session
        _session = None
        if session is not None and session.timer is not None:
            context.prefs.remove_timer(session.timer)
        redraw_3d_views()  # a cube or card an arm held goes back to its place
        redraw()
        rehover()


# --- the play ------------------------------------------------------------------------------


def _wake(session: _Session) -> None:
    """The pointer moved: the stillness starts again, and arms at play go back."""
    now = time.monotonic()
    if session.pose is not None and session.exit_from is None:
        session.exit_from, session.exit_at = session.pose, now
        found = star_play.thing_at(now - session.still_since, session.seed)
        session.exit_thing = found[0] if found is not None else None
    session.still_since = now


def _targets(region) -> dict:
    """The middles of what the arms may grab: the nav cube and the palette card."""
    from . import nav_cube, overlay

    out = {}
    if CONTEXT.view_state is not None and nav_cube.has_cube(CONTEXT.view_state):
        c = nav_cube.home_centre(region, CONTEXT.view_state)
        out[star_play.NAV_CUBE] = (float(c[0]), float(c[1]))
    if document.grid is not None and not state.animating:
        strip = overlay.strip_layout(region)
        plate = overlay.strip_plate(strip, overlay.ui_scale())
        out[star_play.PALETTE] = (plate[0] + plate[2] / 2, plate[1] + plate[3] / 2)
        if document.scene is not None:
            palette = document.scene.palette
            out[star_play.SWATCHES] = [((x + w / 2, y + h / 2), tuple(int(c) for c in palette[i, :3]))
                                       for i, (x, y, w, h) in strip.used if 0 < i < 256]
    rail = chrome.layout_for(region)
    out[star_play.RAIL] = [w.rect for w in rail.widgets if w.glyph and w.rect[2] > 0][:8]
    from . import ui

    out[star_play.HINT] = (region.width / 2.0, ui.HINT_BOTTOM + 10.0)
    return out


def _play(session: _Session, region) -> None:
    """This frame's pose of the play, or None while the star is itself."""
    now = time.monotonic()
    if session.mode != RELEASE or session.open_ray is not None or any(session.extents):
        session.still_since = now  # the play waits for Space held and no arm out
    t = now - session.still_since
    had = session.pose is not None
    scene = None
    if t >= star_play.START or session.exit_from is not None:
        from . import ui

        clock = localtime()
        scene = star_play.Scene(session.rays, session.origin, session.starts, region.width, region.height,
                                _rail_edge(region), _targets(region), session.seed,
                                (clock.tm_hour, clock.tm_min, clock.tm_sec), ui._hint(CONTEXT))
        base = star_play.base_pose(scene, _layout(session, region))
    if t >= star_play.START:
        session.exit_from = None
        session.pose = star_play.pose(scene, base, t)
    elif session.exit_from is not None:
        w = (now - session.exit_at) / star_play.EXIT_S
        if w >= 1.0:
            session.exit_from = None
            session.pose = None
        else:
            session.pose = star_play.blend(session.exit_from, base, star_play.wake_weight(session.exit_thing, w))
    else:
        session.pose = None
    if session.pose is not None or had:
        redraw_3d_views()  # the nav cube and the palette card follow the arms


def _apply(context, region, chip_id: str) -> None:
    """Do what the chip says: the star's own chips here, the rest as the rail does."""
    if chip_id in kit.TOOL_IDS:
        # A tool chip picks the tool in the mode it was last used in.
        wm = context.prefs
        tool = kit.TOOL_IDS[chip_id]
        last = state.tool_modes.get(tool)
        wm.rok_tool = tool
        if last is not None:
            mode_, solid = last
            wm.rok_mode = mode_
            if tool == "ATTACH":
                wm.rok_solid = solid
        return
    if chip_id == star.STAR_MODEL:
        # The Model menu opens on hover: bring the pointer to its button.
        opener = kit.MODEL_ID
        widget = chrome.layout_for(region, context).by_id(opener)
        if widget is not None:
            x, y, w, h = widget.rect
            context.window.cursor_warp(int(region.x + x + w / 2), int(region.y + y + h / 2))
            state.chrome_open = opener
        return
    if chip_id == star.STAR_SHADERS:
        _later("shader_gallery")
        return
    if chip_id.startswith(star.STAR_COLOUR):
        index = int(chip_id[len(star.STAR_COLOUR):])
        if document.scene is not None and 0 < index < 256:
            document.colour = index
            redraw_3d_views()
        return
    if chip_id == star.LAST_COMMAND:
        from . import last_command

        last_command.run()
        return
    chrome._run(context, chip_id, False, chrome.layout_for(region, context), region)
    region.tag_redraw()


def _later(ident: str) -> None:
    """Open a panel the star names. The star has closed already, so it opens
    at once."""
    op = getattr(opkit.ops.rok, ident)
    if op.poll():
        op("INVOKE")


# --- drawing -------------------------------------------------------------------------------


def _recent_chips() -> tuple:
    """The star's Colours arm: the last three colours drawn with, named by
    their favourite name where they have one."""
    from . import palette_ui

    scene = document.scene
    if scene is None:
        return ()
    out = []
    for i in palette_ui.recent_colours():
        rgb = tuple(int(c) for c in scene.palette[i, :3])
        name = palette_ui.favourite_name(rgb) or f"Colour {i}"
        out.append(star.colour_chip(i, rgb, name))
    return tuple(out)


def _active_ids(context) -> set[str]:
    """The chips that show as on: the current tool and its mode."""
    wm = context.prefs
    tool = getattr(wm, "rok_tool", "ATTACH")
    mode_ = getattr(wm, "rok_mode", "VOXEL")
    on = {f"TOOL_{tool}", f"TOOL_{tool}_{mode_}"}
    on.add(f"{star.STAR_COLOUR}{int(getattr(document, 'colour', 0))}")  # the current colour's chip
    return on


def _with_alpha(colour, alpha: float):
    return (colour[0], colour[1], colour[2], colour[3] * alpha)


def _arm_strip(points, width_start: float, width_end: float, colour) -> None:
    """A tapered arm along `points`: quads that narrow from the centre out."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return
    seg = np.diff(pts, axis=0)
    lengths = np.hypot(seg[:, 0], seg[:, 1])
    keep = lengths > 1e-3
    pts = np.concatenate([pts[:1], pts[1:][keep]])
    if len(pts) < 2:
        return
    tangent = np.gradient(pts, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1) / np.maximum(norm, 1e-6)
    widths = np.linspace(width_start, width_end, len(pts))[:, None] / 2.0
    left = pts + normal * widths
    right = pts - normal * widths
    n = len(pts) - 1
    tris = np.empty((n, 6, 2), dtype=np.float32)
    tris[:, 0], tris[:, 1], tris[:, 2] = left[:-1], right[:-1], left[1:]
    tris[:, 3], tris[:, 4], tris[:, 5] = right[:-1], right[1:], left[1:]
    pen.tris(pen.to_3d(tris.reshape(-1, 2)), colour)


def draw(region, scale: float = 1.0) -> None:
    """The star: arms, chips, the centre pill. Reads `_session`."""
    session = _session
    if session is None:
        return
    context = CONTEXT
    layout = _layout(session, region)
    pose = session.pose
    if pose is not None:
        layout = _posed(layout, pose)
    s = layout.scale
    hovered = None if pose is not None else star.chip_at(layout, *session.pointer, session.open_ray)
    on = _active_ids(context)
    # Arms under the chips, in their family's colour; the rest fade while one is out.
    def colour_of(i: int):
        family = FAMILY.get(layout.rays[i].family, FAMILY[star.SCENE])
        if pose is not None and i in pose.tints:
            rgb, k = pose.tints[i]
            family = tuple(f + (c - f) * k for f, c in zip(family[:3], rgb)) + (family[3],)
        return family

    for i, arm in enumerate(layout.arms):
        family = colour_of(i)
        out = i == session.open_ray
        alpha = 0.8 if out else (0.5 if session.open_ray is None else 0.2)
        if pose is not None:
            alpha = 0.75
        _arm_strip(arm, (8.0 if out else 6.0) * s, 2.0 * s, _with_alpha(family, alpha))
    chips = sorted(layout.chips, key=lambda c: (c.ray == session.open_ray, c is hovered))
    for chip in chips:
        if chip.shown <= 0.02:
            continue
        fade = 1.0 if session.open_ray is None or chip.ray == session.open_ray else 0.4
        side = chip.side * (1.08 if chip is hovered else 1.0)
        x, y = chip.centre
        rect = (x - side / 2, y - side / 2, side, side)
        alpha = min(1.0, chip.shown) * fade
        family = colour_of(chip.ray)
        is_on = chip.chip.id in on
        radius = (10 if chip.origin else 8) * s
        round_rect(rect, radius, _with_alpha(CHIP_BG, alpha))
        if is_on:
            round_rect(rect, radius, _with_alpha(family, 0.28 * alpha))  # the current tool and mode
        # A tool's own chip: a heavy edge in its colour. The others a thin one.
        width = 3.2 if chip.origin else (1.8 if chip is hovered else 1.1)
        edge = TEXT if chip is hovered else family
        round_outline(rect, radius, _with_alpha(edge, alpha * (1.0 if chip.origin or chip is hovered else 0.75)), width * s)
        if chip.origin:
            inner = (rect[0] + 1.6 * s, rect[1] + 1.6 * s, rect[2] - 3.2 * s, rect[3] - 3.2 * s)
            round_outline(inner, radius - 1.6 * s, _with_alpha(edge, alpha), 1.6 * s)
        ink = family if is_on else (TEXT if chip is hovered or chip.origin else chrome.GLYPH)
        ink = _with_alpha(ink, alpha)
        if chip.chip.rgb is not None:
            # A colour chip: a swatch of the colour, ringed while it is the current one.
            inset = side * 0.18
            swatch = (rect[0] + inset, rect[1] + inset, rect[2] - 2 * inset, rect[3] - 2 * inset)
            round_rect(swatch, radius * 0.6, tuple(c / 255.0 for c in chip.chip.rgb) + (alpha,))
            if is_on:
                round_outline(swatch, radius * 0.6, _with_alpha(TEXT, alpha), 1.6 * s)
        elif chip.chip.glyph:
            chrome.draw_icon(chip.chip.glyph, rect, s, ink=ink, icon_side=side * 0.74, ring=False)
        else:
            text(chip.chip.text, x, y, 15 * s, ink, "centre")
    if pose is not None:
        _draw_play_extras(pose, layout, s)
        if pose.eye is not None:
            _draw_eye(layout.origin, pose.eye, s)
            return
    # The pill at the centre: what a pick would do.
    wm = context.prefs
    fallback = f"{kit.MODE_NAMES.get(getattr(wm, 'rok_mode', 'VOXEL'), '')}"
    label = star.centre_label(layout, hovered, fallback)
    ox, oy = layout.origin
    dot = FAMILY.get(layout.rays[hovered.ray].family, TEXT) if hovered else DOT_IDLE
    pen.fan(pen.to_3d(kit.circle_tris(ox, oy, 5.0 * s)), dot)
    if label and pose is None:
        w = text_width(label, 12 * s) + 20 * s
        px = min(max(ox - w / 2, _rail_edge(region) + 4 * s), region.width - w - 4 * s)  # whole, in the view
        pill = (px, max(4 * s, oy - 34 * s), w, 22 * s)
        round_rect(pill, 11 * s, PILL)
        text(label, px + w / 2, pill[1] + pill[3] / 2, 12 * s, TEXT if hovered else TEXT_DIM, "centre")


def _posed(layout: star.StarLayout, pose: star_play.Pose) -> star.StarLayout:
    """The star's layout with every chip and arm where the play has it."""
    from dataclasses import replace

    s = layout.scale
    chips = []
    for i, ray in enumerate(layout.rays):
        for k, chip in enumerate(ray.chips):
            if k >= len(pose.chips[i]):
                break
            origin = ray.first and k == 0
            side = star.CHIP * s * pose.sizes[i][k] * (star.ORIGIN if origin else 1.0)
            chips.append(star.PlacedChip(i, k, chip, pose.chips[i][k], side, pose.alphas[i][k], origin))
    arms = tuple(tuple(a) for a in pose.arms)
    return replace(layout, chips=tuple(chips), arms=arms)


DEEP_INK = (0.10, 0.14, 0.12, 0.94)  # the tentacles from the edges
DEEP_SUCKER = (0.55, 0.60, 0.50, 0.55)
SPARK = (1.0, 0.86, 0.55, 1.0)


def _draw_play_extras(pose: star_play.Pose, layout: star.StarLayout, s: float) -> None:
    for rect, a in pose.flashes:  # a rail button the drummers hit
        round_rect(rect, 8 * s, (0.91, 0.36, 0.23, a))
    for x, y in pose.cores:  # the two middles of a star that split
        pen.fan(pen.to_3d(kit.circle_tris(x, y, 6.0 * s)), (0.82, 0.80, 0.77, 1.0))
    for ch, x, y, size, a in pose.texts:
        text(ch, x, y, size * s, _with_alpha(TEXT, a), "centre")
    for x, y, r, a in pose.sparks:
        pen.fan(pen.to_3d(kit.circle_tris(x, y, r * s)), _with_alpha(SPARK, a))
    if pose.bulge is not None:
        x, y, r = pose.bulge
        pen.fan(pen.to_3d(kit.circle_tris(x, y, r * s)), FAMILY["ERASE"])


def _draw_eye(origin, eye, s: float) -> None:
    """The eye in the middle of the deep: it opens, looks about and blinks."""
    opened, lx, ly = eye
    ox, oy = origin
    w, h = 30 * s, 17 * s * max(opened, 0.04)
    lid = [(ox + w * math.cos(a), oy + h * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40)]
    pen.fan(pen.to_3d(lid), (0.92, 0.88, 0.70, 1.0))
    if opened > 0.25:
        px, py = ox + lx * w * 0.45, oy + ly * h * 0.4
        iris = [(px + 9 * s * math.cos(a), py + min(9 * s, h * 0.95) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 28)]
        pen.fan(pen.to_3d(iris), (0.55, 0.75, 0.25, 1.0))
        slit = [(px + 2.2 * s * math.cos(a), py + min(8 * s, h * 0.9) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 20)]
        pen.fan(pen.to_3d(slit), (0.02, 0.02, 0.02, 1.0))
    pen.lines(pen.to_3d([p for a, b in zip(lid, lid[1:]) for p in (a, b)]), (0.05, 0.05, 0.05, 1.0), 1.5 * s)


def _draw_deep(context) -> None:
    """Under the star: the dark of the deep and the tentacles from the edges."""
    session = _session
    region = context.region
    if session is None or session.pose is None or region is None:
        return
    pose = session.pose
    for wedge, rgba in pose.spots:  # disco lights
        pen.tris(pen.to_3d(wedge), rgba)
    if pose.darken > 0.0:
        w, h = float(region.width), float(region.height)
        quad = [(0, 0), (w, 0), (w, h), (0, 0), (w, h), (0, h)]
        pen.tris(pen.to_3d(quad), (0.0, 0.01, 0.02, pose.darken))
    for path, width in pose.tentacles:
        _arm_strip(path, width, 3.0, DEEP_INK)
        for k in range(3, len(path) - 4, 3):  # suckers along one side
            (x0, y0), (x1, y1) = path[k], path[k + 1]
            d = math.hypot(x1 - x0, y1 - y0) or 1.0
            off = width * (1 - k / len(path)) * 0.3
            cx, cy = x0 - (y1 - y0) / d * off, y0 + (x1 - x0) / d * off
            pen.fan(pen.to_3d(kit.circle_tris(cx, cy, max(1.2, width * 0.12 * (1 - k / len(path))))), DEEP_SUCKER)


def _draw(context) -> None:
    draw(context.region)


@register
class StarPickAction(Action):
    """View > Star: how the star picks. The setting is on the window manager;
    a menu-bar item needs an operator."""

    command_id = "rok.star_pick"
    title = "Star Pick"
    tooltip = "How the star (hold Space) picks a chip"
    pick = EnumProp(name="Pick", items=((RELEASE, "Pick on Release", ""), (CLICK, "Pick on Click", "")))

    def execute(self, context):
        context.prefs.rok_star_pick = self.pick
        return {"FINISHED"}


RokPrefs.add_property("rok_star_pick", EnumProp(
    name="Star",
    description="How the star (hold Space) picks a chip",
    items=(
        (RELEASE, "Pick on Release", "Hold Space, let go over a chip"),
        (CLICK, "Pick on Click", "Space opens the star; click a chip"),
    ),
    default=RELEASE,
    update=_write_pick,
))


def _load_pick(context) -> None:
    """Read the saved pick setting once the window is up (quietly: no write back)."""
    wm = context.prefs
    if wm is not None:
        object.__getattribute__(wm, "_values")["rok_star_pick"] = _read_pick()


registry.add_draw("star_deep", 595, "PIXEL", _draw_deep)
registry.add_draw("star", 600, "PIXEL", _draw)
registry.startup.append(_load_pick)
registry.keymap.append(("rok.marking", {"type": "SPACE", "value": "PRESS"}, None))
