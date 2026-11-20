"""The light dome of the View Settings panel: drag a light to move it.

The maths is rok/core/light_dome.py. A press on a light's handle picks it
and drags it (around = turn, in and out = height); a press on the empty disc
moves the picked light there; a press on the ring turns every light together
(Turn All). Shift snaps to 15 degrees of turn and 5 of height. The drag is
one undo step on release, as a slider's is.
"""

from __future__ import annotations

import math

from rok.core import light as light_core
from rok.core import light_dome as kit

from .gfx import pen
from .paint import round_outline, round_rect, text

DISC = (0.0, 0.0, 0.0, 0.30)
GUIDE = (1.0, 1.0, 1.0, 0.08)
RING = (1.0, 1.0, 1.0, 0.07)
RING_LIT = (1.0, 1.0, 1.0, 0.16)
EDGE = (0.228, 0.212, 0.196, 1.0)
EMBER = (0.910, 0.365, 0.227, 1.0)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_FAINT = (0.450, 0.430, 0.410, 1.0)
DARK = (0.06, 0.056, 0.053, 1.0)


def camera_azimuth(app) -> float:
    cam = getattr(app, "camera", None)
    return math.degrees(float(getattr(cam, "yaw", 0.0)))


def _points(dome: kit.Dome, setup: light_core.LightSetup, camera: float) -> list[tuple[float, float]]:
    return [kit.point(dome, light.azimuth + setup.turn, light.elevation, camera) for light in setup.lights]


def press(session, app, item, mx: float, my: float) -> bool:
    """A press on the dome. True when it started a drag."""
    from . import view_panel as panel

    setup = panel._setup()
    dome = kit.dome_in(item.rect)
    camera = camera_azimuth(app)
    k = kit.nearest(_points(dome, setup, camera), mx, my, kit.GRAB * dome.scale)
    if k is not None:
        session.picked = k
        panel._begin(session)
        session.dome_drag = ("LIGHT", k)
        return True
    if kit.on_ring(dome, mx, my):
        panel._begin(session)
        session.dome_drag = ("RING", kit.angle_at(dome, mx, my), setup.turn)
        return True
    if kit.on_disc(dome, mx, my) and setup.lights:
        k = panel._picked(session, setup)
        panel._begin(session)
        session.dome_drag = ("LIGHT", k)
        _move_light(session, app, dome, k, mx, my, False)
        return True
    return False


def _move_light(session, app, dome: kit.Dome, k: int, mx: float, my: float, snap: bool) -> None:
    from . import view_panel as panel

    setup = panel._setup()
    if not 0 <= k < len(setup.lights):
        return
    azimuth, elevation = kit.light_at(dome, mx, my, camera_azimuth(app))
    azimuth -= setup.turn
    if snap:
        azimuth, elevation = 15.0 * round(azimuth / 15.0), 5.0 * round(elevation / 5.0)
    panel._show(app, light_core.with_light(setup, k, azimuth=light_core.wrap(azimuth),
                                           elevation=round(elevation, 1)))


def drag_event(session, app, layout, event, mx: float, my: float) -> None:
    """An event while a dome drag runs: a move moves, the release ends it as one undo step."""
    from . import view_panel as panel

    item = layout.find("DOME")
    drag = session.dome_drag
    if item is None or drag is None:
        session.dome_drag = None
        return
    dome = kit.dome_in(item.rect)
    if event.type in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
        if drag[0] == "LIGHT":
            _move_light(session, app, dome, drag[1], mx, my, bool(event.shift))
        else:
            _kind, start_angle, start_turn = drag
            turn = start_turn + kit.angle_at(dome, mx, my) - start_angle
            if event.shift:
                turn = 15.0 * round(turn / 15.0)
            panel._show(app, _with_turn(panel._setup(), turn))
    elif event.type == "LEFT_MOUSE" and event.value == "RELEASE":
        session.dome_drag = None
        panel._commit(session, app, "Turn All" if drag[0] == "RING" else "Move Light")


def _with_turn(setup: light_core.LightSetup, turn: float) -> light_core.LightSetup:
    from dataclasses import replace

    return replace(setup, turn=round(light_core.wrap(turn), 1))


def draw(session, app, rect, scale: float, mouse) -> None:
    from . import view_panel as panel

    s = scale
    setup = panel._setup()
    dome = kit.dome_in(rect, s)
    camera = camera_azimuth(app)
    cx, cy, r = dome.cx, dome.cy, dome.radius
    round_rect((cx - r, cy - r, 2 * r, 2 * r), r, DISC)
    # The guides: 30 and 60 degrees high, and the view's cross.
    guides = kit.circle(cx, cy, r / 3) + kit.circle(cx, cy, 2 * r / 3) + kit.circle(cx, cy, r)
    guides += [(cx - r, cy), (cx + r, cy), (cx, cy - r), (cx, cy + r)]
    pen.lines(pen.to_3d(guides), GUIDE, 1.0 * s)
    # The ring: a drag on it turns every light.
    mid = (dome.ring_inner + dome.ring_outer) / 2
    ring_lit = (session.dome_drag is not None and session.dome_drag[0] == "RING") or (
        mouse is not None and kit.on_ring(dome, *mouse))
    pen.lines(pen.to_3d(kit.circle(cx, cy, mid, 96)), RING_LIT if ring_lit else RING, kit.RING * s)
    ticks = []
    for k in range(12):
        a = math.radians(kit.screen_angle(30.0 * k + setup.turn, camera))  # they turn with Turn All
        ticks += [(cx + dome.ring_inner * math.cos(a), cy + dome.ring_inner * math.sin(a)),
                  (cx + (dome.ring_inner + 4 * s) * math.cos(a), cy + (dome.ring_inner + 4 * s) * math.sin(a))]
    pen.lines(pen.to_3d(ticks), TEXT_FAINT, 1.0 * s)
    # The model from above, turned as the view is, and the camera at the bottom.
    _footprint(cx, cy, r * 0.28, camera, s)
    eye_y = cy - r + 10 * s
    pen.lines(pen.to_3d([(cx - 5 * s, eye_y - 3 * s), (cx, eye_y + 2 * s), (cx, eye_y + 2 * s),
                         (cx + 5 * s, eye_y - 3 * s)]), TEXT_FAINT, 1.5 * s)
    # The lights; the picked one on top.
    points = _points(dome, setup, camera)
    picked = panel._picked(session, setup)
    order = [k for k in range(len(points)) if k != picked] + ([picked] if points else [])
    hover = kit.nearest(points, *mouse, kit.GRAB * s) if mouse is not None else None
    h = kit.HANDLE * s
    for k in order:
        light = setup.lights[k]
        x, y = points[k]
        alpha = 1.0 if light.on else 0.35
        colour = tuple(c / 255.0 for c in light_core.srgb255(light.colour)) + (alpha,)
        if light.kind == light_core.LAMP:
            round_outline((x - h - 3 * s, y - h - 3 * s, 2 * h + 6 * s, 2 * h + 6 * s), h + 3 * s,
                          colour, 1.5 * s)
        round_rect((x - h, y - h, 2 * h, 2 * h), h, colour)
        ring = EMBER if k == picked else (TEXT if k == hover else DARK)
        round_outline((x - h, y - h, 2 * h, 2 * h), h, ring, (2.0 if k in (picked, hover) else 1.0) * s)
        text(str(k + 1), x, y, 9 * s, DARK, "centre")
        if light.elevation < 0:  # below the horizon: a mark outside the rim
            a = math.atan2(y - cy, x - cx)
            pen.lines(pen.to_3d([(cx + (r + 1 * s) * math.cos(a), cy + (r + 1 * s) * math.sin(a)),
                                 (cx + (r + 3 * s) * math.cos(a), cy + (r + 3 * s) * math.sin(a))]), EMBER, 2 * s)
    # The hints, left and right of the disc.
    x0, y0, w, hh = rect
    left = x0
    for k, line in enumerate(("Drag a light:", "around turns,", "in and out raises.")):
        text(line, left, y0 + hh - (12 + 14 * k) * s, 10 * s, TEXT_FAINT)
    for k, line in enumerate(("The ring turns", "all lights.", "Shift: steps.")):
        text(line, x0 + w, y0 + (40 - 14 * k) * s, 10 * s, TEXT_FAINT, "right")


def _footprint(cx: float, cy: float, size: float, camera: float, s: float) -> None:
    """The scene's box seen from above, turned as the view is."""
    from .object import document

    scene = document.scene
    if scene is None or not scene.models:
        return
    from . import render_view

    lo, hi = render_view._scene_box(scene)
    w, d = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
    k = size / max(w, d, 1)
    turn = math.radians(-90.0 - camera)
    c, sn = math.cos(turn), math.sin(turn)
    corners = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    pts = [(cx + k * (x * c - y * sn), cy + k * (x * sn + y * c)) for x, y in corners]
    segs = []
    for a, b in zip(pts, pts[1:] + pts[:1]):
        segs += [a, b]
    pen.lines(pen.to_3d(segs), TEXT_FAINT, 1.5 * s)
