"""The navigation cube, top right of the view, with Zoom, Pan and Centre.

The cube turns with the camera. A click on a face looks straight at it; a
click near an edge looks along the edge's two faces, near a corner from the
corner (each face is split 3 x 3). The view
glides there. A drag on the cube orbits, as Ctrl-drag does.

Below the cube are round buttons: three rows on the perspective view
(Centre and four views, Pan and Zoom, rendered view and Steady Hand),
and one row of three on a side view. Zoom: drag up or down, or click to
step in. Pan: drag. Centre: Frame All, then Frame Model, then Frame All again.

In a model, two more sit on the disc's rim: Section at the
top right and Mirror at the top left. A click opens three small X, Y and Z
buttons along the rim, away from the corner; each turns its section plane or
its mirror axis on or off, and the row stays open until a click elsewhere.
A drag from a section's X, Y or Z slides that plane along its axis: the
plane follows the pointer, one layer per voxel the pointer passes on screen.
At the window's edge the cursor jumps to the other side and the slide goes on.

When the mouse rests on a button, a tooltip under the buttons says what it does.

World axes are Z up; Front looks from -Y, Right from +X, Top from +Z.
Positions here are region pixels with the origin at the bottom left.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from . import gfx
from . import object as voxel_object
from .gfx import fonts, pen
from .handlers import registry
from .opkit import CONTEXT, Action, register
from .state import state

# --- look -------------------------------------------------------------------------------

CUBE_PX = 34.0  # half the cube's edge on screen
MARGIN = 26.0  # from the top right corner to the disc
DISC_R = 66.0  # the dark disc behind the cube
BUTTON_R = 15.0
BUTTON_GAP = 10.0
DISC = (0.11, 0.10, 0.10, 0.62)
DISC_EDGE = (1.0, 1.0, 1.0, 0.06)
FACE = np.array([0.30, 0.285, 0.27])
FACE_TOP = np.array([0.36, 0.345, 0.33])
EDGE_INK = (0.07, 0.065, 0.06, 1.0)
LABEL = (0.93, 0.91, 0.88)
EMBER = (0.91, 0.36, 0.23)
BUTTON = (0.11, 0.10, 0.10, 0.80)
BUTTON_HOVER = (0.22, 0.21, 0.20, 0.95)
GLYPH = (0.80, 0.78, 0.75, 1.0)
GLYPH_HOT = (1.0, 0.985, 0.96, 1.0)
GLIDE_S = 0.18
SPLIT = 0.5  # |u| or |v| past this on a face is its edge band
DRAG_PX = 3.0

# Faces: (normal, label, (u axis, v axis)); u and v span the face.
FACES = (
    ((0, 0, 1), "TOP", ((1, 0, 0), (0, 1, 0))),
    ((0, 0, -1), "BOTTOM", ((1, 0, 0), (0, -1, 0))),
    ((0, -1, 0), "FRONT", ((1, 0, 0), (0, 0, 1))),
    ((0, 1, 0), "BACK", ((-1, 0, 0), (0, 0, 1))),
    ((1, 0, 0), "RIGHT", ((0, 1, 0), (0, 0, 1))),
    ((-1, 0, 0), "LEFT", ((0, -1, 0), (0, 0, 1))),
)
_frame_model_next = False  # the Centre button's next click: the model (True) or every model (False)
BUTTONS = ("ZOOM", "PAN", "CENTRE", "QUAD", "RENDER", "STEADY")
# The perspective view's buttons, row by row under the cube; a side view keeps Zoom, Pan and Centre in one row.
GRID = (("CENTRE", "QUAD"), ("PAN", "ZOOM"), ("RENDER", "STEADY"))
# The rim buttons: (name, angle of the button, angle step of its X, Y, Z row), degrees.
RIM = (("SECTION", 45.0, -30.0), ("MIRROR", 135.0, 30.0))
AXIS_R = 11.0  # the X, Y and Z buttons' radius
ON = (0.91, 0.36, 0.23, 0.92)  # an axis that is on


@dataclass
class ShownFace:
    normal: np.ndarray
    label: str
    u: np.ndarray
    v: np.ndarray
    quad: np.ndarray  # (4, 2) region pixels
    centre: np.ndarray  # (2,)
    du: np.ndarray  # screen vector of +u (half the edge)
    dv: np.ndarray
    facing: float  # how squarely it faces the eye, 0..1
    depth: float


def _basis(view_state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The view's right, up and back axes in world space."""
    m = view_state.axes_matrix()
    return m[:, 0], m[:, 1], m[:, 2]


def _view_state(view_state):
    return view_state if view_state is not None else CONTEXT.view_state


def has_cube(view_state) -> bool:
    """The quad view's side views are locked: they show only the buttons."""
    return not _view_state(view_state).camera.locked


def home_centre(region, view_state=None) -> np.ndarray:
    """The cube's place: the top right corner of the view."""
    x, y, w, h = _view_state(view_state).area(region)
    return np.array([x + w - MARGIN - DISC_R, y + h - MARGIN - DISC_R])


def centre(region, view_state=None) -> np.ndarray:
    """The cube's middle: its place, or where an arm of the star's play shakes it."""
    from . import marking

    dx, dy, _gone = marking.ui_pose("NAV_CUBE")
    return home_centre(region, view_state) + np.array([dx, dy])


def faces(region, view_state) -> list[ShownFace]:
    """The faces that face the eye, far ones first, as drawn."""
    if not has_cube(view_state):
        return []
    right, up, back = _basis(view_state)
    c = centre(region, view_state)
    out = []
    for n, label, (u, v) in FACES:
        n, u, v = (np.asarray(a, dtype=np.float64) for a in (n, u, v))
        facing = float(n @ back)
        if facing <= 1e-3:
            continue
        to2 = lambda p: c + CUBE_PX * np.array([p @ right, p @ up])  # noqa: E731
        corners = [n + su * u + sv * v for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        quad = np.array([to2(p) for p in corners])
        out.append(ShownFace(n, label, u, v, quad, to2(n), CUBE_PX * np.array([u @ right, u @ up]),
                             CUBE_PX * np.array([v @ right, v @ up]), facing, facing))
    out.sort(key=lambda f: f.depth)
    return out


def pick(region, view_state, x: float, y: float):
    """What is under (x, y): ("FACE" | "EDGE" | "CORNER", direction, face, (iu, iv))
    with iu, iv in -1, 0, 1 for the face's 3 x 3 cell; or a button name; or None."""
    p = np.array([x, y], dtype=np.float64)
    for name, (bx, by, br) in rim_buttons(region, view_state).items():
        if (p[0] - bx) ** 2 + (p[1] - by) ** 2 <= (br + 2) ** 2:
            return name
    for name, (bx, by) in buttons(region, view_state).items():
        if (p[0] - bx) ** 2 + (p[1] - by) ** 2 <= (BUTTON_R + 2) ** 2:
            return name
    for f in reversed(faces(region, view_state)):  # the nearest face first
        m = np.column_stack([f.du, f.dv])
        if abs(np.linalg.det(m)) < 1e-6:
            continue
        a, b = np.linalg.solve(m, p - f.centre)
        if abs(a) > 1.0 or abs(b) > 1.0:
            continue
        iu = 0 if abs(a) <= SPLIT else int(np.sign(a))
        iv = 0 if abs(b) <= SPLIT else int(np.sign(b))
        d = f.normal + iu * f.u + iv * f.v
        kind = "FACE" if (iu, iv) == (0, 0) else ("CORNER" if iu and iv else "EDGE")
        return kind, d / np.linalg.norm(d), f, (iu, iv)
    if has_cube(view_state) and np.linalg.norm(p - centre(region, view_state)) <= DISC_R:
        return "DISC"
    return None


def buttons(region, view_state=None) -> dict[str, tuple[float, float]]:
    """The round buttons: three rows under the cube (GRID), or one row
    in the top right of a side view."""
    view_state = _view_state(view_state)
    c = centre(region, view_state)
    step = 2 * BUTTON_R + BUTTON_GAP
    x, _y, w, _h = view_state.area(region)
    if not has_cube(view_state):
        names = BUTTONS[:3]
        first = c[0] + DISC_R - BUTTON_R - (len(names) - 1) * step
        first = min(first, x + w - BUTTON_GAP - BUTTON_R - (len(names) - 1) * step)
        y = c[1] + DISC_R - BUTTON_R
        return {name: (first + k * step, y) for k, name in enumerate(names)}
    rows = GRID if view_state is CONTEXT.app.view_state else [BUTTONS[:3]]
    out = {}
    for r, names in enumerate(rows):
        y = c[1] - DISC_R - BUTTON_GAP - BUTTON_R - r * step
        first = c[0] - (len(names) - 1) * step / 2.0
        first = min(first, x + w - BUTTON_GAP - BUTTON_R - (len(names) - 1) * step)  # the row stays inside the view
        out.update({name: (first + k * step, y) for k, name in enumerate(names)})
    return out


_open: str | None = None  # the rim button whose X, Y, Z row is out


def has_rim(view_state) -> bool:
    """Section and Mirror: on the cube of the main view, inside a model."""
    view_state = _view_state(view_state)
    return has_cube(view_state) and view_state is CONTEXT.app.view_state and state.level in ("MODEL", "BRICK")


def rim_buttons(region, view_state=None) -> dict[str, tuple[float, float, float]]:
    """name -> (x, y, radius): Section and Mirror on the disc's rim, and the
    X, Y, Z row of the open one."""
    if not has_rim(view_state):
        return {}
    c = centre(region, view_state)
    out = {}
    for name, angle, step in RIM:
        a = math.radians(angle)
        out[name] = (c[0] + DISC_R * math.cos(a), c[1] + DISC_R * math.sin(a), BUTTON_R)
        if _open == name:
            for k, axis in enumerate("XYZ", start=1):
                b = math.radians(angle + k * step)
                out[f"{name}_{axis}"] = (c[0] + DISC_R * math.cos(b), c[1] + DISC_R * math.sin(b), AXIS_R)
    return out


def plane_preview():
    """("SECTION" | "MIRROR", axis) while the mouse rests on a rim X, Y or Z
    button, for the overlay's preview plane; else None."""
    if not isinstance(_hover, str) or _hover.count("_") != 1:
        return None
    name, axis = _hover.split("_")
    if name not in ("SECTION", "MIRROR") or axis not in "XYZ" or not axis or _open != name:
        return None
    return name, "XYZ".index(axis)


def axes_on(name: str) -> tuple[bool, bool, bool]:
    """Which section planes, or which mirror axes, are on."""
    if name == "SECTION":
        grid = voxel_object.document.grid
        sections = grid.sections if grid is not None else {}
        return tuple(k in sections for k in range(3))
    return tuple(bool(v) for v in getattr(CONTEXT.prefs, "rok_mirror", (False, False, False)))


def angles_of(direction) -> tuple[float, float]:
    """(yaw, pitch) of a camera that looks at the target from `direction`."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    pitch = math.asin(max(-1.0, min(1.0, d[2])))
    pitch = max(-math.radians(89.5), min(math.radians(89.5), pitch))  # straight down breaks look_at
    yaw = math.atan2(d[1], d[0]) if abs(d[0]) + abs(d[1]) > 1e-9 else None
    return yaw, pitch


# --- the glide --------------------------------------------------------------------------

_glide: dict | None = None


def glide_to(direction) -> None:
    """Turn the view to look from `direction`, eased over GLIDE_S."""
    global _glide
    cam = CONTEXT.view_state.camera
    yaw, pitch = angles_of(direction)
    if yaw is None:  # straight up or down: keep the turn about Z
        yaw = cam.yaw
    dy = (yaw - cam.yaw + math.pi) % (2 * math.pi) - math.pi  # the short way round
    timer = _glide["timer"] if _glide is not None else CONTEXT.prefs.add_timer(1.0 / 60.0)
    _glide = {"t0": time.monotonic(), "yaw0": cam.yaw, "pitch0": cam.pitch, "dyaw": dy, "pitch1": pitch,
              "timer": timer}
    _step()


def _step() -> None:
    global _glide
    g = _glide
    if g is None:
        return
    cam = CONTEXT.view_state.camera
    t = min(1.0, (time.monotonic() - g["t0"]) / GLIDE_S)
    k = 1.0 - (1.0 - t) ** 3  # ease out
    cam.yaw = g["yaw0"] + g["dyaw"] * k
    cam.pitch = g["pitch0"] + (g["pitch1"] - g["pitch0"]) * k
    CONTEXT.app.camera_moved()
    if t >= 1.0:
        CONTEXT.prefs.remove_timer(g["timer"])
        _glide = None


# --- the operator -------------------------------------------------------------------------

_hover = None  # what pick() gave last
_hover_since = 0.0  # when the mouse came to rest on it

# What each button does, shown when the mouse rests on it.
TIPS = {
    "ZOOM": "Zoom: drag up or down to zoom in or out; a click steps in.",
    "PAN": "Pan: drag to move the view sideways.",
    "CENTRE": "Frame: a click shows the whole scene; the next click frames the active model; then the whole scene again.",
    "QUAD": "Four views: top, front, right and the perspective view side by side; click again for one view.",
    "RENDER": "Rendered view: the perspective view with the path tracer's light and shadows; click again to turn it off.",
    "STEADY": "Steady Hand (H): the brush trails the mouse on a string, so a shaking hand draws a smooth line. Set the length in Rok > Settings.",
    "SECTION": "Section: opens X, Y and Z buttons that cut the model open along that axis. Drag an X, Y or Z button to move its section plane through the model.",
    "MIRROR": "Mirror: opens X, Y and Z buttons that repeat your edits on the other side of the grid's middle.",
    "SECTION_X": "Click: section plane X on or off. Drag this button to move the plane along X, one layer per voxel the pointer passes.",
    "SECTION_Y": "Click: section plane Y on or off. Drag this button to move the plane along Y, one layer per voxel the pointer passes.",
    "SECTION_Z": "Click: section plane Z on or off. Drag this button to move the plane along Z, one layer per voxel the pointer passes.",
    "MIRROR_X": "Mirror X on or off: each edit also happens at the mirrored place on the other side of the grid's middle along X.",
    "MIRROR_Y": "Mirror Y on or off: each edit also happens at the mirrored place on the other side of the grid's middle along Y.",
    "MIRROR_Z": "Mirror Z on or off: each edit also happens at the mirrored place on the other side of the grid's middle along Z.",
}
TIP_W = 230.0  # the widest a tooltip line goes, in pixels
TIP_FONT = 11.0
TIP_LINE = 15.0
TIP_PAD = 7.0


@register
class NavCubeAction(Action):
    """Always running, like the hover operator: sees every event and takes
    the ones on the cube and its buttons."""

    command_id = "rok.nav_cube"
    title = "Navigation Cube"

    def invoke(self, context, event):
        object.__setattr__(self, "_press", None)
        context.prefs.add_modal(self)
        return {"RUNNING", "PASS"}

    def modal(self, context, event):
        global _hover, _hover_since
        if event.type == "TIMER" and _glide is not None:
            _step()
            return {"PASS"}
        press = object.__getattribute__(self, "_press")
        if press is not None:
            return self._drag(context, event, press)
        if _hidden():
            return {"PASS"}  # no cube to pick
        if event.type == "MOUSE_MOVE":
            found = pick(context.region, context.view_state, event.fx, event.fy)
            if _key(found) != _key(_hover):
                _hover = found
                _hover_since = time.monotonic()
                if isinstance(found, str) and found in TIPS:
                    context.app.host.tick(_tip_delay() + 0.02)  # a mouse that rests sends no event
                context.app.request_redraw()
            if found is not None:
                state.hover = None  # the voxel under the cube is not the target
                return {"RUNNING"}
            return {"PASS"}
        if event.type == "LEFT_MOUSE" and event.value == "PRESS" and not event.ctrl:
            found = pick(context.region, context.view_state, event.fx, event.fy)
            if _open is not None and not (isinstance(found, str) and found.startswith(_open)):
                _close_row(context)  # a click elsewhere puts the X, Y, Z row away
            if found is None:
                return {"PASS"}
            object.__setattr__(self, "_press", {"what": found, "x": event.fx, "y": event.fy,
                                                "lx": event.fx, "ly": event.fy, "vx": event.fx, "vy": event.fy,
                                                "dragged": False})
            return {"RUNNING"}
        if event.type == "FOCUS_LOST" and _hover is not None:
            _hover = None
            context.app.request_redraw()
        return {"PASS"}

    def _drag(self, context, event, press):
        if event.type in ("MOUSE_MOVE", "MOUSE_DELTA"):
            if event.type == "MOUSE_DELTA":
                dx, dy = event.dx, event.dy
            elif press.get("relative"):
                return {"RUNNING"}  # the pointer holds still: the raw moves count
            else:
                dx, dy = event.fx - press["lx"], event.fy - press["ly"]
                press["lx"], press["ly"] = event.fx, event.fy
            press["vx"] += dx
            press["vy"] += dy
            if not press["dragged"] and math.hypot(press["vx"] - press["x"], press["vy"] - press["y"]) < DRAG_PX:
                return {"RUNNING"}
            if not press["dragged"] and context.window is not None:
                # A drag: the pointer hides and stays where it
                # was pressed, so the drag goes on at the screen's edge; it shows again there on release.
                press["relative"] = context.window.relative_set(True)
            press["dragged"] = True
            what = press["what"]
            cam = context.view_state.camera
            if what == "ZOOM":
                cam.zoom(math.exp(dy * 0.01))
            elif what == "PAN":
                cam.pan(dx, -dy)
            elif isinstance(what, str) and what.startswith("SECTION_"):
                _slide(context, press, event, what[-1])
            elif what != "CENTRE" and not _is_rim(what):
                cam.orbit(dx, -dy)  # as the Ctrl orbit
            context.app.camera_moved()
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            self._let_go(context, press)
            if not press["dragged"]:
                _click(context, press["what"])
            context.app.request_redraw()
            return {"RUNNING"}
        if event.type in ("ESCAPE", "FOCUS_LOST"):
            self._let_go(context, press)
            return {"RUNNING"}
        return {"RUNNING"}

    def _let_go(self, context, press) -> None:
        object.__setattr__(self, "_press", None)
        if press.get("relative") and context.window is not None:
            context.window.relative_set(False)


def _key(found):
    if isinstance(found, tuple):
        return (found[0], found[2].label, found[3])
    return found


SLIDE_MIN_PX = 6.0  # the fewest pixels per layer; an axis that points at the eye goes up the screen


def slide_step(context, axis: str) -> np.ndarray:
    """Region pixels the pointer moves for one layer along `axis`, at the model's middle."""
    cam = context.view_state.camera
    model = voxel_object.document.model
    k = "XYZ".index(axis)
    if model is not None:
        mid = np.asarray(voxel_object.model_origin(model), dtype=np.float64) + np.asarray(model.grid.size) / 2.0
    else:
        mid = np.asarray(cam.target, dtype=np.float64)
    span = float(model.grid.size[k]) if model is not None else 10.0
    unit = np.zeros(3)
    unit[k] = span
    a, b = cam.project(np.array([mid, mid + unit]))
    v = np.array([b[0] - a[0], a[1] - b[1]]) / span  # one layer; the region's y points up
    n = float(np.linalg.norm(v)) if np.all(np.isfinite(v)) else 0.0
    if n * span < 2 * SLIDE_MIN_PX:
        return np.array([0.0, SLIDE_MIN_PX])  # the axis points at the eye: up the screen
    # Far away a layer is a pixel or two: keep the direction, give each layer SLIDE_MIN_PX.
    return v if n >= SLIDE_MIN_PX else v / n * SLIDE_MIN_PX


def _slide(context, press, event, axis: str) -> None:
    """Move the section plane of `axis` with the pointer; a plane that is off comes on first."""
    from . import opkit

    grid = voxel_object.document.grid
    if grid is None:
        return
    k = "XYZ".index(axis)
    if "start" not in press:
        if k not in grid.sections:
            opkit.call("rok.section", axis=axis, action="TOGGLE")
        if k not in grid.sections:
            return
        press["start"] = grid.sections[k][0]
        press["step"] = slide_step(context, axis)
    step = press["step"]
    moved = np.array([press["vx"] - press["x"], press["vy"] - press["y"]])  # the drag's path, past any edge
    layers = int(round(float(moved @ step) / float(step @ step)))
    want = press["start"] + layers
    if grid.sections.get(k, (None,))[0] != min(max(want, 1), grid.size[k] - 1):
        opkit.call("rok.section", axis=axis, action="SET", coord=want)


def _is_rim(what) -> bool:
    return isinstance(what, str) and what.split("_")[0] in ("SECTION", "MIRROR")


def _close_row(context) -> None:
    global _open
    _open = None
    context.app.request_redraw()


def _click(context, what) -> None:
    global _open, _frame_model_next
    if _is_rim(what):
        name, _sep, axis = what.partition("_")
        if not axis:
            _open = None if _open == name else name
        elif name == "SECTION":
            from . import opkit

            opkit.call("rok.section", axis=axis, action="TOGGLE")
        else:
            from . import opkit

            opkit.call("rok.toggle_mirror", axis="XYZ".index(axis))
        context.app.set_menu()  # the View menu's check marks
        context.app.request_redraw()
        return
    if what == "QUAD":
        if not context.app.quad:
            from . import render_view

            render_view.set_rendered(context.app, False)  # the rendered view shows one view
        context.app.set_quad(not context.app.quad)
        context.app.set_menu()
        return
    if what == "STEADY":
        from . import opkit

        opkit.call("rok.steady_hand")
        return
    if what == "RENDER":
        from . import render_view

        render_view.set_rendered(context.app, not render_view.rendered(context.app))
        return
    if what == "ZOOM":
        context.view_state.camera.zoom(1.25)
        context.app.camera_moved()
    elif what == "CENTRE":
        # Frame All, then Frame Model, then Frame All again.
        model = voxel_object.document.model
        if _frame_model_next and model is not None:
            voxel_object.frame_model(context, model)
        else:
            voxel_object.frame_scene(context)
        _frame_model_next = not _frame_model_next
    elif isinstance(what, tuple):
        glide_to(what[1])


def _start(context) -> None:
    from . import opkit

    opkit.call("rok.nav_cube", "INVOKE")


# Above the hover operator, whichever module was imported first: the
# newest modal sees an event first, and the cube must see its clicks before
# the hover starts a stroke. The app runs `late` startup functions last.
_start.late = True


# --- drawing ------------------------------------------------------------------------------


def _hidden() -> bool:
    """The shader panel and the View Settings panel stand where the cube is,
    so the cube makes room while one is open."""
    from . import marking, shader_panel, view_panel

    return shader_panel.active() or view_panel.active() or marking.ui_pose("NAV_CUBE")[2]


def _draw_back(context) -> None:
    """The discs and the round buttons of every view: rects, under the cubes' triangles."""
    if _hidden():
        return
    region = context.region
    for view_state in context.app.views:
        hot_view = view_state is context.view_state
        if has_cube(view_state):
            c = centre(region, view_state)
            r = DISC_R
            gfx.rrect((c[0] - r, c[1] - r, 2 * r, 2 * r), DISC, r)
            gfx.rrect((c[0] - r, c[1] - r, 2 * r, 2 * r), DISC_EDGE, r, stroke=1.0)
        for name, (bx, by) in buttons(region, view_state).items():
            hot = hot_view and _hover == name
            gfx.rrect((bx - BUTTON_R, by - BUTTON_R, 2 * BUTTON_R, 2 * BUTTON_R), BUTTON_HOVER if hot else BUTTON, BUTTON_R)
        for name, (bx, by, br) in rim_buttons(region, view_state).items():
            hot = hot_view and _hover == name
            parent, _sep, axis = name.partition("_")
            on = bool(axis) and axes_on(parent)["XYZ".index(axis)]
            fill = ON if on else (BUTTON_HOVER if hot or _open == name else BUTTON)
            gfx.rrect((bx - br, by - br, 2 * br, 2 * br), fill, br)


def _draw_front(context) -> None:
    if _hidden():
        return
    for view_state in context.app.views:
        _draw_front_of(context, view_state)


def _draw_front_of(context, view_state) -> None:
    """The cube, its labels, the hovered part and the button glyphs of one view."""
    region = context.region
    hot_view = view_state is context.view_state
    hover = _hover if isinstance(_hover, tuple) and hot_view else None
    for f in faces(region, view_state):
        shade = 0.72 + 0.28 * f.facing
        base = (FACE_TOP if f.normal[2] > 0.5 else FACE) * shade
        pen.fan(f.quad, tuple(base) + (0.97,))
        if hover is not None and hover[2].label == f.label:
            iu, iv = hover[3]
            lo = [(-1.0, -SPLIT), (-SPLIT, SPLIT), (SPLIT, 1.0)]
            a0, a1 = lo[iu + 1]
            b0, b1 = lo[iv + 1]
            cell = np.array([f.centre + a * f.du + b * f.dv for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))])
            pen.fan(cell, EMBER + (0.85,))
        q = f.quad
        edges = np.array([q[0], q[1], q[1], q[2], q[2], q[3], q[3], q[0]])
        pen.lines(edges, EDGE_INK, 1.5)
        if f.facing > 0.45:
            size = 9.0
            fonts.size(0, size)
            w, h = fonts.dimensions(0, f.label)
            alpha = min(1.0, (f.facing - 0.45) / 0.3)
            fonts.color(0, *LABEL, alpha)
            fonts.position(0, float(f.centre[0] - w / 2), float(f.centre[1] - h / 2 + 3.0), 0)
            fonts.draw(0, f.label)
    _draw_axes(region, view_state)
    for name, (bx, by) in buttons(region, view_state).items():
        on = (
            (name == "QUAD" and context.app.quad)
            or (name == "RENDER" and getattr(context.app, "rendered", False))
            or (name == "STEADY" and bool(CONTEXT.prefs.rok_steady_hand))
        )
        ink = GLYPH_HOT if (hot_view and _hover == name) or on else GLYPH
        _glyph(name, bx, by, ink)
    for name, (bx, by, _br) in rim_buttons(region, view_state).items():
        parent, _sep, axis = name.partition("_")
        hot = hot_view and _hover == name
        if axis:
            ink = GLYPH_HOT if hot or axes_on(parent)["XYZ".index(axis)] else GLYPH
            fonts.size(0, 11.0)
            w, h = fonts.dimensions(0, axis)
            fonts.color(0, *ink)
            fonts.position(0, float(bx - w / 2), float(by - h / 2), 0)
            fonts.draw(0, axis)
        else:
            # Ember ink: some plane or axis is on.
            ink = GLYPH_HOT if hot else (EMBER + (1.0,) if any(axes_on(name)) else GLYPH)
            _glyph(name, bx, by, ink)
    if hot_view:
        _draw_tip(region, view_state)


# The axis letters on the cube's edges: each of X, Y and Z on the nearest
# edge that runs along it. View > Show Axis Letters on Nav Cube (on by default).
LETTERS_SETTING = "cube_axis_letters"


def letters_on() -> bool:
    from . import settings

    return settings.get(LETTERS_SETTING) != "off"


@register
class CubeAxisLettersAction(Action):
    command_id = "rok.cube_axis_letters"
    title = "Show Axis Letters on Nav Cube"
    tooltip = "X, Y and Z on the edges of the navigation cube"

    def execute(self, context):
        from . import settings

        settings.put(LETTERS_SETTING, "off" if letters_on() else "on")
        context.app.set_menu()
        context.app.request_redraw()
        return {"FINISHED"}


def edge_letters(region, view_state) -> list[tuple[str, int, tuple[float, float]]]:
    """(letter, axis index, position) of X, Y and Z: each just outside the
    middle of the cube edge that runs along it nearest the eye. None for an
    axis that points at the eye: no edge shows it."""
    if not has_cube(view_state):
        return []
    c = centre(region, view_state)
    right, up, back = _basis(view_state)
    out = []
    for k in range(3):
        axis = np.eye(3)[k]
        if float(np.hypot(axis @ right, axis @ up)) < 0.3:
            continue
        i, j = [a for a in range(3) if a != k]
        best = None
        for si in (-1, 1):
            for sj in (-1, 1):
                m = np.zeros(3)
                m[i], m[j] = si, sj
                near = float(m @ back)  # toward the eye
                if best is None or near > best[0]:
                    best = (near, m)
        m = best[1]
        mid = c + CUBE_PX * np.array([m @ right, m @ up])
        away = mid - c
        n = float(np.linalg.norm(away))
        out.append(("XYZ"[k], k, tuple(mid + (away / n * 6.0 if n > 1e-9 else 0.0))))
    return out


def _draw_axes(region, view_state) -> None:
    if not letters_on():
        return
    from rok.core.gizmo import AXIS_COLOURS

    from . import theme

    bright = theme.is_bright()
    fonts.size(0, 10.0 if bright else 9.0)
    for letter, k, (x, y) in edge_letters(region, view_state):
        rgb = AXIS_COLOURS[k]
        if bright:  # on a light ground: a deeper tone, fully opaque
            rgb = tuple(c * 0.6 for c in rgb)
        w, h = fonts.dimensions(0, letter)
        fonts.color(0, *rgb, 1.0 if bright else 0.85)
        fonts.position(0, float(x - w / 2), float(y - h / 2), 0)
        fonts.draw(0, letter)


def _tip_delay() -> float:
    from rok.core import chrome as kit

    return kit.TOOLTIP_DELAY_S


def tip_lines(text: str, width_of, width: float = TIP_W) -> list[str]:
    """A tooltip's words in lines no wider than `width`."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if width_of(trial) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def tip_rect(region, view_state, lines: int, width: float) -> tuple[float, float, float, float]:
    """Where a button's tooltip goes: under the row of buttons, its right edge at
    the row's right end, kept inside the view."""
    row = buttons(region, view_state)
    right = max(x for x, _y in row.values()) + BUTTON_R
    below = min(y for _x, y in row.values()) - BUTTON_R - 8.0
    w = width + 2 * TIP_PAD
    h = lines * TIP_LINE + 2 * TIP_PAD
    vx, vy, vw, _vh = view_state.area(region)
    x = min(max(vx + 4.0, right - w), vx + vw - 4.0 - w)
    y = max(vy + 4.0, below - h)
    return (x, y, w, h)


def _draw_tip(region, view_state) -> None:
    """The tooltip of the button the mouse rests on, once it is due."""
    if not isinstance(_hover, str) or _hover not in TIPS:
        return
    if time.monotonic() - _hover_since < _tip_delay():
        return
    fonts.size(0, TIP_FONT)
    lines = tip_lines(TIPS[_hover], lambda t: fonts.dimensions(0, t)[0])
    width = max(fonts.dimensions(0, line)[0] for line in lines)
    x, y, w, h = tip_rect(region, view_state, len(lines), width)
    gfx.rrect((x, y, w, h), BUTTON_HOVER[:3] + (0.97,), 6.0)
    gfx.rrect((x, y, w, h), DISC_EDGE, 6.0, stroke=1.0)
    fonts.color(0, *GLYPH_HOT)
    for k, line in enumerate(lines):
        _w, th = fonts.dimensions(0, line)
        ly = y + h - TIP_PAD - (k + 0.5) * TIP_LINE - th / 2
        fonts.position(0, float(x + TIP_PAD), float(ly), 0)
        fonts.draw(0, line)


def _circle(cx, cy, r, n=20) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n + 1)
    pts = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    return np.repeat(pts, 2, axis=0)[1:-1]


_glyph_cache: dict[tuple, list] = {}


def _glyph(name: str, x: float, y: float, ink) -> None:
    """A button's glyph in `ink`. The geometry is built once per place: every
    frame draws the seven glyphs, and the numpy work added up."""
    key = (name, round(x, 3), round(y, 3))
    parts = _glyph_cache.get(key)
    if parts is None:
        parts = _glyph_cache[key] = _glyph_parts(name, x, y)
        if len(_glyph_cache) > 512:  # a resize moves the buttons; the old places go
            for old in list(_glyph_cache)[:256]:
                del _glyph_cache[old]
    for kind, points, width in parts:
        if kind == "lines":
            pen.lines(points, ink, width)
        elif kind == "fan":
            pen.fan(points, ink)
        else:
            gfx.rrect(points, ink, width)


def _glyph_parts(name: str, x: float, y: float) -> list:
    """("lines", points, width) | ("fan", points, None) | ("rrect", rect, radius)."""
    out = []
    lines = lambda pts, w: out.append(("lines", np.asarray(pts, dtype=np.float64), w))  # noqa: E731
    fan = lambda pts: out.append(("fan", np.asarray(pts, dtype=np.float64), None))  # noqa: E731
    rrect = lambda rect, r: out.append(("rrect", rect, r))  # noqa: E731
    if name == "ZOOM":  # a magnifier
        lines(_circle(x - 2, y + 2, 5.5), 1.6)
        lines([[x + 2, y - 2], [x + 7, y - 7]], 2.2)
    elif name == "PAN":  # four arrows
        s, t = 7.5, 2.8
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tip = np.array([x + dx * s, y + dy * s])
            lines([[x, y], tip], 1.5)
            side = np.array([-dy, dx])
            back = tip - np.array([dx, dy]) * t
            lines([tip, back + side * t, tip, back - side * t], 1.5)
    elif name == "SECTION":  # a box cut by a plane: the part beyond it filled
        rrect((x - 5.5, y - 5.5, 11.0, 5.5), 1.0)
        lines([[x - 5.5, y - 5.5], [x - 5.5, y + 5.5], [x - 5.5, y + 5.5], [x + 5.5, y + 5.5],
               [x + 5.5, y + 5.5], [x + 5.5, y - 5.5]], 1.4)
        lines([[x - 9.0, y], [x + 9.0, y]], 1.6)
    elif name == "MIRROR":  # two triangles facing a dashed line
        for sx in (-1, 1):
            tri = np.array([[x + sx * 2.5, y], [x + sx * 8.5, y + 5.0], [x + sx * 8.5, y - 5.0]])
            if sx < 0:
                fan(tri)
            else:
                lines(np.r_[tri, tri[:1]].repeat(2, axis=0)[1:-1], 1.4)
        for k in range(4):
            lines([[x, y + 7.5 - k * 4.5], [x, y + 5.0 - k * 4.5]], 1.4)
    elif name == "RENDER":  # a sun: the rendered view
        fan(_circle(x, y, 3.6, 16)[::2])
        for k in range(8):
            a = k * math.pi / 4
            c, s = math.cos(a), math.sin(a)
            lines([[x + c * 5.6, y + s * 5.6], [x + c * 8.4, y + s * 8.4]], 1.5)
    elif name == "STEADY":  # a ring on a string, the mouse's dot at the far end
        lines(_circle(x - 5.5, y - 3.0, 3.0), 1.5)
        lines([[x - 2.6, y - 1.6], [x - 1.0, y + 1.0], [x - 1.0, y + 1.0], [x + 1.5, y + 4.0],
               [x + 1.5, y + 4.0], [x + 4.0, y + 3.0]], 1.4)
        fan(_circle(x + 5.5, y + 3.2, 2.0, 12)[::2])
    elif name == "QUAD":  # four panes
        for dx, dy in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
            cx, cy = x + dx * 3.6, y + dy * 3.6
            rrect((cx - 2.8, cy - 2.8, 5.6, 5.6), 1.2)
    else:  # a target
        lines(_circle(x, y, 6.0), 1.5)
        fan(_circle(x, y, 1.8, 12)[::2])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            lines([[x + dx * 8.5, y + dy * 8.5], [x + dx * 4.5, y + dy * 4.5]], 1.5)
    return out


registry.add_draw("nav_back", 515, "PIXEL", _draw_back)
registry.add_draw("nav_front", 516, "PIXEL", _draw_front)
registry.startup.append(_start)
