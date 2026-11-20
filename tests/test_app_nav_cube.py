"""The navigation cube (rok/app/nav_cube.py) against a host that only records."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from rok.app import nav_cube
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.opkit import CONTEXT
from rok.app.state import state

FIXTURES = Path(__file__).parent / "fixtures"
W, H = 1440.0, 900.0


class RecordingHost:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "measure_text":
                return (len(args[0]) * 7.0, 16.0)
            return None

        return record


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def app(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    clock = Clock()
    monkeypatch.setattr(nav_cube.time, "monotonic", clock)
    document.reset()
    state.__init__()
    nav_cube._frame_model_next = False
    a = RokApp(RecordingHost(), str(FIXTURES / "bird.vox"))
    a.handle("start", {"width": W, "height": H, "scale": 2.0})
    a.clock = clock
    yield a
    nav_cube._glide = None
    nav_cube._hover = None
    nav_cube._open = None
    document.reset()
    state.__init__()


def host_xy(app, x: float, y: float) -> dict:
    """Region pixels (origin bottom left) to the host's (origin top left)."""
    return {"x": float(x), "y": float(app.region.height - y)}


def click(app, x: float, y: float) -> None:
    app.handle("mouse_move", host_xy(app, x, y))
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def finish_glide(app) -> None:
    app.clock.t += 1.0
    app.handle("tick", {})


def look_from(app, direction) -> None:
    yaw, pitch = nav_cube.angles_of(direction)
    app.camera.yaw, app.camera.pitch = yaw if yaw is not None else 0.0, pitch
    app.camera_moved()


def test_seen_from_the_front_the_front_face_sits_in_the_middle(app):
    look_from(app, (0, -1, 0))
    c = nav_cube.centre(app.region)
    kind, d, face, _cell = nav_cube.pick(app.region, app.view_state, c[0], c[1])
    assert kind == "FACE" and face.label == "FRONT" and np.allclose(d, (0, -1, 0))


def test_a_click_on_the_top_face_glides_the_view_straight_down(app):
    c = nav_cube.centre(app.region)
    found = None
    for dy in np.linspace(0, nav_cube.CUBE_PX, 30):
        found = nav_cube.pick(app.region, app.view_state, c[0], c[1] + dy)
        if isinstance(found, tuple) and found[2].label == "TOP" and found[0] == "FACE":
            break
    assert found[2].label == "TOP"
    face_c = found[2].centre
    click(app, *face_c)
    assert app.timers  # the glide runs on a timer
    finish_glide(app)
    assert app.camera.pitch == pytest.approx(math.radians(89.5))
    assert not app.timers


def test_a_click_near_a_corner_looks_from_that_corner(app):
    look_from(app, (0, -1, 0))
    face = next(f for f in nav_cube.faces(app.region, app.view_state) if f.label == "FRONT")
    p = face.centre + 0.85 * face.du + 0.85 * face.dv  # the top right corner of the front face
    kind, d, _f, cell = nav_cube.pick(app.region, app.view_state, *p)
    assert kind == "CORNER" and cell == (1, 1)
    click(app, *p)
    finish_glide(app)
    eye = app.camera.eye() - app.camera.target
    assert np.allclose(eye / np.linalg.norm(eye), np.array([1, -1, 1]) / math.sqrt(3), atol=1e-6)


def test_a_click_on_the_cube_never_reaches_the_voxels(app):
    before = document.grid.voxels.copy()
    c = nav_cube.centre(app.region)
    click(app, c[0], c[1])
    assert np.array_equal(document.grid.voxels, before)
    assert not document.undo.can_undo


def test_hovering_the_cube_lights_the_cell_and_hides_the_voxel_hover(app):
    c = nav_cube.centre(app.region)
    app.handle("mouse_move", host_xy(app, c[0], c[1]))
    assert isinstance(nav_cube._hover, tuple) and state.hover is None


def test_dragging_the_cube_orbits(app):
    c = nav_cube.centre(app.region)
    yaw = app.camera.yaw
    app.handle("mouse_move", host_xy(app, c[0], c[1]))
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", host_xy(app, c[0] + 40, c[1]))
    app.handle("mouse_up", {"button": "left"})
    assert app.camera.yaw != yaw
    assert nav_cube._glide is None  # a drag is no click


def test_dragging_zoom_up_moves_closer_and_a_click_steps_in(app):
    bx, by = nav_cube.buttons(app.region, app.view_state)["ZOOM"]
    d = app.camera.distance
    app.handle("mouse_move", host_xy(app, bx, by))
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", host_xy(app, bx, by + 60))
    app.handle("mouse_up", {"button": "left"})
    assert app.camera.distance < d
    d = app.camera.distance
    click(app, bx, by)
    assert app.camera.distance == pytest.approx(d / 1.25)


def test_dragging_pan_moves_the_target(app):
    bx, by = nav_cube.buttons(app.region, app.view_state)["PAN"]
    t = app.camera.target.copy()
    app.handle("mouse_move", host_xy(app, bx, by))
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", host_xy(app, bx + 50, by))
    app.handle("mouse_up", {"button": "left"})
    assert not np.allclose(app.camera.target, t)


def test_centre_frames_the_model_again(app):
    app.camera.target = np.array([500.0, 500.0, 500.0])
    app.camera_moved()
    bx, by = nav_cube.buttons(app.region, app.view_state)["CENTRE"]
    click(app, bx, by)
    lo = np.asarray(document.model.position, dtype=float)
    mid = lo + np.asarray(document.grid.size) / 2.0
    assert np.allclose(app.camera.target, mid)


def test_the_cube_and_its_buttons_are_drawn(app):
    app.handle("mouse_move", host_xy(app, *nav_cube.buttons(app.region, app.view_state)["PAN"]))
    app.redraw()
    sent = {c[1][0]: c for c in app.host.calls if c[0] == "set_draw"}
    front = sent["nav_front"][2]
    assert len(front["tris2"]) > 0 and len(front["lines2"]) > 0
    assert any(t[4] in ("TOP", "FRONT", "RIGHT", "LEFT", "BACK", "BOTTOM") for t in front["texts"])
    assert len(sent["nav_back"][2]["rects"]) >= 4 * 10  # the disc, its edge and three buttons


@pytest.mark.parametrize("pitch_deg", [10.0, 35.0, 70.0])
def test_centre_fits_the_whole_model_into_the_view(app, pitch_deg):
    from rok.app.object import model_origin

    app.camera.pitch = math.radians(pitch_deg)
    app.camera_moved()
    bx, by = nav_cube.buttons(app.region, app.view_state)["CENTRE"]
    click(app, bx, by)
    m = document.model
    lo = np.asarray(model_origin(m), dtype=float)
    hi = lo + np.asarray(m.grid.size, dtype=float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    p = app.camera.project(corners)
    assert np.all((p[:, 0] >= 0) & (p[:, 0] <= W) & (p[:, 1] >= 0) & (p[:, 1] <= H))
    # Not lost in the distance either: the box spans most of the room one way
    # (the view lifts the model clear of the bottom; the room is the smaller half twice).
    room = H - 2 * app.camera.lift
    span = max(np.ptp(p[:, 0]) / W, np.ptp(p[:, 1]) / room)
    assert span > 0.7


# --- Section and Mirror on the rim ---------------------------------------------------------


def rim(app, name):
    x, y, _r = nav_cube.rim_buttons(app.region)[name]
    return x, y


def test_section_and_mirror_sit_on_the_rim_top_right_and_top_left(app):
    c = nav_cube.centre(app.region)
    sx, sy = rim(app, "SECTION")
    mx, my = rim(app, "MIRROR")
    assert sx > c[0] and sy > c[1] and mx < c[0] and my > c[1]
    assert set(nav_cube.rim_buttons(app.region)) == {"SECTION", "MIRROR"}  # the rows are closed
    assert sx + nav_cube.BUTTON_R <= app.region.width  # inside the view
    state.level = "SCENE"
    assert nav_cube.rim_buttons(app.region) == {}  # a model's tools


def test_a_click_on_section_opens_x_y_z_and_they_toggle_the_planes(app):
    click(app, *rim(app, "SECTION"))
    row = nav_cube.rim_buttons(app.region)
    assert {"SECTION_X", "SECTION_Y", "SECTION_Z"} <= set(row) and "MIRROR_X" not in row
    for name in ("SECTION_X", "SECTION_Y", "SECTION_Z"):  # clear of the corner button and of each other
        x, y, r = row[name]
        assert math.hypot(x - row["SECTION"][0], y - row["SECTION"][1]) > r + nav_cube.BUTTON_R
    assert document.grid.sections == {}
    click(app, *rim(app, "SECTION_Y"))
    assert set(document.grid.sections) == {1}
    assert nav_cube.axes_on("SECTION") == (False, True, False)
    assert "SECTION_X" in nav_cube.rim_buttons(app.region)  # the row stays open
    click(app, *rim(app, "SECTION_Y"))
    assert document.grid.sections == {}
    click(app, *rim(app, "SECTION"))  # a second click closes the row
    assert "SECTION_X" not in nav_cube.rim_buttons(app.region)


def test_mirror_toggles_its_axes_and_a_click_elsewhere_closes_the_row(app):
    click(app, *rim(app, "MIRROR"))
    click(app, *rim(app, "MIRROR_X"))
    click(app, *rim(app, "MIRROR_Z"))
    assert tuple(CONTEXT.prefs.rok_mirror) == (True, False, True)
    app.redraw()  # the row draws, with its on axes
    click(app, 300, 300)  # the viewport
    assert nav_cube._open is None


def test_opening_one_row_closes_the_other_and_a_drag_does_not_orbit(app):
    click(app, *rim(app, "SECTION"))
    click(app, *rim(app, "MIRROR"))
    assert nav_cube._open == "MIRROR"
    yaw = app.camera.yaw
    x, y = rim(app, "SECTION")
    app.handle("mouse_move", host_xy(app, x, y))
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", host_xy(app, x + 40, y))
    app.handle("mouse_up", {"button": "left"})
    assert app.camera.yaw == yaw


def test_the_view_menu_keeps_only_what_the_nav_cube_does_not_do(app):
    """Framing, the sides, Section, Mirror, Quad and Rendered View are on the cube.
    CAD View is in the menu."""
    menus = [c for c in app.host.calls if c[0] == "set_menu"][-1][1][0]
    labels = [item[1] if len(item) == 4 else item[0] for item in dict(menus)["View"] if item is not None]
    assert labels == ["Customize Rail Menu and Keys…", "Show Animation Menu", "Show History", "Saved Views", "Voxel Grid",
                      "CAD View", "Star: Pick on Click", "Theme", "Show Axis Letters on Nav Cube", "View Settings…",
                      "Render Settings…"]


def test_the_centre_button_frames_all_then_the_model_then_all_again(app, monkeypatch):
    seen = []
    monkeypatch.setattr(nav_cube.voxel_object, "frame_scene", lambda context: seen.append("ALL"))
    monkeypatch.setattr(nav_cube.voxel_object, "frame_model", lambda context, model: seen.append("MODEL"))
    bx, by = nav_cube.buttons(app.region, app.view_state)["CENTRE"]
    for _ in range(3):
        click(app, bx, by)
    assert seen == ["ALL", "MODEL", "ALL"]


def delta(app, dx, dy) -> None:
    """A raw mouse move while the host holds the pointer still: region pixels, y up."""
    app.handle("mouse_delta", {"dx": float(dx), "dy": float(-dy)})


def drag_from(app, name, dx, dy, steps=4):
    """Press on `name` and drag by (dx, dy): the first step moves the pointer
    (the press becomes a drag), the rest come as raw moves, as on the real host."""
    x, y = rim(app, name)
    app.handle("mouse_move", host_xy(app, x, y))
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", host_xy(app, x + dx / steps, y + dy / steps))
    for _k in range(1, steps):
        delta(app, dx / steps, dy / steps)
    app.handle("mouse_up", {"button": "left"})


def test_a_drag_from_a_section_axis_slides_its_plane(app):
    """Click X, Y or Z and move the mouse: the plane moves along the axis."""
    click(app, *rim(app, "SECTION"))
    grid = document.grid
    yaw = app.camera.yaw
    step = nav_cube.slide_step(CONTEXT, "Z")
    drag_from(app, "SECTION_Z", *(step * 3))  # off: it comes on, then follows the pointer
    start = max(1, grid.size[2] // 2)
    assert grid.sections[2][0] == start + 3
    drag_from(app, "SECTION_Z", *(-step * 5))
    assert grid.sections[2][0] == start - 2
    drag_from(app, "SECTION_Z", *(step * 500))  # kept inside the grid
    assert grid.sections[2][0] == grid.size[2] - 1
    assert app.camera.yaw == yaw  # no orbit
    click(app, *rim(app, "SECTION_Z"))  # a plain click still takes it away
    assert 2 not in grid.sections


def test_the_slide_follows_the_axis_on_screen(app):
    look_from(app, (0, -1, 0))  # from the front: x runs right, z up, y into the screen
    sx, sz, sy = (nav_cube.slide_step(CONTEXT, a) for a in "XZY")
    assert sx[0] > 0 and abs(sx[1]) < 1e-6
    assert sz[1] > 0 and abs(sz[0]) < 1e-6
    assert np.allclose(sy, (0.0, nav_cube.SLIDE_MIN_PX))  # along the view: up the screen


def test_the_slide_goes_on_past_the_window_edge(app):
    """The cursor must not run into the screen edge and stop the plane: the
    pointer hides and holds still during the drag; the raw moves count, however far they go."""
    look_from(app, (0, -1, 0))  # x runs right, toward the edge the rim sits at
    app.camera.distance *= 3.0  # small layers: the edge comes before the grid's end
    app.camera_moved()
    click(app, *rim(app, "SECTION"))
    grid = document.grid
    step = nav_cube.slide_step(CONTEXT, "X")[0]
    x, y = rim(app, "SECTION_X")
    app.handle("mouse_move", host_xy(app, x, y))
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", host_xy(app, x + 5, y))  # off: it comes on; the drag starts
    assert ("set_relative", (True,)) in [(c[0], c[1]) for c in app.host.calls]
    start = max(1, grid.size[0] // 2)  # where the plane came on
    far = app.region.width - x + 6 * step  # past the window's right edge
    delta(app, far, 0)
    app.handle("mouse_up", {"button": "left"})
    assert grid.sections[0][0] == min(start + round((far + 5) / step), grid.size[0] - 1)
    assert grid.sections[0][0] > start + round((app.region.width - x) / step)  # it went on past the edge
    assert [c[0] for c in app.host.calls if c[0] == "set_relative"][-1:] and app.host.calls[
        max(i for i, c in enumerate(app.host.calls) if c[0] == "set_relative")][1] == (False,)
    assert not [c for c in app.host.calls if c[0] == "warp_cursor"]


def test_hovering_an_axis_button_previews_its_plane(app, monkeypatch):
    """The X, Y and Z buttons show the plane they turn on."""
    from rok.app import overlay

    drawn = []
    monkeypatch.setattr(overlay, "_draw_plane_preview", lambda size, plane, sections, scale: drawn.append(plane))
    click(app, *rim(app, "MIRROR"))
    x, y = rim(app, "MIRROR_Y")
    app.handle("mouse_move", host_xy(app, x, y))
    assert nav_cube.plane_preview() == ("MIRROR", 1)
    app.redraw()
    assert drawn and drawn[-1] == ("MIRROR", 1)
    click(app, *rim(app, "SECTION"))  # the other row: its own planes
    x, y = rim(app, "SECTION_Z")
    app.handle("mouse_move", host_xy(app, x, y))
    assert nav_cube.plane_preview() == ("SECTION", 2)
    app.handle("mouse_move", host_xy(app, 300, 300))  # off the buttons: no preview
    assert nav_cube.plane_preview() is None


def test_the_section_preview_sits_where_the_plane_is_or_will_be():
    from rok.app import overlay

    quads = []
    orig = overlay._fan, overlay._lines
    overlay._fan = lambda quad, colour: quads.append(np.array(quad))
    overlay._lines = lambda *a, **k: None
    try:
        overlay._draw_plane_preview((20, 10, 8), ("SECTION", 0), {}, 1.0)
        overlay._draw_plane_preview((20, 10, 8), ("SECTION", 0), {0: (6, True)}, 1.0)
        overlay._draw_plane_preview((20, 10, 8), ("MIRROR", 2), {}, 1.0)
    finally:
        overlay._fan, overlay._lines = orig
    assert np.allclose(quads[0][:, 0], 10) and np.allclose(quads[1][:, 0], 6) and np.allclose(quads[2][:, 2], 4)


def test_old_events_after_a_jump_do_not_run_the_slide_away(app):
    """At the screen edge the plane snapped to the far end. After
    the cursor jumps, macOS can still send events from before, at the old edge."""
    look_from(app, (0, -1, 0))
    app.camera.distance *= 3.0
    app.camera_moved()
    click(app, *rim(app, "SECTION"))
    grid = document.grid
    x, y = rim(app, "SECTION_X")
    app.handle("mouse_move", host_xy(app, x, y))
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_move", host_xy(app, x + 5, y))
    right = app.region.width - 1
    app.handle("mouse_move", host_xy(app, right, y))  # the edge: the cursor jumps left
    at_edge = grid.sections[0][0]
    for _ in range(3):  # stale events, still at the old edge
        app.handle("mouse_move", host_xy(app, right, y))
    assert grid.sections[0][0] == at_edge  # no run to the far end
    app.handle("mouse_move", host_xy(app, 8.0, y))  # the first new event, where the cursor landed
    assert grid.sections[0][0] == at_edge
    app.handle("mouse_up", {"button": "left"})


def test_the_sun_button_switches_the_rendered_view_and_the_quad_view_ends_it(app):
    buttons = nav_cube.buttons(app.region)
    assert list(buttons)[-2:] == ["RENDER", "STEADY"]
    x, y = buttons["RENDER"]
    assert x + nav_cube.BUTTON_R <= app.region.width  # the row stays inside the view
    click(app, x, y)
    assert app.rendered
    renders = [c for c in app.host.calls if c[0] == "set_render"]
    assert renders[-1][1] == (True,)
    assert any(c[0] == "set_materials" for c in app.host.calls)  # the glass and lights go to the host
    app.handle("menu", {"id": "rok.quad_view"})
    assert app.quad and not app.rendered  # the rendered view shows one view
    app.handle("menu", {"id": "rok.rendered_view"})
    assert app.rendered and not app.quad
    app.handle("menu", {"id": "rok.rendered_view"})
    assert not app.rendered
    assert [c for c in app.host.calls if c[0] == "set_render"][-1][1] == (False,)


def test_a_render_preset_lights_the_rendered_view_and_is_kept_by_the_file(app):
    from rok.app import render_view
    from rok.core import render_presets

    app.handle("menu", {"id": "rok.render_preset(preset='NIGHT')"})
    assert app.rendered  # picking a preset shows it
    assert document.scene.light.preset == "NIGHT" and document.modified  # the file's light
    sent = [c for c in app.host.calls if c[0] == "set_render"][-1]
    night = render_presets.preset_by_id("NIGHT")
    assert sent[1] == (True,) and sent[2]["tone"] == render_presets.TONE_CODES[night.tone]
    assert len(sent[2]["lamps"]) == 2 and sent[2]["bloom"] == night.bloom
    # A bigger scene: the lamps move out and grow, so the settings go again.
    calls = len([c for c in app.host.calls if c[0] == "set_render"])
    app.flush()
    assert len([c for c in app.host.calls if c[0] == "set_render"]) == calls  # nothing changed: nothing sent
    document.model.position = (40, 0, 0)
    app.flush()
    assert len([c for c in app.host.calls if c[0] == "set_render"]) == calls + 1
    assert render_view.preset_id() == "NIGHT"


def test_a_light_change_is_one_undo_step_and_the_view_follows(app):
    app.handle("menu", {"id": "rok.render_preset(preset='STUDIO')"})
    studio = [c for c in app.host.calls if c[0] == "set_render"][-1][2]
    app.handle("menu", {"id": "rok.render_preset(preset='NIGHT')"})
    app.flush()
    assert [c for c in app.host.calls if c[0] == "set_render"][-1][2]["tone"] != studio["tone"]
    app.handle("menu", {"id": "rok.undo"})
    app.flush()
    assert document.scene.light.preset == "STUDIO"
    assert [c for c in app.host.calls if c[0] == "set_render"][-1][2] == studio  # the view got Studio back


def test_every_button_has_a_tooltip():
    names = set(nav_cube.BUTTONS) | {name for name, _a, _s in nav_cube.RIM}
    names |= {f"{name}_{axis}" for name, _a, _s in nav_cube.RIM for axis in "XYZ"}
    assert names <= set(nav_cube.TIPS)


def test_a_resting_mouse_shows_the_buttons_tooltip_under_the_row(app):
    bx, by = nav_cube.buttons(app.region, app.view_state)["PAN"]
    app.handle("mouse_move", host_xy(app, bx, by))
    app.redraw()
    texts = [t[4] for c in app.host.calls if c[0] == "set_draw" for t in c[2].get("texts", [])]
    assert not any("Pan:" in t for t in texts)  # not yet due
    app.clock.t += 1.0
    app.host.calls.clear()
    app.redraw()
    texts = [t for c in app.host.calls if c[0] == "set_draw" for t in c[2].get("texts", [])]
    tip = [t for t in texts if t[4].startswith("Pan:")]
    assert tip
    x, y, w, h = nav_cube.tip_rect(app.region, app.view_state, 1, 100.0)
    assert y + h < by - nav_cube.BUTTON_R  # under the row
    assert x >= app.view_state.area(app.region)[0]


def test_tooltip_lines_stay_inside_the_width():
    lines = nav_cube.tip_lines(nav_cube.TIPS["RENDER"], lambda t: len(t) * 6.0, 120.0)
    assert len(lines) > 1 and all(len(line) * 6.0 <= 120.0 or " " not in line for line in lines)


def test_the_axis_letters_sit_on_the_edges_and_can_be_turned_off(app):
    from rok.app import opkit, settings

    letters = nav_cube.edge_letters(app.region, app.view_state)
    assert {m[0] for m in letters} == {"X", "Y", "Z"}
    c = nav_cube.centre(app.region, app.view_state)
    for _letter, _k, pos in letters:
        assert nav_cube.CUBE_PX * 0.5 < math.dist(pos, c) < nav_cube.DISC_R  # by the cube, inside the disc
    app.redraw()
    opkit.call("rok.cube_axis_letters", "EXEC")
    assert not nav_cube.letters_on() and settings.get(nav_cube.LETTERS_SETTING) == "off"
    app.redraw()
    opkit.call("rok.cube_axis_letters", "EXEC")
    assert nav_cube.letters_on()


def test_the_steady_hand_button_and_its_key_switch_the_string(app):
    from rok.app.opkit import CONTEXT

    wm = CONTEXT.prefs
    wm.rok_steady_hand = False
    x, y = nav_cube.buttons(app.region)["STEADY"]
    assert x + nav_cube.BUTTON_R <= app.region.width  # the row stays inside the view
    click(app, x, y)
    assert wm.rok_steady_hand
    app.handle("key_down", {"key": "H", "code": "H", "text": "h", "repeat": False})
    app.handle("key_up", {"key": "H", "code": "H", "text": "", "repeat": False})
    assert not wm.rok_steady_hand  # the H key switches it back


def test_the_buttons_sit_in_rows_under_the_cube(app):
    row = nav_cube.buttons(app.region)
    assert tuple(tuple(n for n in row if abs(row[n][1] - y) < 0.5) for y in sorted({round(p[1], 1) for p in row.values()}, reverse=True)) == nav_cube.GRID
    ys = [row[names[0]][1] for names in nav_cube.GRID]
    assert ys[0] > ys[1] > ys[2]  # the rows go down
    for names in nav_cube.GRID:
        xs = [row[name][0] for name in names]
        assert xs == sorted(xs)  # left to right in the grid's order
    columns = [row[names[0]][0] for names in nav_cube.GRID if len(names) == 2]
    assert max(columns) - min(columns) < 0.5  # one column under the other
    centre = nav_cube.centre(app.region, app.view_state)
    assert abs((row["CENTRE"][0] + row["QUAD"][0]) / 2 - centre[0]) < 1.0  # centred under the cube
