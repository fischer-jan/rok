"""View > View Settings: the panel at the right edge (rok/app/view_panel.py,
rok/core/view_panel.py) against a recording host."""

from __future__ import annotations

from pathlib import Path

import pytest

from rok.app import looks, render_view, view_panel
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state
from rok.core import light
from rok.core import view_panel as vp

FIXTURES = Path(__file__).parent / "fixtures"
W, H = 1200.0, 800.0


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

    def named(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def app(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "bird.vox"))
    a.handle("start", {"width": W, "height": H, "scale": 2.0})
    yield a
    view_panel._close(a)
    document.reset()
    state.__init__()


def move(app, x: float, y: float) -> None:
    app.handle("mouse_move", {"x": float(x), "y": float(app.region.height - y)})


def click(app, x: float, y: float) -> None:
    move(app, x, y)
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def layout(app) -> vp.PanelLayout:
    return view_panel._layout(view_panel._session, app.region, app)


def centre(rect) -> tuple[float, float]:
    x, y, w, h = rect
    return x + w / 2, y + h / 2


def click_item(app, kind: str, key: str = "", value=None) -> None:
    item = layout(app).find(kind, key, value)
    assert item is not None, (kind, key, value)
    click(app, *centre(item.rect))


def open_panel(app) -> None:
    app.handle("menu", {"id": "rok.view_settings"})
    assert view_panel.active()


def turn_to(app, page: str) -> None:
    """The other command while the panel is open turns to its page."""
    app.handle("menu", {"id": "rok.render_settings" if page == vp.RENDER else "rok.view_settings"})
    assert view_panel.active() and view_panel._session.tab == page


def last_render(app) -> dict:
    return [c for c in app.host.calls if c[0] == "set_render"][-1]


def test_the_panel_stands_at_the_right_edge_and_leaves_the_model_free(app):
    open_panel(app)
    rect = layout(app).rect
    assert rect[0] + rect[2] <= W and rect[0] > W / 2  # beside the model, not over its middle
    app.redraw()  # both pages draw without an error
    turn_to(app, vp.RENDER)
    app.redraw()


def test_the_layout_fits_a_small_window_with_every_section_open():
    for section in (vp.LIGHTS, vp.SKY, vp.FINISH, vp.QUALITY):
        state_ = vp.State(tab=vp.RENDER, section=section, lights=("SUN", "LAMP", "SUN", "LAMP"), picked=1,
                          backdrop="COLOUR", ground=True, changed=True, menu=True,
                          presets=tuple((f"P{k}", f"Preset {k}") for k in range(9)))
        lay = vp.panel_layout(1.0, 1000.0, 720.0, state_)
        assert lay.rect[1] >= 0 and lay.rect[1] + lay.rect[3] <= 720.0, section
        panel = [i for i in lay.items if i.kind != "MENU"]
        for item in panel:  # every part inside the panel
            x, y, w, h = item.rect
            assert lay.rect[0] <= x and x + w <= lay.rect[0] + lay.rect[2] + 0.01, (section, item)
            assert lay.rect[1] <= y and y + h <= lay.rect[1] + lay.rect[3] + 0.01, (section, item)
        clickable = [i for i in panel if i.kind not in ("LABEL", "NOTE")]
        for a in clickable:  # nothing you click overlaps
            for b in clickable:
                if a is b:
                    continue
                ax, ay, aw, ah = a.rect
                bx, by, bw, bh = b.rect
                assert ax + aw <= bx + 0.01 or bx + bw <= ax + 0.01 or ay + ah <= by + 0.01 or by + bh <= ay + 0.01, (a, b)
    look = vp.panel_layout(1.0, 1000.0, 720.0, vp.State(tab=vp.LOOK, looks=tuple((str(k), "L") for k in range(12))))
    assert look.rect[1] >= 0


def test_a_look_shows_at_once_with_no_apply(app, tmp_path):
    open_panel(app)
    click_item(app, "CHIP", "look", "PAPER")
    assert app.look == "PAPER" and (tmp_path / "config" / looks.FILE_NAME).read_text().strip() == "PAPER"
    click_item(app, "TOGGLE", "occlusion")
    assert app.host.named("set_occlusion")[-1][1][0] == 0.0
    click_item(app, "TOGGLE", "smooth")
    assert app.host.named("set_smooth_edges")[-1][1] == (False,)
    assert view_panel.active()  # still open: nothing to apply


def test_the_model_box_toggle_hides_the_box_and_is_kept(app, tmp_path):
    from rok.app import settings
    from rok.app.state import state

    assert state.model_box  # on until the user turns it off
    open_panel(app)
    click_item(app, "TOGGLE", "box")
    assert not state.model_box and settings.get("model_box") == "off"
    click_item(app, "CLOSE")
    assert not state.model_box  # Done keeps it
    open_panel(app)
    click_item(app, "TOGGLE", "box")
    assert state.model_box and settings.get("model_box") == "on"
    click_item(app, "BUTTON", "revert")  # as it was when the panel opened
    assert not state.model_box and settings.get("model_box") == "off"


def test_render_settings_shows_the_rendered_view_and_close_puts_the_view_back(app):
    open_panel(app)
    assert view_panel._session.tab == vp.LOOK and not render_view.rendered(app)
    turn_to(app, vp.RENDER)
    assert render_view.rendered(app) and last_render(app)[1] == (True,)
    click_item(app, "CLOSE")
    assert not view_panel.active() and not render_view.rendered(app)  # the view it opened with


def test_each_command_opens_its_own_page_with_its_title(app):
    app.handle("menu", {"id": "rok.render_settings"})
    assert view_panel._session.tab == vp.RENDER and render_view.rendered(app)
    assert layout(app).find("TITLE", vp.RENDER).text == "Render Settings"
    assert layout(app).find("TAB", vp.LOOK) is None  # no tabs: each page has its own command
    app.handle("menu", {"id": "rok.render_settings"})  # again: the panel closes
    assert not view_panel.active() and render_view.rendered(app)
    render_view.set_rendered(app, True)
    open_panel(app)  # View Settings shows the editing view, also from the rendered view
    assert view_panel._session.tab == vp.LOOK and not render_view.rendered(app)
    assert layout(app).find("TITLE", vp.LOOK).text == "View Settings"
    click_item(app, "CLOSE")
    assert render_view.rendered(app)  # the view it opened with


def test_a_preset_from_the_list_is_one_undo_step_and_hover_shows_it_first(app):
    open_panel(app)
    turn_to(app, vp.RENDER)
    click_item(app, "PRESET")
    night = layout(app).find("MENU", "NIGHT")
    move(app, *centre(night.rect))
    app.flush()
    assert last_render(app)[2]["tone"] == light.trace_settings(light.from_preset("NIGHT"), (0, 0, 0), (1, 1, 1))["tone"]
    assert not document.modified  # only shown
    move(app, *centre(layout(app).rect))  # off the list: back to Studio
    assert render_view.scene_light() == light.default()
    click_item(app, "MENU", "NIGHT")  # the list is still open
    assert document.scene.light == light.from_preset("NIGHT") and document.modified
    app.handle("menu", {"id": "rok.undo"})
    assert document.scene.light is None


def test_a_slider_drag_is_one_undo_step_on_release(app):
    open_panel(app)
    turn_to(app, vp.RENDER)
    click_item(app, "SECTION", vp.FINISH)
    control = layout(app).control("exposure")
    x0, y = control.track[0] + control.track[2] / 2, control.track[1] + control.track[3] / 2
    move(app, x0, y)
    app.handle("mouse_down", {"button": "left"})
    move(app, control.track[0] + control.track[2] * 0.75, y)
    app.flush()
    assert last_render(app)[2]["exposure"] == pytest.approx(2.0)  # live while dragging
    steps = len(document.undo._undo)
    move(app, control.track[0] + control.track[2], y)
    app.handle("mouse_up", {"button": "left"})
    assert document.scene.light.exposure == pytest.approx(4.0)
    assert len(document.undo._undo) == steps + 1
    app.handle("menu", {"id": "rok.undo"})
    assert render_view.scene_light().exposure == 0.0


def test_lights_add_pick_mute_and_remove(app):
    open_panel(app)
    turn_to(app, vp.RENDER)
    click_item(app, "PRESET")
    click_item(app, "MENU", "STANDARD")
    click_item(app, "ADD")
    assert len(document.scene.light.lights) == 2 and view_panel._session.picked == 1
    click_item(app, "CHIP", "kind", "LAMP")
    assert document.scene.light.lights[1].kind == light.LAMP
    assert layout(app).control("light.distance") is not None  # a lamp has a distance
    click_item(app, "BUTTON", "mute")
    app.flush()
    assert len(last_render(app)[2]["lamps"]) == 1
    click_item(app, "BUTTON", "remove")
    assert len(document.scene.light.lights) == 1


def test_reset_and_revert(app):
    open_panel(app)
    turn_to(app, vp.RENDER)
    click_item(app, "SECTION", vp.FINISH)
    click_item(app, "CHIP", "tone", "CLIP")
    assert layout(app).find("BUTTON", "reset") is not None  # changed from Studio
    click_item(app, "BUTTON", "reset")
    assert light.same_as_preset(document.scene.light)
    turn_to(app, vp.LOOK)
    click_item(app, "CHIP", "look", "CLAY")
    click_item(app, "BUTTON", "revert")
    assert app.look == "STUDIO" and document.scene.light is None


def test_a_colour_from_the_picker(app):
    open_panel(app)
    turn_to(app, vp.RENDER)
    click_item(app, "SWATCH", "light.colour")
    parts = view_panel._picker_parts(view_panel._session, app.region, layout(app))
    sv = parts["SV"]
    click(app, sv[0] + sv[2] - 1, sv[1] + sv[3] - 1)  # full saturation and value
    colour = document.scene.light.lights[0].colour
    assert max(colour) > 0.95 and min(colour) < 0.1


def test_a_plain_click_in_the_view_closes_but_a_navigation_drag_passes(app):
    open_panel(app)
    app.handle("modifiers", {"shift": False, "ctrl": True, "alt": False, "meta": False})
    move(app, 300, 300)
    app.handle("mouse_down", {"button": "left"})  # Ctrl + left drag on nothing orbits
    app.handle("mouse_up", {"button": "left"})
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": False})
    assert view_panel.active()
    voxels = document.model.grid.voxels.copy()
    click(app, 300, 300)
    assert not view_panel.active()
    assert (document.model.grid.voxels == voxels).all()  # the click only closed the panel


def test_the_samples_are_yours_not_the_files(app):
    from rok.app import settings

    open_panel(app)
    turn_to(app, vp.RENDER)
    click_item(app, "SECTION", vp.QUALITY)
    click_item(app, "CHIP", "samples", 2048)
    assert settings.get("render_samples") == "2048" and document.scene.light is None
    app.flush()
    assert last_render(app)[2]["max_samples"] == 2048


def test_a_drag_on_the_dome_moves_the_light_and_the_ring_turns_all(app):
    from rok.app import light_dome
    from rok.core import light_dome as kit

    open_panel(app)
    turn_to(app, vp.RENDER)
    dome_item = layout(app).find("DOME")
    dome = kit.dome_in(dome_item.rect)
    camera = light_dome.camera_azimuth(app)
    setup = render_view.scene_light()
    first = setup.lights[0]
    x, y = kit.point(dome, first.azimuth + setup.turn, first.elevation, camera)
    move(app, x, y)
    app.handle("mouse_down", {"button": "left"})
    move(app, dome.cx, dome.cy + 1)  # nearly straight above
    steps = len(document.undo._undo)
    app.handle("mouse_up", {"button": "left"})
    assert document.scene.light.lights[0].elevation > 85
    assert len(document.undo._undo) == steps + 1
    mid = (dome.ring_inner + dome.ring_outer) / 2
    move(app, dome.cx + mid, dome.cy)
    app.handle("mouse_down", {"button": "left"})
    move(app, dome.cx, dome.cy + mid)  # a quarter turn counter-clockwise
    app.handle("mouse_up", {"button": "left"})
    assert document.scene.light.turn == pytest.approx(90.0, abs=0.5)
