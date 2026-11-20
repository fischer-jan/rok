"""The editor UI: the rail, the colour popup, the star, Cmd+K, the hint
line, the menus and the welcome screen. The host only records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app import chrome, command, last_command, marking, opkit, palette_ui, settings, ui, welcome
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.opkit import CONTEXT
from rok.app.state import state
from rok.core import chrome as kit
from rok.core import star
from rok.core.undo import ModelRecord, StrokeRecorder
from test_foundation import RecordingHost, key

FIXTURES = Path(__file__).parent / "fixtures"
BIRD = FIXTURES / "bird.vox"
RECON = FIXTURES / "robot.vox"


def _reset_ui() -> None:
    chrome.invalidate()
    chrome.hover_stop = None
    chrome.slider_dragging = False
    chrome._fade_timer = None
    marking._session = None
    command._session = None
    palette_ui._popup = None
    palette_ui._drag = None
    palette_ui._eyedrop_active = False
    welcome._session = None
    welcome._shown_at_start = False


def _make_app(tmp_path, monkeypatch, path):
    monkeypatch.setenv("ROK_CONFIG", str(tmp_path / "config"))
    from rok.app import io_ops, recent
    from rok.core import new_sizes

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setattr(io_ops, "_last_new_size", new_sizes.size_of(new_sizes.DEFAULT))
    monkeypatch.delenv("ROK_NO_WELCOME", raising=False)
    document.reset()
    state.__init__()
    _reset_ui()
    a = RokApp(RecordingHost(), str(path) if path else None)
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    return a


@pytest.fixture
def app(tmp_path, monkeypatch):
    a = _make_app(tmp_path, monkeypatch, BIRD)
    yield a
    document.reset()
    state.__init__()
    _reset_ui()


def region_point(app, rx: float, ry: float) -> dict:
    """Host pixels (origin top left) of a region pixel (origin bottom left)."""
    return {"x": float(rx), "y": float(app.region.height - ry)}


def centre_of(rect) -> tuple[float, float]:
    x, y, w, h = rect
    return x + w / 2.0, y + h / 2.0


def move_to(app, rx: float, ry: float) -> None:
    app.handle("mouse_move", region_point(app, rx, ry))


def click_at(app, rx: float, ry: float) -> None:
    move_to(app, rx, ry)
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def widget(ident: str) -> kit.Widget:
    w = chrome.layout_for(CONTEXT.region).by_id(ident)
    assert w is not None, ident
    return w


def layer(app, name: str):
    """The last draw list the app sent for layer `name`: (icons, texts, arrays)."""
    calls = [c for c in app.host.named("set_draw") if c[1][0] == name]
    assert calls, f"no layer {name}"
    _n, _args, kw = calls[-1]
    return kw


def texts_of(app, name: str) -> list[str]:
    return [t[4] for t in layer(app, name)["texts"]]


# --- the rail -----------------------------------------------------------------------


def test_the_rail_draws_its_plate_tools_and_headers(app):
    app.redraw()
    kw = layer(app, "chrome")
    assert len(kw["tris2"]) > 0
    icons = {i[0] for i in kw["icons"]}
    assert {"chrome/1x/CUBE_PLUS", "chrome/1x/CUBE_MINUS", "chrome/1x/MODEL", "chrome/1x/LAYERS",
            "chrome/1x/LEVEL_MODEL", "chrome/1x/LEVEL_ANIM"} <= icons
    assert {"Model", "Layers", "Size"} <= set(texts_of(app, "chrome"))


def test_a_click_on_a_tool_sets_the_tool(app):
    wm = CONTEXT.prefs
    click_at(app, *centre_of(widget("TOOL_ERASE").rect))
    assert wm.rok_tool == "ERASE"
    assert widget("TOOL_ERASE").active and not widget("TOOL_ATTACH").active


def test_a_tool_flyout_opens_on_hover_and_its_item_picks_the_mode(app):
    wm = CONTEXT.prefs
    move_to(app, *centre_of(widget("TOOL_PAINT").rect))
    assert state.chrome_open == "TOOL_PAINT"
    item = next(w for w in kit.flyout_widgets(chrome.layout_for(CONTEXT.region), "TOOL_PAINT")
                if w.id == "TOOL_PAINT_FACE")
    # Along the row to the item: the flyout stays open on the way.
    click_at(app, *centre_of(item.rect))
    assert (wm.rok_tool, wm.rok_mode) == ("PAINT", "FACE")


def test_the_tooltip_shows_after_its_delay(app):
    move_to(app, *centre_of(widget(kit.SIZE_ID).rect))
    assert state.hover_chrome == kit.SIZE_ID
    assert chrome.tooltip_id() is None  # not yet
    assert any(abs(c[1][0] - (kit.TOOLTIP_DELAY_S + 0.02)) < 1e-9 for c in app.host.named("tick"))
    state.chrome_since -= kit.TOOLTIP_DELAY_S + 0.01  # the delay has passed
    assert chrome.tooltip_id() == kit.SIZE_ID
    app.redraw()
    assert widget(kit.SIZE_ID).tooltip in texts_of(app, "chrome_tips")  # its own layer, over the history strip


def test_a_drag_on_the_size_field_changes_the_brush_size(app):
    """The Size field is a number you drag."""
    wm = CONTEXT.prefs
    wm.rok_brush_size = 3
    w = widget(kit.SIZE_ID)
    cx, cy = centre_of(w.rect)
    move_to(app, cx, cy)
    app.handle("mouse_down", {"button": "left"})
    assert chrome.slider_dragging
    # The pointer hides and holds still; the raw moves count (the
    # field sits at the window's left edge, a left drag ran into it).
    assert ("set_relative", (True,)) in [(c[0], c[1]) for c in app.host.calls]
    app.handle("mouse_delta", {"dx": 2.5 * kit.SIZE_DRAG_PX, "dy": 0.0})  # right: two steps
    assert int(wm.rok_brush_size) == 5
    app.handle("mouse_delta", {"dx": -3.5 * kit.SIZE_DRAG_PX, "dy": 0.0})  # left of the press: one step down
    assert int(wm.rok_brush_size) == 2
    app.handle("mouse_up", {"button": "left"})
    assert not chrome.slider_dragging and int(wm.rok_brush_size) == 2
    assert [c[1] for c in app.host.calls if c[0] == "set_relative"][-1] == (False,)
    # Far to the left, past the window's edge: down to the smallest size.
    move_to(app, cx, cy)
    app.handle("mouse_down", {"button": "left"})
    for _k in range(40):
        app.handle("mouse_delta", {"dx": -kit.SIZE_DRAG_PX, "dy": 0.0})
    app.handle("mouse_up", {"button": "left"})
    assert int(wm.rok_brush_size) == kit.BRUSH_SIZES[0]
    wm.rok_brush_size = 2
    # A click without a drag changes nothing.
    click_at(app, cx, cy)
    assert int(wm.rok_brush_size) == 2


def test_a_press_in_the_rail_beside_a_button_is_eaten(app):
    before = CONTEXT.prefs.__dict__.copy()
    assert chrome.press(CONTEXT, CONTEXT.region, 3, 450, True, False) is True
    assert CONTEXT.prefs.__dict__ == before


def test_the_animate_cell_goes_to_animate_mode_and_the_model_cell_back(app):
    cell = widget(kit.LEVEL_ANIM_ID)
    assert cell.enabled
    click_at(app, *centre_of(cell.rect))
    assert state.animating
    click_at(app, *centre_of(widget(kit.LEVEL_MODEL_ID).rect))
    assert not state.animating and state.level == "MODEL"


# --- the colour popup -----------------------------------------------------------------


def _press_part(app, part: str, fx: float = 0.5, fy: float = 0.5, release: bool = True) -> None:
    from rok.app import overlay

    dock = overlay.dock_layout(app.region)  # as the frame after the popup opened lays the card out
    parts = palette_ui._current_layout(palette_ui._popup)
    if part == "STAR" and part not in parts:  # inside the palette card the header's star does it
        part_rect = dict(dock.buttons)["FAVOURITE"]
    else:
        part_rect = parts[part]
    x, y, w, h = part_rect
    move_to(app, x + w * fx, y + h * fy)
    app.handle("mouse_down", {"button": "left"})
    if release:
        app.handle("mouse_up", {"button": "left"})


def test_the_colour_popup_edits_the_colour_as_one_undo_step(app):
    scene = document.scene
    index = document.colour
    before = scene.palette.copy()
    depth = len(document.undo._undo)
    move_to(app, 400, 200)
    opkit.call("rok.colour_popup")
    assert palette_ui.popup_open()
    # A drag across the saturation and value square.
    _press_part(app, "SV", 0.9, 0.9, release=False)
    x, y, w, h = palette_ui._current_layout(palette_ui._popup)["SV"]
    move_to(app, x + w * 0.5, y + h * 0.4)
    app.handle("mouse_up", {"button": "left"})
    dragged = tuple(int(c) for c in scene.palette[index, :3])
    assert dragged != tuple(int(c) for c in before[index, :3])
    # The hex field takes typed ASCII; Enter writes it.
    _press_part(app, "HEX")
    for ch in "12ab3f":
        key(app, ch)
    key(app, "Enter")
    assert tuple(int(c) for c in scene.palette[index, :3]) == (0x12, 0xAB, 0x3F)
    assert CONTEXT.prefs.rok_hex == "#12AB3F"
    assert document.modified
    # The material type, in the same session.
    _press_part(app, "TYPE:METAL")
    assert scene.materials.get(index).type == "METAL"
    assert len(document.undo._undo) == depth + 1  # one popup session, one step
    opkit.call("rok.undo")
    assert np.array_equal(scene.palette, before)
    assert scene.materials.get(index).type != "METAL"


def test_a_slider_of_the_material_type_writes_its_setting(app):
    move_to(app, 400, 200)
    opkit.call("rok.colour_popup")
    CONTEXT.prefs.rok_material_type = "EMIT"
    parts = palette_ui._current_layout(palette_ui._popup)
    assert {"SLIDER:rok_glow", "SLIDER:rok_power"} <= set(parts)
    _press_part(app, "SLIDER:rok_power", 1.0)
    assert CONTEXT.prefs.rok_power == 4


def test_a_click_outside_or_esc_closes_the_popup(app):
    move_to(app, 400, 200)
    opkit.call("rok.colour_popup")
    click_at(app, 1300, 800)
    assert not palette_ui.popup_open()
    opkit.call("rok.colour_popup")
    key(app, "Escape")
    assert not palette_ui.popup_open()


def test_new_colour_makes_a_white_plain_colour_and_changes_no_other(app):
    scene = document.scene
    old = document.colour
    scene.materials.set(old, scene.materials.get(old).with_type("GLASS"))
    palette_before = scene.palette.copy()
    materials_before = [scene.materials.get(i) for i in range(256)]
    opkit.call("rok.colour_new")
    new = document.colour
    assert new != old and document.used_counts()[new] == 0
    assert tuple(scene.palette[new]) == (255, 255, 255, 255)
    assert scene.materials.get(new).type == "DIFFUSE"
    others = [i for i in range(256) if i != new]
    assert np.array_equal(scene.palette[others], palette_before[others])
    assert all(scene.materials.get(i) == materials_before[i] for i in others)
    store = scene.colours
    assert store.id_of(new) in store.unused_ids(document.used_counts())  # the strip shows it at once
    assert palette_ui.popup_open()
    opkit.call("rok.undo")
    assert np.array_equal(scene.palette, palette_before)


def test_new_colour_on_a_full_palette_does_not_open_the_popup(app):
    scene = document.scene
    for model in scene.models:
        model.grid.voxels[...] = 0
    model = scene.models[0]
    flat = model.grid.voxels.reshape(-1)
    flat[:255] = np.arange(1, 256, dtype=np.uint8)
    document.stroke_serial += 1
    colour = scene.palette.copy()
    opkit.call("rok.colour_new")
    assert not palette_ui.popup_open()
    assert np.array_equal(scene.palette, colour)


@pytest.mark.parametrize("text, rgb", [("#FF8000", (255, 128, 0)), ("00ff7f", (0, 255, 127)), ("12345", None), ("GGHHII", None)])
def test_parse_hex(text, rgb):
    assert palette_ui.parse_hex(text) == rgb


def test_a_ctrl_drag_moves_a_colour_in_the_strip_and_changes_no_voxel(app):
    from rok.app import overlay

    scene = document.scene
    store = scene.colours
    used, _unused = overlay.strip_entries()
    a, b = used[0], used[2]
    ia = store.id_of(a)
    palette, voxels = scene.palette.copy(), document.grid.voxels.copy()
    order = list(store.order)
    palette_ui._apply_move(a, b)
    assert store.order.index(ia) == order.index(store.id_of(b))
    assert np.array_equal(scene.palette, palette) and np.array_equal(document.grid.voxels, voxels)
    opkit.call("rok.undo")
    assert store.order == order


def test_a_shift_drag_blends_the_colours_between_two_in_the_strip(app):
    from rok.app import overlay

    store = document.scene.colours
    used, _unused = overlay.strip_entries()
    a, b = used[0], used[4]
    span = store.order[store.order.index(store.id_of(a)):store.order.index(store.id_of(b)) + 1]
    first, last = store.entries[span[0]].rgba, store.entries[span[-1]].rgba
    palette_ui._apply_gradient(a, b)
    mid = store.entries[span[2]].rgba
    assert all(min(f, l) <= m <= max(f, l) for f, l, m in zip(first, last, mid))
    assert tuple(document.scene.palette[store.slot_of(span[2])]) == mid  # its voxels change with it
    opkit.call("rok.undo")
    assert store.entries[span[0]].rgba == first


# --- the star -------------------------------------------------------------------------


def _chip_centre(chip_id: str) -> tuple[float, float]:
    layout = marking._layout(marking._session, CONTEXT.region)
    chip = next(c for c in layout.chips if c.chip.id == chip_id)
    return chip.centre


def test_the_star_opens_on_space_and_picks_on_release(app):
    wm = CONTEXT.prefs
    move_to(app, 760, 450)
    app.handle("key_down", {"key": "Space", "code": "Space", "text": " ", "repeat": False})
    assert marking.active()
    marking._session.since -= 5.0  # the rays have unrolled
    app.redraw()
    assert len(layer(app, "star")["icons"]) > 0
    move_to(app, *_chip_centre("TOOL_ERASE"))
    app.handle("key_up", {"key": "Space", "code": "Space", "text": "", "repeat": False})
    assert not marking.active()
    assert wm.rok_tool == "ERASE"


def test_a_release_in_the_middle_of_the_star_picks_nothing(app):
    wm = CONTEXT.prefs
    move_to(app, 760, 450)
    key(app, "Space")
    assert not marking.active()
    assert getattr(wm, "rok_tool", "ATTACH") == "ATTACH"


def test_the_star_on_click_mode_picks_with_a_click(app):
    wm = CONTEXT.prefs
    wm.rok_star_pick = marking.CLICK
    move_to(app, 760, 450)
    key(app, "Space")  # opens; the release does nothing
    assert marking.active()
    marking._session.since -= 5.0
    click_at(app, *_chip_centre("TOOL_SELECT"))
    assert not marking.active() and wm.rok_tool == "SELECT"
    assert settings.get("star_pick") == "CLICK"  # remembered


def _open_star_on(app, ident):
    move_to(app, 760, 450)
    app.handle("key_down", {"key": "Space", "code": "Space", "text": " ", "repeat": False})
    marking._session.since -= 5.0
    ray = next(i for i, r in enumerate(marking._session.rays) if any(c.id == ident for c in r.chips))
    marking._session.open_ray = ray
    marking._session.extents[ray] = 1.0
    move_to(app, *_chip_centre(ident))
    app.handle("key_up", {"key": "Space", "code": "Space", "text": "", "repeat": False})


def test_the_star_s_last_chip_runs_the_last_command_again(app):
    def ids():
        return {c.id for r in star.rays_for(state.level, False, (), last_command.current()) for c in r.chips}

    assert star.LAST_COMMAND not in ids()  # nothing ran yet
    assert not any(i.startswith("PRIMITIVE_") for i in ids())  # the primitives left the star
    run_op = opkit.call
    run_op("rok.undo")  # undo is no command for the star
    assert last_command.current() is None
    app.handle("menu", {"id": "rok.inverse"})
    assert last_command.current().id == "inverse"
    depth = len(document.undo._undo)
    _open_star_on(app, star.LAST_COMMAND)
    assert len(document.undo._undo) == depth + 1  # Inverse ran again


def test_a_dialog_command_counts_when_its_dialog_is_confirmed(app):
    app.handle("menu", {"id": "rok.loop"})
    assert last_command.current() is None  # the dialog is still open
    key(app, "Escape")
    assert last_command.current() is None  # cancelled: no command
    app.handle("menu", {"id": "rok.loop"})
    key(app, "Enter")
    assert last_command.current().id == "loop"
    chip = star.last_chip(last_command.current())
    assert chip.glyph == "LOOP" and chip.label == "Loop"
    _open_star_on(app, star.LAST_COMMAND)
    assert app.dialog is not None and app.dialog.op.command_id == "rok.loop"  # it asks again


def test_a_primitive_picked_in_the_rail_is_the_last_command(app):
    """A Diamond from the Primitives menu did not show on the star."""
    wm = CONTEXT.prefs
    region = CONTEXT.region
    chrome._run(CONTEXT, "PRIMITIVE_DIAMOND", False, chrome.layout_for(region, CONTEXT), region)
    last = last_command.current()
    assert (last.id, last.label, last.glyph) == ("PRIMITIVE_DIAMOND", "Diamond", "DIAMOND")
    wm.rok_tool, wm.rok_mode = "PAINT", "VOXEL"
    _open_star_on(app, "PRIMITIVE_DIAMOND")
    assert (wm.rok_tool, wm.rok_mode, wm.rok_solid) == ("ATTACH", "BOX", "DIAMOND")
    state.level = kit.LEVEL_SCENE
    assert last_command.current() is None  # a model's primitive: not in the scene
    state.level = kit.LEVEL_MODEL
    app.handle("menu", {"id": "rok.inverse"})
    assert last_command.current().id == "inverse"  # a later command takes over


def test_a_choice_command_keeps_its_choice(app):
    opkit.call("rok.scale", "INVOKE", mode="UP")
    assert last_command.current().props == (("mode", "UP"),)


# --- Cmd+K ------------------------------------------------------------------------------


def test_cmd_k_filters_by_typing_and_runs_the_command(app):
    grid = document.grid
    rec = StrokeRecorder(grid)
    rec.set(1, 1, 1, 7)
    document.undo.push(ModelRecord(document.active_uid, rec.finish()))
    key(app, "k", meta=True)
    assert command.active()
    for ch in "undo":
        key(app, ch)
    assert command._session.query == "undo"
    assert command._hits(command._session)[0].id == "undo"
    app.redraw()
    assert "undo" in texts_of(app, "command")
    key(app, "Enter")
    assert not command.active()
    assert grid.voxels[1, 1, 1] == 0


def test_cmd_k_lists_only_commands_rok_has(app):
    key(app, "k", ctrl=True)
    hits = command._hits(command._session)
    assert hits and all(f"rok.{c.id}" in opkit.REGISTRY for c in hits)
    assert not any(c.id in command.PARKED_IDS for c in hits)
    key(app, "Escape")
    assert not command.active()


def test_the_menu_and_the_key_do_not_toggle_the_palette_shut(app):
    key(app, "k", meta=True)
    app.handle("menu", {"id": "rok.command_search"})  # the same Cmd+K, seen twice
    assert command.active()


def test_cmd_k_arrows_move_the_highlight_and_a_click_outside_closes(app):
    key(app, "k", meta=True)
    key(app, "ArrowDown")
    assert command._session.index == 1
    click_at(app, 5, 5)
    assert not command.active()


# --- the hint line and the menus ---------------------------------------------------


def test_the_hint_line_names_the_tool_and_the_mode(app):
    app.redraw()
    (hint,) = texts_of(app, "hint")
    assert kit.IDLE_STATUS["ATTACH"].split(" · ")[0] in hint
    move_to(app, 760, 450)
    app.handle("key_down", {"key": "Space", "code": "Space", "text": " ", "repeat": False})
    assert ui._hint(CONTEXT) == marking.status_hint()


def test_the_menus_have_the_editor_s_items(app):
    menus = dict(app.host.named("set_menu")[-1][1][0])
    file_ids = [i[0] for i in menus["File"] if i]
    assert "rok.welcome" in file_ids
    assert any(i.startswith("rok.open_recent(") for i in file_ids)  # the bird was opened
    edit_ids = [i[0] for i in menus["Edit"] if i]
    assert {"rok.undo", "rok.command_search", "rok.copy"} <= set(edit_ids)
    view_ids = [i[0] for i in menus["View"] if i]
    assert "rok.grid" in view_ids and "rok.view_settings" in view_ids


def test_view_top_looks_down(app):
    app.handle("menu", {"id": "rok.view(action='TOP')"})
    assert CONTEXT.view_state.view_direction()[2] < -0.99


def test_open_recent_from_the_menu_opens_the_file(app, tmp_path):
    app.handle("menu", {"id": f"rok.open_recent(filepath={str(RECON)!r})"})
    assert document.name == "robot.vox"


# --- the welcome screen -------------------------------------------------------------


def _due_welcome_timer(app) -> None:
    for t in app.timers:
        t.next_at = 0.0
        t.started -= welcome.OPEN_AFTER_S
    app.handle("tick", {})


def test_the_welcome_screen_opens_at_start_without_a_file_and_lists_recent_files(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    recent.remember(str(RECON))
    recent.remember(str(BIRD))
    a = _make_app(tmp_path, monkeypatch, None)
    try:
        assert not welcome.active()
        _due_welcome_timer(a)
        assert welcome.active()
        assert a.modals[-1].command_id == "rok.welcome"  # newest: it sees every event first
        a.redraw()
        shown = texts_of(a, "welcome")
        assert "bird.vox" in shown and "robot.vox" in shown
        assert "Rok" in shown
        # A click on a recent row opens that file.
        layout = welcome._layout(CONTEXT.region)
        row = next(r for r in layout.rows if r.name == "robot.vox")
        click_at(a, *centre_of(row.rect))
        assert not welcome.active()
        assert document.name == "robot.vox"
    finally:
        document.reset()
        state.__init__()
        _reset_ui()


def test_the_welcome_screen_does_not_open_when_a_file_was_given(app):
    _due_welcome_timer(app)
    assert not welcome.active()


def test_esc_closes_the_welcome_screen_and_n_opens_the_new_dialog(app):
    opkit.call("rok.welcome", "INVOKE")
    assert welcome.active()
    key(app, "Escape")
    assert not welcome.active()
    opkit.call("rok.welcome", "INVOKE")
    key(app, "n")
    assert not welcome.active() and app.dialog is not None


def test_view_voxel_grid_toggles_the_voxel_edges_and_its_check_mark(app):
    app.handle("menu", {"id": "rok.grid"})
    assert app.host.named("set_voxel_grid")[-1][1][0] is True
    view = dict(app.host.named("set_menu")[-1][1][0])["View"]
    assert ("rok.grid", "Voxel Grid", "CmdOrCtrl+G", 1) in view
    app.handle("menu", {"id": "rok.grid"})
    assert app.host.named("set_voxel_grid")[-1][1][0] is False


def test_ambient_occlusion_is_on_by_default_and_toggles_from_the_view_menu(app):
    """Occlusion over several cells is on unless switched off, and the choice stays."""
    from rok.app import looks, settings

    calls = app.host.named("set_occlusion")
    assert calls and calls[-1][1] == (looks.AO_STRENGTH, looks.AO_REACH)
    view = dict(app.host.named("set_menu")[-1][1][0])["View"] if False else None  # noqa: F841 (shape differs per host)
    app.handle("menu", {"id": "rok.occlusion"})
    assert app.host.named("set_occlusion")[-1][1] == (0.0, looks.AO_REACH)
    assert settings.get("ambient_occlusion") == "off"
    app.handle("menu", {"id": "rok.occlusion"})
    assert app.host.named("set_occlusion")[-1][1] == (looks.AO_STRENGTH, looks.AO_REACH)
    assert settings.get("ambient_occlusion") == "on"


def test_the_voxel_grid_choice_stays_for_the_next_start(app, tmp_path, monkeypatch):
    settings.put("star_pick", "CLICK")
    app.handle("menu", {"id": "rok.grid"})
    assert app.grid and settings.get("voxel_grid") == "on"
    assert settings.get("star_pick") == "CLICK"  # a write keeps the other settings
    again = _make_app(tmp_path, monkeypatch, BIRD)
    assert again.grid and ("set_voxel_grid", (True,), {}) in again.host.calls
    again.handle("menu", {"id": "rok.grid"})
    assert not _make_app(tmp_path, monkeypatch, BIRD).grid


# --- favourite colours ------------------------------------------------------


def test_the_star_marks_the_colour_as_a_favourite_for_every_file(app, tmp_path, monkeypatch):
    scene = document.scene
    document.colour = 3
    scene.palette[3, :3] = (10, 120, 200)
    opkit.call("rok.colour_popup")
    _press_part(app, "STAR")
    assert palette_ui.favourites() == [(10, 120, 200)]
    assert "0a78c8" in settings.get("favourite_colours")  # kept in the settings file
    # A new file, a new start: the favourite is still there.
    palette_ui.close_popup()
    other = _make_app(tmp_path, monkeypatch, None)
    opkit.call("rok.new", "EXEC", size=(8, 8, 8))
    assert palette_ui.favourites() == [(10, 120, 200)]
    opkit.call("rok.colour_popup")
    layout = palette_ui._current_layout(palette_ui._popup)
    assert "FAV:0" in layout and "FAV:1" not in layout  # the popup shows the row
    # A click on it puts the colour into a free slot of this file and picks it.
    _press_part(other, "FAV:0")
    assert tuple(int(c) for c in document.scene.palette[document.colour, :3]) == (10, 120, 200)
    palette_ui.close_popup()


def test_a_favourite_uses_the_slot_that_already_holds_it(app):
    scene = document.scene
    scene.palette[42, :3] = (1, 2, 3)
    document.colour = 5
    assert palette_ui.use_favourite((1, 2, 3)) == 42 and document.colour == 42


def test_the_star_again_unmarks_it(app):
    document.colour = 4
    document.scene.palette[4, :3] = (200, 50, 50)
    assert palette_ui.toggle_favourite() is True
    assert palette_ui.toggle_favourite() is False
    assert (200, 50, 50) not in palette_ui.favourites()


def test_a_favourite_gets_a_name_by_right_click_and_shows_it_on_hover(app, tmp_path, monkeypatch):
    """Name favourites; the name shows at once when the mouse rests on one."""
    scene = document.scene
    document.colour = 3
    scene.palette[3, :3] = (232, 105, 43)
    opkit.call("rok.colour_popup")
    _press_part(app, "STAR")
    # A right-click on the swatch opens the name dialog.
    x, y, w, h = palette_ui._current_layout(palette_ui._popup)["FAV:0"]
    move_to(app, x + w / 2, y + h / 2)
    app.handle("mouse_down", {"button": "right"})
    app.handle("mouse_up", {"button": "right"})
    assert app.dialog is not None and app.dialog.op.command_id == "rok.favourite_name"
    app.dialog.op.name = "Rust  orange "
    key(app, "Enter")
    assert palette_ui.favourite_name((232, 105, 43)) == "Rust orange"  # one line, trimmed
    # The tooltip: the name, drawn in its own layer while the mouse is on the swatch.
    move_to(app, x + w / 2, y + h / 2)
    assert palette_ui._popup.hover == "FAV:0"
    app.redraw()
    assert "Rust orange" in texts_of(app, "colour_popup_tip")
    # It stays: a new start reads it back.
    palette_ui.close_popup()
    _make_app(tmp_path, monkeypatch, None)
    assert palette_ui.favourite_name((232, 105, 43)) == "Rust orange"
    # Unmarking the favourite takes its name away.
    document.colour = 3
    document.scene.palette[3, :3] = (232, 105, 43)
    assert palette_ui.toggle_favourite() is False
    assert palette_ui.favourite_name((232, 105, 43)) == ""


def test_an_unnamed_favourite_says_how_to_name_it(app):
    document.colour = 3
    document.scene.palette[3, :3] = (1, 2, 3)
    palette_ui.toggle_favourite()
    opkit.call("rok.colour_popup")
    x, y, w, h = palette_ui._current_layout(palette_ui._popup)["FAV:0"]
    move_to(app, x + w / 2, y + h / 2)
    app.redraw()
    assert "#010203 · right-click to name" in texts_of(app, "colour_popup_tip")
    palette_ui.close_popup()


def test_the_strip_has_a_favourites_section_on_top():
    """The favourites get their own section in the palette strip."""
    from rok.core import hud

    lay = hud.strip_layout(1.0, (1, 2, 3), False, favourites=3)
    assert [k for k, _r in lay.favourites] == [0, 1, 2]
    top_of_used = max(r[1] + r[3] for _i, r in lay.used)
    assert all(r[1] >= top_of_used for _k, r in lay.favourites)  # above the colours
    assert lay.fav_divider is not None
    k, rect = lay.favourites[1]
    assert hud.strip_hit_test(lay, rect[0] + 2, rect[1] + 2) == ("FAV", 1)
    assert hud.strip_layout(1.0, (1, 2, 3), True, favourites=2).favourites  # the 256 grid keeps them
    assert hud.strip_layout(1.0, (1,), False).favourites == ()


def test_a_favourite_in_the_strip_is_picked_named_and_shown(app):
    document.colour = 3
    document.scene.palette[3, :3] = (90, 30, 160)
    palette_ui.toggle_favourite()
    palette_ui.set_favourite_name((90, 30, 160), "Royal purple")
    document.colour = 1
    app.redraw()
    from rok.app import overlay

    lay = overlay.strip_layout(CONTEXT.region)
    _k, (x, y, w, h) = lay.favourites[0]
    # The hover shows the name at once.
    move_to(app, x + w / 2, y + h / 2)
    app.redraw()
    assert any("Royal purple" in t for t in texts_of(app, "overlay.favourite_tip"))  # above the strip
    # A click picks it.
    click_at(app, x + w / 2, y + h / 2)
    assert tuple(int(c) for c in document.scene.palette[document.colour, :3]) == (90, 30, 160)
    # A right-click opens the name dialog.
    move_to(app, x + w / 2, y + h / 2)
    app.handle("mouse_down", {"button": "right"})
    app.handle("mouse_up", {"button": "right"})
    assert app.dialog is not None and app.dialog.op.command_id == "rok.favourite_name"
    key(app, "Escape")


# --- Clean Up Palette -------------------------------------------------------


def test_clean_up_palette_asks_then_empties_unused_colours_but_favourites(app):
    scene = document.scene
    used = set(int(i) for i in np.flatnonzero(document.used_counts()))
    free = [i for i in range(1, 256) if i not in used]
    keep, gone = free[0], free[1]
    scene.palette[keep, :3] = (12, 34, 56)
    scene.palette[gone, :3] = (200, 10, 10)
    document.colour = keep
    palette_ui.toggle_favourite()  # (12, 34, 56) is a favourite: it stays
    menus = dict(app.host.named("set_menu")[-1][1][0])
    assert ("rok.palette_cleanup", "Clean Up Palette…", "", -1) in menus["Edit"]
    before = scene.palette.copy()
    app.handle("menu", {"id": "rok.palette_cleanup"})
    (_n, (token, title, text, buttons), _kw), = app.host.named("ask")
    assert "Remove" in buttons and "Favourite colours stay" in text
    assert np.array_equal(scene.palette, before)  # nothing yet: it asks first
    app.handle("dialog", {"token": token, "answer": "Remove"})
    assert tuple(scene.palette[gone]) == (0, 0, 0, 0)
    assert tuple(scene.palette[keep, :3]) == (12, 34, 56)  # the favourite
    assert all(tuple(scene.palette[i]) == tuple(before[i]) for i in used if i)  # used colours stay
    opkit.call("rok.undo")  # one undo step brings them back
    assert np.array_equal(scene.palette, before)


def test_clean_up_palette_with_nothing_to_do_says_so(app):
    scene = document.scene
    used = document.used_counts()
    for i in range(1, 256):
        if not used[i]:
            scene.palette[i] = (0, 0, 0, 0)
    scene.colours.remove(scene.colours.unused_ids(used), used)
    assert palette_ui.cleanup_slots() == []
    app.handle("menu", {"id": "rok.palette_cleanup"})
    assert not app.host.named("ask") and "palette is clean" in (app.message or "")



def test_a_favourite_shows_only_in_the_favourites_row(app):
    """Favourite colours do not appear in the other sections."""
    from rok.app import overlay

    used = int(np.flatnonzero(document.used_counts()[1:])[0]) + 1  # a colour the bird uses
    document.colour = used
    rgb = tuple(int(c) for c in document.scene.palette[used, :3])
    app.redraw()
    assert used in [i for i, _r in overlay.strip_layout(CONTEXT.region).used]
    palette_ui.toggle_favourite()
    app.redraw()
    lay = overlay.strip_layout(CONTEXT.region)
    assert used not in [i for i, _r in lay.used] and used not in [i for i, _r in lay.made]
    assert len(lay.favourites) == 1 and palette_ui.favourites() == [rgb]


def test_animate_mode_turns_the_rail_violet():
    chrome._set_inks(1.0, 1.0)
    assert chrome.RAIL_BG == chrome._ANIM_INKS["RAIL_BG"]
    chrome._set_inks(1.0, 0.0)
    assert chrome.RAIL_BG == chrome._SCENE_INKS["RAIL_BG"]


# --- the Mirror Half preview ------------------------------------------------------------


def _open_mirror_half(app):
    move_to(app, *centre_of(widget(kit.MODEL_ID).rect))
    item = next(w for w in kit.flyout_widgets(chrome.layout_for(CONTEXT.region), kit.MODEL_ID)
                if w.id == kit.MIRROR_HALF_ID)
    click_at(app, *centre_of(item.rect))
    assert state.chrome_open == kit.MIRROR_HALF_ID


def _choice(ident: str):
    return next(w for w in kit.flyout_widgets(chrome.layout_for(CONTEXT.region), kit.MIRROR_HALF_ID)
                if w.id == ident)


def test_a_mirror_half_choice_shows_its_result_while_the_mouse_rests_on_it(app):
    from rok.core import grid_ops

    before = document.grid.voxels.copy()
    expect = grid_ops.mirror_half(before, 2, True)  # the bird is not symmetric in Z
    assert not np.array_equal(expect, before)
    depth = len(document.undo._undo)
    _open_mirror_half(app)
    move_to(app, *centre_of(_choice("MIRROR_HALF_Z_POS").rect))
    assert np.array_equal(document.grid.voxels, expect)  # the real voxels
    assert len(document.undo._undo) == depth and not document.modified
    move_to(app, *centre_of(_choice("MIRROR_HALF_Z_NEG").rect))
    assert np.array_equal(document.grid.voxels, grid_ops.mirror_half(before, 2, False))
    move_to(app, 1300, 500)  # off the rail
    assert np.array_equal(document.grid.voxels, before)
    assert len(document.undo._undo) == depth and not document.modified


def test_a_click_on_the_previewed_choice_is_one_undo_step(app):
    from rok.core import grid_ops

    before = document.grid.voxels.copy()
    _open_mirror_half(app)
    click_at(app, *centre_of(_choice("MIRROR_HALF_Z_POS").rect))
    assert np.array_equal(document.grid.voxels, grid_ops.mirror_half(before, 2, True))
    assert document.modified
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)


def test_an_undo_while_the_preview_shows_takes_the_preview_back_first(app):
    before = document.grid.voxels.copy()
    _open_mirror_half(app)
    move_to(app, *centre_of(_choice("MIRROR_HALF_Z_POS").rect))
    opkit.call("rok.undo")  # nothing to undo: the model is as it was
    assert np.array_equal(document.grid.voxels, before)


def test_a_choice_that_changes_nothing_still_shows_its_mirror_plane(app):
    from rok.app import mirror_preview
    from rok.core import grid_ops

    document.grid.voxels[...] = grid_ops.mirror_half(document.grid.voxels.copy(), 0, True)  # symmetric in X
    _open_mirror_half(app)
    move_to(app, *centre_of(_choice("MIRROR_HALF_X_NEG").rect))
    assert mirror_preview.shown_axis() == 0
    move_to(app, 1300, 500)
    assert mirror_preview.shown_axis() is None


def test_the_star_item_is_one_checkbox_that_flips_between_click_and_release(app):
    from rok.app import marking

    def item():
        view = dict(app.host.named("set_menu")[-1][1][0])["View"]
        return next(i for i in view if i is not None and i[1] == "Star: Pick on Click")

    assert item()[3] == 0 and marking.pick_mode() == marking.RELEASE
    app.handle("menu", {"id": item()[0]})
    assert marking.pick_mode() == marking.CLICK and item()[3] == 1
    app.handle("menu", {"id": item()[0]})
    assert marking.pick_mode() == marking.RELEASE


def test_quad_view_keeps_its_key_without_the_menu_item(app):
    from rok.app.handlers import registry

    assert ("rok.quad_view", {"type": "Q", "value": "PRESS", "ctrl": True, "alt": True}, None) in registry.keymap


def test_the_animation_menu_is_off_by_default_and_the_view_item_keeps_it_on(app, tmp_path, monkeypatch):
    from rok.app import chrome as app_chrome
    from rok.app import settings

    def item():
        view = dict(app.host.named("set_menu")[-1][1][0])["View"]
        return next(i for i in view if i is not None and i[1] == "Show Animation Menu")

    assert item()[3] == 0 and not app_chrome.chrome_state().anim_menu
    app.handle("menu", {"id": "rok.animation_menu"})
    assert item()[3] == 1 and app_chrome.chrome_state().anim_menu
    assert settings.get("animation_menu") == "on"
    again = _make_app(tmp_path, monkeypatch, BIRD)
    assert again.animation_menu  # it stays for the next start
    app.handle("menu", {"id": "rok.animation_menu"})
    assert not app.animation_menu and settings.get("animation_menu") == "off"


def test_an_import_into_a_full_palette_asks_before_it_takes_the_nearest_colours(app):
    crate = Path(__file__).parent / "fixtures" / "agent" / "crate.vox"
    scene = document.scene
    flat = document.grid.voxels.reshape(-1)
    flat[:255] = np.arange(1, 256, dtype=np.uint8)  # every slot holds voxels
    document.stroke_serial += 1
    count, palette = len(scene.models), scene.palette.copy()
    assert opkit.call("rok.model_add_file", "EXEC", filepath=str(crate)) == {"RUNNING"}
    (_n, (token, title, text, buttons), _kw), = app.host.named("ask")
    assert "full" in title and "Use Nearest" in buttons
    assert len(scene.models) == count  # nothing yet: it asks first
    app.handle("dialog", {"token": token, "answer": "Use Nearest"})
    assert len(scene.models) == count + 1 and np.array_equal(scene.palette, palette)


def test_the_colour_you_paint_with_stays_the_same_colour_in_a_new_scene(app):
    palette_ui.use_favourite((12, 200, 90))
    opkit.call("rok.new", "EXEC", size=(8, 8, 8))
    assert tuple(int(c) for c in document.scene.palette[document.colour]) == (12, 200, 90, 255)


class _Clock:
    def __init__(self) -> None:
        import time as real

        self.t = real.monotonic()

    def monotonic(self) -> float:
        return self.t


def _hold_still(app, clock, seconds: float) -> None:
    """Space held, the pointer still: the clock moves on and the star's timer ticks."""
    clock.t += seconds
    marking._play(marking._session, CONTEXT.region)


def test_a_long_still_hold_makes_the_arms_play_and_a_release_then_picks_nothing(app, monkeypatch):
    import types

    from rok.core import star_play

    clock = _Clock()
    monkeypatch.setattr(marking, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    wm = CONTEXT.prefs
    move_to(app, 760, 450)
    app.handle("key_down", {"key": "Space", "code": "Space", "text": " ", "repeat": False})
    marking._session.seed = next(seed for seed in range(5000) if star_play.schedule(seed, 10.0)[0][0] == "EAT")
    _hold_still(app, clock, 3.0)
    assert not marking.playing()  # a short hold is the quick menu
    eat = star_play.schedule(marking._session.seed, 10.0)[0][1]
    clock.t = marking._session.still_since + eat + 1.0
    marking._play(marking._session, CONTEXT.region)
    assert marking.playing()
    app.redraw()  # the play draws without an error
    clock.t = marking._session.still_since + eat + 5.5
    marking._play(marking._session, CONTEXT.region)
    assert marking.ui_pose(star_play.NAV_CUBE)[2]  # swallowed
    app.redraw()
    tool = getattr(wm, "rok_tool", "ATTACH")
    app.handle("key_up", {"key": "Space", "code": "Space", "text": "", "repeat": False})
    assert not marking.active() and getattr(wm, "rok_tool", "ATTACH") == tool  # nothing picked
    assert marking.ui_pose(star_play.NAV_CUBE) == (0.0, 0.0, False)  # the cube is back


def test_a_move_ends_the_play_and_the_star_works_again(app, monkeypatch):
    import types

    from rok.core import star_play

    clock = _Clock()
    monkeypatch.setattr(marking, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    move_to(app, 760, 450)
    app.handle("key_down", {"key": "Space", "code": "Space", "text": " ", "repeat": False})
    first, start = star_play.schedule(marking._session.seed, 10.0)[0]
    _hold_still(app, clock, start + star_play.DURATION[first] / 2)
    assert marking.playing()
    move_to(app, 765, 450)  # a move: the arms go back
    clock.t += star_play.EXIT_S + 0.01
    marking._play(marking._session, CONTEXT.region)
    assert not marking.playing()
    marking._session.since -= 5.0
    move_to(app, *_chip_centre("TOOL_ERASE"))
    app.handle("key_up", {"key": "Space", "code": "Space", "text": "", "repeat": False})
    assert CONTEXT.prefs.rok_tool == "ERASE"
