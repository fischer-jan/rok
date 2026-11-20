"""xs shaders on the GPU (rok_native.ShaderRunner through
rok/app/xs_ops.py), the gallery and the settings panel. These tests run on
the real GPU; they skip where there is none."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app import opkit
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state
from rok.core import xs
from rok.core.palette import default_palette

FIXTURES = Path(__file__).parent / "fixtures"
BIRD = FIXTURES / "bird.vox"

rok_native = pytest.importorskip("rok_native")


@pytest.fixture(scope="module")
def gpu_ok():
    try:
        rok_native.ShaderRunner()
    except Exception as exc:  # no GPU here
        pytest.skip(f"no GPU: {exc}")


def shader(tmp_path: Path, body: str, args: str = ""):
    path = tmp_path / "t.txt"
    path.write_text(f"// xs_begin\n// author : 'Rok test'\n{args}// xs_end\n{body}\n")
    return xs.load(path)


def run(sh, voxels, colour=5, **kw):
    from rok.app import xs_ops

    return xs_ops.run(sh, voxels, colour=colour, palette=default_palette(), **kw)


# --- the runner -------------------------------------------------------------------------


def test_each_cell_gets_the_value_of_its_own_coordinates(gpu_ok, tmp_path):
    sh = shader(tmp_path, "float map(vec3 v) { return floor(v.x) + 10.0 * floor(v.y) + 100.0 * floor(v.z) + 1.0; }")
    out = run(sh, np.zeros((5, 4, 2), np.uint8))
    x, y, z = np.meshgrid(np.arange(5), np.arange(4), np.arange(2), indexing="ij")
    assert np.array_equal(out, (x + 10 * y + 100 * z + 1).astype(np.uint8))


def test_voxel_reads_the_neighbour_the_shader_asks_for(gpu_ok, tmp_path):
    rng = np.random.default_rng(1)
    vox = rng.integers(0, 6, size=(7, 6, 5), dtype=np.uint8)
    sh = shader(tmp_path, "float map(vec3 v) { return voxel(v - vec3(0.0, 1.0, 0.0)); }")
    out = run(sh, vox)
    assert np.array_equal(out[:, 1:], vox[:, :-1]) and not out[:, 0].any()
    sh = shader(tmp_path, "float map(vec3 v) { return voxel(v + vec3(0.0, 0.0, 1.0)); }")
    out = run(sh, vox)
    assert np.array_equal(out[:, :, :-1], vox[:, :, 1:]) and not out[:, :, -1].any()


def test_a_tall_model_spans_several_atlas_rows(gpu_ok, tmp_path):
    vox = np.zeros((3, 3, 300), np.uint8)
    vox[1, 1, ::7] = 9
    sh = shader(tmp_path, "float map(vec3 v) { return voxel(v); }")
    assert xs.atlas_layout(vox.shape)[1] > 1
    assert np.array_equal(run(sh, vox), vox)


def test_arguments_iterations_and_colours_arrive(gpu_ok, tmp_path):
    arg = "// arg : { var = 'm_step' name = 'Step' value = '3' range = '0 10' step = '1' precision = '0' }\n"
    sh = shader(tmp_path, "float map(vec3 v) { return voxel(v) + m_step + iIter; }", arg)
    out = run(sh, np.zeros((2, 2, 2), np.uint8), iterations=3)
    assert np.all(out == 3 + 4 + 5)  # 0+3+0, then +3+1, then +3+2
    sh = shader(tmp_path, "float map(vec3 v) { return color_sel(1.0); }")
    out = run(sh, np.zeros((2, 2, 2), np.uint8), colours=[20, 21, 22])
    assert np.all(out == 21)


def test_a_shader_that_does_not_compile_says_so(gpu_ok, tmp_path):
    from rok.app import xs_ops

    sh = shader(tmp_path, "float map(vec3 v) { return undefined_thing(v); }")
    with pytest.raises(xs_ops.ShaderError):
        run(sh, np.zeros((2, 2, 2), np.uint8))


def test_a_shader_that_leaves_cells_undrawn_is_refused(gpu_ok):
    from rok.app import xs_ops

    sh = xs.load(FIXTURES / "shaders" / "rok_discard.txt")
    with pytest.raises(xs_ops.GpuGaveUp):
        run(sh, np.zeros((8, 8, 4), np.uint8))


def test_the_loop_guard_counts_the_cells_it_stopped(gpu_ok):
    sh = xs.load(FIXTURES / "shaders" / "rok_hidden_loop.txt")
    info: dict = {}
    run(sh, np.zeros((4, 4, 2), np.uint8), info=info)
    assert info["stopped"] == 32


def test_every_bundled_shader_compiles_and_runs(gpu_ok):
    from rok.app import xs_ops

    vox = np.zeros((24, 24, 16), np.uint8)
    vox[:, :, :4] = 20
    for name, path in xs_ops.shader_files().items():
        sh = xs.load(path, name)
        out = run(sh, vox, colours=[10, 11, 12, 13])
        assert out.shape == vox.shape, name


# --- the app ----------------------------------------------------------------------------


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


@pytest.fixture
def app(tmp_path, monkeypatch, gpu_ok):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(BIRD))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    yield a
    document.reset()
    state.__init__()


def test_running_a_shader_changes_the_model_in_one_undo_step(app):
    before = document.grid.voxels.copy()
    result = opkit.call("rok.xs_run", "EXEC", shader="weathering/erode", a0=60.0)
    assert result == {"FINISHED"}
    assert not np.array_equal(document.grid.voxels, before)
    assert document.modified
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)


def test_a_shader_with_its_own_colours_brings_them_into_the_palette(app):
    palette_before = document.scene.palette.copy()
    opkit.call("rok.model_add", "EXEC")  # an empty model to grow rocks on
    assert opkit.call("rok.xs_run", "EXEC", shader="nature/hills") == {"FINISHED"}
    assert document.grid.voxels.any()
    used = set(np.unique(document.grid.voxels)) - {0}
    hills = xs.load(Path(__file__).parents[1] / "rok" / "shaders" / "nature" / "hills.txt").colours
    hexes = {tuple(int(v) for v in c) for c in hills.rgb}
    assert {tuple(document.scene.palette[i][:3]) for i in used} <= hexes
    assert not np.array_equal(document.scene.palette, palette_before)


def test_the_gallery_opens_and_a_click_on_a_card_opens_the_settings(app):
    from rok.app import shader_gallery, shader_panel

    assert opkit.call("rok.shader_gallery", "INVOKE") == {"RUNNING"}
    assert shader_gallery.active()
    app.redraw()
    layout = shader_gallery._layout(app.region, 0.0)
    card = next(c for c in layout.cards if shader_gallery.sg.visible(layout, c.rect))
    x, y, w, h = card.rect
    app.handle("mouse_move", {"x": x + w / 2, "y": app.region.height - (y + h / 2)})
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})
    assert not shader_gallery.active() and shader_panel.active()
    app.handle("key_down", {"key": "Escape", "code": "Escape", "text": "", "repeat": False})
    assert not shader_panel.active()


def test_cmd_k_lists_the_shaders(app):
    from rok.app import xs_ops

    names = [c.name for c in xs_ops.shader_commands()]
    assert "Shader: Nature / Rocks" in names and len(names) == len(xs_ops.shader_files())


def _at(app, rect, button="left"):
    x, y, w, h = rect
    app.handle("mouse_move", {"x": x + w / 2, "y": app.region.height - (y + h / 2)})
    app.handle("mouse_down", {"button": button})
    app.handle("mouse_up", {"button": button})


def test_the_panel_shows_the_colours_a_run_puts_in_and_changes_them(app, tmp_path, monkeypatch):
    """Display the colours a shader will use and let me change
    them. The swatches are the colours of the cells the run changes, e.g. the
    grass of cover; a pick changes them, a right-click puts one back."""
    from rok.app import palette_ui, settings, shader_panel

    palette_ui._save_favourites([(255, 0, 255)])  # a favourite to pick from
    opkit.call("rok.shader_panel", "INVOKE", shader="weathering/cover")
    session = shader_panel._session
    session.values["a0"] = 1.0  # Kind 1: grass
    session.values["a1"] = 100.0
    shader_panel._preview(session)
    grass = session.shader.colours.rgb[2]
    assert session.sources and session.sources[0] == grass  # its own colours first, the grass it uses
    assert session.colours == session.sources
    layout = shader_panel._layout(session, app.region)
    assert len(layout.swatches) == len(session.sources)
    app.redraw()
    _at(app, layout.swatches[0])  # Grass: its picker opens
    assert session.picker == 0
    picker = shader_panel._picker(session, app.region, layout)
    _at(app, picker["Q:0"])  # the favourite
    assert session.colours[0] == (255, 0, 255)
    _at(app, (0, 0, 2, 2))  # a click outside the picker closes it, not the panel
    assert session.picker is None and shader_panel.active()
    assert session.sources[0] == grass  # the swatch still stands for the grass
    base = session.base.copy()  # the model as it was when the panel opened
    _at(app, layout.run)
    after = document.grid.visible
    changed = after[(after > 0) & (after != base)]
    colours = {tuple(int(c) for c in document.scene.palette[i, :3]) for i in np.unique(changed)}
    assert (255, 0, 255) in colours and grass not in colours
    # Remembered for the next time.
    assert "weathering/cover" in settings.get("shader_recolours")
    opkit.call("rok.shader_panel", "INVOKE", shader="weathering/cover")
    session = shader_panel._session
    session.values["a0"] = 1.0
    session.values["a1"] = 100.0
    shader_panel._preview(session)
    assert session.sources[0] == grass and session.colours[0] == (255, 0, 255)
    # A right-click puts it back to the colour the run puts in.
    layout = shader_panel._layout(session, app.region)
    _at(app, layout.swatches[0], button="right")
    assert session.colours[0] == grass
    shader_panel._close()


def test_the_panel_shows_the_shades_a_shader_makes_of_the_model(app):
    """Hull plating paints shades of the model's grey: they are swatches too."""
    from rok.app import shader_panel

    opkit.call("rok.shader_panel", "INVOKE", shader="hard_surface/hull_plating")
    session = shader_panel._session
    shader_panel._preview(session)
    assert len(session.sources) >= 2
    own = set(session.shader.colours.rgb)
    assert not own & set(session.sources)  # no hazard stripes by default
    shader_panel._close()


def test_a_shader_leaves_masked_voxels_alone(app):
    from rok.app import select_ops

    before = document.grid.visible.copy()
    half = np.zeros(before.shape, bool)
    half[: before.shape[0] // 2] = True
    opkit.call("rok.deselect")
    document.selection.add_cells(np.argwhere(half & (before != 0)))
    opkit.call("rok.mask_selection")
    opkit.call("rok.deselect")
    masked = select_ops.mask_of()
    assert masked is not None and masked.any()
    assert opkit.call("rok.xs_run", "EXEC", shader="weathering/erode", a0=60.0) == {"FINISHED"}
    after = document.grid.visible
    assert np.array_equal(after[masked], before[masked])
    assert not np.array_equal(after[~half], before[~half])  # the rest did change


def test_the_panel_shows_the_result_live_and_puts_the_model_back(app):
    from rok.app import shader_panel

    before = document.grid.voxels.copy()
    palette = document.scene.palette.copy()
    depth = len(document.undo._undo)
    opkit.call("rok.shader_panel", "INVOKE", shader="hard_surface/battle_damage")
    session = shader_panel._session
    shown = document.grid.voxels.copy()
    assert not np.array_equal(shown, before)  # the result shows at once
    assert len(document.undo._undo) == depth and not document.modified  # but nothing is done yet
    assert "voxels changed" in session.status
    session.values["a1"] = 100.0  # more hits
    shader_panel._preview(session)
    more = document.grid.voxels.copy()
    assert not np.array_equal(more, shown)
    shader_panel._close()  # Esc, Back or a click outside
    assert np.array_equal(document.grid.voxels, before)
    assert np.array_equal(document.scene.palette, palette)
    assert len(document.undo._undo) == depth and not document.modified


def test_the_panels_previews_leave_no_colours_behind(app):
    from rok.app import overlay, shader_panel

    store = document.scene.colours
    entries = dict(store.entries)
    opkit.call("rok.shader_panel", "INVOKE", shader="nature/jungle")
    session = shader_panel._session
    for seed in (1.0, 2.0, 3.0):
        session.values["a7"] = seed
        shader_panel._preview(session)
        overlay.strip_entries()  # the strip draws: it takes on the colours the preview shows
    shader_panel._close()
    assert store.entries == entries  # the previews left no colour behind


def test_run_from_the_panel_keeps_what_the_preview_showed_in_one_undo_step(app):
    from rok.app import shader_panel

    before = document.grid.voxels.copy()
    depth = len(document.undo._undo)
    opkit.call("rok.shader_panel", "INVOKE", shader="weathering/erode")
    shown = document.grid.voxels.copy()
    app.handle("key_down", {"key": "Enter", "code": "Enter", "text": "", "repeat": False})
    assert shader_panel._session is None
    assert np.array_equal(document.grid.voxels, shown)
    assert len(document.undo._undo) == depth + 1 and document.modified
    opkit.call("rok.undo")
    assert np.array_equal(document.grid.voxels, before)


def test_the_panel_stands_at_the_right_edge_and_the_nav_cube_makes_room(app):
    from rok.app import nav_cube, shader_panel
    from rok.app.opkit import CONTEXT

    assert not nav_cube._hidden()
    opkit.call("rok.shader_panel", "INVOKE", shader="weathering/erode")
    region = CONTEXT.region
    x, y, w, h = shader_panel._layout(shader_panel._session, region).rect
    assert x + w <= region.width and x > region.width / 2  # the right side: the model stays in sight
    assert y + h <= region.height
    assert nav_cube._hidden()
    shader_panel._close()
    assert not nav_cube._hidden()


def test_only_on_top_changes_only_top_voxels_and_run_keeps_the_scope(app):
    from rok.app import shader_panel
    from rok.core import shader_scope

    before = document.grid.voxels.copy()
    top = shader_scope.face_mask(before, "TOP")
    opkit.call("rok.shader_panel", "INVOKE", shader="hard_surface/battle_damage")
    session = shader_panel._session
    session.faces = {"TOP"}
    shader_panel._preview(session)
    shown = document.grid.voxels.copy()
    changed = shown != before
    assert changed.any() and not (changed & ~top).any()
    assert "top" in shader_scope.Where(faces=("TOP",)).summary(False)
    app.handle("key_down", {"key": "Enter", "code": "Enter", "text": "", "repeat": False})
    assert np.array_equal(document.grid.voxels, shown)  # Run did what the preview showed


def test_only_colours_and_not_selected(app):
    before = document.grid.voxels.copy()
    counts = np.bincount(before.ravel(), minlength=256)
    counts[0] = 0
    target, keep = (int(c) for c in np.argsort(-counts)[:2])  # the two most used colours
    opkit.call("rok.xs_run", "EXEC", shader="hard_surface/battle_damage", a0=3.0, a1=60.0,
               only_colours=str(target))
    changed = document.grid.voxels != before
    assert changed.any() and np.all(before[changed] == target)
    opkit.call("rok.undo")
    opkit.call("rok.deselect")
    document.selection.add_cells(np.argwhere(before == keep))
    opkit.call("rok.xs_run", "EXEC", shader="hard_surface/battle_damage", a0=3.0, a1=60.0,
               scope="NOT_SELECTION")
    changed = document.grid.voxels != before
    assert changed.any() and not np.any(before[changed] == keep)


def test_room_lets_a_shader_build_out_from_the_selection(app, tmp_path, monkeypatch):
    from rok.app import xs_ops

    # A shader that fills every empty cell next to a voxel.
    sh = shader(tmp_path, "float map(vec3 v) { if (voxel(v) != 0.0) return voxel(v); "
                          "for (int k = 0; k < 3; k++) { vec3 d = vec3(0.0); d[k] = 1.0; "
                          "if (voxel(v + d) != 0.0 || voxel(v - d) != 0.0) return 5.0; } return 0.0; }")
    monkeypatch.setattr(xs_ops, "shader_by_name", lambda name: sh)
    before = document.grid.voxels.copy()
    cell = tuple(int(c) for c in np.argwhere(before)[0])
    opkit.call("rok.deselect")
    document.selection.add_cells(np.array([cell]))
    opkit.call("rok.xs_run", "EXEC", shader="t")
    assert np.array_equal(document.grid.voxels, before)  # a selection holds only voxels: no room to add
    opkit.call("rok.xs_run", "EXEC", shader="t", room=1)
    added = np.argwhere((document.grid.voxels != 0) & (before == 0))
    assert len(added) and np.all(np.abs(added - np.array(cell)).sum(axis=1) == 1)  # next to the selected voxel


def test_feather_thins_a_shader_out_toward_the_edge_of_the_selection(app, tmp_path, monkeypatch):
    from rok.app import xs_ops

    sh = shader(tmp_path, "float map(vec3 v) { return voxel(v) != 0.0 ? 7.0 : 0.0; }")  # paints everything
    monkeypatch.setattr(xs_ops, "shader_by_name", lambda name: sh)
    opkit.call("rok.new", "EXEC", size=(24, 24, 4))
    cells = np.argwhere(np.ones((24, 24, 2), bool))
    document.grid.set_voxels(cells, 5)
    opkit.call("rok.deselect")
    patch = np.argwhere(np.ones((16, 16, 1), bool)) + (4, 4, 1)
    document.selection.add_cells(patch)
    opkit.call("rok.xs_run", "EXEC", shader="t", feather=4)
    v = document.grid.voxels
    assert np.all(v[8:16, 8:16, 1] == 7)  # the middle: all of it
    assert 0 < np.count_nonzero(v[4, 4:20, 1] == 7) < 10  # the edge: thinned out
    assert np.count_nonzero(v == 7) < len(patch)  # never outside the selection, and fewer than all
    assert not np.any(v[:, :, 0] == 7)


FILL = "float map(vec3 v) { return voxel(v) != 0.0 ? voxel(v) : 3.0; }"  # fills every empty cell


def test_a_shader_builds_in_a_placed_volume_and_nowhere_else(app, tmp_path, monkeypatch):
    from rok.app import xs_ops

    sh = shader(tmp_path, FILL)
    monkeypatch.setattr(xs_ops, "shader_by_name", lambda name: sh)
    opkit.call("rok.new", "EXEC", size=(16, 16, 8))
    opkit.call("rok.xs_run", "EXEC", shader="t", scope="VOLUME", volume="2,2,0,5,6,3")
    v = document.grid.voxels
    assert np.all(v[2:6, 2:7, 0:4] == 3) and np.count_nonzero(v) == 4 * 5 * 4
    opkit.call("rok.undo")
    opkit.call("rok.xs_run", "EXEC", shader="t", scope="VOLUME", volume="0,0,0,8,8,8",
               volume_shape="SPHERE")
    v = document.grid.voxels
    assert v[4, 4, 4] == 3 and v[0, 0, 0] == 0


def test_the_panel_shows_a_volume_and_a_face_handle_drag_moves_it(app, tmp_path, monkeypatch):
    from rok.app import shader_panel, xs_ops
    from rok.app.opkit import CONTEXT

    sh = shader(tmp_path, FILL)
    monkeypatch.setattr(xs_ops, "shader_by_name", lambda name: sh)
    opkit.call("rok.new", "EXEC", size=(16, 16, 8))
    document.grid.set_voxels(np.argwhere(np.ones((16, 16, 1), bool)), 5)  # a floor
    opkit.call("rok.shader_panel", "INVOKE", shader="t")
    session = shader_panel._session
    session.where_open = True
    layout = shader_panel._layout(session, CONTEXT.region)
    rect = dict(layout.scope_chips)["VOLUME"]
    _click_region(app, rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
    assert session.scope == "VOLUME" and session.volume == [0, 0, 0, 15, 15, 0]  # the voxels' box
    assert np.all(document.grid.voxels[:, :, 0] == 5)  # the volume holds only the floor: nothing to fill
    region = CONTEXT.region
    top = shader_panel._to_region(region, shader_panel._face_centre(session.volume, 2, 1))
    up = shader_panel._to_region(region, shader_panel._face_centre(session.volume, 2, 1) + (0, 0, 3))
    _move_region(app, *top)
    assert session.vol_hover == (2, 1)
    app.handle("mouse_down", {"button": "left"})
    _move_region(app, *up)
    app.handle("mouse_up", {"button": "left"})
    assert session.volume[5] == 3  # the top face went up three cells
    assert np.all(document.grid.voxels[:, :, 1:4] == 3)  # and the live preview filled the volume
    assert not document.grid.voxels[:, :, 4:].any()
    shader_panel._close()
    assert not document.grid.voxels[:, :, 1:].any()


def _move_region(app, rx: float, ry: float) -> None:
    app.handle("mouse_move", {"x": float(rx), "y": float(app.region.height - ry)})


def _click_region(app, rx: float, ry: float) -> None:
    _move_region(app, rx, ry)
    app.handle("mouse_down", {"button": "left"})
    app.handle("mouse_up", {"button": "left"})


def test_resting_on_an_arguments_name_shows_its_tooltip(app, monkeypatch):
    from rok.app import shader_panel

    clock = [1000.0]
    monkeypatch.setattr(shader_panel.time, "monotonic", lambda: clock[0])
    opkit.call("rok.shader_panel", "INVOKE", shader="nature/woodland")
    session = shader_panel._session
    layout = shader_panel._layout(session, app.region)
    density = layout.control("a0")
    assert density.label == "Density" and density.tip
    x, y, w, h = density.label_rect
    app.handle("mouse_move", {"x": x + 4, "y": app.region.height - (y + h / 2)})
    assert session.hover == ("LABEL", "a0")

    def drawn() -> str:
        app.host.calls.clear()
        app.redraw()
        return " ".join(str(a) for c in app.host.calls for a in c[1] + tuple(c[2].values()))

    first_words = " ".join(density.tip.split()[:3])
    assert first_words not in drawn()  # not yet due
    clock[0] += 1.0
    assert first_words in drawn()
    shader_panel._close()


def test_the_preview_switch_keeps_a_big_model_still_until_you_run(app):
    """A shader on a big model is slow: with the preview off the panel runs
    nothing on open or on a change; on again, the result shows at once."""
    from rok.app import settings, shader_panel

    settings.put(shader_panel.PREVIEW_SETTING, "off")
    before = document.grid.voxels.copy()
    opkit.call("rok.shader_panel", "INVOKE", shader="weathering/cover")
    session = shader_panel._session
    assert np.array_equal(document.grid.voxels, before) and not session.shown  # nothing ran
    assert session.status == shader_panel.PREVIEW_OFF_NOTE
    session.values["a1"] = 100.0
    app.handle("mouse_move", {"x": 10.0, "y": 10.0})  # an event: a change would preview now
    assert np.array_equal(document.grid.voxels, before)
    layout = shader_panel._layout(session, app.region)
    _at(app, layout.preview_switch)  # on: the result shows
    assert settings.get(shader_panel.PREVIEW_SETTING) == "on"
    assert session.shown and not np.array_equal(document.grid.voxels, before)
    _at(app, shader_panel._layout(session, app.region).preview_switch)  # off: the model goes back
    assert np.array_equal(document.grid.voxels, before) and settings.get(shader_panel.PREVIEW_SETTING) == "off"
    shader_panel._close()
    settings.put(shader_panel.PREVIEW_SETTING, "on")
