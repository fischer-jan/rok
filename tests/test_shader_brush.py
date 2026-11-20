"""The shader brush: the panel's Brush button, then Paint strokes that write
the shader's result where the brush goes (rok/app/shader_brush.py). On the
real GPU; skipped where there is none."""

from __future__ import annotations

import numpy as np
from test_app_shaders import gpu_ok, shader  # noqa: F401 (fixture)
from test_app_tools import app, click, empty, filled, key, on_top, put  # noqa: F401 (fixtures)

from rok.app import opkit, overlay, shader_brush, shader_panel, xs_ops
from rok.app.object import document
from rok.app.opkit import CONTEXT

# Paints every voxel 7 and grows a layer of 3 on top of the model.
PAINT_AND_GROW = ("float map(vec3 v) { if (voxel(v) != 0.0) return 7.0; "
                  "if (voxel(v - vec3(0.0, 0.0, 1.0)) != 0.0) return 3.0; return 0.0; }")


def _arm_from_the_panel(app, tmp_path, monkeypatch) -> None:
    sh = shader(tmp_path, PAINT_AND_GROW)
    monkeypatch.setattr(xs_ops, "shader_by_name", lambda name: sh)
    opkit.call("rok.shader_panel", "INVOKE", shader="t")
    layout = shader_panel._layout(shader_panel._session, CONTEXT.region)
    x, y, w, h = layout.brush
    click(app, {"x": x + w / 2, "y": float(app.region.height - (y + h / 2))})
    assert shader_panel._session is None and shader_brush.armed is not None


def test_the_brush_paints_the_shader_where_it_goes_one_stroke_one_undo(empty, gpu_ok, tmp_path, monkeypatch):  # noqa: F811
    floor = [(x, y, 0) for x in range(16) for y in range(16)]
    put(floor, colour=5)
    _arm_from_the_panel(empty, tmp_path, monkeypatch)
    wm = CONTEXT.prefs
    assert (wm.rok_tool, wm.rok_mode) == ("PAINT", "VOXEL")
    assert np.all(document.grid.voxels[:, :, 0] == 5)  # the panel put the model back: nothing yet
    wm.rok_brush_size = 2
    depth = len(document.undo._undo)
    click(empty, on_top(empty, (8, 8, 0)))
    v = document.grid.voxels
    assert v[8, 8, 0] == 7 and v[8, 8, 1] == 3  # repainted, and the layer grew where the brush went
    assert v[0, 0, 0] == 5 and v[0, 0, 1] == 0  # far from it: as it was
    assert len(document.undo._undo) == depth + 1
    opkit.call("rok.undo")
    assert np.all(document.grid.voxels[:, :, 0] == 5) and not document.grid.voxels[:, :, 1].any()
    key(empty, "Escape")
    assert shader_brush.armed is None
    document.colour = 9
    click(empty, on_top(empty, (8, 8, 0)))  # plain Paint again
    assert document.grid.voxels[8, 8, 0] == 9 and not document.grid.voxels[:, :, 1].any()


def test_the_brush_keeps_to_the_selection_and_the_mask(empty, gpu_ok, tmp_path, monkeypatch):  # noqa: F811
    floor = [(x, y, 0) for x in range(16) for y in range(16)]
    put(floor, colour=5)
    opkit.call("rok.deselect")
    document.selection.add_cells(np.array([(7, 8, 0)]))
    opkit.call("rok.mask_selection")  # one voxel masked
    document.selection.add_cells(np.array([(x, y, 0) for x in range(8) for y in range(16)]))  # the left half
    _arm_from_the_panel(empty, tmp_path, monkeypatch)
    CONTEXT.prefs.rok_brush_size = 4
    click(empty, on_top(empty, (8, 8, 0)))  # the brush reaches both halves
    v = document.grid.voxels
    assert v[6, 8, 0] == 7 and v[9, 8, 0] == 5  # only the selected half
    assert v[7, 8, 0] == 5  # and not the masked voxel
    assert not v[:, :, 1].any()  # a selection holds voxels only: no room to grow (Room 0)


def test_another_tool_puts_the_brush_away(empty, gpu_ok, tmp_path, monkeypatch):  # noqa: F811
    put([(x, y, 0) for x in range(16) for y in range(16)], colour=5)
    _arm_from_the_panel(empty, tmp_path, monkeypatch)
    assert overlay._shader_panel_open() is False
    key(empty, "t")  # Attach
    click(empty, on_top(empty, (8, 8, 0)))
    assert shader_brush.armed is None
    assert (8, 8, 1) in filled()  # a plain Attach
