"""The new UI's widget kit: layout, hit testing, focus, tooltips and glyphs."""

import math

import numpy as np
import pytest

from rok.core.chrome import (
    STAMP_ID,
    DEFAULT_MODEL_ITEMS,
    MODEL_COMMAND_IDS,
    BUTTON_KIND,
    CHIP_KIND,
    COMMANDS,
    GLYPH_NAMES,
    IDLE_STATUS,
    HEADER_H,
    MODE_NAMES,
    RAIL_WIDTH,
    rail_height,
    RING_GLYPHS,
    MODEL_FLYOUT_IDS,
    MODEL_HEADER,
    MODEL_ID,
    SCENE_TOOLS_ID,
    SCENE_TOOL_IDS,
    TOOLS_HEADER,
    FLIP_IDS,
    LEVEL_IDS,
    LEVEL_MODEL_ID,
    LEVEL_SCENE_ID,
    SCENE_FIT_ID,
    SCENE_DELETE_ID,
    SCENE_FLIP_ID,
    SEGMENT_KIND,
    LEVEL_MODEL,
    LEVEL_SCENE,
    NEW_FLYOUT_IDS,
    NEW_HEADER,
    NEW_ID,
    RAIL_INSET,
    SCENE_CUT_ID,
    SCENE_PICK_ID,
    COMMAND_OF_MODEL_ID,
    TOOL_FLYOUT_IDS,
    TOOL_MODE_OF_ID,
    LAYERS_HEADER,
    LAYERS_ID,
    SIZE_HEADER,
    SIZE_ID,
    SIZE_KNOB,
    VISIBLE_HEADER,
    VISIBLE_ID,
    VISIBLE_MODES,
    SLIDER_KIND,
    BRUSH_SIZES,
    PRIMITIVE_COLUMNS,
    PRIMITIVE_IDS,
    PRIMITIVES_ID,
    SOLID_OF_ID,
    primitive_bridge,
    TOOL_HIT,
    TOOL_IDS,
    TOOL_KEYS,
    TOOLTIP_DELAY_S,
    ChromeLayout,
    ChromeState,
    chrome_layout,
    command_hit,
    command_layout,
    command_match,
    contains,
    edge_line,
    flyout_bridge,
    split_tooltip,
    flyout_widgets,
    focus_ids,
    focus_step,
    glyph_lines,
    hit_test,
    rect_lines,
    right_edge_line,
    tooltip_due,
    tooltip_rect,
    ANCHOR_ID,
    ANIM_COLOUR_ID,
    CAPTION_KIND,
    CARD_KIND,
    LEVEL_ANIM_ID,
    MOTION_ID,
    PRESET_IDS,
    PRESET_OF_ID,
    preset_tip,
)
from rok.core.geom2d import in_rect, rect_tris
from rok.core.edit import (
    ATTACH,
    BOX,
    CENTER,
    ERASE,
    FACE,
    LASSO,
    LINE,
    MODES,
    PAINT,
    PATTERN,
    RECT,
    SELECT,
    VOXEL,
)
from rok.core.shape import CONE, CYLINDER, SOLID_ITEMS, SOLIDS, SPHERE

WIDTH = 1200
HEIGHT = 1500  # a real view at 2x (a 1618 px window less the menu and status bars)


def layout(scale=1.0, width=WIDTH, height=HEIGHT, **kwargs) -> ChromeLayout:
    return chrome_layout(scale, ChromeState(**kwargs), width, height)


def centre(widget) -> tuple[float, float]:
    x, y, w, h = widget.rect
    return (x + w / 2.0, y + h / 2.0)


# --- the left rail ----------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_the_rail_is_a_full_height_left_column(scale):
    lay = layout(scale)
    x, y, w, h = lay.rect
    assert (x, y) == (0, 0)
    assert w == round(RAIL_WIDTH * scale)
    assert h == HEIGHT
    assert lay.hit_rect == lay.rect
    assert w < WIDTH  # not a horizontal top strip


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_every_widget_is_whole_pixels_inside_the_rail(scale):
    lay = layout(scale, tool=SELECT)
    sx, sy, sw, sh = lay.rect
    assert lay.widgets
    for widget in lay.widgets:
        x, y, w, h = widget.rect
        assert all(isinstance(v, int) for v in widget.rect), widget.id
        assert sx <= x and x + w <= sx + sw, widget.id
        assert sy <= y and y + h <= sy + sh, widget.id


@pytest.mark.parametrize("height", [400, 600, 900])
def test_a_short_region_squeezes_the_rail_until_it_fits(height):
    """A split screen at 2x: every widget stays on the screen, in its order."""
    lay = layout(2.0, height=height, tool=SELECT, mode=VOXEL)
    assert lay.rect[3] == height
    assert 0.0 < lay.scale < 2.0
    sx, sy, sw, sh = lay.rect
    for widget in lay.widgets:
        x, y, w, h = widget.rect
        assert sx <= x and x + w <= sx + sw, widget.id
        assert sy <= y and y + h <= sy + sh, widget.id
    ids = [w.id for w in lay.widgets]
    assert ids == [w.id for w in layout(2.0, tool=SELECT, mode=VOXEL).widgets]
    assert lay.by_id("TOOL_ATTACH").rect[1] > lay.by_id(LAYERS_ID).rect[1] > lay.by_id(SIZE_ID).rect[1]


def test_a_tall_region_keeps_the_ui_scale():
    lay = layout(2.0, height=HEIGHT, tool=SELECT, mode=VOXEL)
    assert lay.scale == 2.0
    assert rail_height(ChromeState(tool=SELECT, mode=VOXEL)) * 2.0 <= HEIGHT


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_tool_tick_and_chip_hit_sizes(scale):
    lay = layout(scale, tool=SELECT, mode=VOXEL)
    for ident in TOOL_IDS:
        _x, _y, w, h = lay.by_id(ident).rect
        assert (w, h) == (round(TOOL_HIT * scale), round(TOOL_HIT * scale))
    tools = [lay.by_id(i) for i in TOOL_IDS]
    assert tools[0].rect[1] > tools[-1].rect[1]  # Attach above Select
    slider = lay.by_id(SIZE_ID)
    assert slider.rect[1] < lay.by_id(LAYERS_ID).rect[1]  # the slider sits at the bottom


def test_widgets_do_not_overlap():
    lay = layout(tool=SELECT)
    rects = [w.rect for w in lay.widgets]
    for i, (x0, y0, w0, h0) in enumerate(rects):
        for x1, y1, w1, h1 in rects[i + 1 :]:
            apart = x0 + w0 <= x1 or x1 + w1 <= x0 or y0 + h0 <= y1 or y1 + h1 <= y0
            assert apart


def test_the_action_cluster_is_always_the_same_four():
    for tool in (ATTACH, ERASE, PAINT, SELECT):
        for mode in (VOXEL, BOX, PATTERN):
            lay = layout(tool=tool, mode=mode)
            buttons = [w for w in lay.widgets if w.kind == BUTTON_KIND]
            assert [w.id for w in buttons] == list(TOOL_IDS)
            assert [w.active for w in buttons] == [
                TOOL_IDS[ident] == tool for ident in TOOL_IDS
            ]


def test_idle_status_names_option_and_the_hold_keys():
    assert "⌥: erase" in IDLE_STATUS[ATTACH]
    assert "Ctrl, ⌘, ⌥ + move: orbit, pan, zoom" in IDLE_STATUS[ATTACH]
    assert "Space: star" in IDLE_STATUS[ATTACH]
    assert "Ctrl, ⌘, ⌥ + move: orbit, pan, zoom" in IDLE_STATUS[ERASE]
    assert "⌥: erase" in IDLE_STATUS[PAINT]
    assert "⌥ or right-click: deselect" in IDLE_STATUS[SELECT]  # Select never erases


def test_the_cluster_keeps_its_place_whatever_the_chips_show():
    plain = layout(tool=ATTACH, mode=VOXEL)
    busy = layout(tool=SELECT, mode=VOXEL, brush_size=4)
    for ident in TOOL_IDS:
        assert plain.by_id(ident).rect == busy.by_id(ident).rect


def test_a_tooltip_at_the_right_edge_opens_to_the_left():
    from rok.core.chrome import Widget, tooltip_rect

    button = Widget(id="EDGE", kind=CHIP_KIND, rect=(WIDTH - 40, 400, 30, 30), tooltip="Mirror X — Copy your edits")
    tx, _ty, tw, _th = tooltip_rect(button, layout(), WIDTH)
    assert tx + tw <= button.rect[0]


def test_the_cut_ticks_are_gone():
    lay = layout()
    assert not any(w.id.startswith("SECTION") for w in lay.widgets)  # the Cut ticks are gone


def test_the_model_flyout_shows_the_chosen_commands():
    every = flyout_widgets(layout(model_items=tuple(MODEL_COMMAND_IDS)), MODEL_ID)
    assert [w.id for w in every] == list(MODEL_FLYOUT_IDS)
    two = flyout_widgets(layout(model_items=("twist", "loop")), MODEL_ID)
    assert [COMMAND_OF_MODEL_ID[w.id] for w in two] == ["loop", "twist"]  # in the usual order


@pytest.mark.parametrize("items", [DEFAULT_MODEL_ITEMS, tuple(MODEL_COMMAND_IDS), ("fill_all", "clear_all")])
def test_the_model_flyout_is_a_block(items):
    """About as many rows as columns, not one long row."""
    lay = layout(model_items=items)
    row = flyout_widgets(lay, MODEL_ID)
    xs = sorted({w.rect[0] for w in row})
    ys = sorted({w.rect[1] for w in row})
    cols = math.ceil(math.sqrt(len(row)))
    assert len(xs) == cols and len(ys) == -(-len(row) // cols)
    button = lay.by_id(MODEL_ID).rect
    assert max(ys) == button[1] + (button[3] - row[0].rect[3]) // 2  # the first row level with the button
    assert min(ys) >= 0  # inside the view
    # A choice grid hangs under the whole block.
    choices = flyout_widgets(lay, "MODEL_SCALE") if "scale" in items else ()
    if choices:
        extra = [w for w in choices if w not in row]
        assert extra and max(w.rect[1] for w in extra) < min(ys)


def test_the_model_button_opens_the_whole_model_tools():
    assert layout(level=LEVEL_SCENE).by_id(MODEL_ID) is None  # its tools edit one model's voxels
    lay = layout()
    button = lay.by_id(MODEL_ID)
    assert button is not None and button.glyph == "MODEL" and "Whole model" in button.tooltip
    assert button.rect[1] < min(lay.by_id(i).rect[1] for i in TOOL_IDS)  # under the tools
    row = flyout_widgets(lay, MODEL_ID)
    # Loop, Repeat, Diagonal, Lathe and Twist live in Cmd+K.
    assert [COMMAND_OF_MODEL_ID[w.id] for w in row] == list(DEFAULT_MODEL_ITEMS)
    assert not {"loop", "repeat", "diagonal", "lathe", "twist"} & set(DEFAULT_MODEL_ITEMS)
    # Split off the selection first, then the fills.
    assert [COMMAND_OF_MODEL_ID[w.id] for w in row][:4] == ["model_split", "fill_all", "fill_volume", "clear_all"]
    assert all(w.glyph and split_tooltip(w.tooltip)[1] for w in row)
    assert not any(w.active for w in row)
    assert all(w.rect[0] >= lay.rect[2] for w in row)
    assert hit_test(lay, *centre(row[0]), MODEL_ID).id == row[0].id


# --- the column headers ------------------------------------------------------------


@pytest.mark.parametrize("mode", [VOXEL, BOX, PATTERN, LINE])
def test_no_mode_chip_the_tools_run_into_model(mode):
    """The style has no chip: the tool flyouts and the Tab wheel pick it, the
    readout names it. Every style is in some tool's flyout."""
    lay = layout(tool=ATTACH, mode=mode)
    assert not any(
        w.kind == CHIP_KIND and w.id not in (MODEL_ID, LAYERS_ID, VISIBLE_ID, STAMP_ID)
        for w in lay.widgets
    )
    covered = {TOOL_MODE_OF_ID[i][1] for ids in TOOL_FLYOUT_IDS.values() for i in ids}
    assert covered == set(RING_GLYPHS)
    last_tool = min(lay.by_id(i).rect[1] for i in TOOL_IDS)
    assert lay.divider[1] < last_tool and lay.by_id(MODEL_ID).rect[1] < lay.divider[1]


def test_the_column_headers_name_model_layers_and_size_above_their_widgets():
    lay = layout(mode=VOXEL)
    assert [text for text, _rect in lay.headers] == [
        MODEL_HEADER, LAYERS_HEADER, SIZE_HEADER,
    ]
    rects = dict(lay.headers)
    model_rect, layers_rect, size_rect = (rects[k] for k in (MODEL_HEADER, LAYERS_HEADER, SIZE_HEADER))
    assert model_rect[1] > lay.by_id(MODEL_ID).rect[1] > layers_rect[1] > lay.by_id(LAYERS_ID).rect[1]
    assert model_rect[1] < lay.divider[1]
    assert all(r[3] == HEADER_H for r in rects.values())
    assert size_rect[1] >= lay.by_id(SIZE_ID).rect[1] + lay.by_id(SIZE_ID).rect[3]
    for _text, (x, y, w, h) in lay.headers:
        assert 0 <= x and x + w <= lay.rect[2] and 0 <= y and y + h <= lay.rect[3]


def test_the_stroke_button_sits_above_the_size_field_for_the_stroke_tools_only():
    from rok.core.chrome import STROKE_ID, rail_height

    for tool in (ATTACH, ERASE, PAINT):
        st = ChromeState(tool=tool, mode=VOXEL)
        lay = chrome_layout(1.0, st, WIDTH, int(rail_height(st)))
        button, size = lay.by_id(STROKE_ID), lay.by_id(SIZE_ID)
        assert button is not None and button.rect[1] > size.rect[1] + size.rect[3]  # above it (y grows up)
    st = ChromeState(tool=SELECT, mode=VOXEL)
    assert chrome_layout(1.0, st, WIDTH, int(rail_height(st))).by_id(STROKE_ID) is None
    assert rail_height(st) < rail_height(ChromeState(tool=ATTACH, mode=VOXEL))


def test_the_size_slider_exists_in_voxel_mode_for_every_tool():
    for tool in (ATTACH, ERASE, PAINT, SELECT):
        slider = layout(tool=tool, mode=VOXEL, brush_size=3).by_id(SIZE_ID)
        assert slider.kind == SLIDER_KIND
        assert slider.label == "3"
    assert layout(tool=SELECT, mode=BOX).by_id(SIZE_ID) is None
    assert layout(tool=ATTACH, mode=LINE).by_id(SIZE_ID) is None


def test_the_size_slider_has_a_header_over_it():
    lay = layout(tool=ATTACH, mode=VOXEL)
    slider = lay.by_id(SIZE_ID)
    rects = dict(lay.headers)
    assert rects[SIZE_HEADER][1] >= slider.rect[1] + slider.rect[3]
    assert rects[SIZE_HEADER][1] + rects[SIZE_HEADER][3] < lay.by_id(LAYERS_ID).rect[1]
    assert SIZE_HEADER not in dict(layout(tool=ATTACH, mode=BOX).headers)


def test_the_size_slider_has_no_flyout():
    lay = layout(tool=ATTACH, mode=VOXEL)
    assert flyout_widgets(lay, SIZE_ID) == ()


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_size_field_changes_by_one_step_of_drag(scale):
    """The Size slider is a number you drag: a step of SIZE_DRAG_PX points per size, right to grow, left to shrink."""
    from rok.core.chrome import SIZE_DRAG_PX, size_from_drag

    step = SIZE_DRAG_PX * scale
    assert size_from_drag(3, 0.0, scale) == 3
    assert size_from_drag(3, step * 0.9, scale) == 3  # a whole step first
    assert size_from_drag(3, step * 1.0, scale) == 4
    assert size_from_drag(3, step * 2.5, scale) == 5
    assert size_from_drag(3, -step * 2.0, scale) == 1
    assert size_from_drag(3, -step * 50, scale) == BRUSH_SIZES[0]  # clamped
    assert size_from_drag(3, step * 50, scale) == BRUSH_SIZES[-1]
    box = layout(scale, tool=ATTACH, mode=VOXEL).by_id(SIZE_ID)
    assert box.rect[3] < 40 * scale  # a field, not a long track


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_a_horizontal_slider_spreads_its_stops_along_x_and_picks_by_x(scale):
    from rok.core.chrome import Widget, slider_at, slider_stops

    values = ("A", "B", "C", "D")
    widget = Widget(id="H", kind=SLIDER_KIND, rect=(20.0, 30.0, 200.0, 40.0))
    x, _y, w, _h = widget.rect
    stops = slider_stops(widget, values, scale, True)
    assert len(stops) == len(values)
    ys = {sy for _sx, sy in stops}
    assert len(ys) == 1  # every stop sits on the same horizontal line
    xs = [sx for sx, _sy in stops]
    gaps = np.diff(xs)
    assert (gaps > 0).all()  # A at the left
    assert np.allclose(gaps, gaps[0])
    knob = SIZE_KNOB * scale
    assert xs[0] - knob >= x and xs[-1] + knob <= x + w  # the knob fits at both ends

    for value, (sx, _sy) in zip(values, stops):
        assert slider_at(widget, values, sx, scale, True) == value
    step = stops[1][0] - stops[0][0]
    assert slider_at(widget, values, stops[2][0] + 0.4 * step, scale, True) == values[2]
    assert slider_at(widget, values, stops[2][0] + 0.6 * step, scale, True) == values[3]
    assert slider_at(widget, values, -1000, scale, True) == values[0]
    assert slider_at(widget, values, 100000, scale, True) == values[-1]




@pytest.mark.parametrize("mode", [BOX, CENTER, VOXEL, LINE])
def test_there_is_no_solid_chip_on_the_rail(mode):
    lay = layout(tool=ATTACH, mode=mode, solid=CYLINDER)
    assert not any(w.glyph == "CYLINDER" or "SOLID" in w.id for w in lay.widgets)


def test_there_is_no_pattern_chip():
    lay = layout(mode=PATTERN)
    assert not any("PATTERN" in w.id for w in lay.widgets)


# --- hit testing --------------------------------------------------------------


def test_hit_test_finds_every_widget_by_its_centre():
    lay = layout(tool=SELECT, mode=VOXEL)
    for widget in lay.widgets:
        assert hit_test(lay, *centre(widget)) is widget


def test_hit_test_misses_outside_the_rail():
    lay = layout()
    assert hit_test(lay, 5.0, 5.0) is None  # empty plate, no widget
    assert hit_test(lay, WIDTH - 5.0, HEIGHT - 5.0) is None  # top-right of the viewport
    assert hit_test(lay, RAIL_WIDTH + 8.0, HEIGHT - 24.0) is None  # just right of the rail


def test_the_rail_swallows_a_press_beside_a_widget_but_not_the_viewport():
    lay = layout()
    last = lay.widgets[-1]
    beside = (last.rect[0] + last.rect[2] + 2.0, last.rect[1] + 2.0)
    assert hit_test(lay, *beside) is None
    assert contains(lay, *beside)  # eaten: no voxel is attached behind the rail
    assert contains(lay, 5.0, 5.0)  # the plate itself
    # The rest of the viewport stays the view's, so the navigation gizmo keeps its clicks.
    assert not contains(lay, WIDTH - 5.0, HEIGHT - 5.0)
    assert not contains(lay, RAIL_WIDTH + 8.0, HEIGHT - 24.0)
    assert lay.hit_rect[2] == lay.rect[2] == RAIL_WIDTH


def test_an_open_flyout_is_hit_before_the_strip():
    lay = layout(tool=ERASE, mode=BOX)
    flyout = flyout_widgets(lay, "TOOL_ERASE")
    assert [w.id for w in flyout] == list(TOOL_FLYOUT_IDS["TOOL_ERASE"])
    for widget in flyout:
        px, py = centre(widget)
        assert hit_test(lay, px, py, open_id="TOOL_ERASE").id == widget.id
        assert hit_test(lay, px, py) is None  # closed: the pixel is the viewport's
        assert contains(lay, px, py, open_id="TOOL_ERASE")


def test_a_flyout_opens_to_the_right_of_its_button():
    lay = layout(tool=ERASE, mode=CENTER)
    button = lay.by_id("TOOL_ERASE")
    flyout = flyout_widgets(lay, "TOOL_ERASE")
    assert flyout
    for widget in flyout:
        assert widget.rect[0] >= button.rect[0] + button.rect[2]
    assert flyout[1].rect[0] > flyout[0].rect[0]
    assert flyout_widgets(lay, None) == ()
    assert flyout_widgets(lay, "MIRROR_X") == ()  # an item opens nothing


# --- primitives ---------------------------------------------------------------


def test_attach_ends_in_the_primitives_item_and_the_others_do_not():
    lay = layout(tool=ATTACH, mode=VOXEL)
    row = flyout_widgets(lay, "TOOL_ATTACH")
    assert [w.id for w in row] == list(TOOL_FLYOUT_IDS["TOOL_ATTACH"]) + [PRIMITIVES_ID]
    for tool_id in ("TOOL_ERASE", "TOOL_PAINT", "TOOL_SELECT"):
        assert PRIMITIVES_ID not in [w.id for w in flyout_widgets(lay, tool_id)]


def test_the_primitives_item_has_its_own_icon_and_lights_with_a_primitive():
    item = {w.id: w for w in flyout_widgets(layout(tool=ATTACH, mode=BOX, solid=CONE), "TOOL_ATTACH")}
    assert item[PRIMITIVES_ID].glyph == "PRIMITIVES" and item[PRIMITIVES_ID].active
    assert not item["TOOL_ATTACH_BOX"].active  # Box mode with a cone is the cone's
    plain = {w.id: w for w in flyout_widgets(layout(tool=ATTACH, mode=BOX), "TOOL_ATTACH")}
    assert plain["TOOL_ATTACH_BOX"].active and not plain[PRIMITIVES_ID].active
    voxel = {w.id: w for w in flyout_widgets(layout(tool=ATTACH, mode=VOXEL, solid=CONE), "TOOL_ATTACH")}
    assert not voxel[PRIMITIVES_ID].active


def test_the_open_primitives_show_every_solid_in_a_grid_under_the_item():
    lay = layout(tool=ATTACH, mode=BOX, solid=SPHERE)
    widgets = flyout_widgets(lay, PRIMITIVES_ID)
    row = flyout_widgets(lay, "TOOL_ATTACH")
    assert widgets[: len(row)] == row
    grid = widgets[len(row) :]
    assert [w.id for w in grid] == list(PRIMITIVE_IDS)
    assert [SOLID_OF_ID[w.id] for w in grid] == list(SOLIDS)
    assert [w.glyph for w in grid] == list(SOLIDS)
    assert [w.active for w in grid] == [s == SPHERE for s in SOLIDS]
    names = {ident: name for ident, name, _d in SOLID_ITEMS}
    assert all(w.tooltip.startswith(names[SOLID_OF_ID[w.id]]) for w in grid)
    item = row[-1]
    assert all(w.rect[1] + w.rect[3] <= item.rect[1] for w in grid)  # all under the item
    assert grid[0].rect[0] == item.rect[0]  # left edges line up
    cols = sorted({w.rect[0] for w in grid})
    assert len(cols) == min(PRIMITIVE_COLUMNS, len(grid))
    rects = [w.rect for w in grid]
    for i, (x0, y0, w0, h0) in enumerate(rects):
        for x1, y1, w1, h1 in rects[i + 1 :]:
            assert x0 + w0 <= x1 or x1 + w1 <= x0 or y0 + h0 <= y1 or y1 + h1 <= y0


def test_the_open_grid_is_hit_and_its_way_in_counts_as_inside():
    lay = layout(tool=ATTACH, mode=BOX)
    widgets = flyout_widgets(lay, PRIMITIVES_ID)
    grid = [w for w in widgets if w.id in SOLID_OF_ID]
    item = [w for w in widgets if w.id == PRIMITIVES_ID][0]
    for widget in grid:
        assert hit_test(lay, *centre(widget), open_id=PRIMITIVES_ID).id == widget.id
        assert hit_test(lay, *centre(widget), open_id="TOOL_ATTACH") is None  # closed grid
    gap = (item.rect[0] + item.rect[2] / 2.0, (item.rect[1] + grid[0].rect[1] + grid[0].rect[3]) / 2.0)
    assert contains(lay, *gap, open_id=PRIMITIVES_ID)
    # Under the rest of the row, left of the grid, the viewport keeps its clicks.
    first = widgets[0]
    below_first = (first.rect[0] + first.rect[2] / 2.0, grid[-1].rect[1] + 2.0)
    assert not contains(lay, *below_first, open_id=PRIMITIVES_ID)
    assert primitive_bridge(lay, "TOOL_ATTACH") is None


@pytest.mark.parametrize("opener", ["MODEL_MIRROR_HALF", "MODEL_SCALE"])
def test_a_model_tool_with_choices_opens_them_under_its_item(opener):
    from rok.core.chrome import COMMANDS, MODEL_CHOICES, is_flyout_item

    lay = layout(tool=ATTACH, mode=VOXEL)
    widgets = flyout_widgets(lay, opener)
    row = flyout_widgets(lay, MODEL_ID)
    assert widgets[: len(row)] == row  # the Model row stays open above the grid
    choices = MODEL_CHOICES[opener]
    grid = [w for w in widgets if w.id in choices]
    item = next(w for w in row if w.id == opener)
    assert [w.id for w in grid] == list(choices)
    assert all(is_flyout_item(w.id) and w.label and split_tooltip(w.tooltip)[1] for w in grid)
    assert max(w.rect[1] for w in grid) < item.rect[1]  # under the item
    for widget in grid:
        assert hit_test(lay, *centre(widget), open_id=opener).id == widget.id
        assert hit_test(lay, *centre(widget), open_id=MODEL_ID) is None  # closed grid
    gap = (item.rect[0] + item.rect[2] / 2.0, (item.rect[1] + grid[0].rect[1] + grid[0].rect[3]) / 2.0)
    assert contains(lay, *gap, open_id=opener)
    # Cmd+K offers every choice by name, with the operator's arguments.
    op = next(iter(choices.values())).op
    assert {c.props for c in COMMANDS if c.id == op} == {c.props for c in choices.values()}
    tool = split_tooltip(item.tooltip)[0]
    assert all(c.name.startswith(tool + ": ") for c in COMMANDS if c.id == op)


def test_mirror_half_puts_a_column_per_axis_and_plus_onto_minus_on_top():
    from rok.core.chrome import MIRROR_HALF_ID, MODEL_CHOICES

    lay = layout(tool=ATTACH, mode=VOXEL)
    choices = MODEL_CHOICES[MIRROR_HALF_ID]
    grid = [w for w in flyout_widgets(lay, MIRROR_HALF_ID) if w.id in choices]
    props = {w.id: dict(choices[w.id].props) for w in grid}
    assert {(p["axis"], p["from_positive"]) for p in props.values()} == {(a, s) for a in "XYZ" for s in (True, False)}
    top = [w for w in grid if props[w.id]["from_positive"]]
    assert min(w.rect[1] for w in top) > max(w.rect[1] for w in grid if w not in top)


def test_the_gap_between_a_button_and_its_open_flyout_counts_as_inside():
    """The mouse crosses the rail's hairline on its way to the row; that must
    not close the row, and a press there must not reach the voxels."""
    lay = layout(tool=ATTACH, mode=VOXEL)
    row = flyout_widgets(lay, "TOOL_ATTACH")
    button = lay.by_id("TOOL_ATTACH").rect
    gap_x = (lay.rect[2] + row[0].rect[0]) / 2.0  # between the hairline and the first item
    gap_y = button[1] + button[3] / 2.0
    assert not in_rect(lay.hit_rect, gap_x, gap_y)
    assert contains(lay, gap_x, gap_y, "TOOL_ATTACH")
    assert not contains(lay, gap_x, gap_y, None)  # closed: the viewport's
    assert hit_test(lay, gap_x, gap_y, "TOOL_ATTACH") is None  # inside, on no item
    bridge = flyout_bridge(lay, "TOOL_ATTACH")
    assert bridge is not None and bridge[0] <= button[0] and bridge[0] + bridge[2] >= row[-1].rect[0] + row[-1].rect[2]
    assert flyout_bridge(lay, None) is None and flyout_bridge(lay, "MIRROR_X") is None


def test_split_tooltip_gives_the_readout_a_name_and_a_line():
    assert split_tooltip("Fill All — every empty cell") == ("Fill All", "every empty cell")
    assert split_tooltip("Plain") == ("Plain", "")
    row = flyout_widgets(layout(tool=ATTACH), "TOOL_ATTACH")
    assert all(split_tooltip(w.tooltip)[0] and split_tooltip(w.tooltip)[1] for w in row)


def test_every_tool_button_opens_its_modes_to_its_right():
    """Attach, Erase and Paint offer the face modes, Attach Clone too, Erase and Paint the Marquee; Select its click mode,
    Face, the two screen modes, Box and the Connected modes. A click on one
    sets the tool and the mode."""
    from rok.core.edit import SELECT_ONLY_MODES

    face_modes = [m for m in MODES if m not in (RECT, LASSO)]
    for widget_id, tool in TOOL_IDS.items():
        lay = layout(tool=tool, mode=LINE)
        row = [w for w in flyout_widgets(lay, widget_id) if w.id != PRIMITIVES_ID]
        modes = [TOOL_MODE_OF_ID[w.id][1] for w in row]
        if tool == SELECT:
            assert modes == [VOXEL, FACE, RECT, LASSO, BOX, *SELECT_ONLY_MODES]
        elif tool == ERASE:  # the Marquee too
            assert modes == [*face_modes, RECT]
        elif tool == PAINT:  # the Marquee and the Gradient
            assert modes == [*face_modes, RECT, "GRADIENT"]
        else:  # Attach: the Clone stamp after Pattern
            assert modes == [*face_modes, "CLONE"]
        assert all(TOOL_MODE_OF_ID[w.id][0] == tool for w in row)
        assert [w.glyph for w in row] == [RING_GLYPHS[m] for m in modes]
        assert all(MODE_NAMES[m] in w.tooltip for w, m in zip(row, modes))
        button = lay.by_id(widget_id).rect
        assert all(w.rect[0] >= lay.rect[2] for w in row)  # past the rail
        assert all(abs((w.rect[1] + w.rect[3] / 2) - (button[1] + button[3] / 2)) <= 1 for w in row)
        # The current mode is lit only on the current tool's row.
        lit = [w.active for w in row]
        assert lit == [m == LINE for m in modes] if tool != SELECT else lit == [False] * len(modes)
        other = flyout_widgets(layout(tool=ATTACH if tool != ATTACH else ERASE, mode=LINE), widget_id)
        assert not any(w.active for w in other)
        # A press on an item hits it while the row is open, not otherwise.
        item = row[-1]
        assert hit_test(lay, *centre(item), widget_id) is not None
        assert hit_test(lay, *centre(item), None) is None
    assert set(TOOL_FLYOUT_IDS) == set(TOOL_IDS)


# --- focus and tooltips -------------------------------------------------------


def test_the_focus_walks_the_rail_and_wraps():
    lay = layout(tool=SELECT, mode=VOXEL)
    ids = focus_ids(lay)
    assert ids == tuple(w.id for w in lay.widgets)
    assert focus_step(lay, None, 1) == ids[0]
    assert focus_step(lay, None, -1) == ids[-1]
    assert focus_step(lay, ids[0], 1) == ids[1]
    assert focus_step(lay, ids[-1], 1) == ids[0]
    assert focus_step(lay, ids[0], -1) == ids[-1]
    # A widget that went away (the size chip) starts the walk again.
    gone = layout(tool=ATTACH, mode=VOXEL)
    assert focus_step(gone, SIZE_ID, 1) == focus_ids(gone)[0]


def test_a_tooltip_waits_for_the_hover_to_rest():
    assert TOOLTIP_DELAY_S == 0.2
    assert not tooltip_due(None, 100.0)
    assert not tooltip_due(100.0, 100.19)
    assert tooltip_due(100.0, 100.2)
    assert tooltip_due(100.0, 101.0)


def test_a_tooltip_stays_inside_the_region():
    lay = layout()
    first = lay.widgets[0]
    rect = tooltip_rect(first, lay, WIDTH)
    assert rect[0] >= first.rect[0] + first.rect[2]  # to the right of the widget
    assert rect[0] >= 0
    assert rect[0] + rect[2] <= WIDTH
    narrow = tooltip_rect(first, lay, 40)
    assert narrow[0] == 0  # pushed back inside a region narrower than the tooltip


def test_every_widget_has_a_tooltip():
    lay = layout(tool=SELECT, mode=VOXEL)
    assert all(widget.tooltip for widget in lay.widgets)
    attach = lay.by_id("TOOL_ATTACH").tooltip
    assert attach.startswith("Attach")
    assert TOOL_KEYS == {
        "TOOL_ATTACH": "T",
        "TOOL_ERASE": "R",
        "TOOL_PAINT": "G",
        "TOOL_SELECT": "Z",
    }
    assert TOOL_KEYS["TOOL_ATTACH"] in attach
    assert "Add blocks" in attach
    assert TOOL_KEYS["TOOL_ERASE"] in lay.by_id("TOOL_ERASE").tooltip
    assert TOOL_KEYS["TOOL_PAINT"] in lay.by_id("TOOL_PAINT").tooltip
    assert TOOL_KEYS["TOOL_SELECT"] in lay.by_id("TOOL_SELECT").tooltip
    assert "Brush size" in lay.by_id(SIZE_ID).tooltip
    assert "Whole model" in lay.by_id(MODEL_ID).tooltip


# --- glyphs -------------------------------------------------------------------


@pytest.mark.parametrize("name", GLYPH_NAMES)
def test_a_glyph_is_line_pairs_inside_its_rectangle(name):
    rect = (100, 200, 20, 20)
    lines = glyph_lines(name, rect, inset=2.0)
    assert lines.dtype == np.float32
    assert lines.ndim == 2 and lines.shape[1] == 2
    assert lines.shape[0] >= 2 and lines.shape[0] % 2 == 0  # LINES pairs
    x, y, w, h = rect
    assert lines[:, 0].min() >= x and lines[:, 0].max() <= x + w
    assert lines[:, 1].min() >= y and lines[:, 1].max() <= y + h
    # Pixel-snapped: a one-pixel line is sharp on a half pixel.
    assert np.allclose(lines % 1.0, 0.5)


def test_an_unknown_or_tiny_glyph_draws_nothing():
    assert glyph_lines("NO_SUCH_GLYPH", (0, 0, 20, 20)).shape == (0, 2)
    assert glyph_lines("BOX", (0, 0, 4, 4), inset=3.0).shape == (0, 2)


def test_a_glyph_grows_with_its_rectangle():
    small = glyph_lines("MARQUEE", (0, 0, 16, 16))
    big = glyph_lines("MARQUEE", (0, 0, 32, 32))
    assert big[:, 0].max() - big[:, 0].min() > small[:, 0].max() - small[:, 0].min()


def test_rectangle_geometry():
    lines = rect_lines((10, 20, 30, 40))
    assert lines.shape == (8, 2)  # four edges as pairs
    assert np.allclose(lines % 1.0, 0.5)
    tris = rect_tris((10, 20, 30, 40))
    assert tris.shape == (6, 2)
    assert tris[:, 0].min() == 10 and tris[:, 0].max() == 40
    assert tris[:, 1].min() == 20 and tris[:, 1].max() == 60
    edge = edge_line((0, 100, 50, 24))
    assert np.allclose(edge, [[0.0, 100.5], [50.0, 100.5]])
    right = right_edge_line((0, 100, 50, 24))
    assert np.allclose(right, [[49.5, 100.0], [49.5, 124.0]])


# --- command palette ----------------------------------------------------------


def test_an_empty_query_lists_every_command():
    assert command_match("") == COMMANDS
    assert command_match("   ") == COMMANDS
    ids = {c.id for c in COMMANDS}
    assert {"grow", "fill_all", "export_mesh", "view_go", "resize"} <= ids


def test_command_search_is_fuzzy_and_ranks_a_prefix_first():
    hits = command_match("grow")
    assert hits[0].id == "grow"
    mesh = command_match("mesh")
    assert mesh and mesh[0].id == "export_mesh"
    fill = command_match("fll")
    assert fill and fill[0].id == "fill_all"
    assert command_match("no-such-command-xyz") == ()


def test_bridge_lines_is_a_command_palette_entry():
    hits = command_match("bridge")
    assert hits and hits[0].name == "Bridge" and hits[0].id == "bridge"
    assert hits[0].props == ()
    lines = [c for c in COMMANDS if c.name == "Bridge (lines)"]
    assert len(lines) == 1
    assert lines[0].id == "bridge"
    assert dict(lines[0].props) == {"kind": "LINES"}
    assert command_match("lines")[0].name == "Bridge (lines)"


def test_the_command_panel_sits_inside_the_region_and_hit_tests_rows():
    hits = command_match("view")
    lay = command_layout(1.0, WIDTH, HEIGHT, "view", hits, 0)
    x, y, w, h = lay.rect
    assert w <= WIDTH and h <= HEIGHT
    assert 0 <= x and x + w <= WIDTH
    assert 0 <= y and y + h <= HEIGHT
    assert lay.rows
    rx, ry, rw, rh = lay.rows[0]
    assert command_hit(lay, rx + rw / 2.0, ry + rh / 2.0) == 0
    assert command_hit(lay, 0.0, 0.0) is None
    all_hits = command_match("")
    full = command_layout(1.0, WIDTH, HEIGHT, "", all_hits, 99)
    assert len(full.rows) == len(full.hits)
    assert full.index == len(full.hits) - 1



# --- visible only -------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_visible_chip_shows_with_select_in_the_modes_that_take_through_the_model(mode):
    chip = layout(tool=SELECT, mode=mode).by_id(VISIBLE_ID)
    assert (chip is not None) == (mode in VISIBLE_MODES)
    for tool in (ATTACH, ERASE, PAINT):
        assert layout(tool=tool, mode=mode).by_id(VISIBLE_ID) is None


def test_the_visible_chip_lights_when_on_and_has_a_header():
    off = layout(tool=SELECT, mode=RECT)
    on = layout(tool=SELECT, mode=RECT, select_visible=True)
    assert not off.by_id(VISIBLE_ID).active and on.by_id(VISIBLE_ID).active
    assert off.by_id(VISIBLE_ID).glyph == "SEEN" and "Visible only" in off.by_id(VISIBLE_ID).tooltip
    chip = off.by_id(VISIBLE_ID)
    header = dict(off.headers)[VISIBLE_HEADER]
    assert header[1] >= chip.rect[1] + chip.rect[3]  # the header sits over the chip
    assert off.by_id(VISIBLE_ID).rect[1] < off.by_id(LAYERS_ID).rect[1]


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_rail_with_the_visible_chip_still_fits(scale):
    lay = layout(scale, tool=SELECT, mode=LASSO)
    for widget in lay.widgets:
        x, y, w, h = widget.rect
        assert 0 <= y and y + h <= lay.rect[3], widget.id
    rects = [w.rect for w in lay.widgets]
    for i, (x0, y0, w0, h0) in enumerate(rects):
        for x1, y1, w1, h1 in rects[i + 1 :]:
            assert x0 + w0 <= x1 or x1 + w1 <= x0 or y0 + h0 <= y1 or y1 + h1 <= y0



def test_the_layers_button_sits_under_the_model_button_with_its_header():
    lay = layout(tool=ATTACH, mode=VOXEL)
    button = lay.by_id(LAYERS_ID)
    assert button is not None and button.glyph == "LAYERS" and "Layers" in button.tooltip
    assert button.rect[1] < lay.by_id(MODEL_ID).rect[1]  # below Model
    assert button.rect[1] > lay.by_id(SIZE_ID).rect[1]  # above the chips at the bottom
    scene = layout(level=LEVEL_SCENE)
    assert scene.by_id(LAYERS_ID).rect[1] < scene.by_id(NEW_ID).rect[1]  # the scene's last group


# --- the two levels -------------------------------------------------------------


def test_the_level_switch_tops_the_rail_with_the_current_level_lit():
    inside = layout(model_name="Hull", layers=LAYER_ROWS)
    scene_seg, model_seg = inside.by_id(LEVEL_SCENE_ID), inside.by_id(LEVEL_MODEL_ID)
    assert scene_seg.kind == model_seg.kind == SEGMENT_KIND
    assert model_seg.active and not scene_seg.active
    assert scene_seg.glyph and model_seg.glyph and scene_seg.glyph != model_seg.glyph  # symbols, no words
    assert not scene_seg.label and not model_seg.label and not scene_seg.note
    assert "Hull" in model_seg.tooltip
    others = [w for w in inside.widgets if w.id not in LEVEL_IDS]
    assert model_seg.rect[1] > max(w.rect[1] + w.rect[3] for w in others)  # on top of everything
    # Stacked: the model above the scene (the chosen order), one column, no overlap.
    assert scene_seg.rect[0] == model_seg.rect[0] and model_seg.rect[1] >= scene_seg.rect[1] + scene_seg.rect[3]
    assert scene_seg.rect[0] >= 0 and scene_seg.rect[0] + scene_seg.rect[2] <= inside.rect[2]
    assert "(Esc)" in scene_seg.tooltip and "(Enter)" in model_seg.tooltip
    scene = layout(level=LEVEL_SCENE, back_name="Recon", layers=LAYER_ROWS)
    assert scene.by_id(LEVEL_SCENE_ID).active and not scene.by_id(LEVEL_MODEL_ID).active
    assert "Recon" in scene.by_id(LEVEL_MODEL_ID).tooltip  # where a click goes back to
    assert scene.by_id(LEVEL_SCENE_ID).rect == scene_seg.rect  # the switch keeps its place


def test_the_scene_tools_hold_drop_fit_flip_the_group_tools_and_delete():
    from rok.core.chrome import (SCENE_ALIGN_ID, SCENE_ARRAY_ID, SCENE_CARVE_ID, SCENE_DROP_ID, SCENE_DROP_ONTO_ID,
                                  SCENE_PICK_LINKED_ID)

    lay = layout(level=LEVEL_SCENE)
    button = lay.by_id(SCENE_TOOLS_ID)
    assert button is not None and button.tooltip.startswith("Tools")
    row = flyout_widgets(lay, SCENE_TOOLS_ID)
    assert [w.id for w in row] == list(SCENE_TOOL_IDS) == [
        SCENE_DROP_ID, SCENE_DROP_ONTO_ID, SCENE_FIT_ID, SCENE_FLIP_ID, SCENE_PICK_LINKED_ID, SCENE_ALIGN_ID,
        SCENE_ARRAY_ID, SCENE_CARVE_ID, SCENE_DELETE_ID]
    assert all(w.glyph and w.tooltip and is_flyout_item(w.id) for w in row)
    assert all(w.rect[0] >= lay.rect[2] for w in row)
    assert layout().by_id(SCENE_TOOLS_ID) is None  # inside a model the Model button holds the tools


def test_flip_in_the_scene_tools_opens_x_y_and_z_under_it():
    from rok.core.chrome import CHOICE_OF_ID

    lay = layout(level=LEVEL_SCENE)
    items = flyout_widgets(lay, SCENE_FLIP_ID)
    row = [w for w in items if w.id in SCENE_TOOL_IDS]
    grid = [w for w in items if w.id in FLIP_IDS]
    assert [w.id for w in row] == list(SCENE_TOOL_IDS)  # the Tools row stays open
    assert [w.label for w in grid] == ["X", "Y", "Z"]
    flip = next(w for w in row if w.id == SCENE_FLIP_ID)
    assert all(w.rect[1] < flip.rect[1] for w in grid)  # under its item
    assert [CHOICE_OF_ID[w.id].props for w in grid] == [(("axis", 0),), (("axis", 1),), (("axis", 2),)]
    assert hit_test(lay, *centre(grid[1]), SCENE_FLIP_ID).id == FLIP_IDS[1]
    assert contains(lay, *centre(grid[2]), SCENE_FLIP_ID)
    assert any(c.name == "Flip Models: Y" for c in COMMANDS)


def test_the_scene_rail_holds_pick_new_and_layers_and_no_voxel_tools():
    scene = layout(level=LEVEL_SCENE, tool=SELECT, mode=VOXEL)
    ids = {w.id for w in scene.widgets}
    assert set(LEVEL_IDS) | {
        SCENE_PICK_ID, SCENE_CUT_ID, NEW_ID, SCENE_TOOLS_ID, LAYERS_ID,
    } == ids
    assert scene.by_id(SCENE_PICK_ID).active and not scene.by_id(SCENE_CUT_ID).active
    cutting = layout(level=LEVEL_SCENE, scene_tool="CUT")
    assert cutting.by_id(SCENE_CUT_ID).active and not cutting.by_id(SCENE_PICK_ID).active
    assert cutting.by_id(SCENE_CUT_ID).rect[1] < cutting.by_id(SCENE_PICK_ID).rect[1]  # Cut under Pick
    assert [t for t, _r in scene.headers] == [
        NEW_HEADER, TOOLS_HEADER, LAYERS_HEADER,
    ]
    inside = {w.id for w in layout(tool=SELECT, mode=VOXEL).widgets}
    assert set(TOOL_IDS) | {MODEL_ID} <= inside and not ({SCENE_PICK_ID, NEW_ID} & inside)


def test_the_new_flyout_offers_an_empty_model_one_from_a_file_terrain_and_every_shape():
    from rok.core.chrome import NEW_SOLID_OF_ID
    from rok.core.shape import SOLIDS

    lay = layout(level=LEVEL_SCENE)
    items = flyout_widgets(lay, NEW_ID)
    row = [w for w in items if w.id not in NEW_SOLID_OF_ID]
    grid = [w for w in items if w.id in NEW_SOLID_OF_ID]
    assert [w.id for w in row] == list(NEW_FLYOUT_IDS)
    assert [NEW_SOLID_OF_ID[w.id] for w in grid] == list(SOLIDS)
    assert all(w.glyph and split_tooltip(w.tooltip)[1] for w in items)
    assert not any(w.active for w in grid)  # nothing is "chosen" in New's grid
    assert max(w.rect[1] for w in grid) < min(w.rect[1] for w in row)  # the grid hangs under the row
    for w in (row[-1], grid[-1]):
        assert hit_test(lay, *centre(w), NEW_ID).id == w.id


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_scene_rail_fits_its_height_and_nothing_overlaps(scale):
    st = ChromeState(level=LEVEL_SCENE)
    lay = layout(scale, level=LEVEL_SCENE)
    lowest = min(w.rect[1] for w in lay.widgets)
    used = lay.rect[3] - lowest + round(RAIL_INSET * scale)
    assert abs(used - rail_height(st) * scale) <= 3  # rail_height counts every widget
    assert rail_height(st) < rail_height(ChromeState(tool=SELECT, mode=VOXEL))
    rects = [w.rect for w in lay.widgets]
    for i, (x0, y0, w0, h0) in enumerate(rects):
        for x1, y1, w1, h1 in rects[i + 1 :]:
            assert x0 + w0 <= x1 or x1 + w1 <= x0 or y0 + h0 <= y1 or y1 + h1 <= y0


# --- the Layers flyout ----------------------------------------------------------

from rok.core.chrome import (  # noqa: E402
    LAYER_EYE,
    LAYER_MODEL,
    LAYER_MOVE,
    LAYER_ROW,
    LIST_KIND,
    is_flyout_item,
    layer_display_name,
    layer_part,
)

LAYER_ROWS = (
    (0, "0", False, ((11, "Hull"), (12, "Tree"))),
    (1, "Guns", True, ((13, "Recon"),)),
    (2, "2", False, ()),
)


def layered(**kwargs):
    base = dict(tool=ATTACH, mode=VOXEL, layers=LAYER_ROWS, model_uid=11, model_name="Hull")
    base.update(kwargs)
    return layout(**base)


def test_the_list_is_each_layer_then_its_models():
    widgets = flyout_widgets(layered(), LAYERS_ID)
    expected = []
    for layer_id, _n, _h, models in LAYER_ROWS:
        expected += [f"{LAYER_EYE}{layer_id}", f"{LAYER_ROW}{layer_id}", f"{LAYER_MOVE}{layer_id}"]
        expected += [f"{LAYER_MODEL}{uid}" for uid, _m in models]
    assert [w.id for w in widgets] == expected
    assert all(w.kind == LIST_KIND for w in widgets)


def test_the_edited_model_is_the_one_marked_row():
    rows = {w.id: w for w in flyout_widgets(layered(), LAYERS_ID)}
    assert rows[f"{LAYER_MODEL}11"].active
    assert not rows[f"{LAYER_MODEL}12"].active and not rows[f"{LAYER_MODEL}13"].active
    assert not any(w.active for w in flyout_widgets(layered(), LAYERS_ID) if w.id.startswith(LAYER_ROW))
    assert "Click: edit this model" in rows[f"{LAYER_MODEL}12"].tooltip


def test_a_layer_row_is_a_group_named_and_dimmed_when_hidden_or_empty():
    rows = {w.id: w for w in flyout_widgets(layered(), LAYERS_ID)}
    assert rows[f"{LAYER_ROW}0"].label == "Layer 0" and rows[f"{LAYER_ROW}0"].glyph == "GROUP"
    assert rows[f"{LAYER_ROW}1"].label == "Guns (hidden)" and rows[f"{LAYER_ROW}1"].dim
    assert rows[f"{LAYER_ROW}2"].dim and rows[f"{LAYER_ROW}2"].glyph == ""  # empty
    assert rows[f"{LAYER_MODEL}13"].dim  # a model in a hidden group
    assert not rows[f"{LAYER_EYE}1"].active and "Show" in rows[f"{LAYER_EYE}1"].tooltip


def test_the_arrow_moves_the_edited_model_but_not_into_its_own_layer():
    rows = {w.id: w for w in flyout_widgets(layered(), LAYERS_ID)}
    assert not rows[f"{LAYER_MOVE}0"].enabled  # Hull is already in Layer 0
    assert rows[f"{LAYER_MOVE}2"].enabled and "Move Hull here" in rows[f"{LAYER_MOVE}2"].tooltip
    none = {w.id: w for w in flyout_widgets(layered(model_uid=-1, model_name=""), LAYERS_ID)}
    assert not none[f"{LAYER_MOVE}2"].enabled


def test_model_rows_sit_under_their_layer_and_the_list_stays_inside_the_view():
    lay = layered()
    widgets = flyout_widgets(lay, LAYERS_ID)
    rows = [w for w in widgets if w.id.startswith((LAYER_ROW, LAYER_MODEL))]
    ys = [w.rect[1] for w in rows]
    assert ys == sorted(ys, reverse=True) and len(set(ys)) == len(ys)  # one row each, top down
    assert all(w.rect[0] >= lay.rect[2] and w.rect[1] >= 0 for w in widgets)
    for w in widgets:
        if w.enabled:
            assert hit_test(lay, *centre(w), LAYERS_ID).id == w.id


def test_a_long_list_moves_up_to_stay_inside_the_view():
    many = tuple((i, str(i), False, ((100 + i, f"M{i}"),)) for i in range(8))
    lay = layered(layers=many, model_uid=100)
    widgets = flyout_widgets(lay, LAYERS_ID)
    assert min(w.rect[1] for w in widgets) >= 0
    assert max(w.rect[1] + w.rect[3] for w in widgets) <= lay.rect[3]


def test_no_layers_no_list_and_the_ids_are_flyout_items():
    assert flyout_widgets(layered(layers=()), LAYERS_ID) == ()
    assert layer_part(f"{LAYER_MOVE}12") == (LAYER_MOVE, 12)
    assert layer_part(f"{LAYER_MODEL}7") == (LAYER_MODEL, 7)
    assert layer_part("LAYER_ROW_x") is None and layer_part(None) is None
    assert is_flyout_item(f"{LAYER_EYE}3") and not is_flyout_item("MIRROR_X")
    assert layer_display_name(4, "4") == "Layer 4" and layer_display_name(4, "Guns") == "Guns"


def test_the_rail_colour_slides_to_the_level_and_stays():
    from rok.core.chrome import LEVEL_FADE_S, LEVEL_MODEL, level_mix

    assert level_mix(LEVEL_SCENE, 10.0, 10.0) == 0.0
    assert 0.0 < level_mix(LEVEL_SCENE, 10.0, 10.0 + LEVEL_FADE_S / 2) < 1.0
    assert level_mix(LEVEL_SCENE, 10.0, 10.0 + LEVEL_FADE_S * 3) == 1.0
    assert level_mix(LEVEL_MODEL, 10.0, 10.0 + LEVEL_FADE_S * 3) == 0.0
    assert level_mix(LEVEL_MODEL, 0.0, 100.0) == 0.0  # a file opens with no fade


def test_a_double_click_is_two_clicks_on_one_model_soon_after():
    from rok.core.chrome import DOUBLE_CLICK_S, is_double_click

    assert is_double_click((7, 1.0), 7, 1.0 + DOUBLE_CLICK_S / 2)
    assert not is_double_click((7, 1.0), 8, 1.1)  # another model
    assert not is_double_click((7, 1.0), 7, 1.0 + DOUBLE_CLICK_S * 2)  # too late
    assert not is_double_click(None, 7, 1.0)


def test_a_linked_copy_has_a_chain_on_its_row_in_the_layers_list():
    from rok.core.chrome import LAYER_LINK, LAYER_MODEL, flyout_widgets, layer_part

    st = ChromeState(level=LEVEL_SCENE, layers=((0, "0", False, ((1, "Crate"), (2, "Crate 2"), (3, "Ship"))),),
                     linked=((1, ("Crate 2",)), (2, ("Crate",))), model_uid=1, model_name="Crate")
    lay = chrome_layout(1.0, st, 1600, 900)
    items = {w.id: w for w in flyout_widgets(lay, LAYERS_ID)}
    assert {f"{LAYER_LINK}1", f"{LAYER_LINK}2"} <= set(items) and f"{LAYER_LINK}3" not in items
    chain, row = items[f"{LAYER_LINK}2"], items[f"{LAYER_MODEL}2"]
    assert chain.rect[1] == row.rect[1] and chain.rect[0] >= row.rect[0] + row.rect[2]  # beside its row
    assert layer_part(chain.id) == (LAYER_LINK, 2)
    assert chain.tooltip == "Linked copy of Crate: an edit to one changes both. Click: make it unique"
    assert hit_test(lay, *centre(chain), LAYERS_ID).id == chain.id


def test_the_chain_tooltip_names_up_to_three_copies():
    from rok.core.chrome import link_tip, name_list

    assert name_list(["A", "B", "C"]) == "A, B and C"
    assert name_list(["A", "B", "C", "D", "E"]) == "A, B, C and 2 more"
    assert link_tip(("A", "B")) == "Linked copy of A and B: an edit to one changes all of them. Click: make it unique"


# --- Animate mode ----------------------------------------------------------------


def test_the_level_switch_has_animate_as_its_third_cell():
    for level in (LEVEL_MODEL, LEVEL_SCENE):
        lay = layout(level=level, layers=LAYER_ROWS)
        cells = [lay.by_id(ident) for ident in LEVEL_IDS]
        assert all(c is not None and c.kind == SEGMENT_KIND for c in cells)
        # One column, top to bottom in LEVEL_IDS order, no overlap.
        for upper, lower in zip(cells, cells[1:]):
            assert upper.rect[0] == lower.rect[0] and upper.rect[1] >= lower.rect[1] + lower.rect[3]
        assert not lay.by_id(LEVEL_ANIM_ID).active
    animating = layout(animating=True, layers=LAYER_ROWS)
    assert animating.by_id(LEVEL_ANIM_ID).active
    assert not animating.by_id(LEVEL_SCENE_ID).active and not animating.by_id(LEVEL_MODEL_ID).active
    assert animating.by_id(LEVEL_ANIM_ID).rect == layout().by_id(LEVEL_ANIM_ID).rect  # it keeps its place


def test_animate_mode_swaps_the_rail_to_the_animation_tools():
    lay = layout(animating=True, layers=LAYER_ROWS)
    assert lay.state.animating
    ids = {w.id for w in lay.widgets if w.kind != SEGMENT_KIND}
    # Motions, the parts, and (Phase 3) the three paint tools for a part's frames; no Select.
    assert ids == {MOTION_ID, LAYERS_ID, "TOOL_ATTACH", "TOOL_ERASE", "TOOL_PAINT"}
    assert "TOOL_SELECT" not in ids
    assert all(w.tooltip for w in lay.widgets)
    # The build rails have no Animate button: the switch is the way in.
    for level in (LEVEL_MODEL, LEVEL_SCENE):
        assert MOTION_ID not in {w.id for w in layout(level=level).widgets}


def test_rail_height_counts_the_three_cells_and_fits_every_rail():
    for st in (ChromeState(tool=SELECT, mode=VOXEL), ChromeState(level=LEVEL_SCENE), ChromeState(animating=True)):
        assert rail_height(st) * 2.0 <= HEIGHT
        lay = chrome_layout(1.0, st, WIDTH, int(rail_height(st)))
        assert lay.scale == 1.0  # fits exactly: not squeezed
        for widget in lay.widgets:
            assert widget.rect[1] >= RAIL_INSET - 1


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_motion_gallery_opens_beside_the_rail_with_a_card_per_preset(scale):
    from rok.core import anim
    from rok.core.anim import PART_PRESETS

    lay = layout(scale, animating=True)
    gallery = flyout_widgets(lay, MOTION_ID)
    cards = [w for w in gallery if w.kind == CARD_KIND]
    captions = [w for w in gallery if w.kind == CAPTION_KIND]
    assert [PRESET_OF_ID[w.id] for w in cards] == list(PART_PRESETS)  # colour motions come in Phase 3
    assert set(PRESET_OF_ID.values()) == set(anim.PRESETS)
    assert len(captions) == 1 and not any(c.enabled for c in captions)
    for w in gallery:
        assert w.rect[0] >= lay.rect[0] + lay.rect[2]  # right of the rail
        assert w.rect[1] >= 0 and w.rect[1] + w.rect[3] <= lay.rect[3]
        assert is_flyout_item(w.id)
    for card in cards:
        assert card.tooltip == preset_tip(PRESET_OF_ID[card.id])
        assert hit_test(lay, *centre(card), MOTION_ID).id == card.id
        assert contains(lay, *centre(card), MOTION_ID)
    for i, a in enumerate(cards):
        for b in cards[i + 1:]:
            ax, ay, aw, ah = a.rect
            bx, by, bw, bh = b.rect
            assert ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay


def test_the_gallery_stays_inside_a_short_view():
    lay = chrome_layout(1.0, ChromeState(animating=True), WIDTH, int(rail_height(ChromeState(animating=True))))
    for w in flyout_widgets(lay, MOTION_ID):
        assert w.rect[1] >= 0


def test_every_preset_has_an_id_and_a_plain_tip():
    from rok.core import anim

    assert PRESET_IDS == tuple(f"PRESET_{n}" for n in anim.PRESETS)
    for name in anim.PRESETS:
        tip = preset_tip(name)
        assert tip.startswith(name.title() + " — ") and len(tip) > len(name) + 10


def test_the_command_palette_enters_animate_and_runs_every_motion():
    from rok.core import anim

    names = {c.name for c in COMMANDS}
    assert "Animate" in names and "Key the Pose" in names and "Play or Stop" in names
    assert {f"Motion: {anim.PRESET_LABELS[n]}" for n in anim.PART_PRESETS} <= names
    assert command_match("animate")[0].id == "animate"


def test_every_rail_keeps_the_state_it_was_laid_out_for():
    """The scene and Animate rails once forgot the mirror: the layout now holds the
    whole state instead of a copy of some of it."""
    for kwargs in ({}, {"level": LEVEL_SCENE}, {"animating": True}):
        state = ChromeState(mirror=(True, False, True), brush_size=3, **kwargs)
        assert chrome_layout(1.0, state, WIDTH, 2000).state is state


@pytest.mark.parametrize("space, glyph, name", [("ADD", "CUBE_PLUS", "Add"), ("CARVE", "CUBE_MINUS", "Carve"),
                                                 ("HOLES", "HOLLOW", "Holes")])
def test_the_stamp_switch_shows_with_attach_in_pattern_mode(space, glyph, name):
    """A visible switch for the pattern's negative space."""
    lay = layout(tool=ATTACH, mode=PATTERN, pattern_space=space)
    chip = lay.by_id(STAMP_ID)
    assert chip is not None and chip.glyph == glyph and chip.active == (space != "ADD")
    assert name in [h for h, _r in lay.headers]
    assert layout(tool=ATTACH, mode=VOXEL).by_id(STAMP_ID) is None
    assert layout(tool=ERASE, mode=PATTERN).by_id(STAMP_ID) is None


def test_a_models_live_shaders_follow_its_row_with_an_eye():
    from rok.core.chrome import LIVE_EYE, LIVE_ROW

    live = ((11, 7, "Rust", True, ""), (11, 8, "Moss", False, ""))
    widgets = flyout_widgets(layered(live=live), LAYERS_ID)
    ids = [w.id for w in widgets]
    at = ids.index(f"{LAYER_MODEL}11")
    assert ids[at + 1:at + 5] == [f"{LIVE_EYE}7", f"{LIVE_ROW}7", f"{LIVE_EYE}8", f"{LIVE_ROW}8"]
    rows = {w.id: w for w in widgets}
    assert rows[f"{LIVE_EYE}7"].active and not rows[f"{LIVE_EYE}8"].active
    assert rows[f"{LIVE_ROW}8"].dim and rows[f"{LIVE_ROW}7"].label == "Rust"
    assert layer_part(f"{LIVE_ROW}8") == (LIVE_ROW, 8)


# --- the Animation menu ------------------------------------------------------------


def test_the_animation_button_is_off_until_it_is_turned_on():
    from rok.core.chrome import ANIM_MENU_ID

    for kwargs in ({}, {"level": "SCENE"}, {"animating": True}):
        assert layout(**kwargs).by_id(ANIM_MENU_ID) is None
        assert layout(anim_menu=True, **kwargs).by_id(ANIM_MENU_ID) is not None


def test_the_animation_flyout_lists_its_commands_and_greys_what_does_not_run_here():
    from rok.core.chrome import ANIM_MENU_FLYOUT_IDS, ANIM_MENU_ID, COMMAND_OF_ANIM_ID

    building = {w.id: w for w in flyout_widgets(layout(anim_menu=True), ANIM_MENU_ID)}
    assert tuple(building) == ANIM_MENU_FLYOUT_IDS
    assert building["ANIM_MENU_MAKE_PART"].enabled and not building["ANIM_MENU_FRAME_ADD"].enabled
    assert building["ANIM_MENU_MAKE_PART"].tooltip.startswith("Make Part — ")
    animating = {w.id: w for w in flyout_widgets(layout(anim_menu=True, animating=True), ANIM_MENU_ID)}
    assert not animating["ANIM_MENU_MAKE_PART"].enabled and animating["ANIM_MENU_FRAME_ADD"].enabled
    assert set(COMMAND_OF_ANIM_ID.values()) <= {c.id for c in COMMANDS}  # each is a real command


def test_the_rail_grows_by_the_animation_button_and_still_fits_its_plate():
    from rok.core.chrome import rail_height

    for kwargs in ({}, {"level": "SCENE"}, {"animating": True}):
        off, on = layout(**kwargs), layout(anim_menu=True, **kwargs)
        assert rail_height(on.state) > rail_height(off.state)
        assert min(w.rect[1] for w in on.widgets if w.kind != "SEGMENT") >= 0


def test_the_model_flyout_stays_above_the_readout():
    """The flyout block moves up rather than cover the tool readout at the lower left."""
    low = chrome_layout(1.0, ChromeState(readout_top=0), 1000, 400)
    high = chrome_layout(1.0, ChromeState(readout_top=200), 1000, 400)
    bottom = lambda layout: min(w.rect[1] for w in flyout_widgets(layout, MODEL_ID))
    assert bottom(low) < 200 <= bottom(high)
