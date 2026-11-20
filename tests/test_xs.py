"""xs shader files: the header, the search, the wrapped source."""

from pathlib import Path

import numpy as np
import pytest

from rok.core import xs

TORUS = """// xs_begin
// author : 'Rok test'
// arg : { var = 'm_radius' name = 'Radius' value = '3'   range = '1 128'    step = '1'    precision = '0' }
// arg : { name = 'Mode' var = 'm_mode' range = '0 2' value = '1' step = '1' precision = '0' }
// xs_end
float map(vec3 v) { return length(v - i_volume_size * 0.5) < m_radius ? i_color_index : 0.0; }
"""


def test_the_header_gives_the_arguments_in_order():
    args = xs.parse_args(TORUS)
    assert [a.name for a in args] == ["Radius", "Mode"]
    assert [a.var for a in args] == ["m_radius", "m_mode"]
    assert (args[0].value, args[0].low, args[0].high, args[0].step, args[0].precision) == (3, 1, 128, 1, 0)
    assert xs.parse_author(TORUS) == "Rok test"
    assert xs.parse_args("float map(vec3 v) { return 0.0; }") == ()


def test_at_most_sixteen_arguments():
    many = "".join(f"// arg : {{ var = 'a{k}' name = 'A{k}' value = '{k}' }}\n" for k in range(20))
    assert len(xs.parse_args(many)) == xs.MAX_ARGS


def test_the_source_names_the_args_and_keeps_the_file(tmp_path: Path):
    path = tmp_path / "ball.txt"
    path.write_text(TORUS)
    shader = xs.load(path)
    src = xs.fragment_source(shader)
    assert xs.arg_names(shader) == ("m_radius", "m_mode")  # uniforms of their own, not #defines
    assert "#define iArgs i_args" in src  # the older naming
    assert src.index("float voxel(") < src.index("float map(") < src.index("void main()")


def test_prepare_rewrites_what_metal_cannot_take():
    src = """
struct Cell { float d; vec2 uv; };
Cell hit(vec2 p) { return Cell(length(p), p); }
float map(vec3 v) {
  vec3 a = v; vec3 b = floor(v);
  float try = 1.0;
  if (a != b) return try;
  if (a == b) return 2.0;
  float k = 1.0; if (k == 1.0) return 3.0;
  return hit(v.xy).d;
}
"""
    out = xs.prepare(src)
    assert "Cell_make(length(p), p)" in out and "Cell Cell_make(float p_d, vec2 p_uv)" in out
    assert "any(notEqual(a, b))" in out and "all(equal(a, b))" in out
    assert "k == 1.0" in out  # scalars stay
    assert "xs_try" in out and " try " not in out


def test_find_shaders_takes_the_later_folder_on_a_name_clash(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "ball.txt").write_text(TORUS)
    (a / "other.txt").write_text(TORUS)
    (b / "ball.txt").write_text(TORUS)
    (b / "notes.txt").write_text("not a shader")
    found = xs.find_shaders([a, b, tmp_path / "missing"])
    assert [p.stem for p in found] == ["ball", "other"]
    assert found[0].parent == b


def test_labels_read_like_menu_entries():
    assert xs.label("city_blocks3D") == "City Blocks3D"
    assert xs.label("torus") == "Torus"


def test_a_shader_that_reads_color_sel_uses_the_colour_list(tmp_path: Path):
    path = tmp_path / "grad.txt"
    path.write_text("float map(vec3 v) { return color_sel(v.x); }")
    assert xs.uses_colour_selection(xs.load(path))
    path.write_text(TORUS)
    assert not xs.uses_colour_selection(xs.load(path))
    path.write_text("// color_sel( only in a comment\nfloat map(vec3 v) { return 1.0; }")
    assert not xs.uses_colour_selection(xs.load(path))


FIXTURES = Path(__file__).parent / "fixtures" / "shaders"


def test_an_old_shader_takes_its_arguments_from_the_usage_line():
    shader = xs.load(FIXTURES / "rok_old_usage.txt")
    assert [(a.name, a.value, a.precision) for a in shader.args] == [("Colour", 12, 0), ("Proba", 0.25, 2)]
    colour, proba = shader.args
    assert (colour.low, colour.high) == (0, 255)  # a whole number: a palette index
    assert (proba.low, proba.high) == (0, 1)  # a fraction
    assert all(a.var == "" for a in shader.args)  # read through i_args only


def test_decimal_counts_as_precision():
    args = xs.parse_args("// arg : { id = '1'  name = 'Radius'  value = '0.6'  range = '0 1'  step = '0.01'  decimal = '3' }")
    assert args[0].precision == 3


def test_globals_start_in_a_function_that_main_calls():
    shader = xs.load(FIXTURES / "rok_globals_loops.txt")
    out = xs.prepare(shader.source)
    assert "float colour;" in out and "int steps, unused;" in out
    init = out[out.index("void xs_init_globals()") :]
    assert "colour = m_colour;" in init and "steps = int(m_steps);" in init
    assert "const float" not in init
    kept = xs.prepare("const float PI = 3.14;\nfloat f(float x);\nfloat map(vec3 v) { return PI; }")
    assert "const float PI = 3.14;" in kept and "float f(float x);" in kept
    src = xs.fragment_source(shader)
    assert src.index("xs_init_globals();") < src.index("float idx = map(v);")


def test_every_loop_is_guarded():
    out = xs.prepare(xs.load(FIXTURES / "rok_globals_loops.txt").source)
    assert "for (int i = 0; (i < steps) && xs_tick(); i++)" in out
    assert "while ((k < steps) && xs_tick())" in out
    assert xs.prepare("float map(vec3 v) { for (;;) {} do { } while (true); return 0.0; }").count("xs_tick()") == 2
    assert "xs_steps <= u_loop_limit" in xs.fragment_source(xs.load(FIXTURES / "rok_ball.txt"))


def test_a_shader_cannot_reach_the_guard():
    out = xs.prepare("int xs_steps; bool xs_tick() { return true; }\nfloat map(vec3 v) { xs_steps = 0; return 1.0; }")
    assert "usr_xs_tick" in out and "usr_xs_steps = 0" in out
    assert "bool xs_tick" not in out


def test_a_huge_file_is_refused():
    with pytest.raises(xs.Unsupported, match="KB"):
        xs.prepare("float map(vec3 v) { return 0.0; }" + " " * xs.MAX_SOURCE)
    with pytest.raises(xs.Unsupported, match="#include"):
        xs.prepare('#include "other"\nfloat map(vec3 v) { return 0.0; }')


def test_brush_shaders_are_known_by_name_or_usage_line(tmp_path: Path):
    assert xs.is_brush(xs.load(FIXTURES / "rok_stripes.txt"))
    assert not xs.is_brush(xs.load(FIXTURES / "rok_ball.txt"))
    path = tmp_path / "brush_dots.txt"
    path.write_text("float map(vec3 v) { return 1.0; }")
    assert xs.is_brush(xs.load(path))


def test_the_atlas_tiles_the_slices_inside_the_texture_limit():
    assert xs.atlas_layout((256, 256, 256)) == (16, 16)  # 4096 x 4096, not 256 x 65536
    cols, rows = xs.atlas_layout((40, 30, 7))
    assert cols * rows >= 7 and cols <= 7
    rng = np.random.default_rng(3)
    vox = rng.integers(0, 256, (5, 4, 7), dtype=np.uint8)
    img = xs.atlas(vox, 3)
    assert img.shape == (3 * 4, 3 * 5)
    for x, y, z in ((0, 0, 0), (4, 3, 6), (2, 1, 4)):
        r, c = divmod(z, 3)
        assert img[r * 4 + y, c * 5 + x] == vox[x, y, z]


def test_reads_model_ignores_voxel_in_a_comment(tmp_path: Path):
    recolour = xs.load(FIXTURES / "rok_recolour.txt")
    path = tmp_path / "fill.txt"
    path.write_text("// voxel( only in a comment\nfloat map(vec3 v) { return i_color_index; }")
    assert xs.reads_model(recolour) and not xs.reads_model(xs.load(path))


def test_colour_arguments_are_known_by_name():
    target, new = xs.load(FIXTURES / "rok_recolour.txt").args
    assert xs.is_colour_arg(target) and not xs.is_colour_arg(new)
    named = [xs.ShaderArg(n, "", 1, 0, 255, 1, 0) for n in ("Object color", "Original", "Grout Color", "Seed")]
    assert [xs.is_colour_arg(a) for a in named] == [True, True, False, False]
    assert not xs.is_colour_arg(xs.ShaderArg("Target Colors", "", 0, -255, 255, 1, 0))  # a count, not an index


def test_a_one_value_vector_in_a_float_is_that_value():
    out = xs.prepare("float map(vec3 v) { float d = 1.0; float c = d * vec3(1.0 - step(0.5, v.x)); return c; }")
    assert "float c = d * float(1.0 - step(0.5, v.x));" in out
    kept = xs.prepare("float map(vec3 v) { vec3 k = vec3(1.0); float n = length(vec3(v.x, 0.0, 1.0)) + vec2(2.0).x; return n; }")
    assert "vec3 k = vec3(1.0);" in kept and "length(vec3(v.x, 0.0, 1.0))" in kept and "vec2(2.0).x" in kept


def test_model_reads_count_against_the_loop_budget():
    src = xs.fragment_source(xs.load(FIXTURES / "rok_ball.txt"))
    voxel = src[src.index("float voxel(") : src.index("float color_sel(")]
    assert "xs_steps++" in voxel
    assert src.index("int xs_steps;") < src.index("float voxel(")


def test_bands_grow_when_fast_and_shrink_when_slow():
    assert xs.next_band(4, xs.BAND_FAST / 2, 256) == 8
    assert xs.next_band(200, xs.BAND_FAST / 2, 256) == 256  # never past the slice
    assert xs.next_band(8, xs.BAND_SLOW * 2, 256) == 4
    assert xs.next_band(1, xs.BAND_SLOW * 10, 256) == 1  # one row at least
    assert xs.next_band(8, (xs.BAND_FAST + xs.BAND_SLOW) / 2, 256) == 8


# --- colours a shader asks for (rok_colours) -----------------------------------------

FLAME = ((0xFF, 0xD6, 0x3C), (0xFF, 0x8C, 0x1E))


def test_the_plume_fixture_asks_for_two_colours_after_its_marker():
    need = xs.load(FIXTURES / "rok_plume.txt").colours
    assert need == xs.ColourNeed(1, FLAME)


@pytest.mark.parametrize(
    "line",
    [
        "// rok_colours : { rgb = 'FFD63C nothex' }",
        "// rok_colours : { start = '3'  rgb = 'FFD63C' }",
        "// rok_colours : { start = '1' }",
        "float map(vec3 v) { return 0.0; } // no rok_colours line at all",
    ],
)
def test_a_colours_line_rok_cannot_read_counts_as_absent(line):
    assert xs.parse_colours(line) is None


def test_start_defaults_to_after_the_marker_and_hash_is_allowed():
    need = xs.parse_colours("// rok_colours : { rgb = '#FFD63C ff8c1e' }")
    assert need == xs.ColourNeed(1, FLAME)


@pytest.mark.parametrize(
    "line,author",
    [("// author : '@someone'", "@someone"), ("// author : @someone", "@someone"), ("// arg : { id = '0' }", "")],
)
def test_the_author_may_be_quoted_or_bare(line, author):
    assert xs.parse_author(f"// xs_begin\n{line}\n// xs_end\n") == author


def test_a_shader_that_cuts_the_selected_colours_acts_on_the_selection():
    cut = xs.load(FIXTURES / "rok_cut_selected.txt")
    paint = xs.load(FIXTURES / "rok_paint_selected.txt")
    assert xs.uses_colour_selection(cut) and xs.acts_on_selection(cut)
    assert xs.uses_colour_selection(paint) and not xs.acts_on_selection(paint)
    assert not xs.acts_on_selection(xs.load(FIXTURES / "rok_recolour.txt"))  # no selection at all


@pytest.mark.parametrize(
    "line",
    [
        "float map(vec3 v) { return voxel(v) == color_sel(0.0) ? 0.0 : voxel(v); }",
        "float map(vec3 v) { return color_sel(1.0) == voxel(v) ? 0.0 : voxel(v); }",
        "float map(vec3 v) { return isSelectedPal(v) ? 1.0 : 0.0; } float color_sel_n = color_sel(0.0);",
    ],
)
def test_every_way_of_testing_a_selected_colour_counts(tmp_path: Path, line):
    path = tmp_path / "s.txt"
    path.write_text(line)
    assert xs.acts_on_selection(xs.load(path))


def test_a_selected_colour_named_only_in_a_comment_does_not_count(tmp_path: Path):
    path = tmp_path / "s.txt"
    path.write_text("// voxel(v) == color_sel(0)\nfloat map(vec3 v) { return color_sel(0.0); }")
    assert not xs.acts_on_selection(xs.load(path))


def test_a_personal_shader_replaces_one_of_roks_of_the_same_name(tmp_path):
    from rok.core.xs import with_personal

    bundled = {"nature/rocks": tmp_path / "k1", "hard_surface/greebles": tmp_path / "k2"}
    personal = {"greebles": tmp_path / "p1", "mine": tmp_path / "p2", "nature/rocks": tmp_path / "p3"}
    out = with_personal(bundled, personal)
    assert out == {
        "nature/rocks": tmp_path / "p3",
        "hard_surface/greebles": tmp_path / "p1",
        "mine": tmp_path / "p2",
    }


def test_a_rok_about_line_describes_the_shader():
    from rok.core.xs import parse_about

    assert parse_about("// xs_end\n// rok_about : 'Rain streaks running down'\nfloat map") == "Rain streaks running down"
    assert parse_about("float map(vec3 v) { return 0.0; }") == ""


def test_a_colour_line_may_name_its_colours():
    need = xs.parse_colours("// rok_colours : { start = '0'  rgb = 'F4F7FB CBD6E2 7DBE4A'  names = 'Snow, Snow shade' }")
    assert need.names == ("Snow", "Snow shade") and need.name(2) == "Colour 3"
    mixed = need.with_rgb(xs.parse_hex_list("ff0000, -, 00ff00"))
    assert mixed.rgb == ((255, 0, 0), (203, 214, 226), (0, 255, 0)) and mixed.names == need.names
