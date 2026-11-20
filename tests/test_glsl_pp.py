"""Rok's shader preprocessor: macros expanded before the loop guard runs."""

from pathlib import Path

import pytest

from rok.core import glsl_pp, xs

SHADERS = Path(__file__).parent / "fixtures" / "shaders"


def _pp(text: str) -> str:
    return glsl_pp.preprocess(text)


def test_the_macro_fixture_expands_to_its_golden_text():
    source = xs._strip_comments((SHADERS / "rok_macros.txt").read_text())
    assert glsl_pp.preprocess(source) == (SHADERS / "rok_macros.expanded.glsl").read_text()


def test_the_expansion_has_no_directive_left():
    out = _pp(xs._strip_comments((SHADERS / "rok_macros.txt").read_text()))
    assert not any(line.lstrip().startswith("#") for line in out.splitlines())


def test_a_macro_is_not_expanded_inside_itself():
    assert _pp("#define X X + 1\nfloat a = X;").strip() == "float a = X + 1;"
    assert _pp("#define A B\n#define B A\nfloat a = A;").strip() == "float a = A;"


def test_a_function_macro_name_without_a_call_stays_a_name():
    assert _pp("#define F(x) (x)\nfloat F = 1.0; float g = F(2.0);").strip() == "float F = 1.0; float g = (2.0);"


def test_nested_ifs_skip_whole_blocks():
    text = "#if 0\n#if 1\nA\n#else\nB\n#endif\nC\n#else\nD\n#endif\n"
    assert _pp(text).split() == ["D"]
    assert _pp("#define N 3\n#if N > 2 && !defined(M) || N == 0\nyes\n#endif\n").split() == ["yes"]
    assert _pp("#if __VERSION__ >= 300\nnew\n#else\nold\n#endif\n").split() == ["new"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('#include "noise.glsl"\n', "#include"),
        ("#error needs a newer version\n", "#error"),
        ("#if 1\nfloat a;\n", "not closed"),
        ("#endif\n", "without #if"),
        ("#define F(a, b) a\nfloat x = F(1);\n", "takes 2 arguments"),
        ("#if __import__('os')\n#endif\n", "#if"),
        ("#if 1 / 0\n#endif\n", "divides by zero"),
    ],
)
def test_what_cannot_run_is_refused_with_a_reason(text, message):
    with pytest.raises(glsl_pp.PreprocessError, match=message):
        _pp(text)


def test_an_expansion_bomb_stops():
    # Each level doubles the text: 2**40 copies without the limits.
    levels = "".join(f"#define L{k + 1} L{k} L{k}\n" for k in range(40))
    with pytest.raises(glsl_pp.PreprocessError):
        _pp("#define L0 x\n" + levels + "float a = L40;")


def test_a_loop_a_macro_spells_is_guarded():
    out = xs.prepare((SHADERS / "rok_hidden_loop.txt").read_text())
    assert "REPEAT" not in out
    assert "for (; xs_tick();)" in out
