"""Rok's own shader set (rok/shaders): every file follows the set's rules.

The GPU runs are in the smoke test (check_shader_set); these check the files.
"""

from pathlib import Path

import pytest

from rok.core import xs
from rok.core.shader_gallery import GROUP_ORDER as GROUPS

FOLDER = Path(__file__).resolve().parent.parent / "rok" / "shaders"
SHADERS = sorted(FOLDER.rglob("*.txt"))


def test_every_shader_sits_in_a_group():
    assert SHADERS
    for path in SHADERS:
        assert path.parent.parent == FOLDER and path.parent.name in GROUPS, path


@pytest.mark.parametrize("path", SHADERS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_a_shader_of_the_set(path):
    shader = xs.load(path, f"{path.parent.name}/{path.stem}")
    # Written for Rok.
    assert shader.author == "Rok", shader.author
    # Rok runs up to xs.MAX_ARGS arguments (the set may go past
    # the xs format's documented eight).
    assert 0 < len(shader.args) <= xs.MAX_ARGS
    # Every argument has a tooltip, and every rok_tip line names an argument.
    for arg in shader.args:
        assert len(arg.tip) >= 40, f"{arg.name}: no rok_tip line, or too short"
    names = {a.name.strip().lower() for a in shader.args}
    assert set(xs.parse_tips(shader.source)) <= names, set(xs.parse_tips(shader.source)) - names
    by_index = "iArgs" in shader.source or "i_args" in shader.source
    for arg in shader.args:
        assert arg.low <= arg.value <= arg.high, (arg.name, arg.value)
        assert arg.var or by_index, f"{arg.name} has no var name"
    names = [a.name for a in shader.args]
    # A shader that tiles takes both tile numbers.
    assert ("Tile X" in names) == ("Tile Y" in names)
    # A rok_colours line that Rok cannot read would silently do nothing.
    if "rok_colours" in shader.source:
        assert shader.colours is not None
    # A line on what it makes, for the gallery: short enough for a card.
    assert 10 <= len(shader.about) <= 60, shader.about
    # The usage line names the shader as the menu does.
    assert f"// xs {shader.name}" in shader.source
    # Rok takes it for the GPU.
    assert "map" in xs.prepare(shader.source)


def test_the_patterns_only_paint():
    for path in (FOLDER / "patterns").glob("*.txt"):
        assert xs.is_brush(xs.load(path, f"patterns/{path.stem}")), path
