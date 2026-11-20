"""View > Theme (rok/app/theme.py): the Bright table, the switch back to Dark,
the contrast of text on its plates, and the view's background."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

from rok.app import theme

from test_app_looks import app  # noqa: F401 - the recording app fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def dark_after():
    yield
    theme.set_inks(theme.DARK)


def _is_colour(v: ast.expr) -> bool:
    """A literal colour: 3 or 4 numbers in 0..1 (one a float), a hex call, or a numpy array of those."""
    if isinstance(v, ast.Tuple) and len(v.elts) in (3, 4):
        if all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in v.elts):
            vals = [e.value for e in v.elts]
            return any(isinstance(x, float) for x in vals) and all(0 <= x <= 1 for x in vals)
        return False
    if isinstance(v, ast.Call):
        f = v.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
        if name in ("_srgb", "_rgba", "_hex"):
            return bool(v.args) and isinstance(v.args[0], ast.Constant) and isinstance(v.args[0].value, str)
        if name == "array" and v.args and isinstance(v.args[0], (ast.List, ast.Tuple)):
            return _is_colour(ast.Tuple(elts=v.args[0].elts))
    return False


def _module_colours():
    paths = sorted((ROOT / "rok" / "app").glob("*.py")) + [ROOT / "rok" / "core" / "hud.py"]
    for path in paths:
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        if module == "rok.app.theme":
            continue
        for node in ast.parse(path.read_text()).body:
            target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
            value = getattr(node, "value", None)
            if isinstance(target, ast.Name) and target.id.isupper() and value is not None and _is_colour(value):
                yield module, target.id


def test_every_ui_colour_has_a_bright_value_or_is_shared():
    missing = [f"{m}.{name}" for m, name in _module_colours()
               if name not in theme.BRIGHT_INKS.get(m, {}) and name not in theme.BOTH.get(m, set())]
    assert not missing, "add these to theme.BRIGHT_INKS (or theme.BOTH): " + ", ".join(missing)


def test_every_bright_name_exists_with_the_same_shape():
    theme.set_inks(theme.DARK)
    for module_name, inks in theme.BRIGHT_INKS.items():
        module = importlib.import_module(module_name)
        for name, bright in inks.items():
            assert hasattr(module, name), f"{module_name}.{name}"
            dark = getattr(module, name)
            if isinstance(dark, (tuple, np.ndarray)):
                assert len(bright) == len(dark), f"{module_name}.{name}"
            elif isinstance(dark, dict) and bright is not None:
                assert set(bright) >= set(dark) or not dark, f"{module_name}.{name}"


def test_dark_comes_back_exactly():
    from rok.app import chrome, overlay

    before = (chrome.RAIL_BG, chrome._SCENE_INKS, overlay.STRIP_PLATE_BG, overlay.TOOL_COLOURS)
    theme.set_inks(theme.BRIGHT)
    assert overlay.STRIP_PLATE_BG != before[2] and theme.is_bright()
    theme.set_inks(theme.DARK)
    assert (chrome.RAIL_BG, chrome._SCENE_INKS, overlay.STRIP_PLATE_BG, overlay.TOOL_COLOURS) == before
    assert not theme.is_bright()


def _luminance(c) -> float:
    rgb = np.asarray(c[:3], dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(lin @ [0.2126, 0.7152, 0.0722])


def _contrast(a, b) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("module_name, ink, plate", [
    ("rok.app.chrome", "TEXT", "PANEL_BG"),
    ("rok.app.chrome", "GLYPH", "RAIL_BG"),
    ("rok.app.command", "TEXT", "PANEL_BG"),
    ("rok.app.dialog", "TEXT", "PANEL"),
    ("rok.app.dialog", "TEXT", "BUTTON"),
    ("rok.app.view_panel", "TEXT", "PANEL_BG"),
    ("rok.app.shader_panel", "TEXT", "PANEL_BG"),
    ("rok.app.welcome", "TEXT", "PANEL_BG"),
    ("rok.app.history_strip", "TEXT", "CARD_BG"),
    ("rok.app.dock", "TEXT", "DOCK_BG"),
    ("rok.app.marking", "TEXT", "PILL"),
    ("rok.app.overlay", "STRIP_LABEL_INK", "STRIP_PLATE_BG"),
    ("rok.app.ui", "HINT_TEXT", "HINT_PLATE"),
])
def test_bright_text_reads_on_its_plate(module_name, ink, plate):
    theme.set_inks(theme.BRIGHT)
    module = importlib.import_module(module_name)
    assert _contrast(getattr(module, ink), getattr(module, plate)) >= 4.5


@pytest.mark.parametrize("inks", ["_INSIDE_INKS", "_SCENE_INKS", "_ANIM_INKS"])
def test_each_bright_rail_keeps_its_glyphs_readable(inks):
    rail = theme.BRIGHT_INKS["rok.app.chrome"][inks]
    assert _contrast(rail["GLYPH"], rail["RAIL_BG"]) >= 4.5


@pytest.mark.parametrize("ink", ["HUD_TITLE_RGBA", "HUD_TIP_RGBA"])
def test_bright_tool_readout_reads_on_the_view(ink):
    from rok.core import hud

    theme.set_inks(theme.BRIGHT)
    assert _contrast(getattr(hud, ink), tuple(theme.VIEW_BACKGROUND) + (1.0,)) >= 4.5


def test_the_active_tool_tiles_carry_dark_glyphs():
    chrome_inks = theme.BRIGHT_INKS["rok.app.chrome"]
    for accent in list(chrome_inks["TILE_ACCENTS"].values()) + [chrome_inks["EMBER"]]:
        assert _contrast(chrome_inks["ACTIVE_INK"], accent) >= 3.0


def test_bright_lightens_only_the_looks_without_a_background(app, tmp_path):  # noqa: F811
    app.handle("menu", {"id": "rok.theme(name='BRIGHT')"})
    assert app.host.named("set_look")[-1][2]["background"] == theme.VIEW_BACKGROUND
    assert (tmp_path / "config" / theme.FILE_NAME).read_text().strip() == "BRIGHT"
    app.handle("menu", {"id": "rok.look(name='NIGHT')"})
    assert app.host.named("set_look")[-1][2]["background"] == (0.03, 0.04, 0.08)
    app.handle("menu", {"id": "rok.look(name='STUDIO')"})
    app.handle("menu", {"id": "rok.theme(name='DARK')"})
    assert app.host.named("set_look")[-1][2]["background"] is None


def test_the_view_menu_lists_both_themes(app):  # noqa: F811
    from rok.app import ui

    items = dict((it[0], it[1]) for it in ui.view_menu(app)[0][1] if it is not None)
    theme_items = items["Theme"]
    assert [label for _id, label, _k, _on in theme_items] == ["Dark", "Bright"]
    assert [on for *_rest, on in theme_items] == [1, 0]
