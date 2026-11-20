"""The UI theme: View > Theme picks Dark or Bright.

Each drawing module keeps its colours as module constants, in the Dark
theme's values. Bright is a table of new values for those constants, module
by module. `apply` writes the table into the modules (and keeps the Dark
values the first time), so the drawing code reads its own constants as before
and needs no switch of its own. The chosen theme is kept for the next start.

Bright: white and cream plates, deep indigo ink, a coral accent, and a pale
sky behind the voxels. Each level keeps its rail colour, in a light tint:
white for the model, sky blue for the scene, lavender for Animate.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from rok.core.safe_write import write_text

from .handlers import registry
from .opkit import Action, EnumProp, register

DARK, BRIGHT = "DARK", "BRIGHT"
THEMES = {DARK: "Dark", BRIGHT: "Bright"}
DEFAULT = DARK
FILE_NAME = "theme.txt"


def _hex(hex_rgb: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = hex_rgb.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, a)


def _rgb(hex_rgb: str) -> tuple[float, float, float]:
    return _hex(hex_rgb)[:3]


# The Bright palette.
INK = _hex("2E2A4F")  # text and glyphs
INK_DIM = _hex("6C6790")
INK_FAINT = _hex("A29EBF")
PAPER = _hex("FFFFFF")  # plates
CREAM = _hex("FFFAF2")  # panels and dialogs
HAIR = _hex("E7E0F2")  # hairlines and outlines
HOVER = _hex("FFEFD3")  # a hovered row or button: butter
FIELD = _hex("F4F0FA")  # a text field, a track
CORAL = _hex("FF6B57")  # the accent: what is active
CORAL_HOVER = _hex("FF8670")
SUN = _hex("FFC23D")
MINT = _hex("22C98E")
SKY = _hex("3D9BFF")
GRAPE = _hex("8C6CF5")
BERRY = _hex("F0436A")
SHADE = (0.18, 0.16, 0.36)  # indigo for washes and lines in the view
VIEW_BACKGROUND = (0.894, 0.929, 0.969)  # behind the voxels when the look has no background of its own


def _with(colour, a: float) -> tuple[float, float, float, float]:
    return tuple(colour[:3]) + (a,)


_MODEL_RAIL = {
    "RAIL_BG": _hex("FFFFFF", 0.94), "RAIL_EDGE": _hex("ECE5F5"), "WIDGET_HOVER": HOVER,
    "TICK_OFF": INK_FAINT, "GLYPH": INK,
}
_SCENE_RAIL = {
    "RAIL_BG": _hex("D6EAFF", 0.96), "RAIL_EDGE": _hex("A5CCFA"), "WIDGET_HOVER": _hex("BFDDFF"),
    "TICK_OFF": _hex("5E8CC2"), "GLYPH": _hex("1C4B85"),
}
_ANIM_RAIL = {
    "RAIL_BG": _hex("ECE4FF", 0.96), "RAIL_EDGE": _hex("C9B9FF"), "WIDGET_HOVER": _hex("DCCFFF"),
    "TICK_OFF": _hex("8A78D0"), "GLYPH": _hex("43309E"),
}

# Module by module: the constant's name and its Bright value.
BRIGHT_INKS: dict[str, dict[str, object]] = {
    "rok.app.chrome": {
        **_MODEL_RAIL,
        "_INSIDE_INKS": _MODEL_RAIL, "_SCENE_INKS": _SCENE_RAIL, "_ANIM_INKS": _ANIM_RAIL,
        "RAIL_TILE_RADIUS": 12,
        "TILE_ACCENTS": {"CUBE_PLUS": MINT, "CUBE_MINUS": BERRY, "BRUSH": SUN, "MARQUEE": SKY},
        "ACTIVE_FILL_ALPHA": 1.0, "ACTIVE_INK": _hex("1E1B3A"),
        "PANEL_BG": CREAM, "EMBER": CORAL, "FOCUS": CORAL,
        "GLYPH_HOVER": _hex("17142F"), "GLYPH_ON": PAPER, "OUTLINE": HAIR, "TEXT": INK,
        "SLIDER_GROOVE": HAIR, "SLIDER_GHOST": _with(INK, 0.45),
        "SWITCH_TRACK": _hex("EFEAF7", 0.95), "SWITCH_TRACK_EDGE": HAIR,
        "THUMB_SCENE": _hex("8CC3FF"), "THUMB_MODEL": _hex("FFD27A"), "THUMB_ANIM": _hex("B9A3FF"),
        "THUMB_EDGE": _hex("FFFFFF"), "SWITCH_HOVER": _with(SHADE, 0.06), "HAIRLINE": _with(SHADE, 0.12),
        "LIST_DIM": _with(INK, 0.40), "LIST_SELECTED": _hex("FFE3B8"),
        "GALLERY_BG": _hex("F6F1FF", 0.98), "GALLERY_EDGE": _hex("D9CCFF"),
        "CARD_BG": _hex("FFFFFF"), "CARD_HOVER": _hex("8C6CF5", 0.18), "CARD_STAGE": _hex("ECE4FF"),
        "GALLERY_TEXT": _hex("2E2366"), "GALLERY_DIM": _hex("6E62A8"), "GALLERY_FAINT": _hex("A79DD6"),
    },
    "rok.app.key_cursor": {"CURSOR_INK": _with(SHADE, 0.85)},
    "rok.app.overlay": {
        "TOOL_COLOURS": {"ATTACH": _rgb("17B86A"), "ERASE": _rgb("F0435A"), "SELECT": _rgb("F29A00"),
                         "ACTIVATE": SHADE},
        "BOX_COLOUR": _with(SHADE, 0.35), "EMBER_RGB": CORAL[:3], "LINE_RGB": SHADE,
        "TEXT_COLOUR": INK, "AMBER": _rgb("F29A00"), "MASK_RGB": _rgb("3D7BFF"),
        "BUTTON_BG": _with(SHADE, 0.06), "BUTTON_BG_ON": _with(CORAL, 0.9), "EMBER": CORAL,
        "CAGE_GLYPH_BG": _hex("FFFFFF", 0.94), "BAR_BG": _hex("FFFFFF", 1.0), "BAR_EDGE": HAIR, "CAGE_GLYPH_ON": PAPER, "CAGE_GLYPH_HOVER": HOVER,
        "STRIP_PLATE_BG": _hex("FFFFFF", 0.94), "STRIP_PLATE_EDGE": HAIR, "TIP_BG": CREAM,
        "STRIP_CAPTION_INK": _with(INK_DIM, 0.8), "STRIP_LABEL_INK": _with(INK, 0.85),
        "STRIP_TILE_EDGE": _with(SHADE, 0.18), "STRIP_HOVER_RING": INK, "STRIP_DIVIDER": HAIR,
        "STRIP_BUTTON_BG": FIELD, "STRIP_BUTTON_HOVER": HOVER,
        "PANEL_BG": _hex("FFFFFF", 0.90), "PILL_BG": _hex("FFFFFF", 0.94),
        "PANEL_FLAT_BG": _hex("FFFFFF", 0.94), "PANEL_FLAT_EDGE": HAIR,
        "GLYPH_INK": INK, "GLYPH_HOVER_INK": _hex("17142F"),
        "CORNER_CENTRE_RGBA": INK, "CAGE_CHIP_LIT_INK": PAPER,
        "SIZE_FIELD_BG": PAPER, "SIZE_FIELD_INK": INK, "SIZE_FIELD_INK_LIT": PAPER, "SIZE_FIELD_OFF": INK_FAINT,
    },
    "rok.app.nav_cube": {
        "DISC": _hex("FFFFFF", 0.70), "DISC_EDGE": HAIR, "EDGE_INK": _hex("C9BFE0"),
        "FACE": np.array(_rgb("F6F1FF")), "FACE_TOP": np.array(_rgb("FFFFFF")),
        "LABEL": INK[:3], "EMBER": CORAL[:3], "BUTTON": _hex("FFFFFF", 0.92), "BUTTON_HOVER": HOVER,
        "GLYPH": INK, "GLYPH_HOT": CORAL, "ON": _with(CORAL, 0.95),
    },
    "rok.app.history_strip": {
        "CARD_BG": _hex("FFFFFF", 0.94), "CARD_EDGE": HAIR, "TRACK": _with(SHADE, 0.14),
        "DONE": INK, "UNDONE": INK_FAINT, "EMBER": CORAL, "HOVER": HOVER,
        "TEXT": INK, "TEXT_DIM": INK_DIM, "TEXT_FAINT": INK_FAINT,
    },
    "rok.app.colour_picker": {
        "PLATE": CREAM, "EDGE": HAIR, "EMBER": CORAL, "BUTTON_BG": FIELD, "BUTTON_HOVER": HOVER,
        "TEXT": INK, "TEXT_DIM": INK_DIM,
    },
    "rok.app.shader_panel": {
        "PANEL_BG": CREAM, "PANEL_EDGE": HAIR, "STAGE_BG": _hex("E9F1FB"), "TRACK_BG": _with(SHADE, 0.10),
        "BUTTON_BG": FIELD, "BUTTON_HOVER": HOVER, "EMBER": CORAL, "EMBER_DARK": PAPER,
        "TEXT": INK, "TEXT_DIM": INK_DIM, "TEXT_FAINT": INK_FAINT,
    },
    "rok.app.stroke_panel": {
        "PANEL_BG": CREAM, "PANEL_EDGE": HAIR, "WELL_BG": _hex("F4F0FA"), "WELL_EDGE": HAIR,
        "GRID": _with(SHADE, 0.08), "CURVE": CORAL, "CURVE_FILL": _with(CORAL, 0.16), "CURVE_OFF": INK_FAINT,
        "POINT": INK, "HANDLE": INK_DIM, "HANDLE_LINE": _with(SHADE, 0.28),
        "BUTTON_BG": FIELD, "BUTTON_HOVER": HOVER, "EMBER": CORAL, "EMBER_DARK": PAPER,
        "TEXT": INK, "TEXT_DIM": INK_DIM, "TEXT_FAINT": INK_FAINT,
        "SWITCH_OFF": _with(SHADE, 0.12), "SWITCH_THUMB": PAPER, "HAIRLINE": _with(SHADE, 0.10),
    },
    "rok.app.view_panel": {
        "PANEL_BG": CREAM, "PANEL_EDGE": HAIR, "TRACK_BG": _with(SHADE, 0.10), "BUTTON_BG": FIELD,
        "BUTTON_HOVER": HOVER, "SECTION_BG": _with(SHADE, 0.035), "EMBER": CORAL, "EMBER_DARK": PAPER,
        "TEXT": INK, "TEXT_DIM": INK_DIM, "TEXT_FAINT": INK_FAINT,
    },
    "rok.app.shader_gallery": {
        "PANEL_BG": CREAM, "PANEL_EDGE": HAIR, "CARD_BG": PAPER, "CARD_HOVER": HOVER,
        "STAGE_BG": _hex("E9F1FB"), "BUTTON_BG": FIELD, "EMBER": CORAL, "SCRIM": _with(SHADE, 0.18),
        "TEXT": INK, "TEXT_DIM": INK_DIM, "TEXT_FAINT": INK_FAINT,
    },
    "rok.app.command": {
        "PANEL_BG": _hex("FFFAF2", 0.98), "PANEL_EDGE": HAIR, "ROW_HOVER": HOVER,
        "ROW_ON": _with(CORAL, 0.22), "TEXT": INK, "TEXT_DIM": INK_DIM,
    },
    "rok.app.dialog": {
        "PANEL": _hex("FFFAF2", 0.98), "EDGE": HAIR, "SCRIM": _with(SHADE, 0.18), "FIELD": FIELD, "FIELD_HOVER": HOVER,
        "BUTTON": _hex("F1ECF8"), "PICKED": CORAL, "TEXT": INK, "DIM": INK_DIM, "CARD": _with(SHADE, 0.05),
    },
    "rok.app.draw": {
        "EMBER": CORAL, "TEXT": INK, "TEXT_DIM": _with(INK, 0.6), "PLATE": _hex("FFFFFF", 0.92),
        "ATTACH_GREEN": _hex("17B86A"), "ERASE_RED": _hex("F0435A"), "BOX_GREY": _with(SHADE, 0.45),
    },
    "rok.app.palette_ui": {
        "MARKER_DARK": _with(SHADE, 0.7), "SLIDER_FILL": _with(CORAL, 0.6), "TEXT_MARK": _with(CORAL, 0.35),
        "TIP_BG": CREAM, "STAR_ON": SUN, "STAR_OFF": INK_FAINT,
    },
    "rok.app.ui": {"HINT_TEXT": _with(INK, 0.85), "HINT_PLATE": _hex("FFFFFF", 0.82)},
    "rok.app.quad": {"SEAM": _with(SHADE, 0.25), "ACTIVE_EDGE": _with(CORAL, 0.75), "LABEL": _with(INK, 0.75)},
    "rok.app.object": {"FADE_TOWARD": VIEW_BACKGROUND},
    "rok.app.dock": {
        "DOCK_BG": _hex("F6F1FF", 0.97), "DOCK_EDGE": _hex("D9CCFF"), "NAME_BG": _hex("EDE5FF"),
        "ROW_STRIPE": _with(GRAPE, 0.05), "ROW_PICKED": _with(GRAPE, 0.16), "VIOLET": GRAPE,
        "VIOLET_LIGHT": _hex("7A58EE"), "SPAN": _with(GRAPE, 0.35), "SPAN_HOLD": _with(GRAPE, 0.15),
        "EMBER": CORAL, "KEY_FILL": _hex("FFFFFF"), "KEY_RING": _hex("43309E"),
        "PAST_END": _with(GRAPE, 0.08), "BUTTON_BG": _with(GRAPE, 0.08), "BUTTON_HOVER": _with(GRAPE, 0.16),
        "BUTTON_ON": _with(GRAPE, 0.45),
        "TEXT": _hex("2E2366"), "TEXT_DIM": _hex("6E62A8"), "TEXT_FAINT": _hex("A79DD6"),
    },
    "rok.app.light_dome": {
        "DISC": _hex("E9F1FB"), "GUIDE": _with(SHADE, 0.10), "RING": _with(SHADE, 0.08),
        "RING_LIT": _with(SHADE, 0.18), "EDGE": HAIR, "EMBER": CORAL, "TEXT": INK, "TEXT_FAINT": INK_FAINT,
        "DARK": PAPER,
    },
    "rok.app.onion": {"BEFORE": _with(GRAPE, 0.22), "AFTER": _with(CORAL, 0.22)},
    "rok.app.marking": {
        "CHIP_BG": _hex("FFFFFF", 0.96), "PILL": _hex("FFFFFF", 0.96), "TEXT": INK, "TEXT_DIM": INK_DIM,
        "DOT_IDLE": INK_FAINT,
        "FAMILY": None,  # filled below: the star's families, in brighter tones
    },
    "rok.app.scene_tools": {"CUT_FILL": _with(CORAL, 0.25), "CUT_EDGE": CORAL},
    "rok.app.welcome": {
        "PANEL_BG": _hex("FFFAF2", 0.99), "PANEL_EDGE": HAIR, "SCRIM": _with(SHADE, 0.22), "BUTTON_BG": FIELD, "BUTTON_HOVER": HOVER,
        "EMBER": CORAL, "EMBER_HOVER": CORAL_HOVER, "TEXT": INK, "TEXT_DIM": INK_DIM, "TEXT_FAINT": INK_FAINT,
    },
    "rok.core.hud": {
        "HUD_TITLE_RGBA": INK, "HUD_ERASE_RGBA": BERRY, "HUD_TIP_RGBA": _hex("5A557D"),
        "HUD_HALO_RGBA": _hex("FFFFFF", 0.9),
    },
}

# Colours that both themes share: they sit on a swatch, on a picture or on a
# creature's own body, never on a plate or the view's background.
BOTH: dict[str, set[str]] = {
    "rok.app.marking": {"DEEP_INK", "DEEP_SUCKER", "SPARK"},  # the star's playful things
    "rok.app.overlay": {"FRAME_WHITE", "FRAME_BLACK", "FRAME_GREY"},  # rings round a swatch
    "rok.app.palette_ui": {"MARKER_LIGHT"},  # the picker's marker on the colour field
    "rok.app.welcome": {"WHITE"},  # text on the ember button
    "rok.core.hud": {"SLOT0_COLOUR"},  # the empty palette slot's swatch
    # The CAD sheet is white in both themes, so the bar and the side labels stay.
    "rok.app.cad_mode": {"_INK", "_PLATE", "_EDGE", "_CHIP", "_CHIP_ON", "_CHIP_INK"},
}


def _bright_families() -> dict:
    from rok.core import star

    return {
        "ATTACH": MINT, "ERASE": BERRY, "PAINT": SUN, "SELECT": SKY,
        star.MODEL: GRAPE, star.SCENE: _hex("5C8FD6"), star.LAST: _hex("FF9F6B"),
    }


# The Dark values, saved by the first switch.
_dark: dict[str, dict[str, object]] = {}
_current = DARK


def current() -> str:
    return _current


def is_bright() -> bool:
    return _current == BRIGHT


def view_background() -> tuple[float, float, float] | None:
    """The view's background for a look without its own: None keeps the dark default."""
    return VIEW_BACKGROUND if is_bright() else None


def set_inks(name: str) -> None:
    """Write `name`'s colours into the drawing modules."""
    global _current
    if name not in THEMES:
        name = DEFAULT
    if BRIGHT_INKS["rok.app.marking"]["FAMILY"] is None:
        BRIGHT_INKS["rok.app.marking"]["FAMILY"] = _bright_families()
    for module_name, inks in BRIGHT_INKS.items():
        module = importlib.import_module(module_name)
        dark = _dark.setdefault(module_name, {k: getattr(module, k) for k in inks})
        for key, value in (inks if name == BRIGHT else dark).items():
            setattr(module, key, value)
    _current = name


def _file() -> Path:
    from . import recent

    return Path(recent.FOLDER) / FILE_NAME


def apply(app, name: str, save: bool = True) -> None:
    """Show the UI in `name` and remember it for the next start."""
    from . import looks

    set_inks(name)
    looks.apply(app, looks.current(app), save=False)  # the view's background follows the theme
    app.request_redraw()
    if save:
        try:
            _file().parent.mkdir(parents=True, exist_ok=True)
            write_text(_file(), current() + "\n")
        except OSError:
            pass  # the theme still shows; it is only not remembered


@register
class ThemeAction(Action):
    command_id = "rok.theme"
    title = "Theme"
    tooltip = "The colours of the panels, the rail and the background: Dark or Bright"

    name = EnumProp(items=tuple((k, v, "") for k, v in THEMES.items()), default=DEFAULT)

    def execute(self, context):
        apply(context.app, self.name)
        context.app.set_menu()
        return {"FINISHED"}


def menu_items() -> list:
    return [(f"rok.theme(name={k!r})", label, "", int(k == _current)) for k, label in THEMES.items()]


def _start(context) -> None:
    try:
        saved = _file().read_text().strip()
    except OSError:
        saved = DEFAULT
    if saved != DARK:
        apply(context.app, saved, save=False)


registry.startup.append(_start)
