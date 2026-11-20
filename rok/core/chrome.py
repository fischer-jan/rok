"""The new UI's widget kit: layout, hit testing and glyphs.

The chrome's kit. It holds everything about the chrome
that can be decided without a window: where each widget sits at a given UI
scale, what is under a pixel, which widget the keyboard focus goes to next,
when a tooltip is due, and the line lists the glyphs are drawn from.

The chrome is a left rail, full region height:

    Attach / Erase / Paint / Select
    (no mode chip: each tool's flyout picks the build style, Tab opens the
    wheel, and the readout at the lower left names the style)
    ── divider ──
    "Model": one button whose flyout holds the whole-model tools (fill,
    clear, fit, loop, scale, repeat, ...), then "Layers"
    at the bottom, only what applies: the Visible chip for Select, the Size
    slider in Voxel mode
Mirror is not on the rail: X Y Z are round buttons under Recenter and Live
Render in the top right.

Every glyph is drawn in one ink: grey at rest, near white under the mouse,
ember when active. Colour means state, never decoration.

Every measurement is in points; a layout multiplies them by the UI scale
(2.0 on a Retina Mac) and rounds, so every rectangle lands on a pixel. A
region too short for the whole rail (a split screen) squeezes the rail: the
scale goes down until it fits, so no widget is ever off the screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import anim
from .edit import (
    ATTACH,
    BOX,
    BY_COLOUR,
    COLOUR_MODES,
    CENTER,
    COLOUR_CONNECTED,
    COLOUR_CONNECTED_CORNERS,
    CONNECTED,
    CONNECTED_CORNERS,
    ERASE,
    FACE,
    LASSO,
    LINE,
    MODE_ITEMS,
    MODES,
    PAINT,
    SCREEN_MODES,
    PATTERN,
    CLONE,
    GRADIENT,
    RECT,
    SELECT,
    SELECT_ONLY_MODES,
    VOXEL,
)
from .geom2d import Rect, fit_glyph, in_rect, loop_lines
from .shape import BOX_SOLID, SOLID_ITEMS, SOLIDS

# Sizes in points, before the UI scale. 56 @1x, 112 @2x.
RAIL_WIDTH = 56
RAIL_INSET = 12  # tools from the top, chips from the bottom
TOOL_HIT = 48  # Attach / Erase / Paint / Select
TOOL_ICON = 32  # 32 @1x, 64 @2x
TOOL_GAP = 8
DIVIDER_THICKNESS = 1
DIVIDER_MARGIN = 10
TICK_HIT = 36  # mirror and section
TICK_GAP = 4
SECTION_GAP = 8  # between the mirror group and the model group
CHIP_HIT = 40
CHIP_GAP = 8
HEADER_H = 12  # the "Mirror" and "Cut" words over the tick columns
HEADER_GAP = 2
CHIP_HEIGHT = 18  # tooltip height; chips themselves are CHIP_HIT
CHIP_PAD = 6  # left and right of tooltip text
CHIP_CHAR = 6  # width of one character at 11 pt; the layout needs no font
GAP = 4  # between flyout items
FLYOUT_GAP = 4  # between a chip and its hover flyout
TOOLTIP_DELAY_S = 0.2  # HELP copy: 200 ms
# The Size slider (Voxel mode): eight stops on a vertical track, 1 at the bottom.
SIZE_END = 9  # a slider: from the first and last stop to the widget's edge
SIZE_KNOB = 7  # a slider's knob radius
SIZE_TRACK_X = 0.36  # a slider's track, as a fraction of the widget width from the left
SIZE_BOX_H = 26  # the Size field: a number you drag
SIZE_DRAG_PX = 10  # points of drag per step of the brush size
TOLERANCE_DRAG_PX = 4  # points of drag per step of the colour tolerance (0 to 100)
TOLERANCE_HEADER = "Tolerance"
SIZE_HEADER = "Size"

# Command palette (Cmd+K / Ctrl+K).
COMMAND_WIDTH = 420
COMMAND_ROW = 22
COMMAND_MAX = 12
COMMAND_PAD = 8

# Widget kinds. The drawing tells them apart, the hit test does not.
BUTTON_KIND = "BUTTON"
ICON_KIND = "ICON"  # a rail button whose glyph is drawn in code, not a tool
TICK_KIND = "TICK"
CHIP_KIND = "CHIP"
SLIDER_KIND = "SLIDER"

# Widget ids that are acted on. The id says what a press does.
TOOL_IDS = {
    "TOOL_ATTACH": ATTACH,
    "TOOL_ERASE": ERASE,
    "TOOL_PAINT": PAINT,
    "TOOL_SELECT": SELECT,
}
TOOL_KEYS = {"TOOL_ATTACH": "T", "TOOL_ERASE": "R", "TOOL_PAINT": "G", "TOOL_SELECT": "Z"}
MODEL_ID = "MODEL_TOOLS"  # the button under "Model"; its flyout is the whole-model commands
LAYERS_ID = "LAYERS_BUTTON"  # the button under "Layers"; hovering opens the Layers list
# Animate mode, the level switch's third cell: the rail turns violet and holds
# the animation tools; the voxel tools, the palette strip and the file label
# go away, and the timeline dock (rok/core/dock.py) runs along the
# bottom. A click in the view picks a part; the gizmo writes keys.
MOTION_ID = "ANIM_MOTION"  # its flyout is the preset gallery
ANIM_COLOUR_ID = "ANIM_COLOUR"  # the next click on a voxel takes its colour as the target
ANCHOR_ID = "ANIM_ANCHOR"  # the next click on a face puts the picked part's anchor there
MOTION_HEADER = "Motion"
COLOUR_HEADER = "Colour"
ANCHOR_HEADER = "Anchor"
_MOTION_TIP = "Motion — Pick a ready-made motion for the picked part"
_ANIM_COLOUR_TIP = "Colour — Click a voxel next: its colour gets a track of its own"
_ANCHOR_TIP = "Anchor — Click a face next: the picked part turns and scales around it"

# Two levels: inside a model the rail holds the
# voxel tools; in the scene it holds the model tools. No mode name: the rail's
# tools and colour show the level, and the level chip at its top names it.
LEVEL_MODEL, LEVEL_SCENE = "MODEL", "SCENE"
# Brick mode (Edit > Brick Mode) edits the cells of one model, as Model level
# does, but with bricks from the bin instead of the voxel tools. It is not a
# cell of the level switch: there the Model cell stays lit.
LEVEL_BRICK = "BRICK"
# The level switch at the top of the rail:
# a sunken track holding three stacked cells, Model, Scene and Animate from
# the top. A thumb slides to the current one; a
# click on another cell goes there. A boxed set reads as a mode, loose icons
# as tools; a hairline under it sets it apart.
LEVEL_SCENE_ID = "LEVEL_SCENE_SEG"
LEVEL_MODEL_ID = "LEVEL_MODEL_SEG"
LEVEL_ANIM_ID = "LEVEL_ANIM_SEG"
LEVEL_IDS = (LEVEL_MODEL_ID, LEVEL_SCENE_ID, LEVEL_ANIM_ID)  # top to bottom
# The Brick rail: the four brick tools, then the Convert chip that applies.
BRICK_PLACE_ID, BRICK_PULL_ID, BRICK_PAINT_ID, BRICK_PICK_ID = "BRICK_PLACE", "BRICK_PULL", "BRICK_PAINT", "BRICK_PICK"
BRICK_TOOL_IDS = {BRICK_PLACE_ID: "PLACE", BRICK_PULL_ID: "PULL", BRICK_PAINT_ID: "PAINT", BRICK_PICK_ID: "PICK"}
_BRICK_TOOL_TIPS = {
    BRICK_PLACE_ID: "Place — Set the brick in your hand (B)",
    BRICK_PULL_ID: "Pull — Take the brick under the mouse out (U)",
    BRICK_PAINT_ID: "Paint — Give the brick under the mouse the current colour (G)",
    BRICK_PICK_ID: "Pick — Take the shape and colour of the brick under the mouse (I)",
}
BRICK_CONVERT_ID = "BRICK_CONVERT"  # on a voxel model: Convert to Bricks
BRICK_TO_VOXELS_ID = "BRICK_TO_VOXELS"  # on a brick model: Convert to Voxels
CONVERT_HEADER = "Convert"
_BRICK_CONVERT_TIP = "Convert to Bricks — Every voxel of the model becomes a 1×1 brick"
_BRICK_TO_VOXELS_TIP = "Convert to Voxels — The model becomes plain voxels again; the bricks are gone for good"
SEGMENT_KIND = "SEGMENT"
SEGMENT_W, SEGMENT_H = 46, 28  # a cell
SEGMENT_GAP = 2
LEVEL_PAD = 3  # the track around the cells
LEVEL_H = len(LEVEL_IDS) * SEGMENT_H + (len(LEVEL_IDS) - 1) * SEGMENT_GAP + 2 * LEVEL_PAD  # the track's height
LEVEL_GAP = 14  # under it: room for its hairline
SCENE_PICK_ID = "SCENE_PICK"  # the scene's first tool: pick, drag and stretch models
SCENE_CUT_ID = "SCENE_CUT"  # the scene's knife: slice a model in two
# The scene's Tools button: its flyout holds what acts on
# the picked models, as the Model button's does inside a model.
SCENE_TOOLS_ID = "SCENE_TOOLS"
TOOLS_HEADER = "Tools"
_TOOLS_TIP = "Tools — Drop, fit, flip, line up, repeat, carve or delete the picked models"
SCENE_DROP_ID = "SCENE_DROP"  # the picked models go down onto the floor grid
SCENE_DROP_ONTO_ID = "SCENE_DROP_ONTO"  # they go down onto the first voxel below
SCENE_FIT_ID = "SCENE_FIT"  # shrink the picked models' grids to their voxels
SCENE_FLIP_ID = "SCENE_FLIP"  # a click opens X, Y and Z under it
FLIP_IDS = ("SCENE_FLIP_X", "SCENE_FLIP_Y", "SCENE_FLIP_Z")
SCENE_DELETE_ID = "SCENE_DELETE"  # remove the picked models from the scene
SCENE_PICK_LINKED_ID = "SCENE_PICK_LINKED"  # pick every linked copy of the picked model
SCENE_ALIGN_ID = "SCENE_ALIGN"  # line the picked models up, or spread them
SCENE_ARRAY_ID = "SCENE_ARRAY"  # repeat the picked models in a row
SCENE_CARVE_ID = "SCENE_CARVE"  # the picked model cuts its shape out of the others
SCENE_TOOL_ITEMS = (  # (id, glyph, tooltip), in the flyout's order
    (SCENE_DROP_ID, "DROP_FLOOR", "Drop to Floor · G — Put the picked models on the floor grid"),
    (SCENE_DROP_ONTO_ID, "DROP_ONTO", "Drop · Shift G — Move the picked models down until they rest on a voxel"),
    (SCENE_FIT_ID, "FIT_GRID", "Fit — Shrink the picked models' grids to their voxels"),
    (SCENE_FLIP_ID, "FLIP", "Flip — Mirror the picked models across X, Y or Z; click to pick the axis"),
    (SCENE_PICK_LINKED_ID, "PICK_LINKED", "Pick Linked Copies — Pick every linked copy of the picked model"),
    (SCENE_ALIGN_ID, "ALIGN", "Align — Line the picked models up on the picked one, or spread them evenly"),
    (SCENE_ARRAY_ID, "ARRAY", "Array — Repeat the picked models in a row along X, Y or Z"),
    (SCENE_CARVE_ID, "CARVE", "Carve — Cut the picked model's shape out of the models it overlaps"),
    (SCENE_DELETE_ID, "DELETE", "Delete · Delete — Remove the picked models from the scene"),
)
SCENE_TOOL_IDS = tuple(ident for ident, _g, _t in SCENE_TOOL_ITEMS)
NEW_ID = "NEW_BUTTON"  # the scene rail's New button; its flyout makes models
NEW_HEADER = "New"
NEW_FLYOUT_IDS = ("NEW_EMPTY", "NEW_FILE", "NEW_TERRAIN", "NEW_HEIGHTMAP")
_NEW_GLYPHS = {"NEW_EMPTY": "CUBE_PLUS", "NEW_FILE": "FROM_FILE", "NEW_TERRAIN": "TERRAIN", "NEW_HEIGHTMAP": "HEIGHTMAP"}
_NEW_TIPS = {
    "NEW_EMPTY": "Empty model — A new 40 x 40 x 40 model, in the picked model's layer",
    "NEW_FILE": "From file — Add the models of a .vox file",
    "NEW_TERRAIN": "Terrain — A landscape of up to 5 x 5 models that meet without a seam",
    "NEW_HEIGHTMAP": "Heightmap — A model from a picture: the brighter a pixel, the taller its column",
}
_NEW_TIP = "New — Make a model: empty, a shape, or from a file"
_PICK_TIP = "Pick · Drag a model to move it, its dots to stretch it; double-click steps in"
_CUT_TIP = "Cut · C — Click a model to cut it in two; Tab turns the plane"
LEVEL_FADE_S = 0.15  # the rail's colour change between the levels
DOUBLE_CLICK_S = 0.45  # long enough that slow double-clicks count too


def level_mix(level: str, since: float, now: float, prev: str | None = None) -> float:
    """How far the rail's colours are toward the scene's: 0 inside a model,
    1 in the scene, in between for LEVEL_FADE_S after a change. With `prev`,
    the level before the change, a change between two levels that are not
    the scene stays at 0."""
    if prev is not None and LEVEL_SCENE not in (level, prev):
        return 0.0
    t = 1.0 if LEVEL_FADE_S <= 0 else min(1.0, max(0.0, (now - since) / LEVEL_FADE_S))
    return t if level == LEVEL_SCENE else 1.0 - t


def anim_mix(animating: bool, since: float, now: float) -> float:
    """How far the rail's colours are toward Animate's violet: 1 in Animate
    mode, 0 out of it, in between for LEVEL_FADE_S after a change."""
    t = 1.0 if LEVEL_FADE_S <= 0 else min(1.0, max(0.0, (now - since) / LEVEL_FADE_S))
    return t if animating else 1.0 - t


def is_double_click(last: tuple[int, float] | None, uid: int, now: float) -> bool:
    """True when the click on model `uid` at `now` is the second of a double-click."""
    return last is not None and last[0] == uid and 0.0 <= now - last[1] <= DOUBLE_CLICK_S


SIZE_ID = "SIZE_SLIDER"
STROKE_ID = "STROKE_DESIGN"  # above the Size field: opens the Stroke Designer
STROKE_HIT = 32
TOLERANCE_ID = "TOLERANCE_SLIDER"  # Select by colour: how close a colour must be to count
VISIBLE_ID = "CHIP_VISIBLE"  # Select: take only the voxels the view shows
MASK_ID = "CHIP_MASK"  # while the model has a mask: shows it, a click clears it
_MASK_TIP = "Masked — Attach, Erase and Paint leave the masked voxels alone. Click to clear the mask (Option+M)"
STAMP_ID = "CHIP_STAMP"  # Attach in Pattern mode: add, carve or holes
# Paint in Gradient mode: how many colours, the end colour, and dither on or off.
GRADIENT_STEPS_ID = "CHIP_GRADIENT_STEPS"
GRADIENT_END_ID = "CHIP_GRADIENT_END"
GRADIENT_DITHER_ID = "CHIP_GRADIENT_DITHER"
_GRADIENT_TIPS = {
    GRADIENT_STEPS_ID: "How many colours the gradient has: click for one more, Shift+click for one less",
    GRADIENT_END_ID: "The colour the gradient ends in: click to pick it",
    GRADIENT_DITHER_ID: "Mix neighbouring colours where they meet, in a fine ordered pattern",
}
STAMP_GLYPHS = {"ADD": "CUBE_PLUS", "CARVE": "CUBE_MINUS", "HOLES": "HOLLOW"}
STAMP_NAMES = {"ADD": "Add", "CARVE": "Carve", "HOLES": "Holes"}
_STAMP_TIPS = {
    "ADD": "Add — The stamp builds the pattern's shape. Click for Carve (N)",
    "CARVE": "Carve — The stamp cuts the pattern's shape out of the model. Click for Holes (N)",
    "HOLES": "Holes — The stamp sinks in, builds the shape and clears the pattern's empty cells. Click for Add (N)",
}
VISIBLE_HEADER = "Visible"
VISIBLE_MODES = (BOX, CENTER, RECT, LASSO)  # the Select modes that take through the model
# The Primitives item ends Attach's flyout row. A click opens the grid of
# primitives under it: then PRIMITIVES_ID is the open flyout, and its
# "flyout" is Attach's row plus the grid.
PRIMITIVES_ID = "TOOL_ATTACH_PRIMITIVES"
PRIMITIVE_IDS = tuple(f"PRIMITIVE_{solid}" for solid in SOLIDS)
SOLID_OF_ID = dict(zip(PRIMITIVE_IDS, SOLIDS))
PRIMITIVE_COLUMNS = 6
# New's flyout ends in the same grid of shapes: a click makes a model of that
# shape next to the scene, a drag places it where it is dropped.
NEW_SOLID_IDS = tuple(f"NEW_SOLID_{solid}" for solid in SOLIDS)
NEW_SOLID_OF_ID = dict(zip(NEW_SOLID_IDS, SOLIDS))
GRID_SOLID_OF_ID = {**SOLID_OF_ID, **NEW_SOLID_OF_ID}  # both grids, for drawing


def _flyout_modes(tool: str) -> tuple[str, ...]:
    """The modes a tool button offers on hover: Select its click mode, Face
    (the flat face under the mouse), the screen modes and
    Box (a block on a face, with a height), the others every
    face mode. Erase and Paint have the Marquee too: it erases or paints every voxel inside, R rounds it, as
    Select's does. The wheel still has them all."""
    if tool == SELECT:
        return (VOXEL, FACE) + SCREEN_MODES + (BOX,) + SELECT_ONLY_MODES
    face = tuple(m for m in MODES if m not in SCREEN_MODES)
    if tool == ATTACH:
        return face + (CLONE,)  # the stamp with the copied colours
    if tool == PAINT:
        return face + (RECT, GRADIENT)  # a ramp of colours across the model
    return face + (RECT,) if tool == ERASE else face


# Every tool button opens its modes on hover; a click on one sets the tool and
# the mode in one go, so a new user finds every mode without the wheel.
TOOL_FLYOUT_IDS = {
    widget_id: tuple(f"{widget_id}_{mode}" for mode in _flyout_modes(tool))
    for widget_id, tool in TOOL_IDS.items()
}
TOOL_MODE_OF_ID = {
    f"{widget_id}_{mode}": (tool, mode)
    for widget_id, tool in TOOL_IDS.items()
    for mode in _flyout_modes(tool)
}
# The whole-model tools, each a command id in COMMANDS, with its glyph.
MODEL_COMMAND_IDS = (
    "model_split", "fill_all", "fill_volume", "clear_all", "inverse", "fit_grid", "drop_to_floor", "drop_voxels", "smooth",
    "loop", "scale", "repeat", "diagonal", "mirror_half", "lathe", "twist", "rotate_degrees", "remove_slice", "resize",
    "cage", "crease", "convert_to_bricks", "convert_to_voxels",
)
MODEL_GLYPHS = {
    "model_split": "SPLIT",
    "fill_all": "FILL_ALL",
    "fill_volume": "FILL_VOLUME",
    "clear_all": "CLEAR_ALL",
    "inverse": "INVERSE",
    "fit_grid": "FIT_GRID",
    "drop_to_floor": "DROP_FLOOR",
    "drop_voxels": "DROP_VOXELS",
    "smooth": "SMOOTH",
    "loop": "LOOP",
    "scale": "SCALE",
    "repeat": "REPEAT",
    "diagonal": "DIAGONAL",
    "mirror_half": "MIRROR_HALF",
    "lathe": "LATHE",
    "twist": "TWIST",
    "rotate_degrees": "ROTATE_DEGREES",
    "remove_slice": "REMOVE_SLICE",
    "resize": "RESIZE",
    "cage": "CAGE",
    "crease": "CREASE",
    "convert_to_bricks": "BRICK_CONVERT",
    "convert_to_voxels": "BRICK_TO_VOXELS",
}
MODEL_TIPS = {  # the operators' own descriptions, shortened
    "model_split": "Move the selected voxels into a model of their own",
    "fill_all": "Give every filled voxel the current colour",
    "fill_volume": "Make every cell of the grid the current colour",
    "clear_all": "Empty every cell of the grid",
    "inverse": "Empty cells take the current colour; filled cells become empty",
    "fit_grid": "Shrink the grid to the filled voxels",
    "drop_to_floor": "The selected voxels fall until one lands on a voxel; nothing selected: the model falls to the floor",
    "drop_voxels": "Each selected voxel falls on its own, with a small bounce, until it rests on the floor or on another voxel",
    "smooth": "Corners and bumps go, dents fill; again for smoother",
    "loop": "Shift the voxels along an axis; they wrap around at the edge",
    "scale": "Double or halve the model's resolution; click to pick which",
    "repeat": "Tile the model along each axis",
    "diagonal": "Mirror across the x = y diagonal, swapping x and y",
    "mirror_half": "Copy one half onto the other, mirrored; click to pick the axis",
    "lathe": "Spin what you drew round the centre line into a round solid; click to pick the axis",
    "twist": "Turn each layer a little more than the one below",
    "rotate_degrees": "Turn the selection, or the whole model, by an angle you type: for big models the rings are slow",
    "remove_slice": "Take out the selected layers and close the gap",
    "resize": "Change the grid size",
    "cage": "A cage of points around the selection; drag a point to bend the voxels",
    "crease": "Raise bars in the current colour on the sharp edges, as a frame round flat panes",
    "convert_to_bricks": "Every voxel becomes a 1×1 brick; then Brick mode builds on the model",
    "convert_to_voxels": "The model becomes plain voxels again; the bricks are gone for good",
}
MODEL_FLYOUT_IDS = tuple(f"MODEL_{ident.upper()}" for ident in MODEL_COMMAND_IDS)
COMMAND_OF_MODEL_ID = dict(zip(MODEL_FLYOUT_IDS, MODEL_COMMAND_IDS))
# What the Model flyout shows unless the user picks otherwise:
# Loop, Repeat, Diagonal, Lathe and Twist live in Cmd+K. View > Customize Rail
# Menu and Keys in the menu bar picks the list (rok/app/model_menu.py).
# Rotate by Degrees is off by default: for whoever turns
# very big models and finds the gizmo's rings too slow.
DEFAULT_MODEL_ITEMS = tuple(c for c in MODEL_COMMAND_IDS
                            if c not in ("loop", "repeat", "diagonal", "lathe", "twist", "rotate_degrees"))
# Some Model tools need a choice first. A click on one opens its choices in
# a grid under it; then the tool's flyout id is the open flyout: the Model row
# plus the grid. A choice runs `op` with `props`.


@dataclass(frozen=True)
class Choice:
    op: str
    props: tuple
    label: str  # short: it sits on a square tick
    name: str  # for the readout and Cmd+K, after the tool's name
    tip: str


def mirror_half_name(axis: str, positive: bool) -> str:
    """ "+X onto −X": the half that is kept first."""
    keep, lose = ("+", "−") if positive else ("−", "+")
    return f"{keep}{axis} onto {lose}{axis}"


MIRROR_HALF_ID = "MODEL_MIRROR_HALF"
SCALE_ID = "MODEL_SCALE"
LATHE_ID = "MODEL_LATHE"
# Mirror Half: a column per axis, + onto − on top, − onto + below.
MODEL_CHOICES: dict[str, dict[str, Choice]] = {
    MIRROR_HALF_ID: {
        f"MIRROR_HALF_{axis}_{'POS' if positive else 'NEG'}": Choice(
            "mirror_half",
            (("axis", axis), ("from_positive", positive)),
            f"{'+' if positive else '−'}{axis}",
            mirror_half_name(axis, positive),
            f"Copy the {'+' if positive else '−'} half of {axis} onto the other half, mirrored",
        )
        for positive in (True, False)
        for axis in "XYZ"
    },
    SCALE_ID: {
        "SCALE_UP": Choice("scale", (("mode", "UP"),), "×2", "×2", "Every voxel becomes a 2 x 2 x 2 block"),
        "SCALE_DOWN": Choice("scale", (("mode", "DOWN"),), "×½", "×½", "Every 2 x 2 x 2 block becomes one voxel; thin parts stay closed"),
    },
}
MODEL_CHOICES[SCENE_FLIP_ID] = {
    ident: Choice("scene_flip", (("axis", k),), "XYZ"[k], "XYZ"[k], f"Mirror the picked models across {'XYZ'[k]}")
    for k, ident in enumerate(FLIP_IDS)
}
MODEL_CHOICES[LATHE_ID] = {
    f"LATHE_{axis}": Choice("lathe", (("axis", axis),), axis, f"round {axis}",
                            f"Spin the model round the centre line along {axis}")
    for axis in "ZXY"
}
CHOICE_COLUMNS = {MIRROR_HALF_ID: 3, SCALE_ID: 2, SCENE_FLIP_ID: 3, LATHE_ID: 3}
# The flyout each item with choices sits in: its grid hangs under that row.
CHOICE_PARENT = {MIRROR_HALF_ID: MODEL_ID, SCALE_ID: MODEL_ID, SCENE_FLIP_ID: SCENE_TOOLS_ID,
                 LATHE_ID: MODEL_ID}
CHOICE_OF_ID = {ident: c for group in MODEL_CHOICES.values() for ident, c in group.items()}
CHOICE_IDS = tuple(CHOICE_OF_ID)


# The Animation menu: the animation commands in one flyout,
# under a button like the Model button. View > Show Animation Menu turns it on;
# it is off until then. "BUILD" items run outside Animate mode, "ANIMATE" items in it.
ANIM_MENU_ID = "ANIM_MENU"
ANIM_MENU_HEADER = "Animation"
_ANIM_MENU_TIP = "Animation — A collection of shortcuts for animating models"
ANIM_MENU_ITEMS = (  # (command id, glyph, tip, where it runs)
    ("make_part", "SPLIT", "Cut the selected voxels into a part that follows this model", "BUILD"),
    ("part_empty", "ANCHOR", "A part with no voxels: a root for a character, or a socket for a weapon", "BUILD"),
    ("frame_add", "CUBE_PLUS", "Add a frame to the picked part: a copy of it to paint", "ANIMATE"),
    ("frame_delete", "CUBE_MINUS", "Take the current frame off the picked part", "ANIMATE"),
    ("key_pose", "MOTION", "Key the pose: write the part's turn, move and scale at the play head", "ANIMATE"),
    ("onion_skin", "SEEN", "Show the frames before and after, faintly", "ANIMATE"),
    ("clip_add", "MORE", "A new clip, as a tab on the timeline", "ANIMATE"),
    ("marker_add", "PICK", "Put a named marker at the play head: a footstep, a hit", "ANIMATE"),
)
ANIM_MENU_FLYOUT_IDS = tuple(f"ANIM_MENU_{cid.upper()}" for cid, *_rest in ANIM_MENU_ITEMS)
COMMAND_OF_ANIM_ID = dict(zip(ANIM_MENU_FLYOUT_IDS, (cid for cid, *_rest in ANIM_MENU_ITEMS)))
# The brush sizes the Size field runs through.
BRUSH_SIZES = tuple(range(1, 51))
FLYOUT_OPENERS = (MODEL_ID, LAYERS_ID, NEW_ID, SCENE_TOOLS_ID, MOTION_ID, ANIM_MENU_ID) + tuple(TOOL_IDS)
ALL_FLYOUT_IDS = (
    (PRIMITIVES_ID,) + PRIMITIVE_IDS + MODEL_FLYOUT_IDS + ANIM_MENU_FLYOUT_IDS + CHOICE_IDS + NEW_FLYOUT_IDS + NEW_SOLID_IDS
    + SCENE_TOOL_IDS + sum(TOOL_FLYOUT_IDS.values(), ())
)

_TOOL_GLYPHS = {
    "TOOL_ATTACH": "CUBE_PLUS",
    "TOOL_ERASE": "CUBE_MINUS",
    "TOOL_PAINT": "BRUSH",
    "TOOL_SELECT": "MARQUEE",
}
_TOOL_TIPS = {
    "TOOL_ATTACH": f"Attach · {TOOL_KEYS['TOOL_ATTACH']} — Add blocks where you click or drag",
    "TOOL_ERASE": f"Erase · {TOOL_KEYS['TOOL_ERASE']} — Remove blocks",
    "TOOL_PAINT": f"Paint · {TOOL_KEYS['TOOL_PAINT']} — Recolor blocks with the current color",
    "TOOL_SELECT": f"Select · {TOOL_KEYS['TOOL_SELECT']} — Choose blocks to move or edit",
}
MODE_NAMES = {ident: name for ident, name, _desc in MODE_ITEMS}
MODEL_HEADER = "Model"
LAYERS_HEADER = "Layers"
_LAYERS_TIP = "Layers — Show, hide and pick the layers your models sit on"
# The Layers flyout: each layer is a group of models (the layers of a
# .vox file), its models are listed under it, and the highlighted model is the one being edited. Ids carry the
# layer id (eye, header, move arrow) or the model uid (a model row).
LAYER_EYE, LAYER_ROW, LAYER_MOVE, LAYER_MODEL = "LAYER_EYE_", "LAYER_ROW_", "LAYER_MOVE_", "LAYER_MODEL_"
LAYER_LINK = "LAYER_LINK_"  # the chain on a linked copy's row: a click makes it unique
# A model's live shaders, under its row: the eye switches one, a click opens it.
LIVE_EYE, LIVE_ROW = "LIVE_EYE_", "LIVE_ROW_"
LIST_KIND = "LIST"  # every part of the Layers list
LIST_ROW = 26  # the height of one row
LIST_WIDTH = 230  # the whole row: eye, name, move arrow
LIST_ICON = 26  # the eye's and the arrow's square
LIST_INDENT = 22  # a model row's name sits this far in from its layer's name
_MODEL_TIP = "Whole model — Split, fill, clear, fit, drop, smooth, scale, mirror half, resize"
_SIZE_TIP = "Brush size — A sphere of voxels; drag the number left or right"
_STROKE_TIP = "Stroke — Shape thickness, roundness and colour along the stroke"
_TOLERANCE_TIP = "Colour tolerance — 0 takes the clicked colour only; drag right to take close colours too"
_VISIBLE_TIP = "Visible only — Select takes only the voxels you can see, not the ones behind or inside"
_PRIMITIVES_TIP = "Primitives — Click for the shapes: sphere, cone, stairs and more"
_SOLID_LABELS = {ident: name for ident, name, _desc in SOLID_ITEMS}
_SOLID_TIPS = {ident: f"{name} — {desc}" for ident, name, desc in SOLID_ITEMS}
_NEW_SOLID_TIPS = {
    ident: f"{name} — A new {name.lower()} model; drag it into the view to place it" for ident, name, _d in SOLID_ITEMS
}
_GLYPH_OF_SOLID = {ident: ident for ident in SOLIDS}  # each glyph is named as its solid


@dataclass(frozen=True)
class Widget:
    """One thing in the rail: a button, a tick or a chip.

    `rect` is (x, y, w, h) in region pixels. `label` is drawn when there is no
    glyph; the Size slider keeps its value there. `active` lights the widget in the ember
    colour. A widget that is not `enabled` is drawn dim and takes no press.
    """

    id: str
    kind: str
    rect: Rect
    label: str = ""
    glyph: str = ""
    tooltip: str = ""
    active: bool = False
    enabled: bool = True
    note: str = ""  # a list row: the grey text on the right (a layer's model count)
    dim: bool = False  # a list row drawn fainter (an empty or hidden layer); still takes a press


# One layer as the rail shows it: (id, name, hidden, its models as (uid, name)).
LayerRow = tuple[int, str, bool, tuple[tuple[int, str], ...]]


@dataclass(frozen=True)
class ChromeState:
    """What the rail shows. Read from the window manager and the document."""

    tool: str = ATTACH
    mode: str = VOXEL
    mirror: tuple[bool, bool, bool] = (False, False, False)
    brush_size: int = 1
    colour_tolerance: int = 0  # Select's By colour modes: 0 the same colour only
    solid: str = BOX_SOLID
    select_visible: bool = False
    pattern_space: str = "ADD"  # Attach's stamp in Pattern mode: ADD, CARVE or HOLES
    gradient_steps: int = 5  # Paint's Gradient mode: the number of colours
    gradient_dither: bool = False
    masked: bool = False  # the active model has a mask
    layers: tuple[LayerRow, ...] = ()
    linked: tuple[tuple[int, tuple[str, ...]], ...] = ()  # (uid, the other models that share its shape) of linked copies
    live: tuple[tuple[int, int, str, bool, str], ...] = ()  # live shaders: (uid, id, name, on, error)
    model_uid: int = -1  # the model being edited, -1 without one
    model_name: str = ""
    level: str = LEVEL_MODEL
    back_name: str = ""  # in the scene: the model you were last inside; the level chip goes back to it
    scene_tool: str = "PICK"  # in the scene: "PICK" or "CUT"
    animating: bool = False  # Animate mode: the violet rail and the timeline dock
    anim_pick: bool = False  # Colour is armed: the next voxel click takes its colour
    anim_paint: str | None = None  # Animate mode: the paint tool on (ATTACH, ERASE, PAINT), or None to pose
    anchor_ok: bool = False  # the target is a part: Anchor applies
    model_items: tuple[str, ...] = DEFAULT_MODEL_ITEMS  # the Model flyout's commands, by command id
    anim_menu: bool = False  # View > Show Animation Menu: the Animation button is on the rail
    readout_top: int = 0  # the top of the tool readout above the palette card: a flyout block stays above it
    stroke_on: bool = False  # the Stroke Designer is open: its button is lit
    brick_tool: str = "PLACE"  # Brick mode's tool: PLACE, PULL, PAINT or PICK
    brick_hand: tuple | None = None  # Brick mode: (w, d) of the brick in the hand
    brick_model: bool = False  # the active model is a brick model: To Voxels applies, not Convert


@dataclass(frozen=True)
class ChromeLayout:
    """The left rail for one region size and UI scale."""

    widgets: tuple[Widget, ...]
    rect: Rect  # the drawn rail: full height, RAIL_WIDTH wide
    hit_rect: Rect  # the rail swallows a press; flyouts are extra in contains()
    scale: float  # pixels per point in this rail; below the UI scale when squeezed to fit
    divider: Rect = (0, 0, 0, 0)
    headers: tuple[tuple[str, Rect], ...] = ()  # ("Mirror", rect), ("Cut", rect)
    state: ChromeState = ChromeState()  # what the rail was laid out for: the flyouts read it

    def by_id(self, widget_id: str) -> Widget | None:
        for widget in self.widgets:
            if widget.id == widget_id:
                return widget
        return None


def chrome_layout(
    scale: float, st: ChromeState, region_width: int, region_height: int
) -> ChromeLayout:
    """Lay the left rail out for a region of this size.

    Only the widgets that apply are in it: the Size slider exists in Voxel
    mode. That is the proposal's "chips exist only when they apply".

    A region shorter than the rail at this scale gets the rail at a smaller
    scale, just enough to fit, so the section ticks and the chips stay on the
    screen.
    """
    needed = rail_height(st) * scale
    if needed > region_height > 0:
        scale = scale * region_height / needed

    def px(points: float) -> int:
        return int(round(points * scale))

    rail_w = px(RAIL_WIDTH)
    region_h = int(region_height)
    inset = px(RAIL_INSET)
    tool = px(TOOL_HIT)
    chip_s = px(CHIP_HIT)
    tool_gap = px(TOOL_GAP)
    section_gap = px(SECTION_GAP)
    div_m = px(DIVIDER_MARGIN)
    div_h = max(1, px(DIVIDER_THICKNESS))
    chip_gap = px(CHIP_GAP)
    header_h = px(HEADER_H)
    header_gap = px(HEADER_GAP)

    def cx(hit: int) -> int:
        return (rail_w - hit) // 2

    widgets: list[Widget] = []
    headers: list[tuple[str, Rect]] = []
    y_top = region_h - inset
    scene = st.level == LEVEL_SCENE
    level_h, seg_w, seg_h = px(LEVEL_H), px(SEGMENT_W), px(SEGMENT_H)
    pad, gap = px(LEVEL_PAD), px(SEGMENT_GAP)
    x0 = cx(seg_w)
    inside = st.model_name or st.back_name or "a model"
    animating = st.animating
    for k, (widget_id, glyph, tip, here) in enumerate((
        (LEVEL_MODEL_ID, "LEVEL_MODEL", f"Model (Enter) — Edit the voxels of {inside}",
         st.level in (LEVEL_MODEL, LEVEL_BRICK) and not animating),
        (LEVEL_SCENE_ID, "LEVEL_SCENE", "Scene (Esc) — Work on whole models: move, stretch, cut, arrange",
         scene and not animating),
        (LEVEL_ANIM_ID, "LEVEL_ANIM", "Animate — Make the models move: motions, keys and the timeline",
         animating),
    )):
        widgets.append(
            Widget(
                id=widget_id,
                kind=SEGMENT_KIND,
                rect=(x0, y_top - pad - (k + 1) * seg_h - k * gap, seg_w, seg_h),
                glyph=glyph,
                tooltip=tip,
                active=here,
            )
        )
    y_top -= level_h + px(LEVEL_GAP)
    if animating:
        return _anim_rail(st, scale, rail_w, region_h, widgets, headers, y_top, px, cx)
    if scene:
        return _scene_rail(st, scale, rail_w, region_h, widgets, headers, y_top, px, cx)
    if st.level == LEVEL_BRICK:
        return _brick_rail(st, scale, rail_w, region_h, widgets, headers, y_top, px, cx)
    for widget_id, tool_name in TOOL_IDS.items():
        y = y_top - tool
        widgets.append(
            Widget(
                id=widget_id,
                kind=BUTTON_KIND,
                rect=(cx(tool), y, tool, tool),
                glyph=_TOOL_GLYPHS[widget_id],
                tooltip=_TOOL_TIPS[widget_id],
                active=st.tool == tool_name,
            )
        )
        y_top = y - tool_gap

    last_bottom = widgets[-1].rect[1]
    div_y = last_bottom - div_m - div_h
    divider = (px(8), div_y, max(0, rail_w - px(16)), div_h)

    y_top = div_y - div_m
    headers.append((MODEL_HEADER, (0, y_top - header_h, rail_w, header_h)))
    y_top -= header_h + header_gap
    y = y_top - chip_s
    widgets.append(
        Widget(
            id=MODEL_ID,
            kind=CHIP_KIND,
            rect=(cx(chip_s), y, chip_s, chip_s),
            glyph="MODEL",
            tooltip=_MODEL_TIP,
        )
    )

    y_top = y - section_gap
    if st.anim_menu:
        headers.append((ANIM_MENU_HEADER, (0, y_top - header_h, rail_w, header_h)))
        y_top -= header_h + header_gap
        y = y_top - chip_s
        widgets.append(
            Widget(id=ANIM_MENU_ID, kind=CHIP_KIND, rect=(cx(chip_s), y, chip_s, chip_s), glyph="LEVEL_ANIM",
                   tooltip=_ANIM_MENU_TIP)
        )
        y_top = y - section_gap
    headers.append((LAYERS_HEADER, (0, y_top - header_h, rail_w, header_h)))
    y_top -= header_h + header_gap
    y = y_top - chip_s
    widgets.append(
        Widget(
            id=LAYERS_ID,
            kind=CHIP_KIND,
            rect=(cx(chip_s), y, chip_s, chip_s),
            glyph="LAYERS",
            tooltip=_LAYERS_TIP,
        )
    )

    y = inset
    if st.mode == VOXEL:
        # The Size field, with a "Size" header over it like Mirror and Model.
        slider_h = px(size_box_height())
        margin = px(4)
        widgets.append(
            Widget(
                id=SIZE_ID,
                kind=SLIDER_KIND,
                rect=(margin, y, rail_w - 2 * margin, slider_h),
                label=str(int(st.brush_size)),
                tooltip=_SIZE_TIP,
            )
        )
        y += slider_h + header_gap
        headers.append((SIZE_HEADER, (0, y, rail_w, header_h)))
        if _shows_stroke(st):
            y += header_h + px(4)
            hit = px(STROKE_HIT)
            # Before the Size field in the list, so focus walks top to bottom and
            # the Size field stays the last stop.
            widgets.insert(
                len(widgets) - 1,
                Widget(
                    id=STROKE_ID,
                    kind=ICON_KIND,
                    rect=(cx(hit), y, hit, hit),
                    glyph="STROKE",
                    tooltip=_STROKE_TIP,
                    active=st.stroke_on,
                ),
            )
            y += hit + chip_gap
        else:
            y += header_h + chip_gap
    elif st.tool == SELECT and st.mode in COLOUR_MODES:
        # The colour tolerance, a field like Size's.
        slider_h = px(size_box_height())
        margin = px(4)
        widgets.append(
            Widget(
                id=TOLERANCE_ID,
                kind=SLIDER_KIND,
                rect=(margin, y, rail_w - 2 * margin, slider_h),
                label=str(int(st.colour_tolerance)),
                tooltip=_TOLERANCE_TIP,
            )
        )
        y += slider_h + header_gap
        headers.append((TOLERANCE_HEADER, (0, y, rail_w, header_h)))
        y += header_h + chip_gap
    if st.masked and st.level != LEVEL_SCENE:
        widgets.append(
            Widget(
                id=MASK_ID,
                kind=CHIP_KIND,
                rect=(cx(chip_s), y, chip_s, chip_s),
                glyph="MASK",
                tooltip=_MASK_TIP,
                active=True,
            )
        )
        y += chip_s + header_gap
        headers.append(("Masked", (0, y, rail_w, header_h)))
        y += header_h + chip_gap
    if st.tool == ATTACH and st.mode in (PATTERN, CLONE):
        space = st.pattern_space if st.pattern_space in STAMP_GLYPHS else "ADD"
        widgets.append(
            Widget(
                id=STAMP_ID,
                kind=CHIP_KIND,
                rect=(cx(chip_s), y, chip_s, chip_s),
                glyph=STAMP_GLYPHS[space],
                tooltip=_STAMP_TIPS[space],
                active=space != "ADD",
            )
        )
        y += chip_s + header_gap
        headers.append((STAMP_NAMES[space], (0, y, rail_w, header_h)))
        y += header_h + chip_gap
    if st.tool == PAINT and st.mode == GRADIENT:
        for widget_id, label, active in (
            (GRADIENT_STEPS_ID, str(int(st.gradient_steps)), False),
            (GRADIENT_END_ID, "", False),
            (GRADIENT_DITHER_ID, "Dither", bool(st.gradient_dither)),
        ):
            widgets.append(
                Widget(
                    id=widget_id,
                    kind=CHIP_KIND,
                    rect=(cx(chip_s), y, chip_s, chip_s),
                    label=label,
                    tooltip=_GRADIENT_TIPS[widget_id],
                    active=active,
                )
            )
            y += chip_s + chip_gap
        y += header_gap - chip_gap
        headers.append(("Gradient", (0, y, rail_w, header_h)))
        y += header_h + chip_gap
    if _shows_visible(st):
        widgets.append(
            Widget(
                id=VISIBLE_ID,
                kind=CHIP_KIND,
                rect=(cx(chip_s), y, chip_s, chip_s),
                glyph="SEEN",
                tooltip=_VISIBLE_TIP,
                active=bool(st.select_visible),
            )
        )
        y += chip_s + header_gap
        headers.append((VISIBLE_HEADER, (0, y, rail_w, header_h)))
        y += header_h + chip_gap

    plate = (0, 0, rail_w, region_h)
    return ChromeLayout(
        widgets=tuple(widgets),
        rect=plate,
        hit_rect=plate,
        scale=float(scale),
        divider=divider,
        headers=tuple(headers),
        state=st,
    )


def _scene_rail(st, scale, rail_w, region_h, widgets, headers, y_top, px, cx) -> ChromeLayout:
    """The scene's rail, under the level chip: the Pick and Cut tools, then
    New and Tools (buttons with a flyout) and Layers. The Model tools act on
    one model's voxels, so they stay on the rail inside a model."""
    tool, chip_s = px(TOOL_HIT), px(CHIP_HIT)
    header_h, header_gap, section_gap = px(HEADER_H), px(HEADER_GAP), px(SECTION_GAP)
    div_m, div_h = px(DIVIDER_MARGIN), max(1, px(DIVIDER_THICKNESS))
    y = y_top
    for widget_id, glyph, tip, name in ((SCENE_PICK_ID, "PICK", _PICK_TIP, "PICK"),
                                        (SCENE_CUT_ID, "SLICE", _CUT_TIP, "CUT")):
        y -= tool
        widgets.append(
            Widget(id=widget_id, kind=BUTTON_KIND, rect=(cx(tool), y, tool, tool), glyph=glyph,
                   tooltip=tip, active=st.scene_tool == name)
        )
        y -= px(TOOL_GAP)
    y += px(TOOL_GAP)
    div_y = y - div_m - div_h
    divider = (px(8), div_y, max(0, rail_w - px(16)), div_h)
    y_top = div_y - div_m
    for header, widget_id, glyph, tip in (
        (NEW_HEADER, NEW_ID, "CUBE_PLUS", _NEW_TIP),
        (TOOLS_HEADER, SCENE_TOOLS_ID, "MODEL", _TOOLS_TIP),
        *(((ANIM_MENU_HEADER, ANIM_MENU_ID, "LEVEL_ANIM", _ANIM_MENU_TIP),) if st.anim_menu else ()),
        (LAYERS_HEADER, LAYERS_ID, "LAYERS", _LAYERS_TIP),
    ):
        headers.append((header, (0, y_top - header_h, rail_w, header_h)))
        y_top -= header_h + header_gap
        y = y_top - chip_s
        widgets.append(
            Widget(id=widget_id, kind=CHIP_KIND, rect=(cx(chip_s), y, chip_s, chip_s), glyph=glyph, tooltip=tip)
        )
        y_top = y - section_gap
    plate = (0, 0, rail_w, region_h)
    return ChromeLayout(
        widgets=tuple(widgets), rect=plate, hit_rect=plate, scale=float(scale), divider=divider,
        headers=tuple(headers), state=st,
    )


def _brick_rail(st, scale, rail_w, region_h, widgets, headers, y_top, px, cx) -> ChromeLayout:
    """Brick mode's rail, under the level switch: the four brick tools,
    then the Convert chip that applies to the model, then Layers."""
    tool, chip_s = px(TOOL_HIT), px(CHIP_HIT)
    header_h, header_gap, section_gap = px(HEADER_H), px(HEADER_GAP), px(SECTION_GAP)
    div_m, div_h = px(DIVIDER_MARGIN), max(1, px(DIVIDER_THICKNESS))
    y = y_top
    for widget_id, name in BRICK_TOOL_IDS.items():
        y -= tool
        widgets.append(
            Widget(id=widget_id, kind=BUTTON_KIND, rect=(cx(tool), y, tool, tool), glyph=widget_id,
                   tooltip=_BRICK_TOOL_TIPS[widget_id], active=st.brick_tool == name)
        )
        y -= px(TOOL_GAP)
    y += px(TOOL_GAP)
    div_y = y - div_m - div_h
    divider = (px(8), div_y, max(0, rail_w - px(16)), div_h)
    y_top = div_y - div_m
    convert = ((BRICK_TO_VOXELS_ID, _BRICK_TO_VOXELS_TIP) if st.brick_model
               else (BRICK_CONVERT_ID, _BRICK_CONVERT_TIP))
    for header, widget_id, glyph, tip in (
        (CONVERT_HEADER, convert[0], convert[0], convert[1]),
        (LAYERS_HEADER, LAYERS_ID, "LAYERS", _LAYERS_TIP),
    ):
        headers.append((header, (0, y_top - header_h, rail_w, header_h)))
        y_top -= header_h + header_gap
        y = y_top - chip_s
        widgets.append(
            Widget(id=widget_id, kind=CHIP_KIND, rect=(cx(chip_s), y, chip_s, chip_s), glyph=glyph, tooltip=tip)
        )
        y_top = y - section_gap
    plate = (0, 0, rail_w, region_h)
    return ChromeLayout(
        widgets=tuple(widgets), rect=plate, hit_rect=plate, scale=float(scale), divider=divider,
        headers=tuple(headers), state=st,
    )


_ANIM_GROUPS = (
    (MOTION_HEADER, MOTION_ID, "MOTION", _MOTION_TIP),
    (LAYERS_HEADER, LAYERS_ID, "LAYERS", _LAYERS_TIP),
)
_ANIM_MENU_GROUP = (ANIM_MENU_HEADER, ANIM_MENU_ID, "LEVEL_ANIM", _ANIM_MENU_TIP)


def _anim_groups(st) -> tuple:
    """The Animate rail's groups: the Animation menu goes before Layers when it is on."""
    return _ANIM_GROUPS[:1] + ((_ANIM_MENU_GROUP,) if st.anim_menu else ()) + _ANIM_GROUPS[1:]
PAINT_HEADER = "Paint"
ANIM_PAINT_IDS = ("TOOL_ATTACH", "TOOL_ERASE", "TOOL_PAINT")  # on the Animate rail: paint the part's frame
_PAINT_TIPS = {
    "TOOL_ATTACH": "Attach — Add voxels to the picked part's frame at the play head (T)",
    "TOOL_ERASE": "Erase — Take voxels off the picked part's frame (R)",
    "TOOL_PAINT": "Paint — Recolour the picked part's frame (G)",
}


def _anim_rail(st, scale, rail_w, region_h, widgets, headers, y_top, px, cx) -> ChromeLayout:
    """Animate mode's rail, under the level switch: Motion (the preset
    gallery), Colour, Anchor and Layers. Nothing here edits voxels: a click in
    the view picks a part, and the gizmo on it writes keys."""
    chip_s = px(CHIP_HIT)
    header_h, header_gap, section_gap = px(HEADER_H), px(HEADER_GAP), px(SECTION_GAP)
    for header, widget_id, glyph, tip in _anim_groups(st):
        headers.append((header, (0, y_top - header_h, rail_w, header_h)))
        y_top -= header_h + header_gap
        y = y_top - chip_s
        widgets.append(
            Widget(
                id=widget_id, kind=CHIP_KIND, rect=(cx(chip_s), y, chip_s, chip_s), glyph=glyph, tooltip=tip,
                active=widget_id == ANIM_COLOUR_ID and st.anim_pick,
                enabled=widget_id != ANCHOR_ID or st.anchor_ok,
            )
        )
        y_top = y - section_gap
    # Paint (6.5): the three voxel tools, one under the other, for the picked part's frame.
    headers.append((PAINT_HEADER, (0, y_top - header_h, rail_w, header_h)))
    y_top -= header_h + header_gap
    for widget_id in ANIM_PAINT_IDS:
        y = y_top - chip_s
        widgets.append(
            Widget(
                id=widget_id, kind=CHIP_KIND, rect=(cx(chip_s), y, chip_s, chip_s), glyph=TOOL_IDS[widget_id],
                tooltip=_PAINT_TIPS[widget_id], active=TOOL_IDS[widget_id] == st.anim_paint,
            )
        )
        y_top = y - px(TOOL_GAP)
    y_top -= section_gap
    plate = (0, 0, rail_w, region_h)
    return ChromeLayout(
        widgets=tuple(widgets), rect=plate, hit_rect=plate, scale=float(scale),
        headers=tuple(headers), state=st,
    )


def _shows_stroke(st: ChromeState) -> bool:
    """The Stroke Designer's button: only for the tools a designed stroke shapes."""
    return st.mode == VOXEL and st.tool in (ATTACH, ERASE, PAINT)


def rail_height(st: ChromeState) -> float:
    """The height in points the rail needs for this state, top inset to bottom inset."""
    level = LEVEL_H + LEVEL_GAP
    if st.animating:
        # Motion, Layers, Paint (three chips).
        groups = len(_anim_groups(st)) * (HEADER_H + HEADER_GAP + CHIP_HIT + SECTION_GAP)
        paint = HEADER_H + HEADER_GAP + len(ANIM_PAINT_IDS) * (CHIP_HIT + TOOL_GAP)
        return float(RAIL_INSET + level + groups + paint + RAIL_INSET)
    if st.level == LEVEL_BRICK:
        # The four tools, then Convert and Layers.
        groups = 2 * (HEADER_H + HEADER_GAP + CHIP_HIT) + SECTION_GAP
        tools = len(BRICK_TOOL_IDS) * TOOL_HIT + (len(BRICK_TOOL_IDS) - 1) * TOOL_GAP
        return float(RAIL_INSET + level + tools + 2 * DIVIDER_MARGIN + DIVIDER_THICKNESS + groups + RAIL_INSET)
    if st.level == LEVEL_SCENE:
        # New, Tools, Layers.
        count = 3 + int(st.anim_menu)
        groups = count * (HEADER_H + HEADER_GAP + CHIP_HIT) + (count - 1) * SECTION_GAP
        tools = 2 * TOOL_HIT + TOOL_GAP  # Pick and Cut
        return float(RAIL_INSET + level + tools + 2 * DIVIDER_MARGIN + DIVIDER_THICKNESS + groups + RAIL_INSET)
    top = (
        RAIL_INSET
        + level
        + len(TOOL_IDS) * TOOL_HIT
        + len(TOOL_IDS) * TOOL_GAP
        + 2 * DIVIDER_MARGIN
        + DIVIDER_THICKNESS
        + HEADER_H + HEADER_GAP
        + CHIP_HIT  # the model button
        + SECTION_GAP + HEADER_H + HEADER_GAP
        + CHIP_HIT  # the layers button
        + (SECTION_GAP + HEADER_H + HEADER_GAP + CHIP_HIT if st.anim_menu else 0)  # the Animation button
    )
    bottom = RAIL_INSET
    if st.mode == VOXEL:
        bottom += size_box_height() + HEADER_GAP + HEADER_H + CHIP_GAP
        if _shows_stroke(st):
            bottom += 4 + STROKE_HIT
    if _shows_visible(st):
        bottom += CHIP_HIT + HEADER_GAP + HEADER_H + CHIP_GAP
    return float(top + bottom)


def _shows_visible(st: ChromeState) -> bool:
    """The Visible chip: with Select in a mode that takes through the model."""
    return st.tool == SELECT and st.mode in VISIBLE_MODES


def size_box_height() -> float:
    """The Size field's height in points."""
    return float(SIZE_BOX_H)


def size_from_drag(before: int, dx: float, scale: float) -> int:
    """The brush size after a drag of `dx` region pixels on the Size field:
    one step of BRUSH_SIZES every SIZE_DRAG_PX points, right to grow, left to
    shrink; clamped to the sizes."""
    steps = int(dx / (SIZE_DRAG_PX * scale))  # a whole step of drag before the value moves
    k = BRUSH_SIZES.index(before) if before in BRUSH_SIZES else 0
    return BRUSH_SIZES[max(0, min(len(BRUSH_SIZES) - 1, k + steps))]


def tolerance_from_drag(before: int, dx: float, scale: float) -> int:
    """The colour tolerance after a drag of `dx` region pixels: one step every
    TOLERANCE_DRAG_PX points, clamped to 0..100."""
    return max(0, min(100, int(before) + int(dx / (TOLERANCE_DRAG_PX * scale))))


def slider_stops(
    widget: Widget, stops: Sequence, scale: float, horizontal: bool
) -> tuple[tuple[float, float], ...]:
    """The centre (x, y) of every stop of a slider over `stops`, the first at
    the bottom (a vertical slider) or the left (a horizontal one). Spread
    over the widget's own rect, so a squeezed rail stays even."""
    x, y, w, h = widget.rect
    end = SIZE_END * scale
    last = max(1, len(stops) - 1)  # one stop sits at the start, no division by zero
    if horizontal:
        ty = float(round(y + h * SIZE_TRACK_X))
        span = max(0.0, w - 2 * end)
        return tuple((x + end + span * k / last, ty) for k in range(len(stops)))
    tx = float(round(x + w * SIZE_TRACK_X))
    span = max(0.0, h - 2 * end)
    return tuple((tx, y + end + span * k / last) for k in range(len(stops)))


def slider_at(widget: Widget, stops: Sequence, p: float, scale: float, horizontal: bool):
    """The stop nearest the region position `p` (a y for a vertical slider, an
    x for a horizontal one), clamped to the track."""
    pts = slider_stops(widget, stops, scale, horizontal)
    axis = 0 if horizontal else 1
    k = min(range(len(pts)), key=lambda i: abs(pts[i][axis] - p))
    return stops[k]




# --- the Motion gallery -----------------------------------------------------------
#
# The Motion gallery, the flyout of the Animate rail's Motion button: a card
# per preset in two groups, the part's motions and the colour's. The timeline
# dock shows the same cards while it has no track.
CARD_KIND = "CARD"
CAPTION_KIND = "CAPTION"  # a group's name over its cards; never hit
GALLERY_CARD_W = 96
GALLERY_CARD_H = 84
GALLERY_COLS = 4
GALLERY_PAD = 8
PRESET_IDS = tuple(f"PRESET_{name}" for name in anim.PRESETS)
PRESET_OF_ID = dict(zip(PRESET_IDS, anim.PRESETS))
CAPTION_IDS = tuple(f"CAPTION_{kind}" for kind in (anim.PART, anim.COLOUR))
GROUP_CAPTIONS = {anim.PART: "Move the part", anim.COLOUR: "Animate a colour"}
PRESET_BLURBS = {
    "SPIN": "Turns round and round, like a radar dish",
    "SWING": "Rocks over and back, like a wiper",
    "BOB": "Floats up and down",
    "SHAKE": "Jitters in place",
    "GLOW": "The colour brightens and dims",
    "BLINK": "The colour goes dark for a moment, like an eye",
    "FLOW": "The colours run along, like chaser lights",
}


def preset_tip(name: str) -> str:
    """ "Spin — Turns round and round, like a radar dish" """
    return f"{name.title()} — {PRESET_BLURBS.get(name, 'A ready-made motion')}"


def _motion_gallery(layout: ChromeLayout) -> tuple[Widget, ...]:
    """The preset gallery right of the Motion button: a caption over each
    group, then its cards, GALLERY_COLS to a row. It starts level with the
    button and moves up as far as it must to stay inside the view."""
    opener = layout.by_id(MOTION_ID)
    if opener is None:
        return ()
    from .anim import PART_PRESETS

    scale = layout.scale

    def px(points: float) -> int:
        return int(round(points * scale))

    cw, ch, gap, pad = px(GALLERY_CARD_W), px(GALLERY_CARD_H), px(GAP) * 2, px(GALLERY_PAD)
    cap_h = px(HEADER_H + 6)
    # Phase 1 of the animation plan: the part motions. Colour motions come with Phase 3.
    groups = [(kind, names) for kind, names in ((anim.PART, PART_PRESETS),) if names]
    height = sum(cap_h + gap + (-(-len(n) // GALLERY_COLS)) * (ch + gap) for _k, n in groups) - gap + 2 * pad
    x = layout.rect[0] + layout.rect[2] + px(FLYOUT_GAP) + pad
    top = opener.rect[1] + opener.rect[3] - pad
    top = max(top, px(RAIL_INSET) + height - 2 * pad)  # up, so the last card stays inside
    out: list[Widget] = []
    y = top
    for kind, names in groups:
        y -= cap_h
        width = GALLERY_COLS * cw + (GALLERY_COLS - 1) * gap
        out.append(Widget(id=f"CAPTION_{kind}", kind=CAPTION_KIND, rect=(x, y, width, cap_h),
                          label=GROUP_CAPTIONS[kind], enabled=False))
        y -= gap
        for k, name in enumerate(names):
            col, row = k % GALLERY_COLS, k // GALLERY_COLS
            out.append(Widget(
                id=f"PRESET_{name}", kind=CARD_KIND,
                rect=(x + col * (cw + gap), y - (row + 1) * ch - row * gap, cw, ch),
                label=name.title(), tooltip=preset_tip(name),
            ))
        y -= (-(-len(names) // GALLERY_COLS)) * (ch + gap)
    return tuple(out)


def flyout_widgets(layout: ChromeLayout, widget_id: str | None) -> tuple[Widget, ...]:
    """The row a hovered chip or button opens to its right, or nothing.

    A tool button opens the modes it works in, Attach's ending in the
    Primitives item; the Model button opens the whole-model tools. The row
    sits to the right of the opener, vertically centred on it. An open
    Primitives item (PRIMITIVES_ID) is Attach's row plus the grid of
    primitives under the item.
    """
    if widget_id == PRIMITIVES_ID:
        row = flyout_widgets(layout, "TOOL_ATTACH")
        return row + _primitive_grid(layout, row[-1]) if row else ()
    if widget_id in MODEL_CHOICES:
        row = flyout_widgets(layout, CHOICE_PARENT[widget_id])
        item = next((w for w in row if w.id == widget_id), None)
        return row + _choice_grid(layout, item, min(w.rect[1] for w in row)) if item else ()
    if widget_id == LAYERS_ID:
        return _layer_list(layout)
    if widget_id == MOTION_ID:
        return _motion_gallery(layout)
    if widget_id not in FLYOUT_OPENERS:
        return ()
    chip = layout.by_id(widget_id)
    if chip is None:
        return ()
    scale = layout.scale
    side = int(round(TICK_HIT * scale))
    gap = int(round(GAP * scale))
    _x0, y0, _w, h = chip.rect
    # Past the rail hairline, not just past the chip: the chip sits inset in the plate.
    x = layout.rect[0] + layout.rect[2] + int(round(FLYOUT_GAP * scale))
    y = y0 + (h - side) // 2
    out: list[Widget] = []
    if widget_id == SCENE_TOOLS_ID:
        items = [(ident, tip, glyph, False) for ident, glyph, tip in SCENE_TOOL_ITEMS]
    elif widget_id == NEW_ID:
        items = [(ident, _NEW_TIPS[ident], _NEW_GLYPHS[ident], False) for ident in NEW_FLYOUT_IDS]
    elif widget_id == ANIM_MENU_ID:
        names: dict[str, str] = {}
        for c in COMMANDS:
            names.setdefault(c.id, c.name)
        items = [
            (fid, f"{names.get(cid, cid).rstrip('.')} — {tip}", glyph, False)
            for fid, (cid, glyph, tip, _where) in zip(ANIM_MENU_FLYOUT_IDS, ANIM_MENU_ITEMS)
        ]
    elif widget_id == MODEL_ID:
        names: dict[str, str] = {}
        for c in COMMANDS:
            names.setdefault(c.id, c.name.split(":")[0])  # "Mirror Half: +X onto −X" -> "Mirror Half"
        shown = set(layout.state.model_items)
        items = [
            (fid, f"{names.get(cid, cid).rstrip('.')} — {MODEL_TIPS[cid]}", MODEL_GLYPHS[cid], False)
            for fid, cid in COMMAND_OF_MODEL_ID.items()
            if cid in shown
        ]
    else:
        items = []
        for ident in TOOL_FLYOUT_IDS[widget_id]:
            tool, mode = TOOL_MODE_OF_ID[ident]
            active = layout.state.tool == tool and layout.state.mode == mode
            if tool == ATTACH and mode == BOX:
                active = active and layout.state.solid == BOX_SOLID  # a primitive lights its own item
            items.append((ident, f"{MODE_NAMES[mode]} — {MODE_TIPS[mode]}", RING_GLYPHS[mode], active))
        if widget_id == "TOOL_ATTACH":
            items.append(
                (
                    PRIMITIVES_ID,
                    _PRIMITIVES_TIP,
                    "PRIMITIVES",  # its own glyph; the grid marks the chosen shape
                    layout.state.tool == ATTACH and layout.state.mode in (BOX, CENTER) and layout.state.solid != BOX_SOLID,
                )
            )
    # The Model flyout is a block, about as many rows as columns; the others are one row. A block that would run out of the
    # view's bottom moves up.
    columns = max(1, math.ceil(math.sqrt(len(items)))) if widget_id in (MODEL_ID, ANIM_MENU_ID) else max(1, len(items))
    rows = -(-len(items) // columns) if items else 0
    low = max(int(round(RAIL_INSET * scale)), int(layout.state.readout_top) if widget_id in (MODEL_ID, ANIM_MENU_ID) else 0)
    y = max(y, low + (rows - 1) * (side + gap))
    for k, (ident, tooltip, glyph, active) in enumerate(items):
        row, col = divmod(k, columns)
        out.append(
            Widget(
                id=ident,
                kind=BUTTON_KIND,
                rect=(x + col * (side + gap), y - row * (side + gap), side, side),
                glyph=glyph,
                tooltip=tooltip,
                active=active,
                enabled=_anim_item_enabled(layout.state, ident),
            )
        )
    if widget_id == NEW_ID and out:
        out.extend(_primitive_grid(layout, out[0], NEW_SOLID_OF_ID))
    return tuple(out)


def _anim_item_enabled(st: ChromeState, ident: str) -> bool:
    """An Animation menu item works in one place: building (Make Part) or Animate mode (Frame: Add)."""
    if ident not in COMMAND_OF_ANIM_ID:
        return True
    where = next(w for fid, (_c, _g, _t, w) in zip(ANIM_MENU_FLYOUT_IDS, ANIM_MENU_ITEMS) if fid == ident)
    return st.animating == (where == "ANIMATE")


def layer_display_name(layer_id: int, name: str) -> str:
    """A .vox file often names its layers "0" to "7": those read "Layer 3"."""
    return name if name and name != str(layer_id) else f"Layer {layer_id}"


def layer_part(widget_id: str | None) -> tuple[str, int] | None:
    """(LAYER_EYE, LAYER_ROW, LAYER_MOVE or LAYER_MODEL, layer id or model uid)
    of a Layers list id, or None."""
    if not widget_id:
        return None
    for prefix in (LAYER_EYE, LAYER_ROW, LAYER_MOVE, LAYER_MODEL, LAYER_LINK, LIVE_EYE, LIVE_ROW):
        if widget_id.startswith(prefix):
            try:
                return prefix, int(widget_id[len(prefix):])
            except ValueError:
                return None
    return None


def is_flyout_item(widget_id: str | None) -> bool:
    """True for an item inside an open flyout (the mouse stays in the flyout)."""
    return (
        widget_id in ALL_FLYOUT_IDS
        or widget_id in PRESET_OF_ID
        or widget_id in CAPTION_IDS
        or layer_part(widget_id) is not None
    )


def _layer_list(layout: ChromeLayout) -> tuple[Widget, ...]:
    """The Layers flyout, right of the Layers button: per layer a header row
    (the eye, the name, the move arrow), then one row per model on it,
    indented. The model being edited is the active row. The list starts
    level with the button and moves up as far as it must to stay inside the
    view."""
    opener = layout.by_id(LAYERS_ID)
    if opener is None or not layout.state.layers:
        return ()
    scale = layout.scale
    row_h = int(round(LIST_ROW * scale))
    icon = int(round(LIST_ICON * scale))
    width = int(round(LIST_WIDTH * scale))
    live_of: dict[int, list] = {}
    for uid, live_id, live_name, on, error in layout.state.live:
        live_of.setdefault(uid, []).append((live_id, live_name, on, error))
    rows = sum(1 + len(models) + sum(len(live_of.get(uid, ())) for uid, _n in models)
               for _i, _n, _h, models in layout.state.layers)
    x = layout.rect[0] + layout.rect[2] + int(round(FLYOUT_GAP * scale))
    top = opener.rect[1] + opener.rect[3]
    top = max(top, int(round(RAIL_INSET * scale)) + rows * row_h)  # up, so the last row stays inside
    top = min(top, layout.rect[1] + layout.rect[3])
    out: list[Widget] = []
    linked = dict(layout.state.linked)
    k = 0
    for layer_id, name, hidden, models in layout.state.layers:
        y = top - (k + 1) * row_h
        k += 1
        shown = layer_display_name(layer_id, name)
        here = any(uid == layout.state.model_uid for uid, _n in models)
        out.append(Widget(
            id=f"{LAYER_EYE}{layer_id}", kind=LIST_KIND, rect=(x, y, icon, row_h), glyph="SEEN",
            tooltip=f"{'Show' if hidden else 'Hide'} {shown} — {'Show' if hidden else 'Hide'} every model in it",
            active=not hidden, dim=hidden,
        ))
        out.append(Widget(
            id=f"{LAYER_ROW}{layer_id}", kind=LIST_KIND, rect=(x + icon, y, width - 2 * icon, row_h),
            label=shown + (" (hidden)" if hidden else ""), glyph="GROUP" if models else "",
            tooltip=f"{shown} — Click: frame its models; the eye hides them",
            dim=hidden or not models,
        ))
        out.append(Widget(
            id=f"{LAYER_MOVE}{layer_id}", kind=LIST_KIND, rect=(x + width - icon, y, icon, row_h),
            glyph="MOVE", tooltip=f"Move {layout.state.model_name} here — Put it into {shown}",
            enabled=bool(layout.state.model_name) and not here,
        ))
        for uid, model_name in models:
            y = top - (k + 1) * row_h
            k += 1
            edited = uid == layout.state.model_uid
            out.append(Widget(
                id=f"{LAYER_MODEL}{uid}", kind=LIST_KIND, rect=(x + icon, y, width - 2 * icon, row_h),
                label=model_name, active=edited, dim=hidden,
                tooltip=f"{model_name} — {'The model you edit' if edited else 'Click: edit this model'}",
            ))
            if uid in linked:
                out.append(Widget(
                    id=f"{LAYER_LINK}{uid}", kind=LIST_KIND, rect=(x + width - icon, y, icon, row_h),
                    glyph="LINK", dim=hidden, tooltip=link_tip(linked[uid]),
                ))
            for live_id, live_name, on, error in live_of.get(uid, ()):
                y = top - (k + 1) * row_h
                k += 1
                out.append(Widget(
                    id=f"{LIVE_EYE}{live_id}", kind=LIST_KIND, rect=(x + icon, y, icon, row_h), glyph="SEEN",
                    tooltip=f"{'Switch on' if not on else 'Switch off'} {live_name} — A live shader; off, its voxels go",
                    active=on, dim=hidden,
                ))
                out.append(Widget(
                    id=f"{LIVE_ROW}{live_id}", kind=LIST_KIND, rect=(x + 2 * icon, y, width - 3 * icon, row_h),
                    label=live_name, dim=hidden or not on or bool(error),
                    tooltip=f"{live_name} — {error or 'A live shader: it runs again after each edit'}. "
                            "Click: change, bake or remove it",
                ))
    return tuple(out)


def _primitive_grid(layout: ChromeLayout, item: Widget, ids=None) -> tuple[Widget, ...]:
    """The primitives, PRIMITIVE_COLUMNS across, under the Primitives item:
    the grid's left edge under the item's, its top one gap below it. `ids`
    NEW_SOLID_OF_ID: New's grid, where a shape makes a model and none is lit."""
    x0, y0, side, _h = item.rect
    gap = int(round(GAP * layout.scale))
    making = ids is not None
    ids = SOLID_OF_ID if ids is None else ids
    out = []
    for k, (ident, solid) in enumerate(ids.items()):
        row, col = divmod(k, PRIMITIVE_COLUMNS)
        out.append(
            Widget(
                id=ident,
                kind=BUTTON_KIND,
                rect=(x0 + col * (side + gap), y0 - (row + 1) * (side + gap), side, side),
                glyph=_GLYPH_OF_SOLID[solid],
                tooltip=_NEW_SOLID_TIPS[solid] if making else _SOLID_TIPS[solid],
                active=not making and layout.state.tool == ATTACH and layout.state.mode in (BOX, CENTER) and layout.state.solid == solid,
            )
        )
    return tuple(out)


def _choice_grid(layout: ChromeLayout, item: Widget, below: float | None = None) -> tuple[Widget, ...]:
    """A tool's choices under its item, as the primitives hang under theirs.
    `below` is the lowest row of a flyout of several rows: the choices hang
    under the whole block, in the item's column."""
    x0, y0, side, _h = item.rect
    if below is not None:
        y0 = min(y0, int(below))
    gap = int(round(GAP * layout.scale))
    tool = split_tooltip(item.tooltip)[0]
    columns = CHOICE_COLUMNS[item.id]
    out = []
    for k, (ident, choice) in enumerate(MODEL_CHOICES[item.id].items()):
        row, col = divmod(k, columns)
        out.append(
            Widget(
                id=ident,
                kind=TICK_KIND,
                rect=(x0 + col * (side + gap), y0 - (row + 1) * (side + gap), side, side),
                label=choice.label,
                tooltip=f"{tool} {choice.name} — {choice.tip}",
            )
        )
    return tuple(out)


def split_tooltip(text: str) -> tuple[str, str]:
    """ "Name — what it does" -> ("Name", "what it does"). No dash: (text, "")."""
    name, sep, rest = text.partition(" — ")
    return (name.strip(), rest.strip()) if sep else (text.strip(), "")


def hit_test(
    layout: ChromeLayout, px: float, py: float, open_id: str | None = None
) -> Widget | None:
    """The widget under the region pixel, the open flyout's first.

    A widget that is not enabled is not hit: a press falls through to the
    strip, which swallows it.
    """
    for widget in flyout_widgets(layout, open_id):
        if widget.enabled and in_rect(widget.rect, px, py):
            return widget
    for widget in layout.widgets:
        if widget.enabled and in_rect(widget.rect, px, py):
            return widget
    return None


def flyout_bridge(layout: ChromeLayout, open_id: str | None) -> Rect | None:
    """The rectangle from the opener across the gap to the end of its flyout,
    padded a little. While the flyout is open a mouse in here counts as inside,
    so the row does not close on the way from the button to its items."""
    if open_id == PRIMITIVES_ID:
        open_id = "TOOL_ATTACH"  # the row's bridge; the grid has its own (primitive_bridge)
    elif open_id in MODEL_CHOICES:
        open_id = CHOICE_PARENT[open_id]  # likewise for a tool's choices
    row = flyout_widgets(layout, open_id)
    opener = layout.by_id(open_id) if open_id else None
    if not row or opener is None:
        return None
    pad = int(round(FLYOUT_GAP * layout.scale)) + 2
    rects = (opener.rect,) + tuple(w.rect for w in row)
    x0 = min(r[0] for r in rects) - pad
    y0 = min(r[1] for r in rects) - pad
    x1 = max(r[0] + r[2] for r in rects) + pad
    y1 = max(r[1] + r[3] for r in rects) + pad
    return (x0, y0, x1 - x0, y1 - y0)


def contains(layout: ChromeLayout, px: float, py: float, open_id: str | None = None) -> bool:
    """True inside the rail, inside the open flyout, or on the way between the
    rail and its flyout.

    The press handler swallows everything in here, so a click beside a button
    does not reach the voxels behind the rail. The rest of the viewport is not
    in it: the navigation gizmo in the top right keeps its clicks.
    """
    if in_rect(layout.hit_rect, px, py):
        return True
    bridge = flyout_bridge(layout, open_id)
    if bridge is not None and in_rect(bridge, px, py):
        return True
    grid = primitive_bridge(layout, open_id)
    if grid is not None and in_rect(grid, px, py):
        return True
    return any(in_rect(w.rect, px, py) for w in flyout_widgets(layout, open_id))


def primitive_bridge(layout: ChromeLayout, open_id: str | None) -> Rect | None:
    """The open grid of primitives (or of a Model tool's choices) and the gap
    up to its item, padded: the mouse on its way into the grid stays inside.
    Only the grid's own columns, so the space under the rest of the row keeps
    its clicks."""
    grid_ids = SOLID_OF_ID if open_id == PRIMITIVES_ID else MODEL_CHOICES.get(open_id)
    if grid_ids is None:
        return None
    widgets = flyout_widgets(layout, open_id)
    grid = [w for w in widgets if w.id in grid_ids]
    item = next((w for w in widgets if w.id == open_id), None)
    if not grid or item is None:
        return None
    pad = int(round(FLYOUT_GAP * layout.scale)) + 2
    rects = [item.rect] + [w.rect for w in grid]
    x0 = min(r[0] for r in rects) - pad
    y0 = min(r[1] for r in rects) - pad
    x1 = max(r[0] + r[2] for r in rects) + pad
    y1 = max(r[1] + r[3] for r in rects) + pad
    return (x0, y0, x1 - x0, y1 - y0)


def focus_ids(layout: ChromeLayout) -> tuple[str, ...]:
    """The widgets the keyboard focus walks, top to bottom."""
    return tuple(w.id for w in layout.widgets if w.enabled)


def focus_step(layout: ChromeLayout, current: str | None, step: int) -> str | None:
    """The next (step 1) or previous (step -1) focusable widget, wrapping.

    From nothing the focus lands on the first widget going right and on the
    last going left. A widget that went away keeps the focus at its place.
    """
    ids = focus_ids(layout)
    if not ids:
        return None
    if current not in ids:
        return ids[0] if step >= 0 else ids[-1]
    return ids[(ids.index(current) + step) % len(ids)]


def name_list(names, most: int = 3) -> str:
    """'A', 'A and B', 'A, B and C', 'A, B, C and 4 more'."""
    names = list(names)
    if len(names) > most:
        return ", ".join(names[:most]) + f" and {len(names) - most} more"
    if len(names) > 1:
        return ", ".join(names[:-1]) + " and " + names[-1]
    return names[0] if names else ""


def link_tip(others) -> str:
    """The tooltip of a linked copy's chain: which models share its shape."""
    both = "both" if len(others) == 1 else "all of them"
    return f"Linked copy of {name_list(others)}: an edit to one changes {both}. Click: make it unique"


def tooltip_due(hover_started: float | None, now: float, delay: float = TOOLTIP_DELAY_S) -> bool:
    """True once the mouse has rested on a widget for `delay` seconds."""
    if hover_started is None:
        return False
    return (now - hover_started) >= delay


def tooltip_rect(widget: Widget, layout: ChromeLayout, region_width: int) -> Rect:
    """Where a tooltip for `widget` goes: to its right, pushed inside the region;
    to its left when the right has no room (the mirror buttons at the right edge)."""
    scale = layout.scale
    width = int(round((len(widget.tooltip) * CHIP_CHAR + 2 * CHIP_PAD) * scale))
    height = int(round(CHIP_HEIGHT * scale))
    x, y, w, h = widget.rect
    gap = int(round(FLYOUT_GAP * scale))
    tx = max(x + w, layout.rect[0] + layout.rect[2]) + gap
    if tx + width > int(region_width) and x - gap - width >= 0:
        tx = x - gap - width
    ty = y + (h - height) // 2
    tx = min(max(0, tx), max(0, int(region_width) - width))
    region_h = layout.rect[3]
    ty = min(max(0, ty), max(0, int(region_h) - height))
    return (tx, ty, width, height)


# --- glyphs -------------------------------------------------------------------
#
# One-pixel line drawings in the unit square, as LINES pairs. `glyph_lines`
# puts one into a rectangle. No filled blobs, as the proposal asks.


def _pairs(*points: tuple[float, float]) -> np.ndarray:
    return np.array(points, dtype=np.float32).reshape(-1, 2)


def _open_path(points: np.ndarray) -> np.ndarray:
    """An open polyline (N, 2) as LINES pairs (2 N - 2, 2)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return np.stack([pts[:-1], pts[1:]], axis=1).reshape(-1, 2)


def _ellipse(cx: float, cy: float, rx: float, ry: float, steps: int = 12) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1).astype(np.float32)


def _arc(cx, cy, rx, ry, start, end, steps: int = 8) -> np.ndarray:
    """A piece of the ellipse, as a polyline of `steps` points."""
    a = np.linspace(start, end, steps)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1).astype(np.float32)


_HEX = np.array(
    [[0.5, 1.0], [1.0, 0.75], [1.0, 0.25], [0.5, 0.0], [0.0, 0.25], [0.0, 0.75]],
    dtype=np.float32,
)
# The three inner edges of an isometric cube: the centre out to three corners.
_CUBE_INNER = _pairs(
    (0.5, 0.5), (0.5, 0.0),
    (0.5, 0.5), (0.0, 0.75),
    (0.5, 0.5), (1.0, 0.75),
)


def _cube(scale_xy: float = 1.0, dy: float = 0.0) -> np.ndarray:
    """The isometric cube, optionally shrunk about its centre and moved."""
    def fit(points: np.ndarray) -> np.ndarray:
        out = (np.asarray(points, dtype=np.float32) - 0.5) * scale_xy + 0.5
        out[:, 1] += dy
        return out

    return np.concatenate([loop_lines(fit(_HEX)), fit(_CUBE_INNER)], axis=0)


def _plus(cx, cy, r) -> np.ndarray:
    return _pairs((cx - r, cy), (cx + r, cy), (cx, cy - r), (cx, cy + r))


def _minus(cx, cy, r) -> np.ndarray:
    return _pairs((cx - r, cy), (cx + r, cy))


def _marquee() -> np.ndarray:
    """Four corner brackets: the dashed selection frame."""
    lo, hi, arm = 0.05, 0.95, 0.3
    out = []
    for x, sx in ((lo, 1.0), (hi, -1.0)):
        for y, sy in ((lo, 1.0), (hi, -1.0)):
            out.append(_pairs((x, y), (x + sx * arm, y)))
            out.append(_pairs((x, y), (x, y + sy * arm)))
    return np.concatenate(out, axis=0)


def _brush() -> np.ndarray:
    """A handle running to the bottom left, with a ferrule and a tip."""
    return np.concatenate(
        [
            _pairs((0.95, 0.95), (0.45, 0.45)),
            _pairs((0.55, 0.3), (0.3, 0.55)),  # the ferrule, across the handle
            loop_lines(_pairs((0.05, 0.05), (0.42, 0.22), (0.22, 0.42))),  # the tip
        ],
        axis=0,
    )


def _cylinder() -> np.ndarray:
    return np.concatenate(
        [
            loop_lines(_ellipse(0.5, 0.82, 0.38, 0.14)),
            _open_path(_arc(0.5, 0.22, 0.38, 0.14, np.pi, 2.0 * np.pi)),
            _pairs((0.12, 0.82), (0.12, 0.22), (0.88, 0.82), (0.88, 0.22)),
        ],
        axis=0,
    )


def _cone() -> np.ndarray:
    return np.concatenate(
        [
            loop_lines(_ellipse(0.5, 0.22, 0.38, 0.14)),
            _pairs((0.12, 0.22), (0.5, 0.95), (0.88, 0.22), (0.5, 0.95)),
        ],
        axis=0,
    )


def _stamp() -> np.ndarray:
    """The pattern stamp: four small squares."""
    out = []
    for cx in (0.3, 0.7):
        for cy in (0.3, 0.7):
            out.append(loop_lines(_pairs(
                (cx - 0.18, cy - 0.18), (cx + 0.18, cy - 0.18),
                (cx + 0.18, cy + 0.18), (cx - 0.18, cy + 0.18),
            )))
    return np.concatenate(out, axis=0)


def _stroke() -> np.ndarray:
    """A brush stroke seen from above: a round, thick start that bends and
    thins to a point, as a designed taper draws it."""
    top = _open_path(np.array([[0.16, 0.66], [0.34, 0.76], [0.54, 0.64], [0.72, 0.46], [0.92, 0.34]],
                              dtype=np.float32))
    bot = _open_path(np.array([[0.16, 0.26], [0.34, 0.38], [0.52, 0.38], [0.72, 0.34], [0.92, 0.34]],
                              dtype=np.float32))
    cap = _open_path(np.array([[0.16, 0.66], [0.09, 0.58], [0.07, 0.46], [0.09, 0.34], [0.16, 0.26]],
                              dtype=np.float32))
    return np.concatenate([top, bot, cap], axis=0)


def _turn(tip_over: bool = False) -> np.ndarray:
    """A circular arrow. `tip_over` turns it a quarter, for Tip."""
    arc = _open_path(_arc(0.5, 0.5, 0.35, 0.35, 0.35 * np.pi, 1.9 * np.pi))
    head = np.concatenate(
        [
            _pairs((0.5 + 0.35 * np.cos(0.35 * np.pi), 0.5 + 0.35 * np.sin(0.35 * np.pi)),
                   (0.62, 0.95)),
            _pairs((0.5 + 0.35 * np.cos(0.35 * np.pi), 0.5 + 0.35 * np.sin(0.35 * np.pi)),
                   (0.95, 0.72)),
        ],
        axis=0,
    )
    out = np.concatenate([arc, head], axis=0)
    if tip_over:
        turned = out - 0.5
        out = np.stack([turned[:, 1], -turned[:, 0]], axis=1) + 0.5
    return out.astype(np.float32)


_GLYPHS: dict[str, np.ndarray] = {
    "CUBE_PLUS": np.concatenate([_cube(0.75, -0.08), _plus(0.82, 0.82, 0.16)], axis=0),
    "CUBE_MINUS": np.concatenate([_cube(0.75, -0.08), _minus(0.82, 0.82, 0.16)], axis=0),
    "BRUSH": _brush(),
    "MARQUEE": _marquee(),
    "CUT": _pairs((0.1, 0.16), (0.9, 0.16)),
    "BOX": loop_lines(_pairs((0.15, 0.12), (0.85, 0.12), (0.85, 0.82), (0.15, 0.82))),
    "CYLINDER": _cylinder(),
    "PYRAMID": _pairs((0.12, 0.15), (0.88, 0.15), (0.12, 0.15), (0.5, 0.9), (0.88, 0.15), (0.5, 0.9)),
    "CONE": _cone(),
    "PATTERN": _stamp(),
    "TURN": _turn(False),
    "TIP": _turn(True),
    "STROKE": _stroke(),
}

GLYPH_NAMES = tuple(sorted(_GLYPHS))


def glyph_lines(name: str, rect: Rect, inset: float = 0.0) -> np.ndarray:
    """The glyph as LINES pairs in region pixels, inside `rect` minus `inset`.

    An unknown name draws nothing, so a widget without a glyph is no error.
    Coordinates land on half pixels, which is where a one-pixel line is sharp.
    """
    unit = _GLYPHS.get(name)
    if unit is None:
        return np.zeros((0, 2), dtype=np.float32)
    return fit_glyph(unit, rect, inset)


def rect_lines(rect: Rect, inset: float = 0.0) -> np.ndarray:
    """The outline of a rectangle as LINES pairs, on half pixels."""
    x, y, w, h = rect
    x0 = x + inset + 0.5
    y0 = y + inset + 0.5
    x1 = x + w - inset - 0.5
    y1 = y + h - inset - 0.5
    return loop_lines(_pairs((x0, y0), (x1, y0), (x1, y1), (x0, y1))).astype(np.float32)


def edge_line(rect: Rect) -> np.ndarray:
    """The bottom edge of a rectangle as one LINES pair, on a half pixel."""
    x, y, w, _h = rect
    return np.array([[x, y + 0.5], [x + w, y + 0.5]], dtype=np.float32)


def right_edge_line(rect: Rect) -> np.ndarray:
    """The right edge of a rectangle as one LINES pair, on a half pixel."""
    x, y, w, h = rect
    return np.array([[x + w - 0.5, y], [x + w - 0.5, y + h]], dtype=np.float32)


# --- glyphs, tips and status texts of the tools and modes --------------------------
# (the rail, the star and the status bar use them)


RING_GLYPHS = {
    VOXEL: "VOXEL",
    FACE: "FACE",
    BOX: "MODE_BOX",
    LINE: "LINE",
    CENTER: "CENTER",
    RECT: "RECT",
    LASSO: "LASSO",
    PATTERN: "MODE_PATTERN",
    CLONE: "MODE_CLONE",
    GRADIENT: "MODE_GRADIENT",
    CONNECTED: "CONNECTED",
    CONNECTED_CORNERS: "CONNECTED_CORNERS",
    BY_COLOUR: "BY_COLOUR",
    COLOUR_CONNECTED: "COLOUR_CONNECTED",
    COLOUR_CONNECTED_CORNERS: "COLOUR_CONNECTED_CORNERS",
}
RING_TOOL_GLYPHS = {
    ATTACH: "CUBE_PLUS",
    ERASE: "CUBE_MINUS",
    PAINT: "BRUSH",
    SELECT: "MARQUEE",
}
MODE_TIPS = {
    VOXEL: "One block at a time (V)",
    FACE: "Whole flat face of the same color; drag to push it out (F)",
    BOX: "Drag a rectangle, then set how tall it is (B)",
    LINE: "Drag a straight row of blocks (L)",
    CENTER: "Drag a rectangle that grows from the middle (C)",
    RECT: "Draw a rectangle on screen to select what’s inside (M)",
    LASSO: "Draw a free loop on screen to select what’s inside (O)",
    PATTERN: "Stamp your copied cluster onto the surface (P)",
    CLONE: "Stamp your copied cluster with its own colours",
    GRADIENT: "Drag across the model: a ramp from the current colour to the end colour",
    CONNECTED: "Everything joined to the block you click, face to face",
    CONNECTED_CORNERS: "Everything joined to the block you click, corners and edges too",
    BY_COLOUR: "Every block of the colour you click, all over the model",
    COLOUR_CONNECTED: "The colour you click, where it touches the block face to face",
    COLOUR_CONNECTED_CORNERS: "The colour you click, where it touches the block, corners and edges too",
}
TOOL_RING_TIPS = {
    ATTACH: "Attach blocks (T)",
    ERASE: "Erase blocks (R)",
    PAINT: "Paint blocks (G)",
    SELECT: "Select blocks (Z)",
}
MODE_STATUS = {
    VOXEL: "Mode: one block",
    FACE: "Mode: whole face",
    BOX: "Mode: box from a drag",
    LINE: "Mode: straight line",
    CENTER: "Mode: grow from center",
    RECT: "Mode: screen rectangle",
    LASSO: "Mode: freehand loop",
    PATTERN: "Mode: pattern stamp",
    CLONE: "Mode: clone stamp, the copied colours",
    GRADIENT: "Mode: gradient",
    CONNECTED: "Mode: everything joined by a face",
    CONNECTED_CORNERS: "Mode: everything joined, corners too",
    BY_COLOUR: "Mode: every block of one colour",
    COLOUR_CONNECTED: "Mode: one colour, joined by a face",
    COLOUR_CONNECTED_CORNERS: "Mode: one colour, joined by corners too",
}
IDLE_STATUS = {
    ATTACH: "Click or drag to place blocks · ⌥: erase · Ctrl, ⌘, ⌥ + move: orbit, pan, zoom · Space: star",
    ERASE: "Click or drag to remove blocks · Ctrl, ⌘, ⌥ + move: orbit, pan, zoom",
    PAINT: "Click or drag to paint with the current color · ⌥: erase · Ctrl, ⌘, ⌥ + move: orbit, pan, zoom",
    SELECT: "Click or drag to choose blocks · drag a selected block to move them · ⌥ or right-click: deselect · Ctrl, ⌘, ⌥ + move: orbit, pan, zoom",
}
COMMAND_STATUS = "Type to find an action"

def circle_points(cx: float, cy: float, radius: float, steps: int = 48) -> np.ndarray:
    """A closed ring of `steps` points, (steps, 2) float32."""
    a = np.linspace(0.0, 2.0 * np.pi, int(steps), endpoint=False, dtype=np.float32)
    return np.stack(
        [np.float32(cx) + np.float32(radius) * np.cos(a), np.float32(cy) + np.float32(radius) * np.sin(a)],
        axis=1,
    ).astype(np.float32)


def circle_tris(cx: float, cy: float, radius: float, steps: int = 48) -> np.ndarray:
    """A filled disc as a TRI_FAN, (steps + 2, 2) float32, centre first then the ring."""
    rim = circle_points(cx, cy, radius, steps)
    centre = np.array([[cx, cy]], dtype=np.float32)
    return np.concatenate([centre, rim, rim[:1]], axis=0)


def circle_lines(cx: float, cy: float, radius: float, steps: int = 48) -> np.ndarray:
    """The outline of a circle as LINES pairs, (2 * steps, 2) float32."""
    return loop_lines(circle_points(cx, cy, radius, steps))


# --- command palette -------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    """One entry in the Cmd+K list."""

    id: str
    name: str
    key: str = ""
    keywords: str = ""
    props: tuple[tuple[str, object], ...] = ()  # the operator's keyword arguments


COMMANDS: tuple[Command, ...] = (
    Command("undo", "Undo", "Ctrl+Z"),
    Command("redo", "Redo", "Shift+Ctrl+Z"),
    Command("grow", "Grow Voxels...", "Shift+G", "layers"),
    Command("shrink", "Shrink Voxels...", "Shift+D", "layers"),
    Command("hollow", "Hollow...", "", "shell"),
    Command("inset", "Inset...", "Shift+I"),
    Command("bevel", "Bevel"),
    Command("smooth", "Smooth", keywords="soften round blur relax"),
    Command("bridge", "Bridge", keywords="solid slab loft"),
    Command(
        "bridge",
        "Bridge (lines)",
        keywords="lines old pair",
        props=(("kind", "LINES"),),
    ),
    Command("fill_all", "Fill All", "", "whole model"),
    Command("fill_volume", "Fill Volume", "", "whole model"),
    Command("clear_all", "Clear All", "", "whole model"),
    Command("inverse", "Inverse", "", "whole model invert"),
    Command("fit_grid", "Fit Grid", "", "whole model"),
    Command("loop", "Loop...", "", "whole model"),
    Command("repeat", "Repeat..."),
    Command("diagonal", "Diagonal"),
    Command("drop_to_floor", "Drop to Floor", "", "whole model ground bottom gravity fall"),
    Command("drop_voxels", "Drop Voxels", "", "selection each single voxels gravity fall settle loose pile"),
    *(
        Command(c.op, f"{name}: {c.name}", keywords=keywords, props=c.props)
        for opener, name, keywords in ((MIRROR_HALF_ID, "Mirror Half", "mirror copy symmetry"),
                                       (SCALE_ID, "Scale", "up down double halve resolution"),
                                       (LATHE_ID, "Lathe", "turn revolve spin round milling vase column"),
                                       (SCENE_FLIP_ID, "Flip Models", "scene mirror flip picked"))
        for c in MODEL_CHOICES[opener].values()
    ),
    Command("twist", "Twist...", "", "whole model screw turn layers"),
    Command("rotate_degrees", "Rotate by Degrees...", "", "turn angle free selection whole model big slow rings"),
    Command("remove_slice", "Remove Slice", "", "cut layers close gap knife delete"),
    Command("resize", "Resize Grid..."),
    Command("cage", "Cage", keywords="lattice deform bend ffd"),
    Command("crease", "Crease...", keywords="edges bars frame outline panels canopy sharp ridge"),
    Command("brick_mode", "Brick Mode", "", "brick bricks build bin blocks"),
    Command("convert_to_bricks", "Convert to Bricks", "", "brick bricks build lock blocks"),
    Command("convert_to_voxels", "Convert to Voxels", "", "brick bricks unlock plain voxels"),
    Command("palette_cleanup", "Clean Up Palette", "", "unused remove colours tidy empty"),
    Command("palette_grid", "Show All 256 Colours", "", "palette colour grid"),
    Command("palette_import", "Import Palette...", "", "colour"),
    Command("palette_export", "Export Palette...", "", "colour"),
    Command("export_mesh", "Export Mesh...", "", "obj gltf fbx"),
    *(Command("export_animation", f"Export Animation: {label}...", "", "animation clip gltf glb vox game engine",
              props=(("kind", kind),))
      for kind, label in (("GLTF", "glTF"), ("GLTF_BAKED", "glTF, Baked"), ("VOX", ".vox"))),
    Command("export_pictures", "Export Sprite Sheet...", "", "animation clip sprites 2d directions aseprite",
            props=(("kind", "SPRITES"),)),
    Command("export_pictures", "Export GIF", "", "animation clip gif moving picture", props=(("kind", "GIF"),)),
    Command("export_pictures", "Export MP4", "", "animation clip video movie mp4", props=(("kind", "MP4"),)),
    Command("view_save", "Save View", "", "saved views"),
    Command("view_go", "Go to View", "", "saved views"),
    Command("selection_save", "Save Selection", "Ctrl+1", "saved selections keep store remember"),
    Command("selection_recall", "Recall Selection", "1", "saved selections bring back restore"),
    Command("select_all", "Select All", "Ctrl+A"),
    Command("deselect", "Deselect", "Esc"),
    Command("select_invert", "Invert Selection", "Ctrl+I"),
    Command("select_colour", "Select by Colour", "Shift+C"),
    Command("select_connected", "Select Connected", "Shift+L"),
    Command("select_grow", "Grow Selection", "Ctrl++", "expand bigger wider neighbours"),
    Command("select_shrink", "Shrink Selection", "Ctrl+-", "contract smaller narrower"),
    Command("copy", "Copy", "Ctrl+C"),
    Command("cut", "Cut", "Ctrl+X"),
    Command("paste", "Paste", "Ctrl+V"),
    Command("selection_delete", "Delete Selection", "Delete"),
    Command("fill", "Fill Connected Colour", "Shift+F"),
    Command("paint_selection", "Paint Selection", "Shift+P", "colour recolour fill selected"),
    Command("mask_selection", "Mask Selection", "Shift+M", "protect lock freeze keep"),
    Command("mask_clear", "Clear Mask", "Alt+M", "unmask unlock unprotect"),
    Command("flip", "Flip", "", "axis"),
    Command("rotate", "Rotate 90", "", "turn"),
    Command("model_add", "New Model", "", "scene"),
    Command("model_add_file", "Add From File...", "", "scene"),
    Command("terrain_new", "New Terrain...", "", "scene landscape tiles seamless ground"),
    Command("heightmap_new", "New From Heightmap...", "", "scene picture image png height terrain"),
    Command("shader_gallery", "Shaders...", "", "shader gallery browse xs effects generate"),
    Command("live_bake", "Bake Live Shaders", "", "shader live keep apply finish"),
    Command("model_duplicate", "Duplicate Model", "", "scene"),
    Command("model_rename", "Rename Model...", "", "scene"),
    Command("model_delete", "Delete Model", "", "scene"),
    Command("model_split", "Split Selection to Model", "", "scene"),
    Command("model_make_unique", "Make Unique", "", "scene linked copy unlink instance"),
    Command("scene_pick_linked", "Pick Linked Copies", "", "scene linked copy instance select all"),
    Command("scene_align", "Align Models...", "", "scene line up distribute spread centre"),
    Command("scene_array", "Array...", "", "scene repeat copies row duplicate instance"),
    Command("scene_carve", "Carve with Model", "", "scene cut subtract boolean hole cutter"),
    Command("model_merge_all", "Merge All Models", "", "scene"),
    Command("model_merge_active", "Merge Into Active", "", "scene"),
    Command("ui_switch", "Switch Rok UI", "", "classic new"),
    # Animation: Animate mode, its motions and keys.
    Command("animate", "Animate", "", "animation timeline mode motion", props=(("action", "ENTER"),)),
    Command("animate", "Back to Building", "", "animation leave exit build model", props=(("action", "LEAVE"),)),
    *(Command("anim_preset", f"Motion: {anim.PRESET_LABELS[n]}", "", "animation motion preset", props=(("name", n),))
      for n in anim.PART_PRESETS),
    Command("key_pose", "Key the Pose", "K", "animation keyframe"),
    Command("key_delete", "Delete Keys", "Delete", "animation keyframe"),
    Command("key_copy", "Copy Keys", "Ctrl+C", "animation keyframe clipboard"),
    Command("key_paste", "Paste Keys", "Ctrl+V", "animation keyframe clipboard play head"),
    Command("key_paste", "Paste Keys Mirrored", "Shift+Ctrl+V", "animation keyframe clipboard other leg opposite",
            props=(("mirrored", True),)),
    Command("key_value", "Key Value...", "", "animation keyframe type number channel"),
    Command("onion_skin", "Onion Skin", "", "animation ghost before after pose faint"),
    Command("frame_add", "Frame: Add", "", "animation frames paint copy blink voxel frame"),
    Command("frame_delete", "Frame: Delete", "", "animation frames remove voxel frame"),
    Command("anim_paint", "Paint: Attach", "T", "animation paint frame tool", props=(("tool", "ATTACH"),)),
    Command("anim_paint", "Paint: Erase", "R", "animation paint frame tool", props=(("tool", "ERASE"),)),
    Command("anim_paint", "Paint: Paint", "G", "animation paint frame tool", props=(("tool", "PAINT"),)),
    Command("anim_paint", "Paint: Pose", "", "animation gizmo pose stop painting", props=(("tool", "POSE"),)),
    Command("marker_add", "Marker: Add...", "", "animation marker moment footstep hit event"),
    Command("marker_rename", "Marker: Rename...", "", "animation marker name"),
    Command("marker_delete", "Marker: Delete", "", "animation marker remove"),
    Command("clip_settings", "Clip: Root Motion", "", "animation root motion in place export",
            props=(("root_motion", 2),)),
    *(Command("key_ease", f"Ease: {anim.EASE_LABELS[e]}", "", "animation keyframe", props=(("ease", e),))
      for e in anim.EASES),
    Command("play", "Play or Stop", "Shift+Space", "animation timeline"),
    Command("frame_go", "To Start", "", "animation timeline playhead", props=(("frame", 0),)),
    Command("frame_go", "Previous Frame", ",", "animation timeline playhead step back", props=(("step", -1),)),
    Command("frame_go", "Next Frame", ".", "animation timeline playhead step forward", props=(("step", 1),)),
    Command("frame_go", "Previous Key", "Shift+,", "animation timeline playhead keyframe", props=(("to_key", -1),)),
    Command("frame_go", "Next Key", "Shift+.", "animation timeline playhead keyframe", props=(("to_key", 1),)),
    Command("clip_settings", "Loop Mode", "", "animation loop once ping pong"),
    Command("make_part", "Make Part", "", "animation cut selection split part parent character"),
    Command("part_empty", "Empty Part", "", "animation root socket empty part"),
    Command("part_parent", "Part: No Parent", "", "animation unparent root follow", props=(("parent", -1),)),
    Command("clip_add", "Clip: Add", "", "animation new clip tab"),
    Command("clip_rename", "Clip: Rename...", "", "animation clip name tab"),
    Command("clip_duplicate", "Clip: Duplicate", "", "animation copy clip tab"),
    Command("clip_delete", "Clip: Delete", "", "animation remove clip tab"),
    Command("clip_length", "Clip Length...", "", "animation duration seconds fps frame rate"),
)


def command_match(query: str, commands: tuple[Command, ...] | None = None) -> tuple[Command, ...]:
    """Fuzzy filter of the command list, best first. An empty query returns all."""
    items = COMMANDS if commands is None else commands
    q = query.strip().lower()
    if not q:
        return items
    scored: list[tuple[tuple, Command]] = []
    for cmd in items:
        hay = f"{cmd.name} {cmd.id} {cmd.keywords}".lower()
        score = _fuzzy_score(q, hay, cmd.name.lower(), cmd.id.lower())
        if score is not None:
            scored.append((score, cmd))
    scored.sort(key=lambda item: item[0])
    return tuple(cmd for _score, cmd in scored)


def _fuzzy_score(q: str, hay: str, name: str, ident: str) -> tuple | None:
    """Lower is better. None when `q` does not match."""
    if q == name or q == ident:
        return (0, 0, name)
    if name.startswith(q) or ident.startswith(q):
        return (1, len(name), name)
    found = name.find(q)
    if found >= 0:
        return (2, found, name)
    found = ident.find(q)
    if found >= 0:
        return (3, found, name)
    found = hay.find(q)
    if found >= 0:
        return (4, found, name)
    span = _subseq_span(q, name) or _subseq_span(q, ident)
    if span is not None:
        return (5, span, name)
    span = _subseq_span(q, hay)
    if span is not None:
        return (6, span, name)
    return None


def _subseq_span(q: str, text: str) -> int | None:
    """How wide the first subsequence match is. Smaller is tighter, so better."""
    i = 0
    start = None
    for j, ch in enumerate(text):
        if ch != q[i]:
            continue
        if start is None:
            start = j
        i += 1
        if i == len(q):
            return j - start
    return None


@dataclass(frozen=True)
class CommandLayout:
    """The search panel for one region size, query and hit list."""

    rect: Rect  # the whole panel
    query_rect: Rect
    rows: tuple[Rect, ...]  # one per visible hit
    hits: tuple[Command, ...]
    index: int  # highlighted row, 0-based in `hits` (and `rows`)
    scale: float
    query: str


def command_layout(
    scale: float,
    region_width: int,
    region_height: int,
    query: str,
    hits: tuple[Command, ...],
    index: int,
    *,
    max_rows: int = COMMAND_MAX,
) -> CommandLayout:
    """A panel centred near the top of the region."""
    s = float(scale)
    width = min(int(round(COMMAND_WIDTH * s)), max(1, int(region_width) - 16))
    pad = int(round(COMMAND_PAD * s))
    row_h = int(round(COMMAND_ROW * s))
    shown = hits[: max(1, int(max_rows))]
    height = pad + row_h + pad + max(1, len(shown)) * row_h + pad
    x = max(0, (int(region_width) - width) // 2)
    # Sit near the top, centred horizontally. The left rail does not move it.
    top = int(region_height) - int(round(12 * s))
    y = max(0, top - height)
    query_rect = (x + pad, y + height - pad - row_h, width - 2 * pad, row_h)
    rows = []
    row_y = query_rect[1] - pad - row_h
    for _ in shown:
        rows.append((x + pad, row_y, width - 2 * pad, row_h))
        row_y -= row_h
    highlight = 0 if not shown else min(max(0, int(index)), len(shown) - 1)
    return CommandLayout(
        rect=(x, y, width, height),
        query_rect=query_rect,
        rows=tuple(rows),
        hits=shown,
        index=highlight,
        scale=s,
        query=query,
    )


def command_hit(layout: CommandLayout, px: float, py: float) -> int | None:
    """Row index under the pointer, or None outside the rows."""
    for i, rect in enumerate(layout.rows):
        if in_rect(rect, px, py):
            return i
    return None

