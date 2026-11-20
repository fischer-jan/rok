"""The pattern brush: the clipboard box, turned, as the stamp of Pattern mode.

`current_pattern` is the oriented box, cached per clip and rotation. The turn operator changes
`state.pattern_rotation` and rehovers, so the ghost under the mouse turns at
once. `hover_cells` gives the ghost for the hover.
"""

from __future__ import annotations

import numpy as np

from rok.core import pattern
from rok.core.edit import ATTACH, ERASE, STAMP_MODES

from . import object as voxel_object
from .object import document
from .opkit import EnumProp, Action, register
from .ops_common import rehover
from .state import state

PATTERN_HINT = "Pattern: click or drag to stamp the copied voxels   R: turn   Shift+R: tip over   N: add / carve / holes"
CLONE_HINT = "Clone: click or drag to stamp the copied voxels in their colours   R: turn   Shift+R: tip over   N: add / carve / holes"
NO_PATTERN_HINT = "Pattern: copy some voxels first (Cmd+C)"

_cache: tuple[object, tuple, np.ndarray] | None = None  # (clip, rotation, box)


def current_pattern() -> np.ndarray | None:
    """The clipboard box in the current orientation, or None with an empty clipboard."""
    global _cache
    clip = state.clipboard
    if clip is None:
        return None
    rotation = state.pattern_rotation
    if _cache is None or _cache[0] is not clip or _cache[1] != rotation:
        _cache = (clip, rotation, pattern.oriented(clip.voxels, rotation))
    return _cache[2]


# What an Attach stamp does with the pattern: build its shape
# (ADD), carve the shape out of the model (CARVE, Erase's stamp), or sink into
# the surface, build the shape and clear the pattern's empty cells (HOLES), so
# a window frame cuts its opening too.
SPACES = ("ADD", "CARVE", "HOLES")
SPACE_NAMES = {"ADD": "Add", "CARVE": "Carve", "HOLES": "Holes"}


def space() -> str:
    from .opkit import CONTEXT

    value = getattr(CONTEXT.prefs, "rok_pattern_space", "ADD")
    return value if value in SPACES else "ADD"


def stamp_tool(tool: str) -> tuple[str, bool]:
    """(the tool a Pattern stamp runs as, holes) for the pressed tool: Carve
    makes Attach's stamp Erase's; Holes keeps Attach and sinks the box in."""
    if tool != ATTACH:
        return tool, False
    s = space()
    if s == "CARVE":
        return ERASE, False
    return ATTACH, s == "HOLES"


def hover_cells(grid, hit, tool) -> np.ndarray | None:
    """(N, 3) cells the pattern would cover for this hit, or None when there is
    no pattern or the tool cannot stamp on this face."""
    box = current_pattern()
    if box is None:
        return None
    tool, holes = stamp_tool(tool)
    origin = pattern.placement(box, hit, ERASE if holes else tool)
    if origin is None:
        return None
    cells, values = pattern.box_cells(box, origin, grid.size)
    global _hover_values
    _hover_values = (cells, values)  # the Clone ghost's colours, cell by cell
    return cells


_hover_values: tuple[np.ndarray, np.ndarray] | None = None  # the last hover's (cells, clip palette indices)


def hover_colours(cells: np.ndarray) -> list[tuple[tuple[int, int, int], np.ndarray]] | None:
    """The Clone ghost's colours: the hover's `cells` (as hover_cells gave
    them) grouped by the copied colour, as [(rgb, cells)]."""
    clip = state.clipboard
    if clip is None or _hover_values is None or _hover_values[0] is not cells:
        return None
    values = _hover_values[1]
    out = []
    for v in np.unique(values):
        rgb = tuple(int(c) for c in clip.palette[int(v), :3])
        out.append((rgb, np.asarray(cells)[values == v]))
    return out


@register
class PatternSpaceAction(Action):
    command_id = "rok.pattern_space"
    title = "Pattern: Add, Carve or Holes"
    tooltip = "What an Attach stamp does: build the shape, carve it out, or build it and cut its empty cells"
    step = EnumProp(
        name="Step",
        items=(("NEXT", "Next", "The next one"), ("BACK", "Back", "The one before")),
        default="NEXT",
    )

    @classmethod
    def poll(cls, context):
        return (
            document.grid is not None
            and not state.stroke_active
            and state.level == "MODEL"
            and getattr(context.prefs, "rok_mode", None) in STAMP_MODES
        )

    def execute(self, context):
        k = SPACES.index(space()) + (1 if self.step == "NEXT" else -1)
        context.prefs.rok_pattern_space = SPACES[k % len(SPACES)]
        self.report({"INFO"}, f"Pattern: {SPACE_NAMES[space()]}")
        rehover()
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class PatternTurnAction(Action):
    command_id = "rok.pattern_turn"
    title = "Turn Pattern"
    tooltip = "Turn the pattern brush a quarter turn"
    axis = EnumProp(
        name="Axis",
        description="The axis to turn about",
        items=(
            ("Z", "Z", "Turn about the vertical axis"),
            ("X", "X", "Tip over about the x axis"),
        ),
        default="Z",
    )

    @classmethod
    def poll(cls, context):
        # Only in Pattern mode, so bare R falls through to set_tool Erase elsewhere.
        return (
            document.grid is not None
            and not state.stroke_active
            and state.level == "MODEL"  # R turns the picked model in the scene, the brick in Brick mode
            and getattr(context.prefs, "rok_mode", None) in STAMP_MODES
        )

    def execute(self, context):
        if state.clipboard is None:
            self.report({"WARNING"}, NO_PATTERN_HINT)
            return {"CANCELLED"}
        state.pattern_rotation = pattern.turned(state.pattern_rotation, "XYZ".index(self.axis))
        state.pattern_serial += 1
        rehover()
        voxel_object.redraw_3d_views()
        return {"FINISHED"}
