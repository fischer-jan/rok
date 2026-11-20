"""Shared state of voxel mode. Plain dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(frozen=True)
class Hover:
    """What is under the mouse and what a click would do there."""

    hit: object  # rok.core.raycast.Hit
    tool: str  # effective action: "ATTACH" | "ERASE" | "PAINT" | "SELECT"
    target: tuple[int, int, int] | None  # cell a click changes; None = click does nothing
    mode: str = "VOXEL"
    # Face mode: the region as an (N, 3) array, else None. Left out of == and hash:
    # a numpy array has no single truth value, and the key below already says
    # which region it is (voxel, normal, tool, serial).
    cells: object = field(default=None, compare=False)
    serial: int = 0  # document.edit_serial the region was computed for; 0 outside Face mode
    subtract: bool = False  # Select with Option+Shift: the press would remove, not add


def hover_key(hover: Hover | None):
    """What the highlight depends on. Two hovers with the same key draw the same.

    `Hit.t` is left out: it changes with every mouse pixel on one face, and a redraw
    per pixel is what the key avoids.
    """
    if hover is None:
        return None
    return (
        hover.hit.voxel,
        hover.hit.normal,
        hover.tool,
        hover.target,
        hover.mode,
        hover.subtract,
        hover.serial,
    )


@dataclass
class GizmoState:
    """The transform gizmo: what it acts on and where it is drawn."""

    target: str  # "SELECTION" | "MODEL"
    pivot2: tuple[int, int, int]  # doubled units of the space the gizmo is drawn in
    origin: tuple[float, float, float]  # world origin of that space (model_origin)
    length_px: float  # 90 * ui scale
    uid: int  # the model the gizmo belongs to


@dataclass
class ModeState:
    """Mutable state shared by the mode operator, the stroke and the overlay."""

    generation: int = 0  # load_pre and unregister bump it; a stale modal instance exits
    running: bool = False  # rok.voxel_mode is alive
    stroke_active: bool = False
    model_box: bool = True  # View Settings > Model Box: the faint edges around the models
    steady_line: tuple | None = None  # Steady Hand: ((brush x, y), (mouse x, y)) in region pixels while the string is taut
    hover: Hover | None = None
    hover_view: object = None  # perspective matrix of the view the hover was picked in
    corner: tuple | None = None  # (region pointer, CornerLayout, cell_px) of the corner gizmo as drawn
    live_render: str | None = None  # the preset id while the view renders live (View > Live Render)
    hover_strip: tuple[str, int] | None = None  # (kind, index) of the palette strip item under the mouse
    # (region pointer, {part: rect}) of the clickable parts of the file label:
    # "GRID_SIZE" (the X x Y x Z, resizes) and "LABEL_COLOUR" (the colour number, the picker).
    label_parts: tuple[int, dict[str, tuple[int, int, int, int]]] | None = None
    hover_chrome: str | None = None  # widget id of the new UI's strip under the mouse
    chrome_open: str | None = None  # the chip whose hover flyout is open
    chrome_since: float | None = None  # monotonic time the chrome hover began; times the tooltip
    chrome_focus: str | None = None  # the widget the keyboard focus is on
    hover_handle: str | None = None  # gizmo or cage handle under the mouse
    hover_pill: str | None = None  # selection-menu button under the mouse
    cage: object = None  # rok.app.cage_ops.CageState while the lattice cage shows, else None
    pill_since: float | None = None  # monotonic time the pill hover began; times the HUD tip
    hover_model: int | None = None  # uid of the hovered inactive model (an ACTIVATE hover)
    gizmo: GizmoState | None = None  # None = no gizmo shown
    gizmo_screen: object = None  # rok.core.gizmo.GizmoScreen, written by the overlay each draw
    gizmo_region: object = None  # as_pointer() of the region gizmo_screen was projected for
    cage_screen: object = None  # rok.core.cage.CageScreen, written by the overlay each draw
    cage_region: object = None  # as_pointer() of the region cage_screen was projected for
    # Every view of a split or quad view projects its own gizmo and cage; the
    # hit test of a view reads that view's (region as_pointer() -> screen).
    gizmo_screens: dict = field(default_factory=dict)
    cage_screens: dict = field(default_factory=dict)
    anchor_screen: object = None  # (x, y) region pixels of the anchor dot, written by the overlay each draw
    anchor_region: object = None  # as_pointer() of the region anchor_screen was projected for
    gizmo_ghost: tuple[tuple, object] | None = None  # (offset, cells) preview of a model transform
    model_drag_offset: tuple[int, int, int] = (0, 0, 0)  # the live offset of a MODEL move drag
    onion_skin: bool = False  # Animate mode: faint copies of the poses before and after the play head
    anim_paint: str | None = None  # Animate mode: ATTACH, ERASE or PAINT paints the picked part's frame; None poses
    # uid -> 4x4 (grid units, relative to the rest place) of a scene Rotate
    # preview: the Empties are turned so, and the boxes follow. Empty at rest.
    turn_preview: dict = field(default_factory=dict)
    preview: object = None  # rok.core.gesture.Preview during a drag, else None
    last_mouse: tuple[int, int] = (0, 0)  # window pixels of the last event seen
    ctrl: bool = False  # modifier state from the last event
    shift: bool = False
    alt: bool = False  # Option/Alt: erase, never orbit (emulate-3-button is off)
    option_erase_count: int = 0  # Option-erase strokes this session
    hud_tip_faded: bool = False  # idle HUD tip hidden after three Option-erases
    hud_last_tool: str | None = None  # header tool the HUD last saw; a change restores the tip
    start_tries: int = 0
    profile: bool = False  # ROK_PROFILE=1 in the environment
    clipboard: object = None  # rok.core.clipboard.Clip; survives New and Open
    pattern_rotation: tuple = (1, 0, 0, 0, 1, 0, 0, 0, 1)  # how the pattern brush is turned
    pattern_serial: int = 0  # bumped on a turn or a new copy; keys the pattern hover
    # The levels: "MODEL" edits the voxels of
    # document.model; "SCENE" works on whole models, and document.model is the
    # picked one. level_since times the rail's colour change.
    level: str = "MODEL"
    level_since: float = 0.0
    level_prev: str | None = None  # the level before the last change, for the rail's fade
    last_pick: tuple[int, float] | None = None  # (uid, monotonic time) of the last scene click
    # The mode each tool was last used in, and Attach's solid with it: a tool
    # chip on the star picks the tool in that mode again.
    tool_modes: dict = field(default_factory=dict)
    last_inside: int | None = None  # the model you were last inside; the Scene chip goes back to it
    group: set = field(default_factory=set)  # uids of the picked models in the scene, the leader too (empty: one model)
    snap_uids: set = field(default_factory=set)  # the models a scene drag snapped to, outlined while it holds
    scene_tool: str = "PICK"  # the scene's tool: "PICK" (move, stretch) or "CUT"
    # Brick mode (rok/app/brick_mode.py): the tool, the brick in the hand
    # ((w, d) or None) and its quarter turns.
    brick_tool: str = "PLACE"
    brick_hand: tuple | None = None
    brick_turn: int = 0
    # The bin's Random colours switch, and the colour (an index into
    # brick.RANDOM_COLOURS) the next brick takes while it is on.
    brick_random: bool = False
    brick_next: int = 0
    cut_other: bool = False  # Tab: the cutting plane turns to the face's other axis
    cut_aim: tuple[int, int, int] | None = None  # (uid, axis, boundary) of the plane under the mouse
    size_edit: str | None = None  # the grid size being typed into the label, None when not typing
    size_edit_fresh: bool = False  # the whole typed size is selected: the first key replaces it
    # Animation. playing is true
    # while the timeline plays; the hover, the stroke and the
    # overlay read it. `animating` is Animate mode (rok/app/anim_mode.py):
    # the violet rail, the timeline dock, a click picks a part, and Record
    # writes keys only there. anim_since times the rail's colour change.
    playing: bool = False
    animating: bool = False
    anim_since: float = 0.0
    anim_frame: int = 0  # the play head Animate mode left at; entering again goes back to it
    anchor_picking: bool = False  # rok.anchor_pick waits for its click
    posed_pill: tuple | None = None  # (region pointer, rect) of the Posed pill as last drawn
    hover_pill_posed: bool = False
    anim_target: tuple[str, int] | tuple[str, tuple[int, ...]] | None = None
    # ("PART", uid) or ("COLOUR", slot) or ("COLOUR", slots) for a range (Task 3.8); None = the active model
    anim_pick: bool = False  # the one-shot Colour pick is armed
    anim_pending: str | None = None  # a colour preset that waits for the colour pick
    active_track: int | None = None
    anim_property: str | None = None
    selected_keys: tuple[tuple[int, str, int], ...] = ()  # (track index, property, time)
    expanded_tracks: frozenset = frozenset()
    hover_timeline: object = None
    timeline_scroll: float = 0.0  # points; whole rows, clamped so the last row stays visible
    timeline_zoom: float = 0.0  # points per frame; 0 fits the timeline into the dock
    # The transient "Key at frame N" hint (Task 3.12): (text, monotonic time it was set).
    anim_hint: tuple[str, float] | None = None
    # (track_index, property) and monotonic time of the last click on a timeline
    # row, so a second click there within the double-click window adds a key.
    last_row_click: tuple[tuple[int, str | None], float] | None = None


state = ModeState()


def gizmo_screen_for(region) -> object:
    """The projected gizmo geometry, but only when it was drawn for `region`.

    Each 3D view keeps its own, so a quad view's top view is not hit-tested
    against the anchor pixels of the view that drew last.
    """
    if region is None:
        return None
    return state.gizmo_screens.get(region.as_pointer())


def cage_screen_for(region) -> object:
    """The projected cage geometry, but only when it was drawn for `region`."""
    if region is None:
        return None
    return state.cage_screens.get(region.as_pointer())


def anchor_screen_for(region) -> object:
    """The projected anchor dot, but only when it was drawn for `region`."""
    if region is None or state.anchor_screen is None:
        return None
    if state.anchor_region != region.as_pointer():
        return None
    return state.anchor_screen
