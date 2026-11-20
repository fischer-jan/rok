"""The open document and the view of it.

There is one `Document` and a set of `voxel_object` functions; the
functions hand their work to the app (the GPU sync, the camera, redraws).
Other modules call `voxel_object.flush_dirty()` and friends.
"""

from __future__ import annotations

import math

import numpy as np

from rok.core.grid import VoxelGrid
from rok.core.scene import SceneModel, VoxScene
from rok.core.select import Selection
from rok.core.undo import UndoStack

from .opkit import CONTEXT
from .state import state

# Functions told the new value whenever the unsaved flag is set (recovery.py).
modified_hooks: list = []

FADE_KEEP = 0.35  # how much of a colour stays (the shader uses the same numbers)
FADE_TOWARD = (0.19, 0.18, 0.17)


def _app():
    return CONTEXT.app


class Document:
    """The open scene. The numpy grids are the source; the GPU textures are a view."""

    def __init__(self) -> None:
        self.scene: VoxScene | None = None
        self.active_uid: int = 0  # uid of the model the tools edit
        self.filepath: str = ""
        self.undo: UndoStack = UndoStack()
        self.colour: int = 1  # palette index 1..255; survives New and Open
        self._modified: bool = False
        self.palette_version: int = 0  # bumped when the palette changed
        self.edit_serial: int = 0  # bumped when the voxels or the grid shape changed
        self.view_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # always 0: cells are world cells
        self.palette_session: int = 0  # bumped when the colour popup opens
        self.made: set[int] = set()  # slots whose colour you made (made_unused)
        self.pivot_override: tuple[float, float, float] | None = None  # gizmo pivot, grid units
        self.stroke_serial: int = 0  # stroke end, undo, redo, scene ops; keys used_counts
        self.views: dict[int, tuple] = {}  # slot 1..4 -> saved view
        self.saved_selections: dict[tuple[int, int], np.ndarray] = {}  # (model uid, slot 1..4) -> mask
        self.mask_serial = 0  # bumps on every mask change, for the overlay's cache
        self.record: bool = True
        self._used_counts: np.ndarray | None = None
        self._used_serial: int = -1
        self._type_codes: np.ndarray | None = None
        self._type_codes_version: int = -1

    @property
    def model(self) -> SceneModel | None:
        """The active model, or None without a scene or with a stale uid."""
        if self.scene is None:
            return None
        try:
            return self.scene.by_uid(self.active_uid)
        except ValueError:
            return None

    @property
    def grid(self) -> VoxelGrid | None:
        """The grid the tools edit: the active model's own, or in Animate mode
        the frame the play head shows (6.5: paint the frame, not the rest)."""
        model = self.model
        if model is None:
            return None
        if state.animating and model.states:
            from . import anim_mode

            return model.frame_grid(anim_mode.shown_frame(model))
        return model.grid

    @property
    def selection(self) -> Selection | None:
        model = self.model
        return None if model is None else model.selection

    @selection.setter
    def selection(self, selection: Selection) -> None:
        model = self.model
        if model is not None:
            model.selection = selection

    def used_counts(self) -> np.ndarray:
        """(256,) int64 voxel count per palette index, cached by stroke_serial."""
        if self.scene is None:
            return np.zeros(256, dtype=np.int64)
        if self._used_counts is None or self._used_serial != self.stroke_serial:
            self._used_counts = self.scene.used_counts()
            self._used_serial = self.stroke_serial
        return self._used_counts

    def made_unused(self) -> tuple[int, ...]:
        """The colours you made that no voxel uses now and that still differ from
        the default palette: the strip's second section."""
        if self.scene is None or not self.made:
            return ()
        from rok.core.palette import default_palette

        used = self.used_counts()
        default = default_palette()
        palette = self.scene.palette
        return tuple(sorted(i for i in self.made if 0 < i < 256 and int(used[i]) == 0
                            and not np.array_equal(palette[i][:3], default[i][:3])))

    def pinned(self) -> tuple[int, ...]:
        """The slots no new colour may take although no voxel uses them: the
        colour you paint with, the colours you made and have not used yet, and
        the colours a colour motion (glow, blink, flow) animates."""
        from rok.core.anim import COLOUR

        motions = ()
        if self.scene is not None:
            motions = tuple(int(s) for clip in self.scene.clips for track in clip.tracks
                            if track.target.kind == COLOUR for s in track.target.slots)
        return (int(self.colour), *self.made_unused(), *motions)

    def type_codes(self) -> np.ndarray:
        """(256,) uint8 material type code per palette index, cached by palette_version."""
        if self.scene is None:
            return np.zeros(256, dtype=np.uint8)
        if self._type_codes is None or self._type_codes_version != self.palette_version:
            self._type_codes = self.scene.materials.type_codes()
            self._type_codes_version = self.palette_version
        return self._type_codes

    def set_active(self, uid: int) -> None:
        """Make model `uid` the one the tools edit."""
        self.active_uid = uid
        self.pivot_override = None
        state.hover = None
        apply_fade()

    @property
    def modified(self) -> bool:
        return self._modified

    @modified.setter
    def modified(self, value: bool) -> None:
        was = self._modified
        self._modified = bool(value)
        if was != self._modified and _app() is not None:
            _app().update_title()
        for fn in modified_hooks:
            fn(self._modified)

    def reset(self) -> None:
        """Drop the scene. The colour stays."""
        self.scene = None
        self.active_uid = 0
        self.filepath = ""
        self.undo = UndoStack()
        self.modified = False
        self.palette_version += 1
        self.edit_serial += 1
        self.stroke_serial += 1
        self.view_offset = (0.0, 0.0, 0.0)
        self.pivot_override = None
        self.made = set()
        self.views = {}
        self.saved_selections = {}
        self.mask_serial += 1
        self.record = True
        self._used_counts = None
        self._type_codes = None

    @property
    def name(self) -> str:
        from pathlib import Path

        return Path(self.filepath).name if self.filepath else "Untitled.vox"


document = Document()


def _sync_links(record, undone: bool) -> None:
    """After every push, undo and redo: the linked copies of a changed model
    take its new voxels (core/link.py)."""
    from rok.core import link as link_core

    scene = document.scene
    if scene is None:
        return
    if link_core.sync_after(scene, record, undone):
        sync_scene()
        redraw_3d_views()


UndoStack.after_apply = _sync_links


# --- placement ----------------------------------------------------------------------


def settle_scene(scene: VoxScene) -> tuple[int, int, int]:
    """Move every model by one whole-cell shift so the scene stands centred on
    the floor grid: its lowest cell at z = 0, centred in x and y."""
    lo, hi = scene.world_bounds()
    offset = (-(lo[0] + hi[0] + 1) / 2.0, -(lo[1] + hi[1] + 1) / 2.0, -float(lo[2]))
    shift = tuple(int(math.floor(v + 0.5)) for v in offset)
    if shift != (0, 0, 0):
        for model in scene.models:
            model.position = tuple(int(p) + d for p, d in zip(model.position, shift))
    return shift


def model_origin(model: SceneModel) -> tuple[float, float, float]:
    """World location of grid cell (0, 0, 0) of `model`."""
    ox, oy, oz = document.view_offset
    px, py, pz = model.position
    return (ox + px, oy + py, oz + pz)


def model_name(uid: int) -> str:
    return f"RokModel.{uid}"


# --- scene in, scene out -------------------------------------------------------------


def fading() -> bool:
    """True when the models other than the active one are faded: inside a model,
    with more than one."""
    scene = document.scene
    return state.level in ("MODEL", "BRICK") and not state.animating and scene is not None and len(scene.models) > 1


hidden_preview: set[int] = set()  # models a drag hides for a while (a stretch's ghost stands in)
anim_hidden: frozenset = frozenset()  # parts a Show key hides at the play head (anim_mode.shown)


def style_of(model: SceneModel) -> tuple[float, bool]:
    """(fade, hidden) of one model as the view shows it."""
    scene = document.scene
    hidden = (scene is not None and scene.is_hidden(model)) or model.uid in hidden_preview \
        or model.uid in anim_hidden
    fade = 1.0 if (fading() and model.uid != document.active_uid) else 0.0
    return fade, hidden


def sync_scene() -> None:
    """Bring the view in line with the scene after any change to it (flip,
    rotate, resize, undo, redo, the scene operators): models, sizes,
    positions, dirty chunks, fade and layer visibility."""
    app = _app()
    if app is not None:
        app.flush()
    document.edit_serial += 1
    state.hover = None
    state.preview = None


def flush_dirty() -> int:
    """Upload every dirty chunk of every model. Returns the number of dirty
    chunks; when there were some, `edit_serial` goes up (the Face and
    Connected hover floods key on it)."""
    scene = document.scene
    dirty = 0 if scene is None else sum(len(m.grid.dirty_chunks()) for m in scene.models)
    app = _app()
    if app is not None:
        app.flush()
    if dirty:
        document.edit_serial += 1
    return dirty


def rebuild_all() -> None:
    if document.scene is None:
        return
    for model in document.scene.models:
        model.grid.mark_all_dirty()
    flush_dirty()


def apply_fade() -> None:
    sync_scene()
    redraw_3d_views()


def apply_layer_visibility() -> None:
    sync_scene()


def update_palette() -> None:
    """After a palette edit: upload it and tell the overlay."""
    document.palette_version += 1
    sync_scene()
    redraw_3d_views()


def write_colours(writes, made: bool = True, materials=None) -> None:
    """Put new colours ({slot: colour}, on free slots) into the scene's colour
    store, after the current colour in the strip. The change travels with the
    next undo record the command pushes (UndoStack.track). `made`: the
    colours count as made by you."""
    if not writes:
        return None
    store = document.scene.colours
    anchor = store.id_of(document.colour)
    for slot, colour in writes.items():
        material = None if materials is None else materials.get(slot)
        anchor = store.claim(int(slot), colour, material, after=anchor)
    if made:
        document.made.update(int(s) for s in writes)
    update_palette()
    return None


def merge_colours(merged, claimed_by, materials=None) -> None:
    """Put the colours a palette merge wrote (palette.merge_claimed: slot k
    holds incoming colour claimed_by[k]) into the colour store, each with the
    incoming material, after the current colour in the strip."""
    store = document.scene.colours
    anchor = store.id_of(document.colour)
    for slot in range(1, 256):
        source = int(claimed_by[slot])
        if source:
            material = None if materials is None else materials.get(source)
            anchor = store.claim(slot, merged[slot], material, after=anchor)
    update_palette()


def colours_now():
    """The scene's colours as they are, to put back later (colours_back)."""
    store = document.scene.colours
    return store.snapshot(), store.version


def colours_back(saved) -> None:
    """Put the scene's colours back as colours_now saved them, without an undo
    step: a preview's colours leave no trace."""
    if saved is None or document.scene is None:
        return
    snapshot, version = saved
    store = document.scene.colours
    if store.version == version:
        return
    undo = document.undo
    quiet = undo._colours_seen is not None and undo._colours_seen[0] == version
    store.restore(snapshot)
    if quiet:
        undo._colours_now()  # nothing of yours was pending: no step to record
    update_palette()


def revert_colours(snapshot) -> None:
    """Take the colour store back to `snapshot` (a press that came to nothing)."""
    if snapshot is not None:
        document.scene.colours.restore(snapshot)
        update_palette()


def redraw_3d_views() -> None:
    app = _app()
    if app is not None:
        app.request_redraw()


redraw_statusbar = redraw_3d_views


def apply_scene(context, scene: VoxScene, active_uid: int | None = None, settle: bool = True) -> None:
    """Make `scene` the document (New, Open) and show it. With `settle` the scene
    is moved to stand centred on the floor grid (a new model); without it every
    model keeps its position (an opened file: its cells are the file's cells, and
    saving it again moves nothing)."""
    from .handlers import registry

    app = _app()
    if app is not None:
        app.sync.reset()
    old = document.scene
    keep = None if old is None or document.colour < 1 else tuple(int(c) for c in old.palette[document.colour])
    keep_material = None if keep is None else old.materials.get(document.colour)
    document.scene = scene
    document.active_uid = active_uid if active_uid is not None else scene.models[0].uid
    document.undo = UndoStack()
    if keep is not None:
        # The colour you paint with stays: the same colour in the new scene,
        # found or added, not the same number.
        store = scene.colours
        ident = store.find(keep, keep_material)
        if ident is None:
            ident = store.add(keep, keep_material)
        slot = store.lease(ident, scene.used_counts())
        if slot is not None:
            document.colour = slot
    document.undo.track(scene.colours)
    document.modified = False
    document.pivot_override = None
    if settle:
        settle_scene(scene)
    document.view_offset = (0.0, 0.0, 0.0)
    document.stroke_serial += 1
    document._used_counts = None
    if document.colour < 1:
        document.colour = 1
    for model in scene.models:
        model.grid.mark_all_dirty()
    document.palette_version += 1
    document.edit_serial += 1
    state.hover = None
    state.gizmo_ghost = None
    state.model_drag_offset = (0, 0, 0)
    state.turn_preview.clear()
    # A file of several models opens in the scene; one model opens inside it.
    state.level = "SCENE" if len(scene.models) > 1 else "MODEL"
    state.level_since = 0.0
    state.last_pick = None
    state.last_inside = None
    state.group.clear()
    flush_dirty()
    for fn in registry.load_post:
        fn(context)
    frame_scene(context)
    if app is not None:
        app.update_title()
    redraw_3d_views()


def apply_grid(context, grid: VoxelGrid) -> None:
    """Make one grid the whole scene (New)."""
    apply_scene(context, VoxScene.single(grid))


# --- framing -------------------------------------------------------------------------


def _frame(context, lo, hi) -> None:
    """Fit the box lo..hi into the view the mouse works in, from its current angle."""
    view_state = context.view_state
    view_state.camera.fit(lo, hi)
    context.app.camera_moved()


def frame_scene(context) -> None:
    """Point the view at the scene centre, far enough back to see it all."""
    scene = document.scene
    if scene is None:
        return
    lo, hi = scene.world_bounds()
    offset = document.view_offset
    _frame(context, tuple(lo[i] + offset[i] for i in range(3)), tuple(hi[i] + 1 + offset[i] for i in range(3)))


def frame_models(context, models) -> None:
    """Point the view at these models, far enough back to see them all. The view keeps its angle."""
    if not models:
        return
    los, his = [], []
    for model in models:
        origin = model_origin(model)
        los.append(origin)
        his.append(tuple(origin[i] + model.grid.size[i] for i in range(3)))
    lo = [min(p[i] for p in los) for i in range(3)]
    hi = [max(p[i] for p in his) for i in range(3)]
    _frame(context, lo, hi)


def frame_model(context, model: SceneModel) -> None:
    frame_models(context, [model])
