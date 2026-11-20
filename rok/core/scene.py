"""The scene: several models over one shared palette. Pure numpy.

This is the only core module that builds VoxelGrid objects from file data.
Every grid in a scene aliases the scene palette (the same array object), so a
palette edit is one write.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import anim, layers as layers_core, light as light_core, link as link_core
from .colour_store import ColourStore
from .grid import VoxelGrid
from .grid_ops import MAX_SIZE
from .material import MaterialTable
from .select import Selection
from .vox_read import ModelData, SceneData
from .vox_write import DEFAULT_LAYERS

__all__ = ["DEFAULT_LAYERS", "SceneModel", "VoxScene", "merged"]

Cell = tuple[int, int, int]


@dataclass
class SceneModel:
    """One model of the scene: a grid placed at a world cell."""

    uid: int  # unique in the session; VoxScene.next_uid hands them out; never reused
    name: str  # nTRN "_name"; "" when the file has none
    position: Cell  # world cell of grid cell (0, 0, 0); may be negative
    grid: VoxelGrid  # grid.palette IS scene.palette
    selection: Selection  # one per model; not saved; SelectionRecord holds it on undo
    layer: int = 0  # nTRN layer id; written back so the model stays on its file layer
    # Voxel states besides the grid's own (state 0), as a .vox animation
    # swaps them; the timeline's STATE keys say when each shows.
    states: list[np.ndarray] = field(default_factory=list)
    # Linked copies (core/link.py): the group id (0: not linked) and how this
    # model's voxels are turned from the group's shared shape.
    link: int = 0
    orient: tuple = link_core.IDENTITY
    # Animation: the point the part turns about, in
    # doubled grid units (None: its box's middle), and the part it follows.
    pivot2: Cell | None = None
    parent: int | None = None  # a uid
    empty: bool = False  # an Empty part: no voxels of its own, a root or a socket (6.3)
    # Frames (6.5): `grid` is frame 1, `states` hold frames 2.. as arrays (the
    # .vox reads and writes them). A frame painted in Animate mode is a
    # VoxelGrid over its array, made on first use; the array stays shared.
    frame_grids: dict = field(default_factory=dict, repr=False, compare=False)

    def frame_count(self) -> int:
        return 1 + len(self.states)

    def frame_grid(self, index: int) -> VoxelGrid:
        """The grid of frame `index` (0: the model's own grid, k: states[k - 1])."""
        if index <= 0 or index > len(self.states):
            return self.grid
        g = self.frame_grids.get(index)
        if g is None or g.voxels is not self.states[index - 1]:
            g = VoxelGrid(self.grid.size, self.states[index - 1], self.grid.palette)
            g.palette = self.grid.palette
            self.states[index - 1] = g.voxels  # one array for the grid and the state
            self.frame_grids[index] = g
        return g

    def add_frame(self, voxels: np.ndarray) -> int:
        """A new frame holding a copy of `voxels`. Returns its index."""
        self.states.append(np.array(voxels, dtype=np.uint8, copy=True))
        return len(self.states)

    def remove_frame(self, index: int) -> None:
        if 0 < index <= len(self.states):
            del self.states[index - 1]
            self.frame_grids.clear()

    def size(self) -> Cell:
        return self.grid.size

    def box_world(self) -> tuple[Cell, Cell]:
        """Inclusive world box (position, position + size - 1)."""
        lo = tuple(int(p) for p in self.position)
        hi = tuple(int(p) + int(d) - 1 for p, d in zip(lo, self.grid.size))
        return lo, hi

    def display_name(self, index: int) -> str:
        return self.name if self.name else f"Model {index + 1}"


@dataclass
class VoxScene:
    """The models of one .vox file plus everything that passes through a save."""

    models: list[SceneModel] = field(default_factory=list)
    palette: np.ndarray = field(default_factory=lambda: np.zeros((256, 4), dtype=np.uint8))
    palette_tail: bytes = b"\0\0\0\0"
    layers: list[bytes] = field(default_factory=lambda: list(DEFAULT_LAYERS))
    extras: list[tuple[bytes, bytes]] = field(default_factory=list)
    materials: MaterialTable = field(default_factory=MaterialTable)
    next_uid: int = 1
    next_link: int = 1
    # The clips: every one starts from the same
    # rest pose; `timeline` is the active one, which the dock shows.
    clips: list = field(default_factory=lambda: [anim.Clip()])
    active_clip: int = 0
    # Live shaders per model uid (rok.core.live_shader). Not saved: a file
    # keeps their voxels.
    live: dict = field(default_factory=dict)
    # The file's own light (rok.core.light), kept in a .vox chunk; None:
    # the default light.
    light: light_core.LightSetup | None = None
    # The length of one voxel, in metres, and the unit the side labels show
    # ("m" or "ft"). None: the labels stay in cells. One length for the scene,
    # because every model shares the cell grid. A .vox has no place for it.
    cell_metres: float | None = None
    cell_unit: str = "m"
    # The colours (rok.core.colour_store): entries, your order, the leases.
    # Made from the palette on first use when a file brought none.
    _colours: ColourStore | None = field(default=None, repr=False, compare=False)

    @property
    def colours(self) -> ColourStore:
        if self._colours is None or self._colours.palette is not self.palette:
            self._colours = ColourStore.from_palette(self.palette, self.materials, self.used_counts())
        self._colours.materials = self.materials
        self._colours.sync()  # rows written outside the store become colours
        return self._colours

    @colours.setter
    def colours(self, store: ColourStore) -> None:
        self._colours = store

    @property
    def timeline(self) -> anim.Clip:
        if not self.clips:
            self.clips = [anim.Clip()]
        self.active_clip = min(max(int(self.active_clip), 0), len(self.clips) - 1)
        return self.clips[self.active_clip]

    @timeline.setter
    def timeline(self, clip: anim.Clip) -> None:
        if not self.clips:
            self.clips = [clip]
            self.active_clip = 0
            return
        self.active_clip = min(max(int(self.active_clip), 0), len(self.clips) - 1)
        self.clips[self.active_clip] = clip

    def would_cycle(self, uid: int, parent: int | None) -> bool:
        """True when making `parent` the parent of `uid` closes a loop."""
        seen = {uid}
        while parent is not None:
            if parent in seen or not self.has_uid(parent):
                return parent in seen
            seen.add(parent)
            parent = self.by_uid(parent).parent
        return False

    def set_parent(self, uid: int, parent: int | None) -> bool:
        """Make `parent` (a uid, or None) the parent of `uid`. False on a loop or a
        missing model: nothing changes then."""
        if parent is not None and (parent == uid or not self.has_uid(parent) or self.would_cycle(uid, parent)):
            return False
        self.by_uid(uid).parent = parent
        return True

    def depth_of(self, uid: int) -> int:
        """How many parents above the part (a loop counts once)."""
        depth, seen = 0, {uid}
        parent = self.by_uid(uid).parent
        while parent is not None and parent not in seen and self.has_uid(parent):
            depth += 1
            seen.add(parent)
            parent = self.by_uid(parent).parent
        return depth

    def ordered_parts(self) -> list:
        """(model, depth) in tree order: a root in the models' order, then its
        children, each with theirs. A part whose parent is missing is a root."""
        uids = {m.uid for m in self.models}
        children: dict = {}
        roots = []
        for m in self.models:
            if m.parent is None or m.parent not in uids or m.parent == m.uid or self.would_cycle(m.uid, m.parent):
                roots.append(m)
            else:
                children.setdefault(m.parent, []).append(m)
        out: list = []

        def walk(model, depth: int, seen: set) -> None:
            out.append((model, depth))
            for child in children.get(model.uid, []):
                if child.uid not in seen:
                    seen.add(child.uid)
                    walk(child, depth + 1, seen)

        seen = {m.uid for m in roots}
        for root in roots:
            walk(root, 0, seen)
        for m in self.models:  # a part caught in a loop still shows
            if all(entry[0] is not m for entry in out):
                out.append((m, 0))
        return out

    def new_clip_name(self) -> str:
        names = {c.name for c in self.clips}
        n = len(self.clips) + 1
        while f"Clip {n}" in names:
            n += 1
        return f"Clip {n}"

    # --- models --------------------------------------------------------------

    def add(
        self, grid: VoxelGrid, name: str = "", position: Cell | None = None, layer: int = 0
    ) -> SceneModel:
        """Append a model. Position None puts the grid centre at the origin."""
        model = self._wrap(grid, name, position, layer)
        self.models.append(model)
        return model

    def insert(self, index: int, model: SceneModel) -> None:
        model.grid.palette = self.palette
        self.models.insert(index, model)

    def remove(self, index: int) -> SceneModel:
        """Take a model out. ValueError on the last one."""
        if len(self.models) <= 1:
            raise ValueError("a scene keeps at least one model")
        return self.models.pop(index)

    def has_uid(self, uid: int) -> bool:
        return any(m.uid == uid for m in self.models)

    def index_of(self, uid: int) -> int:
        for i, model in enumerate(self.models):
            if model.uid == uid:
                return i
        raise ValueError(f"no model with uid {uid}")

    def by_uid(self, uid: int) -> SceneModel:
        return self.models[self.index_of(uid)]

    def new_uid(self) -> int:
        uid = self.next_uid
        self.next_uid += 1
        return uid

    def shown_names(self) -> set[str]:
        """Every name the Layers list shows: a model without a name shows "Model 3"."""
        return {model.display_name(k) for k, model in enumerate(self.models)}

    def unique_name(self, base: str) -> str:
        """`base`, or "base 2", "base 3" ... when the name is taken."""
        taken = self.shown_names()
        if base not in taken:
            return base
        n = 2
        while f"{base} {n}" in taken:
            n += 1
        return f"{base} {n}"

    def part_name(self, source: str) -> str:
        """The name of a part split off `source`: "source.1", or the next free
        number ("Model 1.2"); a part of a part goes one level down ("Model 1.1.1")."""
        taken = self.shown_names()
        n = 1
        while f"{source}.{n}" in taken:
            n += 1
        return f"{source}.{n}"

    def new_model_name(self) -> str:
        """The name of a new model: "Model N", N one past the count or the next free one,
        in the style of the names models without a name show."""
        taken = self.shown_names()
        n = len(self.models) + 1
        while f"Model {n}" in taken:
            n += 1
        return f"Model {n}"

    # --- queries -------------------------------------------------------------

    def hidden_layers(self) -> set[int]:
        """The ids of the layers the user hid (LAYR _hidden)."""
        return layers_core.hidden_ids(self.layers)

    def is_hidden(self, model: SceneModel) -> bool:
        """True when the model's layer is hidden: not drawn, rendered or edited."""
        return int(model.layer) in self.hidden_layers()

    def world_bounds(self) -> tuple[Cell, Cell]:
        """Inclusive AABB over every model box."""
        if not self.models:
            return (0, 0, 0), (0, 0, 0)
        return _bounds_of(self.models)

    def used_counts(self) -> np.ndarray:
        """(256,) int64: how many voxels of each index the whole scene holds,
        in every model's own grid and in all its frames."""
        counts = np.zeros(256, dtype=np.int64)
        for model in self.models:
            counts += count_colours(model.grid.voxels)
            for voxels in model.states:
                counts += count_colours(voxels)
        return counts

    # --- file data -----------------------------------------------------------

    @classmethod
    def from_data(cls, data: SceneData) -> VoxScene:
        scene = cls(
            palette=np.array(data.palette, dtype=np.uint8, copy=True),
            palette_tail=bytes(data.palette_tail),
            layers=list(data.layers) if data.layers else list(DEFAULT_LAYERS),
            extras=[e for e in data.extras if e[0] != light_core.CHUNK],
            materials=data.materials.copy(),
        )
        for cid, content in data.extras:
            if cid == light_core.CHUNK:
                try:
                    scene.light = light_core.from_chunk(content)
                except ValueError:
                    scene.light = None  # a broken light: the default one
        links: dict[int, int] = {}  # the file's link ids -> the scene's
        for model in data.models:
            grid = VoxelGrid(model.size, model.voxels, scene.palette)
            added = scene.add(grid, model.name, model.position, model.layer)
            if model.link:
                if model.link not in links:
                    links[model.link] = scene.new_link()
                added.link = links[model.link]
                added.orient = link_core.as_orient(model.orient)
            scene.add_file_frames(added, model)
        if scene.timeline.tracks:
            scene.timeline.fps = anim.VOX_FPS
            scene.timeline.length = max(anim.vox_length(t) for t in scene.timeline.tracks) or anim.VOX_FPS
        return scene

    def add_file_frames(self, added: SceneModel, model: ModelData, remap: np.ndarray | None = None) -> None:
        """Give `added` the voxel frames and the animation track of the file's
        `model`. `remap` renumbers their colours as the model's own were."""
        # An entry that is the model's own array or an earlier entry's shows
        # that frame again (a blink's open eyes; the reader shares the array).
        shows = []
        arrays: list = []  # the file's arrays of added.states, in order
        for frame, voxels in model.states:
            index = 0 if voxels is model.voxels else next(
                (i for i, a in enumerate(arrays, start=1) if a is voxels), None)
            if index is None:
                arrays.append(voxels)
                state = voxels if remap is None else remap[voxels]
                added.states.append(np.array(state, dtype=np.uint8, copy=True))
                index = len(added.states)
            shows.append((frame, index))
        track = anim.track_from_vox(added.uid, model.frames, model.state_frame, shows)
        if track is not None:
            track.name = added.display_name(self.models.index(added))
            self.timeline.tracks.append(track)

    def to_vox(self) -> SceneData:
        """The scene as a .vox file holds it: `to_data`, with only the colours
        voxels use, in your order from slot 1 on. The scene does not change."""
        data = self.to_data()
        used = self.used_counts()
        store = self.colours
        store.adopt_used(used)
        lut, palette, table = store.compact(used)
        for model in data.models:
            # One new array per old one: the writer knows a shared frame by identity.
            done = {id(model.voxels): lut[model.voxels]}
            model.states = [(frame, done.setdefault(id(voxels), lut[voxels])) for frame, voxels in model.states]
            model.voxels = done[id(model.voxels)]
        data.palette = palette
        data.materials = table
        return data

    def to_data(self, rest: bool = False) -> SceneData:
        """The scene as a `.vox` holds it. By default the active clip bakes into
        .vox frames (a `.vox` save); `rest` writes the parts at rest with
        no frames, as the Rok file's inner `.vox` must (the clips live in its
        JSON, and a reader applies a frame's turn to the voxels themselves)."""
        models = []
        for model in self.models:
            frames, state_frame, shows = anim.vox_from_track(
                None if rest else self.timeline.track_of(model.uid), len(model.states), self.timeline.fps,
                self.timeline.length
            )
            models.append(
                ModelData(
                    size=model.grid.size,
                    voxels=model.grid.voxels,
                    name=model.name,
                    position=tuple(int(p) for p in model.position),
                    layer=int(model.layer),
                    frames=frames,
                    states=[(f, model.grid.voxels if i == 0 else model.states[i - 1]) for f, i in shows],
                    state_frame=state_frame,
                    link=int(model.link),
                    orient=model.orient,
                )
            )
        return SceneData(
            models=models,
            palette=self.palette,
            palette_tail=self.palette_tail,
            layers=list(self.layers),
            extras=list(self.extras) + ([(light_core.CHUNK, light_core.to_chunk(self.light))]
                                        if self.light is not None else []),
            materials=self.materials,
        )

    def new_link(self) -> int:
        """A fresh link group id."""
        link = self.next_link
        self.next_link += 1
        return link

    @classmethod
    def single(cls, grid: VoxelGrid, position: Cell | None = None) -> VoxScene:
        """One model, default layers, no extras, an empty material table."""
        scene = cls(palette=np.array(grid.palette, dtype=np.uint8, copy=True))
        scene.add(grid, "", position)
        return scene

    # --- helpers -------------------------------------------------------------

    def _wrap(self, grid: VoxelGrid, name: str, position: Cell | None, layer: int) -> SceneModel:
        if position is None:
            position = tuple(-(int(d) // 2) for d in grid.size)
        grid.palette = self.palette
        return SceneModel(
            uid=self.new_uid(),
            name=name,
            position=tuple(int(p) for p in position),
            grid=grid,
            selection=Selection(grid.size),
            layer=int(layer),
        )


def _bounds_of(models: list[SceneModel]) -> tuple[Cell, Cell]:
    """Inclusive AABB over the world boxes of `models`. `models` must not be empty."""
    lo = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)
    for model in models:
        a, b = model.box_world()
        lo = np.minimum(lo, a)
        hi = np.maximum(hi, b)
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)


def merged(models: list[SceneModel]) -> tuple[np.ndarray, Cell]:
    """The union grid of `models` over their world bounds, later models over earlier.

    Every model's voxels are written into the result at its world place, in list
    order, so a model further down the list overwrites the cells it shares with
    one before it. Empty cells overwrite nothing. Returns (voxels, position);
    position is the world low corner. `models` must not be empty. Raises
    TooBigToMerge when the result would pass MAX_SIZE on an axis; check with
    merge_size first, before changing the scene.
    """
    if not models:
        raise ValueError("merged needs at least one model")
    lo, _hi = _bounds_of(models)
    size = merge_size(models)
    if max(size) > MAX_SIZE:
        raise TooBigToMerge(size)
    voxels = np.zeros(size, dtype=np.uint8)
    for model in models:
        x0, y0, z0 = (p - l for p, l in zip(model.position, lo))
        sx, sy, sz = model.grid.size
        filled = model.grid.voxels
        np.copyto(voxels[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz], filled, where=filled != 0)
    return voxels, lo


def count_colours(voxels: np.ndarray) -> np.ndarray:
    """(256,) int64 count per index. The native module counts 16 million cells
    in a millisecond; numpy's bincount widens them to int64 first (36 ms on a
    256-cube, felt at the end of every stroke)."""
    import os

    if not os.environ.get("ROK_PURE_COUNT"):
        try:
            import rok_native

            if hasattr(rok_native, "count_colours"):
                return np.asarray(rok_native.count_colours(np.ascontiguousarray(voxels, dtype=np.uint8)), dtype=np.int64)
        except ImportError:
            pass
    return np.bincount(np.asarray(voxels, dtype=np.uint8).ravel(), minlength=256).astype(np.int64)


class TooBigToMerge(ValueError):
    """The models span more than MAX_SIZE cells on an axis."""


def merge_size(models: list[SceneModel]) -> tuple[int, int, int]:
    """The size of the grid that merged(models) would make."""
    lo, hi = _bounds_of(models)
    return tuple(h - l + 1 for l, h in zip(lo, hi))
