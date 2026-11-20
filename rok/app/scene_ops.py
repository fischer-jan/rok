"""The model list: add, add from file, duplicate, rename, delete, set active, move,
split. Plus the header popover panel and the Edit > Model submenu.

Every change to the list runs through `_scene_change`, which pushes one
SceneRecord. Rename and move change one model and push a ModelRecord.
"""

from __future__ import annotations

from .opkit import Action, ops, register

import copy
import os

import numpy as np
from .opkit import BoolProp, IntProp, IntVectorProp, StringProp
from .opkit import OpenFile

from rok.core import brick as brick_core
from rok.core import link as link_core
from rok.core import terrain
from rok.core.anim import MASK, PART
from rok.core import palette as palette_core
from rok.core.grid import VoxelGrid
from rok.core.grid_ops import MAX_SIZE
from rok.core.scene import merge_size, merged
from rok.core.select import Selection
from rok.core.undo import (
    AnimRecord,
    CompoundRecord,
    GridRecord,
    ModelRecord,
    SceneRecord,
    StrokeRecorder,
    stacked_record,
)
from rok.core.vox_read import VoxError, read_scene
from . import object as voxel_object
from .object import document
from .handlers import registry
from .ops_common import can_edit_scene, commit, redraw, refresh_views

NEW_SIZE = (32, 32, 32)
SHAPE_SIZE = (16, 16, 16)  # a new model of a shape from New's grid; the gizmo stretches it
GAP = 5  # cells between the scene and a model added next to it
TERRAIN_SHADER = "nature/terrain_tiles"  # rok/shaders/nature/terrain_tiles.txt
DUPLICATE_GAP = 2  # cells between a model and its copy


class ListRecord(SceneRecord):
    """A SceneRecord whose undo and redo mark every returning model's grid dirty.

    A model that leaves the list loses its chunk objects with its Empty. When a
    step brings it back, sync_scene rebuilds only dirty chunks, so they are marked
    here.
    """

    def undo(self, scene) -> None:
        super().undo(scene)
        _mark_returned(self.after, self.before)

    def redo(self, scene) -> None:
        super().redo(scene)
        _mark_returned(self.before, self.after)


def _mark_returned(left, entered) -> None:
    gone = {m.uid for m in left}
    for model in entered:
        if model.uid not in gone:
            model.grid.mark_all_dirty()


def _scene_change(fn, inner=None) -> SceneRecord:
    """Run `fn`, which changes `scene.models` and may set the active uid, and push
    one SceneRecord. `fn` may return the inner record (a split's source stroke).
    A deep copy of the timeline before and after `fn` goes on the record too, so a
    deleted model's track goes with it and undo brings it back (section 6.1)."""
    scene = document.scene
    before = tuple(scene.models)
    active_before = document.active_uid
    timeline_before = copy.deepcopy(scene.timeline)
    result = fn()
    if inner is None:
        inner = result
    after = tuple(scene.models)
    timeline_after = copy.deepcopy(scene.timeline)
    record = ListRecord(
        before, after, active_before, document.active_uid, inner, timeline_before, timeline_after
    )
    commit(record, serial=True, sync=True, hover=True)
    return record


def _next_x(scene) -> int:
    """The x cell just right of the whole scene, with a gap."""
    _lo, hi = scene.world_bounds()
    return hi[0] + GAP


def _file_used(data) -> np.ndarray:
    """(256,) voxel count per index over every model of a SceneData, frames too."""
    used = np.zeros(256, dtype=np.int64)
    for model in data.models:
        used += np.bincount(model.voxels.ravel(), minlength=256)
        for _frame, voxels in model.states:
            used += np.bincount(np.asarray(voxels).ravel(), minlength=256)
    used[0] = 0
    return used


def _nearest_count(incoming, incoming_used, merged, remap) -> int:
    """How many used incoming colours ended up on a slot with a different colour."""
    count = 0
    for i in range(1, 256):
        if incoming_used[i] > 0 and not np.array_equal(merged[remap[i]], incoming[i]):
            count += 1
    return count


# --- operators ---------------------------------------------------------------------


def new_grid(solid: str = "") -> VoxelGrid:
    """The grid of a new model: empty, or a shape from New's grid in the
    current colour (a Rock or a Hill lumps differently each time)."""
    if not solid:
        return VoxelGrid.empty(NEW_SIZE)
    import random

    from rok.core.palette import darker_slot
    from rok.core.shape import solid_grid

    colour = int(document.colour)
    voxels = solid_grid(solid, SHAPE_SIZE, colour, seed=random.randrange(1 << 30),
                        frame_colour=darker_slot(document.scene.palette, colour))
    return VoxelGrid(voxels.shape, voxels, document.scene.palette)


@register
class ModelAddAction(Action):
    command_id = "rok.model_add"
    title = "New Model"
    tooltip = "Add an empty 32×32×32 model next to the scene"
    solid = StringProp(name="Shape", description="A shape to fill the new model with; empty: none",
                          default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return can_edit_scene()

    def execute(self, context):
        scene = document.scene
        grid = new_grid(self.solid)

        def add():
            lo, _hi = scene.world_bounds()
            position = (_next_x(scene), -(grid.size[1] // 2), lo[2])
            # A new model joins the layer of the model being edited.
            layer = int(document.model.layer) if document.model is not None else 0
            model = scene.add(grid, scene.new_model_name(), position, layer)
            document.set_active(model.uid)

        _scene_change(add)
        # The new model is the focus: picked, and in the middle of the view.
        voxel_object.frame_model(context, document.model)
        return {"FINISHED"}


@register
class ModelAddFileAction(OpenFile, Action):
    command_id = "rok.model_add_file"
    title = "Add From File"
    tooltip = "Add every model of a .vox file next to the scene"
    filename_ext = ".vox"
    filter_glob = StringProp(default="*.vox", options={"HIDDEN"})
    merge = BoolProp(
        name="Use Nearest Colours",
        description="Colours of the file without a free place in the palette take the nearest colour you have",
        default=False, options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context):
        return can_edit_scene()

    def execute(self, context):
        scene = document.scene
        try:
            data = read_scene(self.filepath)
        except (OSError, VoxError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not data.models:
            self.report({"ERROR"}, "The file has no model")
            return {"CANCELLED"}

        incoming = np.array(data.palette, dtype=np.uint8, copy=True)
        incoming_used = _file_used(data)
        merged, remap, claimed_by = palette_core.merge_claimed(
            scene.palette, document.used_counts(), incoming, incoming_used,
            protect=document.colour, keep=document.made_unused(), materials=scene.materials,
            incoming_materials=data.materials,
        )
        nearest = _nearest_count(incoming, incoming_used, merged, remap)
        if nearest and not self.merge:
            self.merge = True  # the answer runs this action again, merging
            return context.prefs.ask_confirm(
                self, None, title="The palette is full",
                message=f"{nearest} colour{'s' if nearest != 1 else ''} of the file "
                        f"{'have' if nearest != 1 else 'has'} no free place in the palette. "
                        "Use the nearest colours you have instead?", confirm_text="Use Nearest")
        # Each colour the merge wrote carries the file's material for it.
        voxel_object.merge_colours(merged, claimed_by, data.materials)
        palette_record = None  # the colours travel with the scene record (UndoStack.track)

        # Shift in x only, so the file's own placement stays.
        file_lo_x = min(int(m.position[0]) for m in data.models)
        shift = _next_x(scene) - file_lo_x

        added = []

        def add():
            first = None
            for item in data.models:
                grid = VoxelGrid(item.size, remap[item.voxels], scene.palette)
                px, py, pz = (int(p) for p in item.position)
                model = scene.add(grid, item.name, (px + shift, py, pz), item.layer)
                scene.add_file_frames(model, item, remap)  # a blink or a walk comes along
                added.append(model)
                if first is None:
                    first = model
            document.set_active(first.uid)

        _scene_change(add, palette_record)
        voxel_object.update_palette()
        voxel_object.frame_models(context, added)  # the new models are the focus
        message = f"Added {len(data.models)} model{'s' if len(data.models) != 1 else ''}"
        if nearest:
            message += f", {nearest} colour{'s' if nearest != 1 else ''} mapped to the nearest existing colour"
        self.report({"INFO"}, message)
        return {"FINISHED"}


@register
class TerrainNewAction(Action):
    """New > Terrain: up to 5 x 5 models side by side, filled by the bundled
    terrain_tiles shader with their tile numbers, so the landscape runs on
    across every edge (core/terrain.py). One undo step with the palette."""

    command_id = "rok.terrain_new"
    title = "New Terrain"
    tooltip = "Make a landscape of up to 5 x 5 models that meet without a seam"
    @classmethod
    def poll(cls, context):
        import importlib.util

        return can_edit_scene() and importlib.util.find_spec(f"{__package__}.xs_ops") is not None

    across = IntProp(name="Tiles across", default=3, min=1, max=terrain.MAX_TILES)
    deep = IntProp(name="Tiles deep", default=3, min=1, max=terrain.MAX_TILES)
    size = IntProp(name="Tile size", description="The width and depth of each model, in voxels",
                      default=128, min=16, max=terrain.MAX_TILE_SIZE)
    height = IntProp(name="Height", description="The height of each model, in voxels",
                        default=64, min=8, max=terrain.MAX_TILE_SIZE)
    seed = IntProp(name="Seed", description="Another number, another landscape",
                      default=0, min=0, max=9999)
    plateaus = BoolProp(name="Plateaus", description="Flat steps with steep sides instead of rolling hills",
                           default=False)

    def invoke(self, context, event):
        import random

        self.seed = random.randrange(10000)  # a new landscape each time
        return context.prefs.open_dialog(self, title="New Terrain", confirm_text="Make")

    def execute(self, context):
        from . import xs_ops
        shader = xs_ops.shader_by_name(TERRAIN_SHADER)
        if shader is None:
            self.report({"ERROR"}, f"The {TERRAIN_SHADER} shader is missing")
            return {"CANCELLED"}
        scene = document.scene
        empty = np.zeros((self.size, self.size, self.height), dtype=np.uint8)
        sp = xs_ops.shader_palette_for(shader, empty, int(document.colour), scene.palette)
        free_slot = xs_ops.store_slots()  # shared by the tiles: one colour, one slot
        tiles = terrain.tile_layout(self.across, self.deep, self.size, (_next_x(scene), 0, 0))
        # The terrain is centred on y = 0 and stands on the floor (z = 0).
        shift_y = -(self.deep * self.size) // 2
        tiles = [terrain.Tile(t.tile_x, t.tile_y, (t.position[0], t.position[1] + shift_y, 0), t.name)
                 for t in tiles]
        grids = []
        writes, materials = {}, {}
        wm = context.prefs
        wm.progress_begin(0, len(tiles))
        try:
            for k, tile in enumerate(tiles):
                voxels = xs_ops.run(
                    shader, empty, colour=sp.colour, palette=sp.palette, colours=sp.run,
                    args=terrain.shader_args(shader.args, tile.tile_x, tile.tile_y, self.seed,
                                             style=1 if self.plateaus else 0),
                )
                voxels, new = sp.back(voxels, free_slot)
                writes.update(new)
                materials.update(sp.new_materials)
                grids.append(VoxelGrid(voxels.shape, voxels, scene.palette))
                wm.progress_update(k + 1)
        except xs_ops.PaletteFull as exc:
            self.report({"ERROR"}, f"Terrain: {exc}; free some colours first")
            return {"CANCELLED"}
        except xs_ops.ShaderError as exc:
            self.report({"ERROR"}, f"Terrain: {exc}")
            return {"CANCELLED"}
        finally:
            wm.progress_end()

        # Only the colours some tile uses reach the palette.
        palette_record = voxel_object.write_colours(writes, materials=materials)
        added = []

        def add():
            layer = int(document.model.layer) if document.model is not None else 0
            for tile, grid in zip(tiles, grids):
                added.append(scene.add(grid, scene.unique_name(tile.name), tile.position, layer))
            document.set_active(added[0].uid)

        _scene_change(add, palette_record)
        voxel_object.update_palette()
        voxel_object.frame_models(context, added)
        self.report({"INFO"}, f"Made a terrain of {len(added)} tiles (seed {self.seed})")
        return {"FINISHED"}


@register
class HeightmapNewAction(OpenFile, Action):
    """New > Heightmap: a model from a PNG picture seen from above. Each pixel
    is a column, as tall as the pixel is bright (core/heightmap.py)."""

    command_id = "rok.heightmap_new"
    title = "New From Heightmap"
    tooltip = "Make a model from a picture: the brighter a pixel, the taller its column"
    filename_ext = ".png"
    filter_glob = StringProp(default="*.png", options={"HIDDEN"})
    height = IntProp(name="Height", description="The height of the brightest pixel's column, in voxels",
                        default=32, min=2, max=256)
    image_colours = BoolProp(name="Picture's colours",
                                description="Each column takes the palette colour nearest its pixel; "
                                "off: all in the current colour", default=True)

    @classmethod
    def poll(cls, context):
        return can_edit_scene()

    def execute(self, context):
        from rok.core import heightmap
        from rok.core.png import PngError, decode

        try:
            with open(self.filepath, "rb") as f:
                _h, _w, rgba = decode(f.read(), max_pixels=heightmap.MAX_PIXELS)
        except (OSError, PngError) as exc:
            self.report({"ERROR"}, f"Heightmap: {exc}")
            return {"CANCELLED"}
        scene = document.scene
        small = heightmap.shrink(rgba)
        colour = heightmap.colours(small, scene.palette) if self.image_colours else int(document.colour)
        voxels = heightmap.heightmap_voxels(small, self.height, colour)
        if not voxels.any():
            self.report({"ERROR"}, "Heightmap: the picture is all transparent")
            return {"CANCELLED"}
        grid = VoxelGrid(voxels.shape, voxels, scene.palette)
        name = os.path.splitext(os.path.basename(self.filepath))[0] or "Heightmap"

        def add():
            position = (_next_x(scene), -(grid.size[1] // 2), 0)
            layer = int(document.model.layer) if document.model is not None else 0
            model = scene.add(grid, scene.unique_name(name), position, layer)
            document.set_active(model.uid)

        _scene_change(add)
        voxel_object.frame_model(context, document.model)
        self.report({"INFO"}, f"Made a {voxels.shape[0]} x {voxels.shape[1]} model from the picture")
        return {"FINISHED"}


@register
class ModelDuplicateAction(Action):
    command_id = "rok.model_duplicate"
    title = "Duplicate Model"
    tooltip = "Copy the active model and place the copy next to it"
    @classmethod
    def poll(cls, context):
        return can_edit_scene() and document.model is not None

    def execute(self, context):
        scene = document.scene
        source = document.model

        def add():
            grid = brick_core.carry(source.grid, VoxelGrid(source.grid.size, source.grid.voxels, scene.palette))
            px, py, pz = source.position
            position = (px + source.grid.size[0] + DUPLICATE_GAP, py, pz)
            name = scene.unique_name(source.name or "Model")
            model = scene.add(grid, name, position, source.layer)
            document.set_active(model.uid)

        _scene_change(add)
        return {"FINISHED"}


@register
class ModelRenameAction(Action):
    command_id = "rok.model_rename"
    title = "Rename Model"
    tooltip = "Give the active model a name"
    name = StringProp(name="Name", description="The new name of the model", default="")

    @classmethod
    def poll(cls, context):
        return can_edit_scene() and document.model is not None

    def invoke(self, context, event):
        self.name = document.model.name
        return context.prefs.open_dialog(
            self, title="Rename Model", confirm_text="Rename"
        )

    def draw(self, context):
        self.layout.prop(self, "name", text="Name")

    def execute(self, context):
        model = document.model
        name = self.name.strip()
        if name == model.name:
            return {"CANCELLED"}
        record = ModelRecord(model.uid, None, name_before=model.name, name_after=name)
        model.name = name
        timeline = document.scene.timeline
        track = timeline.track_of(model.uid)
        if track is not None:
            # The model's track carries its name; one undo step takes both back.
            before = copy.deepcopy(timeline)
            track.name = name
            record = CompoundRecord((record, AnimRecord(before, timeline)))
        document.undo.push(record)
        document.modified = True
        redraw()
        return {"FINISHED"}


@register
class ModelDeleteAction(Action):
    command_id = "rok.model_delete"
    title = "Delete Model"
    tooltip = "Remove the active model from the scene"
    @classmethod
    def poll(cls, context):
        return can_edit_scene() and document.model is not None

    def execute(self, context):
        scene = document.scene
        if len(scene.models) <= 1:
            self.report({"WARNING"}, "A scene needs one model")
            return {"CANCELLED"}
        index = scene.index_of(document.active_uid)

        def remove():
            uid = scene.models[index].uid
            scene.remove(index)
            scene.timeline.tracks = [
                t for t in scene.timeline.tracks
                if not (t.target.kind in (PART, MASK) and t.target.uid == uid)
            ]
            document.set_active(scene.models[max(index - 1, 0)].uid)

        _scene_change(remove)
        return {"FINISHED"}


@register
class ModelSetActiveAction(Action):
    command_id = "rok.model_set_active"
    title = "Edit This Model"
    tooltip = "Make this model the one the tools edit"
    uid = IntProp(name="Model", description="Which model to edit", default=0)

    @classmethod
    def poll(cls, context):
        return can_edit_scene()

    def execute(self, context):
        scene = document.scene
        try:
            scene.index_of(self.uid)
        except ValueError:
            return {"CANCELLED"}
        document.set_active(self.uid)
        refresh_views()
        return {"FINISHED"}


@register
class ModelMoveAction(Action):
    command_id = "rok.model_move"
    title = "Move Model"
    tooltip = "Move the active model by whole cells"
    offset = IntVectorProp(
        name="Offset", description="How far to move, in cells", size=3, default=(0, 0, 0)
    )

    @classmethod
    def poll(cls, context):
        return can_edit_scene() and document.model is not None

    def execute(self, context):
        model = document.model
        offset = tuple(int(v) for v in self.offset)
        if offset == (0, 0, 0):
            return {"CANCELLED"}
        before = tuple(int(p) for p in model.position)
        after = tuple(b + o for b, o in zip(before, offset))
        model.position = after
        commit(
            ModelRecord(model.uid, None, position_before=before, position_after=after),
            serial=True,
            sync=True,
            hover=True,
        )
        return {"FINISHED"}


class SplitEmpty(Exception):
    """Raised by `split_selection` when `source` has nothing to split: no
    selection, or a selection over cells that are already empty."""


def split_selection(scene, source):
    """Move `source`'s selected voxels into a new model of their own. Returns
    the source's `ModelRecord` (its stroke that erased the selected cells)
    and the new model. Raises `SplitEmpty`, with the message to report, when
    there is nothing to split; nothing is changed then. Called inside a
    `_scene_change` closure, by `ModelSplitAction` and by `rok.anim_preset`
    on a selection (Task 3.9)."""
    from .select_ops import unmasked

    selection = source.selection
    bounds = selection.bounds()
    if bounds is None:
        raise SplitEmpty("Select something first")
    picked = unmasked(selection.mask, source.uid)  # masked voxels stay in the source
    if not picked.any():
        raise SplitEmpty("Every selected voxel is masked: Option+M clears the mask")
    lo, hi = bounds
    box = tuple(slice(lo[i], hi[i] + 1) for i in range(3))
    part = np.where(picked[box], source.grid.voxels[box], 0).astype(np.uint8)
    if not part.any():
        raise SplitEmpty("The selection holds no voxels")
    index = scene.index_of(source.uid)
    name = scene.part_name(source.display_name(index))
    recorder = StrokeRecorder(source.grid)
    recorder.set_many(np.argwhere(picked), 0)
    stroke = recorder.finish()
    selection.clear()
    position = tuple(int(p) + int(o) for p, o in zip(source.position, lo))
    grid = VoxelGrid(part.shape, part, scene.palette)
    model = scene.add(grid, name, position, source.layer)
    document.set_active(model.uid)
    return ModelRecord(source.uid, stroke), model


@register
class ModelSplitAction(Action):
    command_id = "rok.model_split"
    title = "Split Selection to Model"
    tooltip = "Move the selected voxels into a new model of their own"
    @classmethod
    def poll(cls, context):
        return can_edit_scene() and document.model is not None and document.grid.bricks is None

    def execute(self, context):
        scene = document.scene
        source = document.model

        def split():
            record, _model = split_selection(scene, source)
            return record

        try:
            _scene_change(split)
        except SplitEmpty as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


@register
class MakePartAction(Action):
    command_id = "rok.make_part"
    title = "Make Part"
    tooltip = "Cut the selected voxels into a part of their own that follows this model (its parent)"
    @classmethod
    def poll(cls, context):
        return can_edit_scene() and document.model is not None and document.grid.bricks is None

    def execute(self, context):
        scene = document.scene
        source = document.model

        def split():
            record, model = split_selection(scene, source)
            model.parent = source.uid
            return record

        try:
            _scene_change(split)
        except SplitEmpty as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"{document.model.name}: a part of {source.display_name(scene.index_of(source.uid))}")
        return {"FINISHED"}


@register
class PartEmptyAction(Action):
    command_id = "rok.part_empty"
    title = "Empty Part"
    tooltip = "A part with no voxels: a root for a character, or a socket for a weapon. It follows the picked model"
    @classmethod
    def poll(cls, context):
        return can_edit_scene() and document.model is not None

    def execute(self, context):
        scene = document.scene
        source = document.model

        def add():
            names = scene.shown_names()
            n = 1
            while f"Empty {n}" in names:
                n += 1
            grid = VoxelGrid((1, 1, 1), np.zeros((1, 1, 1), dtype=np.uint8), scene.palette)
            lo, hi = source.box_world()
            position = tuple((int(a) + int(b)) // 2 for a, b in zip(lo, hi))
            model = scene.add(grid, f"Empty {n}", position, source.layer)
            model.empty = True
            model.parent = source.uid
            document.set_active(model.uid)
            return None

        _scene_change(add)
        return {"FINISHED"}


def _merged_or_report(op, models):
    """merged(models), or (None, None) with a message when the result would be
    too big. Called before the scene changes, so a refusal loses nothing."""
    size = merge_size(models)
    if max(size) > MAX_SIZE:
        op.report({"WARNING"}, f"Too big to merge: {size[0]} x {size[1]} x {size[2]} (at most {MAX_SIZE} per side)")
        return None, None
    return merged(models)


def merged_masks(models, voxels: np.ndarray, position) -> tuple[np.ndarray, np.ndarray | None]:
    """The merged voxels with every masked voxel back on top (a later model
    must not overwrite one), and the models' masks as one, or None."""
    from .select_ops import mask_of

    union = None
    for model in models:
        mask = mask_of(model.uid)
        if mask is None:
            continue
        if union is None:
            union = np.zeros(voxels.shape, dtype=bool)
        box = tuple(slice(int(p) - int(l), int(p) - int(l) + n)
                    for p, l, n in zip(model.position, position, mask.shape))
        union[box] |= mask
        voxels[box][mask] = model.grid.voxels[mask]
    return voxels, union


ASK_MAKE_UNIQUE = "ask_make_unique"  # settings.txt: "off" makes models unique without the question


def unlink_record(model) -> ModelRecord | None:
    """Take `model` out of its link group (Make unique): its voxels stay as
    they are, now its own. The record for undo; None when it was not linked."""
    if not link_core.is_linked(document.scene, model):
        return None
    record = ModelRecord(model.uid, None, link_before=model.link, link_after=0,
                         orient_before=model.orient, orient_after=link_core.IDENTITY)
    model.link, model.orient = 0, link_core.IDENTITY
    return record


@register
class ModelMakeUniqueAction(Action):
    command_id = "rok.model_make_unique"
    title = "Make Unique"
    tooltip = "Unlink the model from its linked copies: its voxels become its own"
    uid = IntProp(name="Model", description="The model; -1: the picked models", default=-1,
                     options={"SKIP_SAVE"})
    dont_ask = BoolProp(name="Don't ask again", description="Make models unique at once from now on "
                        "(Rok > Settings asks again)", default=False, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return can_edit_scene() and document.model is not None

    def _models(self) -> list:
        from . import scene_tools

        scene = document.scene
        if self.uid >= 0:
            try:
                return [scene.by_uid(self.uid)]
            except ValueError:
                return []
        return scene_tools.group_models() or [document.model]

    def invoke(self, context, event):
        """Asks first: a linked copy that turns unique stops following its copies."""
        from . import settings

        linked = [m for m in self._models() if link_core.is_linked(document.scene, m)]
        if not linked or settings.get(ASK_MAKE_UNIQUE, "on") == "off":
            return self.execute(context)
        self.dont_ask = False
        return context.prefs.open_dialog(self, width=380, title="Make Unique?", confirm_text="Make Unique")

    def draw(self, context):
        from rok.core.chrome import name_list

        scene = document.scene
        layout = self.layout
        for model in self._models():
            if not link_core.is_linked(scene, model):
                continue
            others = [m.display_name(scene.models.index(m)) for m in link_core.members(scene, model.link)
                      if m is not model]
            name = model.display_name(scene.models.index(model))
            layout.label(text=f"{name} stops sharing its shape with {name_list(others)}.")
        layout.label(text="Its voxels stay; later edits change it alone. Cmd+Z links it again.", small=True)
        layout.prop(self, "dont_ask")

    def execute(self, context):
        from . import settings

        if self.dont_ask:
            settings.put(ASK_MAKE_UNIQUE, "off")
        models = self._models()
        if not models:
            return {"CANCELLED"}
        record = stacked_record(*(unlink_record(m) for m in models))
        if record is None:
            self.report({"INFO"}, "Not a linked copy")
            return {"CANCELLED"}
        commit(record, serial=True, sync=True, hover=True)
        return {"FINISHED"}


@register
class ModelMergeAllAction(Action):
    command_id = "rok.model_merge_all"
    title = "Merge All Models"
    tooltip = "Combine every model of the scene into one, later models on top"
    @classmethod
    def poll(cls, context):
        return can_edit_scene() and len(document.scene.models) > 1

    def execute(self, context):
        scene = document.scene
        active = document.model
        name = active.name if active is not None else scene.models[0].name
        count = len(scene.models)
        layer = active.layer if active is not None else 0
        voxels, position = _merged_or_report(self, scene.models)
        if voxels is None:
            return {"CANCELLED"}

        def merge():
            scene.models = []
            model = scene.add(VoxelGrid(voxels.shape, voxels), name, position, layer)
            document.set_active(model.uid)

        _scene_change(merge)
        self.report({"INFO"}, f"Merged {count} models into one")
        return {"FINISHED"}


@register
class ModelMergeActiveAction(Action):
    command_id = "rok.model_merge_active"
    title = "Merge Into Active"
    tooltip = "Combine every model into the active one, which grows to hold them"
    @classmethod
    def poll(cls, context):
        return (can_edit_scene() and document.model is not None and len(document.scene.models) > 1
                and document.grid.bricks is None)  # the active model takes the others' voxels

    def execute(self, context):
        scene = document.scene
        active = document.model
        count = len(scene.models) - 1
        voxels, position = _merged_or_report(self, scene.models)
        if voxels is None:
            return {"CANCELLED"}
        voxels, lock = merged_masks(scene.models, voxels, position)

        def merge():
            from .select_ops import set_mask

            unlinked = unlink_record(active)  # the merge is this model's alone
            grid_before = active.grid.voxels.copy()
            position_before = active.position
            for other in [m for m in scene.models if m.uid != active.uid]:
                scene.models.remove(other)
            masked = set_mask(lock, active.uid)  # before the grid changes size
            active.grid.replace_voxels(voxels)
            active.position = position
            # The grid just changed size: the old selection mask no longer matches
            # it, same as every other size-changing operator in model_ops.py.
            active.selection = Selection(voxels.shape)
            return stacked_record(unlinked, masked, ModelRecord(
                active.uid,
                GridRecord(grid_before, voxels.copy()),
                position_before=position_before,
                position_after=position,
            ))

        _scene_change(merge)
        noun = "model" if count == 1 else "models"
        self.report({"INFO"}, f"Merged {count} {noun} into the active one")
        return {"FINISHED"}
