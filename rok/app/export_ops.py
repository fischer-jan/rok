"""File > Export Mesh: every model's voxels as one mesh file for a game engine.
File > Export Animation: the clips as glTF (smooth or baked) or as a .vox
with .vox frames (rok/core/anim_export.py).

Rok writes the files itself (rok/core/mesh_export.py). The suffix you type
picks the format: .glb (the default), .obj or .ply. It uses the voxels as
they are in the grid; the section plane makes no difference. Nothing changes
the document, so there is no undo record.

Both exports first ask "Merge faces": neighbouring faces of one colour become
one rectangle (rok/core/mesh.py, build_mesh_merged), so a game engine gets
far fewer triangles. Mesh "Automatic" merges a .glb and leaves .obj and .ply
with one quad per voxel face, which is easier to edit in a modeller. The file
browser has no room for options, so a small dialog comes before it; a script
that passes `filepath` skips both.
"""

from __future__ import annotations

from pathlib import Path

from rok.core import anim_export, mesh_export, vox_write

from .object import document
from .opkit import BoolProp, EnumProp, SaveFile, Action, StringProp, register

MERGE_ITEMS = [
    ("AUTO", "Automatic", "Merge a .glb; keep one quad per face in .obj and .ply"),
    ("ON", "On", "Join neighbouring faces of one colour into rectangles: far fewer triangles"),
    ("OFF", "Off", "One quad for each visible voxel face"),
]


def ask_first(op, context):
    """Open the options dialog; its button opens the file browser (`execute` below does that)."""
    op._ask_file = True
    return context.prefs.open_dialog(op, title=op.title, confirm_text="Choose File…")


def pick_file(op, context) -> bool:
    """True when `execute` only opened the file browser: the next call writes."""
    if not getattr(op, "_ask_file", False):
        return False
    op._ask_file = False
    context.prefs.choose_file(op)
    return True


@register
class ExportMeshAction(SaveFile, Action):
    command_id = "rok.export_mesh"
    title = "Export Mesh"
    tooltip = "Save every model as one mesh file: glTF (.glb), OBJ or PLY. The suffix picks the format"

    filename_ext = ".glb"
    filter_glob = StringProp(default="*.glb;*.obj;*.ply", options={"HIDDEN"})
    merge = EnumProp(name="Merge faces", items=MERGE_ITEMS, default="AUTO")

    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def invoke(self, context, event):
        stem = Path(document.filepath).stem if document.filepath else "Untitled"
        folder = Path(document.filepath).parent if document.filepath else None
        self.filepath = str(folder / (stem + ".glb")) if folder else stem + ".glb"
        return ask_first(self, context)

    def execute(self, context):
        scene = document.scene
        if scene is None:
            self.report({"ERROR"}, "No voxel model to export")
            return {"CANCELLED"}
        if pick_file(self, context):
            return {"RUNNING"}
        path = Path(self.filepath)
        if path.suffix.lower() not in mesh_export.FORMATS:
            path = path.with_suffix(".glb")
        models = [(m.name or f"Model {i + 1}", m.position, m.grid.voxels)
                  for i, m in enumerate(scene.models) if not scene.is_hidden(m)]
        try:
            mesh_export.write(path, models, scene.palette, {"AUTO": None, "ON": True, "OFF": False}[self.merge])
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Export: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported to {path.name}")
        return {"FINISHED"}


# --- File > Export Animation ----------------------------------------------------------

ANIMATION_KINDS = (
    # (id, label, suffix, what the export simplifies)
    ("GLTF", "glTF", ".glb", anim_export.SIMPLIFIES[anim_export.SMOOTH]),
    ("GLTF_BAKED", "glTF, Baked", ".glb", anim_export.SIMPLIFIES[anim_export.BAKED]),
    ("VOX", ".vox", ".vox", "the active clip at 10 frames per second; free angles become quarter turns"),
)
_KIND = {k[0]: k for k in ANIMATION_KINDS}


def animation_menu() -> tuple:
    """File > Export Animation: one item per kind."""
    return ("Export Animation", [(f"rok.export_animation(kind='{k}')", f"{label}…", "", -1)
                                 for k, label, _s, _l in ANIMATION_KINDS] + [
        None,
        ("rok.export_pictures(kind='SPRITES')", "Sprite Sheet…", "", -1),
        ("rok.export_pictures(kind='GIF')", "GIF", "", -1),
        ("rok.export_pictures(kind='MP4')", "MP4", "", -1),
    ])


def has_animation(scene) -> bool:
    return scene is not None and (any(c.has_keys() for c in scene.clips) or any(m.states for m in scene.models))


@register
class ExportAnimationAction(SaveFile, Action):
    command_id = "rok.export_animation"
    title = "Export Animation"
    tooltip = ("Save the clips for a game engine: glTF with parts that turn, glTF baked with every frame "
                      "as voxels, or a .vox with frames")

    kind = EnumProp(name="Kind", items=[(k, label, simplifies) for k, label, _s, simplifies in ANIMATION_KINDS],
                        default="GLTF", options={"HIDDEN"})
    merge = BoolProp(name="Merge faces", description="Join neighbouring faces of one colour into rectangles: "
                         "far fewer triangles. Only for the glTF kinds", default=True)
    filename_ext = ".glb"
    filter_glob = StringProp(default="*.glb", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return has_animation(document.scene)

    def invoke(self, context, event):
        _k, label, suffix, _l = _KIND.get(self.kind, ANIMATION_KINDS[0])
        stem = Path(document.filepath).stem if document.filepath else "Untitled"
        folder = Path(document.filepath).parent if document.filepath else None
        if self.kind == "GLTF_BAKED":
            stem += "_baked"
        self.filter_glob = "*" + suffix
        self.filepath = str(folder / (stem + suffix)) if folder else stem + suffix
        if self.kind == "VOX":
            return SaveFile.invoke(self, context, event)  # a .vox has no mesh to merge
        return ask_first(self, context)

    def execute(self, context):
        scene = document.scene
        if not has_animation(scene):
            self.report({"ERROR"}, "No clip has keys and no part has frames")
            return {"CANCELLED"}
        if pick_file(self, context):
            return {"RUNNING"}
        _k, _label, suffix, simplifies = _KIND.get(self.kind, ANIMATION_KINDS[0])
        path = Path(self.filepath)
        if path.suffix.lower() != suffix:
            path = path.with_suffix(suffix)
        try:
            if self.kind == "VOX":
                vox_write.write_scene(path, scene.to_vox())
            else:
                anim_export.write(path, scene, anim_export.BAKED if self.kind == "GLTF_BAKED" else anim_export.SMOOTH,
                                  self.merge)
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"Export: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {path.name}: {simplifies}")
        return {"FINISHED"}
