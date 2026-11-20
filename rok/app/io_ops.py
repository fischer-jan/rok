"""New, Open, Save, Save As, palette import and export, and Quit.

Rok saves a `.vox` as long as a `.vox` can keep the scene. A scene with
live shaders, or with animation a `.vox` would simplify, saves as a Rok
file (`.rok`, rok/core/rokfile.py) next to the `.vox`: Save says so in
one line and asks nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from rok.core import rokfile, new_sizes
from rok.core import palette as palette_core
from rok.core import palette_io, palette_json, png
from rok.core.grid import VoxelGrid
from rok.core import material as material_core
from rok.core.material import MaterialTable
from rok.core.safe_write import write_bytes
from rok.core.scene import VoxScene
from rok.core.vox_read import VoxError, loads_scene, read_scene
from rok.core.vox_write import write_scene

from . import recent
from .opkit import (
    CONTEXT,
    BoolProp,
    EnumProp,
    SaveFile,
    OpenFile,
    IntVectorProp,
    Action,
    StringProp,
    ops,
    register,
)
from .object import apply_grid, apply_scene, document, redraw_3d_views, update_palette

ROK = ".rok"
CONFIRM_TITLE = "Discard unsaved changes?"
CONFIRM_TEXT = "Discard"
PALETTE_GLOB = "*.json;*.vox;*.png;*.gpl;*.txt;*.pal;*.hex"
# The export dialog's formats: (id, label, what it keeps).
PALETTE_FORMATS = [
    ("JSON", "JSON", "Rok's own palette file: the colours with their names, materials and favourite marks"),
    ("PNG", "PNG", "A 256 x 1 strip of the colours, as .vox editors read it"),
    ("GPL", "GPL", "A .gpl palette: the colours as RGB lines"),
    ("TXT", "TXT", "A .txt palette: the colours as AARRGGBB lines"),
    ("PAL", "PAL", "A .pal palette: the colours as RGB lines"),
    ("HEX", "HEX", "One RRGGBB line per colour"),
]

_last_new_size = new_sizes.size_of(new_sizes.DEFAULT)  # what the New dialog offers next
_PRESET_ITEMS = tuple((pid, new_sizes.label(name, size), "") for pid, name, size in new_sizes.STANDARDS)


def _preset_chosen(self, _context):
    self.size = new_sizes.size_of(self.preset)


@register
class NewAction(Action):
    command_id = "rok.new"
    title = "New"
    tooltip = "Start a new scene with one empty model: pick a standard size or type your own"

    preset = EnumProp(name="Standard", items=_PRESET_ITEMS, default=new_sizes.DEFAULT, update=_preset_chosen)
    size = IntVectorProp(
        name="Size", description="Width, depth and height in voxels", size=3,
        default=new_sizes.size_of(new_sizes.DEFAULT), min=new_sizes.MIN_SIZE, max=new_sizes.MAX_SIZE,
    )
    discard = BoolProp(default=False, options={"HIDDEN"})

    def invoke(self, context, event):
        if document.modified and not self.discard:
            return context.prefs.ask_confirm(
                _Then(lambda: ops.rok.new("INVOKE", discard=True)), event,
                title=CONFIRM_TITLE, confirm_text=CONFIRM_TEXT,
            )
        standard = new_sizes.preset_for(_last_new_size)
        if standard is not None:
            self.preset = standard
        self.size = _last_new_size
        return context.prefs.open_dialog(self, title="New", confirm_text="Create", width=460)

    def draw(self, context):
        layout = self.layout
        grid = layout.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=True, align=True)
        for pid, label, _desc in _PRESET_ITEMS:
            grid.prop_enum(self, "preset", pid, text=label)
        layout.separator()
        layout.prop(self, "size", text="Size")

    def execute(self, context):
        global _last_new_size
        size = tuple(int(v) for v in self.size)
        _last_new_size = size
        new_document(context, size)
        return {"FINISHED"}


def new_document(context, size) -> None:
    """One empty model of this size as the whole, untitled scene."""
    apply_grid(context, VoxelGrid.empty(tuple(size)))
    document.filepath = ""
    from .app import RokApp  # noqa: F401

    if CONTEXT.app is not None:
        CONTEXT.app.update_title()


class _Then(Action):
    """A stand-in operator for a confirm: runs `fn` when the answer is yes."""

    title = ""

    def __init__(self, fn) -> None:
        super().__init__()
        object.__setattr__(self, "_fn", fn)

    def execute(self, context):
        object.__getattribute__(self, "_fn")()
        return {"FINISHED"}


@register
class OpenAction(OpenFile, Action):
    command_id = "rok.open"
    title = "Open"
    tooltip = "Open a .vox file or a Rok file"

    filename_ext = ".vox"
    filter_glob = StringProp(default="*.vox;*.rok", options={"HIDDEN"})
    discard = BoolProp(default=False, options={"HIDDEN"})

    def invoke(self, context, event):
        if document.modified and not self.discard:
            return context.prefs.ask_confirm(
                _Then(lambda: ops.rok.open("INVOKE", discard=True)), event,
                title=CONFIRM_TITLE, confirm_text=CONFIRM_TEXT,
            )
        if document.filepath:
            self.filepath = str(Path(document.filepath).parent / " ")[:-1]
        return OpenFile.invoke(self, context, event)

    def execute(self, context):
        return open_path(self, context, self.filepath)


def open_path(operator, context, path: str):
    """Read `path` into the document. Reports and cancels on a bad file."""
    doc = {}
    try:
        if Path(path).suffix.lower() == ROK:
            scene, doc = rokfile.read(path)
        else:
            scene = VoxScene.from_data(read_scene(path))
        if not scene.models:
            raise VoxError("the file has no models")
    except (OSError, VoxError, ValueError) as exc:
        operator.report({"ERROR"}, f"Cannot open {Path(path).name}: {exc}")
        return {"CANCELLED"}
    active = doc.get("active") if isinstance(doc, dict) else None
    uid = scene.models[active].uid if isinstance(active, int) and 0 <= active < len(scene.models) else None
    apply_scene(context, scene, uid, settle=False)
    document.filepath = str(path)
    recent.remember(str(path))
    if CONTEXT.app is not None:
        CONTEXT.app.update_title()
    return {"FINISHED"}


@register
class SaveAsAction(SaveFile, Action):
    command_id = "rok.save_as"
    title = "Save As"
    tooltip = "Save every model to a new file: a .vox, or a Rok file that keeps everything"

    filename_ext = ".vox"
    filter_glob = StringProp(default="*.vox;*.rok", options={"HIDDEN"})

    def invoke(self, context, event):
        default = "Untitled.rok" if document.scene is not None and rokfile.needs_rok(document.scene) else "Untitled.vox"
        self.filepath = document.filepath or default
        return SaveFile.invoke(self, context, event)

    def execute(self, context):
        if document.scene is None:
            self.report({"ERROR"}, "No voxel model to save")
            return {"CANCELLED"}
        path = self.filepath
        if Path(path).suffix.lower() not in (".vox", ROK):
            path = str(Path(path).with_suffix(".vox"))
        if not _write(self, path, as_asked=True):
            return {"CANCELLED"}
        document.filepath = path
        if CONTEXT.app is not None:
            CONTEXT.app.update_title()
        return {"FINISHED"}


@register
class SaveAction(Action):
    command_id = "rok.save"
    title = "Save"
    tooltip = "Save every model to the current file"

    def execute(self, context):
        if document.scene is None:
            self.report({"ERROR"}, "No voxel model to save")
            return {"CANCELLED"}
        if not document.filepath:
            return ops.rok.save_as("INVOKE")
        path = document.filepath
        if not _write(self, path):
            return {"CANCELLED"}
        if document.filepath != path and CONTEXT.app is not None:
            CONTEXT.app.update_title()
        return {"FINISHED"}


@register
class PaletteImportAction(OpenFile, Action):
    command_id = "rok.palette_import"
    title = "Import Palette"
    tooltip = "Load colours from a .json, .vox, .png, .gpl, .txt, .pal or .hex file"

    filename_ext = ""
    filter_glob = StringProp(default=PALETTE_GLOB, options={"HIDDEN"})
    mode = EnumProp(
        name="Mode",
        description="What to do with the colours in the file",
        items=[
            ("REPLACE", "Replace colours 1..N", "The file's colours take the place of colours 1..N; voxels of those colours change with them"),
            ("APPEND", "Add as unused colours", "Add the file's colours to the unused section; no voxel changes"),
        ],
        default="REPLACE",
    )

    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def execute(self, context):
        scene = document.scene
        if scene is None:
            return {"CANCELLED"}
        try:
            incoming, count, used, materials, extras = _read_palette_file(self.filepath)
        except (OSError, VoxError, palette_io.PaletteError, KeyError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        store = scene.colours
        with_materials = Path(self.filepath).suffix.lower() in (".vox", ".json")
        names = extras.get("names", {})
        for slot in range(1, count + 1):
            colour = incoming[slot]
            if self.mode == "APPEND" and not used[slot] and np.array_equal(colour, palette_core.DEFAULT_PALETTE[slot]):
                continue  # an untouched default colour of the file
            material = materials.get(slot) if with_materials else None
            name = names.get(slot, "")
            ident = store.id_of(slot)
            if self.mode == "REPLACE" and ident is not None:
                store.edit(ident, colour, material, name=name or None)
            elif store.find(colour, material) is None:
                store.add(colour, material, name=name)
        if extras.get("favourites"):
            from . import palette_ui

            palette_ui.add_favourites(extras["favourites"])
        if not document.undo.push_colours():
            self.report({"INFO"}, "The palette already holds these colours")
            return {"CANCELLED"}
        document.modified = True
        update_palette()
        return {"FINISHED"}


@register
class PaletteExportAction(SaveFile, Action):
    """A dialog asks which colours (favourites, used, unused: any combination)
    and the format; the file picker follows. A call with a filepath skips both."""

    command_id = "rok.palette_export"
    title = "Export Palette"
    tooltip = ("Save the favourite, used or unused colours as Rok's JSON (with names, materials and "
               "favourite marks), or as .png, .gpl, .txt, .pal or .hex")

    favourites = BoolProp(name="Favourites", description="Your favourite colours, with their names", default=True)
    used = BoolProp(name="Used colours", description="Every colour that voxels of the scene hold", default=True)
    unused = BoolProp(name="Unused colours", description="The palette's colours that no voxel holds", default=False)
    format = EnumProp(name="Format", description="The file format", items=PALETTE_FORMATS, default="JSON")
    filename_ext = ".json"
    filter_glob = StringProp(default=PALETTE_GLOB, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return document.scene is not None

    def invoke(self, context, event):
        return context.prefs.open_dialog(self, title="Export Palette", confirm_text="Export")

    def draw(self, context):
        from . import palette_ui

        layout = self.layout
        scene = document.scene
        counts = scene.used_counts() if scene is not None else None
        store = scene.colours if scene is not None else None
        favs = palette_ui.favourites()
        numbers = {
            "favourites": len(favs),
            "used": len(store.used_ids(counts)) if store is not None else 0,
            "unused": len(store.unused_ids(counts)) if store is not None else 0,
        }
        layout.label("Which colours")
        for name in ("favourites", "used", "unused"):
            layout.prop(self, name, text=f"{_label_of(self, name)}  ·  {numbers[name]}")
        layout.separator()
        layout.prop(self, "format")

    def execute(self, context):
        scene = document.scene
        if scene is None:
            return {"CANCELLED"}
        if not (self.favourites or self.used or self.unused):
            self.report({"ERROR"}, "Choose at least one of Favourites, Used colours and Unused colours")
            return {"CANCELLED"}
        suffix = "." + str(self.format).lower()
        if not self.filepath:
            stem = os.path.splitext(document.filepath)[0] if document.filepath else "palette"
            self.filepath = stem + ("_palette" if document.filepath else "") + suffix
            context.prefs.choose_file(self)
            return {"RUNNING"}
        picked = Path(self.filepath).suffix.lower()
        if picked in palette_io.SUFFIXES and picked != ".vox":
            suffix = picked  # the picker's name wins over the dialog's format
        else:
            self.filepath = str(self.filepath) + suffix
        records = export_records(scene, self.favourites, self.used, self.unused)
        if not records:
            self.report({"INFO"}, "No colours to export")
            return {"CANCELLED"}
        try:
            if suffix == ".json":
                data = palette_json.dumps(records).encode("utf-8")
            else:
                palette = palette_io.colours_to_palette([r.rgba for r in records])
                data = palette_io.palette_bytes(palette, suffix, count=min(len(records), 255))
            write_bytes(self.filepath, data)
        except (OSError, palette_io.PaletteError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {len(records)} colour{'s' if len(records) != 1 else ''}")
        return {"FINISHED"}


def _label_of(op, name: str) -> str:
    prop = getattr(type(op), name)
    return str(getattr(prop, "name", "") or name.capitalize())


def export_records(scene, favourites: bool, used: bool, unused: bool) -> list[palette_json.Record]:
    """The colours to export, in strip order: the palette's entries that are
    chosen, then the favourite colours the palette does not hold (slot None)."""
    from . import palette_ui

    store = scene.colours
    counts = scene.used_counts()
    favs = palette_ui.favourites()
    fav_names = palette_ui.favourite_names()
    used_ids = set(store.used_ids(counts))
    out: list[palette_json.Record] = []
    seen: set[tuple[int, int, int]] = set()
    for ident in store.order:
        entry = store.entries[ident]
        rgb = tuple(int(c) for c in entry.rgba[:3])
        is_fav = rgb in favs
        in_use = ident in used_ids
        if not ((used and in_use) or (unused and not in_use) or (favourites and is_fav)):
            continue
        slot = store.slot_of(ident)
        seen.add(rgb)
        out.append(palette_json.Record(
            rgba=tuple(int(c) for c in entry.rgba),
            slot=slot,
            name=entry.name or (fav_names.get(palette_ui._hex(rgb), "") if is_fav else ""),
            group=entry.group,
            used=int(counts[slot]) if slot is not None else 0,
            favourite=is_fav,
            material=entry.material,
        ))
    if favourites:
        for rgb in favs:
            if rgb not in seen:
                out.append(palette_json.Record(rgba=(*rgb, 255), slot=None, name=fav_names.get(palette_ui._hex(rgb), ""),
                                               favourite=True))
    return out


@register
class QuitAction(Action):
    command_id = "rok.quit"
    title = "Quit"
    tooltip = "Close Rok, and ask first when the scene is unsaved"

    def invoke(self, context, event):
        if document.modified:
            return context.prefs.ask_confirm(self, event, title="Quit without saving?", confirm_text="Quit")
        return self.execute(context)

    def execute(self, context):
        from . import recovery

        recovery.discard()  # the user quits: the work is saved or given up
        if CONTEXT.app is not None:
            CONTEXT.app.host.quit()
        return {"FINISHED"}


def document_dict() -> dict:
    """What a Rok file keeps of the document beside the scene."""
    scene = document.scene
    active = scene.index_of(document.active_uid) if scene.has_uid(document.active_uid) else 0
    return {"active": active, "colour": int(document.colour)}


def _write(operator, path: str, as_asked: bool = False) -> bool:
    """Write the scene to `path`. A `.vox` path for a scene a `.vox` cannot
    keep writes a Rok file next to it instead, and the document follows
    (Save); `as_asked` (Save As) writes the `.vox` anyway and says what it
    leaves out. Report and return False on failure."""
    scene = document.scene
    reasons = rokfile.needs_rok(scene)
    note = ""
    if Path(path).suffix.lower() == ".vox" and reasons:
        if as_asked:
            note = f"; a .vox keeps no {' and no '.join(reasons)}"
        else:
            path = str(Path(path).with_suffix(ROK))
            note = f": it has {' and '.join(reasons)}, which a .vox cannot keep"
    try:
        if Path(path).suffix.lower() == ROK:
            rokfile.write(path, scene, document_dict())
        else:
            write_scene(path, scene.to_vox())
    except OSError as exc:
        operator.report({"ERROR"}, str(exc))
        return False
    document.filepath = path
    document.modified = False
    recent.remember(path)
    operator.report({"INFO"}, f"Saved {Path(path).name}{note}")
    redraw_3d_views()
    return True


def _read_palette_file(path: str) -> tuple[np.ndarray, int, np.ndarray, MaterialTable, dict]:
    """Read a palette file. Returns (palette, count, used, materials, extras);
    extras holds "names" (slot -> name) and "favourites" ((rgb, name) pairs)
    from a Rok .json file, else nothing."""
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        records = palette_io.read_json(palette_io._decode_text(data))
        in_palette = [r for r in records if r.slot is not None][:255]
        palette = palette_io.colours_to_palette([r.rgba for r in in_palette])
        materials = MaterialTable()
        for i, record in enumerate(in_palette, 1):
            if record.material != material_core.DEFAULT:
                materials.set(i, record.material)
        used = np.zeros(256, dtype=np.int64)
        used[1: len(in_palette) + 1] = 1
        extras = {
            "names": {i: r.name for i, r in enumerate(in_palette, 1) if r.name},
            "favourites": [(r.rgba[:3], r.name) for r in records if r.favourite],
        }
        return palette, len(in_palette), used, materials, extras
    if suffix == ".vox":
        file = loads_scene(data)
        used = np.zeros(256, dtype=np.int64)
        for model in file.models:
            used += np.bincount(model.voxels.ravel(), minlength=256)
        used[0] = 0
        if not used.any():
            used[1:] = 1
        return np.array(file.palette, dtype=np.uint8, copy=True), 255, used, file.materials, {}
    palette = palette_io.read_palette_bytes(data, suffix)
    count = _listed_count(data, suffix)
    used = np.zeros(256, dtype=np.int64)
    used[1: count + 1] = 1
    return palette, count, used, MaterialTable(), {}


def _listed_count(data: bytes, suffix: str) -> int:
    if suffix == ".png":
        height, width, _rgba = png.decode(data)
        return min(height * width, 255)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    parser = {
        ".gpl": palette_io.parse_gpl,
        ".txt": palette_io.parse_pdn_txt,
        ".pal": palette_io.parse_jasc,
        ".hex": palette_io.parse_hex,
    }[suffix]
    return min(len(parser(text)), 255)


def file_menu(app) -> list:
    from . import export_ops

    return [
        ("File", [
            ("rok.new", "New…", "CmdOrCtrl+N", -1),
            ("rok.open", "Open…", "CmdOrCtrl+O", -1),
            None,
            ("rok.save", "Save", "CmdOrCtrl+S", -1),
            ("rok.save_as", "Save As…", "CmdOrCtrl+Shift+S", -1),
            None,
            ("rok.palette_import", "Import Palette…", "", -1),
            ("rok.palette_export", "Export Palette…", "", -1),
            None,
            ("rok.export_mesh", "Export Mesh…", "CmdOrCtrl+E", -1),
            export_ops.animation_menu(),
            ("rok.render_picture", "Render Picture", "CmdOrCtrl+R", -1),
        ]),
    ]
