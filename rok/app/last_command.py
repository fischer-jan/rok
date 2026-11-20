"""The last command, for the star's Last chip.

A command is an entry of the Cmd+K list (rok.core.chrome.COMMANDS), however
it ran: the rail, the star, Cmd+K, a key or the menu bar. It counts when its
operator finishes; a dialog's command counts when its dialog is confirmed.
The Last chip runs it again with the properties it was called with, so a
dialog opens again. Undo, redo, files, views and animation do not count.

A primitive picked in the rail counts too: the primitives
left the star for the Last chip. Its chip is the rail's own, so the star
runs it as the rail does.
"""

from __future__ import annotations

from rok.core import chrome as kit
from rok.core.chrome import COMMANDS, Command
from rok.core.shape import SOLID_ITEMS
from rok.core.star import Chip

from . import opkit
from .handlers import registry

SKIP = {
    "undo", "redo", "ui_switch", "view_save", "view_go", "selection_save", "selection_recall", "palette_grid", "palette_import", "palette_export",
    "export_mesh", "save_as", "shader_gallery", "animate", "anim_preset", "key_delete", "key_ease",
    "frame_go", "play", "stop", "loop_toggle", "record_toggle", "anchor_pick", "both_sides", "track_remove",
}

_last: tuple[str, dict] | str | None = None  # (operator id, call properties), or a rail widget id


def command_for(idname: str, props: dict) -> Command | None:
    """The Cmd+K entry an operator call is, or None."""
    prefix, _sep, cid = idname.partition(".")
    if prefix != "rok" or cid in SKIP:
        return None
    plain = None
    for c in COMMANDS:
        if c.id != cid:
            continue
        if c.props and all(props.get(k) == v for k, v in c.props):
            return c
        if not c.props and plain is None:
            plain = c
    return plain


def note(idname: str, props: dict) -> None:
    global _last
    if command_for(idname, props) is not None:
        _last = (idname, dict(props))


def note_widget(widget_id: str) -> None:
    """A primitive picked in the rail: in a model (Attach builds it) or in the scene (New)."""
    global _last
    if widget_id in kit.SOLID_OF_ID or widget_id in kit.NEW_SOLID_OF_ID:
        _last = widget_id


def _widget_chip(widget_id: str) -> Chip | None:
    from .state import state

    in_scene = state.level == kit.LEVEL_SCENE
    solid = (kit.NEW_SOLID_OF_ID if in_scene else kit.SOLID_OF_ID).get(widget_id)
    if solid is None:
        return None  # a model's primitive in the scene, or the scene's in a model
    names = {ident: name for ident, name, _d in SOLID_ITEMS}
    return Chip(widget_id, names.get(solid, solid), solid)


def current() -> Command | Chip | None:
    """The last command, when it can run here and now: a Cmd+K entry, or a primitive's chip."""
    if _last is None:
        return None
    if isinstance(_last, str):
        return _widget_chip(_last)
    idname, props = _last
    cls = opkit.REGISTRY.get(idname)
    if cls is None or not cls.poll(opkit.CONTEXT):
        return None
    return command_for(idname, props)


def run() -> None:
    if isinstance(_last, tuple) and current() is not None:
        idname, props = _last
        opkit.call(idname, "INVOKE", **props)


def _start(_context) -> None:
    global _last
    _last = None  # each start begins without one


opkit.finished.append(note)
registry.startup.append(_start)
