"""The editor's key bindings, in their order.

There are two lists: `TOOL_KEYMAP`, the keymap of the Voxel tool, and `_SPEC`,
the Screen and Window items. Rok has one keymap (`handlers.registry.keymap`);
both lists go into it here, TOOL_KEYMAP first. An item whose action is not
registered is skipped by the dispatcher.

Notes:
- The trackpad has no keymap items: the app does the trackpad itself
  (`RokApp.navigate`).
- The timeline keys (, . 0 and the wheel over the timeline) are parked with
  the animation.
- Cmd with Z, S, O, N and Q belongs to the menu bar (menus.py). The Ctrl
  copies of those items stay here, so both work.
- Rok reports the left modifier keys only, so RIGHT_CTRL never comes.

Emulate 3 Button Mouse is off, so Option/Alt+LMB stays LEFT_MOUSE and erases. A held key
and a plain mouse move moves the view: Ctrl orbits, Cmd pans, Option zooms
(`rok.hold_move`). Space opens the star,
Rok's quick menu at the cursor (`rok.marking`).
"""

from __future__ import annotations

from .handlers import registry


# Option turns a press into erase and Shift into paint (edit.effective_tool;
# Select+Option+Shift stays SELECT and subtracts). Ctrl+LMB does not draw.
def _stroke(tool: str, **modifiers):
    return (
        "rok.stroke",
        {"type": "LEFT_MOUSE", "value": "PRESS", **modifiers},
        {"properties": [("tool", tool)]},
    )


def _key(idname: str, key: str, prop: str, value, **modifiers):
    return (
        idname,
        {"type": key, "value": "PRESS", **modifiers},
        {"properties": [(prop, value)]},
    )


def _plain(idname: str, key: str, **modifiers):
    return (idname, {"type": key, "value": "PRESS", **modifiers}, None)


TOOL_KEYMAP = (
    # Space: the star, the quick menu at the cursor. First,
    # so no plain item catches Space; repeats go to the star's own modal.
    ("rok.marking", {"type": "SPACE", "value": "PRESS"}, None),
    # The key cursor (Model level and Brick mode): the arrows step it a cell
    # on the floor, Shift with Up and Down a layer up and down; while it
    # shows, Return clicks there and Esc puts it away. No modifier, so Option
    # keeps meaning Erase. In the scene the arrows nudge models (they poll there).
    *(("rok.key_cursor_move", {"type": arrow, "value": "PRESS", "shift": shift, "repeat": True},
       {"properties": [("key", key)]})
      for arrow, shift, key in (("UP_ARROW", False, "W"), ("DOWN_ARROW", False, "S"), ("LEFT_ARROW", False, "A"),
                                ("RIGHT_ARROW", False, "D"), ("UP_ARROW", True, "E"), ("DOWN_ARROW", True, "Q"))),
    _key("rok.key_cursor_act", "ENTER", "alt", False),
    _key("rok.key_cursor_act", "ENTER", "alt", True, alt=True),
    _plain("rok.key_cursor_clear", "ESCAPE"),
    # Brick mode (they poll only there): the tools, the bin's keys and the
    # hand. Before the voxel tools' keys, which use B, G, I and the digits.
    _key("rok.brick_tool", "B", "tool", "PLACE"),
    _key("rok.brick_tool", "U", "tool", "PULL"),
    _key("rok.brick_tool", "G", "tool", "PAINT"),
    _key("rok.brick_tool", "I", "tool", "PICK"),
    *(_key("rok.brick_take", f"DIGIT_{n}", "number", n) for n in range(1, 10)),
    _plain("rok.brick_escape", "ESCAPE"),
    _key("rok.brick_turn", "R", "back", False),
    _key("rok.brick_turn", "R", "back", True, shift=True),
    # With the Cut tool in the scene, Tab turns the cutting plane.
    _key("rok.scene_tool", "TAB", "tool", "TOGGLE_AXIS"),
    # Everywhere else Tab steps through Model, Scene and Animate, Shift+Tab back.
    _key("rok.level_cycle", "TAB", "step", 1),
    _key("rok.level_cycle", "TAB", "step", -1, shift=True),
    # Animate mode (they poll only there): play, step, keys.
    _plain("rok.play", "SPACE", shift=True),
    _key("rok.frame_go", "COMMA", "step", -1),
    _key("rok.frame_go", "PERIOD", "step", 1),
    _key("rok.frame_go", "COMMA", "to_key", -1, shift=True),
    _key("rok.frame_go", "PERIOD", "to_key", 1, shift=True),
    _plain("rok.key_pose", "K"),
    _plain("rok.key_delete", "DELETE"),
    _plain("rok.key_delete", "BACKSPACE"),
    # Ctrl+LMB on a model in the scene: a drag that carves (Shift keeps the
    # cutter). Beside every model it passes.
    ("rok.scene_carve_drag", {"type": "LEFT_MOUSE", "value": "PRESS", "ctrl": True}, None),
    ("rok.scene_carve_drag", {"type": "LEFT_MOUSE", "value": "PRESS", "ctrl": True, "shift": True},
     {"properties": [("keep", True)]}),
    # A held key and a plain mouse move: Ctrl orbits, Cmd pans, Option zooms;
    # keys and clicks go through (rok.hold_move).
    ("rok.hold_move", {"type": "LEFT_CTRL", "value": "PRESS", "any": True}, {"properties": [("how", "ORBIT")]}),
    ("rok.hold_move", {"type": "CMD", "value": "PRESS", "any": True}, {"properties": [("how", "PAN")]}),
    ("rok.hold_move", {"type": "LEFT_ALT", "value": "PRESS", "any": True}, {"properties": [("how", "ZOOM")]}),
    _stroke("ACTIVE"),
    _stroke("ACTIVE", shift=True),
    _stroke("ACTIVE", alt=True),
    _stroke("ACTIVE", alt=True, shift=True),
    # Ctrl+LMB on the palette card: a swatch's swap drag.
    _stroke("ACTIVE", ctrl=True),
    (
        "rok.stroke",
        {"type": "RIGHT_MOUSE", "value": "PRESS"},
        {"properties": [("tool", "ERASE")]},
    ),
    # The tool keys. pattern_turn sits first on R: it polls True only
    # in Pattern mode, so bare R sets Erase everywhere else. Shift+G is grow
    # (not Paint). Mirror is buttons on the rail, not X/Y/Z.
    # The scene's keys come first: each polls True only in the scene, so inside
    # a model G, the arrows and the Page keys keep their meaning.
    _plain("rok.scene_drop", "G"),
    _plain("rok.scene_drop_onto", "G", shift=True),
    _key("rok.scene_tool", "C", "tool", "CUT"),
    _key("rok.scene_turn", "R", "back", False),
    _key("rok.scene_turn", "R", "back", True, shift=True),
    _plain("rok.scene_frame", "F"),
    _plain("rok.scene_merge", "J", cmd_key=True),
    _plain("rok.scene_merge", "J", ctrl=True),
    _plain("rok.scene_delete", "DELETE"),
    _plain("rok.scene_delete", "BACKSPACE"),
    _key("rok.scene_tool", "ESCAPE", "tool", "PICK"),
    # Arrows nudge along the floor; Cmd (or Ctrl) with Up and Down raise and
    # lower, as a MacBook has no Page keys. Shift: ten cells.
    *(
        ("rok.scene_nudge", {"type": key, "value": "PRESS", "shift": big, "repeat": True, **mods},
         {"properties": [("direction", direction), ("big", big)]})
        for key, direction, mods in (
            ("RIGHT_ARROW", "RIGHT", {}), ("LEFT_ARROW", "LEFT", {}), ("UP_ARROW", "UP", {}),
            ("DOWN_ARROW", "DOWN", {}),
            ("UP_ARROW", "RAISE", {"cmd_key": True}), ("DOWN_ARROW", "LOWER", {"cmd_key": True}),
            ("UP_ARROW", "RAISE", {"ctrl": True}), ("DOWN_ARROW", "LOWER", {"ctrl": True}),
        )
        for big in (False, True)
    ),
    _key("rok.pattern_turn", "R", "axis", "Z"),
    _key("rok.pattern_turn", "R", "axis", "X", shift=True),
    # N steps the stamp through Add, Carve and Holes; it polls only in Pattern mode.
    _key("rok.pattern_space", "N", "step", "NEXT"),
    _key("rok.pattern_space", "N", "step", "BACK", shift=True),
    _key("rok.set_tool", "T", "tool", "ATTACH"),
    _key("rok.set_tool", "R", "tool", "ERASE"),
    _key("rok.set_tool", "G", "tool", "PAINT"),
    _key("rok.set_tool", "Z", "tool", "SELECT"),
    _key("rok.set_mode", "V", "mode", "VOXEL"),
    _key("rok.set_mode", "F", "mode", "FACE"),
    _key("rok.set_mode", "B", "mode", "BOX"),
    _key("rok.set_mode", "L", "mode", "LINE"),
    _key("rok.set_mode", "C", "mode", "CENTER"),
    _plain("rok.fill", "F", shift=True),
    _plain("rok.cage_done", "ESCAPE"),  # polls only while the cage shows; the next Esc deselects
    _plain("rok.shader_brush_stop", "ESCAPE"),
    _plain("rok.cad_clear_edge", "ESCAPE"),  # polls only while an edge is measured
    # Esc never repeats: a held Esc deselects, and only a new press steps out.
    _plain("rok.deselect", "ESCAPE", repeat=False),
    # With nothing selected, Esc steps out to the scene; Enter steps into the
    # picked model. Each passes through at the other level.
    _key("rok.level", "ESCAPE", "action", "OUT", repeat=False),
    _plain("rok.cage_done", "ENTER"),
    _key("rok.level", "ENTER", "action", "IN"),
    _plain("rok.selection_delete", "DELETE"),
    _plain("rok.selection_delete", "BACKSPACE"),
    _plain("rok.select_all", "A", ctrl=True),
    _plain("rok.select_all", "A", cmd_key=True),
    # The screen modes, the eyedropper, the selection tools, grow and shrink.
    _key("rok.set_mode", "M", "mode", "RECT"),
    _key("rok.set_mode", "O", "mode", "LASSO"),  # O: a loop; Q is a view key
    # WASD + QE move the view (rok/core/view_keys.py): A/D orbit, W/S tilt,
    # Q/E zoom. The operator runs while the key is held; its repeats are eaten.
    *(
        ("rok.view_key", {"type": key, "value": "PRESS", "repeat": False}, {"properties": [("key", key)]})
        for key in "WASDQE"
    ),
    # The pattern brush: P is the mode. R / Shift+R are bound above, before
    # set_tool, so they poll-fail into Erase outside Pattern mode.
    _key("rok.set_mode", "P", "mode", "PATTERN"),
    _plain("rok.paint_selection", "P", shift=True),  # the selection in the current colour
    _plain("rok.mask_selection", "M", shift=True),  # the selection becomes the mask
    _plain("rok.mask_clear", "M", alt=True),
    # The section planes: the keys move the plane last touched (axis CURRENT, the
    # operator's default). Page Down cuts one layer more, Page Up one less.
    _key("rok.section", "PAGE_DOWN", "action", "DOWN"),
    _key("rok.section", "PAGE_UP", "action", "UP"),
    # A MacBook has no Page keys: Cmd or Ctrl with the arrows do the same.
    _key("rok.section", "DOWN_ARROW", "action", "DOWN", cmd_key=True),
    _key("rok.section", "UP_ARROW", "action", "UP", cmd_key=True),
    _key("rok.section", "DOWN_ARROW", "action", "DOWN", ctrl=True),
    _key("rok.section", "UP_ARROW", "action", "UP", ctrl=True),
    _plain("rok.eyedrop", "I"),
    _plain("rok.inset", "I", shift=True),  # the inset dialog for the face under the mouse
    _plain("rok.select_colour", "C", shift=True),
    _plain("rok.select_connected", "L", shift=True),
    _plain("rok.select_invert", "I", ctrl=True),
    _plain("rok.select_invert", "I", cmd_key=True),
    _key("rok.grow", "G", "dialog", False, shift=True),
    _key("rok.shrink", "D", "dialog", False, shift=True),
    # Grow and shrink the selection: Ctrl or Cmd with + and -.
    # The + is Shift+= on a US keyboard and a key of its own on a German one,
    # so Plus and Equal both count, with Shift or without (-1: any).
    *(
        _plain("rok.select_grow", key, shift=-1, **{mod: True})
        for mod in ("ctrl", "cmd_key")
        for key in ("PLUS", "EQUAL")
    ),
    *(_plain("rok.select_shrink", "MINUS", shift=-1, **{mod: True}) for mod in ("ctrl", "cmd_key")),
    # Animate mode first (they poll only there): the picked keys copy and paste.
    _plain("rok.key_copy", "C", ctrl=True),
    _plain("rok.key_copy", "C", cmd_key=True),
    _plain("rok.key_paste", "V", ctrl=True),
    _plain("rok.key_paste", "V", cmd_key=True),
    _key("rok.key_paste", "V", "mirrored", True, ctrl=True, shift=True),
    _key("rok.key_paste", "V", "mirrored", True, cmd_key=True, shift=True),
    # Copy, cut and paste of the selection. Ctrl and Cmd, as for Select All.
    _plain("rok.copy", "C", ctrl=True),
    _plain("rok.copy", "C", cmd_key=True),
    _plain("rok.cut", "X", ctrl=True),
    _plain("rok.cut", "X", cmd_key=True),
    _plain("rok.paste", "V", ctrl=True),
    _plain("rok.paste", "V", cmd_key=True),
    # Saved selections: Ctrl+1..4 saves the selection, 1..4
    # brings it back (Cmd too, as for Copy). Saved views are in View > Saved Views.
    *(
        _key("rok.selection_recall", key, "slot", slot)
        for slot, key in enumerate(("DIGIT_1", "DIGIT_2", "DIGIT_3", "DIGIT_4"), start=1)
    ),
    *(
        _key("rok.selection_save", key, "slot", slot, **{mod: True})
        for mod in ("ctrl", "cmd_key")
        for slot, key in enumerate(("DIGIT_1", "DIGIT_2", "DIGIT_3", "DIGIT_4"), start=1)
    ),
)

# The Screen and Window items. Cmd with these keys is the menu bar's; the
# Ctrl copies come here. Cmd+K has no menu item, so both copies stay.
_SPEC = (
    ("rok.undo", "Z", {"ctrl": True, "repeat": True}),
    ("rok.redo", "Z", {"ctrl": True, "shift": True, "repeat": True}),
    ("rok.save", "S", {"ctrl": True}),
    ("rok.save_as", "S", {"ctrl": True, "shift": True}),
    ("rok.open", "O", {"ctrl": True}),
    ("rok.new", "N", {"ctrl": True}),
    ("rok.quit", "Q", {"ctrl": True}),
    ("rok.command_search", "K", {"ctrl": True}),
    ("rok.preferences", "COMMA", {"ctrl": True}),
    ("rok.command_search", "K", {"cmd_key": True}),
)

SPEC_KEYMAP = tuple((idname, {"type": key, "value": "PRESS", **mods}, None) for idname, key, mods in _SPEC)

registry.keymap.extend(TOOL_KEYMAP)
registry.keymap.extend(SPEC_KEYMAP)
