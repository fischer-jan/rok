"""Import the editor modules; each registers its actions, keymap items,
draw handlers and settings with opkit and handlers.registry.

The order matters: keymap items are tried in the order they were added."""

from __future__ import annotations

from . import export_ops, io_ops, menus, undo_ops  # noqa: F401
from . import keymap_editor  # noqa: F401 - the editor's keys first
from . import navigation  # noqa: F401 - Rok > Settings: the mouse's orbit, pan and zoom
from . import edge_wrap, gradient_ops, mode, overlay, pattern_ops, tools, view_ops  # noqa: F401
from . import looks, theme, nav_cube, quad, render_view, cad_mode, render_picture, view_panel  # noqa: F401 - the cube's modal starts late, above the hover's
# Select before Model before Scene: the menu bar shows them in this order.
from . import select_ops, model_ops, clip_ops, cage_ops, gizmo_ops, layer_ops, level_ops  # noqa: F401, I001
from . import scene_ops, scene_tools, scene_arrange  # noqa: F401, I001
from . import chrome, command, marking, palette_ui, ui, welcome  # noqa: F401  Rok's own UI
from . import history_strip  # noqa: F401 - the undo steps along the bottom (View > Show History)
from . import recovery  # noqa: F401 - the copy of unsaved work, and the question after a crash
from . import mirror_preview  # noqa: F401  Mirror Half's live result
from . import last_command, model_menu  # noqa: F401 - the star's last command; the rail's Model list and its keys
from . import xs_ops, shader_gallery, shader_panel, shader_brush, live_shaders  # noqa: F401, I001  xs shaders
from . import anim_mode, dock, onion, anim_pictures  # noqa: F401, I001  Animate mode, its timeline dock, the onion skin, its pictures
from . import steady_ops  # noqa: F401 - Steady Hand's switch and its key
from . import brick_mode  # noqa: F401 - Brick mode: the bin, the brick tools, Convert
from . import key_cursor  # noqa: F401 - Option+QWEASD steps a cell cursor; Return clicks there
from . import stroke_panel  # noqa: F401 - the Stroke Designer, its button and its curves
from . import view_nav  # noqa: F401 - the view keys, after the editor's

menus.install()
ui.install_menus()
