"""The levels: inside a model, where the tools
edit its voxels, the scene, where the models are the things, and Brick mode,
where bricks from a bin build the model (rok/app/brick_mode.py). Animate
mode is a flag on top (anim_mode.py).

Step in: double-click a model, Enter on the picked one, or a click on a model
in the Layers list. Step out: Esc with nothing selected, or the level chip at
the top of the rail. Nothing here is recorded for undo: the level is a view.
"""

from __future__ import annotations

import time

from rok.core.chrome import LEVEL_BRICK, LEVEL_MODEL, LEVEL_SCENE

from . import object as voxel_object
from .object import document
from .opkit import CONTEXT, EnumProp, Action, register
from .ops_common import _late, redraw
from .state import state


def set_level(level: str) -> bool:
    """Go to `level`. True when it changed."""
    if level == state.level or document.model is None:
        return False
    if state.level in (LEVEL_MODEL, LEVEL_BRICK):
        state.last_inside = document.active_uid
    old = state.level
    state.level_prev = state.level
    state.level = level
    state.level_since = time.monotonic()
    state.last_pick = None
    state.hover = None
    state.hover_model = None
    state.chrome_open = None
    state.preview = None
    voxel_object.apply_fade()
    from . import gizmo_ops

    gizmo_ops.refresh()
    _late("brick_mode", "level_changed")(old, level)
    # chrome and mode import this module: looked up when called.
    _late("chrome", "start_level_fade")()
    _late("mode", "rehover")()
    redraw()
    return True


def step_into_picked() -> None:
    """From the scene into the picked model (the Model cell of the level
    switch, as Enter does). After a pick of a model, from the view
    or the Layers list, Model goes into that one, not back into the
    one you were last inside. A hidden pick falls back to that one."""
    scene = document.scene
    if scene is None:
        return
    model = document.model
    if model is not None and not scene.is_hidden(model):
        step_into(model.uid)
        return
    uids = {m.uid for m in scene.models}
    if state.last_inside in uids:
        step_into(state.last_inside)


def step_into(uid: int) -> None:
    """Make model `uid` the active one and go inside it."""
    if document.scene is None:
        return
    try:
        document.scene.index_of(uid)
    except ValueError:
        return
    state.group.clear()  # inside a model there is one model
    if uid != document.active_uid:
        document.set_active(uid)
    if not set_level(LEVEL_MODEL):
        voxel_object.apply_fade()


_ACTIONS = (
    ("IN", "Step In", "Go inside the picked model and edit its voxels"),
    ("OUT", "Step Out", "Go out to the scene, where the models are the things"),
    ("TOGGLE", "Model or Scene", "Tab: from a model out to the scene, from the scene into the picked model"),
)


ESC_GUARD_S = 0.4  # after an Esc deselected, Esc steps out to the scene only after this long


@register
class LevelAction(Action):
    command_id = "rok.level"
    title = "Scene or Model"
    tooltip = "Step into the picked model, or out to the scene"
    action = EnumProp(name="Action", items=_ACTIONS, default="OUT")

    @classmethod
    def poll(cls, context):
        return document.model is not None and not state.stroke_active and not state.animating

    def execute(self, context):
        if self.action == "TOGGLE":
            # Tab: straight to the other level. A selection stays
            # in its model; Esc still clears it first.
            if state.level == LEVEL_MODEL:
                set_level(LEVEL_SCENE)
                return {"FINISHED"}
            if document.scene.is_hidden(document.model):
                self.report({"WARNING"}, "The picked model is hidden: show its layer to step in")
                return {"CANCELLED"}
            step_into(document.model.uid)
            return {"FINISHED"}
        if self.action == "IN":
            if state.level == LEVEL_BRICK:
                set_level(LEVEL_MODEL)  # Enter: from the bricks to the voxels of the same model
                return {"FINISHED"}
            if state.level != LEVEL_SCENE or document.scene.is_hidden(document.model):
                return {"PASS"}
            step_into(document.model.uid)
        else:
            if state.level == LEVEL_BRICK:
                set_level(LEVEL_SCENE)  # rok.brick_escape empties a full hand first
                return {"FINISHED"}
            if state.level != LEVEL_MODEL:
                return {"PASS"}
            selection = document.selection
            if selection is not None and not selection.is_empty():
                return {"PASS"}  # Esc clears the selection first (rok.deselect)
            from . import select_ops

            event = getattr(CONTEXT.app, "last_event", None)
            if getattr(event, "type", None) == "ESCAPE" and time.monotonic() - select_ops.last_deselect < ESC_GUARD_S:
                return {"CANCELLED"}  # a quick second Esc after a deselect stays in the model
            set_level(LEVEL_SCENE)
        return {"FINISHED"}
