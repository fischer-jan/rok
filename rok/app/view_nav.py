"""Frame All. Orbit, pan and zoom are the hold keys (view_ops.HoldMoveAction)."""

from __future__ import annotations

from . import object as voxel_object
from .handlers import registry
from .opkit import Action, register


@register
class VIEW3D_OT_view_all(Action):
    command_id = "view.frame_all"
    title = "Frame All"

    def execute(self, context):
        voxel_object.frame_scene(context)
        return {"FINISHED"}


registry.keymap.extend([
    ("view.frame_all", {"type": "HOME", "value": "PRESS"}, None),
])
