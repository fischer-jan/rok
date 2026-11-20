"""The shader brush: paint a shader onto the model.

The shader panel's Brush button arms it with the panel's settings and
switches to Paint in Voxel mode. Each press works out what the shader makes
of the whole model (xs_ops.compute, so the selection, the mask, the Where
filters and Feather all hold) and the stroke writes that result inside the
brush sphere (edit.ShaderStroke). Each stroke is one undo step. Esc, or a
press with another tool or mode, puts the brush away.
"""

from __future__ import annotations

from rok.core import shader_scope, xs

from .handlers import registry
from .object import document
from .opkit import Action, register

armed: dict | None = None  # {"name", "label", "kwargs"}: rok.xs_run's settings


def arm(name: str, kwargs: dict) -> None:
    global armed
    armed = {"name": name, "label": xs.label(name.rpartition("/")[2]), "kwargs": dict(kwargs)}


def stop() -> None:
    global armed
    armed = None


def label() -> str:
    return "" if armed is None else armed["label"]


def target():
    """What the armed shader makes of the active model now: an xs_ops.Outcome."""
    from . import xs_ops

    kw = armed["kwargs"]
    raw = [float(kw.get(f"a{k}", 0.0)) for k in range(xs.MAX_ARGS)]
    where = shader_scope.Where(kw.get("scope", ""), shader_scope.parse_faces(kw.get("only_on", "")),
                               shader_scope.parse_colours(kw.get("only_colours", "")),
                               int(kw.get("room", 0)), int(kw.get("feather", 0)),
                               shader_scope.parse_volume(kw.get("volume", "")), kw.get("volume_shape", shader_scope.BOX))
    return xs_ops.compute(armed["name"], raw, colours_rgb=kw.get("colours_rgb", ""), colour=int(document.colour),
                          colour_count=int(kw.get("colour_count", 1)), iterations=int(kw.get("iterations", 1)),
                          where=where, recolour=kw.get("recolour", ""))


@register
class ShaderBrushStopAction(Action):
    command_id = "rok.shader_brush_stop"
    title = "Put the Shader Brush Away"
    tooltip = "Paint paints with the current colour again"
    @classmethod
    def poll(cls, context):
        return armed is not None

    def execute(self, context):
        stop()
        self.report({"INFO"}, "Shader brush put away")
        from .ops_common import redraw

        redraw()
        return {"FINISHED"}


def _forget(_context) -> None:
    stop()  # a new file: the brush's colours and scope were the last file's


registry.load_post.append(_forget)
