"""The layers of a `.vox` scene, used as groups of models: show or hide a layer, edit a model from the list, move the edited model to a
layer. The rail's Layers flyout (drawn by chrome) runs them. rok.core.layers reads and writes the LAYR chunks."""

from __future__ import annotations

from rok.core import layers as layers_core
from rok.core.undo import LayerRecord

from . import object as voxel_object
from .object import document
from .opkit import IntProp, Action, register
from .ops_common import can_touch_model, refresh_views


def _after(record: LayerRecord) -> None:
    """Push the record, keep a visible model active, and show the change."""
    scene = document.scene
    document.undo.push(record)
    document.modified = True
    model = document.model
    if model is not None and scene.is_hidden(model):
        visible = [m for m in scene.models if not scene.is_hidden(m)]
        if visible:
            document.set_active(visible[0].uid)
    voxel_object.apply_layer_visibility()
    refresh_views()


@register
class LayerToggleAction(Action):
    command_id = "rok.layer_toggle"
    title = "Show or Hide Layer"
    tooltip = "Show or hide every model on this layer"
    layer = IntProp(name="Layer", default=0)

    @classmethod
    def poll(cls, context):
        return can_touch_model()

    def execute(self, context):
        scene = document.scene
        info = next((i for i in layers_core.infos(scene.layers) if i.id == self.layer), None)
        if info is None:
            return {"CANCELLED"}
        before = list(scene.layers)
        scene.layers = layers_core.set_hidden(before, self.layer, not info.hidden)
        _after(LayerRecord(before, list(scene.layers)))
        return {"FINISHED"}


@register
class ModelEditFromListAction(Action):
    command_id = "rok.model_edit_from_list"
    title = "Edit Model"
    tooltip = "Edit this model; a hidden layer it sits on is shown first"
    uid = IntProp(name="Model", default=0)

    @classmethod
    def poll(cls, context):
        return can_touch_model()

    def execute(self, context):
        scene = document.scene
        try:
            model = scene.by_uid(self.uid)
        except ValueError:
            return {"CANCELLED"}
        if scene.is_hidden(model):
            before = list(scene.layers)
            scene.layers = layers_core.set_hidden(before, int(model.layer), False)
            document.set_active(model.uid)
            _after(LayerRecord(before, list(scene.layers)))
        else:
            document.set_active(model.uid)
            refresh_views()
        voxel_object.frame_model(context, model)  # the clicked model is the focus
        return {"FINISHED"}


@register
class ModelToLayerAction(Action):
    command_id = "rok.model_to_layer"
    title = "Move Model to Layer"
    tooltip = "Put the model you edit on this layer"
    layer = IntProp(name="Layer", default=0)

    @classmethod
    def poll(cls, context):
        return can_touch_model() and document.model is not None

    def execute(self, context):
        scene = document.scene
        model = document.model
        if int(model.layer) == int(self.layer):
            return {"CANCELLED"}
        move = (model.uid, int(model.layer), int(self.layer))
        model.layer = int(self.layer)
        _after(LayerRecord(list(scene.layers), list(scene.layers), (move,)))
        return {"FINISHED"}
