"""The layers of a .vox file: read, hide and show without touching the other bytes."""

import struct


from rok.core import layers
from rok.core.grid import VoxelGrid
from rok.core.scene import VoxScene
from rok.core.undo import LayerRecord
from rok.core.vox_read import read_scene
from rok.core.vox_write import DEFAULT_LAYERS

from helpers import FIXTURES


def _layr(layer_id, pairs, reserved=-1) -> bytes:
    body = struct.pack("<ii", layer_id, len(pairs))
    for k, v in pairs:
        body += struct.pack("<i", len(k)) + k + struct.pack("<i", len(v)) + v
    return body + struct.pack("<i", reserved)


def test_roks_default_layers_are_eight_shown_ones():
    got = layers.infos(list(DEFAULT_LAYERS))
    assert [(i.id, i.name, i.hidden) for i in got] == [(n, str(n), False) for n in range(8)]


def test_the_golden_scene_layers_read():
    data = read_scene(FIXTURES / "phase3" / "two_models.vox")
    got = layers.infos(data.layers)
    assert got and all(isinstance(i.name, str) for i in got)
    assert layers.hidden_ids(data.layers) == {i.id for i in got if i.hidden}


def test_hiding_adds_the_flag_and_keeps_every_other_byte():
    content = _layr(3, [(b"_name", b"Engines"), (b"_color", b"255 0 0")], reserved=7)
    hidden = layers.with_hidden(content, True)
    assert layers.info(hidden) == layers.LayerInfo(3, "Engines", True)
    assert hidden.endswith(struct.pack("<i", 7))
    assert layers.parse_layr(hidden)[1][:2] == [(b"_name", b"Engines"), (b"_color", b"255 0 0")]
    assert layers.info(layers.with_hidden(hidden, False)).hidden is False


def test_an_existing_flag_keeps_its_place():
    content = _layr(1, [(b"_hidden", b"1"), (b"_name", b"x")])
    shown = layers.with_hidden(content, False)
    assert layers.parse_layr(shown)[1] == [(b"_hidden", b"0"), (b"_name", b"x")]


def test_showing_a_shown_layer_changes_nothing():
    content = _layr(2, [(b"_name", b"a")])
    assert layers.with_hidden(content, False) == content


def test_set_hidden_touches_only_that_layer():
    listed = list(DEFAULT_LAYERS)
    out = layers.set_hidden(listed, 5, True)
    assert out[5] != listed[5] and all(out[i] == listed[i] for i in range(8) if i != 5)
    assert layers.hidden_ids(out) == {5}


def test_an_unreadable_chunk_is_skipped_not_fatal():
    assert layers.info(b"\x01\x00") is None
    assert layers.infos([b"\x01\x00"] + list(DEFAULT_LAYERS[:1])) == [layers.LayerInfo(0, "0", False)]


def test_a_model_on_a_hidden_layer_is_hidden_and_undo_brings_it_back():
    scene = VoxScene.single(VoxelGrid.empty((4, 4, 4)))
    model = scene.models[0]
    assert not scene.is_hidden(model)
    before = list(scene.layers)
    record = LayerRecord(before, layers.set_hidden(before, 2, True), ((model.uid, 0, 2),))
    record.redo(scene)
    assert model.layer == 2 and scene.is_hidden(model)
    record.undo(scene)
    assert model.layer == 0 and not scene.is_hidden(model) and scene.layers == before
