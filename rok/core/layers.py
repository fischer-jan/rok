"""The layers of a .vox file: named groups of models that show or hide together.

A .vox keeps them as LAYR chunks: a layer id, a dict (`_name`, `_hidden` "1"
when hidden, maybe more), and a reserved int. Rok keeps each chunk's bytes
(VoxScene.layers) so a save writes them back unchanged, and only rewrites the
dict of a layer the user shows or hides. Every model names its layer
(SceneModel.layer); a model on a hidden layer is not drawn, rendered or edited.

In Rok's UI "layer" always means this. One animated thing on the timeline is
a track (rok.core.anim), never a layer.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .vox_read import VoxError, _parse_string


@dataclass(frozen=True)
class LayerInfo:
    id: int
    name: str
    hidden: bool


def parse_layr(content: bytes) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    """(layer id, dict pairs in file order, the bytes after the dict)."""
    if len(content) < 8:
        raise VoxError("LAYR chunk too short")
    (layer_id,) = struct.unpack_from("<i", content, 0)
    pairs, end = _parse_dict_pairs(content, 4)
    return layer_id, pairs, bytes(content[end:])


def _parse_dict_pairs(content: bytes, offset: int) -> tuple[list[tuple[bytes, bytes]], int]:
    """A DICT's pairs in file order (vox_read's DICT reader returns a dict)."""
    if offset + 4 > len(content):
        raise VoxError("truncated DICT")
    (count,) = struct.unpack_from("<i", content, offset)
    if count < 0:
        raise VoxError("negative DICT count")
    offset += 4
    pairs = []
    for _ in range(count):
        key, offset = _parse_string(content, offset)
        value, offset = _parse_string(content, offset)
        pairs.append((bytes(key), bytes(value)))
    return pairs, offset


def info(content: bytes) -> LayerInfo | None:
    """The layer a LAYR chunk describes, or None for one Rok cannot read."""
    try:
        layer_id, pairs, _rest = parse_layr(content)
    except (VoxError, struct.error):
        return None
    attrs = dict(pairs)
    return LayerInfo(
        id=layer_id,
        name=attrs.get(b"_name", b"").decode("utf-8", "replace"),
        hidden=attrs.get(b"_hidden", b"0") == b"1",
    )


def infos(layers: list[bytes]) -> list[LayerInfo]:
    """Every readable layer, in file order."""
    return [i for i in (info(content) for content in layers) if i is not None]


def hidden_ids(layers: list[bytes]) -> set[int]:
    return {i.id for i in infos(layers) if i.hidden}


def with_hidden(content: bytes, hidden: bool) -> bytes:
    """The LAYR content with `_hidden` set: an existing key keeps its place and
    becomes "1" or "0", a missing one is added only when hiding. Everything
    else stays as it was."""
    layer_id, pairs, rest = parse_layr(content)
    value = b"1" if hidden else b"0"
    if any(k == b"_hidden" for k, _v in pairs):
        pairs = [(k, value if k == b"_hidden" else v) for k, v in pairs]
    elif hidden:
        pairs = pairs + [(b"_hidden", b"1")]
    else:
        return bytes(content)
    return struct.pack("<i", layer_id) + _encode_pairs(pairs) + rest


def set_hidden(layers: list[bytes], layer_id: int, hidden: bool) -> list[bytes]:
    """A copy of the list with that layer shown or hidden."""
    out = []
    for content in layers:
        i = info(content)
        out.append(with_hidden(content, hidden) if i is not None and i.id == layer_id else bytes(content))
    return out


def with_name(content: bytes, name: str) -> bytes:
    """The LAYR content with `_name` set; everything else stays as it was."""
    layer_id, pairs, rest = parse_layr(content)
    value = name.encode("utf-8")
    if any(k == b"_name" for k, _v in pairs):
        pairs = [(k, value if k == b"_name" else v) for k, v in pairs]
    else:
        pairs = [(b"_name", value)] + pairs
    return struct.pack("<i", layer_id) + _encode_pairs(pairs) + rest


def set_name(layers: list[bytes], layer_id: int, name: str) -> list[bytes]:
    """A copy of the list with that layer named `name`."""
    out = []
    for content in layers:
        i = info(content)
        out.append(with_name(content, name) if i is not None and i.id == layer_id else bytes(content))
    return out


def _encode_pairs(pairs: list[tuple[bytes, bytes]]) -> bytes:
    out = struct.pack("<i", len(pairs))
    for key, value in pairs:
        out += struct.pack("<i", len(key)) + key + struct.pack("<i", len(value)) + value
    return out
