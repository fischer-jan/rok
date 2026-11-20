"""Scene read and write: the three models, the two-model golden and hand-built graphs."""

import struct

import numpy as np
import pytest

from rok.core.grid_ops import rotation_from_byte, rotation_to_byte
from rok.core.material import DEFAULT, MaterialTable, encode_matl
from rok.core.vox_read import VoxError, loads, loads_scene, read_scene, _iter_chunks
from rok.core.vox_write import DEFAULT_LAYERS, dumps, dumps_scene, write_scene

from helpers import FIXTURES, FIXTURE_META, vox_chunk, vox_file, vox_size, vox_xyzi
from reference import _animated_file, _d, _ngrp, _nshp, _nshp_frames, _ntrn, _raw_cube, _spin_file

GOLDEN = FIXTURES / "phase3" / "two_models.vox"

MODEL_POSITIONS = {
    "robot": (-12, -15, 0),
    "bird": (-40, -34, 0),
    "car": (-20, -30, 0),
}

VALID_BYTES = [
    b for b in range(128) if (b & 3) != 3 and ((b >> 2) & 3) != 3 and (b & 3) != ((b >> 2) & 3)
]


# --- hand-built graph chunks (independent of vox_write) ----------------------
# _s, _d, _ntrn, _ngrp, _nshp, _nshp_frames, _raw_cube and _animated_file live
# in tests/reference.py; test_rokfile.py shares them too.


def _chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    content_size, children_size = struct.unpack_from("<ii", data, 12)
    start = 20 + content_size
    return list(_iter_chunks(data, start, start + children_size))


def _xyzi_set(content: bytes) -> set[bytes]:
    (count,) = struct.unpack_from("<i", content, 0)
    return {content[4 + 4 * i : 8 + 4 * i] for i in range(count)}


def _world_cells(model) -> dict[tuple[int, int, int], int]:
    cells = np.argwhere(model.voxels != 0)
    out = {}
    for c in cells:
        w = tuple(int(v) for v in c + np.asarray(model.position))
        out[w] = int(model.voxels[tuple(c)])
    return out


# --- the three models ----------------------------------------------------------


@pytest.mark.parametrize("stem", FIXTURE_META)
def test_model_writes_back_chunk_for_chunk(stem):
    data = (FIXTURES / f"{stem}.vox").read_bytes()
    out = dumps_scene(loads_scene(data))
    original = _chunks(data)
    written = _chunks(out)
    assert [cid for cid, _ in original] == [cid for cid, _ in written]
    for (cid, a), (_, b) in zip(original, written):
        if cid == b"XYZI":
            assert _xyzi_set(a) == _xyzi_set(b)
        else:
            assert a == b, cid
    assert dumps_scene(loads_scene(out)) == out


@pytest.mark.parametrize("stem", FIXTURE_META)
def test_model_position_and_passthrough(stem):
    scene = read_scene(FIXTURES / f"{stem}.vox")
    assert len(scene.models) == 1
    model = scene.models[0]
    assert model.size == FIXTURE_META[stem][0]
    assert model.position == MODEL_POSITIONS[stem]
    assert model.name == ""
    assert len(scene.layers) == 8
    assert all(len(layer) == 26 for layer in scene.layers)
    assert scene.layers == list(DEFAULT_LAYERS)
    ids = [cid for cid, _ in scene.extras]
    assert ids == [b"rOBJ"] * 13
    assert len(scene.materials) == 256
    assert [i for i, _ in scene.materials.items()] == list(range(1, 257))
    assert scene.palette_tail == b"\0\0\0\0"
    assert scene.palette[0].tolist() == [0, 0, 0, 0]


def _matl_id(content: bytes) -> int:
    return struct.unpack_from("<i", content, 0)[0]


@pytest.mark.parametrize("stem", FIXTURE_META)
def test_one_material_edit_changes_exactly_its_matl_chunk(stem):
    data = (FIXTURES / f"{stem}.vox").read_bytes()
    scene = loads_scene(data)
    metal = DEFAULT.with_type("METAL").with_value(b"_rough", 0.52)
    scene.materials.set(1, metal)
    original = _chunks(data)
    written = _chunks(dumps_scene(scene))
    assert [cid for cid, _ in original] == [cid for cid, _ in written]
    changed = []
    for (cid, a), (_, b) in zip(original, written):
        if cid == b"XYZI":
            assert _xyzi_set(a) == _xyzi_set(b)
        elif a != b:
            changed.append((cid, a, b))
    assert len(changed) == 1
    cid, a, b = changed[0]
    assert cid == b"MATL"
    assert _matl_id(a) == 1
    assert b == encode_matl(1, metal)


def test_rok_only_file_gains_256_matl_after_one_set():
    model = loads(vox_file([vox_size(2, 2, 2), vox_xyzi([(0, 0, 0, 1)])]))
    data = dumps(model)
    assert [cid for cid, _ in _chunks(data)].count(b"MATL") == 0
    scene = loads_scene(data)
    assert not scene.materials
    assert scene.materials == MaterialTable()

    glass = DEFAULT.with_type("GLASS")
    scene.materials.set(7, glass)
    out = dumps_scene(scene)
    chunks = _chunks(out)
    ids = [cid for cid, _ in chunks]
    assert ids.count(b"MATL") == 256
    assert ids[ids.index(b"RGBA") + 1 :] == [b"MATL"] * 256
    matl = [c for cid, c in chunks if cid == b"MATL"]
    assert [_matl_id(c) for c in matl] == list(range(1, 257))
    for c in matl:
        i = _matl_id(c)
        assert c == encode_matl(i, glass if i == 7 else DEFAULT)
    assert sum(1 for c in matl if c[4:] == encode_matl(0, DEFAULT)[4:]) == 255

    again = loads_scene(out)
    assert again.materials == scene.materials
    assert again.extras == []
    assert dumps_scene(again) == out


@pytest.mark.parametrize("stem", FIXTURE_META)
def test_loads_returns_the_first_model(stem):
    data = (FIXTURES / f"{stem}.vox").read_bytes()
    model = loads(data)
    scene = loads_scene(data)
    assert model.size == scene.models[0].size
    assert np.array_equal(model.voxels, scene.models[0].voxels)
    assert np.array_equal(model.palette, scene.palette)


# --- the two-model golden --------------------------------------------------------


def test_golden_loads_two_models():
    scene = read_scene(GOLDEN)
    assert [m.name for m in scene.models] == ["Robot", "Car"]
    assert [m.position for m in scene.models] == [(-40, 0, 0), (10, -5, 3)]
    assert [m.size for m in scene.models] == [(24, 30, 20), (40, 60, 14)]
    robot = read_scene(FIXTURES / "robot.vox").models[0]
    car = read_scene(FIXTURES / "car.vox").models[0]
    assert np.array_equal(scene.models[0].voxels, robot.voxels)
    assert np.array_equal(scene.models[1].voxels, car.voxels)
    assert scene.layers == list(DEFAULT_LAYERS)
    assert scene.extras == []
    assert not scene.materials


def test_golden_writes_back_byte_for_byte(tmp_path):
    data = GOLDEN.read_bytes()
    assert dumps_scene(loads_scene(data)) == data
    out = tmp_path / "again.vox"
    write_scene(out, read_scene(GOLDEN))
    assert out.read_bytes() == data


def test_golden_graph_bytes():
    expected = [
        (b"nTRN", bytes.fromhex("00000000 00000000 01000000 ffffffff ffffffff 01000000 00000000")),
        (b"nGRP", bytes.fromhex("01000000 00000000 02000000 02000000 04000000")),
        (
            b"nTRN",
            bytes.fromhex("02000000 01000000 05000000")
            + b"_name"
            + bytes.fromhex("05000000")
            + b"Robot"
            + bytes.fromhex("03000000 ffffffff 00000000 01000000 01000000 02000000")
            + b"_t"
            + bytes.fromhex("09000000")
            + b"-28 15 10",
        ),
        (b"nSHP", bytes.fromhex("03000000 00000000 01000000 00000000 00000000")),
        (
            b"nTRN",
            bytes.fromhex("04000000 01000000 05000000")
            + b"_name"
            + bytes.fromhex("03000000")
            + b"Car"
            + bytes.fromhex("05000000 ffffffff 00000000 01000000 01000000 02000000")
            + b"_t"
            + bytes.fromhex("08000000")
            + b"30 25 10",
        ),
        (b"nSHP", bytes.fromhex("05000000 00000000 01000000 01000000 00000000")),
    ]
    for i in range(8):
        layer = struct.pack("<i", i) + bytes.fromhex("01000000 05000000") + b"_name"
        layer += bytes.fromhex("01000000") + str(i).encode() + bytes.fromhex("ffffffff")
        expected.append((b"LAYR", layer))
    chunks = _chunks(GOLDEN.read_bytes())
    assert [cid for cid, _ in chunks[:4]] == [b"SIZE", b"XYZI", b"SIZE", b"XYZI"]
    assert chunks[4:18] == expected
    assert chunks[18][0] == b"RGBA"
    assert len(chunks) == 19


# --- hand-built graphs -------------------------------------------------------------


def test_group_and_model_translations_add():
    size, arr, _ = _raw_cube()
    data = vox_file(
        [
            vox_size(*size),
            vox_xyzi([(0, 0, 0, 1)]),
            _ntrn(0, 1, layer=-1),
            _ngrp(1, [2]),
            _ntrn(2, 3, frames=[[("_t", "10 0 0")]]),
            _ngrp(3, [4]),
            _ntrn(4, 5, name="inner", frames=[[("_t", "0 5 0")]]),
            _nshp(5, [0]),
        ]
    )
    scene = loads_scene(data)
    assert len(scene.models) == 1
    model = scene.models[0]
    assert model.name == "inner"
    assert model.position == (10 - 1, 5 - 2, 0 - 2)


def test_group_rotation_turns_the_child_translation():
    size, arr, voxels = _raw_cube()
    byte = 17  # M rows (0 -1 0), (1 0 0), (0 0 1): a quarter turn about z
    R_p = rotation_from_byte(byte)
    t_p = np.array([7, -3, 2])
    t_c = np.array([1, 2, 3])
    data = vox_file(
        [
            vox_size(*size),
            vox_xyzi(voxels),
            _ntrn(0, 1, layer=-1, frames=[[("_t", "7 -3 2"), ("_r", str(byte))]]),
            _ngrp(1, [2]),
            _ntrn(2, 3, frames=[[("_t", "1 2 3")]]),
            _nshp(3, [0]),
        ]
    )
    model = loads_scene(data).models[0]
    t_w = R_p @ t_c + t_p
    pivot = np.array(size) // 2
    expected = {}
    for x, y, z, i in voxels:
        w = R_p @ (np.array([x, y, z]) - pivot) + t_w
        expected[tuple(int(v) for v in w)] = i
    assert _world_cells(model) == expected


def test_every_rotated_instance_bakes_to_the_reference_cells():
    size, arr, voxels = _raw_cube()
    t = np.array([5, -4, 9])
    pivot = np.array(size) // 2
    for byte in VALID_BYTES:
        data = vox_file(
            [
                vox_size(*size),
                vox_xyzi(voxels),
                _ntrn(0, 1, layer=-1),
                _ngrp(1, [2]),
                _ntrn(2, 3, frames=[[("_r", str(byte)), ("_t", "5 -4 9")]]),
                _nshp(3, [0]),
            ]
        )
        model = loads_scene(data).models[0]
        R = rotation_from_byte(byte)
        expected = {}
        for x, y, z, i in voxels:
            w = R @ (np.array([x, y, z]) - pivot) + t
            expected[tuple(int(v) for v in w)] = i
        assert _world_cells(model) == expected, byte
        assert model.voxels.dtype == np.uint8
    assert model.voxels.flags.c_contiguous
    assert model.size == model.voxels.shape


def test_rotation_default_byte_is_identity_at_minus_half_size():
    size, arr, voxels = _raw_cube()
    data = vox_file(
        [
            vox_size(*size),
            vox_xyzi(voxels),
            _ntrn(0, 1, layer=-1),
            _ngrp(1, [2]),
            _ntrn(2, 3),
            _nshp(3, [0]),
        ]
    )
    model = loads_scene(data).models[0]
    assert model.position == (-1, -2, -2)
    assert np.array_equal(model.voxels, arr)


def test_model_listed_twice_gives_two_models():
    size, arr, voxels = _raw_cube()
    data = vox_file(
        [
            vox_size(*size),
            vox_xyzi(voxels),
            _ntrn(0, 1, layer=-1),
            _ngrp(1, [2, 4]),
            _ntrn(2, 3, name="a", frames=[[("_t", "0 0 0")]]),
            _nshp(3, [0]),
            _ntrn(4, 5, name="b", frames=[[("_t", "20 0 0")]]),
            _nshp(5, [0]),
        ]
    )
    scene = loads_scene(data)
    assert [m.name for m in scene.models] == ["a", "b"]
    assert [m.position for m in scene.models] == [(-1, -2, -2), (19, -2, -2)]
    a, b = scene.models
    assert np.array_equal(a.voxels, b.voxels)
    assert not np.shares_memory(a.voxels, b.voxels)
    a.voxels[0, 0, 0] = 99
    assert b.voxels[0, 0, 0] == 1


def test_file_without_graph_sits_at_minus_half_size():
    data = vox_file(
        [
            vox_size(4, 6, 8),
            vox_xyzi([(1, 1, 1, 3)]),
            vox_size(3, 3, 3),
            vox_xyzi([(0, 0, 0, 5)]),
        ]
    )
    scene = loads_scene(data)
    assert [m.position for m in scene.models] == [(-2, -3, -4), (-1, -1, -1)]
    assert [m.name for m in scene.models] == ["", ""]
    assert scene.layers == []
    assert scene.extras == []


def test_unreferenced_raw_model_is_appended():
    data = vox_file(
        [
            vox_size(2, 2, 2),
            vox_xyzi([(0, 0, 0, 1)]),
            vox_size(5, 5, 5),
            vox_xyzi([(4, 4, 4, 2)]),
            _ntrn(0, 1, layer=-1),
            _ngrp(1, [2]),
            _ntrn(2, 3, name="listed", frames=[[("_t", "3 3 3")]]),
            _nshp(3, [1]),
        ]
    )
    scene = loads_scene(data)
    assert [m.name for m in scene.models] == ["listed", ""]
    assert scene.models[0].size == (5, 5, 5)
    assert scene.models[0].position == (1, 1, 1)
    assert scene.models[1].size == (2, 2, 2)
    assert scene.models[1].position == (-1, -1, -1)


def test_size_without_xyzi_is_an_empty_model():
    data = vox_file([vox_size(3, 3, 3), vox_size(2, 2, 2), vox_xyzi([(1, 1, 1, 4)])])
    scene = loads_scene(data)
    assert len(scene.models) == 2
    assert int(scene.models[0].voxels.sum()) == 0
    assert scene.models[1].voxels[1, 1, 1] == 4


def test_the_first_frame_places_the_model_and_later_frames_are_kept():
    size, arr, voxels = _raw_cube()
    data = vox_file(
        [
            vox_size(*size),
            vox_xyzi(voxels),
            _ntrn(0, 1, layer=-1),
            _ngrp(1, [2]),
            _ntrn(2, 3, frames=[[("_t", "1 1 1")], [("_t", "50 50 50"), ("_r", "17")]]),
            _nshp(3, [0]),
        ]
    )
    model = loads_scene(data).models[0]
    assert model.position == (0, -1, -1)
    assert np.array_equal(model.voxels, arr)
    assert [f[:2] for f in model.frames] == [(1, 17)]  # no _f: its place in the list


def test_root_is_the_lowest_unparented_ntrn_when_node_zero_is_missing():
    size, arr, voxels = _raw_cube()
    data = vox_file(
        [
            vox_size(*size),
            vox_xyzi(voxels),
            _ntrn(10, 11, layer=-1, frames=[[("_t", "2 0 0")]]),
            _ngrp(11, [12]),
            _ntrn(12, 13, frames=[[("_t", "0 3 0")]]),
            _nshp(13, [0]),
        ]
    )
    model = loads_scene(data).models[0]
    assert model.position == (2 - 1, 3 - 2, -2)


def test_imap_is_dropped_and_extras_keep_order():
    data = vox_file(
        [
            vox_size(2, 2, 2),
            vox_xyzi([(0, 0, 0, 1)]),
            vox_chunk(b"IMAP", bytes(range(256))),
            vox_chunk(b"MATL", b"m1"),
            vox_chunk(b"rOBJ", b"o1"),
            vox_chunk(b"NOTE", b"n1"),
            vox_chunk(b"rCAM", b"c1"),
            vox_chunk(b"MATL", b"m2"),
            vox_chunk(b"ZZZZ", b"unknown"),
        ]
    )
    scene = loads_scene(data)
    assert scene.extras == [
        (b"MATL", b"m1"),
        (b"rOBJ", b"o1"),
        (b"NOTE", b"n1"),
        (b"rCAM", b"c1"),
        (b"MATL", b"m2"),
    ]
    assert not scene.materials  # m1 and m2 do not parse, so they stay extras
    out = dumps_scene(scene)
    ids = [cid for cid, _ in _chunks(out)]
    assert b"IMAP" not in ids
    assert b"ZZZZ" not in ids
    assert ids[-5:] == [b"MATL", b"rOBJ", b"NOTE", b"rCAM", b"MATL"]


def test_palette_tail_round_trips():
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[1:] = 7
    content = np.zeros((256, 4), dtype=np.uint8)
    content[:255] = palette[1:]
    content[255] = (1, 2, 3, 4)
    data = vox_file(
        [
            vox_size(1, 1, 1),
            vox_xyzi([(0, 0, 0, 1)]),
            vox_chunk(b"RGBA", content.tobytes()),
        ]
    )
    scene = loads_scene(data)
    assert scene.palette_tail == b"\x01\x02\x03\x04"
    assert np.array_equal(scene.palette, palette)
    rgba = [c for cid, c in _chunks(dumps_scene(scene)) if cid == b"RGBA"][0]
    assert rgba == content.tobytes()


def test_bad_model_id_raises():
    data = vox_file(
        [
            vox_size(2, 2, 2),
            vox_xyzi([(0, 0, 0, 1)]),
            _ntrn(0, 1, layer=-1),
            _ngrp(1, [2]),
            _ntrn(2, 3),
            _nshp(3, [4]),
        ]
    )
    with pytest.raises(VoxError):
        loads_scene(data)


def test_bad_rotation_byte_raises():
    data = vox_file(
        [
            vox_size(2, 2, 2),
            vox_xyzi([(0, 0, 0, 1)]),
            _ntrn(0, 1, layer=-1),
            _ngrp(1, [2]),
            _ntrn(2, 3, frames=[[("_r", "3")]]),
            _nshp(3, [0]),
        ]
    )
    with pytest.raises(VoxError):
        loads_scene(data)


# --- writer layout -------------------------------------------------------------


def test_written_layout_for_two_models_without_names():
    size, arr, voxels = _raw_cube()
    scene = loads_scene(
        vox_file(
            [
                vox_size(*size),
                vox_xyzi(voxels),
                vox_size(2, 2, 2),
                vox_xyzi([(1, 1, 1, 9)]),
            ]
        )
    )
    chunks = _chunks(dumps_scene(scene))
    ids = [cid for cid, _ in chunks]
    assert ids == (
        [b"SIZE", b"XYZI", b"SIZE", b"XYZI", b"nTRN", b"nGRP", b"nTRN", b"nSHP", b"nTRN", b"nSHP"]
        + [b"LAYR"] * 8
        + [b"RGBA"]
    )
    # Both models sit at -(size // 2), so t = (0, 0, 0) and the frame DICT is empty.
    assert chunks[6][1] == bytes.fromhex(
        "02000000 00000000 03000000 ffffffff 00000000 01000000 00000000"
    )
    assert chunks[8][1] == bytes.fromhex(
        "04000000 00000000 05000000 ffffffff 00000000 01000000 00000000"
    )
    assert chunks[9][1] == bytes.fromhex("05000000 00000000 01000000 01000000 00000000")
    assert len(chunks[4][1]) == 28
    assert len(chunks[7][1]) == 20


# --- layers -------------------------------------------------------------------


def _layr(layer_id: int, hidden: bool = False) -> bytes:
    pairs = [("_name", str(layer_id))] + ([("_hidden", "1")] if hidden else [])
    return vox_chunk(b"LAYR", struct.pack("<i", layer_id) + _d(pairs) + struct.pack("<i", -1))


def _two_layer_file(model_layer: int, layers) -> bytes:
    size, _arr, voxels = _raw_cube()
    return vox_file(
        [
            vox_size(*size),
            vox_xyzi(voxels),
            _ntrn(0, 1, layer=-1),
            _ngrp(1, [2]),
            _ntrn(2, 3, layer=model_layer, name="ship", frames=[[("_t", "4 0 0")]]),
            _nshp(3, [0]),
            *layers,
        ]
    )


def _read_dict(content: bytes, offset: int) -> tuple[dict[str, str], int]:
    """One .vox DICT at `offset`, decoded, and the offset just past it."""
    (count,) = struct.unpack_from("<i", content, offset)
    offset += 4
    pairs = {}
    for _ in range(count):
        (klen,) = struct.unpack_from("<i", content, offset)
        offset += 4
        key = content[offset : offset + klen].decode()
        offset += klen
        (vlen,) = struct.unpack_from("<i", content, offset)
        offset += 4
        value = content[offset : offset + vlen].decode()
        offset += vlen
        pairs[key] = value
    return pairs, offset


def _ntrn_frame_dicts(content: bytes) -> list[dict[str, str]]:
    """Every frame dict of one nTRN chunk's content, in order; the first has
    no `_f` (the file's own first frame, baked into the model's array)."""
    _attrs, offset = _read_dict(content, 4)
    _child, _reserved, _layer, num_frames = struct.unpack_from("<iiii", content, offset)
    offset += 16
    dicts = []
    for _ in range(num_frames):
        d, offset = _read_dict(content, offset)
        dicts.append(d)
    return dicts


def _model_ntrn_layers(data: bytes) -> list[int]:
    """The layer id of every model nTRN (node ids 2, 4, ...) in written order."""
    out = []
    for cid, content in _chunks(data):
        if cid != b"nTRN":
            continue
        node_id = struct.unpack_from("<i", content, 0)[0]
        if node_id == 0:
            continue
        # id, attrs DICT, child, reserved, layer: skip the DICT to reach the layer.
        offset = 4
        (count,) = struct.unpack_from("<i", content, offset)
        offset += 4
        for _ in range(count):
            for _ in range(2):
                (length,) = struct.unpack_from("<i", content, offset)
                offset += 4 + length
        out.append(struct.unpack_from("<i", content, offset + 8)[0])
    return out


def test_model_keeps_its_layer_through_a_round_trip():
    data = _two_layer_file(1, [_layr(0, hidden=True), _layr(1)])
    scene = loads_scene(data)
    assert scene.models[0].layer == 1
    written = dumps_scene(scene)
    assert _model_ntrn_layers(written) == [1]
    assert loads_scene(written).models[0].layer == 1
    # The hidden layer 0 passes through untouched; the model is not on it.
    assert [c for cid, c in _chunks(written) if cid == b"LAYR"] == scene.layers


def test_model_on_an_unlisted_layer_is_written_on_layer_zero():
    data = _two_layer_file(5, [_layr(0), _layr(1)])
    scene = loads_scene(data)
    assert scene.models[0].layer == 5
    assert _model_ntrn_layers(dumps_scene(scene)) == [0]


def test_layer_defaults_to_zero_without_a_graph():
    size, _arr, voxels = _raw_cube()
    scene = loads_scene(vox_file([vox_size(*size), vox_xyzi(voxels)]))
    assert scene.models[0].layer == 0
    assert _model_ntrn_layers(dumps_scene(scene)) == [0]


@pytest.mark.parametrize("stem", ["robot", "bird", "car"])
def test_model_models_sit_on_layer_zero(stem):
    assert read_scene(FIXTURES / f"{stem}.vox").models[0].layer == 0


# --- malformed chunk headers ---------------------------------------------------


def _header_offsets(data: bytes) -> list[int]:
    """Byte offset of each top-level chunk header inside MAIN."""
    content_size, children_size = struct.unpack_from("<ii", data, 12)
    offset = 20 + content_size
    end = offset + children_size
    offsets = []
    while offset < end:
        offsets.append(offset)
        content, children = struct.unpack_from("<ii", data, offset + 4)
        offset += 12 + content + children
    return offsets


def test_negative_content_size_raises():
    """content_size = -12 leaves the walk where it was, so it must be rejected."""
    bad = b"XXXX" + struct.pack("<ii", -12, 0)
    data = vox_file([vox_size(2, 2, 2), vox_xyzi([(0, 0, 0, 1)]), bad])
    with pytest.raises(VoxError):
        loads_scene(data)


def test_negative_children_size_raises():
    bad = b"XXXX" + struct.pack("<ii", 0, -4)
    data = vox_file([vox_size(2, 2, 2), vox_xyzi([(0, 0, 0, 1)]), bad])
    with pytest.raises(VoxError):
        loads_scene(data)


def test_negative_main_size_raises():
    body = vox_size(2, 2, 2) + vox_xyzi([(0, 0, 0, 1)])
    header = b"MAIN" + struct.pack("<ii", -8, len(body))
    data = b"VOX " + struct.pack("<i", 150) + header + body
    with pytest.raises(VoxError):
        loads_scene(data)


def test_a_negative_size_in_any_golden_header_raises():
    data = GOLDEN.read_bytes()
    offsets = _header_offsets(data)
    assert len(offsets) > 5
    for offset in offsets:
        for field in (4, 8):  # content size, then children size
            buf = bytearray(data)
            struct.pack_into("<i", buf, offset + field, -12)
            with pytest.raises(VoxError):
                loads_scene(bytes(buf))


def test_mutated_bytes_only_ever_raise_voxerror():
    """Every parse failure leaves the module as a VoxError, never struct.error."""
    import random

    data = GOLDEN.read_bytes()
    rng = random.Random(7)
    for _ in range(400):
        buf = bytearray(data)
        for _ in range(rng.randint(1, 3)):
            buf[rng.randrange(len(buf))] = rng.randrange(256)
        try:
            loads_scene(bytes(buf))
        except VoxError:
            pass



# --- The .vox animation: frames and states ------------------------------


def _mv_world(voxels, size, r_byte, t) -> dict:
    """Where a .vox player shows each voxel of a raw model in one frame."""
    R = rotation_from_byte(r_byte)
    pivot = np.asarray(size) // 2
    return {tuple(int(v) for v in R @ (np.array((x, y, z)) - pivot) + np.asarray(t)): i for x, y, z, i in voxels}


def _rok_world(model, frame) -> dict:
    """Where Rok's keys put each voxel of a model in one of its later frames."""
    number, r_byte, move = frame
    Q = rotation_from_byte(r_byte)
    pivot_b = np.asarray(model.voxels.shape) // 2
    T = np.asarray(model.position) + pivot_b + np.asarray(move)
    return {
        tuple(int(v) for v in Q @ (c - pivot_b) + T): int(model.voxels[tuple(c)])
        for c in np.argwhere(model.voxels != 0)
    }


@pytest.mark.parametrize("first_r", [4, 17, 40, 98])
def test_every_later_frame_shows_where_the_vox_frame_puts_it(first_r):
    size, arr, voxels = _raw_cube()
    frames = [
        [("_r", str(first_r)), ("_t", "3 -2 5")],
        [("_r", "17"), ("_t", "10 0 0"), ("_f", "4")],
        [("_t", "-6 7 1"), ("_f", "9")],
        [("_r", "98"), ("_t", "0 0 12"), ("_f", "15")],
    ]
    data = vox_file(
        [vox_size(*size), vox_xyzi(voxels), _ntrn(0, 1, layer=-1), _ngrp(1, [2]),
         _ntrn(2, 3, frames=frames), _nshp(3, [0])]
    )
    model = loads_scene(data).models[0]
    assert [f[0] for f in model.frames] == [4, 9, 15]
    for frame, raw in zip(model.frames, frames[1:]):
        pairs = dict(raw)
        expected = _mv_world(voxels, size, int(pairs.get("_r", "4")), [int(v) for v in pairs["_t"].split()])
        assert _rok_world(model, frame) == expected


def test_a_later_frame_under_a_turned_group_follows_the_group():
    size, arr, voxels = _raw_cube()
    data = vox_file(
        [vox_size(*size), vox_xyzi(voxels), _ntrn(0, 1, layer=-1, frames=[[("_r", "17"), ("_t", "5 5 5")]]),
         _ngrp(1, [2]), _ntrn(2, 3, frames=[[("_t", "1 2 3")], [("_r", "40"), ("_t", "4 0 0"), ("_f", "2")]]),
         _nshp(3, [0])]
    )
    model = loads_scene(data).models[0]
    R_g, t_g = rotation_from_byte(17), np.array((5, 5, 5))
    R = R_g @ rotation_from_byte(40)
    t = R_g @ np.array((4, 0, 0)) + t_g
    pivot = np.asarray(size) // 2
    expected = {tuple(int(v) for v in R @ (np.array((x, y, z)) - pivot) + t): i for x, y, z, i in voxels}
    assert _rok_world(model, model.frames[0]) == expected


def test_an_animated_shape_gives_one_model_with_its_states():
    size, arr, voxels = _raw_cube()
    other = [(1, 1, 1, 9), (2, 3, 4, 8)]
    data = vox_file(
        [vox_size(*size), vox_xyzi(voxels), vox_size(*size), vox_xyzi(other),
         _ntrn(0, 1, layer=-1), _ngrp(1, [2]), _ntrn(2, 3), _nshp_frames(3, [(0, 0), (1, 6)])]
    )
    scene = loads_scene(data)
    assert len(scene.models) == 1  # the second model is a state, not a second model
    model = scene.models[0]
    assert model.state_frame == 0 and [f for f, _v in model.states] == [6]
    state = model.states[0][1]
    assert state[1, 1, 1] == 9 and state[2, 3, 4] == 8 and np.count_nonzero(state) == 2


def test_rok_writes_the_animation_back_and_reads_it_the_same():
    first = loads_scene(_animated_file())
    again = loads_scene(dumps_scene(first))
    for a, b in zip(first.models, again.models):
        assert a.position == b.position and np.array_equal(a.voxels, b.voxels)
        assert a.frames == b.frames and a.state_frame == b.state_frame
        assert [f for f, _v in a.states] == [f for f, _v in b.states]
        assert all(np.array_equal(x, y) for (_f, x), (_g, y) in zip(a.states, b.states))
    assert again.models[1].frames == () and again.models[1].states == []


def test_a_rok_animated_file_writes_back_byte_for_byte():
    once = dumps_scene(loads_scene(_animated_file()))
    assert dumps_scene(loads_scene(once)) == once
    golden = FIXTURES / "phase0" / "animated.vox"
    assert golden.read_bytes() == once  # the golden pins the layout Rok writes


def test_the_scene_carries_the_animation_as_a_timeline():
    from rok.core import anim
    from rok.core.scene import VoxScene

    scene = VoxScene.from_data(loads_scene(_animated_file()))
    spinner, still = scene.models
    track = scene.timeline.track_of(spinner.uid)
    assert track.key_times() == [0, 5, 10]
    assert any(track.keys(name) for name in anim.TURN)
    assert [(k.time, k.value) for k in track.keys(anim.STATE)] == [(0, 0), (5, 1)]
    assert len(spinner.states) == 1 and scene.timeline.track_of(still.uid) is None
    assert scene.timeline.length == 11
    assert dumps_scene(scene.to_data()) == dumps_scene(loads_scene(_animated_file()))


def test_a_linear_rotation_bakes_to_vox_frames_at_bake_times():
    from rok.core import anim
    from rok.core.scene import VoxScene

    scene = VoxScene.from_data(loads_scene(_spin_file()))
    model = scene.models[0]
    track = anim.Track(anim.Target(anim.PART, model.uid))
    anim.set_key(track, anim.TURN_Z, anim.Key(0, 0.0, anim.LINEAR))
    anim.set_key(track, anim.TURN_Z, anim.Key(4, 360.0, anim.LINEAR))
    scene.timeline.tracks.append(track)
    scene.timeline.fps = anim.VOX_FPS
    scene.timeline.length = 4  # samples 0 to 3
    assert anim.bake_times(track, anim.ROTATION, 4, scene.timeline.fps) == [0, 1, 2, 3]

    written = dumps_scene(scene.to_data())
    model_ntrn = next(
        content
        for cid, content in _chunks(written)
        if cid == b"nTRN" and struct.unpack_from("<i", content, 0)[0] != 0
    )
    # The bake's frame 0 replaces the first dict (no separate rest dict), so
    # one dict per bake_times entry and every _f appears once.
    dicts = _ntrn_frame_dicts(model_ntrn)
    assert [d["_f"] for d in dicts] == ["0", "1", "2", "3"]
    expected_bytes = [
        rotation_to_byte(np.rint(anim.matrix_from_euler((0, 0, quarter))).astype(np.int64))
        for quarter in (0, 90, 180, 270)
    ]
    assert [int(d.get("_r", anim.IDENTITY_ROTATION)) for d in dicts] == expected_bytes

    golden = FIXTURES / "anim" / "spin.vox"
    assert golden.read_bytes() == written

    reopened = VoxScene.from_data(loads_scene(written))
    reopened_track = reopened.timeline.track_of(model.uid)
    assert [(k.time, k.value, k.ease) for k in reopened_track.keys(anim.TURN_Z)] == [
        (0, 0, anim.HOLD),
        (1, 90, anim.HOLD),
        (2, 180, anim.HOLD),
        (3, 270, anim.HOLD),
    ]

    # The bake path leaves the exact-key path's byte pins untouched.
    assert dumps_scene(loads_scene(_animated_file())) == FIXTURES.joinpath("phase0", "animated.vox").read_bytes()
    assert GOLDEN.read_bytes() == dumps_scene(read_scene(GOLDEN))


# --- graphs that name far more than they hold ---------------------------------


def _doubling_graph(levels: int) -> bytes:
    """Each nGRP lists the next nTRN twice: 2 ** levels shapes from one model."""
    chunks = [vox_size(2, 2, 2), vox_xyzi([(0, 0, 0, 1)])]
    node = 0
    for _ in range(levels):
        chunks.append(_ntrn(node, node + 1, layer=-1))
        chunks.append(_ngrp(node + 1, [node + 2, node + 2]))
        node += 2
    chunks.append(_ntrn(node, node + 1))
    chunks.append(_nshp(node + 1, [0]))
    return vox_file(chunks)


def test_doubling_graph_within_the_limit_loads():
    scene = loads_scene(_doubling_graph(4))
    assert len(scene.models) == 16


def test_doubling_graph_past_the_limit_is_a_vox_error():
    with pytest.raises(VoxError, match="scene graph too large"):
        loads_scene(_doubling_graph(20))


def test_shapes_of_a_big_model_count_against_the_cell_limit():
    from rok.core import vox_read

    chunks = [vox_size(64, 64, 64), vox_xyzi([(0, 0, 0, 1)])]
    children = list(range(2, 2 + 40))
    chunks += [_ntrn(0, 1, layer=-1), _ngrp(1, children)]
    for n in children:
        chunks.append(_nshp(n, [0]))
    data = vox_file(chunks)
    assert len(loads_scene(data).models) == 40
    old = vox_read.MAX_TOTAL_CELLS
    vox_read.MAX_TOTAL_CELLS = 30 * 64 * 64 * 64
    try:
        with pytest.raises(VoxError, match="shown too often"):
            loads_scene(data)
    finally:
        vox_read.MAX_TOTAL_CELLS = old


def test_long_node_chain_loads_without_recursion():
    chunks = [vox_size(2, 2, 2), vox_xyzi([(0, 0, 0, 1)])]
    depth = 5000
    for n in range(depth):
        chunks.append(_ntrn(2 * n, 2 * n + 1, layer=-1 if n == 0 else 0))
        chunks.append(_ngrp(2 * n + 1, [2 * n + 2]))
    chunks.append(_ntrn(2 * depth, 2 * depth + 1, name="deep"))
    chunks.append(_nshp(2 * depth + 1, [0]))
    scene = loads_scene(vox_file(chunks))
    assert [m.name for m in scene.models] == ["deep"]
