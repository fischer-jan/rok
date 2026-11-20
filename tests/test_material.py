"""MATL materials against the three model fixtures. The fixture bytes are the goldens:
scripts/make_models.py writes them from the .vox key text, not with rok.core.material."""

from functools import lru_cache
import struct

import numpy as np
import pytest

from rok.core.material import (
    DEFAULT,
    DEFAULT_PROPS,
    EMIT_DEFAULT_FLUX,
    EMIT_FLUX_BASE,
    KEYS,
    Material,
    MaterialTable,
    carry_materials,
    encode_matl,
    fmt,
    material_atlas,
    parse_matl,
)
from rok.core.palette import merge_claimed

from helpers import FIXTURES, model_palette_counts

MODELS = ("robot", "bird", "car")

# (diffuse, plastic, emit, glass, metal) per model, counted from the files.
TYPE_COUNTS = {
    "robot": (250, 2, 2, 1, 1),
    "bird": (248, 3, 3, 1, 1),
    "car": (243, 3, 5, 1, 4),
}


@lru_cache(maxsize=None)
def _matl_contents(stem: str) -> tuple[bytes, ...]:
    """The content bytes of every MATL chunk of a fixture, in file order.

    A throwaway chunk walker: the tests must not depend on the scene reader.
    """
    data = (FIXTURES / f"{stem}.vox").read_bytes()
    content_size, children_size = struct.unpack_from("<ii", data, 12)
    offset = 20 + content_size
    end = offset + children_size
    out = []
    while offset < end:
        cid = data[offset : offset + 4]
        size, children = struct.unpack_from("<ii", data, offset + 4)
        if cid == b"MATL":
            out.append(data[offset + 12 : offset + 12 + size])
        offset += 12 + size + children
    return tuple(out)


def _chunk(stem: str, material_id: int) -> bytes:
    content = _matl_contents(stem)[material_id - 1]
    assert struct.unpack_from("<i", content, 0)[0] == material_id
    return content


def _table(stem: str) -> MaterialTable:
    return MaterialTable(dict(parse_matl(c) for c in _matl_contents(stem)))


# --- parse and encode --------------------------------------------------------


@pytest.mark.parametrize("stem", MODELS)
def test_every_chunk_parses_and_encodes_byte_exact(stem):
    contents = _matl_contents(stem)
    assert len(contents) == 256
    ids = []
    for content in contents:
        material_id, material = parse_matl(content)
        ids.append(material_id)
        assert tuple(key for key, _ in material.props) == KEYS
        assert encode_matl(material_id, material) == content
    assert ids == list(range(1, 257))


@pytest.mark.parametrize("stem", MODELS)
def test_type_counts_and_plain_slots_are_default(stem):
    table = _table(stem)
    materials = [m for _, m in table.items()]
    counts = tuple(
        sum(1 for m in materials if m.type == kind)
        for kind in ("DIFFUSE", "PLASTIC", "EMIT", "GLASS", "METAL")
    )
    assert counts == TYPE_COUNTS[stem]
    assert all(m.type != "OTHER" for m in materials)
    for m in materials:
        assert m.is_default == (m.type == "DIFFUSE")
    assert table.get(255).is_default and table.get(256).is_default


def test_plain_slots_total_741():
    plain = sum(
        1 for stem in MODELS for _, m in _table(stem).items() if m.is_default
    )
    assert plain == 741


def test_rok_default_equals_the_vox_default():
    assert encode_matl(1, DEFAULT) == _chunk("bird", 1)
    assert DEFAULT.is_default and DEFAULT.type == "DIFFUSE"
    assert DEFAULT.weight == 1 and DEFAULT.rough == 0.1 and DEFAULT.flux == 0
    assert DEFAULT.ior == 0.3 and DEFAULT.spec == 0.5


def test_slot_values():
    robot = _table("robot")
    assert robot.get(185).type == "EMIT"
    assert robot.get(185).weight == 0.05 and robot.get(185).flux == 4
    assert robot.get(209).type == "METAL"
    assert (robot.get(209).weight, robot.get(209).rough, robot.get(209).spec) == (0.59, 0.67, 0.1)
    assert robot.get(177).type == "GLASS"
    assert robot.get(177).weight == 0.43 and robot.get(177).ior == 0.1
    bird = _table("bird")
    assert bird.get(169).type == "GLASS"
    assert (bird.get(169).weight, bird.get(169).rough, bird.get(169).ior) == (0.52, 0.04, 0.23)
    car = _table("car")
    assert car.get(173).type == "GLASS"
    assert car.get(173).text(b"_att", b"") == b"0.77"
    assert car.get(174).type == "EMIT"
    assert car.get(174).weight == 0.84 and car.get(174).flux == 4


# --- fmt ---------------------------------------------------------------------


def test_fmt_round_trips_every_fixture_value():
    for stem in MODELS:
        for content in _matl_contents(stem):
            _, material = parse_matl(content)
            for key, value in material.props:
                if key != b"_type":
                    assert fmt(float(value)) == value


def test_fmt_examples():
    assert fmt(1.0) == b"1"
    assert fmt(0.5) == b"0.5"
    assert fmt(0.05) == b"0.05"
    assert fmt(-0.5) == b"-0.5"
    assert fmt(0.999) == b"1"
    assert fmt(-0.001) == b"0"
    assert fmt(0.123) == b"0.12"
    assert fmt(4) == b"4"
    assert fmt(0) == b"0"
    assert fmt(np.float32(0.7)) == b"0.7"


# --- edited-slot goldens ------------------------------------------------------


def _golden(stem, material_id, material):
    assert encode_matl(material_id, material) == _chunk(stem, material_id)


def test_golden_robot_209_metal():
    metal = (
        DEFAULT.with_type("METAL")
        .with_value(b"_weight", 0.59)
        .with_value(b"_rough", 0.67)
        .with_value(b"_spec", 0.1)
    )
    _golden("robot", 209, metal)
    assert len(encode_matl(209, metal)) == 196


def test_golden_robot_185_light():
    light = DEFAULT.with_type("EMIT").with_value(b"_weight", 0.05).with_value(b"_flux", 4)
    _golden("robot", 185, light)


def test_golden_robot_177_glass():
    _golden(
        "robot",
        177,
        DEFAULT.with_type("GLASS").with_value(b"_weight", 0.43).with_value(b"_ior", 0.1),
    )


def test_golden_robot_169_plastic():
    _golden("robot", 169, DEFAULT.with_type("PLASTIC").with_value(b"_rough", 0.8))


def test_golden_bird_171_light_from_with_type_alone():
    _golden("bird", 171, DEFAULT.with_type("EMIT"))


def test_golden_bird_180_metal():
    _golden("bird", 180, DEFAULT.with_type("METAL").with_value(b"_rough", 0.52))


def test_golden_bird_81_plastic():
    _golden(
        "bird",
        81,
        DEFAULT.with_type("PLASTIC").with_value(b"_rough", 0.78).with_value(b"_spec_p", 0.04),
    )


def test_golden_car_169_light():
    _golden("car", 169, DEFAULT.with_type("EMIT").with_value(b"_flux", 1))


def test_golden_car_173_glass():
    glass = (
        DEFAULT.with_type("GLASS")
        .with_value(b"_weight", 0.85)
        .with_value(b"_rough", 0.81)
        .with_value(b"_ior", 0.24)
        .with_value(b"_att", 0.77)
    )
    _golden("car", 173, glass)


def test_golden_car_174_light():
    _golden(
        "car",
        174,
        DEFAULT.with_type("EMIT").with_value(b"_weight", 0.84).with_value(b"_flux", 4),
    )


def test_golden_car_95_plastic_with_weight():
    _golden("car", 95, DEFAULT.with_type("PLASTIC").with_weight(0.71))


def test_edit_and_restore_bird_169():
    original = _table("bird").get(169)
    edited = original.with_value(b"_rough", 0.5)
    assert edited != original
    assert edited.rough == 0.5 and edited.ior == 0.23
    restored = edited.with_value(b"_rough", 0.04)
    assert restored == original
    assert encode_matl(169, restored) == _chunk("bird", 169)


# --- with_type ---------------------------------------------------------------


def test_with_type_emit_sets_a_power_only_when_flux_is_zero():
    light = DEFAULT.with_type("EMIT")
    assert light.type == "EMIT" and light.flux == EMIT_DEFAULT_FLUX
    assert light.text(b"_flux") == b"2"
    strong = _table("robot").get(185).with_type("DIFFUSE").with_type("EMIT")
    assert strong.flux == 4
    assert strong == _table("robot").get(185)


def test_with_type_keeps_every_other_value():
    metal = _table("robot").get(209)
    plain = metal.with_type("DIFFUSE")
    assert plain.type == "DIFFUSE"
    assert plain.props[1:] == metal.props[1:]
    assert not plain.is_default
    assert plain.with_type("METAL") == metal


def test_with_type_other_raises():
    with pytest.raises(ValueError):
        DEFAULT.with_type("OTHER")
    with pytest.raises(ValueError):
        DEFAULT.with_type("metal")


# --- inline edge cases: what the fixtures cannot hold -------------------------


def test_blend_slot_is_other_and_keeps_its_keys():
    props = ((b"_type", b"_blend"), (b"_weight", b"0.5"), (b"_rough", b"0.2"), (b"_foo", b"7"))
    content = encode_matl(3, Material(props))
    material_id, blend = parse_matl(content)
    assert material_id == 3 and blend.type == "OTHER"
    assert encode_matl(material_id, blend) == content
    metal = blend.with_type("METAL")
    assert metal.type == "METAL"
    assert metal.props == ((b"_type", b"_metal"),) + props[1:]


def test_unknown_key_stays_in_place():
    material = Material(((b"_type", b"_metal"), (b"_foo", b"1"), (b"_rough", b"0.1")))
    edited = material.with_value(b"_rough", 0.3)
    assert edited.props == ((b"_type", b"_metal"), (b"_foo", b"1"), (b"_rough", b"0.3"))
    added = edited.with_value(b"_spec", 0.25)
    assert added.props[-1] == (b"_spec", b"0.25")


def test_weight_alias_is_read_and_written():
    material = Material(((b"_type", b"_metal"), (b"_metal", b"0.7"), (b"_rough", b"0.1")))
    assert material.weight == 0.7
    edited = material.with_weight(0.5)
    assert edited.text(b"_weight") == b"0.5"
    assert edited.text(b"_metal") == b"0.5"
    assert edited.props == (
        (b"_type", b"_metal"),
        (b"_metal", b"0.5"),
        (b"_rough", b"0.1"),
        (b"_weight", b"0.5"),
    )
    # `_weight` wins when both are present; a Plain slot ignores the alias.
    assert Material(((b"_type", b"_metal"), (b"_weight", b"0.3"), (b"_metal", b"0.7"))).weight == 0.3
    assert Material(((b"_type", b"_diffuse"), (b"_metal", b"0.7"))).weight == 1.0
    assert Material(((b"_type", b"_emit"), (b"_emit", b"0.2"))).weight == 0.2
    assert Material(((b"_type", b"_glass"), (b"_alpha", b"0.4"))).weight == 0.4
    # DEFAULT has no alias key, so with_weight never adds one.
    assert DEFAULT.with_type("METAL").with_weight(0.5).text(b"_metal") is None


def test_with_type_adds_a_missing_type_at_the_front():
    material = Material(((b"_weight", b"0.5"), (b"_rough", b"0.2")))
    assert material.type == "OTHER"
    glass = material.with_type("GLASS")
    assert glass.props == ((b"_type", b"_glass"), (b"_weight", b"0.5"), (b"_rough", b"0.2"))


def test_bad_number_text_reads_the_default():
    material = Material(((b"_type", b"_metal"), (b"_weight", b"x"), (b"_flux", b"nan")))
    assert material.weight == 1.0 and material.flux == 0
    assert material.rough == 0.1


# --- parse_matl errors -------------------------------------------------------


def test_parse_matl_rejects_malformed_content():
    with pytest.raises(ValueError):
        parse_matl(b"m1")
    with pytest.raises(ValueError):
        parse_matl(_chunk("robot", 1) + b"\0\0")
    with pytest.raises(ValueError):
        parse_matl(_chunk("robot", 1)[:-1])
    with pytest.raises(ValueError):
        parse_matl(struct.pack("<ii", 1, -1))
    with pytest.raises(ValueError):
        parse_matl(struct.pack("<iii", 1, 1, -4))
    with pytest.raises(ValueError):
        parse_matl(struct.pack("<iii", 1, 1, 99) + b"_type")
    assert parse_matl(struct.pack("<ii", 7, 0)) == (7, Material(()))


# --- MaterialTable -----------------------------------------------------------


def test_empty_table():
    table = MaterialTable()
    assert not table and len(table) == 0
    assert table.items() == []
    assert table.get(1) is DEFAULT and table.get(300) is DEFAULT
    assert table == MaterialTable() and table.copy() == table
    assert np.array_equal(table.type_codes(), np.zeros(256, dtype=np.uint8))


def test_set_on_an_empty_table_fills_256_ids_in_order():
    table = MaterialTable()
    metal = DEFAULT.with_type("METAL")
    table.set(5, metal)
    assert table and len(table) == 256
    assert [i for i, _ in table.items()] == list(range(1, 257))
    assert table.get(5) is metal
    assert all(m is DEFAULT for i, m in table.items() if i != 5)
    assert table.get(256) is DEFAULT


def test_set_on_a_full_table_keeps_the_order():
    table = _table("robot")
    before = [i for i, _ in table.items()]
    light = DEFAULT.with_type("EMIT")
    table.set(1, light)
    assert [i for i, _ in table.items()] == before
    assert table.get(1) == light
    assert table.get(209) == _table("robot").get(209)
    assert table != _table("robot")


def test_set_on_a_partial_table_appends():
    table = MaterialTable({5: DEFAULT, 9: DEFAULT, 200: DEFAULT.with_type("GLASS")})
    table.ensure_full()  # not empty, so nothing happens
    assert len(table) == 3
    table.set(7, DEFAULT.with_type("METAL"))
    assert [i for i, _ in table.items()] == [5, 9, 200, 7]
    assert table.get(7).type == "METAL"
    assert table.get(8) is DEFAULT


def test_restore_and_copy():
    table = _table("robot")
    snapshot = table.copy()
    assert snapshot == table and snapshot is not table
    table.set(185, DEFAULT)
    assert snapshot != table
    assert snapshot.get(185).type == "EMIT"
    table.restore(snapshot)
    assert table == snapshot
    assert table.get(185) is snapshot.get(185)
    empty = MaterialTable()
    table.restore(empty)
    assert not table
    assert snapshot  # the source is untouched


def test_assign_slots():
    fixture = _table("robot")
    # A fixture table into an empty one: ensure_full, then 1..255 from the fixture.
    table = MaterialTable()
    table.assign_slots(fixture)
    assert len(table) == 256
    assert [i for i, _ in table.items()] == list(range(1, 257))
    for i in range(1, 256):
        assert table.get(i) == fixture.get(i)
    assert table.get(256) is DEFAULT
    # An empty table into a fixture table: 1..255 default, id 256 kept.
    table = _table("robot")
    tail = DEFAULT.with_type("METAL")
    table.set(256, tail)
    table.assign_slots(MaterialTable())
    assert all(table.get(i) is DEFAULT for i in range(1, 256))
    assert table.get(256) is tail
    assert len(table) == 256
    # Empty into empty stays empty.
    table = MaterialTable()
    table.assign_slots(MaterialTable())
    assert not table


def test_type_codes_robot():
    codes = _table("robot").type_codes()
    assert codes.shape == (256,) and codes.dtype == np.uint8
    expected = np.zeros(256, dtype=np.uint8)
    expected[[185, 186]] = 3
    expected[209] = 1
    expected[177] = 2
    expected[[169, 193]] = 4
    assert np.array_equal(codes, expected)
    other = MaterialTable({4: Material(((b"_type", b"_blend"),)), 256: DEFAULT.with_type("METAL")})
    assert other.type_codes()[4] == 5 and other.type_codes().sum() == 5


# --- channels and the atlas --------------------------------------------------


def test_channels_rows():
    robot = _table("robot")
    rows = robot.channels()
    assert rows.shape == (256, 6) and rows.dtype == np.float32
    assert rows[0].tolist() == [0, 0, 0, 0, 0, 0]
    assert rows[1].tolist() == pytest.approx([0, 0.9, 0, 0, 1.45, 0.5])
    assert rows[209].tolist() == pytest.approx([0.59, 0.67, 0, 0, 1.45, 0.5])
    assert rows[185][3] == pytest.approx(0.05 * EMIT_FLUX_BASE ** 4, rel=1e-5)
    assert rows[185][[0, 2]].tolist() == [0, 0]
    assert rows[185][1] == pytest.approx(0.9)
    assert rows[177][2] == pytest.approx(0.43)
    assert rows[177][4] == pytest.approx(1.1)
    assert rows[177][1] == pytest.approx(0.1)
    assert _table("car").channels()[95][5] == pytest.approx(0.71)
    # An empty table is plain everywhere but slot 0.
    empty = MaterialTable().channels()
    assert empty[0].tolist() == [0, 0, 0, 0, 0, 0]
    assert np.allclose(empty[1:], [0, 0.9, 0, 0, 1.45, 0.5])


def test_channels_emit_and_glass_edges():
    table = MaterialTable()
    table.set(1, DEFAULT.with_type("EMIT"))
    table.set(2, DEFAULT.with_type("EMIT").with_value(b"_flux", 0).with_weight(0.5))
    table.set(3, DEFAULT.with_type("GLASS").with_value(b"_ior", 1.5))
    table.set(4, DEFAULT.with_type("GLASS").with_value(b"_ior", -0.5))
    rows = table.channels()
    assert rows[1][3] == pytest.approx(EMIT_FLUX_BASE ** EMIT_DEFAULT_FLUX)
    assert rows[2][3] == pytest.approx(0.5)
    assert rows[3][4] == pytest.approx(2.0)
    assert rows[4][4] == pytest.approx(1.0)


def test_channels_clamp_a_foreign_flux():
    # A hand-edited `_flux 700` or `_weight 1e300` would overflow the float32
    # atlas; the render channels are clamped (flux 4, weight 1), the file text stays.
    table = MaterialTable()
    table.set(1, DEFAULT.with_type("EMIT").with_value(b"_flux", 700).with_weight(0.5))
    table.set(2, DEFAULT.with_type("EMIT").with_value(b"_flux", -3))
    table.set(3, DEFAULT.with_type("EMIT").with_value(b"_weight", 1e300))
    assert table.get(1).text(b"_flux") == b"700"
    rows = table.channels()
    assert np.isfinite(rows).all()
    assert rows[1][3] == pytest.approx(0.5 * EMIT_FLUX_BASE ** 4, rel=1e-5)
    assert rows[2][3] == pytest.approx(1.0)
    assert rows[3][3] == pytest.approx(EMIT_FLUX_BASE ** EMIT_DEFAULT_FLUX, rel=1e-5)  # weight 1, the default flux
    image1, image2 = material_atlas(table)
    assert np.isfinite(image1).all() and np.isfinite(image2).all()


def test_material_atlas_texels():
    table = _table("robot")
    image1, image2 = material_atlas(table)
    rows = table.channels()
    assert image1.shape == (16, 16, 4) and image1.dtype == np.float32
    assert image2.shape == (16, 16, 4) and image2.dtype == np.float32
    for i in (0, 1, 169, 177, 185, 209, 255):
        texel = (i // 16, i % 16)
        assert image1[texel][:3].tolist() == rows[i][:3].tolist()
        assert image2[texel][:3].tolist() == rows[i][3:].tolist()
    assert (image1[..., 3] == 1).all() and (image2[..., 3] == 1).all()
    assert image2[185 // 16, 185 % 16, 0] == pytest.approx(0.05 * EMIT_FLUX_BASE ** 4, rel=1e-5)


# --- carry_materials ---------------------------------------------------------


def test_carry_materials_bird_into_robot():
    base, base_used = model_palette_counts("robot")
    incoming, incoming_used = model_palette_counts("bird")
    # The bird's 171 gets the robot's colour 185, so it reuses that slot by exact match.
    incoming[171] = base[185]
    _, remap, claimed_by = merge_claimed(base, base_used, incoming, incoming_used)
    assert remap[171] == 185 and claimed_by[185] == 0
    robot = _table("robot")
    bird = _table("bird")
    before = robot.copy()
    result = carry_materials(robot, bird, claimed_by)
    assert robot == before  # the input is unchanged
    assert result is not robot
    assert len(result) == 256
    # Claimed slots carry the bird's material: 169 -> 1, 177 -> 2, 81 stays 81.
    assert remap[169] == 1 and result.get(1) == bird.get(169)
    assert remap[177] == 2 and result.get(2) == bird.get(177)
    assert result.get(81) == bird.get(81) and result.get(81).type == "PLASTIC"
    assert result.get(180) == bird.get(180) and result.get(180).type == "METAL"
    # A slot reused by exact colour keeps the robot's material.
    assert result.get(185) == robot.get(185)
    assert result.get(185) != bird.get(171)
    # Slots the merge did not write keep theirs.
    for k in (169, 177, 186, 193, 209, 256):
        assert result.get(k) == robot.get(k)
    # An empty incoming table resets every claimed slot to the default.
    plain = carry_materials(robot, MaterialTable(), claimed_by)
    for k in np.nonzero(claimed_by)[0]:
        assert plain.get(int(k)) == DEFAULT
    assert plain.get(185) == robot.get(185)
    # Both tables empty: the result stays empty.
    assert not carry_materials(MaterialTable(), MaterialTable(), claimed_by)
    # An empty scene table gets filled when a claimed slot needs a material.
    filled = carry_materials(MaterialTable(), bird, claimed_by)
    assert len(filled) == 256 and filled.get(1) == bird.get(169)
    assert filled.get(3) is DEFAULT


def test_material_is_frozen_and_compares_by_props():
    a = Material(DEFAULT_PROPS)
    assert a == DEFAULT and a is not DEFAULT
    assert hash(a) == hash(DEFAULT)
    with pytest.raises(AttributeError):
        a.props = ()
