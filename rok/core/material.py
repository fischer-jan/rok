""".vox materials: one MATL chunk per palette slot, kept as bytes.

A Material is the ordered key/value list of one MATL DICT. Parse and encode are
lossless, so an untouched file round trips byte for byte. An edit replaces value
text in place and keeps the .vox key order and number text. Numpy and
stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np

# Rok's five types; anything else is "OTHER" and kept untouched.
TYPES = ("DIFFUSE", "METAL", "GLASS", "EMIT", "PLASTIC")
TYPE_TEXT = {
    "DIFFUSE": b"_diffuse",
    "METAL": b"_metal",
    "GLASS": b"_glass",
    "EMIT": b"_emit",
    "PLASTIC": b"_plastic",
}
TYPE_OF = {v: k for k, v in TYPE_TEXT.items()}

# the .vox key order. Every chunk in the fixtures has exactly these 12.
KEYS = (
    b"_type", b"_weight", b"_rough", b"_spec", b"_spec_p", b"_ior", b"_att",
    b"_g0", b"_g1", b"_gw", b"_flux", b"_ldr",
)
DEFAULT_PROPS: tuple[tuple[bytes, bytes], ...] = (
    (b"_type", b"_diffuse"), (b"_weight", b"1"), (b"_rough", b"0.1"), (b"_spec", b"0.5"),
    (b"_spec_p", b"0.5"), (b"_ior", b"0.3"), (b"_att", b"0"), (b"_g0", b"-0.5"),
    (b"_g1", b"0.8"), (b"_gw", b"0.7"), (b"_flux", b"0"), (b"_ldr", b"0"),
)
# Older .vox files use these key names for the weight. Read when `_weight`
# is absent, written back when a slot has them, never added.
WEIGHT_ALIAS = {"METAL": b"_metal", "EMIT": b"_emit", "GLASS": b"_alpha"}
EMIT_DEFAULT_FLUX = 2  # a slot switched to Light starts with this power, as .vox files have it

# Material channel constants (channel_row says which type uses which).
PLAIN_ROUGH = 0.9
PLAIN_IOR = 1.45
PLAIN_SPEC = 0.5
# emission strength = weight * EMIT_FLUX_BASE ** flux. Each power step is ten
# times brighter: power 1 already lights a room, power 2 burns a white wall out.
EMIT_FLUX_BASE = 10.0
GLASS_IOR_MIN, GLASS_IOR_MAX = 1.0, 2.0

# type_codes(): DIFFUSE, absent and slot 0 are 0.
_TYPE_CODE = {"DIFFUSE": 0, "METAL": 1, "GLASS": 2, "EMIT": 3, "PLASTIC": 4, "OTHER": 5}

Pairs = tuple[tuple[bytes, bytes], ...]


def fmt(value: float | int) -> bytes:
    """The number text of .vox files: an int as is, a float with at most two decimals."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value)).encode("ascii")
    text = "%.2f" % round(float(value), 2)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", "", "-"):
        text = "0"
    return text.encode("ascii")


def parse_number(text: bytes | None, default: float) -> float:
    """float(text), or `default` when the text is not a finite number."""
    if text is None:
        return default
    try:
        value = float(text)
    except (ValueError, TypeError):
        return default
    return value if math.isfinite(value) else default


@dataclass(frozen=True)
class Material:
    """Every key/value pair of one MATL DICT, in file order, as bytes."""

    props: Pairs

    def __post_init__(self) -> None:
        pairs = tuple((bytes(k), bytes(v)) for k, v in self.props)
        object.__setattr__(self, "props", pairs)

    def text(self, key: bytes, default: bytes | None = None) -> bytes | None:
        """The value text of `key`, or `default` when the key is absent."""
        for k, v in self.props:
            if k == key:
                return v
        return default

    @property
    def type(self) -> str:
        return TYPE_OF.get(self.text(b"_type", b""), "OTHER")

    @property
    def weight(self) -> float:
        text = self.text(b"_weight")
        if text is None:
            alias = WEIGHT_ALIAS.get(self.type)
            if alias is not None:
                text = self.text(alias)
        return parse_number(text, 1.0)

    @property
    def rough(self) -> float:
        return parse_number(self.text(b"_rough"), 0.1)

    @property
    def ior(self) -> float:
        return parse_number(self.text(b"_ior"), 0.3)

    @property
    def spec(self) -> float:
        return parse_number(self.text(b"_spec"), 0.5)

    @property
    def flux(self) -> int:
        return int(parse_number(self.text(b"_flux"), 0.0))

    @property
    def is_default(self) -> bool:
        return self.props == DEFAULT_PROPS

    def _replaced(self, key: bytes, value: bytes, *, front: bool) -> Material:
        """The value of `key` replaced in place; a missing key goes to the front or the end."""
        pairs = list(self.props)
        for n, (k, _v) in enumerate(pairs):
            if k == key:
                pairs[n] = (key, value)
                break
        else:
            if front:
                pairs.insert(0, (key, value))
            else:
                pairs.append((key, value))
        return Material(tuple(pairs))

    def with_type(self, kind: str) -> Material:
        """`_type` replaced; every other value stays. A new Light gets a power at once."""
        if kind not in TYPE_TEXT:
            raise ValueError(f"cannot set material type {kind!r}")
        out = self._replaced(b"_type", TYPE_TEXT[kind], front=True)
        if kind == "EMIT" and out.flux == 0:
            out = out.with_value(b"_flux", EMIT_DEFAULT_FLUX)
        return out

    def with_value(self, key: bytes, value: float | int) -> Material:
        return self._replaced(key, fmt(value), front=False)

    def with_weight(self, value: float) -> Material:
        """`_weight`, and the same value into the alias key when the slot carries it."""
        out = self.with_value(b"_weight", value)
        alias = WEIGHT_ALIAS.get(self.type)
        if alias is not None and self.text(alias) is not None:
            out = out.with_value(alias, value)
        return out


DEFAULT = Material(DEFAULT_PROPS)


# --- chunk bytes -------------------------------------------------------------


def _read_string(content: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 4 > len(content):
        raise ValueError("MATL: truncated STRING")
    (length,) = struct.unpack_from("<i", content, offset)
    offset += 4
    if length < 0 or offset + length > len(content):
        raise ValueError("MATL: bad STRING length")
    return content[offset : offset + length], offset + length


def parse_matl(content: bytes) -> tuple[int, Material]:
    """(id, Material) from the content of one MATL chunk. Every byte must be used."""
    content = bytes(content)
    if len(content) < 8:
        raise ValueError("MATL: content too short")
    material_id, count = struct.unpack_from("<ii", content, 0)
    if count < 0:
        raise ValueError("MATL: negative DICT count")
    offset = 8
    props = []
    for _ in range(count):
        key, offset = _read_string(content, offset)
        value, offset = _read_string(content, offset)
        props.append((key, value))
    if offset != len(content):
        raise ValueError("MATL: bytes left over")
    return material_id, Material(tuple(props))


def encode_matl(material_id: int, material: Material) -> bytes:
    """The content of one MATL chunk. encode_matl(*parse_matl(c)) == c."""
    parts = [struct.pack("<ii", material_id, len(material.props))]
    for key, value in material.props:
        parts.append(struct.pack("<i", len(key)) + key)
        parts.append(struct.pack("<i", len(value)) + value)
    return b"".join(parts)


# --- the table ---------------------------------------------------------------


class MaterialTable:
    """MATL id -> Material, in file order. Empty when the file had no MATL."""

    def __init__(self, entries: dict[int, Material] | None = None) -> None:
        self._entries: dict[int, Material] = dict(entries) if entries else {}

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaterialTable):
            return NotImplemented
        return list(self._entries.items()) == list(other._entries.items())

    def __repr__(self) -> str:
        return f"MaterialTable({len(self._entries)} entries)"

    def items(self) -> list[tuple[int, Material]]:
        return list(self._entries.items())

    def copy(self) -> MaterialTable:
        return MaterialTable(self._entries)

    def get(self, index: int) -> Material:
        return self._entries.get(index, DEFAULT)

    def set(self, index: int, material: Material) -> None:
        """Replace the entry in place, or append a new id. An empty table fills first."""
        if not self._entries:
            self.ensure_full()
        self._entries[index] = material

    def ensure_full(self) -> None:
        """Ids 1..256 become DEFAULT, in order, when the table is empty."""
        if self._entries:
            return
        for i in range(1, 257):
            self._entries[i] = DEFAULT

    def restore(self, other: MaterialTable) -> None:
        """In place: the entries become a copy of `other`'s."""
        self._entries.clear()
        self._entries.update(other._entries)

    def assign_slots(self, other: MaterialTable) -> None:
        """Slots 1..255 take other's; id 256 and ids outside 1..256 stay."""
        if not self._entries and not other._entries:
            return
        for i in range(1, 256):
            self.set(i, other.get(i))

    def type_codes(self) -> np.ndarray:
        """(256,) uint8: 0 Plain or absent, 1 Metal, 2 Glass, 3 Emit, 4 Plastic, 5 Other."""
        codes = np.zeros(256, dtype=np.uint8)
        for index, material in self._entries.items():
            if 1 <= index <= 255:
                codes[index] = _TYPE_CODE[material.type]
        return codes

    def channels(self) -> np.ndarray:
        """(256, 6) float32: metallic, roughness, transmission, emission, ior, specular."""
        out = np.zeros((256, 6), dtype=np.float32)
        out[1:] = _PLAIN_ROW
        for index, material in self._entries.items():
            if 1 <= index <= 255:
                out[index] = channel_row(material)
        return out


_PLAIN_ROW = (0.0, PLAIN_ROUGH, 0.0, 0.0, PLAIN_IOR, PLAIN_SPEC)


def channel_row(material: Material) -> tuple[float, float, float, float, float, float]:
    """The material channels of one slot: metallic, roughness, transmission,
    emission, ior, specular. Plain voxels are matte (roughness PLAIN_ROUGH, the
    file's _rough is ignored). Metal: weight is metallic, _rough the roughness.
    Glass: weight is transmission, ior is 1 + _ior (1 to 2). Emit: emission is
    weight * EMIT_FLUX_BASE ** flux. Plastic: weight is the specular level."""
    kind = material.type
    if kind == "METAL":
        return (material.weight, material.rough, 0.0, 0.0, PLAIN_IOR, PLAIN_SPEC)
    if kind == "GLASS":
        ior = min(max(GLASS_IOR_MIN + material.ior, GLASS_IOR_MIN), GLASS_IOR_MAX)
        return (0.0, material.rough, material.weight, 0.0, ior, PLAIN_SPEC)
    if kind == "EMIT":
        # A .vox file holds weight 0..1 and flux 0..4; a foreign value stays in
        # the file but is clamped here, so a huge one cannot put inf into the atlas.
        weight = min(max(material.weight, 0.0), 1.0)
        flux = min(max(material.flux, 0), 4)
        strength = weight * EMIT_FLUX_BASE ** flux
        return (0.0, PLAIN_ROUGH, 0.0, strength, PLAIN_IOR, PLAIN_SPEC)
    if kind == "PLASTIC":
        return (0.0, material.rough, 0.0, 0.0, PLAIN_IOR, material.weight)
    return _PLAIN_ROW


def render_rows(table: MaterialTable) -> np.ndarray:
    """(256, 8) float32 for the rendered view (Host.set_materials): the
    channels, then the type code, then 0."""
    rows = np.zeros((256, 8), dtype=np.float32)
    rows[:, :6] = table.channels()
    rows[:, 6] = table.type_codes()
    return rows


def material_atlas(table: MaterialTable) -> tuple[np.ndarray, np.ndarray]:
    """Two (16, 16, 4) float32 images; texel (i // 16, i % 16) holds slot i, alpha 1.

    Image 1 RGB is metallic, roughness, transmission. Image 2 RGB is emission
    strength, ior, specular level.
    """
    rows = table.channels()
    image1 = np.ones((16, 16, 4), dtype=np.float32)
    image2 = np.ones((16, 16, 4), dtype=np.float32)
    image1[..., :3] = rows[:, 0:3].reshape(16, 16, 3)
    image2[..., :3] = rows[:, 3:6].reshape(16, 16, 3)
    return image1, image2


def carry_materials(
    table: MaterialTable, incoming: MaterialTable, claimed_by: np.ndarray
) -> MaterialTable:
    """A copy of `table` where every slot the merge wrote carries the incoming material.

    `claimed_by[k]` is the incoming index that wrote slot k, 0 when the merge did
    not write it (see palette.merge_claimed). An empty `incoming` carries DEFAULT.
    """
    result = table.copy()
    for k in range(1, 256):
        i = int(claimed_by[k])
        if i == 0:
            continue
        wanted = incoming.get(i)
        if wanted != result.get(k):
            result.set(k, wanted)
    return result
