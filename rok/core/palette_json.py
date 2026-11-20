"""Rok's own palette file: JSON with each colour's hex value, name, group,
material and favourite flag.

    {
      "rok_palette": 1,
      "colours": [
        {"hex": "#FF8800", "slot": 12, "name": "ember", "group": "", "used": 40,
         "favourite": true, "material": {"_type": "_emit", "_emit": "1", "_flux": "2"}}
      ]
    }

`hex` is #RRGGBB, or #RRGGBBAA when the colour is not opaque. `material`
holds the material's properties as text, as a .vox file keeps them, and is
left out for a plain colour. `slot` is null for a favourite colour that is
not in the palette. A reader takes any object with the two top-level keys;
every other key of a colour is optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .material import DEFAULT, Material

VERSION = 1
RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class Record:
    """One colour of the file."""

    rgba: RGBA
    slot: int | None = None
    name: str = ""
    group: str = ""
    used: int = 0
    favourite: bool = False
    material: Material = DEFAULT


def hex_of(rgba) -> str:
    r, g, b, a = (int(v) for v in rgba)
    text = f"#{r:02X}{g:02X}{b:02X}"
    return text if a == 255 else text + f"{a:02X}"


def parse_hex(text: str) -> RGBA:
    """#RRGGBB or #RRGGBBAA (the # is optional) to an RGBA tuple."""
    digits = str(text).strip().lstrip("#")
    if len(digits) not in (6, 8):
        raise ValueError(f"bad hex colour {text!r}")
    try:
        value = int(digits, 16)
    except ValueError as exc:
        raise ValueError(f"bad hex colour {text!r}") from exc
    if len(digits) == 6:
        value = (value << 8) | 0xFF
    return (value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def material_to_dict(material: Material) -> dict[str, str]:
    return {k.decode("ascii", "replace"): v.decode("ascii", "replace") for k, v in material.props}


def material_from_dict(props) -> Material:
    if not isinstance(props, dict):
        raise ValueError("material must be an object of text properties")
    return Material(tuple((str(k).encode("ascii", "replace"), str(v).encode("ascii", "replace"))
                          for k, v in props.items()))


def dumps(records: list[Record]) -> str:
    colours = []
    for record in records:
        item: dict = {"hex": hex_of(record.rgba), "slot": record.slot}
        if record.name:
            item["name"] = record.name
        if record.group:
            item["group"] = record.group
        if record.used:
            item["used"] = int(record.used)
        if record.favourite:
            item["favourite"] = True
        if record.material != DEFAULT:
            item["material"] = material_to_dict(record.material)
        colours.append(item)
    return json.dumps({"rok_palette": VERSION, "colours": colours}, indent=1, ensure_ascii=False) + "\n"


def loads(text: str) -> list[Record]:
    """The file's colours in order. ValueError when the text is not a palette file."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    if not isinstance(data, dict) or "rok_palette" not in data or not isinstance(data.get("colours"), list):
        raise ValueError("not a Rok palette file: it needs rok_palette and a colours list")
    out = []
    for index, item in enumerate(data["colours"]):
        if not isinstance(item, dict) or "hex" not in item:
            raise ValueError(f"colour {index} needs a hex value")
        slot = item.get("slot")
        if slot is not None:
            if not isinstance(slot, int) or isinstance(slot, bool) or not 1 <= slot <= 255:
                raise ValueError(f"colour {index}: slot must be 1 to 255 or null")
        used = item.get("used", 0)
        material = material_from_dict(item["material"]) if "material" in item else DEFAULT
        out.append(Record(
            rgba=parse_hex(item["hex"]),
            slot=slot,
            name=str(item.get("name", "") or "")[:40],
            group=str(item.get("group", "") or "")[:40],
            used=int(used) if isinstance(used, (int, float)) and not isinstance(used, bool) else 0,
            favourite=bool(item.get("favourite", False)),
            material=material,
        ))
    return out
