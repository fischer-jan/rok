"""Palette import and export: .vox, .png, .gpl, .txt (Paint.NET), .pal (JASC), .hex,
and Rok's own .json (core/palette_json.py: names, materials, favourites).

Every reader returns a (256, 4) uint8 palette with slot 0 = (0, 0, 0, 0). A list of
colours fills slots 1..255 in order; slots the file does not list keep the
The .vox default palette. Alpha is 255 unless the format carries one (.png, .txt).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import palette_json, png
from .palette import default_palette
from .safe_write import write_bytes
from .vox_read import VoxError, loads_scene

Colour = tuple[int, int, int, int]

SUFFIXES = (".vox", ".png", ".gpl", ".txt", ".pal", ".hex", ".json")


class PaletteError(Exception):
    """A palette file that cannot be read or written."""


# --- read ---------------------------------------------------------------------


def read_palette(path: str | Path) -> np.ndarray:
    """Read a palette file; the format follows the lower-cased suffix."""
    path = Path(path)
    return read_palette_bytes(path.read_bytes(), path.suffix)


def read_palette_bytes(data: bytes, suffix: str) -> np.ndarray:
    """Parse palette bytes in the format of `suffix` (".png", ".gpl", ...)."""
    suffix = suffix.lower()
    if suffix == ".vox":
        try:
            return loads_scene(data).palette.copy()
        except VoxError as exc:
            raise PaletteError(str(exc)) from exc
    if suffix == ".png":
        try:
            height, width, rgba = png.decode(data)
        except png.PngError as exc:
            raise PaletteError(str(exc)) from exc
        return _png_to_palette(height, width, rgba)
    if suffix not in SUFFIXES:
        raise PaletteError(f"unknown palette format {suffix!r}")
    text = _decode_text(data)
    if suffix == ".json":
        return colours_to_palette([record.rgba for record in read_json(text)])
    if suffix == ".gpl":
        colours = parse_gpl(text)
    elif suffix == ".txt":
        colours = parse_pdn_txt(text)
    elif suffix == ".pal":
        colours = parse_jasc(text)
    else:
        colours = parse_hex(text)
    return colours_to_palette(colours)


def colours_to_palette(colours: list, base: np.ndarray | None = None) -> np.ndarray:
    """Colour k goes to slot k + 1, at most 255; the rest keeps `base`."""
    palette = default_palette() if base is None else np.array(base, dtype=np.uint8).copy()
    if palette.shape != (256, 4):
        raise PaletteError("base palette must be (256, 4)")
    count = min(len(colours), 255)
    if count:
        palette[1 : count + 1] = np.asarray(colours[:count], dtype=np.uint8).reshape(count, 4)
    palette[0] = (0, 0, 0, 0)
    return palette


def read_json(text: str) -> list[palette_json.Record]:
    """The colours of a Rok palette file, with names, materials and favourites."""
    try:
        return palette_json.loads(text)
    except ValueError as exc:
        raise PaletteError(str(exc)) from exc


def _png_to_palette(height: int, width: int, rgba: np.ndarray) -> np.ndarray:
    # Row-major, top row first. A 256 x 1 palette strip has pixel 0 = slot 1
    # and a 256th pixel that colours_to_palette drops.
    flat = rgba.reshape(height * width, 4)
    return colours_to_palette([tuple(int(v) for v in px) for px in flat[:255]])


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


# --- text parsers ---------------------------------------------------------------


def parse_gpl(text: str) -> list[Colour]:
    """GIMP palette: header lines, then `R G B` with an optional name."""
    colours: list[Colour] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("GIMP Palette"):
            continue
        if line.startswith("Name:") or line.startswith("Columns:"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise PaletteError(f"bad GPL line: {raw!r}")
        colours.append(_rgb_ints(parts[:3], raw))
    return colours


def parse_pdn_txt(text: str) -> list[Colour]:
    """Paint.NET palette: one AARRGGBB or RRGGBB hex value per line."""
    colours: list[Colour] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        colours.append(_hex_colour(line, raw))
    return colours


def parse_jasc(text: str) -> list[Colour]:
    """JASC-PAL: `JASC-PAL`, `0100`, a count, then `R G B` lines."""
    if text.startswith("RIFF"):
        raise PaletteError("not a JASC-PAL file")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 3 or lines[0] != "JASC-PAL":
        raise PaletteError("not a JASC-PAL file")
    try:
        count = int(lines[2])
    except ValueError as exc:
        raise PaletteError(f"bad JASC-PAL count: {lines[2]!r}") from exc
    colours: list[Colour] = []
    for raw in lines[3 : 3 + count]:
        parts = raw.split()
        if len(parts) < 3:
            raise PaletteError(f"bad JASC-PAL line: {raw!r}")
        colours.append(_rgb_ints(parts[:3], raw))
    return colours


def parse_hex(text: str) -> list[Colour]:
    """Lospec .hex: RRGGBB per line, `#` optional."""
    colours: list[Colour] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        colours.append(_hex_colour(line, raw))
    return colours


def _rgb_ints(parts: list[str], raw: str) -> Colour:
    try:
        r, g, b = (int(p) for p in parts)
    except ValueError as exc:
        raise PaletteError(f"bad colour line: {raw!r}") from exc
    if not all(0 <= v <= 255 for v in (r, g, b)):
        raise PaletteError(f"colour out of range: {raw!r}")
    return r, g, b, 255


def _hex_colour(line: str, raw: str) -> Colour:
    digits = line[1:] if line.startswith("#") else line
    if len(digits) not in (6, 8):
        raise PaletteError(f"bad hex colour: {raw!r}")
    try:
        value = int(digits, 16)
    except ValueError as exc:
        raise PaletteError(f"bad hex colour: {raw!r}") from exc
    a = (value >> 24) & 0xFF if len(digits) == 8 else 255
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF, a


# --- write --------------------------------------------------------------------


def write_palette(path: str | Path, palette: np.ndarray) -> None:
    """Write the palette in the format of the lower-cased suffix. The old file
    stays until the new one is complete."""
    path = Path(path)
    write_bytes(path, palette_bytes(palette, path.suffix))


def palette_bytes(palette: np.ndarray, suffix: str, count: int = 255) -> bytes:
    """Palette as file bytes in the format of `suffix`. The text formats list
    slots 1..`count`; the .png strip always holds all 255."""
    suffix = suffix.lower()
    palette = np.asarray(palette, dtype=np.uint8)
    if palette.shape != (256, 4):
        raise PaletteError("palette must be (256, 4)")
    count = max(0, min(int(count), 255))
    if suffix == ".png":
        # 256 x 1: pixel k = slot k + 1, the last pixel is empty.
        row = np.zeros((1, 256, 4), dtype=np.uint8)
        row[0, :255] = palette[1:256]
        return png.encode(row)
    if suffix == ".json":
        records = [palette_json.Record(rgba=tuple(int(v) for v in palette[i]), slot=i) for i in range(1, count + 1)]
        return palette_json.dumps(records).encode("utf-8")
    if suffix == ".gpl":
        return write_gpl(palette, count).encode("utf-8")
    if suffix == ".txt":
        return write_pdn_txt(palette, count).encode("utf-8")
    if suffix == ".pal":
        return write_jasc(palette, count).encode("utf-8")
    if suffix == ".hex":
        return write_hex(palette, count).encode("utf-8")
    raise PaletteError(f"cannot write palette format {suffix!r}")


def write_gpl(palette: np.ndarray, count: int = 255) -> str:
    lines = ["GIMP Palette", "Name: Rok", "Columns: 16", "#"]
    for i in range(1, count + 1):
        r, g, b, _ = (int(v) for v in palette[i])
        lines.append(f"{r:3d} {g:3d} {b:3d} Colour {i}")
    return "\n".join(lines) + "\n"


def write_pdn_txt(palette: np.ndarray, count: int = 255) -> str:
    lines = ["; Rok palette"]
    for i in range(1, count + 1):
        r, g, b, a = (int(v) for v in palette[i])
        lines.append(f"{a:02X}{r:02X}{g:02X}{b:02X}")
    return "\n".join(lines) + "\n"


def write_jasc(palette: np.ndarray, count: int = 255) -> str:
    lines = ["JASC-PAL", "0100", str(count)]
    for i in range(1, count + 1):
        r, g, b, _ = (int(v) for v in palette[i])
        lines.append(f"{r} {g} {b}")
    return "\r\n".join(lines) + "\r\n"


def write_hex(palette: np.ndarray, count: int = 255) -> str:
    lines = []
    for i in range(1, count + 1):
        r, g, b, _ = (int(v) for v in palette[i])
        lines.append(f"{r:02x}{g:02x}{b:02x}")
    return "\n".join(lines) + "\n"
