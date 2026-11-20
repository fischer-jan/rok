"""Write the palette fixtures in tests/fixtures/phase3/palettes/.

Run once: `uv run python scripts/make_palette_fixtures.py`.

vox_default.png: 256 x 1 RGBA, pixel k = default palette slot k + 1, the last
pixel empty: the common layout of a 256 x 1 palette strip.
sample.gpl, sample.txt, sample.pal, sample.hex: the same 16 colours (Rok's own
sample set) in each text format, written by hand here so the fixtures do not depend
on the writers they test.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rok.core import png  # noqa: E402
from rok.core.palette import default_palette  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "phase3" / "palettes"

# Rok's own 16 sample colours, (r, g, b, name).
SAMPLE = (
    (12, 14, 20, "night"),
    (38, 52, 92, "deep-blue"),
    (70, 110, 170, "steel-blue"),
    (120, 190, 230, "sky"),
    (40, 200, 150, "mint"),
    (30, 110, 70, "moss"),
    (150, 200, 60, "lime"),
    (240, 220, 110, "straw"),
    (235, 140, 50, "ember"),
    (200, 70, 40, "brick"),
    (150, 40, 90, "plum"),
    (230, 120, 170, "rose"),
    (110, 80, 60, "bark"),
    (120, 120, 130, "slate"),
    (190, 190, 200, "ash"),
    (250, 248, 240, "chalk"),
)


def write_default_png() -> None:
    row = np.zeros((1, 256, 4), dtype=np.uint8)
    row[0, :255] = default_palette()[1:256]
    (OUT / "vox_default.png").write_bytes(png.encode(row))


def write_gpl() -> None:
    lines = ["GIMP Palette", "Name: Sample", "Columns: 16", "#"]
    for r, g, b, name in SAMPLE:
        lines.append(f"{r:3d} {g:3d} {b:3d} {name}")
    (OUT / "sample.gpl").write_text("\n".join(lines) + "\n")


def write_pdn_txt() -> None:
    lines = ["; paint.net Palette File", "; Sample palette"]
    for r, g, b, _ in SAMPLE:
        lines.append(f"FF{r:02X}{g:02X}{b:02X}")
    (OUT / "sample.txt").write_text("\n".join(lines) + "\n")


def write_jasc() -> None:
    lines = ["JASC-PAL", "0100", str(len(SAMPLE))]
    for r, g, b, _ in SAMPLE:
        lines.append(f"{r} {g} {b}")
    (OUT / "sample.pal").write_text("\r\n".join(lines) + "\r\n")


def write_hex() -> None:
    lines = [f"{r:02x}{g:02x}{b:02x}" for r, g, b, _ in SAMPLE]
    (OUT / "sample.hex").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_default_png()
    write_gpl()
    write_pdn_txt()
    write_jasc()
    write_hex()
    print(f"wrote 5 palette fixtures to {OUT}")


if __name__ == "__main__":
    main()
