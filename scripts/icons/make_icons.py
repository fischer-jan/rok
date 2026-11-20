"""Turn generated glyph images into Rok's icon masks.

    uv run python scripts/icons/make_icons.py [NAME ...]

Reads scripts/icons/gen/NAME.png (a glyph on the charcoal background, from
generate.sh) and writes assets/icons/chrome/1x/NAME.png (64 px) and
NAME@2x.png (128 px), plus wheel/NAME.png (96) and NAME@2x.png (192) for the
eight mode glyphs. Every file is a mask: white with the alpha as the ink. The
outline gets alpha 1, the ember fill alpha FACE (a lighter tone of the ink),
and the drawing tints the mask grey, near white or ember. No names: every
glyph in glyphs.tsv. Also writes scripts/icons/contact.png to check by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rok.core import png  # noqa: E402

png.MAX_PIXELS = 1 << 24  # the guard is for icons; the generated images are 1024²
from rok.core.png import decode, encode  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = REPO / "assets" / "icons" / "chrome"
WHEEL = ("VOXEL", "FACE", "MODE_BOX", "LINE", "CENTER", "RECT", "LASSO", "MODE_PATTERN")
FACE = 0.5  # the fill's alpha
MARGIN = 0.10  # of the square, each side
CHAR = np.array([0.165, 0.153, 0.141], np.float32)
GLYPH = np.array([0.78, 0.76, 0.735], np.float32)
EMBER = np.array([0.91, 0.365, 0.227], np.float32)


def load(path: Path) -> np.ndarray:
    _h, _w, rgba = decode(path.read_bytes(), max_pixels=1 << 24)  # decode's default limit is bound at import
    return rgba[..., :3].astype(np.float32) / 255.0


def box_down(img: np.ndarray, t: int) -> np.ndarray:
    s = img.shape[0]
    f = max(1, s // t)
    cut = f * t
    off = (s - cut) // 2
    return img[off : off + cut, off : off + cut].reshape(t, f, t, f, -1).mean(axis=(1, 3))


def dilate(a: np.ndarray, r: int) -> np.ndarray:
    out = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = np.maximum(out, np.roll(np.roll(a, dy, 0), dx, 1))
    return out


def mask_of(rgb: np.ndarray) -> np.ndarray:
    """Alpha per pixel: the outline 1, the ember fill FACE, the background 0."""
    edges = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3)])
    bg = np.median(edges, axis=0)
    d = np.abs(rgb - bg).max(axis=-1)
    cover = np.clip((d - 0.10) / 0.25, 0, 1)  # a dead zone for the JPEG noise
    on = cover > 0.3
    neigh = sum(np.roll(np.roll(on, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - on
    cover = np.where(on & (neigh < 3), 0.0, cover)  # speckles
    emberness = np.clip((rgb[..., 0] - rgb[..., 2] - 0.15) / 0.3, 0, 1)
    return cover * (1.0 - emberness * (1.0 - FACE))


def square(a: np.ndarray, margin: float) -> np.ndarray:
    ys, xs = np.nonzero(a > 0.05)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    pad = int(side * margin)
    out = np.zeros((side + 2 * pad,) * 2, np.float32)
    oy = pad + (side - (y1 - y0)) // 2
    ox = pad + (side - (x1 - x0)) // 2
    out[oy : oy + (y1 - y0), ox : ox + (x1 - x0)] = a[y0:y1, x0:x1]
    return out


def shrink(a: np.ndarray, t: int) -> np.ndarray:
    """Down to t px; the mask grows first so a line keeps at least 1.5 px."""
    r = max(0, int(round(a.shape[0] / t * 0.6)))
    big = dilate(a, r) if r > 0 else a
    return box_down(big[..., None], t)[..., 0]


def write_mask(a: np.ndarray, path: Path) -> None:
    a8 = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(np.stack([np.full_like(a8, 255)] * 3 + [a8], axis=-1)))


def synthetic_more(side: int = 1024) -> np.ndarray:
    """Three dots. The new-clip button on the timeline uses this glyph."""
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32) / side
    a = np.zeros((side, side), np.float32)
    for cx in (0.22, 0.5, 0.78):
        d = np.sqrt((xx - cx) ** 2 + (yy - 0.5) ** 2)
        a = np.maximum(a, np.clip((0.085 - d) / 0.01, 0, 1))
    return a


from anim_glyphs import GLYPHS as ANIM_GLYPHS  # noqa: E402  (scripts/icons is on sys.path when run)
from select_glyphs import GLYPHS as SELECT_GLYPHS  # noqa: E402
from tool_glyphs import GLYPHS as TOOL_GLYPHS  # noqa: E402
from arm_glyphs import GLYPHS as ARM_GLYPHS  # noqa: E402
from scene_glyphs import GLYPHS as SCENE_GLYPHS  # noqa: E402
from stroke_glyphs import GLYPHS as STROKE_GLYPHS  # noqa: E402
from brick_glyphs import GLYPHS as BRICK_GLYPHS  # noqa: E402

SYNTHETIC = {"MORE": synthetic_more, **ANIM_GLYPHS, **SELECT_GLYPHS, **TOOL_GLYPHS, **ARM_GLYPHS, **SCENE_GLYPHS, **STROKE_GLYPHS,
             **BRICK_GLYPHS}


def names_from_tsv() -> list[str]:
    lines = (HERE / "glyphs.tsv").read_text().splitlines()
    return [l.split("\t", 1)[0] for l in lines if l.strip() and not l.startswith("#")] + list(SYNTHETIC)


def contact(tiles: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    gap, per = 16, 9
    rows = [tiles[i : i + per] for i in range(0, len(tiles), per)]
    block = 64 + gap + 64 + gap + 32 + 2 * gap
    canvas = np.tile(CHAR, (gap + len(rows) * block, gap + per * (64 + gap), 1))

    def blit(y, x, m, ink):
        t = m.shape[0]
        m3 = m[..., None]
        canvas[y : y + t, x : x + t] = CHAR * (1 - m3) + ink * m3

    for r, row in enumerate(rows):
        y0 = gap + r * block
        for i, (_name, m64, m32) in enumerate(row):
            x = gap + i * (64 + gap)
            blit(y0, x, m64, GLYPH)
            blit(y0 + 64 + gap, x, m64, EMBER)
            blit(y0 + 2 * (64 + gap), x, m32, GLYPH)
    rgba = np.concatenate([canvas, np.ones(canvas.shape[:2] + (1,), np.float32)], axis=-1)
    (HERE / "contact.png").write_bytes(encode((rgba * 255).round().astype(np.uint8)))


def main(names: list[str]) -> None:
    tiles = []
    for name in names:
        if name in ARM_GLYPHS:  # drawn in their final frame, so the four cubes match
            a = ARM_GLYPHS[name]()
        elif name in SYNTHETIC:
            a = square(SYNTHETIC[name](), MARGIN)
        else:
            a = square(mask_of(load(HERE / "gen" / f"{name}.png")), MARGIN)
        write_mask(shrink(a, 64), OUT / "1x" / f"{name}.png")
        write_mask(shrink(a, 128), OUT / "1x" / f"{name}@2x.png")
        if name in WHEEL:
            write_mask(shrink(a, 96), OUT / "wheel" / f"{name}.png")
            write_mask(shrink(a, 192), OUT / "wheel" / f"{name}@2x.png")
        tiles.append((name, shrink(a, 64), shrink(a, 32)))
    contact(tiles)
    print(f"{len(names)} glyphs -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or names_from_tsv())
