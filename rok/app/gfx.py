"""Drawing calls on top of Rok's draw layers: `pen` for shapes and `fonts`
for text, in two passes:

    with gfx.pixels(drawlist, region):     # the pixel pass
        pen.tris(points, colour)       # region pixels, origin bottom left
        fonts.size(0, 13); fonts.position(0, x, y, 0); fonts.draw(0, "text")
    with gfx.view(drawlist):                # the world pass
        with gfx.translated(origin):        # moved by origin
            pen.lines(points, colour, width)
        with gfx.on_top():                  # no depth test
            ...

Points may be (N, 2) or (N, 3) in both passes (2D points get z = 0).
Colours are (r, g, b, a) floats in sRGB, as the host takes them;
`linear_to_srgb` converts where the code has linear values.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from . import draw

ROOT = Path(__file__).resolve().parents[2]
ICONS = ROOT / "assets" / "icons"

# The text baseline sits this far below the top of the host's text box, as a
# part of the font size (the system font at line height 1.25).
BASELINE = 0.95
# fonts.dimensions heights: capitals and digits, and text with descenders.
CAP_HEIGHT = 0.72
FULL_HEIGHT = 0.95


class _Target:
    def __init__(self) -> None:
        self.list: draw.DrawList | None = None
        self.height = 0.0
        self.mode = "2d"
        self.offset = np.zeros(3)
        self.top = False
        self.linear_colours = False
        self.off2 = (0.0, 0.0)  # 2D offset in region pixels (shifted)
        self.clip = None  # 2D clip rect in region pixels (clipped), after the offset
        self.alpha = 1.0  # opacity multiplier (faded)


_t = _Target()


@contextlib.contextmanager
def shifted(dx: float, dy: float):
    """2D drawing moved by (dx, dy) region pixels."""
    saved = _t.off2
    _t.off2 = (saved[0] + float(dx), saved[1] + float(dy))
    try:
        yield
    finally:
        _t.off2 = saved


@contextlib.contextmanager
def clipped(rect):
    """2D drawing kept inside `rect` (x, y, w, h): rects are cut to it; text,
    triangles and line segments that leave it are left out."""
    saved = _t.clip
    r = tuple(float(v) for v in rect)
    if saved is not None:
        x0, y0 = max(r[0], saved[0]), max(r[1], saved[1])
        x1, y1 = min(r[0] + r[2], saved[0] + saved[2]), min(r[1] + r[3], saved[1] + saved[3])
        r = (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))
    _t.clip = r
    try:
        yield
    finally:
        _t.clip = saved


@contextlib.contextmanager
def faded(alpha: float):
    """2D drawing at `alpha` times its opacity."""
    saved = _t.alpha
    _t.alpha = saved * max(0.0, min(1.0, float(alpha)))
    try:
        yield
    finally:
        _t.alpha = saved


def _inside_clip(x: float, y: float, eps: float = 0.01) -> bool:
    c = _t.clip
    return c is None or (c[0] - eps <= x <= c[0] + c[2] + eps and c[1] - eps <= y <= c[1] + c[3] + eps)


def _clip2(p: np.ndarray, group: int) -> np.ndarray:
    """Points moved by the 2D offset; with a clip, only groups (triangles or
    segments) that stay wholly inside it."""
    return _clip2_kept(p, group)[0]


def _clip2_kept(p: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray | None]:
    """`_clip2`, and which points it kept (None: all of them), for data that
    goes with each point."""
    if _t.off2 != (0.0, 0.0):
        p = p.copy()
        p[:, 0] += _t.off2[0]
        p[:, 1] += _t.off2[1]
    c = _t.clip
    if c is None or len(p) == 0:
        return p, None
    n = len(p) // group * group
    q = p[:n].reshape(-1, group, p.shape[1])
    ok = ((q[:, :, 0] >= c[0] - 0.01) & (q[:, :, 0] <= c[0] + c[2] + 0.01)
          & (q[:, :, 1] >= c[1] - 0.01) & (q[:, :, 1] <= c[1] + c[3] + 0.01)).all(axis=1)
    kept = np.zeros(len(p), dtype=bool)
    kept[:n] = np.repeat(ok, group)
    return q[ok].reshape(-1, p.shape[1]), kept


def _srgb(colour) -> tuple:
    c = tuple(float(v) for v in colour)
    if len(c) == 3:
        c = c + (1.0,)
    if _t.alpha != 1.0:
        c = c[:3] + (c[3] * _t.alpha,)
    if _t.linear_colours:
        return tuple(linear_to_srgb(np.array(c[:3]))) + (c[3],)
    return c


def linear_to_srgb(rgb) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


@contextlib.contextmanager
def pixels(d: draw.DrawList, region, linear: bool = False):
    """Draw in region pixels with the origin at the bottom left."""
    saved = (_t.list, _t.height, _t.mode, _t.linear_colours)
    _t.list, _t.height, _t.mode, _t.linear_colours = d, float(region.height), "2d", linear
    try:
        yield
    finally:
        _t.list, _t.height, _t.mode, _t.linear_colours = saved


@contextlib.contextmanager
def view(d: draw.DrawList, linear: bool = False):
    """Draw in world units, tested against the voxel depth."""
    saved = (_t.list, _t.mode, _t.offset.copy(), _t.top, _t.linear_colours)
    _t.list, _t.mode, _t.offset, _t.top, _t.linear_colours = d, "3d", np.zeros(3), False, linear
    try:
        yield
    finally:
        _t.list, _t.mode, _t.offset, _t.top, _t.linear_colours = saved


@contextlib.contextmanager
def translated(offset):
    saved = _t.offset.copy()
    _t.offset = _t.offset + np.asarray(offset, dtype=np.float64)
    try:
        yield
    finally:
        _t.offset = saved


@contextlib.contextmanager
def on_top(yes: bool = True):
    saved = _t.top
    _t.top = yes
    try:
        yield
    finally:
        _t.top = saved


def current() -> draw.DrawList:
    if _t.list is None:
        raise RuntimeError("draw outside gfx.pixels() or gfx.view()")
    return _t.list


def _flip(points2: np.ndarray) -> np.ndarray:
    out = np.asarray(points2, dtype=np.float32).copy()
    out[:, 1] = _t.height - out[:, 1]
    return out


def _pts(points) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    if p.ndim == 1:
        p = p.reshape(-1, 3 if _t.mode == "3d" else 2)
    return p


class pen:  # noqa: N801 - a namespace named like a module
    """Triangles, lines and points."""

    @staticmethod
    def f32(points) -> np.ndarray:
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def to_3d(points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float32)
        if p.ndim == 2 and p.shape[1] == 2:
            return np.c_[p, np.zeros(len(p), dtype=np.float32)]
        return p

    @staticmethod
    def lines_batch(points) -> np.ndarray:
        return _pts(points)

    @staticmethod
    def tris_batch(points) -> np.ndarray:
        return _pts(points)

    @staticmethod
    def tris(points, colour) -> None:
        p = _pts(points)
        if len(p) == 0:
            return
        if _t.mode == "3d":
            current().tris3(p[:, :3] + _t.offset, _srgb(colour), on_top=_t.top)
        else:
            p = _clip2(p[:, :2], 3)
            if len(p):
                current().tris2(_flip(p), _srgb(colour))

    fill = tris

    @staticmethod
    def lines(points, colour, width: float = 1.0) -> None:
        p = _pts(points)
        if len(p) < 2:
            return
        p = p[: len(p) // 2 * 2]
        if _t.mode == "3d":
            current().lines3(p[:, :3] + _t.offset, _srgb(colour), width, on_top=_t.top)
        else:
            p2 = _clip2(p[:, :2], 2)
            if len(p2) < 2:
                return
            q = _flip(p2).reshape(-1, 4)
            current().lines2(q, _srgb(colour), width)

    polyline = lines

    @staticmethod
    def fan(points, colour) -> None:
        """A convex polygon (as a TRI_FAN)."""
        p = _pts(points)
        n = len(p)
        if n < 3:
            return
        idx = np.array([[0, i, i + 1] for i in range(1, n - 1)]).reshape(-1)
        pen.tris(p[idx], colour)

    @staticmethod
    def rect_fill(rect, colour) -> None:
        x, y, w, h = (float(v) for v in rect)
        pen.tris(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]]), colour)

    @staticmethod
    def flat(points, colours) -> None:
        """Triangles with a colour per vertex."""
        p = _pts(points)
        c = np.asarray(colours, dtype=np.float32)
        if _t.linear_colours:
            c = c.copy()
            c[:, :3] = linear_to_srgb(c[:, :3])
        if _t.mode == "3d":
            current().tris3(p[:, :3] + _t.offset, c, on_top=_t.top)
        else:
            p2, kept = _clip2_kept(p[:, :2], 3)
            if kept is not None:
                c = c[kept]
            if _t.alpha != 1.0:
                c = c.copy()
                c[:, 3] *= _t.alpha
            if len(p2):
                current().tris2(_flip(p2), c)


def rrect(rect, colour, radius: float = 0.0, stroke: float = 0.0) -> None:
    """A rounded rectangle in region pixels (x, y, w, h, origin bottom left),
    soft-edged; with `stroke` only its outline. Within a layer the host draws
    rects after triangles and before lines, icons and text."""
    x, y, w, h = (float(v) for v in rect)
    x, y = x + _t.off2[0], y + _t.off2[1]
    c = _t.clip
    if c is not None:
        x0, y0 = max(x, c[0]), max(y, c[1])
        x1, y1 = min(x + w, c[0] + c[2]), min(y + h, c[1] + c[3])
        if x1 - x0 <= 0.5 or y1 - y0 <= 0.5:
            return
        if (x0, y0, x1, y1) != (x, y, x + w, y + h):
            if stroke:
                return  # a cut outline would draw a false edge
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            radius = min(radius, w / 2.0, h / 2.0)
    current().rect(x, _t.height - y - h, w, h, _srgb(colour), radius, stroke)


class _Fonts:
    """Text: one state (size, position, colour) per font id."""

    SHADOW = 1
    CLIPPING = 2
    WORD_WRAP = 4
    ROTATION = 8

    def __init__(self) -> None:
        self.state: dict[int, dict] = {}
        self.measure = None  # (text, size) -> (w, h): the host's measure_text

    def _s(self, font: int) -> dict:
        return self.state.setdefault(font, {"size": 11.0, "colour": (1, 1, 1, 1), "pos": (0.0, 0.0), "shadow": None})

    def size(self, font: int, size: float, dpi: int | None = None) -> None:
        self._s(font)["size"] = float(size)

    def color(self, font: int, r: float, g: float, b: float, a: float = 1.0) -> None:
        self._s(font)["colour"] = (r, g, b, a)

    def position(self, font: int, x: float, y: float, z: float = 0.0) -> None:
        self._s(font)["pos"] = (float(x), float(y))

    def enable(self, font: int, option: int) -> None:
        pass

    def disable(self, font: int, option: int) -> None:
        pass

    def shadow(self, font: int, level: int, r: float, g: float, b: float, a: float) -> None:
        self._s(font)["shadow"] = (r, g, b, a) if level else None

    def shadow_offset(self, font: int, x: int, y: int) -> None:
        pass

    def clipping(self, *args) -> None:
        pass

    def word_wrap(self, *args) -> None:
        pass

    def dimensions(self, font: int, text: str) -> tuple[float, float]:
        size = self._s(font)["size"]
        if self.measure is not None and text:
            w, _h = self.measure(text, size)
        else:
            w = len(text) * size * 0.55
        tall = any(ch in "gjpqy()[]{}|@$Q,;" for ch in text)
        return float(w), float(size * (FULL_HEIGHT if tall else CAP_HEIGHT))

    def draw(self, font: int, text: str) -> None:
        s = self._s(font)
        x, y = s["pos"]
        x, y = x + _t.off2[0], y + _t.off2[1]
        if _t.clip is not None and not (_inside_clip(x, y) and _inside_clip(x, y + s["size"] * 0.8)):
            return
        top = _t.height - y - s["size"] * BASELINE
        colour = _srgb(s["colour"])
        if s["shadow"] is not None:
            current().text(x + 1, top + 1, text, s["size"], _srgb(s["shadow"]))
        current().text(x, top, text, s["size"], colour)


fonts = _Fonts()


# --- icons ------------------------------------------------------------------------------


class Icons:
    """Loads icon PNGs into the host's atlas on first use."""

    def __init__(self, host) -> None:
        self.host = host
        self.loaded: set[str] = set()
        self.missing: set[str] = set()

    def path_of(self, name: str, scale: float = 2.0, folder: str = "chrome/1x") -> Path | None:
        key = (name, scale > 1.0, folder)
        paths = self.__dict__.setdefault("_paths", {})
        if key in paths:
            return paths[key]
        base = ICONS / folder
        found = None
        for candidate in ((f"{name}@2x.png", f"{name}.png") if scale > 1.0 else (f"{name}.png", f"{name}@2x.png")):
            p = base / candidate
            if p.exists():
                found = p
                break
        paths[key] = found
        return found

    def ensure(self, key: str, path: Path | None) -> bool:
        if key in self.loaded:
            return True
        if key in self.missing or path is None:
            self.missing.add(key)
            return False
        from rok.core import png

        _w, _h, rgba = png.decode(path.read_bytes(), max_pixels=4096 * 4096)
        self.host.add_image(key, np.ascontiguousarray(rgba, dtype=np.uint8))
        self.loaded.add(key)
        return True

    def upright_text(self, text: str, size: float, x: float, y: float, tint=(1, 1, 1, 1), height: float = 0.0) -> bool:
        """`text` turned upright, reading from the bottom up, its bottom-left
        corner at region (x, y), centred in `height` when it is longer. The text runs cannot turn, so the host draws
        it once as a mask, kept in the atlas at twice the size for sharp edges."""
        key = f"upright/{round(size, 2)}/{text}"
        if key not in self.loaded:
            made = self.host.text_mask(text, float(size) * 2.0) if hasattr(self.host, "text_mask") else None
            if made is None:  # a host that cannot draw text
                return False
            w, h, data = made
            mask = np.rot90(np.frombuffer(data, dtype=np.uint8).reshape(int(h), int(w)))
            rgba = np.full(mask.shape + (4,), 255, dtype=np.uint8)
            rgba[..., 3] = mask
            self.host.add_image(key, np.ascontiguousarray(rgba))
            self.loaded.add(key)
            self.__dict__.setdefault("_upright", {})[key] = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
        w, h = self.__dict__["_upright"][key]
        y = float(y) + max(0.0, (float(height) - h) / 2.0)
        current().icon(key, float(x), _t.height - float(y) - h, w, h, _srgb(tint), False)
        return True

    def draw(self, name: str, rect, tint=(1, 1, 1, 1), folder: str = "chrome/1x", image: bool = False) -> bool:
        """Draw icon `name` in the region rect (x, y, w, h), origin bottom left."""
        key = f"{folder}/{name}"
        if not self.ensure(key, self.path_of(name, 2.0, folder)):
            return False
        x, y, w, h = (float(v) for v in rect)
        current().icon(key, x, _t.height - y - h, w, h, _srgb(tint), image)
        return True
