"""Build draw layers for the host.

A layer is sent whole and stays until it is sent again or cleared, so the
app sends only the layers that changed (`Layers.send` compares contents).
Colours are sRGB with alpha, as (r, g, b, a) floats 0..1. 2D positions are
logical pixels with the origin at the top left; 3D positions are world units.
Line widths are logical pixels.
"""

from __future__ import annotations

import hashlib

import numpy as np

# The palette for the UI.
EMBER = (0.91, 0.36, 0.23, 1.0)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.905, 0.885, 0.855, 0.6)
PLATE = (0.11, 0.10, 0.10, 0.82)
ATTACH_GREEN = (0.35, 0.85, 0.40, 1.0)
ERASE_RED = (0.95, 0.30, 0.25, 1.0)
BOX_GREY = (0.62, 0.60, 0.57, 0.55)

_BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),  # along x
    (0, 2), (1, 3), (4, 6), (5, 7),  # along y
    (0, 4), (1, 5), (2, 6), (3, 7),  # along z
)


def box_segments(lo, hi) -> np.ndarray:
    """The twelve edges of the box lo..hi as (24, 3) line pairs."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    corners = np.array([[hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]] for i in range(8)])
    return corners[np.asarray(_BOX_EDGES).reshape(-1)]


def srgb_of(rgba_bytes, alpha: float = 1.0) -> tuple[float, float, float, float]:
    r, g, b = (int(c) / 255.0 for c in rgba_bytes[:3])
    return (r, g, b, alpha)


def _colours(colour, n: int) -> np.ndarray:
    """(n, 4) colours from one colour or from n colours."""
    c = np.asarray(colour, dtype=np.float32)
    if c.ndim == 1:
        if c.shape[0] == 3:
            c = np.append(c, 1.0).astype(np.float32)
        return np.broadcast_to(c, (n, 4))
    if c.shape[1] == 3:
        c = np.c_[c, np.ones(len(c), dtype=np.float32)]
    return c


class DrawList:
    """One layer's content, built with numpy where it matters."""

    def __init__(self) -> None:
        self.parts: dict[str, list[np.ndarray]] = {
            k: [] for k in ("lines3_depth", "lines3_top", "tris3_depth", "tris3_top", "tris2", "rects", "lines2")
        }
        self.icons: list[tuple] = []
        self.texts: list[tuple] = []
        # Set by a drawer that knows what its content is built from: Layers
        # then compares this instead of hashing the content (a big selection's
        # skin cost 16 ms a frame to hash).
        self.key: object = None
        # A shift of the 3D parts in world units, done on the GPU: a new one
        # costs no upload of the content (Host.set_draw_offset).
        self.offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- 3D -----------------------------------------------------------------------

    def lines3(self, pairs, colour, width: float = 1.5, on_top: bool = False) -> None:
        """Segments from (2N, 3) point pairs, as hud.cube_edges gives them."""
        p = np.asarray(pairs, dtype=np.float32).reshape(-1, 2, 3)
        if len(p) == 0:
            return
        n = len(p)
        rows = np.concatenate([p[:, 0], p[:, 1], _colours(colour, n), np.full((n, 1), width, np.float32)], axis=1)
        self.parts["lines3_top" if on_top else "lines3_depth"].append(rows.astype(np.float32).ravel())

    def line3(self, a, b, colour, width: float = 1.5, on_top: bool = False) -> None:
        self.lines3([a, b], colour, width, on_top)

    def box3(self, lo, hi, colour, width: float = 1.5, on_top: bool = False) -> None:
        self.lines3(box_segments(lo, hi), colour, width, on_top)

    def tris3(self, points, colour, on_top: bool = False) -> None:
        """Filled triangles from (3N, 3) points and one colour or (3N, 4) colours."""
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(p) == 0:
            return
        rows = np.concatenate([p, _colours(colour, len(p))], axis=1)
        self.parts["tris3_top" if on_top else "tris3_depth"].append(rows.astype(np.float32).ravel())

    # --- 2D (top-left origin) -------------------------------------------------------

    def tris2(self, points, colour) -> None:
        p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(p) == 0:
            return
        rows = np.concatenate([p, _colours(colour, len(p))], axis=1)
        self.parts["tris2"].append(rows.astype(np.float32).ravel())

    def rect(self, x, y, w, h, colour, radius: float = 0.0, stroke: float = 0.0) -> None:
        c = tuple(colour) + ((1.0,) if len(colour) == 3 else ())
        self.parts["rects"].append(np.array((x, y, w, h, *c, radius, stroke), dtype=np.float32))

    def lines2(self, pairs, colour, width: float = 1.0) -> None:
        p = np.asarray(pairs, dtype=np.float32).reshape(-1, 4)
        if len(p) == 0:
            return
        n = len(p)
        rows = np.concatenate([p, _colours(colour, n), np.full((n, 1), width, np.float32)], axis=1)
        self.parts["lines2"].append(rows.astype(np.float32).ravel())

    def line2(self, x0, y0, x1, y1, colour, width: float = 1.0) -> None:
        self.lines2([[x0, y0, x1, y1]], colour, width)

    def icon(self, name: str, x, y, w, h, tint=(1.0, 1.0, 1.0, 1.0), image: bool = False) -> None:
        t = tuple(float(v) for v in tint) + ((1.0,) if len(tint) == 3 else ())
        self.icons.append((str(name), float(x), float(y), float(w), float(h), t, bool(image)))

    def text(self, x, y, text: str, size: float = 13.0, colour=TEXT) -> None:
        """Text with its top-left corner at (x, y)."""
        c = tuple(int(round(max(0.0, min(1.0, v)) * 255)) for v in colour) + ((255,) if len(colour) == 3 else ())
        self.texts.append((float(x), float(y), float(size), c, str(text)))

    # --- sending --------------------------------------------------------------------

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            k: (np.concatenate(v) if v else np.zeros(0, np.float32)).astype(np.float32, copy=False)
            for k, v in self.parts.items()
        }

    def digest(self) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        for k, a in self.arrays().items():
            h.update(k.encode())
            h.update(a.tobytes())
        h.update(repr(self.icons).encode())
        h.update(repr(self.texts).encode())
        return h.digest()


class Layers:
    """Sends layers to the host, skipping any whose content did not change."""

    def __init__(self, host) -> None:
        self.host = host
        self.sent: dict[str, tuple[int, bytes]] = {}
        self.offsets: dict[str, tuple[float, float, float]] = {}
        self.generation = 0  # bumps when the host forgets every layer: drawers that keep layers of their own send them again

    def send(self, name: str, z: int, d: DrawList) -> bool:
        if d.key is not None:
            key = (z, ("key", d.key))
        elif not d.icons and not d.texts and not any(d.parts.values()):
            key = (z, "empty")  # most handlers draw nothing most frames: no arrays, no hash
        else:
            key = (z, d.digest())
        offset = tuple(float(v) for v in d.offset)
        changed = self.sent.get(name) != key
        if changed:
            self.host.set_draw(name, z, **d.arrays(), icons=d.icons, texts=d.texts)
            self.sent[name] = key
        # The offset after the content: the host makes a new layer when the
        # content first arrives, with no offset, and an offset sent for a layer
        # it does not have yet goes nowhere (every skin block and
        # the Box preview, sent in grid space, drew at the grid's zero corner).
        if offset != (0.0, 0.0, 0.0) and (changed or self.offsets.get(name, (0.0, 0.0, 0.0)) != offset):
            self.host.set_draw_offset(name, *offset)
            self.offsets[name] = offset
        elif offset == (0.0, 0.0, 0.0) and self.offsets.get(name, (0.0, 0.0, 0.0)) != offset:
            self.host.set_draw_offset(name, *offset)
            self.offsets[name] = offset
        return changed

    def clear(self, name: str) -> None:
        if name in self.sent:
            self.host.clear_draw(name)
            del self.sent[name]
        self.offsets.pop(name, None)

    def forget(self) -> None:
        self.sent.clear()
        self.offsets.clear()
        self.generation += 1
