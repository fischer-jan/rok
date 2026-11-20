"""Draw Rok's app icon: a glowing voxel cube with a few small cubes around it on a night-blue tile.

    uv run --no-project --with pillow --with numpy python scripts/icons/app/render_icon.py assets/icon/rok.png [LOGO.png]

Then build assets/icon/Rok.icns from rok.png with sips and iconutil. With a second
path it also writes the welcome screen's logo (assets/icons/welcome/logo.png): the
same cube alone on a clear ground, 256 pixels.
"""
import math, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = sys.argv[1] if len(sys.argv) > 1 else "rok.png"
LOGO = sys.argv[2] if len(sys.argv) > 2 else None
S = 4                                  # supersampling
N, T, OFF = 1024 * S, 824 * S, 100 * S  # canvas, tile, margin (macOS icon grid)

def tile_mask():
    y, x = np.mgrid[0:N, 0:N].astype(np.float32) + 0.5
    r = 0.225 * T
    cx = np.clip(x, OFF + r, OFF + T - r); cy = np.clip(y, OFF + r, OFF + T - r)
    d = np.hypot(x - cx, y - cy)
    inside = np.clip(r - d + 0.5, 0, 1) * ((x > OFF - 1) & (x < OFF + T + 1) & (y > OFF - 1) & (y < OFF + T + 1))
    return inside, x, y

def background(x, y):
    t = np.clip((y - OFF) / T, 0, 1)[..., None]      # 0 top, 1 bottom
    top, bot = np.array([0.10, 0.16, 0.34]), np.array([0.03, 0.05, 0.13])
    rgb = top + (bot - top) * t
    # a soft light behind the cube
    cx, cy = N * 0.5, N * 0.47
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (0.2 * T) ** 2))[..., None]
    rgb = rgb + g * np.array([0.10, 0.32, 0.42])
    return np.clip(rgb, 0, 1)

def iso(cx, cy, e):
    """Corner points of an isometric cube with edge e whose centre is at (cx, cy)."""
    dx, dy = e * math.cos(math.radians(30)), e * 0.5
    top = [(cx, cy - e), (cx + dx, cy - e + dy), (cx, cy - e + 2 * dy), (cx - dx, cy - e + dy)]
    c = (cx, cy - e + 2 * dy)
    left = [(cx - dx, cy - e + dy), c, (c[0], c[1] + e), (cx - dx, cy - e + dy + e)]
    right = [c, (cx + dx, cy - e + dy), (cx + dx, cy - e + dy + e), (c[0], c[1] + e)]
    return top, left, right

def cube(draw, cx, cy, e, top, left, right):
    t, l, r = iso(cx, cy, e)
    draw.polygon(t, fill=top); draw.polygon(l, fill=left); draw.polygon(r, fill=right)

inside, x, y = tile_mask()
rgb = background(x, y)
img = Image.fromarray((np.dstack([rgb, inside]) * 255).astype(np.uint8), "RGBA")

# the glow: the main cube's silhouette, blurred
glow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
cube(ImageDraw.Draw(glow), N * 0.5, N * 0.5, 0.27 * T, *[(120, 235, 255, 255)] * 3)
glow = glow.filter(ImageFilter.GaussianBlur(0.07 * T))
img = Image.alpha_composite(img, glow)

layer = Image.new("RGBA", (N, N), (0, 0, 0, 0))
d = ImageDraw.Draw(layer)
MINT = ((214, 250, 255, 255), (118, 214, 236, 255), (52, 150, 196, 255))
cube(d, N * 0.5, N * 0.5, 0.27 * T, *MINT)
# the specks: (x, y as a part of the tile, edge as a part of the tile, alpha)
for fx, fy, fe, a in [(0.80, 0.24, 0.055, 235), (0.19, 0.30, 0.038, 200), (0.78, 0.74, 0.042, 215),
                      (0.26, 0.80, 0.028, 170), (0.62, 0.15, 0.022, 150), (0.12, 0.60, 0.020, 140)]:
    cols = [c[:3] + (a,) for c in MINT]
    cube(d, OFF + fx * T, OFF + fy * T, fe * T, *cols)
img = Image.alpha_composite(img, layer)

# keep everything inside the tile
alpha = np.minimum(np.asarray(img)[..., 3], (inside * 255).astype(np.uint8))
arr = np.asarray(img).copy(); arr[..., 3] = alpha
Image.fromarray(arr, "RGBA").resize((1024, 1024), Image.LANCZOS).save(OUT)
print("wrote", OUT)

if LOGO:
    # The cube fills the picture from top to bottom, with a thin clear rim.
    e = 0.27 * T
    half = int(e * 1.04)
    alone = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    cube(ImageDraw.Draw(alone), N * 0.5, N * 0.5, e, *MINT)
    c = N // 2
    alone.crop((c - half, c - half, c + half, c + half)).resize((256, 256), Image.LANCZOS).save(LOGO)
    print("wrote", LOGO)
