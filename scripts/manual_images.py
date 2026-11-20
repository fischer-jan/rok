"""Turn the manual's screenshots into the WebP files the manual shows.

  uv run --with pillow --no-project python scripts/manual_images.py SHOTS_DIR

SHOTS_DIR is what `scripts/manual_shots.py` wrote. The files land in
`manual/img`. Screenshots shrink to 1800 pixels wide; the looks are a crop
round the model at 640.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "manual"

# manual/img/<name>.webp <- SHOTS_DIR/<scene>/<file>.png
SHOTS = {
    "welcome": "welcome/02_welcome2", "new_dialog": "welcome/03_new_dialog",
    "window": "bird/01_window", "ghost_attach": "bird/02_ghost_attach", "ghost_erase": "bird/03_ghost_erase",
    "attach_flyout": "bird/04_attach_flyout", "model_flyout": "bird/05_model_flyout", "layers_flyout": "bird/06_layers_flyout",
    "colour_popup": "bird/07_colour_popup", "star": "bird/08_star", "command_palette": "bird/09_command_palette",
    "section": "bird/10_section", "selection": "bird/11_selection", "bright": "bird/24_bright",
    "view_settings": "bird/25_view_settings", "shader_gallery": "bird/26_shader_gallery", "quad": "bird/27_quad",
    "scene": "bird/28_scene", "new_flyout": "bird/29_new_flyout", "gizmo": "bird/30_gizmo",
    "scene_tools_flyout": "bird/31_scene_tools_flyout", "key_cursor": "bird/32_key_cursor",
    "primitives_flyout": "bird_more/01_primitives_flyout", "mode_face": "bird_more/02_mode_face",
    "mode_box": "bird_more/03_mode_box", "mode_line": "bird_more/04_mode_line", "mirror_ghost": "bird_more/06_mirror_ghost",
    "section_open": "bird_more/07_section_open", "cage": "bird_more/08_cage", "history_strip": "bird_more/09_history_strip",
    "render_settings": "bird_more/10_render_settings", "rendered": "bird_more/11_rendered", "cad": "bird_more/12_cad",
    "stroke_designer": "bird_more/13_stroke_designer",
    "brick_mode": "brick/02_brick_mode", "brick_converted": "brick/03_brick_converted", "brick_ghost": "brick/04_brick_ghost",
    "brick_placed": "brick/05_brick_placed", "brick_turned": "brick/06_brick_turned", "brick_random": "brick/07_brick_random",
    "brick_pull": "brick/08_brick_pull", "brick_off": "brick/09_brick_off",
    "walk_scene": "anim/01_walk_scene", "animate": "anim/02_animate", "playing": "anim/04_playing",
    "car_scene": "car/02_scene", "robot_model": "robot/01_model", "robot_scene": "robot/02_scene",
}
LOOKS = ("iso", "soft", "night", "ceramic", "plain", "studio", "sun", "paper", "clay", "flat", "blueprint", "toon")
LOOK_CROP = (700, 330, 2100, 1030)  # the bird, in a 2940 x 1602 screenshot


def save(im: Image.Image, out: Path, width: int, quality: int) -> int:
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, "WEBP", quality=quality, method=6)
    return out.stat().st_size


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    shots = Path(sys.argv[1])
    total = 0
    for name, rel in SHOTS.items():
        src = shots / f"{rel}.png"
        if not src.exists():
            print("missing", src)
            continue
        total += save(Image.open(src), MANUAL / "img" / f"{name}.webp", 1800, 84)
    for k, look in enumerate(LOOKS, start=12):
        src = shots / "bird" / f"{k:02d}_look_{look}.png"
        if src.exists():
            total += save(Image.open(src).crop(LOOK_CROP), MANUAL / "img" / f"look_{look}.webp", 640, 82)
    print(f"{total // 1024} KB written to {MANUAL}")


if __name__ == "__main__":
    main()
