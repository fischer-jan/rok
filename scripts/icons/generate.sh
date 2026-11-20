#!/usr/bin/env bash
# Generate one glyph with an image model (the set was made with xAI's Grok Imagine,
# grok-imagine-image-quality).
#   ROK_IMAGE_TOOL=/path/to/generate scripts/icons/generate.sh NAME "subject"
#     -> scripts/icons/gen/NAME.png
# ROK_IMAGE_TOOL is a command that reads the prompt on stdin and takes
# --quality --aspect 1:1 --resolution 1k -i REFERENCE -o OUT.
# The subject lines live in glyphs.tsv. reference.png keeps every glyph in one style;
# it is not in the repo (git ignores it): ask the maintainer for it.
# Then: uv run python scripts/icons/make_icons.py NAME ...   (or no names: every glyph)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
name="$1"; desc="$2"
tool="${ROK_IMAGE_TOOL:?set ROK_IMAGE_TOOL to an image generator command (see the top of this file)}"
mkdir -p "$HERE/gen"
$tool --quality --aspect 1:1 --resolution 1k \
  -i "$HERE/reference.png" -o "$HERE/gen/$name.png" <<PROMPT
One single icon, centred in a square image and filling about 65 percent of it, on a plain flat dark charcoal background exactly like the reference image. No frame, no border, no text, no label, no shadow, no gradient, no second icon. Style exactly as the reference image: clean isometric geometry drawn with a bold pale warm grey outline of even weight, and the one part the tool acts on filled flat ember orange. Subject: $desc
PROMPT
sips -s format png "$HERE/gen/$name.png" --out "$HERE/gen/$name.png" >/dev/null 2>&1  # the API returns JPEG bytes
