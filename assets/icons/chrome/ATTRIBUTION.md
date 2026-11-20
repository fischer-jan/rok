# Rok icons: where they come from

Every glyph in `1x/` and `wheel/` is AI-generated. It was made on
21 September 2026 with xAI's Grok Imagine (`grok-imagine-image-quality`)
through the xAI API, one image per glyph, from the subject lines in
`scripts/icons/glyphs.tsv` and a style sheet made the same way (see below).
`scripts/icons/make_icons.py` turned the images into the masks here. Nothing
in them is from a third-party icon set. `ROTATE_DEGREES` was added the same
way on 3 October 2026, and `SELECT_ALL`, `DESELECT`, `GROW`, `SHRINK`, `COPY`
and `PASTE` on 11 October 2026, then `MODE_CLONE` and `MODE_GRADIENT`.

## Terms and rights

The images were generated through the xAI API under xAI's Terms of Service -
Enterprise in the version of 14 August 2026, which was in force on all the
dates above (<https://x.ai/legal/terms-of-service-enterprise> links the
previous versions; section "Rights in Input and Output"). Under that section the customer
"owns all right, title, and interest in the Output in perpetuity", and xAI
assigns to the customer all of its own rights in the output. Two limits
apply: the output may not be used to train AI models, and it may not be
presented as made by a human. The version of 27 October 2026 keeps the same
clause.

A picture made by a machine from a prompt may hold no copyright at all,
because copyright needs a human author. So the MIT licence in `LICENSE`
covers these pictures as far as Rok holds rights in them. Where it holds
none, they are free for everyone in any case.

The scripts `scripts/icons/*_glyphs.py` draw some glyphs in code instead,
for example the four tool glyphs `CUBE_PLUS`, `CUBE_MINUS`, `BRUSH` and
`MARQUEE` (`arm_glyphs.py`), so that their cubes have the same size.
`scene_glyphs.py` draws the scene's `PICK_LINKED`, `ALIGN`, `ARRAY` and `CARVE`.
`stroke_glyphs.py` draws `STROKE`, the Stroke Designer's button.
`brick_glyphs.py` draws Brick mode's glyphs (`BRICK_PLACE`,
`BRICK_PULL`, `BRICK_PAINT`, `BRICK_PICK`, `BRICK_CONVERT`, `BRICK_TO_VOXELS`), around one shared brick.

The other pictures under `assets/`:

- `assets/icon/rok.png` and `Rok.icns`: drawn in code by
  `scripts/icons/app/render_icon.py`.
- `assets/icons/welcome/logo.png`: the app icon's cube alone, drawn by the
  same script.
- The style sheet the glyphs were generated from was made with the same
  image model. It is not part of the repo; `scripts/icons/generate.sh` expects
  it at `scripts/icons/reference.png`, which git ignores.
