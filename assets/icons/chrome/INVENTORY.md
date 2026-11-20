# Rok icons

Twenty-four glyphs, isometric cubes in one construction: an outline and one
filled face, the part the tool acts on. Each PNG is a mask, white with the
alpha as the ink: the outline alpha 1, the fill alpha 0.5. The drawing tints
the mask grey at rest, near white under the mouse and ember when active, so
the fill is a lighter tone of whatever the ink is.

| Folder | Size | Used for |
| --- | --- | --- |
| `1x/NAME.png` | 64 px | every glyph at UI scale 1 |
| `1x/NAME@2x.png` | 128 px | every glyph at UI scale 2 (Retina) |
| `wheel/NAME.png`, `NAME@2x.png` | 96 / 192 px | the eight mode glyphs on the Tab wheel |

Tools: `CUBE_PLUS`, `CUBE_MINUS`, `BRUSH`, `MARQUEE`. Modes: `VOXEL`, `FACE`,
`MODE_BOX`, `LINE`, `CENTER`, `RECT`, `LASSO`, `MODE_PATTERN`. Primitives: `PRIMITIVES` on the item at the end of Attach's row; the grid
(each glyph named as its solid): `BOX`, `CYLINDER`, `SPHERE`, `DOME`,
`CAPSULE`, `HALF_CYLINDER`, `PYRAMID`, `CONE`, `DIAMOND`, `OCTAGON_PRISM`, `HEX_PRISM`,
`TUBE`, `WEDGE`, `ROOF`, `STAIRS`, `TORUS`, `ROCK`, `HILL`. Pattern chip and flyout: `PATTERN`, `TURN`,
`TIP`. Section: `CUT`. Selection pills: `HOLLOW`, `SPLIT`, `BRIDGE`, `DELETE`. Model flyout: `SMOOTH` (Smooth), `ROTATE_DEGREES` (Rotate by Degrees). The scene's rail:
`PICK` (the Pick tool), `SLICE` (the Cut tool), `FLIP` (the Flip chip), `PICK_LINKED`, `ALIGN`, `ARRAY` and `CARVE` (Tools items drawn in code, `scripts/icons/scene_glyphs.py`), `LEVEL_SCENE` and `LEVEL_MODEL` (the level switch) and `FROM_FILE` (New from a file; New's own button
and its empty model use `CUBE_PLUS`). The Layers list: `LINK` (a linked copy; a click makes it unique). Attach's Clone mode: `MODE_CLONE`. Paint's Gradient mode: `MODE_GRADIENT`. The Stroke Designer's button over the Size field: `STROKE` (drawn in code, `scripts/icons/stroke_glyphs.py`). Brick mode: the tools `BRICK_PLACE`, `BRICK_PULL`, `BRICK_PAINT`, `BRICK_PICK` and the Convert chips `BRICK_CONVERT`, `BRICK_TO_VOXELS` (drawn in code, `scripts/icons/brick_glyphs.py`). The star's Selection arm: `SELECT_ALL`, `DESELECT`, `GROW`, `SHRINK`, `COPY`, `PASTE` (with `INVERSE`, `DELETE` and `MASK`).

To add or redo a glyph: a subject line in `scripts/icons/glyphs.tsv`,
`scripts/icons/generate.sh NAME "subject"`, then
`uv run python scripts/icons/make_icons.py NAME`, and look at
`scripts/icons/contact.png` at 64 and 32 px before committing.
