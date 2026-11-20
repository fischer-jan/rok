"""Voxel-art scenes: the rules and the feedback that keep an LLM making voxel art
(a small grid, each voxel and colour placed on purpose) rather than smooth 3D
shapes sampled into a fine grid.

Every scene the LLM starts through the MCP (rok_scene do=new, or the New
command) is an art scene. In it:

- the grid is at most ART_MAX_SIDE on any side;
- a budget caps the voxels and the colours (ART_COLOURS by default);
- every colour in use has a name (rok_edit do=palette names them, with ramps);
- the piece starts in a blockout stage: one colour, until it is one connected
  mass that still reads when shown at icon size;
- rok_voxels_set writes at most art_set_cells(grid) cells per call: ART_SET_CELLS
  on a small grid, ART_SET_CELLS_BIG on a grid over ART_DEFAULT on a side;
- a finished picture (rok_render) needs one connected part and the verdict
  "voxel art".

An edit that breaks a rule is undone and the reply says why. Opened files and
scenes the app made stay free; they get the verdict and the craft line only.
"""

from __future__ import annotations

import numpy as np

from rok.core import craft, png, voxel_views

ART_DEFAULT = 32
ART_MAX_SIDE = 64
ART_COLOURS = 16
ART_SET_CELLS = 64
ART_SET_CELLS_BIG = 256  # a 64-cell scene needs bigger groups for the same hand placement
CRAFT_MAX_VOXELS = 200_000  # the craft line is skipped above this (it stays fast)
# The verdict: a piece past any of these is voxelized 3D, not voxel art. Hand-made
# models in the tests reach 13,555 voxels, an 80-cell side and 24 colours; a scene
# sampled from smooth shapes had 1.3 million voxels, a 232-cell side and 96 colours.
VERDICT_VOXELS = 50_000
VERDICT_SIDE = 128
VERDICT_COLOURS = 48
BLOCKOUT_KEEP = 60.0  # percent of the front silhouette that must survive at icon size to end the blockout
ICON = 16  # pixels on the longer side of the icon the blockout must read at
RAMP_STEPS = (2, 6)
FLAT_TARGET = 240  # pixels for the longer side of a flat view, at a whole number of pixels per cell
FLAT_GAP = 8
MARK = (255, 56, 216)
MARK_MAX_SHARE = 0.3  # past this share of the voxels changed, the flat views show plain colours


def art_set_cells(size) -> int:
    """The cells one rok_voxels_set call may write in an art scene of this grid."""
    return ART_SET_CELLS if max(size) <= ART_DEFAULT else ART_SET_CELLS_BIG


def verdict(grids: list[np.ndarray]) -> tuple[str, list[str]]:
    """("voxel art" or "voxelized 3D", the reasons) for a scene's grids."""
    voxels = sum(int((g != 0).sum()) for g in grids)
    longest = 0
    used: set[int] = set()
    for g in grids:
        filled = np.argwhere(g != 0)
        if len(filled):
            longest = max(longest, int((filled.max(axis=0) - filled.min(axis=0) + 1).max()))
        used |= {int(c) for c in np.unique(g[g != 0])}
    reasons = []
    if voxels > VERDICT_VOXELS:
        reasons.append(f"{voxels} voxels (voxel art stays under {VERDICT_VOXELS})")
    if longest > VERDICT_SIDE:
        reasons.append(f"a {longest}-cell side (voxel art stays under {VERDICT_SIDE})")
    if len(used) > VERDICT_COLOURS:
        reasons.append(f"{len(used)} colours (voxel art uses a small palette, under {VERDICT_COLOURS})")
    return ("voxelized 3D" if reasons else "voxel art"), reasons


RAMP_STEP = 0.14  # HSV value between two neighbouring steps of a ramp
RAMP_SPAN = 0.56  # the most value a whole ramp spans, lightest to darkest
RAMP_HUE_SHIFT = 0.025  # how far (of a full turn) the ends lean: light toward warm, dark toward cool
WARM_HUE, COOL_HUE = 0.11, 0.64


def _lean(h: float, target: float, amount: float) -> float:
    """Hue h moved `amount` of a turn toward `target`, the short way round, never past it."""
    d = (target - h + 0.5) % 1.0 - 0.5
    return (h + max(-abs(d), min(abs(d), amount)) * (1 if d >= 0 else -1)) % 1.0


def ramp(hex_: str, steps: int) -> list[tuple[int, int, int]]:
    """`steps` shades of a colour, lightest first. The hue stays: each step changes the
    brightness (HSV value), the ends lean a little, light toward warm and dark toward
    cool, and only past full brightness does a light step turn paler. So a red ramp
    stays red (no pink) and a skin ramp stays skin (no grey-brown)."""
    import colorsys

    base = [int(hex_[k:k + 2], 16) / 255.0 for k in (0, 2, 4)]
    if steps == 1:
        return [tuple(int(round(v * 255)) for v in base)]
    h, sat, val = colorsys.rgb_to_hsv(*base)
    span = min(RAMP_STEP * (steps - 1), RAMP_SPAN)
    out = []
    for k in range(steps):
        t = k / (steps - 1) * 2 - 1  # -1 lightest .. 1 darkest
        vk, sk = val - t * span / 2, sat
        if vk > 1.0:  # no brighter than full: the rest of the step pales the colour
            sk, vk = max(0.0, sat - (vk - 1.0) * 1.2), 1.0
        if t < 0:
            sk *= 0.9
        vk = max(0.08, vk)
        hk = _lean(h, WARM_HUE if t < 0 else COOL_HUE, abs(t) * RAMP_HUE_SHIFT) if sat > 0.05 else h
        out.append(tuple(int(round(v * 255)) for v in colorsys.hsv_to_rgb(hk, sk, vk)))
    return out


def delta_e(a, b) -> float:
    """A plain colour distance in Lab (CIE76)."""
    def lab(c):
        rgb = np.asarray(c, dtype=np.float64) / 255.0
        rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ rgb
        xyz /= np.array([0.95047, 1.0, 1.08883])
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
    return float(np.linalg.norm(lab(a) - lab(b)))


class FlatView:
    """One flat view of the last picture, for rok_probe: the cell under a pixel."""

    FACE = {"front": "-y", "right": "+x", "top": "top"}

    def __init__(self, side, scale, lo, shape, colours, depth, model) -> None:
        self.side, self.scale, self.lo, self.shape = side, scale, np.asarray(lo), shape
        self.colours, self.depth, self.model = colours, depth, model

    def cell(self, x: float, y: float):
        """(cell, colour) under the pixel (x, y) of this view, or None for empty."""
        r, c = int(y // self.scale), int(x // self.scale)
        rows, cols = self.colours.shape
        if not (0 <= r < rows and 0 <= c < cols) or self.colours[r, c] == 0:
            return None
        d = int(self.depth[r, c])
        sx, sy, sz = self.shape
        if self.side == "front":
            local = (c, d, rows - 1 - r)
        elif self.side == "right":
            local = (sx - 1 - d, c, rows - 1 - r)
        else:
            local = (c, rows - 1 - r, sz - 1 - d)
        return [int(v) for v in self.lo + np.asarray(local)], int(self.colours[r, c])


class ArtTools:
    """The art-scene part of the Agent (rok/agent.py)."""

    _art: dict | None = None

    # --- the art scene ----------------------------------------------------------------

    @property
    def art(self) -> dict | None:
        """The rules of the current scene when the LLM started it, else None."""
        a = self._art
        if a is None or self.doc.scene is None or a["scene"] is not self.doc.scene:
            return None
        return a

    def start_art(self, budget=None) -> str:
        from rok.agent import ToolError

        b = dict(budget or {})
        unknown = set(b) - {"voxels", "colours"}
        if unknown:
            raise ToolError(f"budget takes voxels and colours; not {', '.join(sorted(unknown))}")
        colours = int(b.get("colours", ART_COLOURS))
        if not 1 <= colours <= ART_COLOURS:
            raise ToolError(f"budget colours is 1 to {ART_COLOURS}")
        voxels = b.get("voxels")
        if voxels is not None and int(voxels) < 1:
            raise ToolError("budget voxels must be 1 or more")
        self._art = {"scene": self.doc.scene, "voxels": None if voxels is None else int(voxels),
                     "colours": colours, "stage": "blockout"}
        cap = "" if voxels is None else f"{int(voxels)} voxels, "
        return (f"Voxel art scene: budget {cap}{colours} colours; grid at most {ART_MAX_SIDE} per side. "
                "Stage: blockout (one colour until the shape is one connected mass that reads at icon size; "
                "the feedback of each edit tells when). Name colours with rok_edit do=palette before you use them.")

    def set_budget(self, budget) -> str:
        """Change the caps of the open art scene; what is in it stays."""
        from rok.agent import ToolError

        a = self.art
        if a is None:
            raise ToolError("do=budget is for a voxel art scene (one started with rok_scene do=new); "
                            "this scene has no budget")
        b = dict(budget or {})
        if not b or set(b) - {"voxels", "colours"}:
            raise ToolError('budget takes voxels and colours: {"voxels": 2000, "colours": 12}')
        if "colours" in b:
            colours = int(b["colours"])
            if not 1 <= colours <= ART_COLOURS:
                raise ToolError(f"budget colours is 1 to {ART_COLOURS}")
            in_use = len(self._used())
            if colours < in_use:
                raise ToolError(f"the scene uses {in_use} colours; the budget cannot go under that")
            a["colours"] = colours
        if "voxels" in b:
            voxels = int(b["voxels"])
            total = sum(self._used().values())
            if voxels < max(1, total):
                raise ToolError(f"the scene has {total} voxels; the budget cannot go under that")
            a["voxels"] = voxels
        cap = "no voxel cap" if a["voxels"] is None else f"{a['voxels']} voxels"
        return f"Budget is now {cap}, {a['colours']} colours."

    def colour_names(self) -> dict[int, str]:
        """{slot: name} of every named colour of the scene."""
        scene = self.doc.scene
        if scene is None:
            return {}
        store = scene.colours
        out = {}
        for slot in range(1, 256):
            ident = store.id_of(slot)
            if ident is not None and store.entries[ident].name:
                out[slot] = store.entries[ident].name
        return out

    def _used(self) -> dict[int, int]:
        """{slot: voxels} over every model."""
        counts: dict[int, int] = {}
        for m in self.doc.scene.models:
            v = m.grid.voxels
            values, n = np.unique(v[v != 0], return_counts=True)
            for c, k in zip(values.tolist(), n.tolist()):
                counts[c] = counts.get(c, 0) + k
        return counts

    def art_violations(self) -> list[str]:
        """What breaks the rules of the art scene now (empty when nothing)."""
        a = self.art
        if a is None:
            return []
        out = []
        for m in self.doc.scene.models:
            if max(m.grid.size) > ART_MAX_SIDE:
                out.append(f"a grid of {' x '.join(str(s) for s in m.grid.size)} is over {ART_MAX_SIDE} on a side")
        used = self._used()
        total = sum(used.values())
        if a["voxels"] is not None and total > a["voxels"]:
            out.append(f"{total} voxels is {total - a['voxels']} over the budget of {a['voxels']}")
        if len(used) > a["colours"]:
            out.append(f"{len(used)} colours is over the budget of {a['colours']}")
        names = self.colour_names()
        unnamed = sorted(c for c in used if c not in names)
        if unnamed:
            out.append(f"colour{'s' if len(unnamed) > 1 else ''} {', '.join(map(str, unnamed))} "
                       f"{'have' if len(unnamed) > 1 else 'has'} no name: name colours with rok_edit "
                       'do=palette first ({"colours": {"stone": "#8a8478"}, "ramps": {"stone": 3}})')
        if a["stage"] == "blockout" and len(used) > 1:
            out.append(f"the blockout uses {len(used)} colours; it takes one until the feedback reports the "
                       "blockout done (one connected mass that reads at icon size)")
        return out

    def enforce_art(self) -> None:
        """Undo the last step when it broke the rules, and say why."""
        from rok.agent import ToolError

        why = self.art_violations()
        if why:
            self.app_run("rok.undo")
            raise ToolError("Refused, and undone: " + "; ".join(why) + ".")

    def art_line(self) -> str:
        """The budget and stage line of an art scene's feedback."""
        a = self.art
        if a is None:
            return ""
        used = self._used()
        total = sum(used.values())
        cap = "" if a["voxels"] is None else f"/{a['voxels']}"
        return f"Voxel art: voxels {total}{cap}, colours {len(used)}/{a['colours']}, stage {a['stage']}."

    def blockout_status(self) -> str:
        """Ends the blockout when it is ready; a line on where it stands."""
        a = self.art
        if a is None or a["stage"] != "blockout":
            return ""
        model = self.doc.model
        v = model.grid.voxels
        if not (v != 0).any():
            return "Blockout: the model is empty."
        if len(self._used()) > 1:
            return ""  # only inside a batch, before its end refuses the second colour
        parts = craft.parts(v, limit=2)
        crop, _ = voxel_views.cropped(v)
        sil = voxel_views.project(crop, "front")[0] != 0
        factor = max(1, -(-max(sil.shape) // ICON))  # cells per icon pixel
        keep = craft.survival(sil, factor) if factor > 1 else 100.0
        if len(parts) == 1 and keep >= BLOCKOUT_KEEP:
            a["stage"] = "colour"
            return (f"Blockout done: one connected mass, and the front keeps {keep}% as a {ICON}-pixel icon. "
                    "Colours are open.")
        need = []
        if len(parts) > 1:
            need.append(f"join the {len(parts)} parts into one mass")
        if keep < BLOCKOUT_KEEP:
            need.append(f"make the front read at icon size (as a {ICON}-pixel icon it keeps {keep}%; "
                        f"{BLOCKOUT_KEEP:g}% needed: bolder shapes, fewer one-cell details)")
        return "Blockout not done: " + "; ".join(need) + "."

    # --- the palette -------------------------------------------------------------------

    def palette(self, colours=None, ramps=None) -> str:
        """Name colours, each as one slot or as a ramp of shades, lightest first."""
        from rok.agent import ToolError
        from rok.app import object as voxel_object

        if not isinstance(colours, dict) or not colours:
            raise ToolError('palette needs colours, a map of names to hex: {"wood": "#8a5a2b", "leaf": "#4a7a3a"}; '
                            'ramps gives shades: {"wood": 4}')
        ramps = dict(ramps or {})
        stray = set(ramps) - set(colours)
        if stray:
            raise ToolError(f"ramps names {', '.join(sorted(stray))}, which is not in colours")
        scene = self.doc.scene
        store = scene.colours
        used = scene.used_counts()
        names = self.colour_names()
        taken = set(names) | {c for c in range(1, 256) if used[c] > 0}
        free = [c for c in range(1, 256) if c not in taken]
        wanted = []
        for name, hex_ in colours.items():
            h = str(hex_).lstrip("#")
            if len(h) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in h):
                raise ToolError(f"{name}: hex must be like #8a5a2b")
            steps = int(ramps.get(name, 1))
            if steps != 1 and not RAMP_STEPS[0] <= steps <= RAMP_STEPS[1]:
                raise ToolError(f"{name}: a ramp has {RAMP_STEPS[0]} to {RAMP_STEPS[1]} steps")
            shades = ramp(h.lower(), steps)
            for k, rgb in enumerate(shades):
                wanted.append((str(name) if steps == 1 else f"{name} {k + 1}", rgb))
        existing = {n: s for s, n in names.items()}
        if self.art is not None:
            count = len(set(existing) | {n for n, _ in wanted})
            if count > self.art["colours"]:
                raise ToolError(f"that makes {count} named colours; the budget is {self.art['colours']}")
        lines = []
        for name, rgb in wanted:
            slot = existing.get(name)
            if slot is None:
                if not free:
                    raise ToolError("no free palette slot is left")
                slot = free.pop(0)
                ident = store.claim(slot, rgb)
            else:
                ident = store.id_of(slot)
                store.edit(ident, colour=rgb)
            store.edit(ident, name=name)
            self.doc.made.add(slot)
            lines.append(f"{name} = {slot} #{''.join(f'{v:02x}' for v in rgb)}")
        voxel_object.update_palette()
        self.doc.undo.push_colours()  # its own undo step: undoing a later edit keeps the names
        warn = []
        allc = [(n, tuple(int(v) for v in scene.palette[s, :3])) for s, n in self.colour_names().items()]
        for i in range(len(allc)):
            for j in range(i + 1, len(allc)):
                if delta_e(allc[i][1], allc[j][1]) < 5:
                    warn.append(f"{allc[i][0]} and {allc[j][0]} look almost the same")
        text = "Palette: " + ", ".join(lines) + ". Use the numbers as colours."
        if warn:
            text += " Warning: " + "; ".join(warn[:4]) + "."
        return text

    def palette_warnings(self) -> list[str]:
        """Colours used on one voxel only: noise rather than a choice. Two or more (a pair
        of eyes, a row of lamps) are an accent, so they get no warning."""
        used = self._used()
        names = self.colour_names()
        rare = [names.get(c, str(c)) for c, n in used.items() if n < 2]
        if rare and len(used) > 2:
            return [f"colour{'s' if len(rare) > 1 else ''} {', '.join(rare[:5])} "
                    f"{'are' if len(rare) > 1 else 'is'} used on a single voxel"]
        return []

    # --- flat pictures -----------------------------------------------------------------

    def flat_png(self, model, changed: np.ndarray | None = None, scale: int | None = None) -> tuple[bytes, str]:
        """The front, right and top views cell by cell, side by side, and under them
        the front at 16 and 32 pixels (as an icon shows it, then four times larger).
        Changed cells (model cells) are outlined. Returns (PNG, a line on the layout)."""
        v = model.grid.voxels
        crop, lo = voxel_views.cropped(v)
        pal = self.doc.scene.palette
        mark, most = None, False
        if changed is not None and len(changed):
            mark = np.zeros(v.shape, dtype=bool)
            c = changed[np.all((changed >= 0) & (changed < np.asarray(v.shape)), axis=1)]
            mark[tuple(c.T)] = True
            mark = mark[lo[0]:lo[0] + crop.shape[0], lo[1]:lo[1] + crop.shape[1], lo[2]:lo[2] + crop.shape[2]]
            if mark.sum() > MARK_MAX_SHARE * max(1, int((crop != 0).sum())):
                mark, most = None, True
        if scale is None:
            scale = max(2, min(16, FLAT_TARGET // max(crop.shape)))
        views, layout = [], []
        self.last_cameras = []
        x = 0
        for side in voxel_views.SIDES:
            colours, depth = voxel_views.project(crop, side)
            img = voxel_views.scaled(voxel_views.rgb(colours, depth, pal), scale)
            self.last_cameras.append((side, float(x), 0.0, float(img.shape[1]), float(img.shape[0]),
                                      FlatView(side, scale, lo, crop.shape, colours, depth, model)))
            if mark is not None:
                img = self._outline(img, self._visible_marks(crop, mark, side, colours, depth), scale)
            views.append(img)
            layout.append(f"{side.upper()} x {x}..{x + img.shape[1]}")
            x += img.shape[1] + FLAT_GAP
        front_c, front_d = voxel_views.project(crop, "front")
        front = voxel_views.rgb(front_c, front_d, pal)
        icons = [voxel_views.icon(front, front_c, px) for px in (16, 32)]
        icons = icons + [voxel_views.scaled(i, 4) for i in icons]
        top_h = max(i.shape[0] for i in views)
        low_h = max(i.shape[0] for i in icons)
        w = max(x - FLAT_GAP, sum(i.shape[1] for i in icons) + FLAT_GAP * (len(icons) - 1))
        canvas = np.zeros((top_h + FLAT_GAP + low_h, w, 3), np.uint8)
        canvas[...] = (20, 20, 22)
        x = 0
        for img in views:
            canvas[:img.shape[0], x:x + img.shape[1]] = img
            x += img.shape[1] + FLAT_GAP
        x = 0
        for img in icons:
            canvas[top_h + FLAT_GAP:top_h + FLAT_GAP + img.shape[0], x:x + img.shape[1]] = img
            x += img.shape[1] + FLAT_GAP
        rgba = np.concatenate([canvas, np.full(canvas.shape[:2] + (1,), 255, np.uint8)], axis=2)
        hi = lo + np.asarray(crop.shape) - 1
        line = (f"Flat views, {scale} pixels per cell, of cells {lo.tolist()}..{hi.tolist()}: "
                + "; ".join(layout) + f" (y 0..{top_h}). Under them the front at icon size: 16 and 32 pixels, "
                "then the same four times larger. Front: x to the right, z up. Right: y to the right, z up. "
                "Top: x to the right, y up." + (" Changed cells are outlined in magenta." if mark is not None else "")
                + (" Most cells changed, so they are not outlined." if most else ""))
        return png.encode(np.ascontiguousarray(rgba)), line

    @staticmethod
    def _visible_marks(crop, mark, side, colours, depth) -> np.ndarray:
        """For each pixel of a view, whether the cell it shows is marked."""
        m = mark.astype(np.uint8)
        mc, _ = voxel_views.project(np.where(crop != 0, 1 + m, 0).astype(np.uint8), side)
        return mc == 2

    @staticmethod
    def _outline(img: np.ndarray, marks: np.ndarray, scale: int) -> np.ndarray:
        out = img.copy()
        if scale < 3:
            out[np.repeat(np.repeat(marks, scale, 0), scale, 1)] = MARK
            return out
        for r, c in np.argwhere(marks):
            y0, x0 = r * scale, c * scale
            out[y0, x0:x0 + scale] = MARK
            out[y0 + scale - 1, x0:x0 + scale] = MARK
            out[y0:y0 + scale, x0] = MARK
            out[y0:y0 + scale, x0 + scale - 1] = MARK
        return out

    def text_views(self, model) -> tuple[str, dict]:
        """The front, right and top views as rows of characters, with the legend."""
        v = model.grid.voxels
        crop, lo = voxel_views.cropped(v)
        legend = voxel_views.legend_for(crop)
        names = self.colour_names()
        hi = lo + np.asarray(crop.shape) - 1
        parts = []
        data = {"legend": {ch: c for c, ch in legend.items()}, "low": lo.tolist(), "high": hi.tolist()}
        axes = {"front": ("x", 0, "z", 2), "right": ("y", 1, "z", 2), "top": ("x", 0, "y", 1)}
        for side in voxel_views.SIDES:
            rows = voxel_views.text(voxel_views.project(crop, side)[0], legend)
            ca, ci, ra, ri = axes[side]
            parts.append(f"{side.upper()} (columns {ca} {lo[ci]}..{hi[ci]}, rows {ra} {hi[ri]} down to {lo[ri]}):\n"
                         + "\n".join(rows))
            data[side] = rows
        key = ", ".join(f"{ch} = {c}" + (f" {names[c]}" if c in names else "") for c, ch in legend.items())
        text = ("Text views of cells " f"{lo.tolist()}..{hi.tolist()} ('.' empty; {key}). A TOP row reads like a row "
                "of a rok_voxels_set layer.\n" + "\n".join(parts))
        return text, data
