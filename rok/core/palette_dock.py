"""Palette v2: the stacked dock card, bottom left above the HUD (new UI only).

Pure geometry, no drawing: the host draws what `dock_layout` returns. Region
pixels, y up (as `hud.strip_layout`), origin at the card's bottom-left.

Card, top to bottom: the current-colour header (40 px chip, hex + name,
voxel count, three 24 px buttons, the collapse chevron), then the sections
In use, Unused and Favourites, each under a plain heading (11 px semibold,
count right-aligned, 8 px above, a 1 px divider below). An empty section has
no heading. Clicking a heading folds it. Open sections share the card.
The card stays within 40 % of the viewport by scrolling them, and it grows
past that only when one row of each open section does not fit, so In use
and Unused stay on screen together. The header (the current colour) is
never hidden. Collapsed, the card is the chip plus the last 8 recent colours.
Up to 40 colours (used + unused + favourites) swatches are 20 px, ten per
row; above 40 they are 16 px, twelve per row. The card is always 234 px wide
at scale 1 (inner width 10*20 + 9*2 = 218): the header, headings, dividers,
chevrons and scroll clips use that inner width in both modes, and the dense
214 px grid (12*16 + 11*2) is centred in it (2 px each side). Every swatch's
hit cell is the swatch plus its gap, centred on the swatch (18 px dense,
22 px at 20 px), so the cells tile the grid with no dead gaps; drawing keeps
the true swatch rect. Every other target is at least 20 px. Collapsed recent
colours stay 20 px. The voxel count has its own rect on header line 2,
ending `LABEL_CLEAR` before the first button.
"""

from __future__ import annotations

from dataclasses import dataclass

Rect = tuple[int, int, int, int]

SWATCH = 20
SWATCH_DENSE = 16  # above DENSE_AFTER colours in the card
DENSE_AFTER = 40
GAP = 2
COLUMNS = 10
COLUMNS_DENSE = 12  # 12*16 + 11*2 = 214, centred in the 218 px inner width
CHIP = 40
BUTTON = 24
BUTTON_GAP = 4
HEX_LINE = 16  # line 1 of the header: the hex (and name), editable with the editor open
CHEVRON = 20
PAD = 8
HEADING_SIZE = 11  # semibold, light grey
HEADING_H = 24  # the heading's hit band: a target (>= 20), on the 8 px grid
BELOW_DIVIDER = 8  # divider to the first swatch row
EDITOR_FRACTION = 0.60  # the card's cap while the editor is open
HEADING_ABOVE = 8
DIVIDER = 1
MAX_FRACTION = 0.40
RECENT = 8
BUTTONS = ("NEW", "EYEDROP", "FAVOURITE")
SECTIONS = ("USED", "UNUSED", "FAV")
TITLES = {"USED": "In use", "UNUSED": "Unused", "FAV": "Favourites"}
HOVER_TIP = "Ctrl drag to move · Shift drag for a gradient · Alt click to pick"

# Rings around the current swatch: 2 px ember outside, 1 px dark inside;
# hover is a soft white ring.
RING_EMBER = (1.0, 0.45, 0.15, 1.0)
RING_EMBER_W = 2
RING_INNER = (0.05, 0.05, 0.05, 1.0)
RING_INNER_W = 1
RING_HOVER = (1.0, 1.0, 1.0, 0.45)
CARD_BG = (0.11, 0.105, 0.10, 0.92)  # rail-style charcoal
CARD_RADIUS = 10
HEADING_COLOUR = (0.78, 0.78, 0.78, 1.0)
HEADING_INK = (1.0, 1.0, 1.0, 0.80)  # section headings
COUNT_INK = (1.0, 1.0, 1.0, 0.60)  # their counts
SHADOW_ALPHA = 0.35  # the plate's drop shadow: 0/4 px, blur 12
TOP_HIGHLIGHT = (1.0, 1.0, 1.0, 0.06)
DOT = 4  # material dot, px
DOT_INSET = 2
SWATCH_OUTLINE = (1.0, 1.0, 1.0, 0.15)  # 1 px on every swatch, so dark ones read on the plate
LABEL_CLEAR = 8  # clearance between the label text and the buttons


@dataclass(frozen=True)
class Section:
    name: str  # "USED", "UNUSED" or "FAV"
    title: str  # "In use"
    count: int
    heading: Rect  # the whole clickable heading band
    chevron: Rect
    divider: Rect  # 1 px line under the heading
    folded: bool
    swatches: tuple[tuple[int, Rect], ...]  # (key, rect); empty when folded
    scroll: bool = False  # the swatches scroll inside `clip`
    clip: Rect | None = None
    content_height: int = 0  # full height of the rows, for scrolling
    offset: int = 0  # px In use is scrolled by
    keys: tuple[int, ...] = ()  # every colour in the section, also when folded or scrolled out
    hits: tuple[tuple[int, Rect], ...] = ()  # (key, hit cell): swatch + gap, centred on the swatch


@dataclass(frozen=True)
class DockLayout:
    card: Rect
    collapsed: bool
    swatch: int
    chip: Rect
    label: Rect  # hex + name, line 1
    count: Rect  # the voxel count, line 2; ends LABEL_CLEAR before the first button
    buttons: tuple[tuple[str, Rect], ...]
    chevron: Rect
    editor: Rect | None  # inline editor, unfolding upward from the header
    sections: tuple[Section, ...] = ()
    recent: tuple[tuple[int, Rect], ...] = ()  # collapsed only
    recent_hits: tuple[tuple[int, Rect], ...] = ()  # their hit cells
    radius: int = CARD_RADIUS
    gap: int = GAP  # px between swatches

    def section(self, name: str) -> Section | None:
        return next((s for s in self.sections if s.name == name), None)

    def targets(self) -> list[tuple[str, Rect]]:
        """Every clickable thing. A swatch's target is its hit cell (swatch plus
        gap, centred); every other target is >= 20 px."""
        out: list[tuple[str, Rect]] = [("CHIP", self.chip), ("CHEVRON", self.chevron)]
        out += [(n, r) for n, r in self.buttons]
        for s in self.sections:
            out.append(("HEADING:" + s.name, s.heading))
            out += [(f"{s.name}:{k}", r) for k, r in s.hits]
        out += [(f"RECENT:{k}", r) for k, r in self.recent_hits]
        return out


def hit_cell(r: Rect, gap: int) -> Rect:
    """A swatch's hit cell: the swatch plus its gap, centred on the swatch."""
    lo = gap // 2
    return (r[0] - lo, r[1] - lo, r[2] + gap, r[3] + gap)


def _cells(items, gap: int) -> tuple[tuple[int, Rect], ...]:
    return tuple((k, hit_cell(r, gap)) for k, r in items)


def short_count(n: int) -> str:
    """A voxel count for the header: 999, 1.2k, 12k, 1.2M, 12M, 1.0B, 1.2B (cut, not rounded)."""
    n = max(0, int(n))
    for div, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1000, "k")):
        if n >= div:
            tenths = n * 10 // div
            return f"{tenths // 10}{suffix}" if tenths >= 100 else f"{tenths // 10}.{tenths % 10}{suffix}"
    return str(n)


def count_text(n: int, fits=None) -> str:
    """"12,345 voxels" when `fits(text)` says it fits the count rect (always, without
    `fits`), else the short form "12k voxels"."""
    n = max(0, int(n))
    exact = f"{n:,} voxels"
    if fits is None or fits(exact):
        return exact
    return f"{short_count(n)} voxels"


def _rows(n: int, columns: int) -> int:
    return (n + columns - 1) // columns if columns else 0


def dock_layout(
    viewport_height: float,
    used: list[int] | tuple[int, ...],
    unused: list[int] | tuple[int, ...] = (),
    favourites: int = 0,
    *,
    x0: int = 0,
    y0: int = 0,
    scale: float = 1.0,
    folds: dict[str, bool] | None = None,
    collapsed: bool = False,
    recent: list[int] | tuple[int, ...] = (),
    editor_height: int = 0,
    scroll_offset: int = 0,
    scroll_offsets: dict[str, int] | None = None,
    reveal: int | None = None,
) -> DockLayout:
    """The card for these colours. `folds` is your saved fold state per section;
    `editor_height` > 0 opens the inline editor above the header.
    `scroll_offsets` is how far each section is scrolled, in px. `scroll_offset`
    is the In use value when the dict does not name it."""
    s = float(scale)
    px = lambda v: int(round(v * s))  # noqa: E731
    folds = dict(folds or {})
    keys = {"USED": list(dict.fromkeys(int(i) for i in used)),
            "UNUSED": list(dict.fromkeys(int(i) for i in unused)),
            "FAV": list(range(int(favourites)))}
    total = sum(len(v) for v in keys.values())
    dense = (not collapsed) and total > DENSE_AFTER
    cols = COLUMNS_DENSE if dense else COLUMNS
    sw = px(SWATCH_DENSE if dense else SWATCH)
    gap, pad = px(GAP), px(PAD)
    # Card and header always use the comfortable 10*20 grid, so going dense
    # never shrinks the plate or moves the chip, hex, count, buttons or chevron.
    inner = COLUMNS * px(SWATCH) + (COLUMNS - 1) * gap
    chip_s, btn = px(CHIP), px(BUTTON)
    width = inner + 2 * pad  # the grid plus 8 px each side; the header fits itself in
    header_h = chip_s + 2 * pad
    cap = int(viewport_height * MAX_FRACTION)
    grid_w = cols * sw + (cols - 1) * gap
    grid_x = x0 + pad + (inner - grid_w) // 2  # dense: 2 px each side of 214 in 218

    def header(top_y: int):
        chip = (x0 + pad, top_y - pad - chip_s, chip_s, chip_s)
        right = x0 + pad + inner  # the inner width's right edge: chevron and buttons end here
        lx = chip[0] + chip_s + pad
        # Line 1: the hex across the width next to the chip, the chevron at the far right.
        line1_y = chip[1] + chip_s - px(HEX_LINE)
        chev = (right - px(CHEVRON), line1_y, px(CHEVRON), px(CHEVRON))
        label = (lx, line1_y, max(0, chev[0] - px(4) - lx), px(HEX_LINE))
        # Line 2: the count on the left, the three buttons on the right, 4 px apart.
        bgap = px(BUTTON_GAP)
        bx = right - len(BUTTONS) * btn - (len(BUTTONS) - 1) * bgap
        buttons = tuple((n, (bx + i * (btn + bgap), chip[1], btn, btn)) for i, n in enumerate(BUTTONS))
        count = (lx, chip[1], max(0, buttons[0][1][0] - px(LABEL_CLEAR) - lx), btn)
        return chip, label, count, buttons, chev

    if collapsed:
        rec = list(dict.fromkeys(int(i) for i in recent))[:RECENT]
        rsw = px(SWATCH)
        h = header_h + (rsw + pad if rec else 0)
        top = y0 + h
        chip, label, count, buttons, chev = header(top)
        recents = tuple((k, (x0 + pad + i * (rsw + gap), y0 + pad, rsw, rsw)) for i, k in enumerate(rec))
        eh = int(editor_height) if editor_height > 0 else 0
        editor = (x0, top, width, eh) if eh else None
        return DockLayout((x0, y0, width, h + eh), True, rsw, chip, label, count, buttons, chev, editor, (), recents,
                          recent_hits=_cells(recents, gap), gap=gap)

    # Every heading and divider sits on the 8 px grid: 8 above the heading, a
    # 24 px heading band, the 1 px divider, 8 below it to the swatches.
    head_band = px(HEADING_ABOVE) + px(HEADING_H) + px(DIVIDER) + px(BELOW_DIVIDER)
    step = sw + gap  # one swatch row

    def rows_h(rows: int) -> int:
        return max(0, rows * sw + (rows - 1) * gap)

    present = [n for n in SECTIONS if keys[n]]
    folded = {n: bool(folds.get(n, False)) for n in present}
    # A fold is only yours. A long list scrolls inside its section; the height
    # cap never closes one section to make room for another.
    shown = {n: 0 if folded[n] else _rows(len(keys[n]), cols) for n in present}

    def card_body(rows_of: dict[str, int]) -> int:
        h = header_h + pad
        for n in present:
            h += head_band + rows_h(rows_of[n])
        return h

    eh = int(editor_height) if editor_height > 0 else 0
    # The editor and the header always show whole; the sections share what is left.
    cap = int(viewport_height * (EDITOR_FRACTION if eh else MAX_FRACTION)) - eh
    if card_body(shown) > cap:
        open_names = [n for n in present if not folded[n] and _rows(len(keys[n]), cols) > 0]
        wants = {n: _rows(len(keys[n]), cols) for n in open_names}
        budget = cap - (header_h + pad + len(present) * head_band)
        shown = {n: 0 for n in present}
        used_px = 0
        while True:
            # The section with the fewest rows showing takes the next one, so
            # In use and Unused grow together instead of one taking the card.
            candidates = [
                n for n in open_names
                if shown[n] < wants[n] and used_px + (sw if shown[n] == 0 else step) <= budget
            ]
            if not candidates:
                break
            n = min(candidates, key=lambda name: (shown[name], SECTIONS.index(name)))
            used_px += sw if shown[n] == 0 else step
            shown[n] += 1
        for n in open_names:  # one row each, even when that puts the card past the cap
            if shown[n] == 0:
                shown[n] = 1
    offsets = {str(k): int(v) for k, v in (scroll_offsets or {}).items()}
    offsets.setdefault("USED", int(scroll_offset))
    total_h = card_body(shown)
    top = y0 + total_h
    chip, label, count, buttons, chev = header(top)
    y = top - header_h
    out = []
    for n in present:
        y -= px(HEADING_ABOVE)
        hy = y - px(HEADING_H)
        heading = (x0 + pad, hy, inner, px(HEADING_H))
        cy = hy + (px(HEADING_H) - px(CHEVRON)) // 2
        chevr = (x0 + pad + inner - px(CHEVRON), cy, px(CHEVRON), px(CHEVRON))
        div = (x0 + pad, hy - px(DIVIDER), inner, px(DIVIDER))
        y = hy - px(DIVIDER) - px(BELOW_DIVIDER)
        rects: list[tuple[int, Rect]] = []
        clip = None
        content = 0
        sc = False
        off = 0
        if not folded[n]:
            rows = _rows(len(keys[n]), cols)
            content = rows_h(rows)
            for k, key in enumerate(keys[n]):
                c, r = k % cols, k // cols
                rects.append((key, (grid_x + c * step, y - (r + 1) * sw - r * gap, sw, sw)))
            visible = shown[n]
            if visible < rows:
                sc = True
                vis_h = rows_h(visible)
                clip = (x0 + pad, y - vis_h, inner, vis_h)
                first = max(0, min(int(round(offsets.get(n, 0) / step)) if step else 0, rows - visible))
                if reveal is not None and int(reveal) in keys[n]:
                    r = keys[n].index(int(reveal)) // cols  # keep the current colour's row in view
                    first = min(max(first, r - visible + 1), r)
                off = first * step
                # Whole rows only: scrolled rows move up, the ones outside the clip are not drawn or hit.
                rects = [(k_, (r_[0], r_[1] + off, r_[2], r_[3])) for k_, r_ in rects]
                rects = [(k_, r_) for k_, r_ in rects if r_[1] >= clip[1] and r_[1] + r_[3] <= clip[1] + clip[3]]
                y -= vis_h
            else:
                y -= content
        out.append(Section(n, TITLES[n], len(keys[n]), heading, chevr, div, folded[n],
                           tuple(rects), sc, clip, content, off, tuple(keys[n]), _cells(rects, gap)))
    # The editor is part of the card: the plate grows upward above the header,
    # the header (and the current chip) stays where it is.
    editor = (x0, top, width, eh) if eh else None
    return DockLayout((x0, y0, width, total_h + eh), False, sw, chip, label, count, buttons, chev, editor, tuple(out),
                      gap=gap)


def hit(layout: DockLayout, mx: float, my: float) -> str | None:
    """What is under a region pixel: 'CHIP', 'CHEVRON', a button, 'HEADING:USED',
    'USED:<key>' (inside a scroll clip only), 'RECENT:<key>', 'CARD' or None."""
    def inside(r):
        return r[0] <= mx < r[0] + r[2] and r[1] <= my < r[1] + r[3]
    if not inside(layout.card):
        return None
    for s in layout.sections:
        if s.clip is not None and not inside(s.clip):
            continue
        for k, r in s.hits:
            if inside(r):
                return f"{s.name}:{k}"
    for name, r in layout.targets():
        if ":" in name and not name.startswith("HEADING:") and not name.startswith("RECENT:"):
            continue
        if inside(r):
            return name
    if layout.editor is not None and inside(layout.editor):
        return "EDITOR"
    return "CARD"


def fold_state(text: str) -> dict[str, bool]:
    """Parse the saved 'USED=0,UNUSED=1,FAV=0' fold setting."""
    out = {}
    for part in (text or "").split(","):
        name, _, v = part.partition("=")
        if name.strip() in SECTIONS:
            out[name.strip()] = v.strip() == "1"
    return out


def fold_text(folds: dict[str, bool]) -> str:
    return ",".join(f"{n}={int(bool(folds[n]))}" for n in SECTIONS if n in folds)


def shifted(layout: DockLayout, dy: float) -> DockLayout:
    """The card as drawn mid-animation: everything moved by `dy` (the shown
    height minus the final height) and the plate that much taller, its bottom
    edge fixed, so the card grows and shrinks upward."""
    if not dy:
        return layout
    import dataclasses

    up = lambda r: None if r is None else (r[0], r[1] + dy, r[2], r[3])  # noqa: E731
    secs = tuple(dataclasses.replace(
        s, heading=up(s.heading), chevron=up(s.chevron), divider=up(s.divider), clip=up(s.clip),
        swatches=tuple((k, up(r)) for k, r in s.swatches),
        hits=tuple((k, up(r)) for k, r in s.hits)) for s in layout.sections)
    x, y, w, h = layout.card
    return dataclasses.replace(
        layout, card=(x, y, w, max(0.0, h + dy)), chip=up(layout.chip), label=up(layout.label), count=up(layout.count),
        buttons=tuple((n, up(r)) for n, r in layout.buttons), chevron=up(layout.chevron),
        editor=up(layout.editor), sections=secs, recent=tuple((k, up(r)) for k, r in layout.recent),
        recent_hits=tuple((k, up(r)) for k, r in layout.recent_hits))


def gradient_preview(first, last, steps: int) -> list[tuple]:
    """The colours a Shift drag would make: `steps` + 1 colours from first to last."""
    steps = max(1, int(steps))
    return [tuple(int(round(a + (b - a) * k / steps)) for a, b in zip(first, last)) for k in range(steps + 1)]
