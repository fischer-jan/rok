"""The shader gallery's layout: a panel of cards, one per shader, in groups.

The panel sits in the middle of the view. A title row on top (with a button
that opens your shader folder), then each group under its caption, cards in
rows, and a footer that says what the card under the mouse does. When the
groups do not fit, the middle part scrolls.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geom2d import Rect, in_rect

GROUP_ORDER = ("nature", "water", "weathering", "patterns", "hard_surface")
YOURS = ""  # the group of your own shaders: files at the top of your folder

CARD_W = 104
CARD_H = 120
THUMB = 88  # the thumbnail's side on a card
GAP = 10
PAD = 18  # inside the panel's edge
MARGIN = 24  # between the panel and the view's edge
TITLE_H = 42
CAPTION_H = 24
FOOTER_H = 44
MAX_COLUMNS = 7
MIN_COLUMNS = 3
SCROLL_STEP = 60  # pixels per wheel step, at scale 1


@dataclass(frozen=True)
class Entry:
    name: str  # as Rok names it: nature/rocks
    label: str  # Rocks
    about: str  # one line on what it makes


@dataclass(frozen=True)
class Card:
    entry: Entry
    rect: Rect
    thumb: Rect


@dataclass(frozen=True)
class GalleryLayout:
    rect: Rect  # the whole panel
    title: Rect
    folder_button: Rect
    body: Rect  # the part that scrolls; draw and hit only inside it
    footer: Rect
    captions: tuple[tuple[str, Rect], ...]
    cards: tuple[Card, ...]
    scroll: float
    max_scroll: float
    scale: float


def group_of(name: str) -> str:
    return name.rpartition("/")[0]


def grouped(entries) -> list[tuple[str, list[Entry]]]:
    """The entries by group: Rok's groups in GROUP_ORDER, any other folder
    after them, your own last. Each group keeps the entries' order."""
    groups: dict[str, list[Entry]] = {}
    for entry in entries:
        groups.setdefault(group_of(entry.name), []).append(entry)
    order = [g for g in GROUP_ORDER if g in groups]
    order += sorted(g for g in groups if g not in GROUP_ORDER and g != YOURS)
    if YOURS in groups:
        order.append(YOURS)
    return [(g, groups[g]) for g in order]


def caption(group: str) -> str:
    """"hard_surface" -> "Hard surface"; your own group -> "Your shaders"."""
    if group == YOURS:
        return "Your shaders"
    text = group.replace("_", " ").replace("/", " / ")
    return text[:1].upper() + text[1:]


def gallery_layout(scale: float, region_width: int, region_height: int, entries, scroll: float = 0.0) -> GalleryLayout:
    """The panel for a region of this size. `scroll` is how far the body is
    scrolled down, in pixels; it is clamped to what there is to scroll."""
    s = float(scale)
    card_w, card_h, gap, pad = CARD_W * s, CARD_H * s, GAP * s, PAD * s
    room = region_width - 2 * MARGIN * s - 2 * pad + gap
    columns = int(max(MIN_COLUMNS, min(MAX_COLUMNS, room // (card_w + gap))))
    width = columns * card_w + (columns - 1) * gap + 2 * pad
    groups = grouped(entries)
    content = 0.0
    for _group, items in groups:
        rows = -(-len(items) // columns)
        content += CAPTION_H * s + rows * (card_h + gap)
    title_h, footer_h = TITLE_H * s, FOOTER_H * s
    height = min(region_height - 2 * MARGIN * s, title_h + content + footer_h)
    height = max(height, title_h + footer_h + card_h)
    x = max(0.0, (region_width - width) / 2.0)
    y = max(0.0, (region_height - height) / 2.0)
    rect = (x, y, width, height)
    top = y + height
    title = (x, top - title_h, width, title_h)
    footer = (x, y, width, footer_h)
    body = (x, y + footer_h, width, height - title_h - footer_h)
    max_scroll = max(0.0, content - body[3])
    scroll = min(max(0.0, float(scroll)), max_scroll)
    button_w = 120 * s
    folder_button = (x + width - pad - button_w, top - title_h + 10 * s, button_w, title_h - 20 * s)

    captions = []
    cards = []
    cursor = body[1] + body[3] + scroll  # the top of the next row, in region pixels
    for group, items in groups:
        cursor -= CAPTION_H * s
        captions.append((caption(group), (x + pad, cursor, width - 2 * pad, CAPTION_H * s)))
        for k, entry in enumerate(items):
            row, col = divmod(k, columns)
            cx = x + pad + col * (card_w + gap)
            cy = cursor - (row + 1) * (card_h + gap) + gap
            card = (cx, cy, card_w, card_h)
            side = THUMB * s
            thumb = (cx + (card_w - side) / 2.0, cy + card_h - side - 6 * s, side, side)
            cards.append(Card(entry, card, thumb))
        cursor -= (-(-len(items) // columns)) * (card_h + gap)
    return GalleryLayout(rect, title, folder_button, body, footer, tuple(captions), tuple(cards), scroll, max_scroll, s)


def card_at(layout: GalleryLayout, x: float, y: float) -> Card | None:
    """The card under a point, only where the body shows it."""
    if not in_rect(layout.body, x, y):
        return None
    return next((card for card in layout.cards if in_rect(card.rect, x, y)), None)


def visible(layout: GalleryLayout, rect: Rect) -> bool:
    """True when some of `rect` shows in the body."""
    bx, by, bw, bh = layout.body
    x, y, w, h = rect
    return y < by + bh and y + h > by and x < bx + bw and x + w > bx


def in_panel(layout: GalleryLayout, x: float, y: float) -> bool:
    return in_rect(layout.rect, x, y)


def on_folder_button(layout: GalleryLayout, x: float, y: float) -> bool:
    return in_rect(layout.folder_button, x, y)


def scrolled(layout: GalleryLayout, steps: int) -> float:
    """The scroll after `steps` wheel steps (positive: down the list)."""
    return min(max(0.0, layout.scroll + steps * SCROLL_STEP * layout.scale), layout.max_scroll)


def thumb_file(name: str) -> str:
    """The thumbnail's file name for a shader: nature/rocks -> nature__rocks.png."""
    return name.replace("/", "__") + ".png"
