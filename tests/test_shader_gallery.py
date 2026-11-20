"""The shader gallery's layout (rok/core/shader_gallery.py)."""

from pathlib import Path

from rok.core import xs
from rok.core.shader_gallery import (
    GROUP_ORDER,
    Entry,
    caption,
    card_at,
    gallery_layout,
    grouped,
    in_panel,
    on_folder_button,
    scrolled,
    thumb_file,
    visible,
)

SHADERS = Path(__file__).resolve().parent.parent / "rok" / "shaders"
THUMBS = Path(__file__).resolve().parent.parent / "assets" / "icons" / "shaders"


def _entries():
    out = []
    for path in sorted(SHADERS.rglob("*.txt")):
        name = f"{path.parent.name}/{path.stem}"
        shader = xs.load(path, name)
        out.append(Entry(name, xs.label(path.stem), shader.about))
    return out


def test_groups_come_in_order_with_your_own_last():
    entries = [Entry("mine", "Mine", ""), Entry("patterns/a", "A", ""), Entry("nature/b", "B", ""),
               Entry("odd/c", "C", "")]
    assert [g for g, _items in grouped(entries)] == ["nature", "patterns", "odd", ""]
    assert caption("hard_surface") == "Hard surface" and caption("") == "Your shaders"


def test_every_shader_of_the_set_has_a_card_and_a_thumbnail():
    entries = _entries()
    lay = gallery_layout(1.0, 1400, 2000, entries)
    assert [c.entry.name for c in lay.cards] == [e.name for g, items in grouped(entries) for e in items]
    assert [text for text, _r in lay.captions] == [caption(g) for g in GROUP_ORDER]
    for entry in entries:
        assert (THUMBS / thumb_file(entry.name)).is_file(), entry.name


def test_cards_sit_inside_the_panel_without_overlapping():
    lay = gallery_layout(2.0, 2900, 3600, _entries())  # tall enough for every group: no scroll
    x, y, w, h = lay.rect
    rects = [c.rect for c in lay.cards]
    for cx, cy, cw, ch in rects:
        assert x <= cx and cx + cw <= x + w and lay.body[1] <= cy and cy + ch <= lay.body[1] + lay.body[3]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]
    assert lay.max_scroll == 0.0


def test_a_short_view_scrolls_and_only_shown_cards_are_hit():
    entries = _entries()
    lay = gallery_layout(1.0, 900, 500, entries)
    assert lay.max_scroll > 0 and lay.rect[3] <= 500
    first = lay.cards[0]
    cx, cy = first.rect[0] + 5, first.rect[1] + 5
    assert card_at(lay, cx, cy) is first
    down = gallery_layout(1.0, 900, 500, entries, scroll=scrolled(lay, 100))
    assert down.scroll == lay.max_scroll
    assert not visible(down, down.cards[0].rect)
    assert visible(down, down.cards[-1].rect)
    assert card_at(down, *(down.cards[0].rect[:2])) is None  # scrolled out of the body


def test_the_panel_knows_its_folder_button_and_its_outside():
    lay = gallery_layout(1.0, 1400, 1000, _entries())
    bx, by, bw, bh = lay.folder_button
    assert on_folder_button(lay, bx + 2, by + 2) and in_panel(lay, bx + 2, by + 2)
    assert not in_panel(lay, 1, 1)


def test_the_thumbnails_are_stored_to_load_fast():
    """Every row unfiltered (filter byte 0): Rok's decoder then only unzips
    (the gallery opened a second late with filtered PNGs)."""
    import struct
    import zlib

    for path in THUMBS.glob("*.png"):
        data = path.read_bytes()
        pos, idat, width, height = 8, b"", 0, 0
        while pos < len(data):
            length, kind = struct.unpack(">I4s", data[pos:pos + 8])
            body = data[pos + 8:pos + 8 + length]
            if kind == b"IHDR":
                width, height = struct.unpack(">II", body[:8])
            elif kind == b"IDAT":
                idat += body
            pos += 12 + length
        raw = zlib.decompress(idat)
        stride = width * 4 + 1
        assert all(raw[row * stride] == 0 for row in range(height)), path.name


def test_a_personal_shader_shows_the_png_beside_its_file(monkeypatch):
    from rok.app import shader_gallery, xs_ops

    folder = Path(__file__).parent / "fixtures" / "marker_shaders"
    monkeypatch.setattr(xs_ops, "user_shader_folder", lambda: folder)
    assert shader_gallery.thumb_path("plume") == folder / "plume.png"
    assert shader_gallery.thumb_path("nature/rocks") == THUMBS / thumb_file("nature/rocks")
