"""The shader gallery: a panel of cards, one per shader, in the middle of the view.

Opened from the top bar's Shaders and from Cmd+K. Each card shows a rendered
thumbnail of what the shader makes (assets/icons/shaders, made by
scripts/make_shader_thumbs.py) and its name; the footer says what the card
under the mouse does. A click opens the shader's settings (rok.xs_run). The
wheel scrolls when the groups do not fit; Esc, a right-click or a click
outside the panel closes it. The layout lives in rok.core.shader_gallery.
"""

from __future__ import annotations

from .opkit import Action, ops, register

from pathlib import Path

import numpy as np

from rok.core import shader_gallery as sg
from rok.core import xs
from .gfx import pen
from . import object as voxel_object
from . import overlay
from .handlers import registry
from .opkit import CONTEXT
from .paint import clip_text as _clip_text
from .paint import round_outline as _round_outline
from .paint import round_rect as _round
from .paint import text as _text

THUMBS = Path(__file__).resolve().parents[2] / "assets" / "icons" / "shaders"

PANEL_BG = (0.060, 0.056, 0.053, 0.97)
PANEL_EDGE = (0.228, 0.212, 0.196, 1.0)
CARD_BG = (1.0, 1.0, 1.0, 0.045)
CARD_HOVER = (0.208, 0.192, 0.180, 1.0)
STAGE_BG = (0.0, 0.0, 0.0, 0.22)
SCRIM = (0.0, 0.0, 0.0, 0.35)  # over the view behind the gallery
BUTTON_BG = (1.0, 1.0, 1.0, 0.07)
EMBER = (0.910, 0.365, 0.227, 1.0)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)
TEXT_FAINT = (0.450, 0.430, 0.410, 1.0)

_session: "_Session | None" = None
_thumbs: dict[str, str | None] = {}  # shader name -> atlas key, None without a picture


class _Session:
    __slots__ = ("window", "region_ptr", "scroll", "hover", "entries")

    def __init__(self, window, region_ptr) -> None:
        self.window = window
        self.region_ptr = region_ptr
        self.scroll = 0.0
        self.hover: str | None = None  # a shader name, or "FOLDER" for the button
        self.entries = _entries()  # read once when the gallery opens, not on every move


def active() -> bool:
    return _session is not None


def _entries() -> list[sg.Entry]:
    from . import xs_ops  # late: xs_ops registers after this module

    out = []
    for name, path in xs_ops.shader_files().items():
        try:
            shader = xs.load(path, name)
        except OSError:
            continue
        out.append(sg.Entry(name, xs.label(name.rpartition("/")[2]), shader.about))
    return out


def _layout(region, scroll: float) -> sg.GalleryLayout:
    entries = _session.entries if _session is not None else _entries()
    return sg.gallery_layout(overlay.ui_scale(), region.width, region.height, entries, scroll)


def _view_region(window):
    """The view's region: Rok has one view."""
    return CONTEXT.region


def _region_of(session: _Session):
    return CONTEXT.region


@register
class ShaderGalleryAction(Action):
    command_id = "rok.shader_gallery"
    title = "Shaders"
    tooltip = "Browse the shaders: a card for each, with what it makes; click one to run it"
    @classmethod
    def poll(cls, context):
        return voxel_object.document.scene is not None

    def invoke(self, context, event):
        global _session
        region = _view_region(context.window)
        if region is None:
            return {"CANCELLED"}
        if _session is not None:
            _close()
            return {"CANCELLED"}
        _session = _Session(context.window, region.as_pointer())
        for entry in _session.entries:
            _thumb(entry.name)  # all pictures now (a few ms), not one by one while drawing
        context.prefs.add_modal(self)
        voxel_object.redraw_3d_views()
        return {"RUNNING"}

    def modal(self, context, event):
        session = _session
        if session is None:
            return {"CANCELLED"}
        region = _region_of(session)
        if region is None:
            _close()
            return {"CANCELLED"}
        kind, value = event.type, event.value
        mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
        layout = _layout(region, session.scroll)
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            card = sg.card_at(layout, mx, my)
            hover = "FOLDER" if sg.on_folder_button(layout, mx, my) else (card.entry.name if card else None)
            if hover != session.hover:
                session.hover = hover
                region.tag_redraw()
            return {"RUNNING"}
        if kind in {"WHEEL_UP", "WHEEL_DOWN", "TRACKPAD_PAN"}:
            if kind == "TRACKPAD_PAN":
                steps = -event.dy / 30.0
            else:
                steps = 1 if kind == "WHEEL_DOWN" else -1
            session.scroll = sg.scrolled(layout, steps)
            card = sg.card_at(_layout(region, session.scroll), mx, my)
            session.hover = card.entry.name if card else None
            region.tag_redraw()
            return {"RUNNING"}
        if kind == "LEFT_MOUSE" and value == "PRESS":
            if sg.on_folder_button(layout, mx, my):
                _close()
                ops.rok.xs_folder()
                return {"FINISHED"}
            card = sg.card_at(layout, mx, my)
            if card is not None:
                _close()
                if ops.rok.xs_run.poll():
                    ops.rok.shader_panel("INVOKE", shader=card.entry.name)
                else:
                    self.report({"INFO"}, "Open a model first: a shader runs on the model")
                return {"FINISHED"}
            if not sg.in_panel(layout, mx, my):
                _close()
                return {"CANCELLED"}
            return {"RUNNING"}
        if value == "PRESS" and kind in {"ESCAPE", "RIGHT_MOUSE", "TAB"}:
            _close()
            return {"CANCELLED"}
        if kind == "FOCUS_LOST":
            _close()
            return {"CANCELLED"}
        # Keys and clicks elsewhere stay with the gallery while it is open.
        return {"RUNNING"}

    def cancel(self, context):
        _close()


def _close() -> None:
    global _session
    _session = None
    voxel_object.redraw_3d_views()


# --- drawing ------------------------------------------------------------------------


def thumb_path(name: str) -> Path:
    """Where a shader's thumbnail is: Rok's own in assets/icons/shaders, or for
    a shader of yours a PNG of the same name beside its file."""
    path = THUMBS / sg.thumb_file(name)
    if path.is_file():
        return path
    from . import xs_ops

    source = xs_ops.shader_files().get(name)
    return source.with_suffix(".png") if source is not None else path


def _thumb(name: str) -> str | None:
    """The atlas key of a shader's thumbnail, loaded on first use; None when it has none."""
    if name in _thumbs:
        return _thumbs[name]
    key = None
    path = thumb_path(name)
    app = CONTEXT.app
    if app is not None and path.is_file():
        key = f"shaders/{name}"
        if not app.icons.ensure(key, path):
            key = None
    _thumbs[name] = key
    return key


def _draw_image(key: str, rect) -> None:
    x, y, w, h = (float(v) for v in rect)
    draw_image(key, (x, y, w, h))


def draw_image(key: str, rect) -> None:
    """A picture from the atlas in region pixels (origin bottom left)."""
    from .gfx import current

    x, y, w, h = rect
    current().icon(key, x, float(CONTEXT.region.height) - y - h, w, h, (1.0, 1.0, 1.0, 1.0), True)


def _draw_card(card: sg.Card, scale: float, hovered: bool) -> None:
    x, y, w, h = card.rect
    radius = 8 * scale
    _round(card.rect, radius, CARD_HOVER if hovered else CARD_BG)
    if hovered:
        _round_outline(card.rect, radius, EMBER, 1.5 * scale)
    _round(card.thumb, 6 * scale, STAGE_BG)
    tex = _thumb(card.entry.name)
    if tex is not None:
        _draw_image(tex, card.thumb)
    else:
        # Your own shader: its initial on the stage.
        tx, ty, tw, th = card.thumb
        _text(card.entry.label[:1].upper(), tx + tw / 2, ty + th / 2, 34 * scale, TEXT_FAINT, "centre")
    label_cy = y + (card.thumb[1] - y) / 2.0
    label = _clip_text(card.entry.label, w - 8 * scale, 12 * scale)
    _text(label, x + w / 2, label_cy, 12 * scale, TEXT if hovered else TEXT_DIM, "centre")


def draw(region, scale: float) -> None:
    """The gallery, over everything else in the view. Reads `_session`."""
    session = _session
    if session is None or region.as_pointer() != session.region_ptr:
        return
    layout = _layout(region, session.scroll)
    s = layout.scale
    if True:
        # A dim veil over the view, then the panel.
        pen.rect_fill((0, 0, region.width, region.height), SCRIM)
        _round(layout.rect, 14 * s, PANEL_BG)
        _round_outline(layout.rect, 14 * s, PANEL_EDGE, 1.0 * s)
        tx, ty, tw, th = layout.title
        pad = sg.PAD * s
        _text("Shaders", tx + pad, ty + th / 2, 17 * s, TEXT)
        fb = layout.folder_button
        _round(fb, fb[3] / 2, CARD_HOVER if session.hover == "FOLDER" else BUTTON_BG)
        _text("Your folder", fb[0] + fb[2] / 2, fb[1] + fb[3] / 2, 11 * s,
              TEXT if session.hover == "FOLDER" else TEXT_DIM, "centre")

        bx, by, bw, bh = layout.body
        # The host has no clipping yet: only what fits the body whole is drawn.
        inside = lambda r: r[1] >= by - 0.5 and r[1] + r[3] <= by + bh + 0.5  # noqa: E731
        for text, rect in layout.captions:
            if inside(rect):
                _text(text.upper(), rect[0] + 2 * s, rect[1] + rect[3] / 2, 10 * s, TEXT_FAINT)
        for card in layout.cards:
            if inside(card.rect):
                _draw_card(card, s, card.entry.name == session.hover)
        if layout.max_scroll > 0:
            # A thin bar at the right edge: where the body is in the whole list.
            whole = bh + layout.max_scroll
            bar_h = max(24 * s, bh * bh / whole)
            bar_y = by + (bh - bar_h) * (1.0 - layout.scroll / layout.max_scroll)
            _round((bx + bw - 7 * s, bar_y, 3 * s, bar_h), 1.5 * s, (1.0, 1.0, 1.0, 0.18))

        fx, fy, fw, fh = layout.footer
        pen.lines(pen.to_3d(np.array([(fx + pad, fy + fh), (fx + fw - pad, fy + fh)], np.float32)),
                            PANEL_EDGE, 1.0 * s)
        hovered = next((c.entry for c in layout.cards if c.entry.name == session.hover), None)
        if hovered is not None:
            name_w = _text(hovered.label, fx + pad, fy + fh / 2, 13 * s, TEXT)
            about = hovered.about or "Your own shader"
            _text(_clip_text(about, fw - 3 * pad - name_w, 12 * s), fx + 2 * pad + name_w, fy + fh / 2, 12 * s, TEXT_DIM)
        else:
            _text("Pick a shader to set it up. It runs on the model, or on the selection. Esc closes.",
                  fx + pad, fy + fh / 2, 11 * s, TEXT_FAINT)


def _draw_handler(context) -> None:
    if _session is not None and context.region is not None:
        draw(context.region, overlay.ui_scale())


registry.add_draw("shader_gallery", 750, "PIXEL", _draw_handler)
