"""Action dialogs (`prefs.open_dialog`), drawn in Rok's colours.

An action's `draw(self, context)` fills `self.layout` with layout calls
(`label`, `prop`, `prop_enum`, `separator`, `row`, `column`, `grid_flow`);
without a `draw`, every property that is not HIDDEN gets a row. The dialog
lays the rows out top to bottom and handles the mouse and the keyboard:

- enum: one button per item (a click picks it);
- bool: a check box, or a chip with `toggle=True`;
- int and float with `slider=True`: a slider (a click or a drag sets the value);
- int, float, vectors, string: fields; a click types into one (printable
  ASCII), Tab moves to the next, Enter takes the value, Esc drops the edit.
  A horizontal drag on a number field changes it.
- a string with subtype HOTKEY: a key field. A click waits for a key press
  and takes it with its modifiers; Backspace clears it, Esc keeps the old one.
- Enter confirms the dialog when no field is being typed into; Esc cancels.

Positions here are host pixels (origin top left).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import draw, opkit
from .opkit import CONTEXT, Event

PANEL = (0.13, 0.12, 0.115, 0.97)
EDGE = (0.30, 0.28, 0.26, 1.0)
FIELD = (0.08, 0.075, 0.07, 1.0)
FIELD_HOVER = (0.18, 0.17, 0.16, 1.0)
BUTTON = (0.22, 0.21, 0.20, 1.0)
PICKED = (0.91, 0.36, 0.23, 1.0)
SCRIM = (0.0, 0.0, 0.0, 0.35)  # over the view behind the dialog
TEXT = draw.TEXT
DIM = draw.TEXT_DIM
ROW_H = 26.0
GAP = 6.0
PAD = 16.0
FONT = 13.0
CARD = (0.165, 0.155, 0.15, 1.0)  # a card: a group of rows
CARD_PAD = 12.0
HEAD_H = 26.0  # a card's title row
KNOB_R = 8.0
VALUE_W = 64.0  # the number at the right of a slider


# --- the layout ----------------------------------------------------------------------


@dataclass
class Node:
    kind: str  # "label" | "prop" | "prop_enum" | "separator" | "row" | "column" | "grid"
    data: object = None
    name: str = ""
    text: str | None = None
    value: str | None = None
    expand: bool = False
    columns: int = 0
    children: list = field(default_factory=list)
    colour: tuple | None = None  # a label or a property's name in this colour (an axis colour)
    style: str = ""  # label: "small"; prop: "toggle" | "slider"
    weights: tuple = ()  # row: the relative widths of the children
    suffix: str = ""  # slider: the unit after the number


def _rgba(colour) -> tuple:
    c = tuple(float(v) for v in colour)
    return c if len(c) == 4 else c + (1.0,)


class Layout:
    def __init__(self, node: Node | None = None) -> None:
        self.node = node or Node("column")
        self.enabled = True
        self.active = True
        self.alert = False
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.alignment = "EXPAND"

    def _add(self, node: Node) -> Node:
        self.node.children.append(node)
        return node

    def label(self, text: str = "", icon: str = "", colour=None, small: bool = False, **_kw) -> None:
        self._add(Node("label", text=text, colour=colour, style="small" if small else ""))

    def prop(self, data, property: str, text: str | None = None, expand: bool = False, colour=None,
             toggle: bool = False, slider: bool = False, suffix: str = "", **_kw) -> None:
        style = "slider" if slider else ("toggle" if toggle else "")
        self._add(Node("prop", data=data, name=property, text=text, expand=expand, colour=colour,
                       style=style, suffix=suffix))

    def prop_enum(self, data, property: str, value: str, text: str | None = None, **_kw) -> None:
        self._add(Node("prop_enum", data=data, name=property, value=value, text=text))

    def separator(self, factor: float = 1.0, **_kw) -> None:
        self._add(Node("separator"))

    def row(self, align: bool = False, heading: str = "", weights: tuple = (), **_kw) -> Layout:
        return Layout(self._add(Node("row", weights=tuple(weights))))

    def card(self, title: str = "") -> Layout:
        """A group of rows on a lighter plate, with a title."""
        return Layout(self._add(Node("card", text=title)))

    def column(self, align: bool = False, heading: str = "", **_kw) -> Layout:
        return Layout(self._add(Node("column")))

    def box(self) -> Layout:
        return self.column()

    def split(self, factor: float = 0.5, align: bool = False) -> Layout:
        return self.row()

    def grid_flow(self, row_major: bool = True, columns: int = 0, even_columns: bool = False,
                  even_rows: bool = False, align: bool = False) -> Layout:
        return Layout(self._add(Node("grid", columns=max(1, int(columns or 1)))))

    def operator(self, idname: str, text: str = "", **_kw):
        return None  # buttons that run operators are not part of Rok's dialogs


# --- the dialog --------------------------------------------------------------------------


@dataclass
class Hot:
    """One clickable thing of the laid-out dialog."""

    kind: str  # "enum" | "bool" | "field" | "ok" | "cancel"
    rect: tuple[float, float, float, float]
    data: object = None
    name: str = ""
    value: object = None
    index: int = -1  # vector component, -1 for a scalar


def _prop(data, name: str):
    return type(data)._props().get(name) if isinstance(data, opkit.PropGroup) else None


def _label_of(data, name: str) -> str:
    p = _prop(data, name)
    return (p.name if p is not None and p.name else name.replace("_", " ").title())


def _fmt(prop, value) -> str:
    if isinstance(prop, opkit.FloatProp) or isinstance(value, float):
        return f"{float(value):.{getattr(prop, 'precision', 2)}f}"
    return str(value)


class Dialog:
    def __init__(self, app, op, width: int = 300, title: str = "", confirm: str = "OK") -> None:
        self.app = app
        self.op = op
        self.width = max(260.0, float(width))
        self.title = title or op.title
        self.confirm = confirm
        self.hots: list[Hot] = []
        self.hover: Hot | None = None
        self.editing: Hot | None = None
        self.text = ""
        self.capturing: Hot | None = None  # a HOTKEY field waiting for a key
        self.drag: tuple[Hot, float, float] | None = None
        self.slide: Hot | None = None  # the slider being dragged
        self.panel = (0.0, 0.0, 0.0, 0.0)

    # --- layout ----------------------------------------------------------------

    def _layout(self) -> Node:
        layout = Layout()
        object.__setattr__(self.op, "layout", layout)
        draw_fn = getattr(type(self.op), "draw", None)
        if draw_fn is not None:
            draw_fn(self.op, CONTEXT)
        else:
            for name, prop in type(self.op)._props().items():
                if "HIDDEN" not in prop.options:
                    layout.prop(self.op, name)
        return layout.node

    def draw(self) -> None:
        d = draw.DrawList()
        self.hots = []
        root = self._layout()
        items: list = []
        inner_w = self.width - 2 * PAD
        height = self._measure(root, inner_w)
        w = self.width
        h = PAD + 22 + height + GAP * 2 + ROW_H + PAD
        x = (self.app.width - w) / 2
        y = max(20.0, (self.app.height - h) / 2)
        self.panel = (x, y, w, h)
        d.rect(0, 0, self.app.width, self.app.height, SCRIM)
        d.rect(x, y, w, h, PANEL, radius=10)
        d.rect(x, y, w, h, EDGE, radius=10, stroke=1.0)
        d.text(x + PAD, y + PAD - 2, self.title, 15, TEXT)
        self._place(root, x + PAD, y + PAD + 26, inner_w, d, items)
        # Buttons, bottom right.
        bw = 96.0
        by = y + h - PAD - ROW_H
        ok = Hot("ok", (x + w - PAD - bw, by, bw, ROW_H))
        cancel = Hot("cancel", (x + w - PAD - 2 * bw - GAP, by, bw, ROW_H))
        for hot, label, colour in ((cancel, "Cancel", BUTTON), (ok, self.confirm, PICKED)):
            hx, hy, hw, hh = hot.rect
            lit = self.hover is not None and self.hover.kind == hot.kind
            d.rect(hx, hy, hw, hh, colour if not lit else tuple(min(1.0, c * 1.15) for c in colour[:3]) + (1.0,), radius=6)
            tw, _ = self.app.measure_text(label, FONT)
            d.text(hx + (hw - tw) / 2, hy + 5, label, FONT, TEXT)
            self.hots.append(hot)
        cur = gfx_current()
        _merge(cur, d)

    def _measure(self, node: Node, width: float) -> float:
        if node.kind == "row":
            widths = self._widths(node, width)
            return max([ROW_H] + [self._measure(c, w) for c, w in zip(node.children, widths)])
        if node.kind == "card":
            return self._card_height(node, width)
        if node.kind == "separator":
            return GAP * 2
        if node.kind in ("label",):
            return 20.0 if node.style == "small" else ROW_H
        if node.kind in ("prop", "prop_enum"):
            return self._prop_height(node, width)
        if node.kind == "grid":
            n = len(node.children)
            rows = (n + node.columns - 1) // node.columns
            return rows * (ROW_H + GAP)
        return max(0.0, sum(self._measure(c, width) + GAP for c in node.children) - GAP)

    @staticmethod
    def _widths(node: Node, width: float) -> list[float]:
        """The widths of a row's children: equal, or in the proportion of its weights."""
        n = max(1, len(node.children))
        free = width - GAP * (n - 1)
        weights = list(node.weights) if len(node.weights) == len(node.children) else [1.0] * n
        total = sum(weights) or 1.0
        return [free * w / total for w in weights]

    def _card_height(self, node: Node, width: float) -> float:
        inner = width - 2 * CARD_PAD
        body = max(0.0, sum(self._measure(c, inner) + GAP for c in node.children) - GAP)
        return 2 * CARD_PAD + (HEAD_H if node.text else 0.0) + body

    def _prop_height(self, node: Node, width: float) -> float:
        prop = _prop(node.data, node.name)
        if isinstance(prop, opkit.EnumProp) and node.kind == "prop":
            items = prop.item_list(node.data, CONTEXT)
            label = node.text if node.text is not None else _label_of(node.data, node.name)
            per_row = self._enum_per_row(items, width - (110.0 if label else 0.0))
            rows = (len(items) + per_row - 1) // per_row
            return rows * (ROW_H + 4)
        return ROW_H

    def _enum_per_row(self, items, width: float) -> int:
        """How many enum buttons share a row: short labels (X, Y, Z) fit side by side."""
        widest = max((self.app.measure_text(str(i[1]), 12)[0] for i in items), default=0.0) + 16.0
        cell = min(90.0, max(36.0, widest))
        return max(1, min(len(items), int((width + 4) // (cell + 4))))

    def _place(self, node: Node, x: float, y: float, width: float, d: draw.DrawList, items: list) -> float:
        if node.kind == "column":
            top = y
            for c in node.children:
                y += self._place(c, x, y, width, d, items) + GAP
            return max(0.0, y - top - GAP)
        if node.kind == "row":
            h = 0.0
            for c, cw in zip(node.children, self._widths(node, width)):
                h = max(h, self._place(c, x, y, cw, d, items))
                x += cw + GAP
            return h
        if node.kind == "card":
            height = self._card_height(node, width)
            d.rect(x, y, width, height, CARD, radius=8)
            top = y + CARD_PAD
            if node.text:
                d.text(x + CARD_PAD, top + 2, node.text, 14, TEXT)
                top += HEAD_H
            for c in node.children:
                top += self._place(c, x + CARD_PAD, top, width - 2 * CARD_PAD, d, items) + GAP
            return height
        if node.kind == "grid":
            cols = node.columns
            cw = (width - GAP * (cols - 1)) / cols
            for i, c in enumerate(node.children):
                r, k = divmod(i, cols)
                self._place(c, x + k * (cw + GAP), y + r * (ROW_H + GAP), cw, d, items)
            return self._measure(node, width)
        if node.kind == "separator":
            return GAP * 2
        if node.kind == "label":
            small = node.style == "small"
            d.text(x, y + (2 if small else 5), node.text or "", 11.5 if small else FONT,
                   _rgba(node.colour) if node.colour else DIM)
            return 20.0 if small else ROW_H
        if node.kind == "prop_enum":
            self._enum_button(node.data, node.name, node.value, node.text, (x, y, width, ROW_H), d)
            return ROW_H
        return self._place_prop(node, x, y, width, d)

    def _enum_button(self, data, name, value, text, rect, d) -> None:
        prop = _prop(data, name)
        items = {i[0]: i[1] for i in prop.item_list(data, CONTEXT)} if prop else {}
        picked = getattr(data, name) == value
        hot = Hot("enum", rect, data, name, value)
        lit = self.hover is not None and self.hover.kind == "enum" and self.hover.data is data and self.hover.name == name and self.hover.value == value
        d.rect(*rect, PICKED if picked else (FIELD_HOVER if lit else BUTTON), radius=5)
        label = text if text is not None else items.get(value, str(value))
        tw, _ = self.app.measure_text(label, 12)
        d.text(rect[0] + max(4.0, (rect[2] - tw) / 2), rect[1] + 6, label, 12, TEXT)
        self.hots.append(hot)

    def _place_prop(self, node: Node, x: float, y: float, width: float, d) -> float:
        data, name = node.data, node.name
        prop = _prop(data, name)
        label = node.text if node.text is not None else _label_of(data, name)
        value = getattr(data, name)
        label_w = 110.0 if label else 0.0
        if label and not isinstance(prop, opkit.BoolProp):
            d.text(x, y + 5, label, FONT, _rgba(node.colour) if node.colour else DIM)
        fx, fw = x + label_w, width - label_w
        if isinstance(prop, opkit.EnumProp):
            items = prop.item_list(data, CONTEXT)
            per_row = self._enum_per_row(items, fw)
            bw = (fw - 4 * (per_row - 1)) / per_row
            for i, item in enumerate(items):
                r, k = divmod(i, per_row)
                self._enum_button(data, name, item[0], None, (fx + k * (bw + 4), y + r * (ROW_H + 4), bw, ROW_H), d)
            rows = (len(items) + per_row - 1) // per_row
            return rows * (ROW_H + 4)
        if isinstance(prop, opkit.BoolProp) and node.style == "toggle":
            lit = self._same(self.hover, Hot("bool", (x, y, width, ROW_H), data, name))
            d.rect(x, y, width, ROW_H, PICKED if value else (FIELD_HOVER if lit else BUTTON), radius=6)
            tw, _ = self.app.measure_text(label, 12)
            d.text(x + max(4.0, (width - tw) / 2), y + 6, label, 12, TEXT)
            self.hots.append(Hot("bool", (x, y, width, ROW_H), data, name))
            return ROW_H
        if isinstance(prop, (opkit.IntProp, opkit.FloatProp)) and node.style == "slider":
            self._slider(data, name, prop, value, (fx, y, fw, ROW_H), node.suffix, d)
            return ROW_H
        if isinstance(prop, opkit.BoolProp):
            box = (x, y + 4, 18, 18)
            d.rect(*box, PICKED if value else FIELD, radius=4)
            d.rect(*box, EDGE, radius=4, stroke=1.0)
            if value:
                d.line2(x + 4, y + 13, x + 8, y + 18, TEXT, 2.0)
                d.line2(x + 8, y + 18, x + 15, y + 8, TEXT, 2.0)
            d.text(x + 26, y + 5, label, FONT, TEXT)
            self.hots.append(Hot("bool", (x, y, width, ROW_H), data, name))
            return ROW_H
        if isinstance(prop, (opkit.IntVectorProp, opkit.FloatVectorProp)):
            n = len(value)
            cw = (fw - 4 * (n - 1)) / n
            for i in range(n):
                self._field(data, name, prop, value[i], (fx + i * (cw + 4), y, cw, ROW_H), i, d)
            return ROW_H
        self._field(data, name, prop, value, (fx, y, fw, ROW_H), -1, d)
        return ROW_H

    @staticmethod
    def _slide_fraction(prop, value) -> float:
        lo, hi = float(prop.min or 0), float(prop.max if prop.max is not None else 100)
        return 0.0 if hi <= lo else max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))

    def _slider(self, data, name, prop, value, rect, suffix, d) -> None:
        """A track with a knob, and the number with its unit at the right."""
        x, y, w, h = rect
        hot = Hot("slider", rect[:2] + (w - VALUE_W, h), data, name)
        lit = self._same(self.hover, hot) or self._same(self.slide, hot)
        left, right = x + KNOB_R, x + w - VALUE_W - KNOB_R
        cy = y + h / 2
        knob = left + (right - left) * self._slide_fraction(prop, value)
        d.rect(left - 2, cy - 2.5, right - left + 4, 5, FIELD, radius=2.5)
        d.rect(left - 2, cy - 2.5, knob - left + 2, 5, PICKED, radius=2.5)
        r = KNOB_R + (1.0 if lit else 0.0)
        d.rect(knob - r, cy - r, 2 * r, 2 * r, TEXT, radius=r)
        text = _fmt(prop, value) + suffix
        tw, _ = self.app.measure_text(text, FONT)
        d.text(x + w - tw, y + 5, text, FONT, TEXT)
        self.hots.append(hot)

    def _slide_to(self, hot: Hot, fx: float) -> None:
        prop = _prop(hot.data, hot.name)
        x, _y, w, _h = hot.rect
        left, right = x + KNOB_R, x + w - KNOB_R
        frac = 0.0 if right <= left else max(0.0, min(1.0, (fx - left) / (right - left)))
        lo, hi = float(prop.min or 0), float(prop.max if prop.max is not None else 100)
        self._set(hot, lo + frac * (hi - lo))

    def _field(self, data, name, prop, value, rect, index, d) -> None:
        if getattr(prop, "subtype", "") == "HOTKEY":
            self._key_field(data, name, value, rect, d)
            return
        hot = Hot("field", rect, data, name, None, index)
        editing = self._same(self.editing, hot)
        lit = self._same(self.hover, hot)
        d.rect(*rect, FIELD_HOVER if (lit or editing) else FIELD, radius=5)
        if editing:
            d.rect(*rect, PICKED, radius=5, stroke=1.5)
        text = self.text if editing else _fmt(prop, value)
        tw, _ = self.app.measure_text(text or " ", FONT)
        tx = rect[0] + (rect[2] - tw) / 2 if not isinstance(prop, opkit.StringProp) else rect[0] + 8
        d.text(tx, rect[1] + 5, text, FONT, TEXT)
        if editing:
            cx = tx + tw + 1
            d.line2(cx, rect[1] + 5, cx, rect[1] + rect[3] - 5, TEXT, 1.0)
        self.hots.append(hot)

    def _key_field(self, data, name, value, rect, d) -> None:
        from . import model_menu

        hot = Hot("key", rect, data, name)
        waiting = self._same(self.capturing, hot)
        d.rect(*rect, FIELD_HOVER if (waiting or self._same(self.hover, hot)) else FIELD, radius=5)
        if waiting:
            d.rect(*rect, PICKED, radius=5, stroke=1.5)
        text, colour = ("Press a key…", DIM) if waiting else ((model_menu.label(value), TEXT) if value else ("—", DIM))
        tw, _ = self.app.measure_text(text, FONT)
        d.text(rect[0] + (rect[2] - tw) / 2, rect[1] + 5, text, FONT, colour)
        self.hots.append(hot)

    def _capture(self, ev: Event) -> None:
        """A key press while a key field waits."""
        from . import model_menu

        hot = self.capturing
        t = ev.type
        if t in model_menu.MODIFIER_TYPES or "MOUSE" in t or t.startswith(("WHEEL", "TRACKPAD", "TIMER", "WINDOW", "NDOF")):
            return
        if t == "ESCAPE":
            pass
        elif t in ("BACKSPACE", "DELETE"):
            setattr(hot.data, hot.name, "")
        else:
            setattr(hot.data, hot.name, model_menu.key_of(ev))
        self.capturing = None

    @staticmethod
    def _same(a: Hot | None, b: Hot) -> bool:
        return a is not None and a.kind == b.kind and a.data is b.data and a.name == b.name and a.index == b.index and a.value == b.value

    # --- events ----------------------------------------------------------------

    def _hot_at(self, ev: Event) -> Hot | None:
        x, y = ev.fx, self.app.height - ev.fy
        for hot in reversed(self.hots):
            hx, hy, hw, hh = hot.rect
            if hx <= x <= hx + hw and hy <= y <= hy + hh:
                return hot
        return None

    def handle(self, ev: Event) -> bool:
        """Every event goes to the dialog while it is open."""
        app = self.app
        if ev.type == "MOUSE_MOVE":
            if self.slide is not None:
                self._slide_to(self.slide, ev.fx)
            if self.drag is not None:
                hot, x0, v0 = self.drag
                prop = _prop(hot.data, hot.name)
                step = 0.1 if isinstance(prop, opkit.FloatProp) else 1.0
                delta = round((ev.fx - x0) / 6.0) * step
                self._set(hot, v0 + delta)
            self.hover = self._hot_at(ev)
            app.request_redraw()
            return True
        if ev.type == "LEFT_MOUSE" and ev.value == "PRESS":
            hot = self._hot_at(ev)
            self.capturing = None
            if hot is None:
                return True
            if hot.kind == "key":
                self._commit_edit()
                self.capturing = hot
            elif hot.kind == "ok":
                self._commit_edit()
                self._finish(True)
            elif hot.kind == "cancel":
                self._finish(False)
            elif hot.kind == "enum":
                self._commit_edit()
                setattr(hot.data, hot.name, hot.value)
            elif hot.kind == "slider":
                self._commit_edit()
                self.slide = hot
                self._slide_to(hot, ev.fx)
            elif hot.kind == "bool":
                self._commit_edit()
                setattr(hot.data, hot.name, not getattr(hot.data, hot.name))
            elif hot.kind == "field":
                if not self._same(self.editing, hot):
                    self._commit_edit()
                    prop = _prop(hot.data, hot.name)
                    value = self._value(hot)
                    if isinstance(prop, (opkit.IntProp, opkit.FloatProp, opkit.IntVectorProp, opkit.FloatVectorProp)):
                        self.drag = (hot, ev.fx, float(value))
                    self.editing = hot
                    self.text = _fmt(prop, value)
            app.request_redraw()
            return True
        if ev.type == "LEFT_MOUSE" and ev.value == "RELEASE":
            self.slide = None
            if self.drag is not None:
                hot, x0, _v0 = self.drag
                self.drag = None
                if abs(ev.fx - x0) > 3:  # a drag, not a click: no typing
                    self.editing = None
            app.request_redraw()
            return True
        if ev.value != "PRESS":
            return True
        if self.capturing is not None:
            self._capture(ev)
            app.request_redraw()
            return True
        if self.editing is not None:
            if ev.type in ("ENTER", "NUMPAD_ENTER"):
                self._commit_edit()
            elif ev.type == "ESCAPE":
                self.editing = None
            elif ev.type == "TAB":
                self._commit_edit(next_field=True)
            elif ev.type == "BACKSPACE":
                self.text = self.text[:-1]
            elif ev.unicode and ev.unicode.isprintable() and ev.unicode.isascii() and not ev.cmd_key:
                self.text += ev.unicode
            app.request_redraw()
            return True
        if ev.type in ("ENTER", "NUMPAD_ENTER"):
            self._finish(True)
        elif ev.type == "ESCAPE":
            self._finish(False)
        return True

    def _value(self, hot: Hot):
        v = getattr(hot.data, hot.name)
        return v[hot.index] if hot.index >= 0 else v

    def _set(self, hot: Hot, value) -> None:
        prop = _prop(hot.data, hot.name)
        if hot.index >= 0:
            v = list(getattr(hot.data, hot.name))
            v[hot.index] = int(round(value)) if isinstance(prop, opkit.IntVectorProp) else float(value)
            setattr(hot.data, hot.name, tuple(v))
        elif isinstance(prop, opkit.IntProp):
            setattr(hot.data, hot.name, int(round(float(value))))
        else:
            setattr(hot.data, hot.name, value)

    def _commit_edit(self, next_field: bool = False) -> None:
        hot = self.editing
        if hot is None:
            return
        prop = _prop(hot.data, hot.name)
        try:
            if isinstance(prop, opkit.StringProp):
                self._set(hot, self.text)
            else:
                self._set(hot, float(self.text))
        except ValueError:
            pass
        self.editing = None
        if next_field:
            fields = [h for h in self.hots if h.kind == "field"]
            for i, h in enumerate(fields):
                if self._same(h, hot) and i + 1 < len(fields):
                    self.editing = fields[i + 1]
                    self.text = _fmt(_prop(self.editing.data, self.editing.name), self._value(self.editing))
                    break

    def _finish(self, ok: bool) -> None:
        app = self.app
        app.dialog = None
        app.request_redraw()
        if ok:
            result = opkit.run(self.op.execute(CONTEXT) or set(), self.op)
            app._after_op(self.op, result)
        else:
            self.op.cancel(CONTEXT)


def gfx_current() -> draw.DrawList:
    from . import gfx

    return gfx.current()


def _merge(into: draw.DrawList, d: draw.DrawList) -> None:
    for k, v in d.parts.items():
        into.parts[k].extend(v)
    into.icons.extend(d.icons)
    into.texts.extend(d.texts)
