"""Actions, properties, events and context for Rok's editing layer.

Every command of the editing layer is an action. This module gives them:

- `Action` with `poll`, `invoke`, `execute`, `modal`, `report`, and
  properties declared as class attributes (`IntProp(...)` and friends).
- `ops.rok.<name>(...)` runs an action by its command id, with `.poll()`;
  the first positional argument may be "INVOKE" (open its dialog or start
  its drag) or "EXEC" (run it at once).
- `Event`. `mouse_region_x/y` have the origin at the bottom left of the
  view, and so do all region pixels here.
- `context`: `region`, `view_state`, `prefs` (the `rok_*` settings, modal
  handlers, timers, dialogs), `area.tag_redraw()`, `window.cursor_warp()`
  and `window.cursor_set()`.

The app (`rok.app.app`) owns the one context and feeds it events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from rok.core.undo import naming

# --- properties -----------------------------------------------------------------------


class Prop:
    kind = "any"

    def __init__(self, name: str = "", description: str = "", default: Any = None, update: Callable | None = None,
                 options: set | None = None, **extra) -> None:
        # extra may hold get= and set= functions (self, [value]).
        self.name = name
        self.description = description
        self.default = default
        self.update = update
        self.options = options or set()
        self.extra = extra
        self.attr = ""

    def __set_name__(self, owner, attr: str) -> None:
        self.attr = attr

    def coerce(self, value):
        return value


class BoolProp(Prop):
    kind = "bool"

    def __init__(self, default: bool = False, **kw) -> None:
        super().__init__(default=bool(default), **kw)

    def coerce(self, value):
        return bool(value)


class IntProp(Prop):
    kind = "int"

    def __init__(self, default: int = 0, min: int | None = None, max: int | None = None,
                 soft_min: int | None = None, soft_max: int | None = None, **kw) -> None:
        super().__init__(default=int(default), **kw)
        self.min, self.max = min, max
        self.soft_min, self.soft_max = soft_min, soft_max

    def coerce(self, value):
        v = int(value)
        if self.min is not None:
            v = max(self.min, v)
        if self.max is not None:
            v = min(self.max, v)
        return v


class FloatProp(Prop):
    kind = "float"

    def __init__(self, default: float = 0.0, min: float | None = None, max: float | None = None,
                 soft_min: float | None = None, soft_max: float | None = None, step: float = 1.0,
                 precision: int = 2, subtype: str = "NONE", **kw) -> None:
        super().__init__(default=float(default), **kw)
        self.min, self.max = min, max
        self.soft_min, self.soft_max = soft_min, soft_max
        self.precision = precision
        self.subtype = subtype

    def coerce(self, value):
        v = float(value)
        if self.min is not None:
            v = max(self.min, v)
        if self.max is not None:
            v = min(self.max, v)
        return v


class StringProp(Prop):
    kind = "string"

    def __init__(self, default: str = "", subtype: str = "NONE", **kw) -> None:
        super().__init__(default=str(default), **kw)
        self.subtype = subtype

    def coerce(self, value):
        return str(value)


class EnumProp(Prop):
    """items: (id, label, description[, icon[, number]]) tuples, or a function
    (self, context) -> such tuples."""

    kind = "enum"

    def __init__(self, items=(), default=None, **kw) -> None:
        self.items = items
        if default is None and not callable(items) and items:
            default = items[0][0]
        super().__init__(default=default, **kw)

    def item_list(self, owner=None, context=None):
        items = self.items(owner, context) if callable(self.items) else self.items
        return [tuple(i) for i in items]

    def coerce(self, value):
        return str(value)


class IntVectorProp(Prop):
    kind = "ivec"

    def __init__(self, default=(0, 0, 0), size: int = 3, min: int | None = None, max: int | None = None, **kw) -> None:
        super().__init__(default=tuple(int(v) for v in default), **kw)
        self.size = size
        self.min, self.max = min, max

    def coerce(self, value):
        out = []
        for v in value:
            v = int(v)
            if self.min is not None:
                v = max(self.min, v)
            if self.max is not None:
                v = min(self.max, v)
            out.append(v)
        return tuple(out)


class FloatVectorProp(Prop):
    kind = "fvec"

    def __init__(self, default=(0.0, 0.0, 0.0), size: int = 3, min: float | None = None, max: float | None = None,
                 subtype: str = "NONE", **kw) -> None:
        super().__init__(default=tuple(float(v) for v in default), **kw)
        self.size = size
        self.min, self.max = min, max
        self.subtype = subtype

    def coerce(self, value):
        return tuple(float(v) for v in value)


class BoolVectorProp(Prop):
    kind = "bvec"

    def __init__(self, default=(False, False, False), size: int = 3, **kw) -> None:
        super().__init__(default=tuple(bool(v) for v in default), **kw)
        self.size = size

    def coerce(self, value):
        return tuple(bool(v) for v in value)


def props_of(cls) -> dict[str, Prop]:
    """Every property a class declares, its bases' too, in declaration order."""
    out: dict[str, Prop] = {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, Prop):
                out[name] = value
    return out


class PropGroup:
    """An object whose class attributes are properties; reading one gives its value.
    A property with `get` / `set` functions computes its value.
    `add_property` adds one later: the modules add theirs to the window
    manager when they register."""

    _prop_cache: dict = {}

    def __init__(self, **values) -> None:
        object.__setattr__(self, "_values", {})
        for name, value in values.items():
            setattr(self, name, value)

    @classmethod
    def _props(cls) -> dict[str, Prop]:
        cached = PropGroup._prop_cache.get(cls)
        if cached is None:
            cached = props_of(cls)
            PropGroup._prop_cache[cls] = cached
        return cached

    @classmethod
    def add_property(cls, name: str, prop: Prop) -> None:
        prop.attr = name
        setattr(cls, name, prop)
        PropGroup._prop_cache.clear()

    @classmethod
    def remove_property(cls, name: str) -> None:
        if isinstance(cls.__dict__.get(name), Prop):
            delattr(cls, name)
            PropGroup._prop_cache.clear()

    def __getattribute__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        prop = type(self)._props().get(name)
        if prop is None:
            return object.__getattribute__(self, name)
        getter = prop.extra.get("get")
        if getter is not None:
            return prop.coerce(getter(self))
        values = object.__getattribute__(self, "_values")
        return values.get(name, prop.default)

    def __setattr__(self, name, value) -> None:
        prop = type(self)._props().get(name)
        if prop is None:
            object.__setattr__(self, name, value)
            return
        value = prop.coerce(value)
        setter = prop.extra.get("set")
        if setter is not None:
            setter(self, value)
        else:
            values = object.__getattribute__(self, "_values")
            changed = values.get(name, prop.default) != value
            values[name] = value
            if not changed:
                return
        if prop.update is not None:
            prop.update(self, CONTEXT)


class Prefs(PropGroup):
    """context.prefs: the rok_* settings plus helpers for modal actions,
    timers, dialogs and file choosers, which hand their work to the app."""

    clipboard = ""

    @property
    def _app(self):
        return CONTEXT.app

    def add_modal(self, op) -> bool:
        self._app.modal_add(op)
        return True

    def add_timer(self, time_step: float, window=None) -> Timer:
        return self._app.timer_add(float(time_step))

    def remove_timer(self, timer) -> None:
        self._app.timer_remove(timer)

    def open_dialog(self, op, width: int = 300, title: str = "", confirm_text: str = ""):
        self._app.open_dialog(op, width=width, title=title or op.title, confirm=confirm_text or "OK")
        return {"RUNNING"}

    def open_popup(self, op, event):
        return self.open_dialog(op)

    def ask_confirm(self, op, event, title: str = "", message: str = "", confirm_text: str = "", icon: str = ""):
        self._app.confirm(op, title=title or op.title, message=message or op.tooltip, confirm=confirm_text or "OK")
        return {"RUNNING"}

    def choose_file(self, op) -> None:
        self._app.file_select(op)

    def progress_begin(self, lo: float, hi: float) -> None:
        pass

    def progress_update(self, value: float) -> None:
        pass

    def progress_end(self) -> None:
        pass


# --- events ---------------------------------------------------------------------------


@dataclass
class Event:
    type: str
    value: str = "NOTHING"  # "PRESS" | "RELEASE" | "NOTHING"
    shift: bool = False
    ctrl: bool = False
    alt: bool = False
    cmd_key: bool = False
    mouse_region_x: int = 0
    mouse_region_y: int = 0
    mouse_x: int = 0
    mouse_y: int = 0
    mouse_prev_x: int = 0
    mouse_prev_y: int = 0
    unicode: str = ""
    is_repeat: bool = False
    # Trackpad and wheel: logical pixels (scroll) or a factor delta (pinch).
    dx: float = 0.0
    dy: float = 0.0
    precise: bool = False
    # Float region position, for code that wants it exact.
    fx: float = 0.0
    fy: float = 0.0


# Named keys from the host (winit names) to Rok's event types.
NAMED_KEYS = {
    "Space": "SPACE", "Tab": "TAB", "Enter": "ENTER", "Escape": "ESCAPE", "Backspace": "BACKSPACE",
    "Delete": "DELETE", "ArrowUp": "UP_ARROW", "ArrowDown": "DOWN_ARROW", "ArrowLeft": "LEFT_ARROW",
    "ArrowRight": "RIGHT_ARROW", "PageUp": "PAGE_UP", "PageDown": "PAGE_DOWN", "Home": "HOME",
    "End": "END", "Shift": "LEFT_SHIFT", "Control": "LEFT_CTRL", "Alt": "LEFT_ALT", "Super": "CMD",
    "F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4", "F5": "F5", "F6": "F6",
}
CHAR_KEYS = {
    "0": "DIGIT_0", "1": "DIGIT_1", "2": "DIGIT_2", "3": "DIGIT_3", "4": "DIGIT_4", "5": "DIGIT_5", "6": "DIGIT_6",
    "7": "DIGIT_7", "8": "DIGIT_8", "9": "DIGIT_9", ",": "COMMA", ".": "PERIOD", "-": "MINUS", "+": "PLUS",
    "=": "EQUAL", "/": "SLASH", "[": "LEFT_BRACKET", "]": "RIGHT_BRACKET", ";": "SEMICOLON",
    "'": "QUOTE", "`": "BACKQUOTE", "\\": "BACKSLASH", " ": "SPACE",
}
MOUSE_BUTTONS = {"left": "LEFT_MOUSE", "right": "RIGHT_MOUSE", "middle": "MIDDLE_MOUSE"}


def event_type_of_key(key: str) -> str:
    if key in NAMED_KEYS:
        return NAMED_KEYS[key]
    if len(key) == 1:
        if key.isalpha():
            return key.upper()
        return CHAR_KEYS.get(key, key)
    return key.upper()


# --- context --------------------------------------------------------------------------


class Area:
    def __init__(self, app) -> None:
        self._app = app
        self.type = "VIEW"

    def tag_redraw(self) -> None:
        self._app.request_redraw()

    def header_text_set(self, text=None) -> None:
        self._app.set_status(text)


class Window:
    def __init__(self, app) -> None:
        self._app = app
        self.cursor = "DEFAULT"

    def cursor_warp(self, x: int, y: int) -> None:
        self._app.warp_cursor(x, y)

    def relative_set(self, on: bool) -> bool:
        """Hide the pointer and hold it still; raw moves come as MOUSE_DELTA
        events. False when the host cannot (then MOUSE_MOVE is all there is)."""
        host = self._app.host
        if not hasattr(host, "set_relative"):
            return False
        host.set_relative(bool(on))
        return True

    CURSORS = {
        "DEFAULT": "default", "CROSSHAIR": "crosshair", "HAND": "grab", "SCROLL_XY": "move",
        "MOVE_X": "ew-resize", "MOVE_Y": "ns-resize", "TEXT": "text", "WAIT": "wait",
        "NONE": "none", "EYEDROPPER": "crosshair", "PAINT_BRUSH": "crosshair", "KNIFE": "crosshair",
    }

    def cursor_set(self, name: str) -> None:
        self.cursor = name
        self._app.host.set_cursor(self.CURSORS.get(name, "default"))

    def cursor_modal_set(self, name: str) -> None:
        self.cursor_set(name)

    def cursor_modal_restore(self) -> None:
        self.cursor_set("DEFAULT")


@dataclass
class Timer:
    interval: float
    next_at: float
    owner: Any = None
    time_duration: float = 0.0
    started: float = field(default_factory=time.monotonic)
    callback: Any = None  # called when due, in place of a TIMER event


class Context:
    """The one context. `app` fills region, view_state and prefs."""

    def __init__(self) -> None:
        self.app = None
        self.region = None
        self.view_state = None
        self.prefs = None
        self.area = None
        self.window = None
        self.scene = None


CONTEXT = Context()


# --- operators ------------------------------------------------------------------------


class Action(PropGroup):
    command_id = ""
    title = ""
    tooltip = ""
    def __init__(self, **values) -> None:
        PropGroup.__init__(self)
        for name, prop in type(self)._props().items():
            if name in values and values[name] is not None:
                # Set quietly: update callbacks run for edits, not for the call's own values.
                object.__getattribute__(self, "_values")[name] = prop.coerce(values[name])
        object.__setattr__(self, "_reports", [])
        object.__setattr__(self, "layout", None)

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        return {"FINISHED"}

    def modal(self, context, event):
        return {"FINISHED"}

    def cancel(self, context) -> None:
        pass

    def report(self, level, message: str) -> None:
        object.__getattribute__(self, "_reports").append((set(level), str(message)))
        app = CONTEXT.app
        if app is not None:
            app.say(str(message))

    @property
    def properties(self):
        return self


class OpenFile:
    """An action that opens a file: invoke shows the file browser first."""

    filepath = StringProp(subtype="FILE_PATH", options={"HIDDEN"})
    filter_glob = StringProp(default="*", options={"HIDDEN"})

    def invoke(self, context, event):
        context.prefs.choose_file(self)
        return {"RUNNING"}


class SaveFile:
    """An action that saves a file: invoke shows a Save dialog with `filename_ext`."""

    filename_ext = ""
    filepath = StringProp(subtype="FILE_PATH", options={"HIDDEN"})
    filter_glob = StringProp(default="*", options={"HIDDEN"})
    check_existing = BoolProp(default=True, options={"HIDDEN"})

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "untitled" + self.filename_ext
        context.prefs.choose_file(self)
        return {"RUNNING"}


REGISTRY: dict[str, type[Action]] = {}


def register(cls: type[Action]) -> type[Action]:
    """Class decorator: make `ops.rok.<name>` run this action."""
    if not cls.command_id:
        raise ValueError(f"{cls.__name__} has no command_id")
    REGISTRY[cls.command_id] = cls
    return cls


class _OpCall:
    def __init__(self, idname: str) -> None:
        self.idname = idname

    @property
    def cls(self) -> type[Action] | None:
        return REGISTRY.get(self.idname)

    def poll(self) -> bool:
        cls = self.cls
        return cls is not None and bool(cls.poll(CONTEXT))

    def command_class(self):
        return self.cls

    def __call__(self, *args, **props):
        cls = self.cls
        if cls is None:
            raise RuntimeError(f"no operator {self.idname}")
        how = args[0] if args else "EXEC"
        for fn in before:
            fn(self.idname)
        if not cls.poll(CONTEXT):
            return {"CANCELLED"}
        op = cls(**props)
        object.__setattr__(op, "_call_props", dict(props))
        with naming(undo_label(op)):
            if how == "INVOKE":
                app = CONTEXT.app
                event = app.last_event if app is not None else Event("NONE")
                return run(op.invoke(CONTEXT, event), op)
            return run(op.execute(CONTEXT), op)


def undo_label(op) -> str:
    """The name of the undo steps `op` makes: its `undo_label` when it has
    one (a tool's stroke), else its title."""
    return str(getattr(op, "undo_label", "") or op.title or "")


# Functions told the idname before an operator runs (rok/app/mirror_preview.py).
before: list = []

# Functions told (idname, props) when an operator finishes; props are the
# ones it was called with (rok/app/last_command.py).
finished: list = []


def run(result, op) -> set:
    result = set(result) if result is not None else {"FINISHED"}
    if "FINISHED" in result:
        props = op.__dict__.get("_call_props", {})
        for fn in finished:
            fn(op.command_id, props)
    return result


class _Namespace:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __getattr__(self, name: str) -> _OpCall:
        return _OpCall(f"{self.prefix}.{name}")


class _Ops:
    def __getattr__(self, prefix: str) -> _Namespace:
        return _Namespace(prefix)


ops = _Ops()


def call(idname: str, *args, **props):
    """Run an operator by its full id, as a keymap item or a menu does."""
    prefix, name = idname.split(".", 1)
    return getattr(getattr(ops, prefix), name)(*args, **props)
