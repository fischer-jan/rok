"""View > Customize Rail Menu and Keys: which whole-model commands the rail's Model
flyout and the star's Model arms show, and a key for each.

Every command stays in Cmd+K whatever the flyout shows. The choice is saved in
Rok's settings folder (model_menu.json) as
{"shown": [command ids], "known": [command ids], "keys": {command id: "Shift+Ctrl+L"}}.
"known" lists the commands the file was saved with: a command added later that
the flyout shows by default (the Cage) shows up for you too, once.

A key is the modifiers (Shift, Ctrl, Option, Cmd, in this order) and the
event type, joined by "+". A custom key goes first in the keymap, so it wins
over a key Rok already gives to something else; the dialog says which.
"""

from __future__ import annotations

import json
from pathlib import Path

from rok.core.chrome import COMMANDS, DEFAULT_MODEL_ITEMS, MODEL_COMMAND_IDS
from rok.core.safe_write import write_text

from .handlers import registry
from .opkit import BoolProp, Action, StringProp, register

FILE_NAME = "model_menu.json"
MODIFIERS = (("Shift", "shift"), ("Ctrl", "ctrl"), ("Option", "alt"), ("Cmd", "cmd_key"))
# Event types that are only a modifier: the key capture waits past them.
MODIFIER_TYPES = {
    "LEFT_SHIFT", "RIGHT_SHIFT", "LEFT_CTRL", "RIGHT_CTRL", "LEFT_ALT", "RIGHT_ALT", "CMD",
}
_DIGITS = {"DIGIT_0": "0", "DIGIT_1": "1", "DIGIT_2": "2", "DIGIT_3": "3", "DIGIT_4": "4",
           "DIGIT_5": "5", "DIGIT_6": "6", "DIGIT_7": "7", "DIGIT_8": "8", "DIGIT_9": "9"}

_shown: tuple[str, ...] | None = None
_keys: dict[str, str] = {}
_installed: list = []  # the keymap entries this module added


def _file() -> Path:
    from . import recent

    return Path(recent.FOLDER) / FILE_NAME


def names() -> dict[str, str]:
    """Command id -> the name Cmd+K shows, without the dots."""
    out: dict[str, str] = {}
    for c in COMMANDS:
        out.setdefault(c.id, c.name.split(":")[0].rstrip("."))
    return out


def load() -> None:
    global _shown, _keys
    try:
        saved = json.loads(_file().read_text())
    except (OSError, ValueError):
        saved = {}
    shown = saved.get("shown") if isinstance(saved, dict) else None
    known = saved.get("known") if isinstance(saved, dict) else None
    if not isinstance(known, list):
        known = list(MODEL_COMMAND_IDS)
    if isinstance(shown, list):
        new = {c for c in DEFAULT_MODEL_ITEMS if c not in known}  # a new default command shows
        _shown = tuple(c for c in MODEL_COMMAND_IDS if c in shown or c in new)
    else:
        _shown = DEFAULT_MODEL_ITEMS
    keys = saved.get("keys") if isinstance(saved, dict) else None
    keys = {c: k for c, k in (keys or {}).items() if isinstance(k, str)}
    _keys = {c: k for c, k in keys.items() if c in MODEL_COMMAND_IDS and parse(k)}


def items() -> tuple[str, ...]:
    """The command ids the Model flyout shows."""
    if _shown is None:
        load()
    return _shown


def keys() -> dict[str, str]:
    if _shown is None:
        load()
    return dict(_keys)


def save(shown, new_keys: dict[str, str]) -> None:
    global _shown, _keys
    _shown = tuple(c for c in MODEL_COMMAND_IDS if c in set(shown))
    _keys = {c: k for c, k in new_keys.items() if c in MODEL_COMMAND_IDS and k and parse(k)}
    try:
        _file().parent.mkdir(parents=True, exist_ok=True)
        write_text(_file(), json.dumps({"shown": list(_shown), "known": list(MODEL_COMMAND_IDS), "keys": _keys}, indent=1) + "\n")
    except OSError:
        pass  # the choice holds until Rok quits; it is only not remembered


# --- keys ----------------------------------------------------------------------------


def key_of(event) -> str:
    """The key string of a key press."""
    parts = [label for label, attr in MODIFIERS if getattr(event, attr, False)]
    return "+".join(parts + [event.type])


def parse(key: str) -> dict | None:
    """A key string -> a keymap spec ({"type": ..., "shift": ...}), or None."""
    *mods, kind = key.split("+") if key else [""]
    if not kind or kind in MODIFIER_TYPES:
        return None
    attrs = dict(MODIFIERS)
    if any(m not in attrs for m in mods):
        return None
    spec = {"type": kind, "value": "PRESS"}
    for m in mods:
        spec[attrs[m]] = True
    return spec


def label(key: str) -> str:
    """A key string as people read it: "Shift+Ctrl+DIGIT_1" -> "Shift+Ctrl+1"."""
    if not key:
        return ""
    *mods, kind = key.split("+")
    kind = _DIGITS.get(kind, kind if len(kind) == 1 else kind.replace("_", " ").title())
    return "+".join(mods + [kind])


def install(app=None) -> None:
    """Put the custom keys first in the keymap, in place of the ones before."""
    for entry in _installed:
        if entry in registry.keymap:
            registry.keymap.remove(entry)
    _installed.clear()
    for cid, key in keys().items():
        spec = parse(key)
        if spec is not None:
            _installed.append((f"rok.{cid}", spec, None))
    registry.keymap[:0] = _installed
    if app is not None:
        app._keymap = None


def taken(cid: str, key: str) -> list[str]:
    """What else a key runs now, as operator ids, leaving out this module's entries."""
    spec = parse(key)
    if spec is None:
        return []
    from . import keymap

    want = keymap.item((f"rok.{cid}", spec, None))
    out = []
    for entry in registry.keymap:
        if entry in _installed:
            continue
        it = keymap.item(entry)
        if it.type == want.type and not it.any and it.value in ("PRESS", "ANY") and all(
            getattr(it, m) == -1 or bool(getattr(it, m)) == bool(getattr(want, m)) for _l, m in MODIFIERS
        ):
            out.append(it.idname)
    return out


# --- the dialog ----------------------------------------------------------------------


_props = {}
for _cid in MODEL_COMMAND_IDS:
    _props[f"show_{_cid}"] = BoolProp(name=_cid.replace("_", " ").title(), default=True)
    _props[f"key_{_cid}"] = StringProp(name="Key", subtype="HOTKEY")


class _Base(Action):
    command_id = "rok.model_menu"
    title = "Customize Rail Menu"
    tooltip = "Pick which commands the rail's Model menu shows, and give them keys"

    def invoke(self, context, event):
        shown, ks = set(items()), keys()
        for cid in MODEL_COMMAND_IDS:
            setattr(self, f"show_{cid}", cid in shown)
            setattr(self, f"key_{cid}", ks.get(cid, ""))
        return context.prefs.open_dialog(
            self, width=460, title="Model Menu and Keys", confirm_text="Save"
        )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Ticked commands show in the rail and the star. All stay in Cmd+K.")
        layout.label(text="The star leaves out the ones that change the model without asking.")
        called = names()
        clashes = []
        for cid in MODEL_COMMAND_IDS:
            row = layout.row()
            row.prop(self, f"show_{cid}", text=called.get(cid, cid))
            row.prop(self, f"key_{cid}", text="")
            key = getattr(self, f"key_{cid}")
            for other in taken(cid, key):
                clashes.append(f"{label(key)} ran {other}; now it runs {called.get(cid, cid)}")
        layout.label(text="Click a key field and press the keys. Backspace clears it.")
        for text in clashes:
            layout.label(text=text)

    def execute(self, context):
        shown = [cid for cid in MODEL_COMMAND_IDS if getattr(self, f"show_{cid}")]
        save(shown, {cid: getattr(self, f"key_{cid}") for cid in MODEL_COMMAND_IDS})
        app = context.app
        install(app)
        app.set_menu()
        app.request_redraw()
        return {"FINISHED"}


ModelMenuAction = register(type("ModelMenuAction", (_Base,), dict(_props)))


def _start(context) -> None:
    load()
    install(getattr(context, "app", None))


registry.startup.append(_start)
