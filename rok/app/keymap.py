"""Keymap items, matched against events.

An item is (command id, {"type": ..., "value": ..., "shift": ..., ...}, props);
props is None or
{"properties": [(name, value), ...]}. Modifiers left out of the spec must be
up; "any": True accepts any modifiers. "cmd_key" is Cmd.
Items are tried in order; an action whose poll fails, or that returns
PASS or CANCELLED, lets the next item try.
"""

from __future__ import annotations

from dataclasses import dataclass

from .opkit import Event


@dataclass
class Item:
    idname: str
    type: str
    value: str
    shift: object = False
    ctrl: object = False
    alt: object = False
    cmd_key: object = False
    any: bool = False
    repeat: bool = True
    props: dict | None = None

    def matches(self, ev: Event) -> bool:
        if ev.type != self.type:
            return False
        if self.value not in ("ANY", ev.value):
            return False
        if ev.is_repeat and not self.repeat:
            return False
        if self.any:
            return True
        for mod in ("shift", "ctrl", "alt", "cmd_key"):
            want = getattr(self, mod)
            if want == -1:  # KM_ANY for this modifier
                continue
            if bool(want) != bool(getattr(ev, mod)):
                return False
        return True


def item(entry) -> Item:
    idname, spec, extra = entry
    props = {}
    if extra and extra.get("properties"):
        props = dict(extra["properties"])
    return Item(
        idname=idname,
        type=spec["type"],
        value=spec.get("value", "PRESS"),
        shift=spec.get("shift", False),
        ctrl=spec.get("ctrl", False),
        alt=spec.get("alt", False),
        cmd_key=spec.get("cmd_key", False),
        any=bool(spec.get("any", False)),
        repeat=bool(spec.get("repeat", True)),
        props=props,
    )


def items(entries) -> list[Item]:
    return [item(e) for e in entries]
