"""Registries the modules add to when they are imported.

- `draw`: draw handlers. Each gets its own layer; `kind` is "VIEW" (world
  units) or "PIXEL" (region pixels, origin bottom left).
- `startup`: functions run once the window is up;
  one with `fn.late = True` runs after the others.
- `load_post`: functions run after New or Open.
- `keymap`: keymap items (see keymap.py).
- `menus`: the menu bar spec builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DrawHandler:
    name: str
    z: int
    kind: str  # "VIEW" | "PIXEL"
    fn: Callable
    linear: bool = False  # the handler's colours are linear, not sRGB


@dataclass
class Registry:
    draw: list[DrawHandler] = field(default_factory=list)
    startup: list[Callable] = field(default_factory=list)
    load_post: list[Callable] = field(default_factory=list)
    keymap: list = field(default_factory=list)
    menus: list[Callable] = field(default_factory=list)

    def add_draw(self, name: str, z: int, kind: str, fn: Callable, linear: bool = False) -> None:
        self.draw = [h for h in self.draw if h.name != name]
        self.draw.append(DrawHandler(name, z, kind, fn, linear))
        self.draw.sort(key=lambda h: (h.z, h.name))


registry = Registry()
