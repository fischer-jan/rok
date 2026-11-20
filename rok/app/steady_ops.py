"""Steady Hand on the app side: the switch, its key (H) and its saved settings.

The maths is `rok.core.steady_hand`; a Voxel stroke reads the two settings
in `tools.py`. The switch and the string length stay between starts.
"""

from __future__ import annotations

from . import settings
from .handlers import registry
from .opkit import Action, register

KEY = "H"


def load(prefs) -> None:
    """Read the saved switch and length into the settings."""
    values = settings.read()
    prefs.rok_steady_hand = values.get("steady_hand") == "on"
    try:
        prefs.rok_steady_length = int(values.get("steady_length", prefs.rok_steady_length))
    except ValueError:
        pass


def save(prefs) -> None:
    settings.put("steady_hand", "on" if prefs.rok_steady_hand else "off")
    settings.put("steady_length", str(int(prefs.rok_steady_length)))


def say(prefs) -> str:
    return f"Steady Hand on, {int(prefs.rok_steady_length)} px" if prefs.rok_steady_hand else "Steady Hand off"


@register
class SteadyHandAction(Action):
    command_id = "rok.steady_hand"
    title = "Steady Hand"
    tooltip = f"Steady Hand ({KEY}): the brush trails the mouse on a string, so a shaking hand draws a smooth line"

    def execute(self, context):
        prefs = context.prefs
        prefs.rok_steady_hand = not prefs.rok_steady_hand
        save(prefs)
        if context.app is not None:
            context.app.say(say(prefs), 2.0)
            context.app.request_redraw()
        return {"FINISHED"}


def _start(context) -> None:
    load(context.prefs)


registry.startup.append(_start)
registry.keymap.append(("rok.steady_hand", {"type": KEY, "value": "PRESS"}, None))
