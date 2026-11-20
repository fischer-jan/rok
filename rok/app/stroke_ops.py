"""Stroke Designer settings: on or off, and the curves, kept between starts.

The panel and the stroke read these. The maths is rok.core.stroke_design.
"""

from __future__ import annotations

from rok.core.stroke_design import default_design, pack, unpack

from . import settings
from .app import RokPrefs
from .handlers import registry
from .opkit import BoolProp, StringProp

RokPrefs.add_property("rok_stroke_design", BoolProp(
    name="Stroke",
    description="Shape thickness, roundness and colour along a stroke",
    default=False,
))
RokPrefs.add_property("rok_stroke_curves", StringProp(
    name="Stroke Curves",
    description="The Stroke Designer's curves, its switches and its second colour",
    default=pack(default_design()),
))


def load(prefs) -> None:
    """Read the saved switch and curves into the settings."""
    values = settings.read()
    prefs.rok_stroke_design = values.get("stroke_design") == "on"
    text = values.get("stroke_curves")
    if text:
        prefs.rok_stroke_curves = pack(unpack(text))


def save(prefs) -> None:
    settings.put("stroke_design", "on" if prefs.rok_stroke_design else "off")
    settings.put("stroke_curves", str(prefs.rok_stroke_curves))


def design_of(prefs):
    """The curves when the designer is on, else None (a stroke stays as it is)."""
    if prefs is None or not getattr(prefs, "rok_stroke_design", False):
        return None
    return unpack(getattr(prefs, "rok_stroke_curves", "") or None)


def stored(prefs):
    """The curves, whether or not the designer is on."""
    if prefs is None:
        return default_design()
    return unpack(getattr(prefs, "rok_stroke_curves", "") or None)


def store(prefs, design, *, write: bool = True) -> None:
    """Keep `design`. `write` saves it for the next start."""
    prefs.rok_stroke_curves = pack(design)
    if write:
        save(prefs)


def _start(context) -> None:
    load(context.prefs)


registry.startup.append(_start)
