"""Palette v2 motion: small time-driven tweens, no host timers.

Everything reads `now()`, which tests (and the headless GIF frames) replace
with `set_clock`, so an animation is a pure function of time: the same
times give the same frames.
"""

from __future__ import annotations

import time

HOVER_TIP_DELAY = 0.40  # s before a swatch's tip shows in the HUD line
CHIP_FADE = 0.08  # the header chip fades to a new current colour
FOLD = 0.12  # a section folding, the card collapsing; the chevron turns along
EDITOR = 0.15  # the editor sliding up out of the header
SWAP = 0.10  # two swatches trading places after a Ctrl drag
FLY_BACK = 0.12  # the drag ghost flying home after a release off any swatch
CONTENT_FADE = 0.08  # a section's swatches fading in and out as it opens and folds
FLY = 0.6  # a colour that becomes used or unused flies to its new section
MAX_FLIGHTS = 24  # more at once is a new file or a palette swap, not an edit

_clock = time.monotonic


def now() -> float:
    return _clock()


def set_clock(fn=None) -> None:
    """Use `fn()` as the time (None: the real clock again)."""
    global _clock
    _clock = fn or time.monotonic


def ease_out(t: float) -> float:
    """Cubic ease-out, 0..1 -> 0..1."""
    t = min(1.0, max(0.0, float(t)))
    return 1.0 - (1.0 - t) ** 3


def lerp(a, b, t: float):
    if isinstance(a, (tuple, list)):
        return tuple(x + (y - x) * t for x, y in zip(a, b))
    return a + (b - a) * t


class Tweens:
    """Named values that move to a new target over a duration.

    `value(name, target, duration)` returns where the value is now; a new
    target starts a tween from the value shown at that moment, so a change in
    the middle of a tween does not jump.
    """

    def __init__(self) -> None:
        self._t: dict[str, tuple] = {}  # name -> (from, to, start, duration, eased)

    def value(self, name: str, target, duration: float, eased: bool = True, at: float | None = None):
        t_now = now() if at is None else at
        cur = self._t.get(name)
        if cur is None:
            self._t[name] = (target, target, t_now, duration, eased)
            return target
        if cur[1] != target:
            shown = self._at(cur, t_now)
            self._t[name] = (shown, target, t_now, duration, eased)
            cur = self._t[name]
        return self._at(cur, t_now)

    @staticmethod
    def _at(cur, t_now: float):
        a, b, start, duration, eased = cur
        if duration <= 0:
            return b
        t = (t_now - start) / duration
        if t >= 1.0:
            return b
        return lerp(a, b, ease_out(t) if eased else max(0.0, t))

    def running(self, at: float | None = None) -> bool:
        t_now = now() if at is None else at
        return any(t_now - c[2] < c[3] and c[0] != c[1] for c in self._t.values())

    def jump(self, name: str, target) -> None:
        """Set a value with no motion."""
        self._t[name] = (target, target, now(), 0.0, True)

    def clear(self) -> None:
        self._t.clear()


class Hover:
    """How long the mouse has rested on one item."""

    def __init__(self) -> None:
        self.item = None
        self.since = 0.0

    def update(self, item, at: float | None = None) -> float:
        t_now = now() if at is None else at
        if item != self.item:
            self.item, self.since = item, t_now
        return t_now - self.since if item is not None else 0.0


class Flights:
    """Colours that change section (In use <-> Unused) fly from the old place to the new.

    `update` is given where every colour is now; a colour that was in the other
    section on the previous call starts a flight from where it was drawn then.
    Where a section is folded (or the swatch scrolled out of its clip) the
    flight starts or ends at that section's anchor, its heading chevron.
    """

    def __init__(self) -> None:
        self._section: dict[int, str] = {}  # key -> section, as of the last update
        self._rect: dict[int, tuple] = {}
        self._anchor: dict[str, tuple] = {}
        self._fly: dict[int, tuple] = {}  # key -> (from rect, start time)

    def update(self, sections: dict[str, tuple[int, ...]], rects: dict[int, tuple],
               anchors: dict[str, tuple], at: float | None = None) -> None:
        """`sections`: {name: keys} for the sections that fly (not Favourites);
        `rects`: {key: rect} of the swatches drawn; `anchors`: {name: rect}."""
        t_now = now() if at is None else at
        member = {k: name for name, keys in sections.items() for k in keys}
        moved = {k: name for k, name in member.items() if k in self._section and self._section[k] != name}
        if 0 < len(moved) <= MAX_FLIGHTS:
            for k in moved:
                start = self._rect.get(k) or self._anchor.get(self._section[k])
                if start is not None:
                    self._fly[k] = (start, t_now)
        self._section, self._rect, self._anchor = member, dict(rects), dict(anchors)

    def rects(self, rects: dict[int, tuple], anchors: dict[str, tuple], at: float | None = None) -> dict[int, tuple]:
        """{key: rect} of the swatches in flight at `at`; finished flights are dropped."""
        from .hud import flying_rect

        t_now = now() if at is None else at
        out = {}
        for k, (start, t0) in list(self._fly.items()):
            t = (t_now - t0) / FLY
            end = rects.get(k) or anchors.get(self._section.get(k, ""))
            if t >= 1.0 or end is None:
                del self._fly[k]
                continue
            out[k] = flying_rect(start, end, t)
        return out

    def running(self) -> bool:
        return bool(self._fly)

    def clear(self) -> None:
        self._section.clear(); self._rect.clear(); self._anchor.clear(); self._fly.clear()  # noqa: E702
