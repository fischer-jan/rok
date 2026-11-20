"""The history strip's geometry and hit testing.

The strip is one dark card along the bottom of the view:

    ┌──────────────────────────────────────────────────────────────┐
    │ ●──●──●──●──◆──○──○                          Mirror X  14:32 │
    └──────────────────────────────────────────────────────────────┘

The first dot is the start (the document before the oldest step it still
holds); each further dot is one undo step, oldest left. The dots up to the
current place are done steps; the ones after it are undone steps that Redo
can bring back. The right end names the current step, or the step under the
mouse.

Measurements are in points; `strip_layout` multiplies by the UI scale.
Region pixels have y up: a rect is (x, bottom, width, height). numpy-free.
"""

from __future__ import annotations

from dataclasses import dataclass

Rect = tuple[float, float, float, float]

HEIGHT = 34
PAD = 14  # left and right, inside the card
LABEL_W = 220  # the name at the right end
MAX_SPACING = 26  # between two dots, when there is room
DOT_R = 4.0
LINE_SPACING = 4.0  # under this the dots draw as one line with ticks


@dataclass(frozen=True)
class StripLayout:
    rect: Rect  # the card
    track: tuple[float, float]  # x of the first and the last place
    centre_y: float  # the dots' y
    label: Rect  # the name at the right end
    count: int  # places: the start and one per step
    position: int  # the current place: 0 is the start, k is after step k
    scale: float

    @property
    def spacing(self) -> float:
        if self.count < 2:
            return 0.0
        return (self.track[1] - self.track[0]) / (self.count - 1)

    @property
    def as_line(self) -> bool:
        """Too many places for dots: draw a line with ticks."""
        return self.count > 1 and self.spacing < LINE_SPACING * self.scale

    def x_of(self, place: int) -> float:
        return self.track[0] + self.spacing * place

    def place_at(self, x: float) -> int:
        """The place nearest to `x`, clamped to the strip's ends."""
        if self.count < 2:
            return 0
        k = round((float(x) - self.track[0]) / self.spacing)
        return max(0, min(self.count - 1, int(k)))

    def contains(self, x: float, y: float) -> bool:
        rx, ry, rw, rh = self.rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def on_track(self, x: float, y: float) -> bool:
        """Inside the card and left of the name: a press there scrubs."""
        return self.contains(x, y) and x < self.label[0]


def strip_layout(scale: float, left: float, right: float, bottom: float, count: int, position: int) -> StripLayout:
    """The card between x = `left` and `right`, its bottom at `bottom`, for
    `count` places (the start and one per step) with the document at `position`."""
    s = float(scale)
    count = max(1, int(count))
    position = max(0, min(count - 1, int(position)))
    width = max(right - left, (2 * PAD + LABEL_W + 40) * s)
    rect = (float(left), float(bottom), float(width), HEIGHT * s)
    label_w = LABEL_W * s
    label = (left + width - PAD * s - label_w, float(bottom), label_w, HEIGHT * s)
    first = left + PAD * s + DOT_R * s
    room = label[0] - 16 * s - first
    span = room if count < 2 else min(room, (count - 1) * MAX_SPACING * s)
    track = (first, first + max(0.0, span))
    return StripLayout(rect, track, bottom + HEIGHT * s / 2.0, label, count, position, s)
