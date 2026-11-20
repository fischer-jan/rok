"""Typing a grid size into the label at the lower left: click the size, type "40 40 40", press Enter."""

from __future__ import annotations

import re

from .grid_ops import MAX_SIZE, MIN_SIZE

ALLOWED = set("0123456789 x,*X")  # digits and the separators people type
MAX_CHARS = 16


def start_text(size) -> str:
    """What the field shows when it opens: the size, as it would be typed."""
    return " ".join(str(int(v)) for v in size)


def parse_size(text: str) -> tuple[int, int, int] | None:
    """The size in `text`: three whole numbers ("40 30 20", "40x30x20",
    "40, 30, 20"), or one for a cube ("40"). None when it is not a size or a
    number is outside MIN_SIZE..MAX_SIZE."""
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if len(numbers) == 1:
        numbers *= 3
    if len(numbers) != 3 or not all(MIN_SIZE <= n <= MAX_SIZE for n in numbers):
        return None
    return (numbers[0], numbers[1], numbers[2])


def type_char(text: str, fresh: bool, char: str) -> tuple[str, bool]:
    """(text, fresh) after typing `char`. While `fresh` (the whole size is
    selected, as when the field opens) the first character replaces it."""
    if char not in ALLOWED:
        return text, fresh
    base = "" if fresh else text
    if len(base) >= MAX_CHARS:
        return base, False
    return base + char, False


def backspace(text: str, fresh: bool) -> tuple[str, bool]:
    """(text, fresh) after Backspace: a fresh field empties, else one character goes."""
    return ("", False) if fresh else (text[:-1], False)
