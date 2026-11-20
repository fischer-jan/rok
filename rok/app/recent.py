"""The recent-files list, kept in ~/Library/Application Support/Rok/recent.txt.
The list logic is rok.core.recent."""

from __future__ import annotations

import os
from pathlib import Path

from rok.core import recent as recent_core
from rok.core.safe_write import write_text

APP_FOLDER = Path.home() / "Library" / "Application Support" / "Rok"
FOLDER = Path(os.environ.get("ROK_CONFIG", APP_FOLDER))
FILE = FOLDER / "recent.txt"


def load() -> list[str]:
    try:
        return recent_core.parse(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []  # no list yet, or a damaged one


def remember(path: str) -> None:
    entries = recent_core.remember(load(), str(Path(path).resolve()))
    try:
        FOLDER.mkdir(parents=True, exist_ok=True)
        write_text(FILE, recent_core.serialise(entries))
    except OSError:
        pass
