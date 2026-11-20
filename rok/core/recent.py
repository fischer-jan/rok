"""The recent-files list: pure list logic. The app reads and writes the file."""

from __future__ import annotations

LIMIT = 10


def remember(entries: list[str], path: str, limit: int = LIMIT) -> list[str]:
    """`path` first, the rest in their order without it, at most `limit` entries."""
    path = str(path)
    out = [path] + [e for e in entries if e != path]
    return out[:limit]


def parse(text: str) -> list[str]:
    """One path per line; blank lines and duplicates are dropped, order kept."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line and line not in out:
            out.append(line)
    return out


def serialise(entries: list[str]) -> str:
    return "".join(f"{e}\n" for e in entries)
