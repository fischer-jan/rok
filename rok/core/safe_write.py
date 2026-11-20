"""Write a file so the old one stays until the new one is complete.

The bytes go to a `.part` file beside the target, then a rename puts them in
place. A full disk, a share that drops or a crash during the write leaves the
old file as it was, and the half-written part is removed.
"""

from __future__ import annotations

from pathlib import Path


def write_bytes(path: str | Path, data: bytes) -> None:
    """Write `data` to `path` through a `.part` file and a rename."""
    path = Path(path)
    part = path.with_name(path.name + ".part")
    try:
        part.write_bytes(data)
        part.replace(path)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write `text` to `path` the same way."""
    write_bytes(path, text.encode(encoding))
