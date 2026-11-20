"""Small preferences that stay between starts: key=value lines in
settings.txt in Rok's config folder. A write keeps the other keys."""

from __future__ import annotations

from rok.core.safe_write import write_text

FILE_NAME = "settings.txt"


def _path():
    from . import recent

    return recent.FOLDER / FILE_NAME


def read() -> dict[str, str]:
    try:
        text = _path().read_text(encoding="utf-8")
    except (OSError, ValueError):
        return {}  # no file yet, or a damaged one: start from the defaults
    out = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key:
            out[key] = value
    return out


def get(key: str, default: str = "") -> str:
    return read().get(key, default)


def put(key: str, value: str) -> None:
    values = read()
    values[key] = value
    try:
        _path().parent.mkdir(parents=True, exist_ok=True)
        write_text(_path(), "".join(f"{k}={v}\n" for k, v in values.items()))
    except OSError:
        pass  # an unwritable folder: the choice lasts for this session
