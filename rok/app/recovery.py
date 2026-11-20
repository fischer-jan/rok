"""The recovery copy: unsaved work survives a crash.

While the document has unsaved changes, Rok writes a Rok file of it to the
settings folder's Recovery folder, WAIT_S after the first change and then
again WAIT_S after each later one. The user's own file changes only when they
save. Each running Rok has its own copy, `<pid>.rok`, and beside it
`<pid>.json` with the original path and the time.

The copy goes away when the document has no unsaved changes any more (Save,
New, Open) and when Rok quits. So a copy whose Rok no longer runs is work
that a crash left behind: at the next start Rok asks to restore it.

A write never stops a stroke or a drag: it waits for them. The scene is copied
on the main thread (a few milliseconds) and written in a worker thread (half a
second at a 256 cube).
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path

from rok.core import rokfile
from rok.core.safe_write import write_text

from . import object as voxel_object
from . import recent
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action
from .state import state

WAIT_S = 30.0  # after a change, before the copy is written
RETRY_S = 2.0  # while a stroke or a drag runs
KEEP_DAYS = 14  # an older copy is deleted without a question

_timer = None  # the app timer of the next write, or None
_lock = threading.Lock()
_generation = 0  # bumps when the copy is discarded; a write of an older one deletes its result
_worker: threading.Thread | None = None


def folder() -> Path:
    return recent.FOLDER / "Recovery"


def copy_path(pid: int | None = None) -> Path:
    return folder() / f"{os.getpid() if pid is None else pid}.rok"


def meta_path(pid: int | None = None) -> Path:
    return folder() / f"{os.getpid() if pid is None else pid}.json"


def _app():
    app = CONTEXT.app
    if app is None or not getattr(app, "_started", False) or not hasattr(app, "timer_add"):
        return None
    return app


# --- writing ------------------------------------------------------------------------


def changed(modified: bool) -> None:
    """The document's unsaved flag was set (object.Document.modified)."""
    if modified:
        schedule()
    else:
        discard()


def schedule() -> None:
    """Write the copy WAIT_S from now, unless a write is already waiting."""
    global _timer
    app = _app()
    if app is None or _timer is not None:
        return
    _timer = app.timer_add(WAIT_S)
    _timer.callback = _due


def _busy(app) -> bool:
    return bool(state.stroke_active or getattr(app, "buttons_down", 0)
                or (_worker is not None and _worker.is_alive()))


def _due() -> None:
    global _timer
    app = _app()
    if app is None:
        _timer = None
        return
    if _busy(app):
        _timer.next_at = time.monotonic() + RETRY_S
        return
    app.timer_remove(_timer)
    _timer = None
    write_now()


def write_now(wait: bool = False) -> None:
    """Copy the document now and write it in a worker thread (`wait`: in this
    thread, for tests). Nothing without unsaved changes."""
    global _worker
    if document.scene is None or not document.modified:
        return
    scene = copy.deepcopy(document.scene)
    from .io_ops import document_dict  # late: io_ops imports this module

    doc = document_dict()
    meta = {"path": document.filepath, "name": document.name, "time": time.time()}
    with _lock:
        generation = _generation
    if wait:
        _write(scene, doc, meta, generation)
        return
    _worker = threading.Thread(target=_write, args=(scene, doc, meta, generation), daemon=True)
    _worker.start()


def _write(scene, doc: dict, meta: dict, generation: int) -> None:
    try:
        folder().mkdir(parents=True, exist_ok=True)
        rokfile.write(copy_path(), scene, doc)
        write_text(meta_path(), json.dumps(meta))
    except OSError:
        return  # the next change tries again; a full disk must not stop the editor
    with _lock:
        stale = generation != _generation
    if stale:  # discarded while this write ran
        _delete(os.getpid())


def discard() -> None:
    """No unsaved work any more: drop the waiting write and delete the copy."""
    global _timer, _generation
    if _timer is not None:
        app = CONTEXT.app
        if app is not None and hasattr(app, "timer_remove"):
            app.timer_remove(_timer)
        _timer = None
    with _lock:
        _generation += 1
    _delete(os.getpid())


def _delete(pid: int) -> None:
    for path in (copy_path(pid), meta_path(pid), copy_path(pid).with_name(copy_path(pid).name + ".part")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


# --- finding and restoring ---------------------------------------------------------


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it runs, as another user
    return True


def left_behind() -> list[dict]:
    """The copies of Roks that no longer run, newest first. Each is the
    copy's meta with "pid" added. Deletes copies older than KEEP_DAYS and
    the half-written files of dead Roks."""
    found = []
    try:
        names = os.listdir(folder())
    except OSError:
        return []
    now = time.time()
    pids = {int(n.split(".")[0]) for n in names if n.split(".")[0].isdigit()}
    for pid in pids:
        if _alive(pid):
            continue
        try:
            meta = json.loads(meta_path(pid).read_text())
            ok = copy_path(pid).is_file() and isinstance(meta, dict)
        except (OSError, ValueError):
            ok = False
        if not ok or now - float(meta.get("time", 0)) > KEEP_DAYS * 86400:
            _delete(pid)
            continue
        meta["pid"] = pid
        found.append(meta)
    return sorted(found, key=lambda m: -float(m.get("time", 0)))


def restore(pid: int) -> bool:
    """Make the copy of `pid` the document: unsaved, with the original path,
    so Save writes the original file. Writes this Rok's own copy of it, then
    deletes the old one. False when it cannot be read (it is deleted then too)."""
    try:
        meta = json.loads(meta_path(pid).read_text())
        scene, doc = rokfile.read(copy_path(pid))
    except (OSError, ValueError):
        _delete(pid)
        return False
    active = doc.get("active") if isinstance(doc, dict) else None
    uid = scene.models[active].uid if isinstance(active, int) and 0 <= active < len(scene.models) else None
    voxel_object.apply_scene(CONTEXT, scene, uid, settle=False)
    document.filepath = str(meta.get("path") or "")
    if isinstance(doc, dict) and isinstance(doc.get("colour"), int):
        document.colour = doc["colour"]
    document.modified = True
    # The new copy first, at once: a second crash soon after must still find the work.
    write_now(wait=True)
    if copy_path().is_file():
        _delete(pid)
    from . import welcome

    welcome.close()
    app = CONTEXT.app
    if app is not None:
        app.update_title()
    return True


def _when(meta: dict) -> str:
    t = float(meta.get("time", 0))
    same_day = time.strftime("%Y%m%d", time.localtime(t)) == time.strftime("%Y%m%d")
    return time.strftime("%H:%M" if same_day else "%d.%m. %H:%M", time.localtime(t))


class _Answer(Action):
    """The restore question's answer: Restore runs execute, Discard cancel."""

    title = ""

    def __init__(self, pid: int, rest: list[dict]) -> None:
        super().__init__()
        object.__setattr__(self, "_pid", pid)
        object.__setattr__(self, "_rest", rest)

    def execute(self, context):
        if not restore(object.__getattribute__(self, "_pid")):
            if context.app is not None:
                context.app.say("The recovery copy could not be read")
        ask(object.__getattribute__(self, "_rest"))
        return {"FINISHED"}

    def cancel(self, context):
        _delete(object.__getattribute__(self, "_pid"))
        ask(object.__getattribute__(self, "_rest"))


RESTORE = "Restore"
DISCARD = "Discard"


def ask(copies: list[dict]) -> None:
    """Ask about the first copy; the answer asks about the next."""
    app = CONTEXT.app
    if not copies or app is None or not hasattr(app, "pending"):
        return
    from . import welcome

    welcome.skip_at_start()
    meta, rest = copies[0], copies[1:]
    name = meta.get("name") or "Untitled"
    token = app._new_token("confirm")
    app.pending[token] = ("confirm", _Answer(int(meta["pid"]), rest), RESTORE)
    app.host.ask(token, "Restore unsaved work?",
                 f"Rok closed before you saved {name}. The copy is from {_when(meta)}.", [RESTORE, DISCARD])


def check_at_start(context=None) -> None:
    if os.environ.get("ROK_NO_RECOVERY"):
        return
    ask(left_behind())


check_at_start.late = True
registry.startup.append(check_at_start)
voxel_object.modified_hooks.append(changed)
