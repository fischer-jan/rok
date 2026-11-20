"""The open window as an agent target: the MCP server's tools, run in the app.

When Rok runs with a window it listens on a Unix socket, `rok.sock` in its
config folder, readable by the user only. Each connection sends JSON lines
`{"id": .., "tool": .., "args": {..}}` and gets back
`{"id": .., "text": .., "data": {..}, "error": bool, "png": base64 or null}`.

A thread of its own accepts and reads; the tool itself runs on the app's
thread, between events: the reader queues the call and wakes the event loop
(`Host.waker()`), the "wake" event runs it (rok/agent.py), and its picture
comes back as a "screenshot" event. A call waits while a mouse button is
down or a stroke runs, so it never cuts into the user's drag. Every call is one
undo step, and its first line shows in the window.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import socket
import threading
import time
from pathlib import Path

SOCKET_NAME = "rok.sock"
REPLY_TIMEOUT = 180.0
RETRY = 0.1  # seconds between tries while the window is busy


def socket_path() -> Path:
    from . import recent

    return Path(recent.FOLDER) / SOCKET_NAME


class _Call:
    def __init__(self, request: dict) -> None:
        self.request = request
        self.reply: queue.Queue = queue.Queue(maxsize=1)
        self.token: str | None = None
        self.result = None
        self.picture = None
        self.tokens: list[str] = []  # the screenshots the picture needs (a strip of frames: several)
        self.shots: dict = {}  # token -> (width, height, rgba) as they come back


class Live:
    """The listener and the queue of calls for one app."""

    def __init__(self, app, waker, path: Path) -> None:
        self.app = app
        self.waker = waker
        self.path = path
        self.calls: queue.Queue[_Call] = queue.Queue()
        self.waiting: dict[str, _Call] = {}  # picture token -> the call that waits for it
        self.agent = None
        self._token = 0
        self._server: socket.socket | None = None

    # --- the listener thread ---------------------------------------------------------

    def start(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(self.path))
                probe.close()
                return False  # another Rok window listens already
            except OSError:
                self.path.unlink(missing_ok=True)  # left behind by a Rok that ended
            finally:
                probe.close()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            server.bind(str(self.path))
        finally:
            os.umask(old)
        os.chmod(self.path, 0o600)
        server.listen(4)
        self._server = server
        threading.Thread(target=self._accept, name="rok-live", daemon=True).start()
        return True

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.close()
            finally:
                self.path.unlink(missing_ok=True)
                self._server = None

    def _accept(self) -> None:
        while self._server is not None:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), name="rok-live-conn", daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as f:
            for line in f:
                try:
                    request = json.loads(line)
                except ValueError:
                    request = None
                if not isinstance(request, dict):
                    f.write(b'{"error": true, "text": "not a JSON object"}\n')
                    f.flush()
                    continue
                call = _Call(request)
                self.calls.put(call)
                self.waker.wake()
                try:
                    answer = call.reply.get(timeout=REPLY_TIMEOUT)
                except queue.Empty:
                    answer = {"error": True, "text": "Rok did not answer in time."}
                answer["id"] = request.get("id")
                f.write(json.dumps(answer).encode() + b"\n")
                f.flush()

    # --- the app thread --------------------------------------------------------------

    def busy(self) -> bool:
        from .state import state

        return bool(self.app.buttons_down or state.stroke_active or self.app.dialog is not None)

    def on_wake(self) -> None:
        """Run the queued calls, unless the user is in the middle of something."""
        if self.busy():
            if not self.calls.empty():
                self.app.host.tick(RETRY)
            return
        while not self.calls.empty():
            self._run(self.calls.get_nowait())

    def _run(self, call: _Call) -> None:
        """Answer one call. A fault on the way to the answer is logged and
        answered too, so the client never waits out the timeout."""
        try:
            self._answer(call)
        except Exception as exc:
            self.app.report_error("agent call")
            try:
                call.reply.put_nowait({"text": f"Rok could not run this call: {type(exc).__name__}: {exc}",
                                       "data": None, "error": True, "png": None})
            except queue.Full:
                pass  # the call was answered before the fault

    def _answer(self, call: _Call) -> None:
        from rok.agent import Agent

        if self.agent is None:
            self.agent = Agent(self.app, self.app.host)
        req = call.request
        settings = req.get("settings") or {}
        if settings.get("feedback") in ("image", "text"):
            self.agent.feedback = settings["feedback"]
        if isinstance(settings.get("size"), int) and 64 <= settings["size"] <= 2000:
            self.agent.size = (settings["size"], settings["size"])
        reply = self.agent.call(str(req.get("tool", "")), req.get("args") or {})
        first = reply.text.splitlines()[0] if reply.text else ""
        if first:
            self.app.say(f"Agent: {first}", 4.0)
        image = base64.b64encode(reply.image).decode("ascii") if reply.image else None
        call.result = {"text": reply.text, "data": reply.data, "error": bool(reply.error), "png": image}
        if reply.picture is None:
            call.reply.put(call.result)
            return
        self._token += 1
        call.token = f"live-{self._token}"
        call.picture = reply.picture
        call.tokens = self.agent.picture_tokens(reply.picture, call.token)
        for token in call.tokens:
            self.waiting[token] = call
        self.agent.queue_picture(reply.picture, call.token)

    def on_screenshot(self, data: dict) -> bool:
        """A picture for a waiting call: encode it and answer. False when the
        picture is not one of ours."""
        token = str(data.get("token"))
        call = self.waiting.pop(token, None)
        if call is None:
            return False
        from rok.core import png

        call.shots[token] = (int(data["width"]), int(data["height"]), data["rgba"])
        if len(call.shots) < len(call.tokens):
            return True  # a strip of frames: more pictures to come
        encoded = png.encode(self.agent.assemble(call.picture, [call.shots[t] for t in call.tokens]))
        self.agent.finish_picture(call.picture, encoded)
        if call.picture.send:
            call.result["png"] = base64.b64encode(encoded).decode("ascii")
        call.reply.put(call.result)
        return True


_live: Live | None = None


def start(app) -> Live | None:
    """Listen for agent calls, when the host has a window to wake."""
    global _live
    if os.environ.get("ROK_NO_SOCKET"):
        return None
    waker_of = getattr(type(app.host), "waker", None)
    if waker_of is None:
        return None  # a test host, or no native module
    waker = app.host.waker()
    if waker is None:
        return None  # a headless run: Python drives it, no one to wake
    live = Live(app, waker, socket_path())
    try:
        if not live.start():
            return None
    except OSError as exc:  # the editor works without it
        import sys

        print(f"rok: no agent socket at {live.path}: {exc}", file=sys.stderr)
        return None
    _live = live
    import atexit

    atexit.register(live.stop)
    return live


def current() -> Live | None:
    return _live


def connect(path: Path | None = None, timeout: float = 0.5) -> socket.socket | None:
    """A connection to an open Rok window, or None when none listens."""
    path = path or socket_path()
    if not path.exists():
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(path))
    except OSError:
        s.close()
        return None
    s.settimeout(None)
    return s


def wait_for_socket(path: Path, seconds: float = 20.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        s = connect(path)
        if s is not None:
            s.close()
            return True
        time.sleep(0.1)
    return False
