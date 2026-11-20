"""A fault in a tool or a bad request still gets an answer (rok/agent.py, rok/app/live.py)."""

from __future__ import annotations

import json
import queue
import socket
import threading

import pytest

pytest.importorskip("rok_native")

from rok.app import live  # noqa: E402


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session()
    yield s
    s.close()


def test_a_fault_in_a_tool_is_an_error_reply_with_the_traceback(session, monkeypatch):
    def broken(**kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(session.agent, "tool_look", broken)
    reply = session.agent.call("rok_look", {})
    assert reply.error
    assert reply.text == "rok_look failed: ValueError: boom"
    assert "ValueError: boom" in reply.data["traceback"]


def test_a_type_error_inside_a_tool_is_not_blamed_on_the_parameters(session, monkeypatch):
    def broken(**kwargs):
        return None + 1

    monkeypatch.setattr(session.agent, "tool_look", broken)
    reply = session.agent.call("rok_look", {})
    assert reply.error
    assert reply.text.startswith("rok_look failed: TypeError")


def test_an_unknown_parameter_names_the_tool_description(session):
    reply = session.agent.call("rok_look", {"colour_of_the_sky": 3})
    assert reply.error
    assert "The tool description lists its parameters" in reply.text


def test_save_and_export_expand_the_home_folder(session, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    session.run("rok.new", size=tuple([8, 8, 8]), discard=True)
    session.tool("rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [3, 3, 3], "colour": 5})
    session.tool("rok_scene", {"do": "save", "path": "~/home_test.vox"})
    assert (tmp_path / "home_test.vox").is_file()
    session.tool("rok_scene", {"do": "export", "path": "~/home_test.glb"})
    assert (tmp_path / "home_test.glb").is_file()
    assert not (tmp_path / "~").exists()


class _Waker:
    def wake(self) -> None:
        pass


class _App:
    host = None

    def __init__(self) -> None:
        self.errors: list[str] = []

    def report_error(self, kind: str) -> None:
        self.errors.append(kind)

    def say(self, text: str, seconds: float) -> None:
        pass


def test_a_request_that_is_not_an_object_is_answered_and_not_queued(tmp_path):
    server = live.Live(_App(), _Waker(), tmp_path / "live.sock")
    ours, theirs = socket.socketpair()
    thread = threading.Thread(target=server._serve, args=(theirs,), daemon=True)
    thread.start()
    with ours, ours.makefile("rwb") as f:
        f.write(b"[1, 2, 3]\nnot json\n")
        f.flush()
        first = json.loads(f.readline())
        second = json.loads(f.readline())
    thread.join(timeout=5)
    assert first["error"] and second["error"]
    assert server.calls.empty()


def test_a_fault_on_the_way_to_the_answer_is_logged_and_answered(tmp_path):
    app = _App()
    server = live.Live(app, _Waker(), tmp_path / "live.sock")

    class _Agent:
        feedback = "image"
        size = (256, 256)

        def call(self, tool, args):
            raise KeyError("missing")

    server.agent = _Agent()
    call = live._Call({"tool": "rok_look", "args": {}})
    server._run(call)
    answer = call.reply.get_nowait()
    assert answer["error"]
    assert "KeyError" in answer["text"]
    assert app.errors == ["agent call"]
    with pytest.raises(queue.Empty):
        call.reply.get_nowait()
