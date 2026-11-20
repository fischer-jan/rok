"""View > Look (rok/app/looks.py) against a recording host."""

from __future__ import annotations

from pathlib import Path

import pytest

from rok.app import looks
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state

FIXTURES = Path(__file__).parent / "fixtures"


class RecordingHost:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "measure_text":
                return (len(args[0]) * 7.0, 16.0)
            return None

        return record

    def named(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def app(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "bird.vox"))
    a.handle("start", {"width": 1200.0, "height": 800.0, "scale": 2.0})
    yield a
    document.reset()
    state.__init__()


def test_a_look_goes_to_the_host_and_is_remembered(app, tmp_path):
    app.handle("menu", {"id": "rok.look(name='PAPER')"})
    kwargs = app.host.named("set_look")[-1][2]
    assert kwargs["ink"] == 1.0 and kwargs["background"] == (0.95, 0.94, 0.91)
    assert (tmp_path / "config" / looks.FILE_NAME).read_text().strip() == "PAPER"


def test_the_next_start_shows_the_remembered_look(app, tmp_path):
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / looks.FILE_NAME).write_text("CLAY\n")
    other = RokApp(RecordingHost(), str(FIXTURES / "bird.vox"))
    other.handle("start", {"width": 1200.0, "height": 800.0, "scale": 2.0})
    assert other.look == "CLAY" and other.host.named("set_look")[-1][2]["clay"] == 1.0


def test_an_unknown_saved_look_falls_back_to_studio(app, tmp_path):
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / looks.FILE_NAME).write_text("SPARKLY\n")
    other = RokApp(RecordingHost(), str(FIXTURES / "bird.vox"))
    other.handle("start", {"width": 1200.0, "height": 800.0, "scale": 2.0})
    assert other.look == "STUDIO"
