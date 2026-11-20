"""The safe writer: the old file stays until the new one is complete."""

from pathlib import Path

import pytest

from rok.app import recent, settings
from rok.core.safe_write import write_bytes, write_text


def test_write_replaces_the_file_and_leaves_no_part(tmp_path):
    target = tmp_path / "robot.vox"
    target.write_bytes(b"old")
    write_bytes(target, b"new")
    assert target.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [target]


def test_a_failed_write_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "robot.vox"
    target.write_bytes(b"old")

    def fail(self, data):
        self.open("wb").write(data[:1])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError):
        write_text(target, "new")
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_a_damaged_settings_file_reads_as_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(recent, "FOLDER", tmp_path)
    (tmp_path / settings.FILE_NAME).write_bytes(b"look=iso\n\xff\xfe\n")
    assert settings.read() == {}
    settings.put("look", "soft")
    assert settings.read() == {"look": "soft"}


def test_a_damaged_recent_list_reads_as_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(recent, "FOLDER", tmp_path)
    monkeypatch.setattr(recent, "FILE", tmp_path / "recent.txt")
    (tmp_path / "recent.txt").write_bytes(b"\xff\xfe/a.vox\n")
    assert recent.load() == []
