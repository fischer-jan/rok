"""Export Palette: the dialog, the file picker and the JSON that comes out; Import Palette reads it back."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rok.app import opkit, palette_ui
from rok.app.app import RokApp
from rok.app.object import document
from rok.app.state import state
from rok.core.material import DEFAULT, Material

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


@pytest.fixture
def app(tmp_path, monkeypatch):
    from rok.app import recent

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    monkeypatch.setenv("ROK_NO_WELCOME", "1")
    document.reset()
    state.__init__()
    a = RokApp(RecordingHost(), str(FIXTURES / "bird.vox"))
    a.handle("start", {"width": 1440.0, "height": 900.0, "scale": 2.0})
    return a


def _used_slot() -> int:
    counts = document.scene.used_counts()
    return int(np.nonzero(counts[1:])[0][0]) + 1


def _export(app, path: Path, **props) -> dict | str:
    opkit.ops.rok.palette_export("INVOKE")
    assert app.dialog is not None and app.dialog.op.command_id == "rok.palette_export"
    op = app.dialog.op
    for name, value in props.items():
        setattr(op, name, value)
    app.dialog._finish(True)
    assert app.dialog is None
    assert any(call[0] == "file_dialog" for call in app.host.calls), "the file picker follows the dialog"
    (token,) = list(app.pending)
    app.on_dialog(token, path=str(path))
    assert path.is_file()
    return json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()


def test_json_export_holds_used_colours_and_favourites(app, tmp_path):
    scene = document.scene
    slot = _used_slot()
    rgb = tuple(int(v) for v in scene.palette[slot][:3])
    palette_ui.add_favourites([(rgb, "bird"), ((1, 2, 3), "sky")])
    data = _export(app, tmp_path / "out.json", favourites=True, used=True, unused=False)
    counts = scene.used_counts()
    used_slots = {int(s) for s in np.nonzero(counts[1:])[0] + 1}
    in_palette = [c for c in data["colours"] if c["slot"] is not None]
    assert {c["slot"] for c in in_palette} == used_slots
    mine = next(c for c in in_palette if c["slot"] == slot)
    assert mine["favourite"] is True and mine["name"] == "bird" and mine["used"] == int(counts[slot])
    assert mine["hex"] == "#{:02X}{:02X}{:02X}".format(*rgb)
    (sky,) = [c for c in data["colours"] if c["slot"] is None]
    assert sky == {"hex": "#010203", "slot": None, "name": "sky", "favourite": True}


def test_unused_only_export_as_hex(app, tmp_path):
    scene = document.scene
    counts = scene.used_counts()
    unused = scene.colours.unused_ids(counts)
    text = _export(app, tmp_path / "out.hex", favourites=False, used=False, unused=True, format="HEX")
    assert len(text.splitlines()) == len(unused)


def test_nothing_chosen_is_refused(app, tmp_path):
    opkit.ops.rok.palette_export("INVOKE")
    op = app.dialog.op
    op.favourites = op.used = op.unused = False
    app.dialog._finish(True)
    assert not any(call[0] == "file_dialog" for call in app.host.calls)
    assert not (tmp_path / "out.json").exists()


def test_import_reads_names_materials_and_favourites_back(app, tmp_path):
    emit = Material(((b"_type", b"_emit"), (b"_emit", b"1"), (b"_flux", b"3")))
    path = tmp_path / "in.json"
    path.write_text(json.dumps({"rok_palette": 1, "colours": [
        {"hex": "#112233", "slot": 1, "name": "lamp", "favourite": True,
         "material": {"_type": "_emit", "_emit": "1", "_flux": "3"}},
        {"hex": "#445566", "slot": 2},
        {"hex": "#778899", "slot": None, "name": "mist", "favourite": True},
    ]}))
    result = opkit.ops.rok.palette_import("EXEC", filepath=str(path), mode="APPEND")
    assert result == {"FINISHED"}
    store = document.scene.colours
    lamp = store.find((0x11, 0x22, 0x33, 255), emit)
    assert lamp is not None and store.entries[lamp].name == "lamp"
    assert store.find((0x44, 0x55, 0x66, 255), DEFAULT) is not None
    assert store.find((0x77, 0x88, 0x99, 255)) is None, "a favourite without a slot stays out of the palette"
    assert palette_ui.favourites()[:2] == [(0x11, 0x22, 0x33), (0x77, 0x88, 0x99)]
    assert palette_ui.favourite_name((0x77, 0x88, 0x99)) == "mist"


def test_imported_favourites_keep_yours_in_place(app):
    palette_ui.add_favourites([((1, 1, 1), "one"), ((2, 2, 2), "two")])
    assert palette_ui.add_favourites([((2, 2, 2), "other name"), ((3, 3, 3), "three"), ((3, 3, 3), "")]) == 1
    assert palette_ui.favourites() == [(3, 3, 3), (1, 1, 1), (2, 2, 2)]
    assert palette_ui.favourite_name((2, 2, 2)) == "two"
    assert palette_ui.favourite_name((3, 3, 3)) == "three"


def test_favourites_hold_sixty_four_and_the_popup_shows_sixteen(app):
    palette_ui.add_favourites([((i, 0, 0), "") for i in range(70)])
    assert len(palette_ui.favourites()) == palette_ui.FAV_MAX == 64
    parts = palette_ui.popup_layout(0.0, 0.0, "METAL", favs=len(palette_ui.favourites()))
    assert "FAV:15" in parts and "FAV:16" not in parts
