"""The headless session (rok/headless.py): the app on a host with no window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rok_native")

from rok.core import png  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "headless"


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session(FIXTURES / "bird.vox")
    yield s
    s.close()


def _golden(name: str, rgba: np.ndarray) -> None:
    """Compare with tests/fixtures/headless/<name>.png; ROK_GOLDEN=1 writes it."""
    import os

    path = GOLDEN / f"{name}.png"
    if os.environ.get("ROK_GOLDEN") or not path.exists():
        path.write_bytes(png.encode(rgba))
    _w, _h, want = png.decode(path.read_bytes(), max_pixels=1 << 22)
    assert want.shape == rgba.shape
    diff = np.abs(want.astype(np.int16) - rgba.astype(np.int16)).max(axis=2)
    assert float(diff.mean()) < 1.0 and float((diff > 16).mean()) < 0.002


def _background(rgba: np.ndarray) -> np.ndarray:
    return rgba[2, 2, :3]


def test_a_session_renders_one_view_at_the_size_asked(session):
    rgba = session.render("top", size=(320, 200))
    assert rgba.shape == (200, 320, 4)
    model = np.any(np.abs(rgba[..., :3].astype(int) - _background(rgba)) > 30, axis=2)
    assert 0.05 < model.mean() < 0.9  # the bird fills the view, framed


def test_the_front_and_quad_pictures_match_their_goldens(session):
    _golden("bird_front", session.render("front", size=(400, 300)))
    _golden("bird_quad", session.render("quad", size=(600, 600)))


def test_each_quad_tile_shows_the_model(session):
    rgba = session.render("quad", size=(400, 400))
    for y, x in ((0, 0), (0, 200), (200, 0), (200, 200)):
        tile = rgba[y:y + 200, x:x + 200, :3].astype(int)
        assert np.any(np.abs(tile - tile[2, 2]) > 30, axis=2).mean() > 0.02


def test_an_edit_shows_in_the_next_picture_and_undo_takes_it_back(session):
    before = session.render("top", size=(300, 300))
    voxels = session.document.grid.voxels.copy()
    result = session.run("rok.clear_all")
    assert result.ok
    assert not session.document.grid.voxels.any()
    after = session.render("top", size=(300, 300))
    assert not np.array_equal(before, after)
    session.run("rok.undo")
    assert np.array_equal(session.document.grid.voxels, voxels)
    assert np.abs(session.render("top", size=(300, 300)).astype(int) - before).max() < 20


def test_the_editor_view_with_its_panels_is_a_different_picture(session):
    clean = session.render("iso", size=(1000, 1000))
    ui = session.render(ui=True)
    assert ui.shape == clean.shape  # the session's window size is 1000 x 1000
    assert not np.array_equal(ui, clean)


def test_an_unknown_operator_or_view_says_what_exists(session):
    from rok.headless import SessionError

    with pytest.raises(SessionError, match="commands"):
        session.run("rok.no_such_thing")
    with pytest.raises(SessionError, match="iso"):
        session.render("sideways")
    assert "rok.mirror_half" in session.commands()


def test_a_second_session_waits_for_the_first_to_close(session):
    from rok.headless import Session, SessionError

    with pytest.raises(SessionError, match="one Session"):
        Session()


def test_the_command_line_runs_a_script_with_a_session(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "quad.png"
    script = tmp_path / "script.py"
    script.write_text(f"open({str(out)!r}, 'wb').write(session.render_png('quad', (200, 200)))\n"
                      "print(tuple(session.document.grid.size))\n")
    done = subprocess.run([sys.executable, "-m", "rok", "run", str(script), str(FIXTURES / "bird.vox")],
                          capture_output=True, text=True, timeout=120, cwd=Path(__file__).parents[1])
    assert done.returncode == 0, done.stderr
    assert "(80, 68, 22)" in done.stdout
    _w, _h, rgba = png.decode(out.read_bytes(), max_pixels=1 << 22)
    assert rgba.shape == (200, 200, 4)


def test_an_opened_file_keeps_its_cells_and_saving_moves_nothing(session, tmp_path):
    """Opening must not move the models: an LLM's cells are the file's cells."""
    from rok.core.vox_read import read_scene

    src = FIXTURES / "headless" / "off_centre.vox"
    want = [tuple(m.position) for m in read_scene(src).models]
    try:
        assert session.open(src).ok
        assert [tuple(m.position) for m in session.document.scene.models] == want
        out = tmp_path / "again.vox"
        assert session.save(out).ok
        assert [tuple(m.position) for m in read_scene(out).models] == want
    finally:
        session.open(FIXTURES / "bird.vox")
