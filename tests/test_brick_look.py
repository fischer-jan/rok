"""The brick look in the renderer: seams between bricks,
through the headless session (the same sync and host as the window)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rok_native")

from rok.core import brick  # noqa: E402
from rok.core.brick import BrickTable  # noqa: E402

from test_headless import _golden  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session(FIXTURES / "brick" / "house.vox")
    yield s
    s.close()


def _set_table(session, table) -> None:
    grid = session.document.grid
    grid.bricks = table
    grid.mark_all_dirty()
    from rok.app import object as voxel_object

    voxel_object.flush_dirty()  # the sync sends the voxels and the seams field


def test_the_house_as_bricks_matches_its_golden_and_differs_from_its_voxels(session):
    data = json.loads((FIXTURES / "brick" / "house.json").read_text())
    _set_table(session, None)
    voxels = session.render("iso", size=(400, 300), look="ISO")
    _set_table(session, BrickTable.from_json(data["bricks"]))
    bricks = session.render("iso", size=(400, 300), look="ISO")
    _golden("brick_house_iso", bricks)
    diff = np.abs(bricks[..., :3].astype(int) - voxels[..., :3].astype(int)).max(axis=2)
    assert (diff > 24).mean() > 0.002  # the seams change a part of the picture
    _set_table(session, None)
    again = session.render("iso", size=(400, 300), look="ISO")
    assert np.abs(again.astype(int) - voxels.astype(int)).max() < 20  # Convert to Voxels takes the look away


def test_two_bricks_of_one_colour_show_a_seam_one_brick_shows_none(session):
    session.run("rok.new", size=(8, 4, 2), discard=True)
    grid = session.document.grid
    with brick.unlocked(grid):
        grid.set_voxels(np.argwhere(np.ones((8, 2, 1), dtype=bool)), 5)
    one = BrickTable([brick.Brick(0, 0, 0, 8, 2, 5)])
    two = BrickTable([brick.Brick(0, 0, 0, 4, 2, 5), brick.Brick(4, 0, 0, 4, 2, 5)])
    _set_table(session, one)
    a = session.render("top", size=(400, 200), look="SOFT").astype(int)
    _set_table(session, two)
    b = session.render("top", size=(400, 200), look="SOFT").astype(int)
    darker = (a[..., :3].sum(axis=2) - b[..., :3].sum(axis=2)) > 30
    cols = np.flatnonzero(darker.any(axis=0))
    assert len(cols) > 0 and cols.max() - cols.min() < 20  # one thin line across the middle
    assert abs((cols.min() + cols.max()) / 2 - 200) < 20
