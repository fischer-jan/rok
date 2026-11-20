"""The lights of View > Look's Iso, Soft, Night and Ceramic, drawn by the real renderer.

The model: two white boxes side by side. The right one's colour is a Light
(emit) colour in the test of Night's glow.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rok_native")

from rok.app import looks  # noqa: E402
from rok.core.material import Material  # noqa: E402
from rok.core.palette import default_palette  # noqa: E402
from rok.core.vox_read import VoxModel  # noqa: E402
from rok.core.vox_write import write_vox  # noqa: E402

WHITE, LAMP = 1, 2  # palette slots: both white
NEW = ("ISO", "SOFT", "NIGHT", "CERAMIC")


def _boxes(path):
    v = np.zeros((20, 8, 8), dtype=np.uint8)
    v[0:8, :, :] = WHITE
    v[12:20, :, :] = LAMP
    pal = default_palette().copy()
    pal[LAMP] = pal[WHITE]
    write_vox(path, VoxModel(v.shape, v, pal))


def _render(path, look, lamp=False):
    from rok.app.object import document
    from rok.headless import Session

    with Session(path) as s:
        s.app.set_occlusion(False)
        if lamp:
            document.scene.materials.set(LAMP, Material(((b"_type", b"_emit"), (b"_emit", b"1"), (b"_flux", b"1"))))
            s.app.sync.flush(document.scene)  # send the materials, as the agent tools do
            s.process()
        return s.render("iso", (400, 300), look=look, fit="voxels").astype(float)


@pytest.fixture(scope="module")
def boxes(tmp_path_factory):
    path = tmp_path_factory.mktemp("looks") / "boxes.vox"
    _boxes(path)
    return path


def _faces(img):
    """The face tones of the left box, brightest first: every grey that covers
    more than 8 % of the box's pixels."""
    bg = img[2, 2, :3]
    model = np.abs(img[..., :3] - bg).max(axis=2) > 12
    xs = np.where(model.any(axis=0))[0]
    left_half = model.copy()
    left_half[:, (xs.min() + xs.max()) // 2:] = False
    tones = img[..., :3][left_half].mean(axis=1).round()
    values, counts = np.unique(tones, return_counts=True)
    big = values[counts > 0.08 * counts.sum()]
    return np.sort(big)[::-1]


def test_the_new_looks_are_listed_with_their_lights():
    assert set(NEW) <= set(looks.LOOKS)
    lights = [looks.LOOKS[k].light for k in NEW]
    assert lights == [looks.ISO, looks.SOFT, looks.NIGHT, looks.CERAMIC]


def test_iso_gives_the_three_faces_three_clear_tones(boxes):
    tones = _faces(_render(boxes, "ISO"))
    assert len(tones) >= 3
    top, side, dark = tones[:3]
    # Pixel-art steps: full, 75 % and 50 % (255, 191, 127).
    assert top >= 250 and 0.70 < side / top < 0.80 and 0.45 < dark / top < 0.55


@pytest.mark.parametrize("look", NEW)
def test_each_new_look_draws_its_own_picture(boxes, look):
    plain = _render(boxes, "PLAIN")
    img = _render(boxes, look)
    assert np.abs(img - plain).max(axis=2).mean() > 5


def test_night_makes_a_light_colour_glow(boxes):
    off = _render(boxes, "NIGHT")
    on = _render(boxes, "NIGHT", lamp=True)
    w = off.shape[1]
    # The right box (the Light colour) is far brighter; the left box stays as it was.
    right_off, right_on = off[:, w // 2:, :3].mean(), on[:, w // 2:, :3].mean()
    assert right_on > right_off + 15
    assert np.abs(on[:, : w // 3, :3] - off[:, : w // 3, :3]).max() < 2


def test_ceramic_adds_a_glaze_over_soft_diffuse_light(boxes):
    img = _render(boxes, "CERAMIC")
    bg = img[2, 2, :3]
    model = np.abs(img[..., :3] - bg).max(axis=2) > 12
    value = img[..., :3][model].mean(axis=1)
    # The faces stay below white; the glaze is a thin white band along the
    # edges (faces up to 242, 1.7 % of the pixels at 250 or more).
    assert _faces(img)[0] < 248
    assert 0.005 < (value >= 250).mean() < 0.1
