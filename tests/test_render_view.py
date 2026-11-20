"""The rendered view: the path tracer in the viewport (native/src/trace.rs,
rok/app/render_view.py). The picture tests run headless on the real GPU."""

from __future__ import annotations

import numpy as np
import pytest

from rok.core.material import MaterialTable, render_rows

rok_native = pytest.importorskip("rok_native")

RED, GLASS, GREY, LIGHT = 10, 11, 12, 13


def test_render_rows_carry_the_channels_and_the_type():
    table = MaterialTable()
    table.set(GLASS, table.get(GLASS).with_type("GLASS").with_weight(0.8))
    table.set(LIGHT, table.get(LIGHT).with_type("EMIT").with_value(b"_flux", 1))
    rows = render_rows(table)
    assert rows.shape == (256, 8) and rows.dtype == np.float32
    assert rows[GLASS, 2] == pytest.approx(0.8) and rows[GLASS, 6] == 2  # transmission, type
    assert rows[LIGHT, 3] > 0 and rows[LIGHT, 6] == 3  # emission, type
    assert rows[RED, 6] == 0 and rows[RED, 1] == pytest.approx(0.9)  # plain


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session(None, 640, 480)
    yield s
    s.close()


def _rendered_on(session) -> None:
    """The rendered view on, as its button does: a later tick (a swatch that
    flies into the strip) then keeps the test's own sample count."""
    from rok.app import render_view

    render_view.set_rendered(session.app, True)


def _no_flights() -> None:
    """Land the strip's swatches that are still flying to their section, so
    two pictures compare the scene, not a moment of that animation."""
    from rok.app import overlay

    overlay._swatch_flights.clear()
    overlay._swatch_last.clear()
    overlay._FLIGHTS.clear()  # the palette dock's own flights


def _scene(session) -> None:
    """A red wall behind a pane: clear glass on the left, opaque grey on the
    right; a light block in front."""
    from rok.app import object as voxel_object
    from rok.app.object import document

    session.run("rok.new", size=tuple([24, 24, 16]), discard=True)
    pal = document.scene.palette
    pal[RED] = (220, 30, 30, 255)
    pal[GLASS] = (245, 250, 255, 255)
    pal[GREY] = (120, 120, 120, 255)
    pal[LIGHT] = (255, 240, 200, 255)
    mt = document.scene.materials
    mt.set(GLASS, mt.get(GLASS).with_type("GLASS").with_weight(1.0))
    mt.set(LIGHT, mt.get(LIGHT).with_type("EMIT").with_value(b"_flux", 2))
    for low, high, colour in (((2, 18, 0), (21, 19, 13), RED), ((2, 10, 0), (11, 10, 13), GLASS),
                              ((12, 10, 0), (21, 10, 13), GREY), ((10, 2, 0), (13, 4, 2), LIGHT)):
        reply, _png = session.tool("rok_shape", {"kind": "box", "low": list(low), "high": list(high), "colour": colour})
        assert not reply.error, reply.text
    voxel_object.update_palette()
    # From the front (-y), a little above: the glass pane in front of the wall.
    import math

    from rok.app.object import model_origin

    cam = session.app.camera
    cam.target = np.asarray(model_origin(document.model), dtype=np.float64) + (12.0, 12.0, 6.0)
    cam.yaw, cam.pitch, cam.distance = -math.pi / 2, 0.12, 55.0
    session.app.send_camera()
    session.process()


def _at(session, cell) -> tuple[int, int]:
    """The screenshot pixel (row, column) of a point in the model's cells."""
    from rok.app.object import document, model_origin

    app = session.app
    o = np.asarray(model_origin(document.model), dtype=np.float64)
    x, y = app.camera.project([o + np.asarray(cell, dtype=np.float64)])[0]
    return int(round(y)), int(round(x))


def _patch(img, rc, r=3) -> np.ndarray:
    y, x = rc
    return img[y - r:y + r + 1, x - r:x + r + 1, :3].reshape(-1, 3).astype(np.float64).mean(axis=0)


def test_the_rendered_view_sees_through_glass_and_lights_up(session):
    from rok.app import render_view

    _scene(session)
    _rendered_on(session)
    session.host.set_render(True, max_samples=48)
    img = session.render(ui=True)
    assert session.host.render_samples() == 48  # it waits for all its samples
    through_glass = _patch(img, _at(session, (6.5, 10.0, 7.5)))
    opaque = _patch(img, _at(session, (16.5, 10.0, 7.5)))
    light = _patch(img, _at(session, (11.5, 2.0, 1.5)))
    def red(c):
        return c[0] - (c[1] + c[2]) / 2

    assert red(through_glass) > 80  # the red wall shows through the glass
    assert red(opaque) < 40  # the grey pane hides it (the warm light tints it a little)
    assert light.min() > 200  # the light block glows
    session.host.set_render(False)
    session.app.rendered = False
    plain = session.render(ui=True)
    assert abs(_patch(plain, _at(session, (6.5, 10.0, 7.5)))[0] - through_glass[0]) > 10  # back to the editing look
    assert render_view.rendered(session.app) is False


def test_an_edit_updates_the_rendered_scene_in_place_as_a_full_build_would(session):
    """An edit reaches the rendered view by box (no full build); the picture
    must be the one a full build of the same scene gives, pixel for pixel."""
    if session.host.adapter_info().get("software"):
        # On a CPU renderer (lavapipe) the first picture after an edit differs
        # from a full build in a few hundred pixels; later pictures agree. Open.
        pytest.skip("the first picture after an in-place edit differs on a software GPU")
    from rok.app import object as voxel_object
    from rok.app.object import document

    _scene(session)
    _rendered_on(session)
    session.host.set_render(True, max_samples=8)
    _no_flights()
    before = session.render(ui=True)
    cells = np.array([(x, 6, z) for x in range(4, 9) for z in range(0, 3)])
    document.grid.set_voxels(cells, LIGHT)  # a new light, in front of the glass
    document.grid.set_voxels(np.array([(14, 10, 5)]), 0)  # a hole in the grey pane
    voxel_object.sync_scene()
    session.process()
    _no_flights()
    edited = session.render(ui=True)
    assert not np.array_equal(edited, before)
    session.app.sync.reset()  # the host forgets everything: a full build
    voxel_object.sync_scene()
    session.process()
    _no_flights()
    full = session.render(ui=True)
    assert np.array_equal(edited, full)
    session.host.set_render(False)
    session.app.rendered = False


def test_bloom_glows_around_a_light_and_the_outline_inks_the_edges(session):
    from rok.core import light

    _scene(session)
    session.app.rendered = True
    near_light = _at(session, (11.5, 2.0, 4.0))  # just above the light block, on the background
    plain = dict(max_samples=16, ground=None, backdrop=(0.2, 0.2, 0.2), tone=1)
    session.host.set_render(True, **plain)
    flat = session.render(ui=True)
    session.host.set_render(True, bloom=1.0, **plain)
    glow = session.render(ui=True)
    assert _patch(glow, near_light, 2).mean() > _patch(flat, near_light, 2).mean() + 15
    # Diorama's ink: the pane's top edge against the background gets darker.
    edge = _at(session, (16.5, 10.0, 14.0))
    lo, hi = (0, 0, 0), (23, 23, 15)
    settings = light.trace_settings(light.from_preset("DIORAMA"), lo, hi)
    session.host.set_render(True, **{**settings, "max_samples": 16, "outline": 0.0, "voxel_grid": 0.0})
    clean = session.render(ui=True)
    session.host.set_render(True, **{**settings, "max_samples": 16})
    inked = session.render(ui=True)
    def darkest(img):  # the ink is a line one pixel wide: the darkest pixel near the edge
        y, x = edge
        return img[y - 3:y + 4, x - 3:x + 4, :3].astype(float).mean(axis=2).min()

    assert darkest(inked) < darkest(clean) - 30
    session.host.set_render(False)
    session.app.rendered = False


def test_render_picture_saves_the_presets_picture_next_to_the_file(session, tmp_path):
    from rok.app import opkit
    from rok.app.object import document
    from rok.core import png, render_presets as rp

    import dataclasses

    from rok.app import render_picture

    _scene(session)
    document.filepath = str(tmp_path / "panes.vox")
    for pid, size in (("ISO", (256, 256)), ("STUDIO", (320, 240)), ("NIGHT", (320, 240))):
        # Small and few samples for the test: the presets' own sizes are 2K.
        quick = dataclasses.replace(rp.preset_by_id(pid), resolution=size, samples=8)
        path = render_picture.send(session.app, quick, own=False)  # the preset's own light
        session.process()
        assert path == tmp_path / f"panes_{pid.lower()}.png" and path.exists()
        h, w, rgba = png.decode(path.read_bytes(), max_pixels=1 << 20)
        assert (w, h) == size
        if pid == "ISO":
            assert rgba[0, 0, 3] == 0 and rgba[h // 2, w // 2, 3] == 255  # transparent around the model
            assert int(rgba[0, 0, :3].max()) == 0  # a clear pixel holds no sky colour
        else:
            assert rgba[..., 3].min() == 255
    assert opkit.REGISTRY["rok.render_picture"].poll(None)


def test_a_clear_picture_keeps_the_face_colour(session, tmp_path):
    """A pixel the model only partly covers keeps the face colour. The sky
    stays out of it, and a fully clear pixel is black."""
    from rok.core import png

    reply, _png = session.tool("rok_scene", {"do": "new", "size": [6, 6, 6],
                                               "budget": {"voxels": 64, "colours": 2}, "discard": True})
    assert not reply.error, reply.text
    reply, _png = session.tool("rok_edit", {"do": "palette", "colours": {"green": "#2a4630"}})
    assert not reply.error, reply.text
    reply, _png = session.tool("rok_shape", {"kind": "box", "low": [1, 1, 1], "high": [4, 4, 4], "colour": 1})
    assert not reply.error, reply.text
    path = tmp_path / "clear.png"
    reply, _png = session.tool("rok_render", {
        "preset": "iso", "azimuth": 262, "elevation": 14, "fov": 28,
        "ortho": False, "samples": 32, "size": [160, 160], "path": str(path),
    })
    assert not reply.error, reply.text
    _h, _w, rgba = png.decode(path.read_bytes(), 160 * 160)
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3]
    assert rgb[alpha == 0].max() == 0
    solid = rgb[alpha == 255]
    edge = rgb[(alpha >= 128) & (alpha < 255)]
    assert solid.shape[0] > 100 and edge.shape[0] > 20
    # The old picture stored the sky, about 154, in these covered pixels.
    assert edge.mean() < 80
    assert edge.mean() > 8


def _cover(session, tmp_path, size) -> np.ndarray:
    """Which pixels a column at the grid's +x side covers in a clear picture, in a grid of `size`."""
    from rok.core import png

    reply, _png = session.tool("rok_scene", {"do": "new", "size": list(size),
                                               "budget": {"voxels": 64, "colours": 2}, "discard": True})
    assert not reply.error, reply.text
    reply, _png = session.tool("rok_edit", {"do": "palette", "colours": {"green": "#2a4630"}})
    assert not reply.error, reply.text
    reply, _png = session.tool("rok_shape", {"kind": "box", "low": [4, 4, 0], "high": [4, 7, 7], "colour": 1})
    assert not reply.error, reply.text
    path = tmp_path / f"column_{size[0]}.png"
    reply, _png = session.tool("rok_render", {
        "preset": "iso", "azimuth": 256, "elevation": 12, "fov": 28,
        "ortho": False, "samples": 4, "size": [200, 200], "path": str(path),
    })
    assert not reply.error, reply.text
    _h, _w, rgba = png.decode(path.read_bytes(), 200 * 200)
    return rgba[..., 3] > 0


def test_a_ray_that_left_the_grid_hits_nothing(session, tmp_path):
    """A grid whose size is not a multiple of four ends inside a block of the occupancy masks.
    A ray that crosses such an empty block and leaves the grid there is gone: it must not come
    back in and hit a voxel at the grid's edge. The same column in a grid of whole blocks gives
    the picture to compare with."""
    part = _cover(session, tmp_path, (5, 12, 8))  # the column stands in the grid's last layer
    whole = _cover(session, tmp_path, (8, 12, 8))
    assert whole.sum() > 2000
    # The fault drew the column's front face about twice as wide.
    differ, covered = int((part != whole).sum()), int(whole.sum())
    assert differ < 0.02 * covered


def test_outline_and_voxel_grid_grow_thicker_past_black(session):
    """Up to 1 the strength is how dark the line is; past it the line grows wide."""
    from rok.core import light

    _scene(session)
    session.app.rendered = True
    edge = _at(session, (16.5, 10.0, 14.0))
    lo, hi = (0, 0, 0), (23, 23, 15)
    base = {**light.trace_settings(light.from_preset("DIORAMA"), lo, hi), "max_samples": 16}

    def picture(**over):
        session.host.set_render(True, **{**base, **over})
        return session.render(ui=True)

    def dark_pixels(img, centre, half=12):
        y, x = centre
        patch = img[y - half:y + half + 1, x - half:x + half + 1, :3].astype(float).mean(axis=2)
        return int((patch < 40).sum())

    thin = picture(outline=1.0, voxel_grid=0.0)
    thick = picture(outline=3.0, voxel_grid=0.0)
    assert dark_pixels(thick, edge) > dark_pixels(thin, edge) + 8  # the maximum draws a much wider line
    face = _at(session, (12.0, 6.0, 14.0))
    soft = picture(outline=0.0, voxel_grid=0.5)
    hard = picture(outline=0.0, voxel_grid=2.0)
    assert _patch(hard, face, 12).mean() < _patch(soft, face, 12).mean() - 2  # more of the face is ink
    session.host.set_render(False)
    session.app.rendered = False


def test_grid_lines_fade_out_when_a_voxel_is_only_a_few_pixels_wide(session):
    """Lines every two pixels would only turn the face dark: they fade out as voxels shrink."""
    from rok.core import light

    _scene(session)
    session.app.rendered = True
    base = {**light.trace_settings(light.from_preset("DIORAMA"), (0, 0, 0), (23, 23, 15)), "max_samples": 32,
            "outline": 0.0}

    def face_mean(grid):
        session.host.set_render(True, **{**base, "voxel_grid": grid})
        img = session.render(ui=True)
        y, x = _at(session, (12.0, 6.0, 14.0))
        return _patch(img, (y, x), 14).mean()

    near = face_mean(0.0) - face_mean(1.0)  # close up the lines darken the face
    assert near > 4
    session.app.camera.zoom(0.12)  # far away: a voxel is about two pixels
    # Send the camera now: camera_moved alone leaves it to the next tick, and
    # a picture taken before that tick shows the near view at the far pixel.
    session.app.send_camera()
    session.process()
    far = face_mean(0.0) - face_mean(1.0)
    assert far < near * 0.35  # the face stays itself
    session.host.set_render(False)
    session.app.rendered = False


def test_outline_ink_is_averaged_over_the_samples(session):
    """Each sample decides for itself whether it is next to an edge, so more samples
    give the line soft edges (pixels only partly inked). A fixed ink would not change."""
    from rok.core import light

    _scene(session)
    session.app.rendered = True

    def partial_pixels(samples):
        base = {**light.trace_settings(light.from_preset("DIORAMA"), (0, 0, 0), (23, 23, 15)),
                "max_samples": samples, "voxel_grid": 0.0}
        pictures = []
        for outline in (0.0, 1.0):
            session.host.set_render(True, **{**base, "outline": outline})
            pictures.append(session.render(ui=True))
        h, w = pictures[0].shape[:2]
        clean, inked = (p[h // 4:3 * h // 4, w // 4:3 * w // 4, :3].astype(float).sum(axis=2) for p in pictures)
        ratio = inked / np.maximum(clean, 1.0)
        return int(((ratio > 0.15) & (ratio < 0.85)).sum())

    assert partial_pixels(64) > partial_pixels(1) * 1.15
    session.host.set_render(False)
    session.app.rendered = False


def test_the_outline_stays_on_the_models_and_not_on_the_ground(session):
    """Seen at a low angle the ground steps back fast from pixel to pixel: it gets no ink, and
    neither does its horizon."""
    from rok.core import light

    _scene(session)
    _no_flights()
    session.app.rendered = True
    session.app.camera.pitch, session.app.camera.distance = 0.05, 70.0
    session.app.send_camera()
    session.process()
    base = {**light.trace_settings(light.from_preset("DIORAMA"), (0, 0, 0), (23, 23, 15)), "max_samples": 16,
            "voxel_grid": 0.0, "ground": (0.2, 0.5, 0.2), "backdrop": (0.9, 0.9, 0.95)}
    pictures = []
    for outline in (0.0, 1.0):
        session.host.set_render(True, **{**base, "outline": outline})
        pictures.append(session.render(ui=True)[:, :, :3].astype(float).sum(axis=2))
    clean, inked = pictures
    h, w = clean.shape
    horizon = (clean - inked)[int(h * 0.15):int(h * 0.6), int(w * 0.68):int(w * 0.74)]  # ground and sky, no model, no panels
    assert (horizon > 150).sum() == 0  # the old band was black on green: far more
    around_model = (clean - inked)[int(h * 0.1):int(h * 0.6), int(w * 0.3):int(w * 0.7)]
    assert (around_model > 30).sum() > 50  # the models keep their outline
    session.host.set_render(False)
    session.app.rendered = False


def test_the_outline_goes_all_the_way_round_a_model_on_the_floor(session):
    """Against the floor the line stays: down the sides to the foot and along the front
    edges, where the floor beside the model is as near as the model or nearer."""
    import math

    from rok.app.object import document, model_origin
    from rok.core import light

    session.run("rok.new", size=(24, 24, 16), discard=True)
    reply, _png = session.tool("rok_shape", {"kind": "box", "low": [8, 8, 0], "high": [15, 15, 7], "colour": GREY})
    assert not reply.error, reply.text
    _no_flights()
    cam = session.app.camera
    cam.target = np.asarray(model_origin(document.model), dtype=np.float64) + (12.0, 12.0, 4.0)
    cam.yaw, cam.pitch, cam.distance = -math.pi / 2 + 0.6, 0.5, 45.0
    session.app.send_camera()
    session.process()
    session.app.rendered = True
    base = {**light.trace_settings(light.from_preset("DIORAMA"), (0, 0, 0), (23, 23, 15)), "max_samples": 16,
            "voxel_grid": 0.0, "ground": (0.75, 0.75, 0.75), "backdrop": (0.75, 0.75, 0.75)}
    pictures = []
    for outline in (0.0, 1.0):
        session.host.set_render(True, **{**base, "outline": outline})
        pictures.append(session.render(ui=True)[:, :, :3].astype(float).mean(axis=2))
    clean, inked = pictures

    def darkened(cell):  # the ink is a thin line: the most darkened pixel near the edge
        y, x = _at(session, cell)
        return (clean - inked)[y - 3:y + 4, x - 3:x + 4].max()

    corners = [(8, 8), (16, 8), (16, 16), (8, 16)]
    rows = [_at(session, (cx, cy, 0))[0] for cx, cy in corners]
    near = int(np.argmax(rows))  # lowest on screen: the corner nearest the camera
    cx, cy = corners[near]
    edges = []
    for nx, ny in (corners[(near + 1) % 4], corners[(near + 3) % 4]):
        edges += [(cx + (nx - cx) * f, cy + (ny - cy) * f, 0.0) for f in (0.3, 0.5, 0.7)]  # front, on the floor
    cols = [_at(session, (qx, qy, 0))[1] for qx, qy in corners]
    for side in (int(np.argmin(cols)), int(np.argmax(cols))):  # the left and right edges, at the foot
        sx, sy = corners[side]
        edges += [(sx, sy, 0.6), (sx, sy, 1.5)]
    missing = [cell for cell in edges if darkened(cell) < 40]
    assert not missing, f"no outline at {missing}"
    session.host.set_render(False)
    session.app.rendered = False


def test_the_voxel_grid_stays_on_the_models_and_not_on_the_ground(session):
    from rok.core import light

    _scene(session)
    _no_flights()
    session.app.rendered = True
    session.app.camera.pitch, session.app.camera.distance = 0.3, 70.0
    session.app.send_camera()
    session.process()
    base = {**light.trace_settings(light.from_preset("DIORAMA"), (0, 0, 0), (23, 23, 15)), "max_samples": 16,
            "outline": 0.0, "ground": (0.2, 0.5, 0.2), "backdrop": (0.9, 0.9, 0.95)}
    pictures = []
    for grid in (0.0, 1.0):
        session.host.set_render(True, **{**base, "voxel_grid": grid})
        pictures.append(session.render(ui=True)[:, :, :3].astype(float).sum(axis=2))
    clean, gridded = pictures
    h, w = clean.shape
    ground = (clean - gridded)[int(h * 0.75):int(h * 0.92), int(w * 0.68):int(w * 0.74)]  # ground only, no model, no panels
    assert (ground > 100).sum() == 0
    on_model = (clean - gridded)[int(h * 0.1):int(h * 0.6), int(w * 0.3):int(w * 0.7)]
    assert (on_model > 30).sum() > 50  # the models keep their grid
    session.host.set_render(False)
    session.app.rendered = False
