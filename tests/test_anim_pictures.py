"""Pictures of a clip: rok/core/anim_pictures.py, gif.py and video.py, and
the export run through the real renderer in a headless session."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rok.core import anim, gif, video
from rok.core import anim_pictures as pics

ANIM = Path(__file__).parent / "fixtures" / "anim"


# --- times and directions -------------------------------------------------------------


def test_video_times_play_once_round():
    assert pics.video_times(anim.Clip(length=4)) == [0, 1, 2, 3]
    assert pics.video_times(anim.Clip(length=4, loop_mode=anim.ONCE)) == [0, 1, 2, 3]
    assert pics.video_times(anim.Clip(length=4, loop_mode=anim.PING_PONG)) == [0, 1, 2, 3, 2, 1]


def test_sheet_times_follow_the_sheet_rate():
    clip = anim.Clip(fps=24, length=24)
    assert pics.sheet_times(clip) == [float(k) for k in range(24)]
    assert pics.sheet_times(clip, 12) == [float(k) for k in range(0, 24, 2)]
    assert len(pics.sheet_times(anim.Clip(fps=24, length=19), 12)) == 10  # 19 frames at 24: 0.79 s, 10 at 12


def test_directions_are_even_turns():
    assert pics.direction_angles(4) == [0, 90, 180, 270]
    assert len(pics.direction_angles(16)) == 16
    with pytest.raises(ValueError):
        pics.direction_angles(6)


# --- the matte ---------------------------------------------------------------------------


def _srgb(linear):
    return np.rint(pics._to_srgb(np.asarray(linear, dtype=np.float64))).astype(np.uint8)


def test_matte_gives_back_colour_and_coverage():
    """A colour over black and over white, blended in linear light as the
    renderer does, comes back with its coverage as alpha."""
    colour = np.array([0.6, 0.2, 0.05])
    cover = np.array([1.0, 0.5, 0.25, 0.0])
    on_black = _srgb(cover[:, None] * colour)[None]
    on_white = _srgb(cover[:, None] * colour + (1 - cover[:, None]))[None]
    out = pics.matte(on_black, on_white)[0]
    np.testing.assert_allclose(out[:, 3], np.rint(cover * 255), atol=2)
    want = _srgb(colour)
    for pixel in out[:2]:
        np.testing.assert_allclose(pixel[:3], want, atol=3)
    assert tuple(out[3]) == (0, 0, 0, 0)


# --- the sheet -----------------------------------------------------------------------


def test_assemble_lays_out_directions_by_frames():
    clip = anim.Clip(name="Walk", fps=24, length=6, loop_mode=anim.PING_PONG,
                     markers=[anim.Marker(3, "step")])
    cells = [[np.full((8, 10, 4), 10 * d + f, np.uint8) for f in range(3)] for d in range(4)]
    sheet = pics.assemble(cells, clip, 12, pics.direction_angles(4), "walk.png", pivot=(5, 7))
    assert sheet.image.shape == (32, 30, 4)
    assert sheet.image[8 * 2, 10 * 1, 0] == 21  # row 2 (180 degrees), column 1
    meta = sheet.json["meta"]
    assert [t["name"] for t in meta["frameTags"]] == ["Walk 0", "Walk 90", "Walk 180", "Walk 270"]
    assert meta["frameTags"][1] == {"name": "Walk 90", "from": 3, "to": 5, "direction": "pingpong"}
    assert meta["size"] == {"w": 30, "h": 32} and meta["image"] == "walk.png"
    assert meta["markers"] == [{"name": "step", "frame": 2}]  # clip frame 3 at 24 is column 1.5: the next
    assert meta["slices"][0]["keys"][0]["pivot"] == {"x": 5, "y": 7}
    frame = sheet.json["frames"][4]
    assert frame["frame"] == {"x": 10, "y": 8, "w": 10, "h": 8} and frame["duration"] == 83


# --- GIF -----------------------------------------------------------------------------


def _decode_gif(data: bytes) -> list[np.ndarray]:
    """A small GIF reader for the files gif.encode writes: global palette,
    frames that keep the one before, a transparent index."""
    assert data[:6] == b"GIF89a"
    w, h, flags = struct.unpack_from("<HHB", data, 6)
    size = 2 << (flags & 7)
    palette = np.frombuffer(data, np.uint8, size * 3, 13).reshape(size, 3)
    i = 13 + size * 3
    canvas = np.zeros((h, w, 3), np.uint8)
    frames, transparent = [], None
    while data[i] != 0x3B:
        if data[i] == 0x21:
            label = data[i + 1]
            if label == 0xF9:
                packed, _delay, index = struct.unpack_from("<BHB", data, i + 3)
                transparent = index if packed & 1 else None
            i += 2
            while data[i]:
                i += data[i] + 1
            i += 1
            continue
        x, y, fw, fh, _f = struct.unpack_from("<HHHHB", data, i + 1)
        min_size = data[i + 10]
        i += 11
        raw = bytearray()
        while data[i]:
            raw += data[i + 1:i + 1 + data[i]]
            i += data[i] + 1
        i += 1
        indices = _lzw_decode(bytes(raw), min_size)[:fw * fh]
        sub = np.array(indices, np.uint8).reshape(fh, fw)
        region = canvas[y:y + fh, x:x + fw]
        keep = sub == transparent if transparent is not None else np.zeros_like(sub, bool)
        region[~keep] = palette[sub[~keep]]
        frames.append(canvas.copy())
    return frames


def _lzw_decode(data: bytes, min_size: int) -> list[int]:
    clear, end = 1 << min_size, (1 << min_size) + 1
    size, pos, out = min_size + 1, 0, []
    table: list[list[int]] = []
    prev = None
    nbits = len(data) * 8
    while pos + size <= nbits:
        code = int.from_bytes(data[pos // 8:pos // 8 + 3].ljust(3, b"\0"), "little") >> (pos % 8) & ((1 << size) - 1)
        pos += size
        if code == clear:
            table = [[k] for k in range(clear)] + [[], []]
            size, prev = min_size + 1, None
            continue
        if code == end:
            break
        entry = table[code] if code < len(table) else table[prev] + [table[prev][0]]
        out += entry
        if prev is not None:
            table.append(table[prev] + [entry[0]])
            if len(table) == (1 << size) and size < 12:
                size += 1
        prev = code
    return out


def test_gif_frames_come_back_exactly():
    rng = np.random.default_rng(3)
    base = (rng.integers(0, 30, (40, 50, 1)) * np.array([7, 5, 3])).astype(np.uint8)
    moved = base.copy()
    moved[10:20, 5:15] = (250, 10, 10)
    frames = [base, moved, base]
    palette, indices = gif.palette_of(frames)
    decoded = _decode_gif(gif.encode(frames, 10))
    assert len(decoded) == 3
    for got, idx in zip(decoded, indices):
        np.testing.assert_array_equal(got, palette[idx])
    np.testing.assert_array_equal(decoded[1], moved)  # 30 colours: kept exactly


def test_gif_with_many_colours_keeps_them_close():
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)
    palette, _ = gif.palette_of([frame])
    assert len(palette) == 255
    got = _decode_gif(gif.encode([frame], 10))[0]
    assert np.abs(got.astype(int) - frame).mean() < 30


def test_gif_delays_keep_the_frame_rate():
    assert sum(gif.delays(24, 24)) == 100 and set(gif.delays(24, 24)) == {4, 5}


@pytest.mark.skipif(video.find_ffmpeg() is None, reason="ffmpeg is not installed")
def test_ffmpeg_reads_rok_gifs_and_videos(tmp_path):
    ffmpeg = video.find_ffmpeg()
    frames = [np.full((30, 41, 3), v, np.uint8) for v in (20, 120, 220)]
    (tmp_path / "a.gif").write_bytes(gif.encode(frames, 12))
    video.write_mp4(tmp_path / "a.mp4", frames, 12)
    for name, w in (("a.gif", 41), ("a.mp4", 42)):  # an odd width gets one pixel more in a video
        raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(tmp_path / name), "-fps_mode", "passthrough",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        got = np.frombuffer(raw, np.uint8).reshape(-1, 30, w, 3)
        assert len(got) == 3
        np.testing.assert_allclose(got[:, 10, 10, 0], [20, 120, 220], atol=4)


def test_a_video_without_ffmpeg_says_how_to_get_it(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "find_ffmpeg", lambda: None)
    with pytest.raises(OSError, match="brew install ffmpeg"):
        video.write_mp4(tmp_path / "a.mp4", [np.zeros((2, 2, 3), np.uint8)], 10)


# --- the export through the renderer -------------------------------------------------


@pytest.fixture(scope="module")
def walk_session(tmp_path_factory):
    pytest.importorskip("rok_native")
    from rok.headless import Session

    folder = tmp_path_factory.mktemp("pictures")
    shutil.copy(ANIM / "walk.rok", folder / "walk.rok")
    s = Session(folder / "walk.rok", 900, 600)
    yield s, folder
    s.close()


def _export(session, kind: str, **props) -> list[str]:
    from rok.app import anim_pictures

    result = session.run("rok.export_pictures", kind=kind, **props)
    assert result.ok
    for _ in range(1000):
        if not anim_pictures.running():
            break
        session.process()
    assert not anim_pictures.running()
    return session._messages


def test_sprite_sheet_of_the_walk(walk_session):
    from rok.core import png

    session, folder = walk_session
    messages = _export(session, "SPRITES", directions="4", size=64, fps=12)
    assert messages[-1] == "Saved walk_Walk.png, walk_Walk.json"
    data = json.loads((folder / "walk_Walk.json").read_text())
    cols = len(pics.sheet_times(session.document.scene.timeline, 12))
    assert data["meta"]["size"] == {"w": 64 * cols, "h": 64 * 4}
    assert len(data["frames"]) == 4 * cols and data["meta"]["fps"] == 12
    h, w, rgba = png.decode((folder / "walk_Walk.png").read_bytes(), max_pixels=1 << 24)
    assert (w, h) == (64 * cols, 256)
    alpha = rgba[..., 3]
    assert alpha[0, 0] == 0 and (alpha == 255).any()  # a transparent background round an opaque walker
    cells = rgba.reshape(4, 64, cols, 64, 4)
    assert np.abs(cells[0, :, 0].astype(int) - cells[0, :, cols // 2]).max() > 50  # the legs moved
    for d in range(4):  # every cell holds the walker, framed the same
        assert all((cells[d, :, c, :, 3] == 255).sum() > 100 for c in range(cols))


@pytest.mark.skipif(video.find_ffmpeg() is None, reason="ffmpeg is not installed")
def test_gif_and_mp4_of_the_walk(walk_session):
    session, folder = walk_session
    clip = session.document.scene.timeline
    assert _export(session, "GIF")[-1] == "Saved walk_Walk.gif"
    assert _export(session, "MP4")[-1] == "Saved walk_Walk.mp4"
    ffmpeg = video.find_ffmpeg()
    for name in ("walk_Walk.gif", "walk_Walk.mp4"):
        raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(folder / name), "-fps_mode", "passthrough",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        probe = subprocess.run([ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
                                "stream=width,height", "-of", "csv=p=0", str(folder / name)],
                               capture_output=True, text=True, check=True).stdout.split()[0]
        w, h = (int(v) for v in probe.split(","))
        frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
        assert len(frames) == clip.length
        assert np.abs(frames[0].astype(int) - frames[len(frames) // 2]).max() > 50  # the legs moved


def test_a_job_that_gets_no_picture_ends(monkeypatch):
    from rok.app import anim_pictures

    job = anim_pictures.Job(anim_pictures.GIF, anim.Clip(), [0.0], [], Path("x.gif"), 24, sent=100.0)
    monkeypatch.setattr(anim_pictures, "_job", job)
    monkeypatch.setattr(anim_pictures.time, "monotonic", lambda: 100.0 + anim_pictures.STALL - 1)
    assert anim_pictures.running()
    monkeypatch.setattr(anim_pictures.time, "monotonic", lambda: 100.0 + anim_pictures.STALL + 1)
    assert not anim_pictures.running() and anim_pictures._job is None
