"""MP4 videos through ffmpeg. Rok does not encode video itself; it hands
the frames to ffmpeg, which must be installed (`brew install ffmpeg`)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

# An app started from the Finder has a short PATH: look where Homebrew puts it too.
_PLACES = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
MISSING = "MP4 needs ffmpeg: install it with brew install ffmpeg"


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    return next((p for p in _PLACES if Path(p).is_file()), None)


def write_mp4(path, frames: list[np.ndarray], fps: float, ffmpeg: str | None = None) -> None:
    """Write `frames` (H x W x 3 or 4 uint8, one size) as an H.264 MP4 that
    plays everywhere (yuv420p; an odd side gets one black pixel more)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    if ffmpeg is None:
        raise OSError(MISSING)
    if not frames:
        raise ValueError("a video needs at least one frame")
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}",
           "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-movflags", "+faststart", str(path)]
    data = b"".join(np.ascontiguousarray(np.asarray(f)[..., :3], dtype=np.uint8).tobytes() for f in frames)
    done = subprocess.run(cmd, input=data, capture_output=True)
    if done.returncode != 0:
        raise OSError(f"ffmpeg: {done.stderr.decode(errors='replace').strip()[-300:]}")


__all__ = ["MISSING", "find_ffmpeg", "write_mp4"]
