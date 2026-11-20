"""ROK_PROFILE=1: profile every event; print the slow ones.

ROK_TIMING (native/src/host.rs) says which event was slow. This says why:
for an event over SLOW_MS it prints the tool, the mode, the brush, the model
size and the functions that took the time, on stderr. The profiler slows
every event down a little, so read the slow ones, not the totals.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

SLOW_MS = 15.0
TOP = 12

enabled = bool(os.environ.get("ROK_PROFILE"))


def run(kind: str, fn) -> None:
    """Run fn(); with ROK_PROFILE set, report it when it was slow."""
    if not enabled:
        fn()
        return
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    try:
        fn()
    finally:
        prof.disable()
        ms = (time.perf_counter() - t0) * 1000.0
        if ms > SLOW_MS:
            _report(kind, ms, prof)


def _report(kind: str, ms: float, prof: cProfile.Profile) -> None:
    from .object import document
    from .opkit import CONTEXT
    from .state import state

    wm = CONTEXT.prefs
    grid = document.grid
    size = "x".join(str(int(v)) for v in grid.voxels.shape) if grid is not None else "-"
    what = (f"tool {getattr(wm, 'rok_tool', '?')} mode {getattr(wm, 'rok_mode', '?')} "
            f"brush {getattr(wm, 'rok_brush_size', '?')} grid {size} level {state.level} "
            f"stroke {state.stroke_active}")
    print(f"rok profile: {kind} {ms:.1f} ms | {what}", file=sys.stderr)
    for order, count in (("cumulative", TOP + 6), ("tottime", TOP)):
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats(order).print_stats(count)
        lines = [ln for ln in out.getvalue().splitlines() if ln.strip()][-count - 1:]
        print(f"rok profile:   by {order} time:", file=sys.stderr)
        for ln in lines:
            print(f"rok profile:   {ln}", file=sys.stderr)
    sys.stderr.flush()
