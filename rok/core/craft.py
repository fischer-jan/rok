"""Checks a voxel artist makes by eye, as numbers: jagged steps on a silhouette's
edge, features one cell thin, how much of a silhouette survives when the piece
is shown small, which colours each separate part is made of, and symmetry.
Numpy only; the views come from rok.core.voxel_views.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rok.core import region, voxel_views

MIN_SLOPE = 4  # flat runs in one slope before its rhythm is judged
THIN_RUN = 4  # a line one cell thin this long or longer is a thin feature


@dataclass
class Jag:
    side: str  # the view: front, right or top
    edge: str  # which edge of the silhouette: top, bottom, left or right
    runs: list[int]  # the flat runs of the slope, in order
    cells: tuple[int, int]  # the first and last column (or row) of the slope, in view columns (rows)


@dataclass
class Report:
    jags: list[Jag] = field(default_factory=list)
    thin: list[dict] = field(default_factory=list)  # {"side", "low", "high", "cells"}: in view rows and columns
    survives: dict = field(default_factory=dict)  # {side: {"half": %, "quarter": %}}
    parts: list[dict] = field(default_factory=list)  # {"voxels", "low", "high", "colours"}
    symmetry: dict = field(default_factory=dict)  # {axis: %}


def edge_heights(sil: np.ndarray, edge: str) -> np.ndarray:
    """For each column (or row) of a silhouette, how far its edge on `edge` lies
    from that side of the picture; -1 where the column is empty."""
    s = np.asarray(sil, dtype=bool)
    if edge in ("left", "right"):
        s = s.T
    if edge in ("bottom", "right"):
        s = s[::-1]
    hit = s.any(axis=0)
    return np.where(hit, s.argmax(axis=0), -1)


def slopes(h: np.ndarray) -> list[tuple[int, int, list[int]]]:
    """The slopes of an edge: stretches where the edge only climbs or only falls,
    one cell at a time. Returns (first column, last column, flat run lengths)."""
    out = []
    n = len(h)
    i = 0
    while i < n:
        if h[i] < 0:
            i += 1
            continue
        # gather flat runs while each step to the next run is +1 (or each is -1)
        runs, start, direction = [], i, 0
        j = i
        while j < n and h[j] >= 0:
            k = j
            while k + 1 < n and h[k + 1] == h[j]:
                k += 1
            runs.append(k - j + 1)
            if k + 1 >= n or h[k + 1] < 0:
                j = k + 1
                break
            step = int(h[k + 1] - h[j])
            if abs(step) != 1 or (direction and step != direction):
                j = k + 1
                break
            direction = step
            j = k + 1
        if len(runs) >= 2:
            out.append((start, j - 1, runs))
        i = j if j > i else i + 1
    return out


def jagged(runs: list[int]) -> bool:
    """A slope's rhythm is broken when its inner runs (not the two ends, which
    meet other shapes) differ by two or more: 1, 3, 1, 2 is jagged; 1, 2, 1, 2
    and 2, 2, 3 are even."""
    inner = runs[1:-1]
    return len(runs) >= MIN_SLOPE and len(inner) >= 2 and max(inner) - min(inner) >= 2


def jags(sil: np.ndarray, side: str) -> list[Jag]:
    found = []
    for edge in ("top", "bottom", "left", "right"):
        h = edge_heights(sil, edge)
        for a, b, runs in slopes(h):
            if jagged(runs):
                found.append(Jag(side, edge, runs, (a, b)))
    return found


def thin_features(sil: np.ndarray) -> list[dict]:
    """Lines one cell thin and at least THIN_RUN long, standing out of the
    silhouette: an antenna, a spike, a loose strand. In view rows and columns."""
    s = np.asarray(sil, dtype=bool)
    pad = np.pad(s, 1)
    left, right = pad[1:-1, :-2], pad[1:-1, 2:]
    up, down = pad[:-2, 1:-1], pad[2:, 1:-1]
    vertical = s & ~left & ~right  # one cell wide across
    horizontal = s & ~up & ~down  # one cell high
    out = []
    for mask in (vertical, horizontal):
        for part in region.components(mask[..., None]):
            cells = part[:, :2]
            if len(cells) >= THIN_RUN:
                lo, hi = cells.min(axis=0), cells.max(axis=0)
                out.append({"low": lo.tolist(), "high": hi.tolist(), "cells": int(len(cells))})
    return out


def survival(sil: np.ndarray, factor: int) -> float:
    """How much of a silhouette stays when it is shown `factor` times smaller:
    each block of factor x factor cells shows when at least half of it is
    filled. Percent of the filled cells that the small version still covers."""
    s = np.asarray(sil, dtype=bool)
    if not s.any():
        return 100.0
    h, w = s.shape
    ph, pw = -h % factor, -w % factor
    p = np.pad(s, ((0, ph), (0, pw)))
    blocks = p.reshape(p.shape[0] // factor, factor, p.shape[1] // factor, factor).mean(axis=(1, 3))
    kept = np.repeat(np.repeat(blocks >= 0.5, factor, axis=0), factor, axis=1)[:h, :w]
    return round(100.0 * float((kept & s).sum()) / float(s.sum()), 1)


def symmetry(voxels: np.ndarray) -> dict:
    """Mirror symmetry of the shape about the middle of its voxels, per axis, in percent."""
    v = np.asarray(voxels)
    mask = v != 0
    if not mask.any():
        return {}
    lo, hi = np.argwhere(mask).min(axis=0), np.argwhere(mask).max(axis=0) + 1
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    out = {}
    for a, name in enumerate("xyz"):
        flipped = np.flip(sub, axis=a)
        filled = sub | flipped
        out[name] = round(100.0 * float((sub == flipped)[filled].sum()) / max(int(filled.sum()), 1), 1)
    return out


def parts(voxels: np.ndarray, limit: int = 20) -> list[dict]:
    """The separate parts (face to face), largest first, with the colours each is made of."""
    v = np.asarray(voxels)
    out = []
    for p in region.components(v != 0)[:limit]:
        values, counts = np.unique(v[tuple(p.T)], return_counts=True)
        order = np.argsort(-counts)
        out.append({"voxels": int(len(p)), "low": p.min(axis=0).tolist(), "high": p.max(axis=0).tolist(),
                    "colours": {int(values[i]): int(counts[i]) for i in order}})
    return out


def check(voxels: np.ndarray, with_parts: bool = True) -> Report:
    """Every check over the three flat views and the grid."""
    v = np.asarray(voxels)
    rep = Report()
    if not (v != 0).any():
        return rep
    crop, _lo = voxel_views.cropped(v)
    for side in voxel_views.SIDES:
        colours, _depth = voxel_views.project(crop, side)
        sil = colours != 0
        rep.jags.extend(jags(sil, side))
        for t in thin_features(sil):
            rep.thin.append(dict(t, side=side))
        rep.survives[side] = {"half": survival(sil, 2), "quarter": survival(sil, 4)}
    if with_parts:
        rep.parts = parts(v)
    rep.symmetry = symmetry(v)
    return rep


def summary(rep: Report, palette_names: dict | None = None) -> str:
    """One line for an edit's feedback."""
    if not rep.symmetry:
        return "Craft: the model is empty."
    bits = []
    loose = rep.parts[1:]
    if loose:
        def colour_of(p):
            c = next(iter(p["colours"]))
            return (palette_names or {}).get(c, f"colour {c}")
        bits.append(f"{len(loose)} floating part{'s' if len(loose) != 1 else ''} ("
                    + ", ".join(f"{p['voxels']} vox of {colour_of(p)} at {p['low']}" for p in loose[:3])
                    + (", ..." if len(loose) > 3 else "") + ")")
    else:
        bits.append("one part")
    bits.append("symmetry " + " ".join(f"{a} {p}%" for a, p in rep.symmetry.items()))
    per_side = {s: sum(1 for j in rep.jags if j.side == s) for s in voxel_views.SIDES}
    if any(per_side.values()):
        bits.append("jagged slopes " + ", ".join(f"{s} {n}" for s, n in per_side.items() if n))
    if rep.thin:
        bits.append(f"{len(rep.thin)} thin line{'s' if len(rep.thin) != 1 else ''} one cell wide")
    worst = min(rep.survives.items(), key=lambda kv: kv[1]["quarter"])
    bits.append(f"at 1/4 size the {worst[0]} keeps {worst[1]['quarter']}%")
    return "Craft: " + "; ".join(bits) + "."
