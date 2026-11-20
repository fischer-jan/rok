"""xs shaders: parse, find, and wrap for the GPU.

An xs shader is a GLSL file with one function, `float map(vec3 v)`,
called for every cell centre of the model; it returns the colour index the
cell becomes (0 empty). It may read the model (`voxel(v)`), the selected
colours (`color_sel(k)`) and the palette (`palette(i)`), and the uniforms
`i_volume_size`, `i_color_index`, `i_num_color_sels`, `i_args[8]`, `i_iter`,
`i_mirror`, `i_axis`. Its arguments sit in a comment header:

    // xs_begin
    // arg : { var = 'm_radius' name = 'Radius' value = '3' range = '1 128' step = '1' precision = '0' }
    // xs_end

Each argument is `i_args[k]` in the order of the header, and also a uniform
of its own under its `var` name, so a function parameter of the same name
shadows it. Older files use camelCase names (`iArgs`,
`iVolumeSize`, ...); `fragment_source` aliases them. The GLSL is
translated for Metal on a Mac, and `fragment_source` rewrites the few things
that translation cannot take: struct constructor calls, vector ==/!= in a
condition, and C++ keywords used as names. This module is pure.

A shader that needs certain colours can name them in a Rok-only line (put it
after xs_end):

    // rok_colours : { start = '1'  rgb = 'FFD63C FF8C1E CD3C14 78190A' }

Rok finds these colours by their hex, not by a slot number. The shader
counts its colours from the selected one on (start 0), or after it (start 1:
a marker, such as painted nozzle faces), as the xs format has it: each
run gets a palette of its own with the colours in a row
(rok.core.shader_palette), and the result comes back to your colours. Only
the colours the result uses reach the palette, in one undo step with the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import glsl_pp

MAX_ARGS = 16  # the xs format documents i_args[8]; bundled shaders declare up to 14
LOOP_BUDGET = 1 << 17  # loop steps plus model reads per cell before the guard stops loops
BAND_FAST, BAND_SLOW = 0.03, 0.25  # seconds per band: grow below, shrink above
RUN_SECONDS = 30.0  # a run that takes longer stops

# Why a budget of 131,072: an Apple GPU drops a draw once one cell's shader
# runs for about half a second, silently (it reads back empty), and after a
# second drop it drops every later draw of the process. A loop step that
# reads the model costs some 2.4 microseconds on an M5, so 131,072 units
# (steps plus reads) stay near 0.16 s per cell, four times below that.


def next_band(rows: int, seconds: float, height: int) -> int:
    """The rows the next band draws, after one of `rows` rows took `seconds`:
    doubled while bands are fast, halved when slow. A cheap shader soon draws
    whole slices; a slow one keeps each GPU command short, since one long
    command can hang the GPU (and the screen with it)."""
    if seconds < BAND_FAST:
        return min(height, rows * 2)
    if seconds > BAND_SLOW:
        return max(1, rows // 2)
    return min(height, rows)
MAX_SOURCE = 256 * 1024  # bytes; the largest bundled shader has 7 KB


class Unsupported(ValueError):
    """The shader uses something Rok does not run (see prepare)."""

_ARG = re.compile(r"//\s*arg\s*:\s*\{([^}]*)\}")
_FIELD = re.compile(r"(\w+)\s*=\s*'([^']*)'")
_BLOCK = re.compile(r"//\s*xs_begin(.*?)//\s*xs_end", re.S)


@dataclass(frozen=True)
class ShaderArg:
    name: str
    var: str
    value: float
    low: float
    high: float
    step: float
    precision: int
    tip: str = ""  # what it does and when, from the shader's rok_tip lines


@dataclass(frozen=True)
class ColourNeed:
    """The colours a shader asks for in a `rok_colours` line, hottest first.

    `start` is where they go, counted from the colour the run starts with: 1
    after a marker colour (the marker, then the flame), 0 from that
    colour on (the selected colour is the hottest)."""

    start: int
    rgb: tuple[tuple[int, int, int], ...]
    # What each colour is for, from `names = 'Snow, Snow shade, ...'` in the
    # line (the panel lets you pick them); may be shorter.
    names: tuple[str, ...] = ()
    # How clear each colour is as glass, 0 to 1, from `glass = '0.8 0.6 0'`:
    # 0 (or none listed) is a plain colour; may be shorter.
    glass: tuple[float, ...] = ()

    def name(self, k: int) -> str:
        return self.names[k] if k < len(self.names) and self.names[k] else f"Colour {k + 1}"

    def with_rgb(self, rgb) -> "ColourNeed":
        """The same need with your colours instead, as many as it has; None keeps one."""
        new = list(self.rgb)
        for k, c in enumerate(list(rgb)[: len(new)]):
            if c is not None:
                new[k] = tuple(int(v) for v in c)
        return ColourNeed(self.start, tuple(new), self.names, self.glass)

    def material(self, k: int):
        """Colour k's material: glass that clear, or None for a plain colour."""
        clear = self.glass[k] if k < len(self.glass) else 0.0
        if clear <= 0.0:
            return None
        from .material import DEFAULT

        return DEFAULT.with_type("GLASS").with_weight(min(clear, 1.0))


def parse_hex_list(text: str) -> list[tuple[int, int, int] | None]:
    """'FFAA00, -, 112233' -> [(255, 170, 0), None, (17, 34, 51)]: a dash or a
    word that is no colour keeps the shader's own colour at that place."""
    out: list[tuple[int, int, int] | None] = []
    for word in re.split(r"[,\s]+", str(text or "").strip()):
        if not word:
            continue
        if re.fullmatch(r"#?[0-9A-Fa-f]{6}", word):
            value = int(word.lstrip("#"), 16)
            out.append(((value >> 16) & 255, (value >> 8) & 255, value & 255))
        else:
            out.append(None)
    return out


@dataclass(frozen=True)
class Shader:
    name: str  # the file name without .txt
    path: Path
    args: tuple[ShaderArg, ...]
    author: str
    source: str
    colours: ColourNeed | None = None
    about: str = ""  # one line on what it makes, from a rok_about line
    flags: frozenset = frozenset()  # Rok-only run rules, from a rok_flags line (see parse_flags)
    removes: str = ""  # why a run empties cells, from a rok_removes line


def _float(text: str, default: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_args(source: str) -> tuple[ShaderArg, ...]:
    """The arguments of the header, in order, at most MAX_ARGS."""
    out: list[ShaderArg] = []
    for match in _ARG.finditer(source):
        fields = dict(_FIELD.findall(match.group(1)))
        low, _sep, high = fields.get("range", "0 100").partition(" ")
        value = _float(fields.get("value", "0"), 0.0)
        out.append(
            ShaderArg(
                name=fields.get("name", fields.get("var", f"Arg {len(out) + 1}")),
                var=fields.get("var", ""),
                value=value,
                low=_float(low, 0.0),
                high=_float(high.strip(), max(100.0, value)),
                step=_float(fields.get("step", "1"), 1.0),
                precision=int(_float(fields.get("precision", fields.get("decimal", "0")), 0.0)),
            )
        )
    return with_tips(tuple(out[:MAX_ARGS]) or parse_usage_args(source), source)


# A Rok-only line per argument says what it does and when it has an effect:
#   // rok_tip Density : How many of the places for a tree get one. ...
# Several lines for the same argument join into one text.
_TIP_LINE = re.compile(r"^[ \t]*//[ \t]*rok_tip[ \t]+([^:\n]+?)[ \t]*:[ \t]*(.*?)[ \t]*$", re.M)


def parse_tips(source: str) -> dict[str, str]:
    """The rok_tip lines: argument name (lower case) -> its tooltip."""
    tips: dict[str, str] = {}
    for name, text in _TIP_LINE.findall(source):
        key = name.strip().lower()
        tips[key] = f"{tips[key]} {text}".strip() if key in tips else text
    return tips


def with_tips(args: tuple[ShaderArg, ...], source: str) -> tuple[ShaderArg, ...]:
    """The arguments with the tooltips their rok_tip lines give."""
    tips = parse_tips(source)
    return tuple(replace(a, tip=tips.get(a.name.strip().lower(), a.tip)) for a in args)


# Older shaders have no header; a comment shows the usage line instead:
#   // 'xs scatter amount seed' : e.g. 'xs scatter 0.5 42'
_USAGE = re.compile(
    r"//\s*['\u2019]xs\s+[\w/]+((?:[ \t]+\w+)*)\s*['\u2019]\s*:\s*e\.g\.\s*['\u2019]xs\s+[\w/]+([^'\u2019\n]*)['\u2019]"
)


def parse_usage_args(source: str) -> tuple[ShaderArg, ...]:
    """The arguments of an old shader, named after its usage line, with the
    example's values as defaults. The file gives no range: a whole number
    gets 0 to 255 (or twice itself), a fraction 0 to 1 (or twice itself)."""
    m = _USAGE.search(source)
    if m is None:
        return ()
    names, values = m.group(1).split(), m.group(2).split()
    out = []
    for k, name in enumerate(names[:MAX_ARGS]):
        text = values[k] if k < len(values) else "0"
        value = _float(text, 0.0)
        decimals = len(text.partition(".")[2]) if "." in text else 0
        high = max(255.0 if decimals == 0 else 1.0, abs(value) * 2)
        out.append(ShaderArg(name[:1].upper() + name[1:], "", value, min(0.0, value), high, 1.0 if decimals == 0 else 0.01, decimals))
    return tuple(out)


def parse_author(source: str) -> str:
    block = _BLOCK.search(source)
    if block is None:
        return ""
    # Quoted as the format has it, or bare (`author : @someone`).
    found = re.search(r"author\s*[:=]\s*(?:'([^']*)'|([^\s']+))", block.group(1))
    return (found.group(1) if found.group(1) is not None else found.group(2)) if found else ""


SUBFOLDER_DEPTH = 3  # patterns/bricks is one level; deeper trees are not shader folders


def load(path: Path, name: str | None = None) -> Shader:
    """The shader file; name is its path in the shader folder (patterns/bricks),
    the file name when not given."""
    source = Path(path).read_text(encoding="utf-8", errors="replace")
    return Shader(
        name or Path(path).stem, Path(path), parse_args(source), parse_author(source), source,
        parse_colours(source), parse_about(source), parse_flags(source), parse_removes(source),
    )


_COLOURS_LINE = re.compile(r"^[ \t]*//[ \t]*rok_colours[ \t]*:[ \t]*\{([^}\n]*)\}", re.M)


_ABOUT_LINE = re.compile(r"//\s*rok_about\s*:\s*'([^'\n]*)'")


_REMOVES_LINE = re.compile(r"//\s*rok_removes\s*:\s*'([^'\n]*)'")


_FLAGS_LINE = re.compile(r"//\s*rok_flags\s*:\s*'([^'\n]*)'")
JOINED = "joined"
OWN_GRID = "own_grid"
FLAGS = (JOINED, OWN_GRID)


def parse_flags(source: str) -> frozenset:
    """The Rok-only `// rok_flags : 'joined own_grid'` line: how Rok runs the
    shader. Unknown words are dropped.

    joined: the new voxels must hang together. A new part that touches the rest
    only edge to edge or corner to corner gets one or two voxels that join it
    face to face; a tiny part that touches nothing is left out (trees: no leaf
    floats).
    own_grid: the shader builds one thing that fills its grid (a tree, a
    volcano). With a volume scope it runs on the volume's box as if that were
    the whole grid, so the thing is built inside the volume."""
    found = _FLAGS_LINE.search(source)
    if found is None:
        return frozenset()
    return frozenset(w for w in found.group(1).split() if w in FLAGS)


def parse_about(source: str) -> str:
    """The Rok-only `// rok_about : 'what it makes'` line, or "". Rok shows it
    in the shader gallery and the dialog."""
    found = _ABOUT_LINE.search(source)
    return found.group(1).strip() if found else ""


def parse_removes(source: str) -> str:
    """The Rok-only `// rok_removes : 'why it empties cells'` line, or "". A
    shader that adds detail can also cut into the model; the MCP reply quotes
    this line when a run empties cells, so the removal does not read as damage."""
    found = _REMOVES_LINE.search(source)
    return found.group(1).strip() if found else ""


def parse_colours(source: str) -> ColourNeed | None:
    """The `rok_colours` comment line, or None:

        // rok_colours : { start = '1'  rgb = 'FFD63C FF8C1E CD3C14 78190A' }

    `glass = '0.8 0.6'` makes the first colours glass, that clear (0 to 1).
    A line Rok cannot read counts as absent, so the shader runs as before."""
    found = _COLOURS_LINE.search(source)
    if found is None:
        return None
    body = found.group(1)
    start = re.search(r"start\s*=\s*'([^']*)'", body)
    rgb = re.search(r"rgb\s*=\s*'([^']*)'", body)
    if rgb is None:
        return None
    colours = []
    for word in rgb.group(1).split():
        if not re.fullmatch(r"#?[0-9A-Fa-f]{6}", word):
            return None
        value = int(word.lstrip("#"), 16)
        colours.append(((value >> 16) & 255, (value >> 8) & 255, value & 255))
    offset = int(_float(start.group(1), 1)) if start else 1
    if not colours or offset not in (0, 1):
        return None
    names = re.search(r"names\s*=\s*'([^']*)'", body)
    named = tuple(n.strip() for n in names.group(1).split(",")) if names else ()
    glass = re.search(r"glass\s*=\s*'([^']*)'", body)
    clear = tuple(min(max(_float(w, 0.0), 0.0), 1.0) for w in glass.group(1).split()) if glass else ()
    return ColourNeed(offset, tuple(colours), named, clear)


def find_named(folders) -> dict[str, Path]:
    """Every .txt shader in the folders and their subfolders, by name: the
    path inside the folder without .txt, as the xs format names it
    (`patterns/bricks`). A later folder's file of the same name wins. Sorted: the top level first, then each subfolder."""
    found: dict[str, Path] = {}
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.txt"):
            rel = path.relative_to(folder)
            if len(rel.parts) > SUBFOLDER_DEPTH or any(part.startswith(".") for part in rel.parts):
                continue
            if "map" in path.read_text(encoding="utf-8", errors="replace"):
                found[rel.with_suffix("").as_posix()] = path

    def order(name: str):
        folder, _sep, leaf = name.rpartition("/")
        return (folder.lower(), leaf.lower())

    return {name: found[name] for name in sorted(found, key=order)}


def with_personal(bundled: dict[str, Path], personal: dict[str, Path]) -> dict[str, Path]:
    """Rok's own shaders (by name, nature/rocks) with the personal folder's
    (find_named of each). A personal file named like one of Rok's (rocks, or
    nature/rocks) replaces it under Rok's name, so it is listed once; the
    other personal files follow."""
    out = dict(bundled)
    by_leaf = {name.rpartition("/")[2]: name for name in bundled}
    for name, path in personal.items():
        target = name if name in bundled else by_leaf.get(name)
        out[target or name] = path
    return out


def find_shaders(folders) -> tuple[Path, ...]:
    """The files of find_named, in its order."""
    return tuple(find_named(folders).values())


def uses_colour_selection(shader: Shader) -> bool:
    """True when the shader reads the selected palette colours."""
    text = _strip_comments(shader.source)
    return bool(re.search(r"\bcolor_sel\s*\(|\bi_num_color_sels\b|\biNumColorSels\b", text))


# A selected colour compared with something: `x == color_sel(k)`,
# `color_sel(k) == x`, or isSelected...() helpers.
_ACTS_ON_SELECTION = re.compile(
    r"==\s*color_sel\s*\(|\bcolor_sel\s*\([^()]*\)\s*==|\bisSelected\w*\s*\("
)


def acts_on_selection(shader: Shader) -> bool:
    """True when the shader acts on the voxels of the selected colours (a knife
    cuts them, pyramid builds on them), not only paints with them (noise).

    In the xs format the selection is the colours the user picked, by default just
    the current one. Rok stands in with a run of slots; for a shader that
    acts on them, more than the current colour would touch colours the user
    never picked, so such a shader starts with one."""
    return uses_colour_selection(shader) and bool(_ACTS_ON_SELECTION.search(_strip_comments(shader.source)))


def reads_model(shader: Shader) -> bool:
    """True when the shader reads the model (calls voxel()): such a shader
    mostly changes the voxels of one colour, the current one."""
    return bool(re.search(r"\bvoxel\s*\(", _strip_comments(shader.source)))


_COLOUR_ARG = re.compile(r"\b(target|object|source)\b.*\bcolou?r\b|^original$", re.I)


def is_colour_arg(arg: ShaderArg) -> bool:
    """True for an argument that names the colour to work on ('Target Color',
    'Object color', replace's 'Original'): it starts at the run's colour."""
    return bool(_COLOUR_ARG.search(arg.name.strip())) and arg.low >= 0 and arg.high >= 1


def is_brush(shader: Shader) -> bool:
    """True for a paint shader (Rok's patterns/ group, or a brush/ set):
    Rok lets it recolour filled cells only, and never add or erase one."""
    return shader.name.startswith(("brush_", "brush/", "patterns/")) or bool(re.search(r"\bxs\s+brush/", shader.source))


def label(name: str) -> str:
    """"city_blocks3D" -> "City Blocks3D", "patterns/bricks" -> "Patterns / Bricks":
    the menu text."""
    return " / ".join(
        " ".join(part[:1].upper() + part[1:] for part in re.split(r"[_\s]+", piece) if part)
        for piece in name.split("/")
    )


def atlas_layout(shape) -> tuple[int, int]:
    """(tiles per row, rows) for the z slices of a model: about square."""
    sx, sy, sz = (int(v) for v in shape)
    cols = max(1, int(np.ceil(np.sqrt(sz * sy / max(sx, 1)))))
    cols = min(cols, sz)
    return cols, -(-sz // cols)


def atlas(voxels: np.ndarray, cols: int) -> np.ndarray:
    """The z slices as tiles: slice z at column z % cols, row z // cols; a
    tile's row is y, its column x."""
    sx, sy, sz = voxels.shape
    rows = -(-sz // cols)
    out = np.zeros((rows * sy, cols * sx), dtype=np.uint8)
    for z in range(sz):
        r, c = divmod(z, cols)
        out[r * sy : (r + 1) * sy, c * sx : (c + 1) * sx] = voxels[:, :, z].T
    return out



# The helpers the xs format provides, over three textures: the volume as a 2D
# atlas of z slices in rows of u_tiles tiles (slice z at column z % u_tiles,
# row z / u_tiles), the selected colours and the palette, all as RGBA with the
# index in red / 255.
PRELUDE = """
#define iArgs i_args
#define iVolumeSize i_volume_size
#define iColorIndex i_color_index
#define iNumColorSels i_num_color_sels
#define iIter i_iter
#define iFrame i_iter
#define iMirror i_mirror
#define iAxis i_axis
int xs_steps;
bool xs_tick() { xs_steps++; return xs_steps <= u_loop_limit; }
float voxel(vec3 v) {
  xs_steps++;  // a model read costs about as much as a loop step: both count
  ivec3 c = ivec3(floor(v));
  ivec3 s = ivec3(i_volume_size);
  if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= s.x || c.y >= s.y || c.z >= s.z) return 0.0;
  ivec2 tile = ivec2(c.z % u_tiles, c.z / u_tiles);
  return floor(texelFetch(u_vol, tile * s.xy + c.xy, 0).r * 255.0 + 0.5);
}
float color_sel(float k) {
  int i = clamp(int(k), 0, max(i_num_color_sels - 1, 0));
  return floor(texelFetch(u_sel, ivec2(i, 0), 0).r * 255.0 + 0.5);
}
vec4 palette(float index) { return texelFetch(u_pal, ivec2(int(index), 0), 0); }
"""

MAIN = """
void main() {
  xs_steps = 0;
  xs_init_globals();
  vec3 v = vec3(gl_FragCoord.xy, u_z + 0.5);
  float idx = map(v);
  float stopped = xs_steps > u_loop_limit ? 1.0 : 0.0;
  // Alpha 1 marks the cell as drawn: xs_ops.run clears to alpha 0 first and
  // takes a cell left at 0 as a draw the GPU dropped.
  fragColor = vec4(clamp(floor(idx + 0.5), 0.0, 255.0) / 255.0, stopped, 0.0, 1.0);
}
"""


# Words that are names in GLSL but reserved in the C++ that Metal is written in.
_CPP_WORDS = (
    "try", "catch", "throw", "class", "template", "typename", "namespace", "new", "delete",
    "this", "public", "private", "protected", "operator", "friend", "virtual", "auto",
    "register", "union", "enum", "sizeof", "explicit", "mutable", "typedef", "using",
    "device", "constant", "thread", "threadgroup", "kernel", "vertex", "fragment",
)
_VEC_TYPES = "vec2|vec3|vec4|ivec2|ivec3|ivec4|bvec2|bvec3|bvec4"


def arg_names(shader: Shader) -> tuple[str, ...]:
    """The `var` names the GPU gets as uniforms of their own."""
    seen, out = set(), []
    for arg in shader.args:
        if arg.var and re.fullmatch(r"[A-Za-z_]\w*", arg.var) and arg.var not in seen:
            seen.add(arg.var)
            out.append(arg.var)
    return tuple(out)


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def _rename_cpp_words(source: str) -> str:
    for word in _CPP_WORDS:
        source = re.sub(rf"\b{word}\b", f"xs_{word}", source)
    return source


def _struct_constructors(source: str) -> str:
    """`Cell(a, b, c)` -> `Cell_make(a, b, c)`, with Cell_make written after the struct."""
    out = source
    for match in list(re.finditer(r"struct\s+(\w+)\s*\{([^}]*)\}\s*;", source)):
        name, body = match.group(1), match.group(2)
        fields = []
        for decl in body.split(";"):
            decl = decl.strip()
            m = re.fullmatch(r"(\w+)\s+(.+)", decl)
            if not m:
                continue
            for field in m.group(2).split(","):
                fields.append((m.group(1), field.strip()))
        if not fields:
            continue
        params = ", ".join(f"{t} p_{f}" for t, f in fields)
        assigns = " ".join(f"r.{f} = p_{f};" for _t, f in fields)
        maker = f"\n{name} {name}_make({params}) {{ {name} r; {assigns} return r; }}\n"
        struct_text = match.group(0)
        # Calls only: not the struct line, not "Cell x", not a function named Cell_...
        out = re.sub(rf"(?<![\w.]){name}\s*\(", f"{name}_make(", out)
        out = out.replace(struct_text, struct_text + maker, 1)
    return out


def _vector_compares(source: str) -> str:
    """`a != b` / `a == b` on vector variables -> any(notEqual(a, b)) / all(equal(a, b))."""
    names = set(re.findall(rf"\b(?:{_VEC_TYPES})\s+(\w+)\b", source))
    if not names:
        return source

    def repl(m):
        a, op, b = m.group(1), m.group(2), m.group(3)
        if a in names and b in names:
            return f"any(notEqual({a}, {b}))" if op == "!=" else f"all(equal({a}, {b}))"
        return m.group(0)

    return re.sub(r"\b(\w+)\s*(==|!=)\s*(\w+)\b(?!\s*\()", repl, source)


def _split_top(text: str, sep: str) -> list[str]:
    """Split on sep where no bracket is open."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _hoist_globals(source: str) -> str:
    """`float width = m_width;` at file level -> `float width;` there, and the
    assignment in xs_init_globals(), which main() calls before map().

    GLSL lets a global start from a uniform; the Metal translation reads
    such a global as 0 or garbage, and a loop bound by it can hang the GPU.
    Constants (`const float PI = ...`) stay as they are."""
    out, inits = [], []
    depth, seg_start, i = 0, 0, 0
    while i < len(source):
        ch = source[i]
        if ch == "#" and depth == 0 and source[seg_start:i].strip() == "":
            end = source.find("\n", i)
            end = len(source) if end < 0 else end + 1
            out.append(source[seg_start:end])
            seg_start = i = end
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(source[seg_start : i + 1])
                seg_start = i + 1
        elif ch == ";" and depth == 0:
            out.append(_hoist_statement(source[seg_start:i], inits) + ";")
            seg_start = i + 1
        i += 1
    out.append(source[seg_start:])
    body = "".join(inits)
    return "".join(out) + f"\nvoid xs_init_globals() {{{body}\n}}\n"


_DECL = re.compile(r"(\s*)((?:(?:highp|mediump|lowp)\s+)?[A-Za-z_]\w*)\s+(.*)", re.S)


def _hoist_statement(stmt: str, inits: list[str]) -> str:
    """One file-level statement, with its initialisers moved into inits."""
    m = _DECL.fullmatch(stmt)
    if m is None or m.group(2).split()[-1] in ("const", "uniform", "struct", "return", "precision"):
        return stmt
    if "=" not in stmt or re.match(r"\s*(const|uniform|struct)\b", stmt):
        return stmt
    decls = []
    for part in _split_top(m.group(3), ","):
        name, eq, value = part.partition("=")
        if "(" in name:  # a prototype or a call, not a declarator
            return stmt
        decls.append(name.strip())
        if not eq:
            continue
        target = re.sub(r"\[.*\]", "", name).strip()
        inits.append(f"\n  {target} = {value.strip()};")
    return f"{m.group(1)}{m.group(2)} {', '.join(decls)}"


def _scalar_splats(source: str) -> str:
    """`float c = d * vec3(x);` -> `float c = d * float(x);`: a one-argument
    vector in a float initialiser is x in every part, and OpenGL drivers take
    its first; Metal refuses to put a vector into a float (gear.txt does this)."""

    def fix(m):
        init = m.group(2)
        out, pos = [], 0
        for call in re.finditer(r"\b[ib]?vec[234]\s*\(", init):
            if call.start() < pos:
                continue
            close = _matching(init, call.end() - 1)
            if close < 0 or len(_split_top(init[call.end() : close], ",")) != 1:
                continue
            if init[close + 1 :].lstrip().startswith("."):  # vec2(2.0).x is a float already
                continue
            out.append(init[pos : call.start()] + "float(")
            pos = call.end()
        out.append(init[pos:])
        return m.group(1) + "".join(out) + ";"

    return re.sub(r"(\bfloat\s+\w+\s*=)([^;{}]*);", fix, source)


def _matching(text: str, open_at: int) -> int:
    """The index of the bracket that closes the one at open_at, or -1."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _guard_loops(source: str) -> str:
    """Every loop also calls xs_tick(), which fails after u_loop_limit steps in
    one cell: a runaway loop ends instead of hanging the GPU (and the screen
    with it). `for (a; b; c)` -> `for (a; (b) && xs_tick(); c)`,
    `while (b)` -> `while ((b) && xs_tick())`; a do-while gets it in its while."""
    out, pos = [], 0
    for m in re.finditer(r"\b(for|while)\s*\(", source):
        if m.start() < pos:
            continue
        open_at = m.end() - 1
        close = _matching(source, open_at)
        if close < 0:
            return source
        inner = source[open_at + 1 : close]
        if m.group(1) == "for":
            parts = _split_top(inner, ";")
            if len(parts) != 3:
                return source
            cond = parts[1].strip()
            parts[1] = f" ({cond}) && xs_tick()" if cond else " xs_tick()"
            inner = ";".join(parts)
        else:
            inner = f"({inner.strip()}) && xs_tick()"
        out.append(source[pos:open_at + 1] + _guard_loops(inner) + ")")
        pos = close + 1
    out.append(source[pos:])
    return "".join(out)


def prepare(source: str) -> str:
    """The shader's own text, sanitised for the GPU.

    A shader runs as a fragment shader: it reads only the model, the colours
    and the palette Rok binds, and writes one index per cell, so the harm it
    can do is a hang or a driver crash. Against that: a size limit, the
    macros expanded here (rok.core.glsl_pp) so no macro hides a loop, the
    xs_ names reserved for Rok (a shader cannot replace the guard), and
    every loop guarded. The rest rewrites what the Metal translation
    cannot take. Raises Unsupported."""
    if len(source) > MAX_SOURCE:
        raise Unsupported(f"it is larger than {MAX_SOURCE // 1024} KB")
    source = _strip_comments(source)
    try:
        source = glsl_pp.preprocess(source)
    except glsl_pp.PreprocessError as exc:
        raise Unsupported(str(exc)) from exc
    source = re.sub(r"\bxs_", "usr_xs_", source)
    source = _rename_cpp_words(source)
    source = _struct_constructors(source)
    source = _vector_compares(source)
    source = _scalar_splats(source)
    source = _hoist_globals(source)
    return _guard_loops(source)


def fragment_source(shader: Shader) -> str:
    """The shader file, prepared for Metal, between the helpers and a main that
    sets the file's globals and writes map() of this pixel's cell. The
    arguments are uniforms (see arg_names)."""
    return PRELUDE + prepare(shader.source) + MAIN
