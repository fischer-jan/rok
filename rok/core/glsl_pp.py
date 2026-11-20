"""A small C preprocessor for xs shaders: expand the macros in Rok.

Rok expands a shader's macros itself instead of leaving them to the GPU
compiler, so that the loop guard in rok.core.xs sees every loop, including
one a macro spells (`#define REPEAT for`). Supported: object and function
macros (with `##` and `__VA_ARGS__`), #undef, and #if / #ifdef / #ifndef /
#elif / #else / #endif, whose conditions are evaluated without eval().
#version, #extension, #pragma and #line are dropped: Rok sets up the
shader itself. #include has nothing to include (the xs format has no include
path) and #error is the shader's own refusal: both raise
PreprocessError, as does a macro set that expands without end.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

MAX_EXPANSIONS = 100_000
MAX_OUTPUT = 4 * 1024 * 1024  # characters
PREDEFINED = {"__VERSION__": "330"}  # xs shaders are written for OpenGL 3.3
_DROPPED = {"version", "extension", "pragma", "line"}
_IDENT = re.compile(r"[A-Za-z_]\w*")


class PreprocessError(ValueError):
    """The shader's preprocessor lines cannot be run."""


@dataclass(frozen=True)
class Macro:
    params: tuple[str, ...] | None  # None for an object macro
    body: str
    variadic: bool = False


class _Budget:
    def __init__(self) -> None:
        self.left = MAX_EXPANSIONS

    def spend(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise PreprocessError("its macros expand without end")


def _closing(text: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_args(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def expand(text: str, macros: dict[str, Macro], disabled: frozenset = frozenset(), budget: _Budget | None = None) -> str:
    """text with every macro replaced, rescanned; a macro is not expanded
    inside its own replacement (the C rule that stops self-reference)."""
    budget = budget or _Budget()
    out, pos = [], 0
    for m in _IDENT.finditer(text):
        if m.start() < pos:
            continue
        name = m.group(0)
        macro = macros.get(name)
        if macro is None or name in disabled:
            continue
        if macro.params is None:
            budget.spend()
            out.append(text[pos : m.start()])
            out.append(expand(macro.body, macros, disabled | {name}, budget))
            pos = m.end()
        else:
            rest = text[m.end() :]
            gap = len(rest) - len(rest.lstrip())
            open_at = m.end() + gap
            if open_at >= len(text) or text[open_at] != "(":
                continue  # a function macro's name without a call stays a name
            close = _closing(text, open_at)
            if close < 0:
                raise PreprocessError(f"the call of macro {name} is not closed")
            budget.spend()
            args = _split_args(text[open_at + 1 : close])
            if args == [""] and not macro.params:
                args = []
            body = _substitute(name, macro, args, macros, disabled, budget)
            out.append(text[pos : m.start()])
            out.append(expand(body, macros, disabled | {name}, budget))
            pos = close + 1
        if sum(len(o) for o in out) > MAX_OUTPUT:
            raise PreprocessError("its macros expand to more text than Rok runs")
    out.append(text[pos:])
    return "".join(out)


def _substitute(name: str, macro: Macro, args: list[str], macros, disabled, budget) -> str:
    params = list(macro.params or ())
    if macro.variadic:
        if len(args) < len(params):
            raise PreprocessError(f"macro {name} gets too few arguments")
        values = args[: len(params)] + [", ".join(args[len(params) :])]
        params.append("__VA_ARGS__")
    else:
        if len(args) != len(params):
            raise PreprocessError(f"macro {name} takes {len(params)} arguments, not {len(args)}")
        values = args
    raw = dict(zip(params, values))
    done = {p: expand(v, macros, disabled, budget) for p, v in raw.items()}
    body = macro.body
    # `a ## b` pastes the raw arguments; everything else gets them expanded.
    body = re.sub(
        r"(\w+)\s*##\s*(\w+)",
        lambda m: raw.get(m.group(1), m.group(1)) + raw.get(m.group(2), m.group(2)),
        body,
    )
    body = re.sub(r"#\s*(\w+)", lambda m: f'"{raw[m.group(1)]}"' if m.group(1) in raw else m.group(0), body)
    if not params:
        return body
    return re.sub(r"\b(" + "|".join(map(re.escape, params)) + r")\b", lambda m: done[m.group(1)], body)


# --- #if conditions ---------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
}
_CMPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _value(node) -> int:
    if isinstance(node, ast.Expression):
        return _value(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return int(node.value)
    if isinstance(node, ast.UnaryOp):
        v = _value(node.operand)
        if isinstance(node.op, ast.Not):
            return int(not v)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Invert):
            return ~v
    if isinstance(node, ast.BoolOp):
        values = [_value(v) for v in node.values]
        return int(all(values)) if isinstance(node.op, ast.And) else int(any(values))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and right == 0:
            raise PreprocessError("an #if divides by zero")
        if isinstance(node.op, (ast.LShift, ast.RShift)) and not 0 <= right < 64:
            raise PreprocessError("an #if shifts too far")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.Compare):
        left = _value(node.left)
        for op, right_node in zip(node.ops, node.comparators):
            right = _value(right_node)
            if type(op) not in _CMPS or not _CMPS[type(op)](left, right):
                return 0
            left = right
        return 1
    if isinstance(node, ast.IfExp):
        return _value(node.body) if _value(node.test) else _value(node.orelse)
    raise PreprocessError("an #if condition uses something Rok does not evaluate")


def evaluate(condition: str, macros: dict[str, Macro]) -> bool:
    """A #if / #elif condition, as C does it: defined(), macros, then names
    that are left count as 0. Parsed with ast and evaluated by hand."""
    text = re.sub(r"\bdefined\s*\(\s*(\w+)\s*\)|\bdefined\s+(\w+)", lambda m: "1" if (m.group(1) or m.group(2)) in macros else "0", condition)
    text = expand(text, macros)
    text = re.sub(r"\bdefined\s*\(\s*(\w+)\s*\)|\bdefined\s+(\w+)", lambda m: "1" if (m.group(1) or m.group(2)) in macros else "0", text)
    text = _IDENT.sub("0", text)
    text = re.sub(r"(\d+)[uUlL]+\b", r"\1", text)
    text = text.replace("&&", " and ").replace("||", " or ").replace("/", "//")
    text = re.sub(r"!(?!=)", " not ", text)
    text = re.sub(r"(.+?)\?(.+?):(.+)", r"(\2) if (\1) else (\3)", text)
    try:
        tree = ast.parse(text.strip() or "0", mode="eval")
    except SyntaxError as exc:
        raise PreprocessError(f"it has an #if Rok cannot read: '{condition.strip()[:40]}'") from exc
    return bool(_value(tree))


# --- the directives ---------------------------------------------------------------------

_DIRECTIVE = re.compile(r"\s*#\s*(\w*)(.*)")
_DEFINE = re.compile(r"([A-Za-z_]\w*)(\(([^)]*)\))?\s*(.*)", re.S)


def preprocess(source: str) -> str:
    """source (comments already removed) with its directives run and its
    macros expanded: plain GLSL without a single # line. Raises PreprocessError."""
    source = re.sub(r"\\\r?\n", " ", source)  # continued lines
    macros: dict[str, Macro] = {name: Macro(None, value) for name, value in PREDEFINED.items()}
    budget = _Budget()
    out: list[str] = []
    pending: list[str] = []
    # One frame per open #if: (this branch runs, a branch already ran, the frame's parent runs).
    stack: list[list[bool]] = []

    def active() -> bool:
        return all(frame[0] for frame in stack)

    def flush() -> None:
        if pending:
            out.append(expand("\n".join(pending), macros, budget=budget))
            pending.clear()

    for line in source.splitlines():
        m = _DIRECTIVE.fullmatch(line)
        if m is None:
            if active():
                pending.append(line)
            continue
        word, rest = m.group(1), m.group(2).strip()
        if word in ("if", "ifdef", "ifndef"):
            if not active():
                stack.append([False, True])  # skip the whole block
                continue
            if word == "if":
                taken = evaluate(rest, macros)
            else:
                taken = (rest.split()[0] if rest else "") in macros
                taken = taken if word == "ifdef" else not taken
            stack.append([taken, taken])
        elif word in ("elif", "else"):
            if not stack:
                raise PreprocessError(f"#{word} without #if")
            frame = stack.pop()
            if frame[1]:
                stack.append([False, True])
            else:
                parent_runs = active()
                taken = parent_runs and (word == "else" or evaluate(rest, macros))
                stack.append([taken, taken])
        elif word == "endif":
            if not stack:
                raise PreprocessError("#endif without #if")
            stack.pop()
        elif not active():
            continue
        elif word == "define":
            flush()
            d = _DEFINE.fullmatch(rest)
            if d is None:
                raise PreprocessError(f"it has a #define Rok cannot read: '#{word} {rest[:30]}'")
            name, has_params, params, body = d.group(1), d.group(2), d.group(3), d.group(4).strip()
            if has_params and not rest[len(name) :].startswith("("):
                body = f"{has_params} {body}".strip()  # `#define X (1)` is an object macro
                has_params = None
            if has_params:
                names = [p.strip() for p in params.split(",") if p.strip()]
                variadic = bool(names) and names[-1] == "..."
                macros[name] = Macro(tuple(names[:-1] if variadic else names), body, variadic)
            else:
                macros[name] = Macro(None, body)
        elif word == "undef":
            flush()
            macros.pop(rest.split()[0] if rest else "", None)
        elif word in _DROPPED or word == "":
            continue
        elif word == "include":
            raise PreprocessError("it uses #include, and a shader has nothing to include")
        elif word == "error":
            raise PreprocessError(f"it stops itself: #error {rest[:40]}")
        else:
            raise PreprocessError(f"it uses #{word}, which Rok does not run")
    if stack:
        raise PreprocessError("an #if is not closed with #endif")
    flush()
    return "\n".join(out) + "\n"
