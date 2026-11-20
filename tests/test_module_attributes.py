"""Every `module.name` that rok, the scripts and the tests read must exist.

A name that moved or went away leaves no trace for ruff's undefined-name check, and
the headless smoke test only meets it where a line runs. Draw code runs in a real
window only, so such a name would first fail there. This reads the code with `ast`
and needs no app."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    p
    for folder in ("rok", "scripts", "tests")
    for p in (ROOT / folder).rglob("*.py")
    if "__pycache__" not in p.parts
)


def _module_file(dotted: str) -> Path | None:
    """The file of a rok package module, or None for anything else."""
    parts = dotted.split(".")
    if parts[0] != "rok":
        return None
    base = ROOT.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _dotted_of(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _bound_names(target) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


@lru_cache(maxsize=None)
def defined_names(path: Path) -> frozenset[str]:
    """Everything a module binds at its top level, also inside if and try blocks."""
    names: set[str] = set()

    def visit(body) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    names.update(_bound_names(t))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                names.update(_bound_names(node.target))
            elif isinstance(node, ast.Import):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.update(a.asname or a.name for a in node.names)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                names.update(_bound_names(node.target))
                visit(node.body + node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        names.update(_bound_names(item.optional_vars))
                visit(node.body)
            elif isinstance(node, (ast.If, ast.While)):
                visit(node.body + node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body + node.orelse + node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)

    visit(ast.parse(path.read_text()).body)
    return frozenset(names)


def _aliases(tree: ast.AST, path: Path) -> dict[str, set[Path]]:
    """Name -> the rok module files it may stand for, from every import in the file."""
    package = _dotted_of(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    out: dict[str, set[Path]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            for a in node.names:
                target = _module_file(f"{module}.{a.name}")
                if target is not None:
                    out.setdefault(a.asname or a.name, set()).add(target)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    target = _module_file(a.name)
                    if target is not None:
                        out.setdefault(a.asname, set()).add(target)
    return out


def _local_names(func) -> set[str]:
    """The names a function binds itself; a module alias among them is shadowed there."""
    names: set[str] = set()
    args = func.args
    for a in args.posonlyargs + args.args + args.kwonlyargs:
        names.add(a.arg)
    for a in (args.vararg, args.kwarg):
        if a is not None:
            names.add(a.arg)
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not func:
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def missing_attributes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    aliases = _aliases(tree, path)
    if not aliases:
        return []
    found: list[str] = []

    def scan(node, shadowed: frozenset[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            shadowed = shadowed | _local_names(node)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and node.value.id not in shadowed
        ):
            targets = aliases[node.value.id]
            if not any(node.attr in defined_names(t) or _module_file(f"{_dotted_of(t)}.{node.attr}") for t in targets):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.value.id}.{node.attr}")
        for child in ast.iter_child_nodes(node):
            scan(child, shadowed)

    scan(tree, frozenset())
    return found


def test_every_module_attribute_that_is_read_exists():
    problems = [line for path in FILES for line in missing_attributes(path)]
    assert not problems, "\n".join(problems)


def test_the_check_finds_a_name_that_is_gone(tmp_path):
    """The check itself: a module and a reader of a name it does not have."""
    source = ROOT / "rok" / "core" / "geom2d.py"
    reader = tmp_path / "reader.py"
    reader.write_text("from rok.core import geom2d\n\ngeom2d.in_rect\ngeom2d.no_such_name\n")
    tree = ast.parse(reader.read_text())
    targets = _aliases(tree, ROOT / "tests" / "reader.py")["geom2d"]
    assert targets == {source}
    assert "in_rect" in defined_names(source) and "no_such_name" not in defined_names(source)
