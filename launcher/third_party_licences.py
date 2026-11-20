"""Write the licence texts of everything a packaged Rok ships but did not write.

The bundle holds three kinds of third-party code:

- the Rust crates linked into rok_native (from `cargo metadata`, the texts
  from each crate's folder in the cargo registry),
- CPython and the libraries the python-build-standalone build links in
  (the texts in launcher/licences/python, taken from that build's archive),
- the Python packages in the bundle's site-packages (numpy; the texts from
  each package's dist-info).

A crate without its own licence file gets the standard text of its licence.
Crates with the same texts share one copy.

Usage: third_party_licences.py SITE_PACKAGES OUT_FILE
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON_TEXTS = REPO / "launcher" / "licences" / "python"
TARGET = "aarch64-apple-darwin"
LICENCE_PREFIXES = ("LICEN", "COPYING", "NOTICE", "UNLICENSE")
RULE = "=" * 78

MIT = """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def licence_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.upper().startswith(LICENCE_PREFIXES))


def linked_crates() -> list[dict]:
    """The crates rok_native links on macOS: normal and build dependencies, not dev ones."""
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--filter-platform", TARGET,
         "--manifest-path", str(REPO / "native" / "Cargo.toml")],
        check=True, capture_output=True, text=True).stdout)
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen, todo = set(), [root]
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        for dep in nodes[node]["deps"]:
            if any(k["kind"] in (None, "build") for k in dep["dep_kinds"]):
                todo.append(dep["pkg"])
    seen.discard(root)
    return sorted((packages[i] for i in seen), key=lambda p: (p["name"], p["version"]))


def apache_text(crates: list[dict]) -> str:
    """The Apache 2.0 text, from the first crate that ships it as its own file."""
    for p in crates:
        for f in licence_files(Path(p["manifest_path"]).parent):
            text = f.read_text(errors="replace")
            if "Apache License" in text and "Version 2.0" in text and "TERMS AND CONDITIONS" in text:
                return text
    raise SystemExit("no crate ships the Apache 2.0 text")


def crate_texts(p: dict, apache: str) -> str:
    files = licence_files(Path(p["manifest_path"]).parent)
    if files:
        return "\n\n".join(f"--- {f.name}\n\n{f.read_text(errors='replace').strip()}" for f in files)
    # No file in the crate: the standard text of the licence it names.
    expr = p.get("license") or ""
    authors = ", ".join(p.get("authors") or []) or f"the {p['name']} authors"
    if "MIT" in expr:
        return f"--- MIT (the crate names: {expr})\n\nCopyright (c) {authors}\n\n{MIT.strip()}"
    if "Apache-2.0" in expr:
        return f"--- Apache-2.0 (the crate names: {expr})\n\n{apache.strip()}"
    raise SystemExit(f"{p['name']} {p['version']}: no licence file and no known licence ({expr!r})")


def python_packages(site: Path) -> list[tuple[str, str]]:
    out = []
    for info in sorted(site.glob("*.dist-info")):
        if info.name.startswith("rok_native-"):
            continue  # Rok's own
        files = sorted(p for p in info.rglob("*") if p.is_file() and p.name.upper().startswith(LICENCE_PREFIXES))
        if not files:
            raise SystemExit(f"{info.name}: no licence file")
        name = info.name.removesuffix(".dist-info").replace("-", " ", 1)
        out.append((name, "\n\n".join(f"--- {f.name}\n\n{f.read_text(errors='replace').strip()}" for f in files)))
    return out


def main(site: Path, out: Path) -> None:
    parts = [
        "Third-party software in Rok\n\n"
        "Rok includes the software listed below. Each part is under its own\n"
        "licence, given in full. Rok's own licence is in LICENSE.\n",
    ]

    parts.append(f"{RULE}\nPYTHON AND THE LIBRARIES IT LINKS\n{RULE}\n")
    for f in sorted(PYTHON_TEXTS.glob("LICENSE.*.txt")):
        name = f.name.removeprefix("LICENSE.").removesuffix(".txt")
        parts.append(f"{name}\n\n{f.read_text(errors='replace').strip()}\n")

    parts.append(f"{RULE}\nPYTHON PACKAGES\n{RULE}\n")
    for name, text in python_packages(site):
        parts.append(f"{name}\n\n{text}\n")

    parts.append(f"{RULE}\nRUST CRATES\n{RULE}\n")
    crates = linked_crates()
    apache = apache_text(crates)
    groups: dict[str, list[str]] = {}
    for p in crates:
        groups.setdefault(crate_texts(p, apache), []).append(f"{p['name']} {p['version']} ({p.get('license') or 'see text'})")
    for text, names in groups.items():
        parts.append("\n".join(names) + "\n\n" + text + "\n")

    out.write_text(("\n" + "-" * 78 + "\n\n").join(parts))
    print(f"{out}: {len(crates)} crates, {len(groups)} texts")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
