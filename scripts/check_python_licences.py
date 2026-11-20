"""Every Python package in uv.lock has a permissive licence.

uv.lock names the packages. scripts/python_licences.toml gives the licence of
each one as an SPDX expression, and the licences Rok allows. The check fails
when a package of uv.lock is missing from that file, or when its expression
cannot be met with the allowed licences ("A OR B" needs one of them, "A AND B"
needs both).

`--verify` also asks PyPI for the licence of every locked version and fails when
the file says something else, so a line typed wrongly, or a package that changed
its licence in a new version, does not pass. scripts/check.sh verifies on every
push that changes uv.lock or the licence file. Where PyPI gives no licence the
script can read, a person reads the licence and adds "name==version" to
`checked_by_hand`; a new version asks again.

    python scripts/check_python_licences.py            # check against the file
    python scripts/check_python_licences.py --verify   # and compare the file with PyPI
    python scripts/check_python_licences.py --lookup   # show what PyPI says about the missing packages
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "uv.lock"
LICENCES = ROOT / "scripts" / "python_licences.toml"


def permitted(expression: str, allow: set[str]) -> bool:
    """True when the SPDX expression can be met with the allowed licences."""
    tokens = re.findall(r"\(|\)|[^\s()]+", expression)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def take() -> str:
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def one() -> bool:
        if peek() is None:
            raise ValueError("the expression ends too early")
        if peek() == "(":
            take()
            inside = either()
            if peek() != ")":
                raise ValueError("a bracket is not closed")
            take()
            return inside
        name = take()
        if peek() == "WITH":
            take()
            name = f"{name} WITH {take()}"
        return name in allow

    def both() -> bool:
        ok = one()
        while peek() == "AND":
            take()
            ok = one() and ok
        return ok

    def either() -> bool:
        ok = both()
        while peek() == "OR":
            take()
            ok = both() or ok
        return ok

    if not tokens:
        return False
    ok = either()
    if pos != len(tokens):
        raise ValueError(f"cannot read the expression at {tokens[pos]!r}")
    return ok


def locked_packages(lock: Path) -> dict[str, str]:
    """The packages that uv.lock takes from a package index: name -> version."""
    found = {}
    for package in tomllib.loads(lock.read_text())["package"]:
        if "registry" in package.get("source", {}):
            found[package["name"]] = package["version"]
    return found


def problems(lock: Path, licences: Path) -> list[str]:
    table = tomllib.loads(licences.read_text())
    allow, known = set(table["allow"]), table["packages"]
    found = []
    for name in sorted(locked_packages(lock)):
        if name not in known:
            found.append(f"{name}: no licence in {licences.name}")
            continue
        try:
            ok = permitted(known[name], allow)
        except ValueError as error:
            found.append(f"{name}: {known[name]!r}: {error}")
            continue
        if not ok:
            found.append(f"{name}: {known[name]!r} is not met by the allowed licences")
    return found


# What the older PyPI fields say, and the SPDX names that agree with it. "BSD
# License" does not say which BSD licence, so both common ones agree.
OLD_NAMES = {
    "mit": {"MIT"},
    "mit license": {"MIT"},
    "mit no attribution license (mit-0)": {"MIT-0"},
    "bsd license": {"BSD-2-Clause", "BSD-3-Clause"},
    "apache software license": {"Apache-2.0"},
    "apache-2.0": {"Apache-2.0"},
    "isc license (iscl)": {"ISC"},
    "psf": {"PSF-2.0"},
    "python software foundation license": {"PSF-2.0"},
    "zlib/libpng license": {"Zlib"},
}


def pypi_info(name: str, version: str) -> dict:
    request = urllib.request.Request(f"https://pypi.org/pypi/{name}/{version}/json",
                                     headers={"User-Agent": "licence-check"})
    with urllib.request.urlopen(request, timeout=20) as reply:
        return json.load(reply)["info"]


def pypi_says(info: dict) -> tuple[str | None, set[str]]:
    """What a PyPI record says: its SPDX expression, or else the SPDX names
    that agree with its older licence field and classifiers."""
    expression = (info.get("license_expression") or "").strip()
    if expression:
        return " ".join(expression.split()), set()
    words = [c.split(" :: ")[-1] for c in info.get("classifiers", []) if c.startswith("License ::")]
    first_line = (info.get("license") or "").strip().splitlines()[:1]
    names: set[str] = set()
    for word in words + first_line:
        names |= OLD_NAMES.get(word.strip().lower(), set())
    return None, names


def disagreement(listed: str, info: dict) -> str | None:
    """None when the listed expression agrees with the PyPI record, or the reason."""
    expression, names = pypi_says(info)
    listed = " ".join(listed.split())
    if expression is not None:
        return None if expression == listed else f"PyPI says {expression!r}"
    if not names:
        return "PyPI gives no licence the script can read; a person must read it (checked_by_hand)"
    return None if listed in names else f"PyPI says one of {sorted(names)}"


def verify(lock: Path, licences: Path, fetch=pypi_info) -> list[str]:
    """The lines of the licence file that PyPI does not confirm."""
    table = tomllib.loads(licences.read_text())
    known, by_hand = table["packages"], set(table.get("checked_by_hand", []))
    todo = [(n, v) for n, v in sorted(locked_packages(lock).items()) if n in known and f"{n}=={v}" not in by_hand]
    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for (name, version), info in zip(todo, pool.map(lambda nv: fetch(*nv), todo)):
            reason = disagreement(known[name], info)
            if reason:
                found.append(f"{name} {version}: the file says {known[name]!r}, {reason}")
    return found


def lookup(lock: Path, licences: Path) -> None:
    known = tomllib.loads(licences.read_text())["packages"]
    for name, version in sorted(locked_packages(lock).items()):
        if name in known:
            continue
        info = pypi_info(name, version)
        first_line = (info.get("license") or "").strip().splitlines()[:1]
        tags = [c.split(" :: ")[-1] for c in info.get("classifiers", []) if c.startswith("License")]
        print(f"{name} {version}: expression={info.get('license_expression')!r} "
              f"licence={first_line} classifiers={tags}")


def main(argv: list[str]) -> int:
    if "--lookup" in argv:
        lookup(LOCK, LICENCES)
        return 0
    found = problems(LOCK, LICENCES)
    if not found and "--verify" in argv:
        found = verify(LOCK, LICENCES)
    for line in found:
        print(line)
    if found:
        return 1
    confirmed = ", confirmed by PyPI" if "--verify" in argv else ""
    print(f"{len(locked_packages(LOCK))} Python packages, all with a permissive licence{confirmed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
