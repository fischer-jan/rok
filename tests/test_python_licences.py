"""scripts/check_python_licences.py: an SPDX expression is met or not, and a
package that the licence file does not know fails the check."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ALLOW = {"MIT", "Apache-2.0", "BSD-3-Clause"}


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_python_licences", ROOT / "scripts" / "check_python_licences.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("expression, met", [
    ("MIT", True),
    ("GPL-3.0-only", False),
    ("Apache-2.0 OR GPL-3.0-only", True),  # one side is enough
    ("MIT AND GPL-3.0-only", False),  # both sides are needed
    ("MIT AND (GPL-3.0-only OR BSD-3-Clause)", True),
    ("(MIT OR GPL-3.0-only) AND LGPL-2.1-only", False),
    ("Apache-2.0 WITH LLVM-exception", False),  # the exception is not in the list
    ("", False),
])
def test_an_expression_is_met_only_by_allowed_licences(check, expression, met):
    assert check.permitted(expression, ALLOW) is met


def test_a_broken_expression_is_an_error(check):
    with pytest.raises(ValueError):
        check.permitted("MIT AND", ALLOW)
    with pytest.raises(ValueError):
        check.permitted("(MIT", ALLOW)


def test_an_unknown_package_and_a_strict_licence_fail(check, tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "known"\nversion = "1"\nsource = { registry = "https://example.invalid/simple" }\n'
        '[[package]]\nname = "strict"\nversion = "1"\nsource = { registry = "https://example.invalid/simple" }\n'
        '[[package]]\nname = "new"\nversion = "1"\nsource = { registry = "https://example.invalid/simple" }\n'
        '[[package]]\nname = "the-project"\nversion = "1"\nsource = { virtual = "." }\n')
    licences = tmp_path / "python_licences.toml"
    licences.write_text('allow = ["MIT"]\n[packages]\nknown = "MIT"\nstrict = "GPL-3.0-only"\n')
    found = check.problems(lock, licences)
    assert len(found) == 2
    assert found[0].startswith("new: no licence") and found[1].startswith("strict:")


@pytest.mark.parametrize("listed, info, agrees", [
    ("MIT", {"license_expression": "MIT"}, True),
    ("MIT", {"license_expression": "GPL-3.0-only"}, False),  # a line typed wrongly
    ("Apache-2.0 OR BSD-3-Clause", {"license_expression": "Apache-2.0  OR BSD-3-Clause"}, True),  # spaces do not count
    ("MIT", {"license": "MIT", "classifiers": []}, True),  # the older licence field
    ("BSD-3-Clause", {"classifiers": ["License :: OSI Approved :: BSD License"]}, True),
    ("MIT", {"classifiers": ["License :: OSI Approved :: BSD License"]}, False),
    ("MIT", {"license": "see the file COPYING", "classifiers": []}, False),  # nothing the script can read
])
def test_a_line_must_agree_with_what_pypi_says(check, listed, info, agrees):
    assert (check.disagreement(listed, info) is None) is agrees


def test_verify_names_the_wrong_line_and_skips_what_a_person_checked(check, tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "right"\nversion = "1"\nsource = { registry = "https://example.invalid/simple" }\n'
        '[[package]]\nname = "wrong"\nversion = "2"\nsource = { registry = "https://example.invalid/simple" }\n'
        '[[package]]\nname = "odd"\nversion = "3"\nsource = { registry = "https://example.invalid/simple" }\n')
    licences = tmp_path / "python_licences.toml"
    licences.write_text('allow = ["MIT"]\nchecked_by_hand = ["odd==3"]\n'
                        '[packages]\nright = "MIT"\nwrong = "MIT"\nodd = "MIT"\n')
    records = {"right": {"license_expression": "MIT"}, "wrong": {"license_expression": "LGPL-3.0-only"},
               "odd": {"license": "see the file COPYING"}}
    found = check.verify(lock, licences, fetch=lambda name, version: records[name])
    assert len(found) == 1 and found[0].startswith("wrong 2:") and "LGPL-3.0-only" in found[0]
    licences.write_text(licences.read_text().replace('"odd==3"', '"odd==2"'))  # the person checked an older version
    assert len(check.verify(lock, licences, fetch=lambda name, version: records[name])) == 2


def test_the_repo_passes(check):
    assert check.problems(check.LOCK, check.LICENCES) == []
