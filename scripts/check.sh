#!/usr/bin/env bash
# Rok's checks in one go. They run on this machine.
#
#   scripts/check.sh          # all checks
#   scripts/check.sh BASE     # the same; the review looks at the commits after BASE
#
# Every step runs, and a failed step makes the script exit 1.
#   names     ruff: an undefined name only fails when its line runs
#   build     maturin: the native module as a wheel, unpacked into a temporary
#             folder. The tests run against that copy, so they test the Rust
#             code of this checkout, and the module in the venv stays as it is.
#   test      pytest, the whole suite, in parallel (pytest-xdist)
#   lint      cargo clippy, warnings are errors
#   licences  only permissive licences: cargo deny for the Rust crates,
#             scripts/check_python_licences.py for the Python packages of uv.lock
#             (compared with PyPI when uv.lock or the licence file changed)
#   audit     cargo audit: a vulnerable, unsound or yanked crate fails;
#             unmaintained crates are counted (native/.cargo/audit.toml)
#   secrets   gitleaks: keys and passwords in the git history
# A clone can hold a review script (.claude/ci/review.sh). It runs last and only
# when every check passed, it only gives information, and it cannot fail the run.
#
# Tools: uv, cargo-deny, cargo-audit, gitleaks.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
# The venv and the build cache live in the main checkout, also for a worktree.
MAIN="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
VENV="$MAIN/.venv-rok"
export CARGO_TARGET_DIR="$MAIN/native/target"
LOGS="$(mktemp -d)"
trap 'rm -rf "$LOGS"' EXIT
FAILED=()
T0=$SECONDS

# step <name> <command...>: keep the output, show its end when the step fails.
step() {
  local name="$1"; shift
  local t=$SECONDS
  if "$@" >"$LOGS/$name.log" 2>&1; then
    printf '▸ %-9s ok    %3ss\n' "$name" "$((SECONDS - t))"
  else
    printf '✗ %-9s FAIL  %3ss\n' "$name" "$((SECONDS - t))"
    tail -n 40 "$LOGS/$name.log" | sed 's/^/    /'
    FAILED+=("$name")
  fi
}

# The commits to look at come after BASE: the argument, or where the branch left main.
BASE="${1:-}"
[[ -n "$BASE" ]] || BASE="$(git merge-base origin/main HEAD 2>/dev/null || echo origin/main)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "! The working tree has uncommitted changes. The checks test the files on"
  echo "  disk, and those are not the same as the commits."
fi

step names    uvx ruff check --quiet --select F821,F822,F823 rok scripts tests
build_native() {
  uvx maturin build --release -m native/Cargo.toml -i "$VENV/bin/python" -o "$LOGS/wheel" \
    && mkdir -p "$LOGS/site" && unzip -q -o "$LOGS"/wheel/*.whl -d "$LOGS/site"
}
step build    build_native
step test     env PYTHONPATH="$LOGS/site" "$VENV/bin/python" -m pytest -q -p no:cacheprovider -n auto tests
step lint     cargo clippy --locked --manifest-path native/Cargo.toml --all-targets -- -D warnings
# The licence file is compared with PyPI when the Python packages or the file changed.
PY_LICENCES=""
git diff --quiet "$BASE" -- uv.lock scripts/python_licences.toml 2>/dev/null || PY_LICENCES="--verify"
step licences bash -c '(cd native && cargo deny check licenses) && "$0" scripts/check_python_licences.py $1' "$VENV/bin/python" "$PY_LICENCES"
step audit    bash -c 'cd native && cargo audit --deny unsound --deny yanked'
if [[ -f "$LOGS/audit.log" ]]; then
  UNMAINTAINED="$(grep -cE '^Warning: +unmaintained' "$LOGS/audit.log" || true)"
  [[ "$UNMAINTAINED" -eq 0 ]] || echo "    $UNMAINTAINED unmaintained crates (allowed)"
fi
step secrets  gitleaks git --no-banner --redact

# The review sends the diff to a service, so it runs only when every check passed:
# a diff that failed the secrets check never leaves this machine.
REVIEW="$REPO/.claude/ci/review.sh"
if [[ -f "$REVIEW" ]]; then
  if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "▸ review    advisory, commits after $BASE"
    bash "$REVIEW" "$BASE" HEAD
  else
    echo "▸ review    skipped: a check failed"
  fi
fi

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "All checks passed in $((SECONDS - T0))s."
else
  echo "CHECKS FAILED in $((SECONDS - T0))s: ${FAILED[*]}"
  exit 1
fi
