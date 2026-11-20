#!/usr/bin/env bash
# Linux with no GPU and no display: build the native module headless, run the
# tests and (with --smoke) render one picture through the MCP server.
#
#   scripts/linux_headless.sh [--smoke OUT.png]
#
# Needs: rust (cargo), uv, a software Vulkan driver (Debian/Ubuntu:
# mesa-vulkan-drivers, which has lavapipe).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"
export PATH="${HOME}/.cargo/bin:${PATH}"
VENV="${REPO}/.venv-rok"

[ -x "${VENV}/bin/python" ] || uv venv "${VENV}"
uv pip install --python "${VENV}/bin/python" numpy pytest pytest-timeout "mcp>=2.2"

echo "== native module (headless: no GTK, no window)"
VIRTUAL_ENV="${VENV}" timeout 1500 uvx maturin develop --release -m native/Cargo.toml \
  --no-default-features --features headless

echo "== pytest (each test limited to 120 s, see pyproject.toml)"
timeout 1500 "${VENV}/bin/python" -m pytest -q tests

if [ "${1:-}" = "--smoke" ]; then
  OUT="${2:-/tmp/rok-linux-render.png}"
  echo "== MCP smoke test -> ${OUT}"
  timeout 300 "${VENV}/bin/python" scripts/linux_mcp_smoke.py "${OUT}"
fi
