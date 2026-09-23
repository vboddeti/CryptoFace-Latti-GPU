#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
else
    python3 -m venv .bootstrap-uv
    .bootstrap-uv/bin/python -m pip install --upgrade pip uv
    UV_BIN="${ROOT_DIR}/.bootstrap-uv/bin/uv"
fi

"${UV_BIN}" venv .venv --python 3.12
"${UV_BIN}" pip install --python .venv/bin/python -r requirements.txt

echo "[install_python_deps] environment ready; run: source .venv/bin/activate"
echo "[install_python_deps] next: bash scripts/build_submission.sh"
