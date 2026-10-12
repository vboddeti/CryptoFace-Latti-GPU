#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
umask 077
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" -B "$ROOT/scripts/setup_runtime.py" "$@"
