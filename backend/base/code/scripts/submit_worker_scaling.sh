#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

scripts/preflight_worker_scaling.sh
exec sbatch "$@" scripts/diagnose_worker_scaling_slurm.sh
