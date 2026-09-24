#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=640G
#SBATCH --partition=general-short
#SBATCH --chdir=/mnt/gs21/scratch/vishnu/cryptoface
#SBATCH --output=/mnt/gs21/scratch/vishnu/cryptoface/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/gs21/scratch/vishnu/cryptoface/slurm-logs/%x-%j.err
set -euo pipefail
ulimit -c 0
umask 077
: "${FHE_SOURCE:?}"
: "${FHE_PYTHON:?}"
: "${FHE_SCRATCH:?}"
: "${FHE_BASELINE:?}"
: "${FHE_BACKEND_BUILD:?}"
: "${FHE_MATCHING_VALIDATION:?}"
: "${FHE_DATASET:?}"
: "${FHE_RELEASE_SHA256:?}"
: "${SLURM_JOB_ID:?}"
[[ "$(sha256sum "$FHE_SOURCE/release.json" | cut -d ' ' -f 1)" == "$FHE_RELEASE_SHA256" ]]
source "$FHE_BASELINE/source/scripts/load_h200_modules.sh"
export PYTHONDONTWRITEBYTECODE=1
mode="${1:?single or small}"
extra=()
case "$mode" in
  single) size=0 ;;
  small)
    size=1
    if [[ "${FHE_SMALL_INDEPENDENT:-0}" != 1 ]]; then
      : "${FHE_SINGLE_JOB:?}"
      extra+=(--previous "$FHE_SCRATCH/reproducible/gpu-matching-v2-single-$FHE_SINGLE_JOB/completed.json")
    fi
    ;;
  *) exit 2 ;;
esac
work="$FHE_SCRATCH/reproducible/gpu-matching-v2-$mode-$SLURM_JOB_ID"
cd "$FHE_SCRATCH"
"$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" prepare \
  --scratch-root "$FHE_SCRATCH" --work "$work" \
  --backend-build "$FHE_BACKEND_BUILD" --dataset "$FHE_DATASET"
if [[ "$mode" == small && "${FHE_SMALL_INDEPENDENT:-0}" == 1 ]]; then
  "$FHE_PYTHON" -B "${FHE_INDEPENDENT_RUNNER:?}" --source "$FHE_SOURCE" \
    --scratch-root "$FHE_SCRATCH" --work "$work" \
    --matching-validation "$FHE_MATCHING_VALIDATION" \
    --runner-sha256 "${FHE_INDEPENDENT_RUNNER_SHA256:?}"
else
  "$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" run \
    --scratch-root "$FHE_SCRATCH" --work "$work" --size "$size" \
    --matching-validation "$FHE_MATCHING_VALIDATION" "${extra[@]}"
fi
