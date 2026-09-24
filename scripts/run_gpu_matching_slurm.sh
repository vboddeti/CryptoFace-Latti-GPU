#!/usr/bin/env bash
#SBATCH --job-name=cf-gpu-match-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=640G
#SBATCH --time=01:00:00
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
: "${FHE_DATASET:?}"
: "${FHE_RELEASE_SHA256:?}"
[[ "$(sha256sum "$FHE_SOURCE/release.json" | cut -d ' ' -f 1)" == "$FHE_RELEASE_SHA256" ]]
source "$FHE_BASELINE/source/scripts/load_h200_modules.sh"
export PYTHONDONTWRITEBYTECODE=1
mode="${1:?single or small}"
if [[ "$mode" == single ]]; then
    build="$FHE_SCRATCH/reproducible/gpu-matching-v2-build-$SLURM_JOB_ID"
    "$FHE_PYTHON" -B "$FHE_SOURCE/scripts/build_gpu_matching.py" \
        --scratch-root "$FHE_SCRATCH" --baseline-build "$FHE_BASELINE" --work "$build"
    diagnostic="$FHE_SCRATCH/reproducible/gpu-matching-v2-check-$SLURM_JOB_ID"
    "$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" prepare --scratch-root "$FHE_SCRATCH" \
        --work "$diagnostic" --backend-build "$build" --dataset "$FHE_DATASET"
    "$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" run --scratch-root "$FHE_SCRATCH" \
        --work "$diagnostic" --size 0 --smoke --matching-check
    work="$FHE_SCRATCH/reproducible/gpu-matching-v2-single-$SLURM_JOB_ID"
    "$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" prepare --scratch-root "$FHE_SCRATCH" \
        --work "$work" --backend-build "$build" --dataset "$FHE_DATASET"
    "$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" run --scratch-root "$FHE_SCRATCH" \
        --work "$work" --size 0 --matching-validation "$diagnostic/completed.json"
elif [[ "$mode" == small ]]; then
    : "${FHE_SINGLE_JOB:?}"
    build="$FHE_SCRATCH/reproducible/gpu-matching-v2-build-$FHE_SINGLE_JOB"
    diagnostic="$FHE_SCRATCH/reproducible/gpu-matching-v2-check-$FHE_SINGLE_JOB"
    previous="$FHE_SCRATCH/reproducible/gpu-matching-v2-single-$FHE_SINGLE_JOB/completed.json"
    work="$FHE_SCRATCH/reproducible/gpu-matching-v2-small-$SLURM_JOB_ID"
    "$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" prepare --scratch-root "$FHE_SCRATCH" \
        --work "$work" --backend-build "$build" --dataset "$FHE_DATASET"
    "$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" run --scratch-root "$FHE_SCRATCH" \
        --work "$work" --size 1 --previous "$previous" --matching-validation "$diagnostic/completed.json"
else
    exit 2
fi
