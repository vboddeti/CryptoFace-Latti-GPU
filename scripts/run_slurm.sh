#!/usr/bin/env bash
#SBATCH --job-name=cf-fhe-upstream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=640G
#SBATCH --time=06:00:00
#SBATCH --partition=general-short
#SBATCH --chdir=/mnt/gs21/scratch/vishnu/cryptoface
#SBATCH --output=/mnt/gs21/scratch/vishnu/cryptoface/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/gs21/scratch/vishnu/cryptoface/slurm-logs/%x-%j.err
set -euo pipefail
ulimit -c 0
umask 077
# Cluster paths are explicitly supplied; no keys/builds/caches go in the checkout.
: "${FHE_SOURCE:?Set FHE_SOURCE to this release checkout}"
: "${FHE_PYTHON:?Set FHE_PYTHON to the validated Python executable}"
: "${FHE_SCRATCH:?Set FHE_SCRATCH to the scratch root}"
: "${FHE_BACKEND_BUILD:?Set FHE_BACKEND_BUILD to a verified backend build}"
: "${FHE_DATASET:?Set FHE_DATASET to the real face_dataset.h5}"
: "${FHE_RELEASE_SHA256:?Pin the release.json SHA-256 at submission time}"
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
[[ "$(sha256sum "$FHE_SOURCE/release.json" | cut -d ' ' -f 1)" == "$FHE_RELEASE_SHA256" ]]
size="${1:?Supply size 0, 1, 2, or 3}"
mode="${2:-benchmark}"
case "$size" in 0|1|2|3) ;; *) exit 2 ;; esac
case "$mode" in smoke|benchmark) ;; *) exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1
work="${FHE_SCRATCH}/reproducible/fhe-face-gpu-v1-${size}-${SLURM_JOB_ID}"
cd "$FHE_SCRATCH"
"$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" prepare \
  --scratch-root "$FHE_SCRATCH" --work "$work" \
  --backend-build "$FHE_BACKEND_BUILD" --dataset "$FHE_DATASET"
# The audited backend source contains the cluster's CUDA/GCC/NTL loader.
source "$FHE_BACKEND_BUILD/source/scripts/load_h200_modules.sh"
extra=()
if [[ "$mode" == smoke ]]; then extra+=(--smoke); fi
if [[ -n "${FHE_PREVIOUS:-}" ]]; then extra+=(--previous "$FHE_PREVIOUS"); fi
"$FHE_PYTHON" -B "$FHE_SOURCE/scripts/benchmark.py" run \
  --scratch-root "$FHE_SCRATCH" --work "$work" --size "$size" "${extra[@]}"
