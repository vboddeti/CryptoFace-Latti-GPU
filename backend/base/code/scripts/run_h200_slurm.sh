#!/usr/bin/env bash
#SBATCH --job-name=cryptoface-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=general-short
#SBATCH --constraint=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/gs21/scratch/vishnu/cryptoface/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/gs21/scratch/vishnu/cryptoface/slurm-logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

SIZE="${CRYPTOFACE_SIZE:-0}"
SEED="${CRYPTOFACE_SEED:-42}"
NUM_RUNS="${CRYPTOFACE_NUM_RUNS:-1}"
RUN_OFFSET="${CRYPTOFACE_RUN_OFFSET:-0}"

cd "${ROOT}"
source scripts/load_h200_modules.sh
VENV_DIR="${CRYPTOFACE_VENV:-/mnt/gs21/scratch/vishnu/cryptoface/home-artifacts/CryptoFace-Latti-GPU-main/.venv}"
source "${VENV_DIR}/bin/activate"

# Prefer CUDA indices visible inside the Slurm job. This cluster remaps an
# allocated physical GPU (for example SLURM_JOB_GPUS=2) to logical CUDA device
# 0, so each native worker must use the job-local namespace.
if [[ -z "${CRYPTOFACE_GPUS:-}" ]]; then
  allocated_gpus="${CUDA_VISIBLE_DEVICES:-}"
  allocated_gpus="${allocated_gpus// /,}"
  if [[ "${allocated_gpus}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    export CRYPTOFACE_GPUS="${allocated_gpus}"
  else
    echo "Unable to derive logical CUDA indices from the Slurm allocation." >&2
    echo "Submit with --export=ALL,CRYPTOFACE_GPUS=0[,1,...] on this cluster." >&2
    exit 2
  fi
fi

environment_dir="${ROOT}/measurements/slurm-environments"
mkdir -p "${environment_dir}"
environment_log="${environment_dir}/${SLURM_JOB_ID:-manual}.txt"

{
  date -Iseconds
  hostname
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "CRYPTOFACE_GPUS=${CRYPTOFACE_GPUS}"
  echo "CRYPTOFACE_FACE_CPU_THREADS=${CRYPTOFACE_FACE_CPU_THREADS}"
  echo "CRYPTOFACE_SIZE=${SIZE}"
  echo "CRYPTOFACE_SEED=${SEED}"
  echo "CRYPTOFACE_NUM_RUNS=${NUM_RUNS}"
  echo "CRYPTOFACE_RUN_OFFSET=${RUN_OFFSET}"
  echo "CRYPTOFACE_MEASUREDIR=${CRYPTOFACE_MEASUREDIR:-}"
  module -t list 2>&1
  nvcc --version
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,compute_cap \
    --format=csv,noheader
} > "${environment_log}"

CRYPTOFACE_RUN_OFFSET="${RUN_OFFSET}" python3 harness/run_submission.py "${SIZE}" \
  --seed "${SEED}" \
  --num_runs "${NUM_RUNS}"
