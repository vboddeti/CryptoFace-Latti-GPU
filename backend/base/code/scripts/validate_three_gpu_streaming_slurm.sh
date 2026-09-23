#!/usr/bin/env bash
#SBATCH --job-name=cf-stream-3g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=general-short
#SBATCH --constraint=h200
#SBATCH --gres=gpu:h200:3
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
cd "${ROOT}"

source scripts/load_h200_modules.sh
VENV_DIR="${CRYPTOFACE_VENV:-${ROOT}/.venv}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Missing Python virtual environment: ${VENV_DIR}" >&2
  exit 2
fi
source "${VENV_DIR}/bin/activate"

EXPECTED_GPU_COUNT="${CRYPTOFACE_EXPECTED_GPU_COUNT:-3}"
if [[ ! "${EXPECTED_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CRYPTOFACE_EXPECTED_GPU_COUNT must be a positive integer" >&2
  exit 2
fi
if [[ -z "${CRYPTOFACE_GPUS:-}" ]]; then
  CRYPTOFACE_GPUS="$(seq -s, 0 "$((EXPECTED_GPU_COUNT - 1))")"
fi
export CRYPTOFACE_GPUS
export CRYPTOFACE_GPU_KEY_CACHE=1
export CRYPTOFACE_GPU_EVENT_CACHE=0
export CRYPTOFACE_STREAM_SCORING=1
export LATTISENSE_GPU_TIMING=1
export LATTISENSE_GPU_EVENT_CACHE_STATS=1
export LATTISENSE_GPU_PIN_REQUEST_H2D_STATS=1

MEASUREDIR="measurements/${EXPECTED_GPU_COUNT}-gpu-streaming-${SLURM_JOB_ID:-manual}"
mkdir -p "${MEASUREDIR}"

{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "CRYPTOFACE_GPUS=${CRYPTOFACE_GPUS}"
  module list 2>&1
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,power.limit,clocks.max.sm,clocks.max.mem,compute_cap --format=csv
} > "${MEASUREDIR}/environment.txt"

python3 -m unittest \
  submission/test_gpu_runtime.py \
  submission/test_streamed_encrypted_compute.py

CRYPTOFACE_MEASUREDIR="${MEASUREDIR}" \
  CRYPTOFACE_RUN_OFFSET=0 \
  python3 harness/run_submission.py 1 --seed 42 --num_runs 1

RESULT="${MEASUREDIR}/results-1.json"
jq -e --argjson expected_gpu_count "${EXPECTED_GPU_COUNT}" '
  .Quality["Comparison to ArcFace baseline"].passed == true and
  .["Server Reported"].additional_measurements["Pair count"] == 128 and
  .["Server Reported"].additional_measurements["Scheduling mode"] == "streamed_encrypted_score" and
  .["Server Reported"].additional_measurements["Encrypted inference/matching overlap"] > 0 and
  .["Server Reported"].additional_measurements.workers.gpu_count == $expected_gpu_count
' "${RESULT}" >/dev/null

jq '{
  encrypted_computation_seconds: (.["Server Reported"]["Encrypted computation"] | rtrimstr("s") | tonumber),
  total_server_seconds: (.["Server Reported"].Total | rtrimstr("s") | tonumber),
  pair_count: .["Server Reported"].additional_measurements["Pair count"],
  encrypted_embedding_inference_seconds: .["Server Reported"].additional_measurements["Encrypted embedding inference"],
  encrypted_pair_matching_seconds: .["Server Reported"].additional_measurements["Encrypted pair matching"],
  overlap_seconds: .["Server Reported"].additional_measurements["Encrypted inference/matching overlap"],
  gpu_count: .["Server Reported"].additional_measurements.workers.gpu_count,
  pair_interval_seconds: ((.["Server Reported"]["Encrypted computation"] | rtrimstr("s") | tonumber) / 128),
  arcface_passed: .Quality["Comparison to ArcFace baseline"].passed
}' "${RESULT}" | tee "${MEASUREDIR}/summary.json"

echo "[validate_multi_gpu_streaming] PASS ${MEASUREDIR}"
