#!/usr/bin/env bash
#SBATCH --job-name=cf-stream-cache-1g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=general-short
#SBATCH --constraint=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
cd "${ROOT}"

source scripts/load_h200_modules.sh
source .venv/bin/activate

LATTI_AI_SOURCE="${ROOT}/.deps/latti-ai"
LATTI_AI_BUILD="${LATTI_AI_SOURCE}/build-h200"
MEASUREDIR="measurements/slurm-streaming-cache-smoke"
SIZE="${CRYPTOFACE_VALIDATION_SIZE:-1}"

case "${SIZE}" in
  0) INSTANCE_DIR="single" ;;
  1) INSTANCE_DIR="small" ;;
  *) echo "Unsupported validation size: ${SIZE}" >&2; exit 2 ;;
esac

scripts/apply_latti_source_patches.sh "${LATTI_AI_SOURCE}"
grep -q 'std::unique_ptr<lattisense::FheTaskGpu> gpu_task' \
  "${LATTI_AI_SOURCE}/inference/inference_task/inference_process.h"
grep -q 'Created persistent GPU task for device' \
  "${LATTI_AI_SOURCE}/inference/inference_task/inference_process.cpp"

BUILD_DIR="${LATTI_AI_BUILD}" \
LATTISENSE_CUDA_ARCH=90 \
JOBS="${SLURM_CPUS_PER_TASK:-64}" \
  "${LATTI_AI_SOURCE}/scripts/build_h200.sh"

LATTI_AI_SOURCE="${LATTI_AI_SOURCE}" \
LATTI_AI_BUILD="${LATTI_AI_BUILD}" \
  submission/native/build_runtime.sh

python3 scripts/write_build_manifest.py \
  --output submission/build/build_manifest.json \
  --latti-ai-source "${LATTI_AI_SOURCE}" \
  --native-binary submission/build/latti_stage_runtime \
  --runtime "${LATTI_AI_BUILD}/inference/lattisense/liblattisense.so" \
  --cuda-architecture sm_90

python3 -m unittest \
  submission/test_gpu_runtime.py \
  submission/test_streamed_encrypted_compute.py

export CRYPTOFACE_GPUS=0
export CRYPTOFACE_MEASUREDIR="${MEASUREDIR}"
export CRYPTOFACE_RUN_OFFSET=0
python3 harness/run_submission.py "${SIZE}" --seed 42 --num_runs 1

GPU_LOG="io/${INSTANCE_DIR}/logs/server_gpu_0.log"
SCORER_LOG="io/${INSTANCE_DIR}/logs/server_scorer.log"
grep -Eq 'request_hits=0 request_misses=53' "${GPU_LOG}"
grep -Eq 'request_hits=53 request_misses=0' "${GPU_LOG}"
grep -q 'Created persistent GPU task for device' "${GPU_LOG}"
grep -q '# request_end' "${SCORER_LOG}"

jq -e --argjson validation_size "${SIZE}" '
  .["Server Reported"].additional_measurements["Scheduling mode"] ==
    "streamed_encrypted_score" and
  (if $validation_size == 0 then
    (.Quality["Encrypted model quality"].score as $score |
      ($score | type) == "number" and ($score | isfinite))
  else
    .Quality["Comparison to ArcFace baseline"].passed == true and
    .["Server Reported"].additional_measurements[
      "Encrypted inference/matching overlap"
    ] > 0
  end)
' "${MEASUREDIR}/results-1.json"

sha256sum \
  submission/build/latti_stage_runtime \
  "${LATTI_AI_BUILD}/inference/lattisense/liblattisense.so" \
  "${LATTI_AI_SOURCE}/inference/inference_task/inference_process.cpp"

echo "[validate_streaming_optimizations] PASS"
