#!/usr/bin/env bash
#SBATCH --job-name=cf-stream-ab-1g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=general-short
#SBATCH --constraint=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
cd "${ROOT}"

source scripts/load_h200_modules.sh
source .venv/bin/activate

export CRYPTOFACE_GPUS=0
export CRYPTOFACE_KEEP_STAGE_INPUTS=1
export LATTISENSE_GPU_TIMING=1
export LATTISENSE_GPU_EVENT_CACHE_STATS=1
export LATTISENSE_GPU_PIN_REQUEST_H2D_STATS=1

DIAGNOSTIC_PAIR_LIMIT="${CRYPTOFACE_PAIR_LIMIT:-16}"
DIAGNOSTIC_DIR="measurements/streaming-ab-${SLURM_JOB_ID:-manual}"
mkdir -p "${DIAGNOSTIC_DIR}"

# The home checkout intentionally does not carry the 80 MiB indexed dataset.
# Reuse the validated scratch copy unless an explicit store was supplied.
DATASET_STORE="${ROOT}/datasets/face_dataset.h5"
DATASET_SOURCE="${CRYPTOFACE_DATASET_STORE:-/mnt/gs21/scratch/vishnu/cryptoface/CryptoFace-Latti-GPU/datasets/face_dataset.h5}"
if [[ ! -f "${DATASET_STORE}" ]]; then
  if [[ ! -f "${DATASET_SOURCE}" ]]; then
    echo "[diagnose_streaming_regression] Missing indexed dataset: ${DATASET_STORE}" >&2
    echo "[diagnose_streaming_regression] Fallback is also missing: ${DATASET_SOURCE}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${DATASET_STORE}")"
  ln -sfn "${DATASET_SOURCE}" "${DATASET_STORE}"
fi

{
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "CRYPTOFACE_PAIR_LIMIT=${DIAGNOSTIC_PAIR_LIMIT}"
    module list 2>&1
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,power.limit,clocks.max.sm,clocks.max.mem,compute_cap --format=csv
} > "${DIAGNOSTIC_DIR}/environment.txt"

python3 -m unittest \
    submission/test_gpu_runtime.py \
    submission/test_streamed_encrypted_compute.py

export CRYPTOFACE_PAIR_LIMIT="${DIAGNOSTIC_PAIR_LIMIT}"

# Reuse the validated client/evaluation contexts and model, but create one
# deterministic encrypted input set shared by every A/B case.
python3 harness/generate_input.py 1 --seed 191664963
python3 harness/materialize_input_store.py 1
python3 submission/client_preprocess_input.py 1
python3 submission/client_encode_encrypt_input.py 1

# Avoid giving the first case an artificial page-cache advantage from the
# immediately preceding encryption writes.
PYTHONPATH=submission python3 -c \
    'from pathlib import Path; from common import release_file_cache; [release_file_cache(p) for p in Path("io/small/ciphertexts_upload").glob("*.ct")]'

run_case() {
    local case_name="$1"
    local key_cache="$2"
    local event_cache="$3"
    local stream_scoring="$4"
    local case_dir="${DIAGNOSTIC_DIR}/${case_name}"
    local telemetry_pid
    local status

    mkdir -p "${case_dir}"
    export CRYPTOFACE_GPU_KEY_CACHE="${key_cache}"
    export CRYPTOFACE_GPU_EVENT_CACHE="${event_cache}"
    export CRYPTOFACE_STREAM_SCORING="${stream_scoring}"

    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu --format=csv \
        > "${case_dir}/gpu-before.csv"
    nvidia-smi dmon -s pucvmet -d 1 \
        > "${case_dir}/gpu-dmon.log" 2>&1 &
    telemetry_pid=$!

    set +e
    /usr/bin/time -v -o "${case_dir}/process-time.txt" \
        python3 submission/server_encrypted_compute.py 1 \
        > "${case_dir}/stdout.log" 2> "${case_dir}/stderr.log"
    status=$?
    set -e

    kill "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true

    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu --format=csv \
        > "${case_dir}/gpu-after.csv"
    cp io/small/server_reported.json "${case_dir}/server_reported.json"
    cp io/small/logs/server_gpu_0.log "${case_dir}/server_gpu_0.log"
    cp io/small/logs/server_scorer.log "${case_dir}/server_scorer.log"

    jq --arg case_name "${case_name}" \
       --argjson exit_code "${status}" \
       --arg key_cache "${key_cache}" \
       --arg event_cache "${event_cache}" \
       --arg stream_scoring "${stream_scoring}" \
       '{
          case: $case_name,
          exit_code: $exit_code,
          key_cache: $key_cache,
          event_cache: $event_cache,
          stream_scoring: $stream_scoring,
          report: .
        }' "${case_dir}/server_reported.json" > "${case_dir}/result.json"

    if [[ "${status}" -ne 0 ]]; then
        echo "[diagnose_streaming_regression] ${case_name} failed: ${status}" >&2
        return "${status}"
    fi
}

# Repeat the old-style control at both ends to expose thermal, clock, or
# filesystem drift across the allocation.
run_case cache_off_seq_a 0 0 0
run_case cache_on_noevent_seq 1 0 0
run_case cache_on_event_seq 1 1 0
run_case cache_on_event_stream 1 1 1
run_case cache_off_seq_b 0 0 0

jq -s '.' "${DIAGNOSTIC_DIR}"/*/result.json > "${DIAGNOSTIC_DIR}/all-results.json"
echo "[diagnose_streaming_regression] PASS ${DIAGNOSTIC_DIR}"
