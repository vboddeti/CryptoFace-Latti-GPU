#!/usr/bin/env bash
#SBATCH --job-name=cf-worker-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=general-short
#SBATCH --constraint=h200
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/gs21/scratch/vishnu/cryptoface/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/gs21/scratch/vishnu/cryptoface/slurm-logs/%x-%j.err

set -euo pipefail

ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
cd "${ROOT}"

source scripts/load_h200_modules.sh

VENV_DIR="${CRYPTOFACE_VENV:-/mnt/gs21/scratch/vishnu/cryptoface/home-artifacts/CryptoFace-Latti-GPU-main/.venv}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Missing Python environment: ${VENV_DIR}" >&2
    exit 2
fi
source "${VENV_DIR}/bin/activate"

CPU_AFFINITY_LIST="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
CPU_AFFINITY_COUNT="$(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
EXPECTED_CPU_COUNT="${SLURM_CPUS_PER_TASK:-${SLURM_JOB_CPUS_PER_NODE:-64}}"
EXPECTED_CPU_COUNT="${EXPECTED_CPU_COUNT%%(*}"
if [[ ! "${EXPECTED_CPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Unable to determine allocated CPU count: ${EXPECTED_CPU_COUNT}" >&2
    exit 2
fi
if (( CPU_AFFINITY_COUNT < EXPECTED_CPU_COUNT )); then
    echo "CPU affinity exposes ${CPU_AFFINITY_COUNT} CPUs, but Slurm allocated ${EXPECTED_CPU_COUNT}; refusing to start GPU workers" >&2
    echo "Visible CPU affinity: ${CPU_AFFINITY_LIST}" >&2
    exit 3
fi

# These are generated ciphertext/key artifacts only.  The executable, Python
# harness, configuration, and model validation all come from ROOT (home).
INPUT_ARTIFACT_ROOT="${CRYPTOFACE_DIAGNOSTIC_INPUT_ROOT:-/mnt/gs21/scratch/vishnu/cryptoface/CryptoFace-Latti-GPU}"
PAIR_LIMIT="${CRYPTOFACE_PAIR_LIMIT:-12}"
if [[ ! "${PAIR_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CRYPTOFACE_PAIR_LIMIT must be a positive integer" >&2
    exit 2
fi

EVAL_CONTEXT="${INPUT_ARTIFACT_ROOT}/io/large/public_keys/eval_context.bin"
FIRST_INPUT="${INPUT_ARTIFACT_ROOT}/io/large/ciphertexts_upload/p000000000000_i0___Split_output_0.ct"
if [[ ! -f "${EVAL_CONTEXT}" || ! -f "${FIRST_INPUT}" ]]; then
    echo "Missing retained diagnostic input artifacts below ${INPUT_ARTIFACT_ROOT}" >&2
    exit 2
fi

JOB_TAG="${SLURM_JOB_ID:-manual}"
SCRATCH_RUN="/mnt/gs21/scratch/vishnu/cryptoface/diagnostics/worker-scaling-${JOB_TAG}"
DURABLE_DIR="${ROOT}/measurements/worker-scaling-${JOB_TAG}"
mkdir -p "${SCRATCH_RUN}" "${DURABLE_DIR}"

export CRYPTOFACE_ARTIFACT_ROOT="${INPUT_ARTIFACT_ROOT}"
export CRYPTOFACE_PAIR_LIMIT="${PAIR_LIMIT}"
export CRYPTOFACE_KEEP_STAGE_INPUTS=1
export CRYPTOFACE_GPU_KEY_CACHE=1
export CRYPTOFACE_GPU_EVENT_CACHE=0
export CRYPTOFACE_STREAM_SCORING=1
export LATTISENSE_GPU_TIMING=1
export LATTISENSE_GPU_EVENT_CACHE_STATS=1
export LATTISENSE_GPU_PIN_REQUEST_H2D_STATS=1

{
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
    echo "SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "ROOT=${ROOT}"
    echo "INPUT_ARTIFACT_ROOT=${INPUT_ARTIFACT_ROOT}"
    echo "SCRATCH_RUN=${SCRATCH_RUN}"
    echo "PAIR_LIMIT=${PAIR_LIMIT}"
    echo "VENV_DIR=${VENV_DIR}"
    echo "CPU_AFFINITY_COUNT=${CPU_AFFINITY_COUNT}"
    echo "CPU_AFFINITY_LIST=${CPU_AFFINITY_LIST}"
    git rev-parse HEAD
    git status --short
    sha256sum submission/build/latti_stage_runtime
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,power.limit,clocks.max.sm,clocks.max.mem,compute_cap --format=csv
    lscpu
    if command -v numactl >/dev/null 2>&1; then
        numactl --hardware
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        scontrol show job "${SLURM_JOB_ID}"
    fi
} >"${DURABLE_DIR}/environment.txt" 2>&1

monitor_pids=()

stop_monitors() {
    local pid
    for pid in "${monitor_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    done
    monitor_pids=()
}

start_monitors() {
    local case_dir="$1"
    nvidia-smi \
        --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,temperature.gpu \
        --format=csv -l 1 >"${case_dir}/gpu-query.csv" 2>&1 &
    monitor_pids+=("$!")

    nvidia-smi dmon -s pucvmet -d 1 -o DT >"${case_dir}/gpu-dmon.log" 2>&1 &
    monitor_pids+=("$!")

    (
        while true; do
            date --iso-8601=ns
            while read -r pid; do
                [[ -n "${pid}" ]] || continue
                ps -o pid,ppid,psr,nlwp,%cpu,%mem,etime,cmd -p "${pid}" || true
                taskset -pc "${pid}" 2>&1 || true
                grep -E 'Cpus_allowed_list|Mems_allowed_list' "/proc/${pid}/status" 2>/dev/null || true
            done < <(pgrep -u "$(id -u)" -f 'latti_stage_runtime.*server' || true)
            sleep 5
        done
    ) >"${case_dir}/process-affinity.log" 2>&1 &
    monitor_pids+=("$!")
}

run_case() {
    local label="$1"
    local gpu_list="$2"
    local warm_count="$3"
    local case_dir="${SCRATCH_RUN}/${label}"
    local status=0
    mkdir -p "${case_dir}"

    export CRYPTOFACE_GPUS="${gpu_list}"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
        >"${case_dir}/compute-apps-before.csv" 2>&1 || true
    start_monitors "${case_dir}"

    /usr/bin/time -v -o "${case_dir}/process-time.txt" \
        python submission/server_encrypted_compute.py 3 \
        >"${case_dir}/stdout.log" 2>"${case_dir}/stderr.log" || status=$?
    stop_monitors

    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
        >"${case_dir}/compute-apps-after.csv" 2>&1 || true

    if [[ -f "${INPUT_ARTIFACT_ROOT}/io/large/server_reported.json" ]]; then
        cp "${INPUT_ARTIFACT_ROOT}/io/large/server_reported.json" \
            "${DURABLE_DIR}/${label}-server_reported.json"
    fi
    if [[ -d "${INPUT_ARTIFACT_ROOT}/io/large/logs" ]]; then
        cp "${INPUT_ARTIFACT_ROOT}/io/large/logs/server_gpu_"*.log "${case_dir}/" 2>/dev/null || true
        cp "${INPUT_ARTIFACT_ROOT}/io/large/logs/server_scorer.log" "${case_dir}/" 2>/dev/null || true
    fi

    if [[ "${status}" -ne 0 ]]; then
        echo "${label} failed with status ${status}" >&2
        return "${status}"
    fi

    jq \
        --arg case_name "${label}" \
        --arg gpu_list "${gpu_list}" \
        --argjson pair_limit "${PAIR_LIMIT}" \
        --argjson warm_count "${warm_count}" \
        -f scripts/worker_scaling_case_summary.jq \
        "${DURABLE_DIR}/${label}-server_reported.json" \
        >"${DURABLE_DIR}/${label}-summary.json"
}

trap stop_monitors EXIT INT TERM

# Repeating the one-worker control after the four-worker case distinguishes
# four-way contention from node/clock drift during the allocation.
run_case one_gpu_before 0 1
run_case four_gpu_concurrent 0,1,2,3 4
run_case one_gpu_after 0 1

jq -s -f scripts/worker_scaling_combined_summary.jq \
    "${DURABLE_DIR}/one_gpu_before-summary.json" \
    "${DURABLE_DIR}/four_gpu_concurrent-summary.json" \
    "${DURABLE_DIR}/one_gpu_after-summary.json" \
    >"${DURABLE_DIR}/summary.json"

echo "[diagnose_worker_scaling] PASS ${DURABLE_DIR}"
echo "[diagnose_worker_scaling] telemetry ${SCRATCH_RUN}"
