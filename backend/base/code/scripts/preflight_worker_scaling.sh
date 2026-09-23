#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
JOB_SCRIPT="${ROOT}/scripts/diagnose_worker_scaling_slurm.sh"
CASE_FILTER="${ROOT}/scripts/worker_scaling_case_summary.jq"
COMBINED_FILTER="${ROOT}/scripts/worker_scaling_combined_summary.jq"
FIXTURE="${1:-${ROOT}/measurements/worker-scaling-16614565/one_gpu_before-server_reported.json}"
INPUT_ARTIFACT_ROOT="${CRYPTOFACE_DIAGNOSTIC_INPUT_ROOT:-/mnt/gs21/scratch/vishnu/cryptoface/CryptoFace-Latti-GPU}"
VENV_DIR="${CRYPTOFACE_VENV:-/mnt/gs21/scratch/vishnu/cryptoface/home-artifacts/CryptoFace-Latti-GPU-main/.venv}"

for command in bash jq nvidia-smi sha256sum; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Missing required command: ${command}" >&2
        exit 2
    fi
done

bash -n "${JOB_SCRIPT}"

required_files=(
    "${CASE_FILTER}"
    "${COMBINED_FILTER}"
    "${FIXTURE}"
    "${VENV_DIR}/bin/activate"
    "${ROOT}/submission/build/latti_stage_runtime"
    "${INPUT_ARTIFACT_ROOT}/io/large/public_keys/eval_context.bin"
    "${INPUT_ARTIFACT_ROOT}/io/large/ciphertexts_upload/p000000000000_i0___Split_output_0.ct"
    "${INPUT_ARTIFACT_ROOT}/io/large/server_model.json"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required preflight file: ${path}" >&2
        exit 2
    fi
done

PYTHONPATH=submission "${VENV_DIR}/bin/python" -m unittest -q \
    submission.test_streamed_encrypted_compute \
    submission.test_gpu_runtime
PYTHONPATH=submission "${VENV_DIR}/bin/python" -c \
    'from common import load_submission_config; from gpu_runtime import require_runtime; require_runtime(load_submission_config())'

jq -e '.task_dir and .model_parameters_sha256 and .mega_ag_sha256' \
    "${INPUT_ARTIFACT_ROOT}/io/large/server_model.json" >/dev/null

preflight_dir="$(mktemp -d /tmp/cryptoface-worker-preflight.XXXXXX)"
trap 'rm -rf -- "${preflight_dir}"' EXIT

for spec in \
    "one_gpu_before 0 1" \
    "four_gpu_concurrent 0,1,2,3 4" \
    "one_gpu_after 0 1"; do
    read -r case_name gpu_list warm_count <<<"${spec}"
    jq \
        --arg case_name "${case_name}" \
        --arg gpu_list "${gpu_list}" \
        --argjson pair_limit 12 \
        --argjson warm_count "${warm_count}" \
        -f "${CASE_FILTER}" "${FIXTURE}" \
        >"${preflight_dir}/${case_name}-summary.json"
done

jq -s -f "${COMBINED_FILTER}" \
    "${preflight_dir}/one_gpu_before-summary.json" \
    "${preflight_dir}/four_gpu_concurrent-summary.json" \
    "${preflight_dir}/one_gpu_after-summary.json" \
    >"${preflight_dir}/summary.json"

jq -e '
    (.cases | length) == 3 and
    .cases[0].case == "one_gpu_before" and
    .cases[1].case == "four_gpu_concurrent" and
    .cases[2].case == "one_gpu_after" and
    ([
        .four_vs_one_before_hot_ratio,
        .four_vs_one_after_hot_ratio,
        .one_gpu_drift_ratio
    ] | all(type == "number"))
' "${preflight_dir}/summary.json" >/dev/null

echo "[preflight_worker_scaling] PASS"
