#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${CRYPTOFACE_ARTIFACT_ROOT:-}"
if [[ -z "${ARTIFACT_ROOT}" && -f "${ROOT}/.cryptoface-artifact-root" ]]; then
    IFS= read -r ARTIFACT_ROOT < "${ROOT}/.cryptoface-artifact-root"
fi
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ROOT}}"
if [[ "${ARTIFACT_ROOT}" != /* ]]; then
    ARTIFACT_ROOT="${ROOT}/${ARTIFACT_ROOT}"
fi

DEPS_DIR="${DEPS_DIR:-${ARTIFACT_ROOT}/.deps}"
SUBMISSION_BUILD_DIR="${CRYPTOFACE_BUILD_DIR:-${ARTIFACT_ROOT}/submission-build}"
LATTI_AI_SOURCE="${DEPS_DIR}/latti-ai"
LATTI_AI_BUILD="${LATTI_AI_BUILD:-${LATTI_AI_SOURCE}/build-h200}"
LATTI_AI_REPO="https://github.com/human-analysis/latti-ai.git"
LATTI_AI_COMMIT="88b4bac84b938f74e6936b19580181f97542e175"
LATTISENSE_COMMIT="809b35830550370b4570c9d45ae9e33c78a97156"

for command_name in cmake git go h5c++ nvidia-smi nvcc patch python3; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "[build_submission] missing required command: ${command_name}" >&2
        exit 2
    fi
done

if ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    | tr -d ' ' | grep -qx '9.0'; then
    echo "[build_submission] no compute-capability 9.0 GPU was found" >&2
    exit 2
fi

mkdir -p "${DEPS_DIR}" "${SUBMISSION_BUILD_DIR}"
if [[ ! -d "${LATTI_AI_SOURCE}/.git" ]]; then
    git clone --recursive "${LATTI_AI_REPO}" "${LATTI_AI_SOURCE}"
fi
git -C "${LATTI_AI_SOURCE}" fetch origin "${LATTI_AI_COMMIT}"
git -C "${LATTI_AI_SOURCE}" checkout --detach "${LATTI_AI_COMMIT}"
git -C "${LATTI_AI_SOURCE}" submodule sync --recursive
git -C "${LATTI_AI_SOURCE}" submodule update --init --recursive
"${ROOT}/scripts/apply_latti_source_patches.sh" "${LATTI_AI_SOURCE}"

actual_latti_ai="$(git -C "${LATTI_AI_SOURCE}" rev-parse HEAD)"
actual_lattisense="$(git -C "${LATTI_AI_SOURCE}/inference/lattisense" rev-parse HEAD)"
if [[ "${actual_latti_ai}" != "${LATTI_AI_COMMIT}" ]]; then
    echo "[build_submission] Latti-AI revision mismatch" >&2
    exit 1
fi
if [[ "${actual_lattisense}" != "${LATTISENSE_COMMIT}" ]]; then
    echo "[build_submission] LattiSense revision mismatch" >&2
    exit 1
fi

BUILD_DIR="${LATTI_AI_BUILD}" LATTISENSE_CUDA_ARCH=90 \
    "${LATTI_AI_SOURCE}/scripts/build_h200.sh"

LATTI_AI_SOURCE="${LATTI_AI_SOURCE}" LATTI_AI_BUILD="${LATTI_AI_BUILD}" \
    LATTI_STAGE_RUNTIME_OUTPUT="${SUBMISSION_BUILD_DIR}/latti_stage_runtime" \
    "${ROOT}/submission/native/build_runtime.sh"

python3 "${ROOT}/scripts/write_build_manifest.py" \
    --output "${SUBMISSION_BUILD_DIR}/build_manifest.json" \
    --latti-ai-source "${LATTI_AI_SOURCE}" \
    --native-binary "${SUBMISSION_BUILD_DIR}/latti_stage_runtime" \
    --runtime "${LATTI_AI_BUILD}/inference/lattisense/liblattisense.so" \
    --cuda-architecture sm_90

echo "[build_submission] H200 source build complete"
