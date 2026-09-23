#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATTI_AI_SOURCE="${1:-${ROOT}/.deps/latti-ai}"
PATCH_FILE="${ROOT}/submission/native/latti-persistent-gpu-task.patch"

if [[ ! -d "${LATTI_AI_SOURCE}/inference/inference_task" ]]; then
  echo "[apply_latti_source_patches] invalid Latti-AI source: ${LATTI_AI_SOURCE}" >&2
  exit 2
fi

HEADER="${LATTI_AI_SOURCE}/inference/inference_task/inference_process.h"
SOURCE="${LATTI_AI_SOURCE}/inference/inference_task/inference_process.cpp"

header_is_patched=0
source_is_patched=0
grep -q 'std::unique_ptr<lattisense::FheTaskGpu> gpu_task' "${HEADER}" && header_is_patched=1
grep -q 'Created persistent GPU task for device' "${SOURCE}" && source_is_patched=1

if [[ "${header_is_patched}" -eq 1 && "${source_is_patched}" -eq 1 ]]; then
  echo "[apply_latti_source_patches] persistent GPU task patch already applied"
elif [[ "${header_is_patched}" -ne "${source_is_patched}" ]]; then
  echo "[apply_latti_source_patches] source is only partially patched" >&2
  exit 2
elif patch --batch --forward --dry-run -d "${LATTI_AI_SOURCE}" -p1 < "${PATCH_FILE}" >/dev/null; then
  patch --batch -d "${LATTI_AI_SOURCE}" -p1 < "${PATCH_FILE}"
  echo "[apply_latti_source_patches] applied persistent GPU task patch"
else
    echo "[apply_latti_source_patches] patch does not apply cleanly: ${PATCH_FILE}" >&2
    exit 2
fi

LATTISENSE_SOURCE="${LATTI_AI_SOURCE}/inference/lattisense"
ABI_SOURCE="${LATTISENSE_SOURCE}/fhe_ops_lib/lattigo/go_sdk/c_struct_import_export.go"
ABI_PATCH_FILE="${ROOT}/submission/native/lattisense-safe-abi-export.patch"
ABI_UPGRADE_PATCH_FILE="${ROOT}/submission/native/lattisense-remove-abi-serialization.patch"

if [[ ! -f "${ABI_SOURCE}" ]]; then
    echo "[apply_latti_source_patches] missing LattiSense ABI source: ${ABI_SOURCE}" >&2
    exit 2
fi

if grep -q 'Transform a Go-owned copy before exporting it' "${ABI_SOURCE}" &&
   ! grep -q 'abiBridgeMu' "${ABI_SOURCE}"; then
    echo "[apply_latti_source_patches] safe concurrent ABI export patch already applied"
elif grep -q 'Transform a Go-owned copy before exporting it' "${ABI_SOURCE}" &&
     grep -q 'abiBridgeMu' "${ABI_SOURCE}" &&
     patch --batch --forward --dry-run -d "${LATTISENSE_SOURCE}" -p1 \
        < "${ABI_UPGRADE_PATCH_FILE}" >/dev/null; then
    patch --batch -d "${LATTISENSE_SOURCE}" -p1 < "${ABI_UPGRADE_PATCH_FILE}"
    echo "[apply_latti_source_patches] removed global ABI bridge serialization"
elif patch --batch --forward --dry-run -d "${LATTISENSE_SOURCE}" -p1 \
        < "${ABI_PATCH_FILE}" >/dev/null; then
    patch --batch -d "${LATTISENSE_SOURCE}" -p1 < "${ABI_PATCH_FILE}"
    echo "[apply_latti_source_patches] applied safe concurrent ABI export patch"
else
    echo "[apply_latti_source_patches] ABI patch does not apply cleanly: ${ABI_PATCH_FILE}" >&2
    exit 2
fi
