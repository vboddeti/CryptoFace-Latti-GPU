#!/usr/bin/env bash
#SBATCH --job-name=cf-abi-fix-4g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=general-short
#SBATCH --constraint=h200
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${ROOT}"
source scripts/load_h200_modules.sh
source .venv/bin/activate

# Ignore any login-shell override (for example a stale single-GPU value) and
# use the complete four-GPU Slurm allocation in the job-local CUDA namespace.
allocated_gpus="${CUDA_VISIBLE_DEVICES:-}"
allocated_gpus="${allocated_gpus// /,}"
if [[ ! "${allocated_gpus}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
    echo "Expected four CUDA devices from Slurm; got '${allocated_gpus}'." >&2
    exit 2
fi
export CRYPTOFACE_GPUS="${allocated_gpus}"

GO_SDK="${ROOT}/.deps/latti-ai/inference/lattisense/fhe_ops_lib/lattigo/go_sdk"
SOURCE="${GO_SDK}/c_struct_import_export.go"
LIBRARY="${GO_SDK}/liblattigo.so"
BACKUP="${GO_SDK}/liblattigo.so.pre-abi-fix"
CACHE_ROOT="${ROOT}/.cache/abi-fix"
BUILD_TMP="$(mktemp -d "${ROOT}/.abi-fix-build.XXXXXX")"
trap 'rm -rf "${BUILD_TMP}"' EXIT

mkdir -p "${CACHE_ROOT}/go-build" "${CACHE_ROOT}/go-tmp"
export GOCACHE="${CACHE_ROOT}/go-build"
export GOTMPDIR="${CACHE_ROOT}/go-tmp"
export GOTELEMETRY=off

environment_dir="${ROOT}/measurements/slurm-environments"
mkdir -p "${environment_dir}"
environment_log="${environment_dir}/${SLURM_JOB_ID:-manual}.txt"
{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "CRYPTOFACE_GPUS=${CRYPTOFACE_GPUS}"
  echo "CRYPTOFACE_SIZE=${CRYPTOFACE_SIZE:-1}"
  echo "CRYPTOFACE_SEED=${CRYPTOFACE_SEED:-42}"
  echo "CRYPTOFACE_RUN_OFFSET=${CRYPTOFACE_RUN_OFFSET:-1}"
  echo "CRYPTOFACE_MEASUREDIR=${CRYPTOFACE_MEASUREDIR:-measurements/small-4gpu-fix-validation}"
  echo "ABI_SOURCE_SHA256=$(sha256sum "${SOURCE}" | cut -d' ' -f1)"
  module -t list 2>&1
  go version
  cmake --version | head -n 1
  nvcc --version | tail -n 1
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,compute_cap --format=csv,noheader
} | tee "${environment_log}"

cd "${GO_SDK}"
go test ./...
go build -buildmode=c-shared -o "${BUILD_TMP}/liblattigo.so" \
  main.go bootstrap.go c_struct_import_export.go conversion.go multiparty.go

if [[ ! -e "${BACKUP}" ]]; then
  cp -p "${LIBRARY}" "${BACKUP}"
fi
mv "${BUILD_TMP}/liblattigo.so" "${LIBRARY}"
echo "ABI_LIBRARY_SHA256=$(sha256sum "${LIBRARY}" | cut -d' ' -f1)" | tee -a "${environment_log}"

cd "${ROOT}"
LATTI_AI_SOURCE="${ROOT}/.deps/latti-ai"
LATTI_AI_BUILD="${LATTI_AI_SOURCE}/build-h200"
CMAKE_CACHE="${LATTI_AI_BUILD}/CMakeCache.txt"

# A build directory copied from another checkout retains absolute source,
# RPATH, and DT_NEEDED entries. Move it aside and configure a scratch-local
# tree so the runtime cannot silently load a stale library from home storage.
if [[ -f "${CMAKE_CACHE}" ]]; then
  cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CMAKE_CACHE}")"
  if [[ "${cached_source}" != "${LATTI_AI_SOURCE}" ]]; then
    stale_build="${LATTI_AI_BUILD}.stale-${SLURM_JOB_ID:-manual}"
    if [[ -e "${stale_build}" ]]; then
      echo "Refusing to overwrite existing stale-build backup: ${stale_build}" >&2
      exit 2
    fi
    mv "${LATTI_AI_BUILD}" "${stale_build}"
    echo "MOVED_STALE_BUILD=${stale_build}" | tee -a "${environment_log}"
  fi
fi

scripts/apply_latti_source_patches.sh "${LATTI_AI_SOURCE}"

BUILD_DIR="${LATTI_AI_BUILD}" \
LATTISENSE_CUDA_ARCH=90 \
JOBS="${SLURM_CPUS_PER_TASK:-64}" \
  "${LATTI_AI_SOURCE}/scripts/build_h200.sh"

LATTI_AI_SOURCE="${LATTI_AI_SOURCE}" \
LATTI_AI_BUILD="${LATTI_AI_BUILD}" \
  submission/native/build_runtime.sh

NATIVE_RUNTIME="${ROOT}/submission/build/latti_stage_runtime"
python3 scripts/write_build_manifest.py \
  --output "${ROOT}/submission/build/build_manifest.json" \
  --latti-ai-source "${LATTI_AI_SOURCE}" \
  --native-binary "${NATIVE_RUNTIME}" \
  --runtime "${LATTI_AI_BUILD}/inference/lattisense/liblattisense.so" \
  --cuda-architecture sm_90

if readelf -d "${NATIVE_RUNTIME}" | grep -F 'Shared library: [/' | grep -Fq 'liblattigo.so'; then
  echo "Native runtime still contains an absolute liblattigo dependency." >&2
  readelf -d "${NATIVE_RUNTIME}" | grep -F 'liblattigo.so' >&2
  exit 2
fi

resolved_lattigo="$(ldd "${NATIVE_RUNTIME}" | awk '
  /liblattigo\.so/ {
    for (i = 1; i <= NF; i++) {
      if ($i == "=>") { print $(i + 1); exit }
    }
    if ($1 ~ /^\//) { print $1; exit }
  }
')"
if [[ -z "${resolved_lattigo}" ]] || \
   [[ "$(realpath "${resolved_lattigo}")" != "$(realpath "${LIBRARY}")" ]]; then
  echo "Native runtime resolves liblattigo outside scratch: ${resolved_lattigo:-missing}" >&2
  ldd "${NATIVE_RUNTIME}" >&2
  exit 2
fi

{
  echo "NATIVE_RUNTIME_SHA256=$(sha256sum "${NATIVE_RUNTIME}" | cut -d' ' -f1)"
  echo "NATIVE_LIBLATTIGO=${resolved_lattigo}"
  if ! readelf -d "${NATIVE_RUNTIME}" | grep -F 'liblattigo.so'; then
    echo "NATIVE_DIRECT_LIBLATTIGO_NEEDED=none (resolved transitively through libabi.so)"
  fi
} | tee -a "${environment_log}"

python3 submission/test_gpu_runtime.py

export CRYPTOFACE_RUN_OFFSET="${CRYPTOFACE_RUN_OFFSET:-1}"
export CRYPTOFACE_MEASUREDIR="${CRYPTOFACE_MEASUREDIR:-measurements/small-4gpu-fix-validation}"
python3 harness/run_submission.py "${CRYPTOFACE_SIZE:-1}" \
  --seed "${CRYPTOFACE_SEED:-42}" --num_runs 1
