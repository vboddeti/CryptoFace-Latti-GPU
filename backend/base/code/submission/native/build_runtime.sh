#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LATTI_AI_SOURCE="${LATTI_AI_SOURCE:?Set LATTI_AI_SOURCE to the pinned latti-ai checkout}"
LATTI_AI_BUILD="${LATTI_AI_BUILD:-${LATTI_AI_SOURCE}/build-h200}"
H5CXX="${H5CXX:-h5c++}"
OUTPUT_BINARY="${LATTI_STAGE_RUNTIME_OUTPUT:-${ROOT}/submission/build/latti_stage_runtime}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/cryptoface-latti-native.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

RUNTIME_LIBRARY="${LATTI_AI_BUILD}/inference/lattisense/liblattisense.so"
LATTIGO_LIBRARY_DIR="${LATTI_AI_SOURCE}/inference/lattisense/fhe_ops_lib/lattigo/go_sdk"
HEONGPU_LIBRARY_DIR="${LATTI_AI_SOURCE}/inference/lattisense/backends/HEonGPU/install/lib"
for artifact in \
    "${LATTI_AI_BUILD}/inference/inference_task/libinference_task.a" \
    "${LATTI_AI_BUILD}/inference/fhe_layers/libfhe_layers.a" \
    "${LATTI_AI_BUILD}/inference/liblog.a" \
    "${RUNTIME_LIBRARY}" \
    "${LATTIGO_LIBRARY_DIR}/liblattigo.so"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "[build_runtime] missing Latti-AI build artifact: ${artifact}" >&2
        exit 2
    fi
done

mkdir -p "${WORK}/inference/interface" "$(dirname "${OUTPUT_BINARY}")"
cp "${LATTI_AI_SOURCE}/inference/interface/inference_client.h" "${WORK}/inference/interface/"
cp "${LATTI_AI_SOURCE}/inference/interface/inference_client.cpp" "${WORK}/inference/interface/"
cp "${LATTI_AI_SOURCE}/inference/interface/inference_server.h" "${WORK}/inference/interface/"
cp "${LATTI_AI_SOURCE}/inference/interface/inference_server.cpp" "${WORK}/inference/interface/"
patch -d "${WORK}" -p1 < "${ROOT}/submission/native/latti-stage-interface.patch"

COMMON_FLAGS=(
  -maes -fopenmp -O3 -DNDEBUG -std=gnu++20 -fPIC
  -D_GLIBCXX_USE_CXX11_ABI=1 -DH5_BUILT_AS_DYNAMIC_LIB
  -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -D_LARGEFILE_SOURCE
  -D_POSIX_C_SOURCE=200809L
  "-DSOURCE_PATH=\"${LATTI_AI_SOURCE}/inference\""
  -I"${WORK}/inference" -I"${WORK}" -I"${LATTI_AI_SOURCE}/inference"
  -I"${LATTI_AI_SOURCE}/inference/lib"
  -I"${LATTI_AI_SOURCE}/inference/lattisense"
  -I"${LATTI_AI_SOURCE}/inference/lattisense/lib"
)

"${H5CXX}" "${COMMON_FLAGS[@]}" -c "${WORK}/inference/interface/inference_client.cpp" -o "${WORK}/inference_client.o"
"${H5CXX}" "${COMMON_FLAGS[@]}" -c "${WORK}/inference/interface/inference_server.cpp" -o "${WORK}/inference_server.o"
"${H5CXX}" "${COMMON_FLAGS[@]}" -c "${ROOT}/submission/native/latti_stage_runtime.cpp" -o "${WORK}/latti_stage_runtime.o"

"${H5CXX}" -maes -fopenmp -O3 -DNDEBUG \
  "${WORK}/latti_stage_runtime.o" \
  "${WORK}/inference_client.o" \
  "${WORK}/inference_server.o" \
  -o "${OUTPUT_BINARY}" \
  -Wl,-rpath,'\$ORIGIN/../../.deps/latti-ai/build-h200/inference/lattisense' \
  -Wl,-rpath,'\$ORIGIN/../../.deps/latti-ai/inference/lattisense/fhe_ops_lib/lattigo/go_sdk' \
  -Wl,-rpath,'\$ORIGIN/../../.deps/latti-ai/inference/lattisense/backends/HEonGPU/install/lib' \
  -Wl,-rpath-link,"${LATTIGO_LIBRARY_DIR}" \
  -Wl,-rpath-link,"${HEONGPU_LIBRARY_DIR}" \
  -Wl,--start-group \
  "${LATTI_AI_BUILD}/inference/inference_task/libinference_task.a" \
  "${LATTI_AI_BUILD}/inference/fhe_layers/libfhe_layers.a" \
  "${LATTI_AI_BUILD}/inference/liblog.a" \
  -Wl,--end-group \
  "${RUNTIME_LIBRARY}" \
  -Wl,--no-as-needed -L"${LATTIGO_LIBRARY_DIR}" -llattigo -Wl,--as-needed \
  -ldl -lm -lrt -pthread

echo "[build_runtime] built ${OUTPUT_BINARY}"
