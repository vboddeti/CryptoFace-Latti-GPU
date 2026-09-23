#!/usr/bin/env bash
# Shared module environment for H200 builds, validation, and Slurm jobs on
# dev-amd24-h200. Source this file before activating the project virtualenv.

CRYPTOFACE_GPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRYPTOFACE_NTL_ROOT="${CRYPTOFACE_GPU_ROOT}/.deps/ntl-sysroot/usr"

module purge
module load foss/2023a
module load CMake/3.31.8-GCCcore-12.3.0
module load CUDA/12.9.1
module load Go/1.22.1
module load GMP/6.2.1-GCCcore-12.3.0
module load HDF5/1.14.0-gompi-2023a
module load Ninja/1.11.1-GCCcore-12.3.0
module load Graphviz/8.1.0-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0

# The login environment's global Python packages leak into uv build isolation
# and break pygraphviz/Orion builds. Project virtualenvs provide their own
# packages, so keep the global path out of build and job processes.
unset PYTHONPATH

# InsightFace's ONNX Runtime otherwise creates a worker for nearly every CPU
# visible on this 192-core host. Besides oversubscribing Slurm allocations,
# that exhausts the interactive shell's aggregate CPU-time limit. Both the
# encrypted-input preprocessor and the ArcFace quality baseline honor this.
export CRYPTOFACE_FACE_CPU_THREADS="${CRYPTOFACE_FACE_CPU_THREADS:-8}"

# NTL is required by the preserved HEonGPU CKKS sources, but this cluster does
# not provide an NTL module. The shared local prefix is populated once from the
# pinned Ubuntu 22.04 libntl-dev/libntl44/libgf2x3 packages.
if [[ ! -f "${CRYPTOFACE_NTL_ROOT}/include/NTL/RR.h" ]]; then
    echo "Missing local NTL prefix: ${CRYPTOFACE_NTL_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi
export CPATH="${CRYPTOFACE_NTL_ROOT}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CRYPTOFACE_NTL_ROOT}/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CRYPTOFACE_NTL_ROOT}/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
