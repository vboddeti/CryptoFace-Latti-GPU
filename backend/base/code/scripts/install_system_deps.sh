#!/usr/bin/env bash
# Install host build and Python prerequisites. The NVIDIA driver and CUDA
# toolkit are machine-specific and must already provide nvcc for the H200.
set -euo pipefail

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
    SUDO="sudo"
fi

${SUDO} apt-get update
${SUDO} apt-get install -y \
    build-essential \
    cmake \
    git \
    golang-go \
    libgl1 \
    libglib2.0-0 \
    libgmp-dev \
    libhdf5-dev \
    libomp-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    patch \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv

for command_name in cmake git go h5c++ nvidia-smi nvcc patch; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "[install_system_deps] missing required command: ${command_name}" >&2
        echo "Install an NVIDIA CUDA toolkit containing nvcc before building." >&2
        exit 2
    fi
done

echo "[install_system_deps] host prerequisites ready"
echo "[install_system_deps] next: bash scripts/install_python_deps.sh"
