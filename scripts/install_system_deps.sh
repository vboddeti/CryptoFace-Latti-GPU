#!/usr/bin/env bash
# Ubuntu 24.04 prerequisites. Install NVIDIA's CUDA 12.9 toolkit separately.
set -euo pipefail
if (( EUID != 0 )); then
  echo 'Run with sudo: bash scripts/install_system_deps.sh' >&2
  exit 2
fi
apt-get update
apt-get install -y build-essential git patch cmake ninja-build pkg-config \
  golang-go libntl-dev libgmp-dev libhdf5-dev libgraphviz-dev \
  python3-dev python3-venv libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
