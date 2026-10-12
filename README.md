# CryptoFace GPU

Encrypted face recognition with GPU feature extraction and encrypted pair matching,
using the unchanged [FHE face-recognition harness](https://github.com/fhe-benchmarking/face-recognition/tree/a7c12d34680bdd39198a349cf1ebeec89ae88061).
Raw results and their original source manifests: [single](measurements/single/), [small](measurements/small/).

## Hardware and dependencies

The validated configuration uses **four NVIDIA H200 GPUs with 141 GB each**.
Reserve near-idle GPUs and ample host memory; our validation allocation uses
64 CPU cores and 640 GiB RAM. CUDA architecture remains `sm_90`; smaller GPUs
are not validated. Linux x86-64, Python 3.11 or 3.12, GCC with C++20 support,
CUDA 12.9, Go 1.22+, CMake 3.26.4+, HDF5 C++, NTL, and GMP are required.

The builder fetches these exact Git revisions, then applies the hash-checked
patches and missing build sources in `backend/`:

| Repository | Base revision |
| --- | --- |
| [human-analysis/latti-ai](https://github.com/human-analysis/latti-ai) | `88b4bac84b938f74e6936b19580181f97542e175` |
| [human-analysis/lattisense](https://github.com/human-analysis/lattisense) | `809b35830550370b4570c9d45ae9e33c78a97156` |

Both repositories must be publicly readable for anonymous setup. While private,
Git credentials with read access to both are required. Offline setup can use
`--repository-source /path/to/latti-ai` with its initialized LattiSense submodule.

The **already-packed** server weights are downloaded from the immutable URL in
[external-inputs.json](external-inputs.json). The required SHA-256 is
`dcc3bf9c4f5a5805666b5b46d244d50d0756636a8b0000765158e912b3eaec62`.
Use `--model-input /path/to/model_parameters.h5` for offline setup.
Stage 3 validates and materializes these weights under `io/server_data/`, where
the harness measures their size. Packing and native compilation occur before the
timed harness run; their costs are not included in the reported runtime.

## Install and build

Install NVIDIA's CUDA 12.9 toolkit and a compatible driver first. On Ubuntu
24.04, install the remaining system dependencies:

```bash
sudo bash scripts/install_system_deps.sh
```

Choose an existing writable scratch location outside your home and the source
checkout. Keep the Python environment, builds, data, caches, and runtime artifacts
there. Set the CUDA paths to your actual installation.

```bash
export SOURCE="$PWD"
export SCRATCH=/path/to/fast-scratch/cryptoface
export PATH=/usr/local/cuda-12.9/bin:"$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:"${LD_LIBRARY_PATH:-}"
mkdir -p "$SCRATCH/tmp" "$SCRATCH/cache"
ulimit -c 0
umask 077
export TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/cache"
export PIP_CACHE_DIR="$SCRATCH/cache/pip" PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
python3 -m venv "$SCRATCH/venv"
source "$SCRATCH/venv/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$SOURCE/requirements.txt"
export PYTHON="$SCRATCH/venv/bin/python"
bash "$SOURCE/scripts/build_submission.sh" --scratch-root "$SCRATCH"
```

The setup runs from a hash-verified scratch source copy and builds both the
feature-extraction backend and GPU matching runtime. It downloads the real
benchmark dataset through the unchanged harness and provisions InsightFace
`buffalo_l`. Downloads, model hashes, and checked Python versions are verified
before a run is prepared. A missing standard InsightFace model directory is
symlinked to the scratch copy; an existing directory is verified, never replaced.

The default work directory is `$SCRATCH/reproducible/standalone`. A failed build
must be retried with a fresh `--work "$SCRATCH/reproducible/another-name"`; it
does not silently reuse partial artifacts. Build diagnostics are in
`backend/logs/rebuild.log` and `gpu/build.log` below that directory.
Use `--ntl-root /path/to/sysroot` if NTL is not installed under `/usr`.
The original cluster environment is available with `--cluster-toolchain`;
the default uses installed tools on `PATH` and requires no module system.

## Run the unchanged harness

After setup, activate the prepared scratch environment:

```bash
source "$SCRATCH/reproducible/standalone/activate.sh"
python3 ./harness/run_submission.py 0 --num_runs 3 --seed 42
```

This is the standard harness entry point requested by the benchmark. Do not run
it from the unprepared source checkout. Variant 0 is one real pair; variants
1, 2, and 3 are 128, 256, and 1,024 pairs. Use small before medium, and validate
both before large. The harness performs production face preprocessing and
reports its own quality acceptance result. Never substitute synthetic inputs.

For a matching correctness diagnostic and separately guarded three-run results,
use fresh prepared directories instead of overwriting earlier measurements:

```bash
export WORK="$SCRATCH/reproducible/standalone"
export SOURCE="$WORK/source"
prepare() {
  "$PYTHON" -B "$SOURCE/scripts/benchmark.py" prepare --scratch-root "$SCRATCH" \
    --work "$WORK/$1" --backend-build "$WORK/gpu" \
    --dataset "$WORK/run/source/datasets/face_dataset.h5"
}
run() {
  local name="$1"; shift
  "$PYTHON" -B "$SOURCE/scripts/benchmark.py" run --scratch-root "$SCRATCH" \
    --work "$WORK/$name" "$@"
}
prepare check
run check --size 0 --smoke --matching-check
prepare single
run single --size 0 --matching-validation "$WORK/check/completed.json"
prepare small
run small --size 1 --matching-validation "$WORK/check/completed.json" \
  --previous "$WORK/single/completed.json"
prepare medium
run medium --size 2 --matching-validation "$WORK/check/completed.json" \
  --previous "$WORK/small/completed.json"
```

Stop on any failed check. The matching diagnostic uses the existing `1e-6`
tolerance and is not a performance result. Formal results use three runs; use
fresh directories for repeats and retain all `results-<n>.json` files and
provenance. Setup does not alter the upstream harness or claim new benchmark
measurements. Existing published results remain tied to their original manifests.
Run `python3 -B scripts/summarize_publication.py` to verify their provenance.

## Computation and security

Client preprocessing performs face detection, landmark alignment, and patch
extraction before encryption. The server evaluates the CryptoFace embedding
network and pairwise dot product on CKKS ciphertexts; only the client decrypts
the resulting similarity score. The ArcFace reference is harness-owned and does
not replace encrypted inference. Secret keys remain client-side.

The configuration uses ring dimension 65,536, 32 Q limbs, 3 P limbs, and a sparse
ternary secret of Hamming weight 640. Parameters, key generation, encrypted
operations, and numerical tolerances are unchanged. The documented
[security estimate](submission/latti/validation/SECURITY.md) gives
128.318682 classical bits under its stated estimator model and attack suite.
This is a scoped estimate, not a system-wide side-channel proof.

This repository follows the workload interface and measurement requirements of
the [benchmark guidelines](https://github.com/fhe-benchmarking/fhe-benchmarking.github.io#how-to-submit-solutions).
It remains an independent repository; the requested fork workflow is deferred.
