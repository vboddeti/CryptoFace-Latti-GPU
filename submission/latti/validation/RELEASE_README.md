# CryptoFace native `sm_120` minimal reproduction

> Historical validation record only. Prebuilt runtime artifacts and source
> bundles described below are intentionally not included in the H200 package.
> The harness package builds the pinned sources locally for `sm_90`.

This is the minimal preserved release of the fastest correctness-validated
native Blackwell CryptoFace runtime. It contains only the runnable application,
model, fixed 128-pair dataset, reference embeddings, required shared libraries,
validation evidence, security report, and exact backend source Git bundles.

## Validated configuration

- CKKS degree 65,536; Q32/P3; default scale `2^44`
- main sparse-secret Hamming weight 640; bootstrap ephemeral weight 32
- eight physical bootstraps per image
- 128.318682 classical security bits
- native CUDA architecture `sm_120`
- four GPU task streams
- bounded persistent cache of 47,509 dependency events
- bootstrap host submission lock; already-enqueued kernels remain concurrent
- GPU evaluation-key cache, pinned request H2D, and shared node-0 affinity

The checked runtime is `lib/liblattisense.so`, SHA-256
`98502f81ceaf814a689260d31ec2860b9691a58599468ca3e41df4d2efc4cb6c`.

## Verify the release

```bash
./scripts/check_artifacts.sh
```

## Install the small Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Required latency gates

Idle single pair on GPU 1:

```bash
PYTHON=.venv/bin/python GPUS=1 PAIRS=1 ./run.sh
```

Sustained 128-pair throughput on GPUs 1, 2, and 3:

```bash
PYTHON=.venv/bin/python GPUS=1,2,3 PAIRS=128 ./run.sh
```

The runner always enables the four required runtime flags and constrains the
process to node-0 CPUs `0-31,64-95`. Override `CPUSET` only when running on a
different topology. `PAIRS` may be from 1 through 128.

Worker setup generates and imports the FHE context once per GPU and can take
about five minutes before measured processing begins. New output is written
under `run/`; preserved validation evidence remains under `results/`.

## Validated measurements

- single pair: 26.949 s processing; 12.873/9.607 s cold/hot core
- 64-pair race stress: 537.745 s; 8.402 s pair interval
- 128 pairs: 1,070.512 s; 8.363 s pair interval; 0.2391 images/s
- full hot core mean: 10.094 s
- full accuracy: relative L2 0.026827; maximum absolute error 0.006033
- all 256 embeddings finite; peak GPU memory 88,056 MiB

A fresh run from this packaged release also passed on 2026-07-26: the single
pair took 27.527 s, and 128 pairs took 1,097.963 s (8.578 s/pair and 0.2332
images/s). All 256 embeddings were finite; relative L2 was 0.026621 and maximum
absolute error was 0.006788. GPU memory was fully released after process exit.

These are matched-session results. The older `sm_86` run remains the lowest
historical absolute measurement at 7.885 s/pair under faster host conditions.

## Recover and rebuild the exact backend source

The source archives are Git bundles, not textual patches. They retain the
complete ancestry of every exact validated backend commit.

```bash
./scripts/materialize_sources.sh
./scripts/build_runtime.sh
```

The rebuild script targets native `sm_120` and leaves its output under `work/`.
It never overwrites the checked runtime. HEonGPU's pinned external CMake
dependencies may require network access on a cold build. GMP is not expected
to be installed system-wide: the required header, licenses, and shared library
are carried in this release and selected explicitly by the build script.

See `PROVENANCE.md`, `VALIDATION.json`, `SECURITY.md`, and `PERFORMANCE.md` for
the exact revisions and validation record.
