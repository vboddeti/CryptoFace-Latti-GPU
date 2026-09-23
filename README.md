# CryptoFace GPU — upstream FHE benchmark submission v1

This is a new submission version, separate from the optimization campaign.
Its `harness/` is byte-identical to `fhe-benchmarking/face-recognition` revision
**a7c12d34680bdd39198a349cf1ebeec89ae88061**, upstream HEAD checked 2026-09-17.
Do not edit it. `release.json` pins all source files and the upstream harness.
No reference monkey-patching or historical CPU-reference patch is applied.

## Status

The backend is the frozen, previously validated `ntt-generic-twiddle-v1`.
The historical **3.732899565 compute seconds/pair** is not a measurement of
this new submission, an official three-run average, or single-request latency.
Publish only new results produced by this version's unchanged harness.

Only three captured submission files change: two stage adapters now use
upstream `params.iodir().parent` instead of the custom `params.io_root()`;
native build paths default to the run directory instead of a home-specific
marker file. GPU arithmetic, encrypted stages, CKKS parameters, weights,
tolerances and client/server key boundaries are unchanged.

The immutable `backend/` archive includes historical custom-harness source and
documentation for build provenance. It is not executed as the benchmark:
only the top-level upstream `harness/` is used.

## Prerequisites

- Four NVIDIA H200 GPUs and the validated Python environment (`requirements.txt`).
- Existing real benchmark `face_dataset.h5`, verified against its pinned hash.
- All five pinned `buffalo_l` ONNX inputs preprovisioned under
  `~/.insightface/models/buffalo_l`; this can be a scratch-backed input symlink.
- Pinned Latti-AI/LattiSense sources, NTL and model weights for a fresh build.

The launcher checks inputs first; it does not request model/dataset downloads.
Private dependency access is currently required, so this is not yet a publicly
self-contained clean-clone distribution. The backend builder uses the existing
H200 cluster module loader; a different toolchain needs separate validation.
Build paths must be below `SCRATCH_ROOT/reproducible/`.

## Verify and build

```bash
python -B scripts/benchmark.py verify
python -B -m unittest discover -s tests
python -B scripts/build_backend.py \
  --scratch-root /path/to/scratch \
  --work /path/to/scratch/reproducible/backend-v1 \
  --repository-source /path/to/pinned/latti-ai \
  --ntl-root /path/to/ntl-sysroot \
  --model-input /path/to/model_parameters.h5
```

A previously verified fresh build of this exact backend may also be used.
The launcher rechecks dependency/source maps, patch provenance and binaries.

## Measure through the official harness

Each workload gets a fresh scratch directory; previous results are never
overwritten. Preparation makes a hash-verified source copy there. Native cwd,
keys, ciphertexts, dataset selections, caches, logs and temporary files stay
in scratch; core dumps and Python bytecode are disabled.

```bash
python -B scripts/benchmark.py prepare \
  --scratch-root /path/to/scratch \
  --work /path/to/scratch/reproducible/v1-small \
  --backend-build /path/to/scratch/reproducible/backend-v1 \
  --dataset /path/to/real/face_dataset.h5
python -B scripts/benchmark.py run \
  --scratch-root /path/to/scratch \
  --work /path/to/scratch/reproducible/v1-small --size 1
```

This invokes `python -B harness/run_submission.py 1 --num_runs 3 --seed 42`
from the scratch copy, with upstream sampling, reference model and acceptance
checks intact. Sizes are single=0 (1 pair), small=1 (128), medium=2 (256),
large=3 (1024). A one-real-pair integration check uses `--size 0 --smoke` and
is explicitly not a complete three-run measurement.

Prepare a new medium directory, then run with its passing small evidence:

```bash
python -B scripts/benchmark.py run \
  --scratch-root /path/to/scratch \
  --work /path/to/scratch/reproducible/v1-medium --size 2 \
  --previous /path/to/scratch/reproducible/v1-small/completed.json
```

Large requires passing medium. Source, dataset, settings and backend identities
must match across gates. `scripts/run_slurm.sh` provides prepare + run in a new
job-specific directory; supply its documented `FHE_*` variables and appropriate
cluster directives. No existing pending jobs are modified.
Pin `FHE_RELEASE_SHA256` to `sha256sum release.json` when submitting; the queued
launcher refuses a changed release rather than silently running new code.

## Collect and publish

```bash
python -B scripts/benchmark.py collect \
  --scratch-root /path/to/scratch \
  --work /path/to/scratch/reproducible/v1-small \
  --output /path/to/code/measurements/gpu-v1-small-runset
```

Only final result JSONs, run provenance and release identity are collected.
Report all three runs and their average, hardware and full harness metrics;
harness total, server total and encrypted compute are distinct.

Official submission also requires an explanation of at least 128-bit security,
public source or an accepted access arrangement, and organizer review. Reusing
unchanged parameters is not a fresh security estimate. The historical
intermittent Go-heap corruption remains a reliability caveat. Do not claim
official acceptance merely because this version runs.

## Future backend versions

1. Copy this release to a new version; retain the upstream `harness/` pin.
2. Update backend/adapter code and review security/correctness. Regenerate the
   release ID and source hashes deliberately; do not edit historical versions.
3. Run real-image smoke, small, medium, then optionally large.
4. Retain three new results per submitted batch size with code, build and input
   identities. Never relabel previous measurements as belonging to new code.
5. Commit qualified code/results and update the benchmark repository link.

An upstream harness upgrade is a separately pinned change requiring comparison
and fresh measurements. Never patch the harness to accelerate one backend.
