# CryptoFace GPU submission

This directory contains the GPU submission for the face-recognition benchmark.
With the benchmark's own Python requirements installed, run the harness from
the repository root:

```bash
python3 ./harness/run_submission.py 0
```

No submission-specific setup command, virtual-environment activation, scratch
directory selection, or environment variables are required. The first submission
stage automatically installs its private Python/build environment, downloads the
checksum-pinned packed weights and source recipe, clones the pinned public
Latti-AI/LattiSense revisions, and compiles the GPU runtime. Later stages and runs
reuse the completed cache. Concurrent first runs share a setup lock; a failed
build is retained for diagnosis and a later invocation retries in a new directory.

Setup discovers CUDA 12.9 through `CUDACXX`, `CUDA_HOME`, `CUDA_PATH`, `PATH`, and
standard installation directories. If unavailable, it installs the matching
user-space toolkit automatically. It never installs or changes the NVIDIA driver
and requires no administrator privileges. Downloads require internet access on
the first run. Setup progress identifies the detailed build log.

The supported platform is Linux x86-64 (glibc 2.28 or newer) with an NVIDIA H200, compute
capability 9.0, and at least 140,000 MiB free device memory. The driver must support
CUDA 12.9. GPUs are discovered automatically; only eligible idle GPUs are used.
For a four-GPU run, recommended host resources are 64 CPU cores and 640 GiB RAM.
Allow ample disk space for the encrypted evaluation context and the benchmark's
normal `io/` outputs. Other GPU architectures are not validated.

The automatic cache is private to the current user beneath the system temporary
directory. `TMPDIR` is honored; `CRYPTOFACE_CACHE_DIR` is an optional override,
not a prerequisite. Core dumps are disabled and native processes run from the
cache. Benchmark inputs, outputs, and measurements retain the harness's paths.
The first Stage 2 invocation includes setup time in the harness's stage timing.

The packed model is fixed trained data, independent of the benchmark inputs and
generated keys. Stage 3 materializes it in the harness's server-data directory so
the harness measures its size. Keys are generated afresh for every harness run;
the shared build cache does not retain client keys or ciphertexts.

Client preprocessing detects and aligns faces and extracts patches before
encryption. The server evaluates the embedding network and pairwise dot product
on CKKS ciphertexts; only the client decrypts the similarity score. The harness's
ArcFace reference remains separate from encrypted inference.

The build recipe and source revisions are pinned in `bootstrap.json`. See
`latti/validation/SECURITY.md` for the CKKS parameters and security estimate.
Single-pair runs check functionality; use the benchmark's small and medium
workloads for dataset-level quality assessment.
