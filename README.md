# CryptoFace GPU

Encrypted face recognition with GPU feature extraction and encrypted pair matching,
using the unchanged [FHE face-recognition harness](https://github.com/fhe-benchmarking/face-recognition/tree/a7c12d34680bdd39198a349cf1ebeec89ae88061).

## Run through the benchmark

Install the benchmark's requirements and run `python3 ./harness/run_submission.py 0`
from this repository's root.
The first submission stage automatically provisions its dependencies, discovers
or installs CUDA 12.9, downloads the pinned weights and backend sources, and
builds the runtime. Later stages and runs reuse the private cache. No manual
scratch setup or environment activation is required. The host must already have
a compatible NVIDIA driver and supported GPU; first setup requires internet.
See [submission/readme.md](submission/readme.md) for the complete instructions.

Automatic setup is included in the first Stage 2 timing. The harness itself is
unchanged.

## Hardware and dependencies

Supported GPUs are **NVIDIA H200 GPUs with 141 GB each**, compute capability 9.0,
and at least 140,000 MiB of free device memory. For a four-GPU run, recommended
host resources are 64 CPU cores and 640 GiB RAM. The native CUDA target is `sm_90`;
other GPU architectures are not validated. The bootstrap provisions its private
Python and native build dependencies on Linux x86-64 (glibc 2.28 or newer).
The host must have an NVIDIA driver that supports CUDA 12.9.

The builder fetches these exact Git revisions, then applies the hash-checked
patches and missing build sources from the immutable recipe in
[submission/bootstrap.json](submission/bootstrap.json):

| Repository | Base revision |
| --- | --- |
| [human-analysis/latti-ai](https://github.com/human-analysis/latti-ai) | `88b4bac84b938f74e6936b19580181f97542e175` |
| [human-analysis/lattisense](https://github.com/human-analysis/lattisense) | `809b35830550370b4570c9d45ae9e33c78a97156` |

Both repositories are publicly readable for anonymous setup.

The **already-packed** server weights are downloaded from the immutable URL in
[submission/bootstrap.json](submission/bootstrap.json). The required SHA-256 is
`dcc3bf9c4f5a5805666b5b46d244d50d0756636a8b0000765158e912b3eaec62`.
Stage 3 validates and materializes these weights under `io/server_data/`, where
the harness measures their size. Native compilation is automatic during the
first Stage 2 invocation and is included in that invocation's measured time.

## Computation and security

Client preprocessing performs face detection, landmark alignment, and patch
extraction before encryption. The server evaluates the CryptoFace embedding
network and pairwise dot product on CKKS ciphertexts; only the client decrypts
the resulting similarity score. The ArcFace reference is harness-owned and does
not replace encrypted inference. Secret keys remain client-side.

The configuration uses ring dimension 65,536, 32 Q limbs, 3 P limbs, and a sparse
ternary secret of Hamming weight 640. The documented
[security estimate](submission/latti/validation/SECURITY.md) gives
128.318682 classical bits under its stated estimator model and attack suite.
This is a scoped estimate, not a system-wide side-channel proof.

This repository follows the workload interface and measurement requirements of
the [benchmark guidelines](https://github.com/fhe-benchmarking/fhe-benchmarking.github.io#how-to-submit-solutions).
