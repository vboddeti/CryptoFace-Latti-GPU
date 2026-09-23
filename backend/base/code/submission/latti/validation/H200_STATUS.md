# H200 source-build status

The harness adapter, packed CryptoFace model, native interface patch, and exact
Latti-AI/LattiSense source revisions are packaged for a clean H200 build. The
build targets CUDA architecture `sm_90` and records hashes of the resulting
native executable and `liblattisense.so` in
`submission/build/build_manifest.json`.

The preserved numerical, security, race-stress, and throughput evidence in
this directory was collected with the prior `sm_120` Blackwell build. It is
useful as a regression oracle, but it does not establish that the H200 build is
correct.

Promote the H200 build only after these gates pass, in order:

1. `bash scripts/build_submission.sh` completes and `cuobjdump` reports `sm_90`.
2. `python3 harness/run_submission.py 0 --seed 42` completes with a finite score.
3. The decrypted 256-element embeddings match the preserved oracle within the
   existing maximum relative-L2 threshold of `0.03`.
4. Repeated single-pair runs remain finite and release GPU memory after exit.
5. The 128-pair harness run completes and passes the ArcFace quality gate.

The adapter uses one persistent native worker per selected GPU. Validation on
this host observed roughly 129 GiB per worker, so automatic discovery selects
only near-idle devices. Multi-GPU use is available through `CRYPTOFACE_GPUS`
after the single-device gate passes.

## Three-H200 streamed baseline (2026-08-29)

The direct 128-pair run in
`measurements/local-three-gpu-streaming-20260829/results-1.json` passed the
ArcFace comparison gate with three persistent H200 workers. It used the GPU
key cache, disabled the regressing dependency-event cache, and streamed public-
context encrypted scoring while later embeddings were still running.

- Encrypted inference plus exposed matching: 1,595.218 seconds, or 12.463
  seconds/pair amortized.
- Server wall time including 142.090 seconds of one-time worker setup:
  1,760.710 seconds, or 13.756 seconds/pair.
- Mean native `run_fhe_gpu_task` time: 18.624 seconds/image across all 256
  images. Wrapper/export phases rounded to zero at the runtime's timer
  resolution.
- Matching overlap: 1,567.144 of 1,568.478 seconds (99.915%), leaving only
  1.334 seconds of matching exposed over the complete run.
- The comparable preserved three-worker native Blackwell reference is
  1,070.512 seconds, or 8.363 seconds/pair. The H200 encrypted-compute
  interval is therefore 49.01% slower; the remaining gap is inside native GPU
  inference rather than the streaming scheduler.

The current performance-critical GPU wrapper/task sources otherwise match the
published `vboddeti/temp` bundle. The build is Release `-O3` for `sm_90`; the
material source deltas are the validated safe Go/C ABI ownership fix and the
plaintext cache described below.

## H200 profiler diagnosis and safe plaintext-cache fix (2026-08-29)

The live `CUDA/12.9.1` module provides Nsight Systems 2025.1.3 and Nsight
Compute 2025.2.1. A complete request in
`measurements/nsys-2025-temp-direct-gpu3-20260829/trace.nsys-rep` contained
340,432 kernel launches, 11.893 seconds of summed kernel time, a 12.205-second
GPU activity span, and 82.1% union GPU activity. NTT kernels consumed 66.3% of
kernel time, base conversion 11.0%, divide/rescale 9.7%, and key switching
6.1%. The request also issued 62,605 H2D copies totaling 29.9 GiB and 0.740
seconds.

The regression was in the safe Go/C ABI export fix, not the scheduler,
serialization, cache flags, or H200 hardware. Every
`ExportCkksPlaintextMul` call performed `CopyNew()` and then a second C export
copy. Tens of thousands of immutable model-plaintext operations per image made
that safety copy dominant. The fix retains safe owned memory and caches each
transformed plaintext by parameter handle, plaintext handle, and `mf_nbits`.
`ReleaseHandle` invalidates associated cache entries.

The definitive local 128-pair run on physical H200 GPUs 1, 2, and 3 is recorded
in `measurements/local-three-gpu-safe-plaintext-cache-20260829/results-1.json`:

- Server encrypted compute: 935.859 seconds, or 7.311 seconds/pair. This is
  41.33% faster than the broken 12.463 seconds/pair baseline and 12.58% faster
  than the preserved 8.363 seconds/pair native Blackwell reference.
- Full harness encrypted-computation interval: 1,100.792 seconds, or 8.600
  seconds/pair. Server wall time including worker setup: 1,098.385 seconds, or
  8.581 seconds/pair.
- Native hot mean after excluding one cold request per worker: 10.802
  seconds/image across 253 images, implying 7.201 seconds/pair on three GPUs.
- All 256 embeddings and 128 scores completed. The encrypted model passed the
  ArcFace comparison gate: EER 0.07018, EER gap 0.01754 against the harness
  model, below the maximum allowed gap of 0.15.

The rebuilt native library is Release `sm_90`, has SHA256
`199fe2dc5f2c54f38c7a30e4f0c31b88bee7fa3250272eea5814534b0336de40`, and is
recorded in `submission/build/build_manifest.json`.
