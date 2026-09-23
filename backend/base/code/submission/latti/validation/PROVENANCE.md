# Provenance

> This file records the historical validated `sm_120` release. The H200
> package deliberately omits its prebuilt libraries and Git bundles. Exact
> source is now pinned through `human-analysis/latti-ai` and
> `human-analysis/lattisense`; see `H200_STATUS.md` for the port status.

## Release lineage

The runtime, trained model weights, dataset, reference embeddings, and
executable derive from the validated secure bundle
`latti-ai-8boot-q44p61-h640-gpu-key-cache-repro` at commit
`4cff63a`. The benchmark task graph was recompiled with embedding output level
2, then transformed with the release's checked Q44/P61 parameter procedure.
This leaves one level for encrypted pair matching and removes the need for any
post-decryption score correction.

The original working source was independently frozen before cleanup at:

`/research/hal-vishnu/external/latti-ai-frozen-source-20260726`

The snapshot contains the same repository SHAs and identical binary Git-diff
hashes for `latti-ai`, LattiSense, HEonGPU, and Lattigo as the active tree at
the time of preservation.

## Exact backend commits

| Component | Commit |
| --- | --- |
| LattiSense | `fbb85da05139af3b6e164fea76ee02f98ace3442` |
| HEonGPU | `aaf76f94bfb2fca2f3b02ff63bea72fb8d4b1680` |
| HEonGPU GPU-NTT | `c6507e99e12b7cf658cda4a2dadedaad7fdee56b` |
| HEonGPU GPU-FFT | `b743607c1173000718ca1f84016fd03093ebb6fc` |
| HEonGPU RNGonGPU | `d9aaa6b5d7cc67a2ca307ef9d6203f25cc2458b4` |
| RNGonGPU GPU-NTT | `32332ce0f367b7e04612e66ae63da3812fb3b8e3` |
| Lattigo | `19d1e4edc162d641ebb6b636beda08496a1a070e` |

Every commit and its ancestry is included under `source/bundles/`. The
`scripts/materialize_sources.sh` gate clones those bundles and verifies every
resulting `HEAD` before reporting success.

## Runtime and immutable input hashes

- `lib/liblattisense.so`:
  `98502f81ceaf814a689260d31ec2860b9691a58599468ca3e41df4d2efc4cb6c`
- `bin/inference_batch`:
  `7610a560eac1cd7a4226ac7d7d5630d91e30c1fb53bb44baa0677ac421d58866`
- `data/lfw_128.bin`:
  `1e8d514acba03e433a70142fbc380432c7c9a468deadb2543cca709ca3a3bdb5`
- `reference/lfw_128_embeddings.npz`:
  `1307762a759ae92c47d74361d2ef685facfae5c839b977e96bf08b8c164eccfe`
- `model/server/mega_ag.json`:
  `ab6393f8d138dbd0b341a07dc1b96e4a697fda31087dd73c81aee3d8df029be8`

`MANIFEST.sha256` covers every tracked release artifact other than the manifest
itself.

GMP is a local dependency rather than a system package. The release includes
`source/deps/gmp/gmp.h`, the applicable GMP license texts, and
`lib/libgmp.so.10`; `scripts/build_runtime.sh` passes both paths explicitly to
CMake.

## Required runtime contract

`run.sh` sets:

```text
LATTISENSE_GPU_KEY_CACHE=1
LATTISENSE_GPU_PIN_REQUEST_H2D=1
LATTISENSE_GPU_EVENT_CACHE=1
LATTISENSE_GPU_BOOTSTRAP_SUBMISSION_LOCK=1
```

It explicitly disables the unsafe whole-operator submission lock and old host
ABI cache. The validated deployment uses one persistent worker per GPU and
shared node-0 CPU affinity `0-31,64-95`.

## Rejected unsafe sibling

Four streams with event reuse but without bootstrap submission locking passed
short tests and later produced one catastrophic embedding per worker. That
runtime is intentionally absent from this minimal release. The included
runtime passed single-pair, 16-pair, 64-pair race-stress, and full 128-pair
gates.
