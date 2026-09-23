# Validated native Blackwell `sm_120` release

Date: 2026-07-25

## Decision

Validated under matched-session single-pair, bounded, race-stress, and full
128-pair gates. Native `sm_120` now beats the validated `sm_86` runtime when
both are measured in the same session. The older historical `sm_86` run remains
the lowest absolute throughput measurement because host conditions were faster
then; do not mix the two control sessions.

## Configuration

- LattiSense base repair: `cb88bdb`
- HEonGPU: `aaf76f9`
- GPU-NTT Blackwell arithmetic fix: `c6507e9`
- Degree 65,536; Q32/P3; main H=640; bootstrap H=32
- Eight physical bootstraps per image
- Classical security: **128.318682 bits**, unchanged
- Four GPU task streams instead of the one-stream correctness fallback
- Persistent, task-scoped cache of 47,509 dependency events; the cache is
  bounded by graph output count and destroyed with the persistent runtime
- Serialize only host submission of bootstrap nodes. Already-enqueued CUDA
  work remains concurrent across all four streams.
- GPU key cache, pinned request H2D, and shared node-0 CPU affinity remain
  enabled and unchanged.

Required runtime flags:

```bash
LATTISENSE_GPU_KEY_CACHE=1
LATTISENSE_GPU_PIN_REQUEST_H2D=1
LATTISENSE_GPU_EVENT_CACHE=1
LATTISENSE_GPU_BOOTSTRAP_SUBMISSION_LOCK=1
taskset -c 0-31,64-95 ...
```

## Results

### Idle single pair

| Metric | Native candidate | Historical `sm_86` | Change |
| --- | ---: | ---: | ---: |
| Pair processing | 26.949 s | 27.347 s | -1.46% |
| Cold core | 12.873 s | 14.616 s | -11.93% |
| Hot core | 9.607 s | 9.471 s | +1.44% |

Native accuracy passed: relative-L2 0.020821 and maximum absolute error
0.003843.

### Matched-session 16-pair gate

| Metric | Native candidate | `sm_86` control | Change |
| --- | ---: | ---: | ---: |
| Processing | 378.642 s | 398.116 s | -4.89% |
| Overall core mean | 9.675 s | 10.301 s | -6.08% |
| Hot core mean | 9.568 s | 10.126 s | -5.51% |
| Images/s | 0.0845 | 0.0804 | +5.14% |

Native accuracy passed: relative-L2 0.024227 and maximum absolute error
0.005345.

### Multi-GPU race stress

The 64-pair/128-image, three-GPU stress passed after every worker reached
42-43 local requests. Processing was 537.745 s, interval 8.402 s/pair, hot core
10.104 s, relative-L2 0.026622, and maximum absolute error 0.005568.

### Matched-session 128-pair gate

| Metric | Native candidate | `sm_86` control | Change |
| --- | ---: | ---: | ---: |
| Processing | 1,070.512 s | 1,098.510 s | -2.55% |
| Pair interval | 8.363 s | 8.582 s | -2.55% |
| Images/s | 0.2391 | 0.2330 | +2.62% |
| Overall core mean / p50 / p95 | 10.159 / 10.099 / 10.580 s | 10.512 / 10.348 / 11.306 s | mean -3.36% |
| Hot core mean / p50 / p95 | 10.094 / 10.098 / 10.539 s | 10.419 / 10.344 / 11.176 s | mean -3.12% |

Native accuracy passed for all 256 embeddings: relative-L2 0.026827 and
maximum absolute error 0.006033. The matched `sm_86` control also passed.

Mean GPU utilization was 69.92% / 69.20% / 67.72% for native versus
65.45% / 62.55% / 64.07% for `sm_86`. Native peak memory was 88,056 MiB,
96 MiB above the matched control. All workers returned to idle memory.

The historical `sm_86` result remains 1,009.239 s processing, 7.885 s pair
interval, and 10.039 s hot core. The matched-session controls demonstrate that
the current host session is slower; the native candidate is a code-path win,
but it does not replace that historical absolute record.

## Race finding and rejected arms

Four streams plus event reuse without bootstrap submission locking was fast
but unsafe. The full gate produced exactly three catastrophic embeddings, one
per worker, at local request ordinals 29, 44, and 46. Relative-L2 reached
`3.09e19`. Locking every operator submission fixed the pair numerically but
regressed it to 30.589 s and 14.730/11.323 s cold/hot. Locking only bootstrap
submission retained performance and passed the 64- and 128-pair gates.

## Evidence

- Runtime SHA-256: `98502f81ceaf814a689260d31ec2860b9691a58599468ca3e41df4d2efc4cb6c`
- Fresh pair metrics: `results/release_pair.json`
  (SHA-256 `b513ee20f77428760a728aeb2f3e4fd1189b1dd1b47e05711a790fdcc678397f`)
- Fresh full-run metrics: `results/release_128pairs.json`
  (SHA-256 `c282d6ecfe32dee841c917a8349b6e919ada65924d9778a3ed168b95bdd85a96`)
- Corresponding embeddings and validation reports are retained beside those
  files. Rejected experimental runtimes and their debug output are
  intentionally not included.
