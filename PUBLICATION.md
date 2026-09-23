# CryptoFace GPU face-recognition results

CKKS-encrypted face recognition on four NVIDIA H200 GPUs, using the unmodified
FHE face-recognition harness at commit
`a7c12d34680bdd39198a349cf1ebeec89ae88061`.

## Results

| Variant | Pairs/run | Runs | Encrypted compute/run | Compute/pair | Harness total/run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 128 | 3 | 483.501942 s | 3.777359 s | 4287.0109 s |

Small is the arithmetic mean of three runs using `--num_runs 3 --seed 42`.
All three runs pass the upstream EER acceptance gate. This submission reports
only small; single, medium and large results are not included.

Compute/pair is amortized encrypted computation, not end-to-end request latency.
Harness totals retain setup, preprocessing, encryption, decryption and CPU ArcFace
reference evaluation. No overhead is subtracted. The website's Compute columns
contain full-workload times; Decrypt combines decryption and postprocessing.

| Quality: mean across runs | Small |
| --- | ---: |
| Encrypted EER | 0.068947 |
| ArcFace EER | 0.063099 |
| Encrypted TAR@FAR=1% | 0.844327 |
| ArcFace TAR@FAR=1% | 0.936901 |

Hardware: four H200 GPUs, 143771 MiB reported per GPU, driver 580.173.02.
The small allocation used 64 CPUs and 640 GiB host RAM. These are not
matched-hardware measurements against the website's CPU reference.

## Source and reproduction

- [Small raw results, provenance and release manifest](measurements/small/).
- [Full measurement summary](measurements/publication-summary.json).
- [Build and execution instructions](README.md).
- Verify source with `python3 -B scripts/benchmark.py verify`; recompute the
  summary with `python3 -B scripts/summarize_publication.py`.

Four persistent GPU workers compute encrypted embeddings. One CPU worker performs
encrypted pair matching concurrently with inference, using only public evaluation
material. Public evaluation-key caching, rotation hoisting and public NTT-twiddle
optimizations accelerate inference. Client secret keys remain on the client path.

Reproduction requires the pinned Latti-AI/LattiSense sources, the H200 toolchain,
NTL, model weights, real face-pair data and pinned InsightFace models. Dependency
and input hashes are recorded in the manifests. Private backend access is required;
an approved reviewer-access procedure must be in place before publication.
This is not yet a self-contained public clean-clone distribution.

Run native work from a fresh hash-verified scratch copy, with core dumps disabled
and temporary files/caches in scratch. Only code and final JSON evidence are
retained in this repository.

## Security and limitations

The [security justification](submission/latti/validation/SECURITY.md) and
[estimator report](submission/latti/validation/security/estimator_report.json)
give a classical LWE estimate of 128.318682 bits under the specified MATZOV/GSA
model. Parameters include N=65536, 32 Q limbs plus 3 P limbs, Gaussian error
sigma=3.2 and sparse ternary secret weight 640. The estimate is not a proof of
the complete bootstrap protocol or implementation; bootstrap ephemeral secret
weight 32 and the narrow security margin require review.

To reproduce the estimate, run the included estimator script with the pinned
lattice-estimator checkout and explicitly supply
`--parameter submission/latti/model/server/ckks_parameter.json`.
An intermittent Go-heap-corruption issue remains unresolved; the reported runs
all completed successfully. Numerical tests do not establish system-wide
security or reliability.
