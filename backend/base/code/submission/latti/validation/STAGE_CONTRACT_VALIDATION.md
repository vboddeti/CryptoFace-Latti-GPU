# Stage-contract validation

The corrected packaged GPU adapter completed the full single-pair benchmark
contract on 2026-08-15 with GPU 1, an NVIDIA RTX PRO 6000 Blackwell `sm_120`
device.
The validated executable has SHA-256
`d4631e867eb22dd66c17ca9ff1acc88e58a27be39cdef8e6a065bcdeb930af40`.

Key generation, public-only server restoration, encrypted input handling, two
encrypted model evaluations, homomorphic pair matching, scalar-only client
decryption, and the harness quality check all passed. The raw genuine-pair
score was 0.529363157.

The key stage serialized an 8,207,375,531-byte client context and an
8,193,080,444-byte public evaluation context. The server command used only the
public artifact. Its persistent worker setup took 323.04 seconds; two encrypted
embeddings took 55.66 seconds and encrypted pair matching took 3.91 seconds.
The full cold harness run took 1,483.14 seconds.

## Normalization and scalar check

A separate diagnostic retained the two encrypted embeddings long enough to
decrypt them on the client. They had 256 elements, norms 0.997752 and 1.007934,
and a direct dot product of 0.529019. These norms reflect the model's existing
polynomial L2-normalization approximation; the GPU task uses the same
coefficients as the Orion model.

The prior task exposed normalized embeddings at CKKS level 1. Ciphertext
multiplication and rescaling therefore placed the scalar reduction at level 0,
where insufficient modulus headroom caused a one-period decoding wrap. Adding
2048 after decryption masked that circuit-level error and has been removed.

The corrected task exposes embeddings at level 2, so matching finishes at level
1. With no score correction, the encrypted pair score decoded to 0.529363157,
an absolute error of about 0.000344 from the direct embedding dot product.
No embedding is exposed in the benchmark path.

Exact hashes, timings, bandwidth, and diagnostic values are recorded in
`results/stage_contract_single.json`. Batched EER/TAR accuracy will be
regenerated with the corrected task; it is not claimed by this smoke test.
