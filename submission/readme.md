# CryptoFace Latti source-built H200 GPU submission

This submission evaluates the 64x64 four-patch CryptoFace model with the
source-built `sm_90` LattiSense runtime. It follows the benchmark's
stage-by-stage client/server contract: stages 3 and 7 load only the public
evaluation context, encrypted inputs, and server model. The client secret
context is generated in stage 2 and loaded only by stages 6 and 8.

## Cryptographic configuration

- CKKS polynomial modulus degree: 65,536
- Q/P basis: 32 Q limbs and 3 P limbs
- main sparse-secret Hamming weight: 640
- bootstrap ephemeral Hamming weight: 32
- eight physical bootstraps per image
- estimated classical security: 128.318682 bits
- native CUDA architecture: `sm_90`

## Hardware

The submission automatically uses every NVIDIA GPU reporting compute capability
9.0 and at least 90 GiB of free memory through `nvidia-smi`. The historical
Blackwell worker peaked near 88,056 MiB, so one H200 worker fits within a 142 GiB
device. Set `CRYPTOFACE_GPUS` to a comma-separated list only when an explicit
compatible subset is required.

## Stage behavior

1. `client_key_generation` generates and serializes the private client context
   and a public-only evaluation context.
2. `server_preprocess_model` validates the locally built runtime and copies the
   server-owned packed model into the harness's persistent `io/server_data`.
3. `client_preprocess_input` performs InsightFace detection, landmark
   alignment, 64x64 normalization, and four-way 32x32 patch extraction.
4. `client_encode_encrypt_input` encrypts patches in one persistent client
   process.
5. `server_encrypted_compute` starts one persistent worker per GPU, evaluates
   both images, and homomorphically computes the 256-dimensional inner product.
   The normalized embedding leaves the model at CKKS level 2; ciphertext
   multiplication and rescaling leave the scalar score at level 1. The server
   never loads or receives the secret context.
6. `client_decrypt_decode` decrypts only the scalar similarity scores and
   writes the raw decoded values. No score offset or post-decryption calibration
   is applied.

GPU worker setup generates GPU-resident evaluation-key state and may take five
to seven minutes. The server report separates setup, encrypted embedding
inference, encrypted matching, and total stage lifetime.

## Local validation

```bash
bash scripts/install_system_deps.sh
bash scripts/install_python_deps.sh
bash scripts/build_submission.sh
source .venv/bin/activate
CRYPTOFACE_GPUS=0 python3 harness/run_submission.py 0 --seed 42
```

The 128-, 256-, and 1,024-pair instances should be run only after the
single-pair stage-isolation and numerical checks pass.

## Provenance

The trained model, task graph, security reports, and preserved Blackwell
validation come from the promoted
`latti-ai-cryptoface-minimal-sm120` release at commit
`215c30a3a0ba8213c10aa1fed7c9eac437ef763f`. The benchmark task graph was
recompiled to expose level-2 embeddings and then converted with the same
checked Q44/P61 procedure used by that release. Latti-AI and LattiSense are
instead checked out from the exact `human-analysis` commits in `config.yml`,
built for `sm_90`, and recorded in a local build manifest. The H200 result is
not considered validated until it passes the gates in `H200_STATUS.md`.
