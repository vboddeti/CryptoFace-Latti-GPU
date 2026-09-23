#!/usr/bin/env python3
"""Stage 6: encrypt aligned CryptoFace patches with the client context."""

from __future__ import annotations

import json
import time

import h5py

from common import pair_stem, parse_stage_args, release_file_cache
from gpu_runtime import (
    NativeSession,
    atomic_write_json,
    client_command,
    require_runtime,
    runtime_environment,
)


def main() -> None:
    _size, cfg, params = parse_stage_args(resolve_checkpoint=False)
    require_runtime(cfg)
    secret_context = params.iodir() / "client_secret" / "secret_context.bin"
    if not secret_context.is_file():
        raise FileNotFoundError(f"Missing client secret context: {secret_context}")

    patch_path = params.io_intermediate_dir() / "preprocessed_patches.h5"
    if not patch_path.is_file():
        raise FileNotFoundError(f"Missing preprocessed patches: {patch_path}")
    upload_dir = params.iodir() / "ciphertexts_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    log_path = params.iodir() / "logs" / "client_encrypt.log"
    pair_count = params.get_batch_size()
    input_names = cfg["input_names"]
    encrypted_images = 0
    started = time.monotonic()

    session = NativeSession(
        client_command(secret_context),
        env=runtime_environment(),
        log_path=log_path,
        ready_marker="[StageClient] Ready.",
    )
    try:
        session.start()
        with h5py.File(patch_path, "r") as patch_store:
            patches = patch_store["patches"]
            expected = (pair_count, 2, len(input_names), 1, 3, 32, 32)
            if patches.shape != expected:
                raise ValueError(f"Unexpected patch tensor shape {patches.shape}; expected {expected}")
            for pair_index in range(pair_count):
                for image_index in range(2):
                    request_id = f"{pair_stem(pair_index)}_i{image_index}"
                    inputs = {
                        name: patches[pair_index, image_index, branch]
                        .astype("float64", copy=False)
                        .reshape(-1)
                        .tolist()
                        for branch, name in enumerate(input_names)
                    }
                    response = session.request(
                        {
                            "op": "encrypt",
                            "id": request_id,
                            "inputs": inputs,
                            "output_dir": str(upload_dir),
                        }
                    )
                    if sorted(response["outputs"]) != sorted(input_names):
                        raise RuntimeError(f"Incomplete encryption response for {request_id}")
                    encrypted_images += 1
                    print(
                        f"[client_encode_encrypt_input] {encrypted_images}/{2 * pair_count} images",
                        flush=True,
                    )
    finally:
        session.close()

    release_file_cache(patch_path)
    patch_path.unlink()
    encrypted_bytes = sum(path.stat().st_size for path in upload_dir.glob("*.ct"))
    report_path = params.iodir() / "submission_reported.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "schema_version": 1,
        "Bandwidth": {},
    }
    report["Bandwidth"]["Encrypted inputs"] = encrypted_bytes
    atomic_write_json(report_path, report)
    print(
        f"[client_encode_encrypt_input] Encrypted {encrypted_images} images in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
