#!/usr/bin/env python3
"""Stage 8: decrypt only the encrypted pair scores."""

from __future__ import annotations

import time
import json
import math
import os

from common import pair_stem, parse_stage_args
from gpu_runtime import NativeSession, client_command, require_runtime, runtime_environment


def main() -> None:
    _size, cfg, params = parse_stage_args(resolve_checkpoint=False)
    require_runtime(cfg)
    secret_context = params.iodir() / "client_secret" / "secret_context.bin"
    if not secret_context.is_file():
        raise FileNotFoundError(f"Missing client secret context: {secret_context}")
    download_dir = params.iodir() / "ciphertexts_download"
    output_path = params.get_encrypted_model_predictions_file()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".txt.tmp")
    pair_count = params.get_batch_size()
    started = time.monotonic()
    verify_matching = os.environ.get("CRYPTOFACE_VERIFY_GPU_MATCHING") == "1"
    max_matching_error = 0.0

    session = NativeSession(
        client_command(secret_context),
        env=runtime_environment(),
        log_path=params.iodir() / "logs" / "client_decrypt.log",
        ready_marker="[StageClient] Ready.",
    )
    try:
        session.start()
        with temporary.open("w", encoding="utf-8") as output:
            for pair_index in range(pair_count):
                score_path = download_dir / f"{pair_stem(pair_index)}_score.ct"
                if not score_path.is_file():
                    raise FileNotFoundError(f"Missing encrypted score: {score_path}")
                response = session.request(
                    {
                        "op": "decrypt",
                        "id": pair_stem(pair_index),
                        "input": str(score_path),
                    }
                )
                values = response.get("output")
                if not isinstance(values, list) or not values:
                    raise RuntimeError(f"Invalid decrypted output for pair {pair_index}")
                score = float(values[0])
                if not math.isfinite(score):
                    raise RuntimeError("Non-finite decrypted score")
                if verify_matching:
                    reference_path = str(score_path) + ".cpu-reference.ct"
                    reference = session.request({"op": "decrypt", "id": f"reference_{pair_index}",
                                                 "input": reference_path})
                    reference_score = float(reference["output"][0])
                    error = abs(score - reference_score)
                    if not math.isfinite(reference_score) or error > 1e-6:
                        raise RuntimeError(f"GPU/CPU matching correctness failed: {error}")
                    max_matching_error = max(max_matching_error, error)
                output.write(f"{score:.9f}\n")
    finally:
        session.close()
    temporary.replace(output_path)
    if verify_matching:
        (params.iodir() / "gpu-matching-validation.json").write_text(json.dumps({
            "passed": True, "pairs": pair_count, "absolute_tolerance": 1e-6,
            "max_absolute_error": max_matching_error,
            "reference": "unchanged encrypted CPU inner product on identical ciphertexts",
        }, indent=2) + "\n")
    print(
        f"[client_decrypt_decode] Decrypted {pair_count} scores in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
