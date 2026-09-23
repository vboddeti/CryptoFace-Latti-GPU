#!/usr/bin/env python3
"""Stage 2: generate client secret material and public evaluation material."""

from __future__ import annotations

import subprocess
import time

from common import parse_stage_args
from gpu_runtime import (
    MODEL_DIR,
    NATIVE_BINARY,
    atomic_write_json,
    require_runtime,
    runtime_environment,
    sha256_file,
)


def main() -> None:
    size, cfg, params = parse_stage_args(resolve_checkpoint=False)
    require_runtime(cfg)
    started = time.monotonic()
    params.iodir().mkdir(parents=True, exist_ok=True)
    params.io_root().mkdir(parents=True, exist_ok=True)
    (params.io_root() / "current_size.txt").write_text(f"{size}\n")

    secret_dir = params.iodir() / "client_secret"
    public_dir = params.iodir() / "public_keys"
    secret_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    secret_context = secret_dir / "secret_context.bin"
    eval_context = public_dir / "eval_context.bin"

    command = [
        str(NATIVE_BINARY), "keygen",
        "--task-dir", str(MODEL_DIR),
        "--secret-out", str(secret_context),
        "--eval-out", str(eval_context),
    ]
    print("[client_key_generation] Generating CKKS client/evaluation contexts...", flush=True)
    subprocess.run(command, check=True, env=runtime_environment())
    if not secret_context.is_file() or not eval_context.is_file():
        raise RuntimeError("Native key generation did not create both contexts")

    atomic_write_json(
        secret_dir / "manifest.json",
        {
            "schema_version": 1,
            "secret_context_sha256": sha256_file(secret_context),
            "secret_context_size_bytes": secret_context.stat().st_size,
        },
    )
    atomic_write_json(
        public_dir / "manifest.json",
        {
            "schema_version": 1,
            "eval_context_sha256": sha256_file(eval_context),
            "eval_context_size_bytes": eval_context.stat().st_size,
            "contains_secret_key": False,
        },
    )
    print(
        f"[client_key_generation] Complete in {time.monotonic() - started:.1f}s; "
        f"public context={eval_context.stat().st_size / 2**30:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
