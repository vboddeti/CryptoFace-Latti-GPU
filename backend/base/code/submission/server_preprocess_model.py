#!/usr/bin/env python3
"""Stage 3: validate and expose the precompiled, server-owned GPU model."""

from __future__ import annotations

import json
import shutil
import time

from common import get_face_params, load_submission_config, mute_logs
from gpu_runtime import (
    BUILD_MANIFEST,
    MODEL_DIR,
    atomic_write_json,
    directory_size,
    require_runtime,
    sha256_file,
)


def main() -> None:
    mute_logs()
    cfg = load_submission_config()
    size_file = get_face_params(0).io_root() / "current_size.txt"
    if not size_file.is_file():
        raise FileNotFoundError(f"Missing benchmark size marker: {size_file}")
    params = get_face_params(int(size_file.read_text().strip()))
    started = time.monotonic()
    require_runtime(cfg)
    eval_context = params.iodir() / "public_keys" / "eval_context.bin"
    if not eval_context.is_file():
        raise FileNotFoundError(
            f"Missing public evaluation context: {eval_context}; run stage 2 first"
        )

    server_data_root = params.get_server_data_dir()
    server_data_root.mkdir(parents=True, exist_ok=True)
    task_dir = server_data_root / f"latti-{cfg['mega_ag_sha256'][:24]}"
    server_dir = task_dir / "server"
    cache_valid = (
        (server_dir / "model_parameters.h5").is_file()
        and (server_dir / "mega_ag.json").is_file()
        and sha256_file(server_dir / "model_parameters.h5")
        == cfg["model_parameters_sha256"]
        and sha256_file(server_dir / "mega_ag.json") == cfg["mega_ag_sha256"]
    )
    if not cache_valid:
        temporary = task_dir.with_name(task_dir.name + ".tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.copytree(MODEL_DIR / "server", temporary / "server")
        shutil.rmtree(task_dir, ignore_errors=True)
        temporary.replace(task_dir)

    packed_model_bytes = directory_size(server_dir)
    reference = {
        "schema_version": 1,
        "task_dir": str(task_dir.resolve()),
        "packed_model_size_bytes": packed_model_bytes,
        "model_parameters_sha256": sha256_file(server_dir / "model_parameters.h5"),
        "mega_ag_sha256": sha256_file(server_dir / "mega_ag.json"),
    }
    atomic_write_json(params.iodir() / "server_model.json", reference)
    build_manifest = json.loads(BUILD_MANIFEST.read_text())
    atomic_write_json(
        params.iodir() / "provenance.json",
        {
            "schema_version": 1,
            "submission": "CryptoFace Latti source-built H200 GPU",
            "latti_ai_commit": cfg["latti_ai_commit"],
            "lattisense_commit": cfg["lattisense_commit"],
            "heongpu_commit": cfg["heongpu_commit"],
            "runtime_sha256": build_manifest["artifacts"]["lattisense_runtime"]["sha256"],
            "native_binary_sha256": build_manifest["artifacts"]["native_binary"]["sha256"],
            "security_classical_bits": cfg["security_classical_bits"],
            "poly_modulus_degree": cfg["poly_modulus_degree"],
            "q_limbs": cfg["q_limbs"],
            "p_limbs": cfg["p_limbs"],
            "physical_bootstraps_per_image": cfg["physical_bootstraps_per_image"],
            "cuda_architecture": cfg["cuda_architecture"],
        },
    )
    print(
        f"[server_preprocess_model] Materialized packed model in io/server_data in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
