#!/usr/bin/env python3
"""Write a reproducible manifest for locally built native artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latti-ai-source", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cuda-architecture", required=True)
    args = parser.parse_args()

    artifacts = {}
    for name, path in (
        ("native_binary", args.native_binary),
        ("lattisense_runtime", args.runtime),
    ):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "schema_version": 1,
        "cuda_architecture": args.cuda_architecture,
        "latti_ai_commit": git_head(args.latti_ai_source),
        "lattisense_commit": git_head(
            args.latti_ai_source / "inference" / "lattisense"
        ),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
