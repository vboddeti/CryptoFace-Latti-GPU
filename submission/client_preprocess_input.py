#!/usr/bin/env python3
"""Stage 5: detect, align, normalize, and split faces into 32x32 patches."""

from __future__ import annotations

import sys

import h5py

from common import (
    decode_image,
    load_detector,
    parse_stage_args,
    preprocess_one_image,
    release_file_cache,
)


def main() -> None:
    _size, cfg, params = parse_stage_args(resolve_checkpoint=False)
    pairs_path = params.get_test_input_file()
    if not pairs_path.is_file():
        print(f"[client_preprocess_input] ERROR: {pairs_path} not found", flush=True)
        sys.exit(1)

    output_path = params.io_intermediate_dir() / "preprocessed_patches.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".h5.tmp")
    pair_count = params.get_batch_size()
    branch_count = len(cfg["input_names"])
    print(f"[client_preprocess_input] {pair_count} pairs -> {output_path}", flush=True)
    detector = load_detector()

    with h5py.File(pairs_path, "r") as pairs, h5py.File(temporary, "w") as output:
        if not {"image0", "image1"}.issubset(pairs):
            raise ValueError(f"Input store is missing image datasets: {pairs_path}")
        if len(pairs["image0"]) != pair_count or len(pairs["image1"]) != pair_count:
            raise ValueError(
                f"Expected {pair_count} pairs, found "
                f"{len(pairs['image0'])}/{len(pairs['image1'])}"
            )
        patches_store = output.create_dataset(
            "patches",
            shape=(pair_count, 2, branch_count, 1, 3, 32, 32),
            dtype="float32",
            chunks=(1, 2, branch_count, 1, 3, 32, 32),
        )
        for pair_index in range(pair_count):
            for image_index in range(2):
                patches = preprocess_one_image(
                    detector,
                    decode_image(pairs[f"image{image_index}"][pair_index]),
                    int(cfg["input_size"]),
                )
                if len(patches) != branch_count:
                    raise ValueError(
                        f"Expected {branch_count} patches, got {len(patches)}"
                    )
                for branch_index, patch in enumerate(patches):
                    patches_store[pair_index, image_index, branch_index] = patch.numpy()
        output.attrs["pair_count"] = pair_count
        output.attrs["branch_count"] = branch_count
    temporary.replace(output_path)
    release_file_cache(pairs_path)
    release_file_cache(output_path)
    print(
        f"[client_preprocess_input] Saved {pair_count * 2 * branch_count} patches",
        flush=True,
    )


if __name__ == "__main__":
    main()
