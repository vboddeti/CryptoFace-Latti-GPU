#!/usr/bin/env python3
"""Rebuild the frozen GPU backend without modifying the official harness."""
import argparse
import os
from pathlib import Path
import resource
import sys

from benchmark import ROOT, digest, module, scratch_path, verify_release


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--repository-source', type=Path, required=True,
                        help='Existing pinned latti-ai checkout with its lattisense submodule; no downloads')
    parser.add_argument('--ntl-root', type=Path, required=True)
    parser.add_argument('--model-input', type=Path, required=True)
    args = parser.parse_args()
    release = verify_release()
    work = scratch_path(args.work, args.scratch_root)
    if not work.is_relative_to(args.scratch_root.resolve() / 'reproducible'):
        parser.error('Use a new path below SCRATCH_ROOT/reproducible for backend builds')
    builder = module(ROOT / 'backend/rebuild_host_key_sources.py', 'fhe_frozen_builder')
    # Relocate only the storage guard; the hash-pinned source/patch recipe is unchanged.
    builder.SCRATCH = args.scratch_root.resolve(strict=True)
    sys.argv = [str(ROOT / 'backend/rebuild_host_key_sources.py'), '--build',
                '--bundle', str(ROOT / 'backend/base'), '--work', str(work),
                '--repository-source', str(args.repository_source.resolve(strict=True)),
                '--ntl-root', str(args.ntl_root.resolve(strict=True)),
                '--model-input', str(args.model_input.resolve(strict=True)),
                '--lattisense-extra-patch', str(ROOT / 'backend/optimization.patch')]
    builder.main()
    print(f"Built {release['release_id']} in {work}; release SHA256 {digest(ROOT / 'release.json')}")


if __name__ == '__main__':
    main()
