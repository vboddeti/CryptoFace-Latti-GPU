#!/usr/bin/env python3
"""Emit a release manifest after comparing harness bytes with a local upstream commit.

Read-only: review its stdout and install release.json deliberately in a new version.
Never use sealing to silently approve changed upstream harness files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from benchmark import ROOT, digest, load


def manifest(repository, revision, release_id):
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', revision + '^{commit}'], text=True).strip()
    names = subprocess.check_output(['git', '-C', str(repository), 'ls-tree', '-r', '--name-only', commit, 'harness'], text=True).splitlines()
    upstream = {}
    for name in names:
        data = subprocess.check_output(['git', '-C', str(repository), 'show', f'{commit}:{name}'])
        if (ROOT / name).is_symlink() or (ROOT / name).read_bytes() != data:
            raise ValueError('Refusing to seal a modified upstream harness: ' + name)
        upstream[name] = hashlib.sha256(data).hexdigest()
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'harness').rglob('*') if p.is_file()}
    if actual != set(upstream):
        raise ValueError('Harness file set differs from upstream')
    files = {}
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in ('measurements', '.git') or relative.as_posix() == 'release.json':
            continue
        if path.is_symlink():
            raise ValueError('Source release must not contain symlinks: ' + str(relative))
        if path.is_file():
            if '__pycache__' in relative.parts or path.suffix in ('.pyc', '.so', '.h5', '.log') or path.name.startswith('core.'):
                raise ValueError('Generated/binary artifact in source release: ' + str(relative))
            files[relative.as_posix()] = digest(path)
    backend = load(ROOT / 'backend/manifest.json')
    return {
        'schema_version': 1, 'release_id': release_id,
        'upstream_repository': 'https://github.com/fhe-benchmarking/face-recognition',
        'upstream_revision': commit, 'upstream_harness_sha256': upstream,
        'backend_family': backend['family'] if 'family' in backend else 'ntt-generic-twiddle-v1',
        'backend_manifest_sha256': digest(ROOT / 'backend/manifest.json'),
        'settings': load(ROOT / 'backend-settings.json'), 'files': files,
        'measurement_status': 'Unmeasured with the unmodified upstream harness; see separate run provenance',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-repository', type=Path, required=True)
    parser.add_argument('--upstream-revision', required=True)
    parser.add_argument('--release-id', required=True)
    args = parser.parse_args()
    print(json.dumps(manifest(args.upstream_repository, args.upstream_revision, args.release_id), indent=2))
