#!/usr/bin/env python3
"""Build only the new native adapter against the hash-verified frozen backend."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from benchmark import ROOT, digest, load, module, scratch_path, verify_release, write_new


def baseline_identity(build, scratch):
    replay = module(ROOT / 'backend/replay.py', 'baseline_matching_verifier')
    replay.SCRATCH = scratch.resolve(strict=True)
    manifest = replay.verify_package(ROOT / 'backend')
    return manifest, replay.verify_build(ROOT / 'backend', build, manifest, None)


def verify(build, scratch):
    record = load(build / 'gpu-matching-build.json')
    if record['release_sha256'] != digest(ROOT / 'release.json'):
        raise ValueError('GPU matching build belongs to a different source release')
    baseline = scratch_path(Path(record['baseline_build']), scratch)
    manifest, original = baseline_identity(baseline, scratch)
    if original != record['baseline_native']:
        raise ValueError('Frozen backend changed')
    verify_release(build / 'source')
    if (build / '.deps').resolve(strict=True) != (baseline / '.deps').resolve(strict=True):
        raise ValueError('Unexpected GPU matching dependencies')
    for name, expected in record['artifacts'].items():
        if digest(build / name) != expected:
            raise ValueError(f'GPU matching artifact changed: {name}')
    return manifest, {'binary_sha256': digest(build / 'submission-build/latti_stage_runtime'),
                      'runtime_sha256': original['runtime_sha256'],
                      'gpu_matching_build_sha256': digest(build / 'gpu-matching-build.json')}


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--baseline-build', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    release = verify_release()
    work = scratch_path(args.work, args.scratch_root)
    baseline = scratch_path(args.baseline_build, args.scratch_root)
    if work.exists():
        raise ValueError('Use a new build directory')
    _, original = baseline_identity(baseline, args.scratch_root)
    work.mkdir(mode=0o700, parents=True)
    source = work / 'source'
    for name in ('release.json', *release['files']):
        destination = source / name
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    verify_release(source)
    (work / '.deps').symlink_to(baseline / '.deps', target_is_directory=True)
    (source / '.deps').symlink_to(work / '.deps', target_is_directory=True)
    model = 'submission/latti/model/server/model_parameters.h5'
    shutil.copy2(baseline / 'source' / model, source / model)
    from benchmark import execution_environment
    env = execution_environment(work, release['settings'])
    env.update(LATTI_AI_SOURCE=str(work / '.deps/latti-ai'),
               LATTI_STAGE_RUNTIME_OUTPUT=str(work / 'submission-build/latti_stage_runtime'),
               PYTHONDONTWRITEBYTECODE='1')
    with (work / 'build.log').open('x') as log:
        subprocess.run([sys.executable, '-B', str(source / 'scripts/generate_matching_task.py'),
                        '--frontend-root', str(work / '.deps/latti-ai/inference/lattisense'),
                        '--parameters', str(source / 'submission/latti/model/server/ckks_parameter.json'),
                        '--output', str(work / 'matching-task')], cwd=work, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['bash', str(source / 'submission/native/build_runtime.sh')],
                       cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = load(baseline / 'submission-build/build_manifest.json')
    metadata['artifacts']['native_binary']['sha256'] = digest(work / 'submission-build/latti_stage_runtime')
    write_new(work / 'submission-build/build_manifest.json', metadata)
    artifacts = {p.relative_to(work).as_posix(): digest(p)
                 for folder in ('submission-build', 'matching-task')
                 for p in (work / folder).rglob('*') if p.is_file()}
    write_new(work / 'gpu-matching-build.json', {
        'schema_version': 1, 'release_sha256': digest(ROOT / 'release.json'),
        'baseline_build': str(baseline), 'baseline_native': original, 'artifacts': artifacts})
    verify(work, args.scratch_root)
    print(json.dumps({'build': str(work), 'verified': True}))


if __name__ == '__main__':
    main()
