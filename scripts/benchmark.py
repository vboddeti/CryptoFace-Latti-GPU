#!/usr/bin/env python3
"""Versioned, scratch-only execution of the unmodified upstream FHE harness.

No monkey-patching, custom reference implementation, or synthetic inference inputs.
This launcher is outside harness/: the organizer's CLI remains the entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('single', 'small', 'medium', 'large')
PAIRS = (1, 128, 256, 1024)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def seconds(value):
    if isinstance(value, str):
        match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)', value.strip())
        if not match:
            raise ValueError(f'Unexpected harness timing format: {value!r}')
        value = float(match[1]) * {'ms': 0.001, 's': 1, 'm': 60, 'h': 3600}[match[2]]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Expected positive finite timing')
    return value


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify_release(root=ROOT):
    release = load(root / 'release.json')
    for name, expected in release['files'].items():
        path = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected):
            raise ValueError(f'Release source changed: {name}')
    actual = {p.relative_to(root).as_posix() for p in (root / 'harness').rglob('*') if p.is_file()}
    if actual != set(release['upstream_harness_sha256']):
        raise ValueError('Upstream harness file set changed')
    for name, expected in release['upstream_harness_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Upstream harness changed: {name}')
    return release


def scratch_path(path, scratch):
    scratch = scratch.resolve(strict=True)
    result = path.resolve()
    # Explicit scratch root, never a home/source directory or a filesystem root.
    home = Path.home().resolve()
    if (scratch == Path('/') or scratch == home or scratch.is_relative_to(home)
            or scratch == ROOT or scratch.is_relative_to(ROOT)
            or result == scratch or not result.is_relative_to(scratch)):
        raise ValueError('Require a dedicated work directory strictly inside non-home scratch')
    return result


def verify_backend(build, scratch, root=ROOT):
    if (build / 'gpu-matching-build.json').is_file():
        builder = module(root / 'scripts/build_gpu_matching.py', 'matching_builder')
        builder.ROOT = root
        return builder.verify(build, scratch)
    raise ValueError('This release requires a verified GPU matching build')


def verify_original_backend(build, scratch, root=ROOT):
    replay = module(root / 'backend/replay.py', 'fhe_backend_verifier')
    replay.SCRATCH = scratch.resolve(strict=True)
    manifest = replay.verify_package(root / 'backend')
    native = replay.verify_build(root / 'backend', build.resolve(strict=True), manifest, None)
    return manifest, native


def verify_inputs(dataset, manifest):
    expected = manifest['replay_environment']
    if digest(dataset) != expected['dataset_sha256']:
        raise ValueError('Expected the pinned real face dataset; synthetic/replacement inputs forbidden')
    # The unchanged upstream harness uses InsightFace's standard model location.
    # These are read-only inputs provisioned ahead of time, not a download request.
    models = Path.home() / '.insightface/models/buffalo_l'
    for name, sha in expected['model_sha256'].items():
        if digest(models / name) != sha:
            raise ValueError(f'Preprovisioned ArcFace input changed: {name}')
    for name, version in expected['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Python dependency differs: {name}; expected {version}')


def execution_environment(work, settings):
    env = os.environ.copy()
    # Do not inherit experimental overrides or profiler toggles into an official run.
    for key in list(env):
        if key.startswith(('CRYPTOFACE_', 'LATTISENSE_', 'HEONGPU_')) or key in ('PYTHONPATH', 'PYTHONSTARTUP'):
            env.pop(key)
    env.update(settings)
    env.update(PYTHONDONTWRITEBYTECODE='1', CRYPTOFACE_ARTIFACT_ROOT=str(work / 'source'),
               CRYPTOFACE_BUILD_DIR=str(work / 'source/submission-build'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key, leaf in {
        'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp', 'XDG_CACHE_HOME': 'cache',
        'CUDA_CACHE_PATH': 'cache/cuda', 'TORCH_HOME': 'cache/torch',
        'HF_HOME': 'cache/huggingface', 'MPLCONFIGDIR': 'cache/matplotlib',
        'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data',
        'GOCACHE': 'cache/go', 'GOMODCACHE': 'cache/go-mod',
    }.items():
        path = work / leaf
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    return env


def prepare(args):
    release = verify_release()
    scratch = args.scratch_root.resolve(strict=True)
    work = scratch_path(args.work, scratch)
    if work.exists() or args.work.is_symlink():
        raise ValueError('Work directory must be new; never reuse previous measurements')
    build = scratch_path(args.backend_build, scratch)
    manifest, native = verify_backend(build, scratch)
    dataset = args.dataset.resolve(strict=True)
    verify_inputs(dataset, manifest)
    model = build / 'source/submission/latti/model/server/model_parameters.h5'
    model_sha = load(ROOT / 'backend/base/manifest.json')['required_external_inputs'][
        'submission/latti/model/server/model_parameters.h5']['sha256']
    if digest(model) != model_sha:
        raise ValueError('Encrypted-model weights differ from the validated backend')
    work.mkdir(parents=True, mode=0o700)
    source = work / 'source'
    source.mkdir(mode=0o700)
    for name in ('release.json', *release['files']):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(ROOT / name, destination)
    verify_release(source)
    shutil.copy2(model, source / 'submission/latti/model/server/model_parameters.h5')
    (source / '.deps').symlink_to(build / '.deps', target_is_directory=True)
    (source / 'submission-build').symlink_to(build / 'submission-build', target_is_directory=True)
    (source / 'datasets').mkdir(mode=0o700)
    (source / 'datasets/face_dataset.h5').symlink_to(dataset)
    provenance = {
        'schema_version': 1, 'release_id': release['release_id'],
        'release_sha256': digest(ROOT / 'release.json'),
        'upstream_revision': release['upstream_revision'],
        'backend_manifest_sha256': digest(ROOT / 'backend/manifest.json'),
        'backend_build': str(build), 'native': native, 'dataset_sha256': digest(dataset),
        'model_sha256': model_sha, 'scratch_root': str(scratch), 'source': str(source),
        'settings': release['settings'], 'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'inference_executed': False,
    }
    write_new(work / 'prepared.json', provenance)
    print(json.dumps(provenance, indent=2))


def assess_results(source, size, repetitions):
    reports = []
    for number in range(1, repetitions + 1):
        path = source / 'measurements' / NAMES[size] / f'results-{number}.json'
        result = load(path)
        reported = result['Server Reported']
        if reported['additional_measurements'].get('Matching backend') != 'gpu':
            raise ValueError('Result does not prove GPU matching execution')
        if reported['additional_measurements']['Pair count'] != PAIRS[size]:
            raise ValueError('Reported pair count does not match the upstream workload')
        if size and result['Quality']['Comparison to ArcFace baseline']['passed'] is not True:
            raise ValueError('Upstream quality gate failed')
        total = seconds(result['Timing']['Total'])
        compute = seconds(reported['Encrypted computation'])
        reports.append({'path': str(path.relative_to(source)), 'sha256': digest(path),
                        'compute_seconds_per_pair': compute / PAIRS[size],
                        'harness_total_seconds': total})
    return reports


def check_previous(path, prepared, size):
    previous = load(path)
    if previous.get('repetitions') != 3 or previous.get('matching_diagnostic'):
        raise ValueError('Previous workload must be an official three-run measurement')
    if previous.get('completed') is not True or previous.get('size') != size - 1:
        raise ValueError('Require a successful preceding smaller workload')
    for key in ('release_sha256', 'native', 'dataset_sha256', 'settings'):
        if previous['prepared'][key] != prepared[key]:
            raise ValueError(f'Previous workload has a different {key}')
    source = Path(previous['prepared']['source'])
    if assess_results(source, size - 1, previous['repetitions']) != previous['results']:
        raise ValueError('Previous workload result evidence changed')


def run(args):
    release = verify_release()
    work = scratch_path(args.work, args.scratch_root)
    prepared = load(work / 'prepared.json')
    source = work / 'source'
    if (prepared['source'] != str(source) or prepared['release_sha256'] != digest(ROOT / 'release.json')
            or prepared['settings'] != release['settings']):
        raise ValueError('Prepared work does not match this release')
    verify_release(source)
    manifest, native = verify_backend(Path(prepared['backend_build']), args.scratch_root)
    if native != prepared['native']:
        raise ValueError('Prepared native binaries changed')
    verify_inputs(source / 'datasets/face_dataset.h5', manifest)
    if args.size >= 1:
        if args.previous is None:
            raise ValueError('Require preceding smaller three-run workload with unchanged identities')
        check_previous(args.previous.resolve(strict=True), prepared, args.size)
    if (work / 'run-started.json').exists() or (source / 'measurements').exists():
        raise ValueError('Never rerun or overwrite an existing measurement directory')
    repetitions = 1 if args.smoke else 3
    if args.matching_check and not args.smoke:
        raise ValueError('CPU comparison is diagnostic only, not a benchmark')
    if args.smoke and args.size != 0:
        raise ValueError('Smoke mode is only for one real pair; official batches use three runs')
    env = execution_environment(work, release['settings'])
    if args.matching_check:
        env['CRYPTOFACE_VERIFY_GPU_MATCHING'] = '1'
    else:
        if args.matching_validation is None:
            raise ValueError('GPU matching numerical validation must precede benchmarks')
        validation = load(args.matching_validation)
        if (not validation.get('completed') or not validation.get('matching_diagnostic')
                or validation.get('matching_validation', {}).get('passed') is not True
                or validation['matching_validation'].get('absolute_tolerance') != 1e-6):
            raise ValueError('Missing successful unchanged-tolerance GPU matching check')
        for key in ('release_sha256', 'native', 'dataset_sha256', 'settings'):
            if validation['prepared'][key] != prepared[key]:
                raise ValueError(f'Matching validation identity differs: {key}')
    command = [sys.executable, '-B', 'harness/run_submission.py', str(args.size),
               '--num_runs', str(repetitions), '--seed', str(args.seed)]
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                         '--format=csv,noheader'], text=True)
    started = {'command': command, 'prepared': prepared, 'size': args.size,
               'matching_diagnostic': args.matching_check,
               'repetitions': repetitions, 'seed': args.seed, 'gpu_inventory': inventory,
               'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'cwd': str(source),
               'core_dump_limit_bytes': resource.getrlimit(resource.RLIMIT_CORE)[0],
               'started_utc': datetime.now(timezone.utc).isoformat()}
    write_new(work / 'run-started.json', started)
    with (work / 'harness.log').open('x') as log:
        outcome = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    if outcome.returncode:
        raise RuntimeError(f'Harness failed ({outcome.returncode}); inspect scratch harness.log')
    verify_release(source)
    reports = assess_results(source, args.size, repetitions)
    _, after = verify_backend(Path(prepared['backend_build']), args.scratch_root)
    if after != native:
        raise ValueError('Native build identity changed during measurement')
    summary = dict(started, completed=True, results=reports,
                   finished_utc=datetime.now(timezone.utc).isoformat(),
                   three_run_measurement_complete=(repetitions == 3),
                   benchmark_acceptance='Pending organizer review; no automatic publication',
                   mean_compute_seconds_per_pair=sum(r['compute_seconds_per_pair'] for r in reports) / repetitions)
    if args.matching_check:
        summary['matching_validation'] = load(source / 'io' / NAMES[args.size] / 'gpu-matching-validation.json')
        if (summary['matching_validation'].get('passed') is not True
                or summary['matching_validation'].get('pairs') != PAIRS[args.size]
                or summary['matching_validation'].get('absolute_tolerance') != 1e-6):
            raise ValueError('GPU matching comparison failed')
    write_new(work / 'completed.json', summary)
    print(json.dumps(summary, indent=2))


def collect(args):
    work = scratch_path(args.work, args.scratch_root)
    summary = load(work / 'completed.json')
    source = work / 'source'
    verify_release(source)
    if not summary['completed'] or assess_results(source, summary['size'], summary['repetitions']) != summary['results']:
        raise ValueError('Incomplete or changed result evidence')
    # Publish only final JSONs, never keys, ciphertexts, logs or build products.
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError('Final-results destination already exists')
    destination.mkdir(parents=True, mode=0o700)
    for entry in summary['results']:
        shutil.copy2(source / entry['path'], destination / Path(entry['path']).name)
    shutil.copy2(work / 'completed.json', destination / 'provenance.json')
    shutil.copy2(source / 'release.json', destination / 'release.json')
    print(destination)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify')
    for command in ('prepare', 'run', 'collect'):
        sub = commands.add_parser(command)
        sub.add_argument('--scratch-root', type=Path, required=True)
        sub.add_argument('--work', type=Path, required=True)
        if command == 'prepare':
            sub.add_argument('--backend-build', type=Path, required=True)
            sub.add_argument('--dataset', type=Path, required=True)
        elif command == 'run':
            sub.add_argument('--size', type=int, choices=range(4), required=True)
            sub.add_argument('--seed', type=int, default=42)
            sub.add_argument('--previous', type=Path)
            sub.add_argument('--smoke', action='store_true')
            sub.add_argument('--matching-check', action='store_true')
            sub.add_argument('--matching-validation', type=Path)
        else:
            sub.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        release = verify_release()
        print(f"PASS: {release['release_id']}; unmodified upstream harness {release['upstream_revision']}")
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
