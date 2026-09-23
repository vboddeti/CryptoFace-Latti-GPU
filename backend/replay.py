#!/usr/bin/env python3
"""Verify or replay the captured exact public-twiddle using only bundled code and external inputs."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

SCRATCH = Path('/mnt/gs21/scratch/vishnu/cryptoface')
DEPENDENCIES = {'latti-ai': '.deps/latti-ai', 'lattisense': '.deps/latti-ai/inference/lattisense'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe manifest path')
    target = root / relative
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Missing, symlinked or escaping manifest file: ' + name)
    return target


def verify_package(bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256(checked_path(bundle, name)) != digest:
            raise ValueError('Changed package file: ' + name)
    if manifest['files'].get('replay.py') != sha256(Path(__file__)):
        raise ValueError('Replay entry point differs from package')
    return manifest


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def indexed_map(root):
    entries = {}
    for entry in git(root, 'ls-files', '--stage', '-z').split(b'\0'):
        if not entry:
            continue
        meta, filename = entry.decode().split('\t', 1)
        mode, revision, stage = meta.split()
        if stage != '0':
            raise ValueError('Conflicted dependency index')
        path = root / filename
        if mode == '160000':
            digest = revision
        elif mode == '120000':
            if not path.is_symlink():
                raise ValueError('Dependency symlink changed type')
            digest = hashlib.sha256(os.readlink(path).encode()).hexdigest()
        else:
            digest = sha256(checked_path(root, filename))
        entries[filename] = {'index': meta, 'sha256': digest}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def verify_build(bundle, build, manifest, validator):
    if build == SCRATCH.resolve() or not build.is_relative_to((SCRATCH / 'reproducible').resolve()):
        raise ValueError('Require a dedicated scratch build')
    base = json.loads((bundle / 'base/manifest.json').read_text())
    source = build / 'source'
    link = source / '.deps'
    if not link.is_symlink() or link.resolve(strict=True) != (build / '.deps').resolve(strict=True):
        raise ValueError('Rebuild dependency link differs from its isolated build')
    paths = list(source.rglob('*'))
    if any(path.is_symlink() and path != link for path in paths):
        raise ValueError('Unexpected symlink in rebuilt application')
    source_hashes = {str(path.relative_to(source)): sha256(path) for path in paths if path.is_file()}
    if source_hashes != base['measured_build']['source_hashes']:
        raise ValueError('Rebuilt application differs from base capture')
    provenance = json.loads((build / 'optimization-provenance.json').read_text())
    if (provenance['base_bundle_manifest_sha256'] != manifest['base_bundle_manifest_sha256']
            or provenance['extra_patch_sha256'] != manifest['optimization_patch_sha256']
            or provenance['recipe_sha256'] != manifest['files']['rebuild_host_key_sources.py']):
        raise ValueError('Build recipe or optimization provenance differs')
    if sha256(build / 'extra-lattisense.patch') != manifest['optimization_patch_sha256']:
        raise ValueError('Build patch changed')
    for name, relative in DEPENDENCIES.items():
        root = build / relative
        if git(root, 'rev-parse', 'HEAD').decode().strip() != manifest['dependencies'][name]:
            raise ValueError('Dependency commit differs')
        if indexed_map(root) != manifest['dependency_source_maps'][name]:
            raise ValueError('Dependency source contents differ')
        metadata = base['dependencies'][name]
        untracked = set(filter(None, git(root, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')))
        if untracked != set(metadata['untracked_source_sha256']):
            raise ValueError('Unexpected untracked dependency files')
        for key in ('untracked_source_sha256', 'ignored_build_source_sha256'):
            for filename, digest in metadata[key].items():
                if sha256(checked_path(root, filename)) != digest:
                    raise ValueError('Extra dependency source changed')
    return {'binary_sha256': sha256(build / 'submission-build/latti_stage_runtime'),
            'runtime_sha256': sha256(build / '.deps/latti-ai/build-h200/inference/lattisense/liblattisense.so')}


def verify_environment(manifest):
    expected = manifest['replay_environment']
    for name, version in expected['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Reference Python package differs: ' + name)
    router = Path(importlib.metadata.distribution('insightface').locate_file('insightface/model_zoo/model_zoo.py'))
    if sha256(router) != expected['router_sha256']:
        raise ValueError('Reference session router changed')
    models = Path(expected['model_directory'])
    if {p.name: sha256(p) for p in models.glob('*.onnx')} != expected['model_sha256']:
        raise ValueError('Reference model inputs changed; do not download replacements')
    dataset = SCRATCH / 'CryptoFace-Latti-GPU/datasets/face_dataset.h5'
    if sha256(dataset) != expected['dataset_sha256']:
        raise ValueError('Real face dataset differs')


def check_small(report, binding, assess_result):
    if (report.get('size') != 1 or report.get('binding') != binding
            or report.get('assessment', {}).get('correctness_passed') is not True
            or report['assessment'].get('pair_count') != 128
            or report['assessment'].get('gpu_count') != 4):
        raise ValueError('Medium requires a correctness-passing unchanged package small replay')
    result = Path(report['result_directory']).resolve() / 'results-1.json'
    if not result.is_relative_to(SCRATCH.resolve()) or sha256(result) != report['result_sha256']:
        raise ValueError('Small replay raw result changed or is outside scratch')
    if assess_result(json.loads(result.read_text()), 1)['correctness_passed'] is not True:
        raise ValueError('Recomputed small replay correctness failed')


def twiddle_settings(provider, *args, **kwargs):
    """Record identical opt-in semantics to the accepted native/matched lane."""
    settings = dict(provider(*args, **kwargs))
    settings.update(LATTISENSE_GPU_TWIDDLE_SHOUP='1',
                    LATTISENSE_GPU_TWIDDLE_SHOUP_VERIFY='0')
    return settings


def check_replay_provenance(provenance, expected_settings, source):
    if source.is_symlink() or source.resolve() == SCRATCH.resolve() or not source.resolve().is_relative_to(SCRATCH.resolve()):
        raise ValueError('Require a dedicated native scratch source directory')
    if any(provenance['settings'].get(k) != v for k, v in expected_settings.items()):
        raise ValueError('Replay settings differ matched public-twiddle settings')
    if provenance.get('core_dump_limit_bytes') != 0:
        raise ValueError('Missing native core-dump guard')
    if Path(provenance['runtime_working_directory']).resolve() != source.resolve():
        raise ValueError('Native replay did not use the verified scratch source cwd')


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--size', type=int, choices=[1, 2], required=True)
    parser.add_argument('--small-evidence', type=Path)
    parser.add_argument('--replay', action='store_true', help='Otherwise perform read-only verification')
    args = parser.parse_args()
    bundle, build = args.bundle.resolve(strict=True), args.build.resolve(strict=True)
    manifest = verify_package(bundle)
    # All imported code has been hash-checked above; no dependency on the analysis folder.
    sys.path.insert(0, str(bundle / 'validation'))
    validator = load_module('ntt_generic_twiddle_replay_validation', bundle / 'validation/run_real_validation.py')
    reference = load_module('ntt_generic_twiddle_replay_reference', bundle / 'source_generators/create_arcface_thread_patch.py')
    native = verify_build(bundle, build, manifest, validator)
    verify_environment(manifest)
    original_settings = validator.candidate_settings
    validator.candidate_settings = lambda *a, **k: twiddle_settings(original_settings, *a, **k)
    expected_settings = validator.candidate_settings('host_keys', 4, host_key_cache_gib=24)
    binding = {'settings': expected_settings, 'package_manifest_sha256': sha256(bundle / 'manifest.json'), 'native': native,
               'application_sha256': manifest['measured_application_sha256'],
               'dataset_sha256': manifest['replay_environment']['dataset_sha256']}
    if args.size == 2:
        if not args.small_evidence:
            raise ValueError('Medium requires small replay evidence')
        check_small(json.loads(args.small_evidence.read_text()), binding, validator.assess_result)
    if not args.replay:
        print(json.dumps({'verification_passed': True, 'inference_executed': False, 'binding': binding}, indent=2))
        return
    job = os.environ.get('SLURM_JOB_ID', '')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if not job.isdecimal() or len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('Require a four-GPU Slurm allocation')
    campaign = SCRATCH / f'reproducible/ntt-generic-twiddle-package-replay-{args.size}-{job}'
    campaign.mkdir(parents=True, exist_ok=False)
    validator.prepare_runtime(campaign)
    source = campaign / 'source'
    shutil.copytree(build / 'source', source, ignore=shutil.ignore_patterns('.deps'))
    target = source / reference.RELATIVE
    target.write_text(reference.transform(target.read_text()))
    if validator.source_hashes(source) != manifest['measured_application_sha256']:
        raise ValueError('Patched scratch application differs from measured application')
    validator.CAMPAIGN = campaign
    validator.VARIANT_FAMILIES['host_keys'] = str(build.relative_to(SCRATCH / 'reproducible'))
    sys.argv = ['run_real_validation.py', '--size', str(args.size), '--variant', 'host_keys',
                '--threads', '4', '--host-key-cache-gib', '24', '--run-tag', 'package-replay']
    validator.main()
    name = validator.validation_run_name(args.size, job, 'package-replay')
    results = campaign / 'results' / name
    assessment = json.loads((results / 'summary.json').read_text())
    provenance = json.loads((results / 'provenance.json').read_text())
    # The shared validator makes its own final native source copy. The campaign
    # source above is the staging copy, not the cwd used by encrypted workers.
    runtime_source = SCRATCH / 'reproducible/hoisted-latency-v2' / name / 'source'
    check_replay_provenance(provenance, expected_settings, runtime_source)
    if validator.source_hashes(runtime_source) != manifest['measured_application_sha256']:
        raise ValueError('Final native scratch source differs measured application')
    if (verify_build(bundle, build, manifest, validator) != native
            or verify_package(bundle) != manifest
            or validator.source_hashes(source) != manifest['measured_application_sha256']):
        raise ValueError('Package, source or build changed during replay')
    verify_environment(manifest)
    report = {'size': args.size, 'binding': binding, 'assessment': assessment,
              'result_directory': str(results), 'result_sha256': sha256(results / 'results-1.json'),
              'registry_promotion': False,
              'scope': 'Single-candidate reproduction, not a matched optimization comparison'}
    destination = campaign / 'replay.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'replay_evidence': str(destination), **report}, indent=2))


if __name__ == '__main__':
    main()
