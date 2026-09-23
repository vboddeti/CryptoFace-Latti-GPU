#!/usr/bin/env python3
"""Capture bounded code-only public-twiddle sources, without builds or registry writes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import rebuild_host_key_sources as rebuild
import validate_ntt_generic_twiddle_fast_reference as validation

HERE = Path(__file__).resolve().parent
DESTINATION = HERE.parents[1] / 'reproduction-sources/ntt-generic-twiddle-v1'
FORBIDDEN = {'.so', '.a', '.o', '.h5', '.hdf5', '.npz', '.npy', '.bin', '.pyc',
             '.sqlite', '.nsys-rep', '.log', '.pt', '.pth', '.onnx'}


def add_file(files, name, source, digest):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or name in files:
        raise ValueError('Unsafe or duplicate bundle path')
    if source.is_symlink() or not source.is_file():
        raise ValueError('Require a regular source file')
    data = source.read_bytes()
    if source.suffix in FORBIDDEN or len(data) > 20 << 20 or b'\0' in data:
        raise ValueError('Refuse binary, generated or oversized source artifact')
    data.decode('utf-8')
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Source file changed: ' + str(source))
    files[name] = (source, digest)


def base_files(bundle):
    manifest = rebuild.verify(bundle)
    files = {}
    external = manifest['required_external_inputs']
    for name, digest in manifest['measured_build']['source_hashes'].items():
        if name not in external:
            add_file(files, 'base/code/' + name, bundle / 'code' / name, digest)
    for name, metadata in manifest['dependencies'].items():
        add_file(files, f'base/{name}.patch', bundle / f'{name}.patch', metadata['patch_sha256'])
        for category, key in [('untracked', 'untracked_source_sha256'),
                              ('ignored', 'ignored_build_source_sha256')]:
            for filename, digest in metadata[key].items():
                relative = f'{category}/{name}/{filename}'
                add_file(files, 'base/' + relative, bundle / relative, digest)
    for name in ('manifest.json', 'rebuild_host_key_sources.py'):
        add_file(files, 'base/' + name, bundle / name, rebuild.sha256(bundle / name))
    return files, manifest


def collect(small_path, medium_path):
    small = json.loads(small_path.read_text())
    builds = validation.build_audit()
    original = validation.validation.source_hashes(validation.CAMPAIGN / 'source')
    expected = validation.patched_source_hashes(original)
    validation.check_encrypted_small(small, builds, expected, small['reference_evidence_sha256'])
    medium = json.loads(medium_path.read_text())
    if (medium.get('size') != 2 or medium.get('candidate_variant') != validation.VARIANT
        or medium.get('promotion_gate_passed') is not True
        or medium.get('build_audit') != builds
        or medium.get('patched_source_hashes') != expected
        or medium.get('reference_evidence_sha256') != small['reference_evidence_sha256']
        or medium.get('launcher_sha256') != rebuild.sha256(Path(validation.__file__))):
        raise ValueError('Require passing unchanged matched medium evidence')
    if validation.checked_matched(medium['conditions'], 2, builds, expected)['promotion_gate_passed'] is not True:
        raise ValueError('Recomputed matched medium gate failed')
    for first, second in zip(small['conditions'], medium['conditions']):
        if first['provenance']['input_store_sha256'] != second['provenance']['input_store_sha256']:
            raise ValueError('Real dataset differs between small and medium')
    files, base = base_files(validation.BUNDLE)
    additions = {
        'README.md': HERE / 'ntt-generic-twiddle-reproducer-README.md',
        'optimization.patch': validation.REBUILT / 'extra-lattisense.patch',
        'reference.patch': HERE / 'arcface-thread-limit.patch',
        'rebuild_host_key_sources.py': HERE / 'rebuild_host_key_sources.py',
        'source_generators/create_moddown_p3_entry_patch.py': HERE / 'create_moddown_p3_entry_patch.py',
        'source_generators/create_ntt_config_cache_patch.py': HERE / 'create_ntt_config_cache_patch.py',
        'source_generators/create_ntt_n16_specialization_patch.py': HERE / 'create_ntt_n16_specialization_patch.py',
        'source_generators/create_arcface_thread_patch.py': HERE / 'create_arcface_thread_patch.py',
        'validation/run_real_validation.py': validation.CAMPAIGN / 'run_real_validation.py',
        'validation/scratch_runtime.py': validation.CAMPAIGN / 'scratch_runtime.py',
        'validation/matched_driver.py': HERE / 'validate_ntt_generic_twiddle_fast_reference.py',
        'capture_ntt_generic_twiddle_reproducer.py': Path(__file__),
    }
    additions.update({
        'source_generators/create_ntt_public_twiddle_patch.py': HERE / 'create_ntt_public_twiddle_patch.py',
        'source_generators/ntt_public_twiddle_templates.py': HERE / 'ntt_public_twiddle_templates.py',
        'source_generators/create_ntt_generic_twiddle_patch.py': HERE / 'create_ntt_generic_twiddle_patch.py',
        'replay.py': HERE / 'replay_ntt_generic_twiddle_bundle.py',
    })
    if rebuild.sha256(additions['optimization.patch']) != builds['rebuilt']['patch_sha256']:
        raise ValueError('Cumulative public-twiddle patch differs audited measured source')
    for name, source in additions.items():
        add_file(files, name, source, rebuild.sha256(source))
    if sum(source.stat().st_size for source, _ in files.values()) > 20 << 20:
        raise ValueError('Code bundle unexpectedly exceeds 20 MiB')
    manifest = {
        'schema_version': 1, 'family': 'ntt-generic-twiddle-v1',
        'status': 'code-capture-ready-replay-unverified', 'registry_promotion': False,
        'small_evidence': {'path': str(small_path.resolve()), 'sha256': rebuild.sha256(small_path)},
        'small_seconds_per_pair': small['candidate_seconds_per_pair'],
        'medium_validation': 'required-before-promotion',
        'base_bundle_manifest_sha256': rebuild.sha256(validation.BUNDLE / 'manifest.json'),
        'optimization_patch_sha256': files['optimization.patch'][1],
        'reference_patch_sha256': files['reference.patch'][1],
        'measured_application_sha256': expected,
        'measured_native_sha256': builds['rebuilt']['rebuilt'],
        'dependencies': {name: data['commit'] for name, data in base['dependencies'].items()},
        'required_external_inputs': base['required_external_inputs'],
        'files': {name: digest for name, (_, digest) in files.items()},
        'reproduction_status': {
            'source_capture_verified': True, 'package_rebuild_executed': False,
            'package_small_replay_verified': False, 'package_medium_replay_verified': False},
        'limitations': [
            'External pinned Git objects, NTL sysroot, model/data inputs and Python environment required.',
            'Captured validation files record the measured engine; their original path assumptions are not a portable replay launcher.',
            'Do not treat this source capture as completed reproduction or change the successful-only registry.'],
    }
    manifest['medium_evidence'] = {'path': str(medium_path.resolve()), 'sha256': rebuild.sha256(medium_path)}
    manifest['matched_gates_verified'] = {'small': True, 'medium': True}
    return files, manifest


def capture(files, manifest, destination):
    if destination != DESTINATION or destination.exists() or destination.is_symlink():
        raise ValueError('Require the new designated code-only reproduction folder')
    destination.mkdir(parents=True, exist_ok=False)
    for name, (source, digest) in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if rebuild.sha256(target) != digest:
            raise ValueError('Source changed during capture; partial bundle is not valid')
    (destination / 'capture-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--small', required=True, type=Path)
    parser.add_argument('--medium', required=True, type=Path)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    files, manifest = collect(args.small.resolve(strict=True), args.medium.resolve(strict=True))
    if args.capture:
        capture(files, manifest, DESTINATION)
    print(json.dumps({'source_capture_checks_passed': True, 'captured': args.capture,
                      'destination': str(DESTINATION), 'file_count': len(files),
                      'source_bytes': sum(path.stat().st_size for path, _ in files.values()),
                      'replay_verified': False}, indent=2))


if __name__ == '__main__':
    main()
