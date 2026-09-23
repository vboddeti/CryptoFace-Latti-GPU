#!/usr/bin/env python3
"""Match generic-forward public-twiddle against the ordered-public-twiddle parent.

Encrypted code and shared validators are unchanged. Source snapshots, native
work, logs and caches stay in scratch; only final JSON evidence goes to home.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import subprocess

import audit_ntt_generic_twiddle_native as compound_audit
from assess_native_candidate import assess as assess_native
import create_arcface_thread_patch as reference_patch
import validate_arcface_threads_full as reference_gate

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / 'hoisted-latency-v2'
sys.path.insert(0, str(CAMPAIGN))
import run_real_validation as validation
from run_matched_validation import assess_matched
from scratch_runtime import prepare_runtime

SCRATCH = validation.SCRATCH
BUNDLE = HERE.parents[1] / 'reproduction-sources/scoped-host-key-cache-v1'
REBUILT = SCRATCH / 'reproducible/converted-cache-analysis/ntt-generic-twiddle-17399674'
PARENT = SCRATCH / 'reproducible/converted-cache-analysis/ntt-public-twiddle-17391334'
PATCH_SHA256 = 'a4cc05db1072817f0ca95465006eb0cea96c4e70fb28fb5b8928cf63b9160a8c'
EXPECTED_PARENT = {
    'binary_sha256': '0ec5f4c7058682cb98338c2e6cc0f8dc80808320c32687589570d559191ed72c',
    'runtime_sha256': '76691ce37195be15c6f12aba3609dfca6f173a4f407fd91e621b727c6488c019',
    'inference_binary_sha256': '98c575e94ece192a4228de5c6896e1788653e4f24d7ae1f5470449a460d1a562',
}
EXPECTED_REBUILT_RUNTIME = '1c6b3a767785481fc5e8a130d9b188cc74ee653bd73f1348104afa43aa75dff0'
# Unarmed until native17407055 completes and its full evidence is independently checked.
# A source build or diagnostic alone cannot authorize matched small.
NATIVE_SHA256 = 'aafa7a21b04cff57ae3a1961af55fac20dbbf478cab08b81934b81b3bd0a9c95'
NATIVE_JOB = '17407055'
NATIVE_OUTPUT = SCRATCH / ('reproducible/converted-cache-analysis/ntt-generic-twiddle-native-' + NATIVE_JOB)
NATIVE_LAUNCHER = SCRATCH / ('reproducible/converted-cache-analysis/ntt-generic-twiddle-native-launcher-' + NATIVE_JOB)
REAL_BUNDLE_SHA256 = 'd1ac68951c1bce0f2d79bd7ccc29df36ac3a282239b4b6577487ccc50e3cd7d9'
PARENT_INFERENCE_SHA256 = '98c575e94ece192a4228de5c6896e1788653e4f24d7ae1f5470449a460d1a562'
TAGS = ('ntt-generic-twiddle-fast-ref-parent-before', 'ntt-generic-twiddle-fast-ref-candidate', 'ntt-generic-twiddle-fast-ref-parent-after')
VARIANT = 'generic_public_twiddle_fast_reference'


_ORIGINAL_CANDIDATE_SETTINGS = validation.candidate_settings


def matched_settings(*args, **kwargs):
    """Use identical public opt-in flags for parent and candidate; record both."""
    settings = _ORIGINAL_CANDIDATE_SETTINGS(*args, **kwargs)
    settings.update(LATTISENSE_GPU_TWIDDLE_SHOUP='1',
                    LATTISENSE_GPU_TWIDDLE_SHOUP_VERIFY='0')
    return settings


def reference_admission(small, medium, small_sha256):
    reference_gate.check_small(small)
    if (medium.get('size') != 2 or medium.get('correctness_passed') is not True
            or medium.get('timing_screen_passed') is not True
            or medium.get('diagnostic_sha256') != reference_gate.sha256(Path(reference_gate.__file__))
            or medium.get('small_evidence_sha256') != small_sha256):
        raise ValueError('Require passing unchanged full medium reference evidence')
    verified = reference_gate.compare(medium['conditions'], 2)
    if not verified['correctness_passed'] or not verified['timing_screen_passed']:
        raise ValueError('Recomputed medium reference gate failed')
    for key in ('source_sha256', 'model_sha256', 'packages', 'library_source_sha256'):
        if small['conditions'][0][key] != medium['conditions'][0][key]:
            raise ValueError('Reference environment differs between small and medium')
    return verified


def build_audit():
    native_build = compound_audit.audit_candidate()
    compound = native_build['source']
    if compound['parent']['rebuilt'] != EXPECTED_PARENT or compound['rebuilt']['runtime_sha256'] != EXPECTED_REBUILT_RUNTIME:
        raise ValueError('Audited Generic-public-twiddle or parent differs from native-screen builds')
    if validation.sha256(HERE / 'arcface-thread-limit.patch') != PATCH_SHA256:
        raise ValueError('CPU reference patch changed')
    generated = reference_patch.make_patch(CAMPAIGN / 'source' / reference_patch.RELATIVE)
    if hashlib.sha256(generated.encode()).hexdigest() != PATCH_SHA256:
        raise ValueError('CPU reference patch generation changed')
    return {'parent': compound['parent']['rebuilt'], 'rebuilt': compound,
            'native_build_admission': native_build,
            'reference_patch_sha256': PATCH_SHA256,
            'patch_generator_sha256': validation.sha256(Path(reference_patch.__file__)),
            'shared_validator_sha256': validation.sha256(Path(validation.__file__)),
            'native_assessor_sha256': validation.sha256(HERE / 'assess_native_candidate.py'),
            'lane_driver_sha256': validation.sha256(Path(__file__))}


def checked_native_summary(path):
    if (not isinstance(NATIVE_SHA256, str) or len(NATIVE_SHA256) != 64
            or any(c not in '0123456789abcdef' for c in NATIVE_SHA256)):
        raise ValueError('Native evidence is not pinned; candidate is not armed')
    if validation.sha256(path) != NATIVE_SHA256:
        raise ValueError('Native evidence differs from the verified completed screen')
    return json.loads(path.read_text())


def native_admission(summary, builds):
    compound_audit.check_plan(summary['plan'])
    if (summary.get('requests_per_condition') != 30
            or summary.get('warmup_requests') != 6
            or summary.get('absolute_tolerance') != 0.001
            or summary.get('input_provenance', {}).get('bundle_sha256') != REAL_BUNDLE_SHA256):
        raise ValueError('Native screen must use the pinned real bundle and 30/6 requests')
    result = assess_native(summary, 'generic_public_twiddle')
    if result['timing_screen_passed'] is not True:
        raise ValueError('Generic-public-twiddle native screen has not passed')
    expected = ('parent_before', 'generic_public_twiddle', 'parent_after')
    settings = matched_settings('host_keys', 4, host_key_cache_gib=24)
    for index, name in enumerate(expected):
        condition = summary['conditions'][name]
        runtime = builds['rebuilt']['rebuilt']['runtime_sha256'] if index == 1 else builds['parent']['runtime_sha256']
        binary = (builds['rebuilt']['rebuilt']['inference_binary_sha256']
                  if index == 1 else PARENT_INFERENCE_SHA256)
        if (condition.get('binary_sha256') != binary
                or condition.get('binary') != summary['plan'][index]['binary']):
            raise ValueError('Native executable differs from the audited plan')
        if condition['runtime_sha256'] != runtime:
            raise ValueError('Native runtime differs from the audited Generic-public-twiddle plan')
        for key in ('LATTISENSE_GPU_SUBMISSION_THREADS', 'LATTISENSE_GPU_HOST_KEY_CACHE_MAX_BYTES',
                    'LATTISENSE_GPU_IMMUTABLE_EVAL_KEYS', 'LATTISENSE_GPU_PLAINTEXT_CACHE_MAX_BYTES',
                    'LATTISENSE_GPU_CONVERTED_PLAINTEXT_MAX_BYTES', 'LATTISENSE_GPU_KEY_EXPORT_PROFILE', 'LATTISENSE_GPU_TWIDDLE_SHOUP',
                    'LATTISENSE_GPU_TWIDDLE_SHOUP_VERIFY'):
            if condition['env'].get(key) != settings[key]:
                raise ValueError('Native cache/thread settings differ')
    return result


def check_completed_native_evidence(summary, builds, screen):
    scheduler = subprocess.run(['sacct', '-X', '-n', '-P', '-j', NATIVE_JOB,
                                '--format=JobID,State,ExitCode'],
                               capture_output=True, text=True, check=True, timeout=30)
    rows = [line.strip().split('|') for line in scheduler.stdout.splitlines() if line.strip()]
    if rows != [[NATIVE_JOB, 'COMPLETED', '0:0']]:
        raise ValueError('Matched small requires terminal successful unprofiled native screen')
    if (summary.get('runtime_working_directory') != str(NATIVE_OUTPUT)
            or summary.get('core_dump_limit_bytes') != 0
            or summary != json.loads((NATIVE_OUTPUT / 'summary.json').read_text())
            or validation.sha256(NATIVE_OUTPUT / 'summary.json') != NATIVE_SHA256):
        raise ValueError('Native summary differs from pinned scratch execution evidence')
    expected_screen = {**screen, 'summary': str((NATIVE_OUTPUT / 'summary.json').resolve()),
                       'summary_sha256': NATIVE_SHA256}
    if expected_screen != json.loads((NATIVE_OUTPUT / 'screen.json').read_text()):
        raise ValueError('Saved native screen differs from independent assessment')
    current = builds['native_build_admission']
    before = json.loads((NATIVE_LAUNCHER / 'source-audit-before.json').read_text())
    after = json.loads((NATIVE_LAUNCHER / 'source-audit-after.json').read_text())
    if before != after or current != before:
        raise ValueError('Current source/build admission differs from before/after native run')
    return {'job_id': NATIVE_JOB, 'scheduler': scheduler.stdout.strip(),
            'summary_sha256': NATIVE_SHA256,
            'source_audit_before_sha256': validation.sha256(NATIVE_LAUNCHER / 'source-audit-before.json'),
            'source_audit_after_sha256': validation.sha256(NATIVE_LAUNCHER / 'source-audit-after.json')}


def patched_source_hashes(original):
    result = dict(original)
    text = (CAMPAIGN / 'source' / reference_patch.RELATIVE).read_text()
    result[reference_patch.RELATIVE] = hashlib.sha256(reference_patch.transform(text).encode()).hexdigest()
    return result


def stage_source(isolated, original, expected_source):
    isolated = isolated.resolve()
    if isolated == SCRATCH.resolve() or not isolated.is_relative_to(SCRATCH.resolve()):
        raise ValueError('Stage the modified harness only inside dedicated scratch')
    source = isolated / 'source'
    shutil.copytree(CAMPAIGN / 'source', source)
    if validation.source_hashes(source) != original:
        raise ValueError('Initial scratch source copy differs')
    target = source / reference_patch.RELATIVE
    target.write_text(reference_patch.transform(target.read_text()))
    if validation.source_hashes(source) != expected_source:
        raise ValueError('Scratch patch changed unintended source')
    return source


def checked_matched(reports, size, builds, expected_source):
    if [r['tag'] for r in reports] != list(TAGS):
        raise ValueError('Require ordered fast-reference matched conditions')
    for index, report in enumerate(reports):
        candidate = index == 1
        expected = builds['rebuilt']['rebuilt'] if candidate else builds['parent']
        provenance, assessment = report['provenance'], report['assessment']
        if assessment['pair_count'] != {1: 128, 2: 256}[size] or assessment['gpu_count'] != 4:
            raise ValueError('Wrong encrypted workload size or GPU count')
        if provenance['arguments']['size'] != size or provenance['source_hashes'] != expected_source:
            raise ValueError('Wrong measured source or workload')
        if provenance.get('core_dump_limit_bytes') != 0:
            raise ValueError('Missing native core-dump guard')
        cwd = Path(provenance['runtime_working_directory']).resolve()
        if cwd == SCRATCH.resolve() or not cwd.is_relative_to(SCRATCH.resolve()) or str(cwd) != provenance['source']:
            raise ValueError('Native working directory must be its scratch source snapshot')
        for key in ('binary_sha256', 'runtime_sha256'):
            if provenance[key] != expected[key]:
                raise ValueError('Wrong measured native build')
        settings = matched_settings('host_keys', 4, host_key_cache_gib=24)
        if any(provenance['settings'].get(key) != value for key, value in settings.items()):
            raise ValueError('Wrong encrypted cache/thread settings')
    return assess_matched(reports)


def check_encrypted_small(small, builds, expected_source, reference_hashes):
    if (small.get('size') != 1 or small.get('candidate_variant') != VARIANT
            or small.get('promotion_gate_passed') is not True
            or small.get('build_audit') != builds
            or small.get('patched_source_hashes') != expected_source
            or small.get('reference_evidence_sha256') != reference_hashes
            or small.get('launcher_sha256') != validation.sha256(Path(__file__))):
        raise ValueError('Medium requires passing unchanged matched encrypted small evidence')
    if checked_matched(small['conditions'], 1, builds, expected_source)['promotion_gate_passed'] is not True:
        raise ValueError('Recomputed encrypted small gate failed')


def smoke_admission(smoke, small, expected_source, small_sha256):
    if (smoke.get('correctness_passed') is not True or smoke.get('pair_count') != 128
            or smoke.get('patch_sha256') != PATCH_SHA256
            or smoke.get('source_sha256') != expected_source
            or smoke.get('reference_sha256') != small_sha256
            or smoke.get('core_dump_limit_bytes') != 0):
        raise ValueError('Require matching actual patched-loader small evidence')
    scores = smoke['scores']
    if len(scores) != 128 or not all(math.isfinite(v) and -1 <= v <= 1 for v in scores):
        raise ValueError('Invalid patched-loader scores')
    if max(abs(a - b) for a, b in zip(scores, small['conditions'][0]['scores'])) > 1e-6:
        raise ValueError('Patched-loader scores differ from the default reference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', required=True, type=int, choices=[1, 2])
    parser.add_argument('--native-summary', required=True, type=Path)
    parser.add_argument('--reference-small', required=True, type=Path)
    parser.add_argument('--reference-medium', required=True, type=Path)
    parser.add_argument('--smoke-evidence', required=True, type=Path)
    parser.add_argument('--small-evidence', type=Path)
    parser.add_argument('--preflight-only', action='store_true',
                        help='Run read-only admission checks without launching or staging work')
    args = parser.parse_args()
    os.umask(0o077)
    for name in ('reference_small', 'reference_medium', 'smoke_evidence', 'small_evidence', 'native_summary'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve(strict=True))
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdecimal() and not args.preflight_only:
        raise ValueError('Require a Slurm allocation')
    native_summary = checked_native_summary(args.native_summary)
    reference_small = json.loads(args.reference_small.read_text())
    reference_medium = json.loads(args.reference_medium.read_text())
    reference_hashes = {'small': validation.sha256(args.reference_small), 'medium': validation.sha256(args.reference_medium),
                        'patched_smoke': validation.sha256(args.smoke_evidence)}
    reference_admission(reference_small, reference_medium, reference_hashes['small'])
    reference = reference_medium['conditions'][0]
    if any(importlib.metadata.version(name) != version for name, version in reference['packages'].items()):
        raise ValueError('Python reference packages differ from validated environment')
    router = Path(importlib.metadata.distribution('insightface').locate_file('insightface/model_zoo/model_zoo.py'))
    if validation.sha256(router) != reference['library_source_sha256']['model_zoo.py']:
        raise ValueError('Installed reference session router changed')
    if {p.name: validation.sha256(p) for p in reference_gate.MODELS.glob('*.onnx')} != reference['model_sha256']:
        raise ValueError('Existing reference model inputs changed; do not download replacements')
    builds = build_audit()
    native = native_admission(native_summary, builds)
    native['completed_screen_evidence'] = check_completed_native_evidence(native_summary, builds, native)
    reference_hashes['native'] = NATIVE_SHA256
    original = validation.source_hashes(CAMPAIGN / 'source')
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    if original != manifest['measured_build']['source_hashes']:
        raise ValueError('Shared application differs from frozen source')
    prior = json.loads((CAMPAIGN / 'results-matched-1-17141256.json').read_text())
    expected_dataset = prior['conditions'][0]['provenance']['input_store_sha256']
    dataset = SCRATCH / 'CryptoFace-Latti-GPU/datasets/face_dataset.h5'
    if validation.sha256(dataset) != expected_dataset:
        raise ValueError('Encrypted dataset differs from the retained reference workloads')
    expected_source = patched_source_hashes(original)
    smoke_admission(json.loads(args.smoke_evidence.read_text()), reference_small, expected_source, reference_hashes['small'])
    small = None
    if args.size == 2:
        if args.small_evidence is None:
            raise ValueError('Medium requires matched encrypted small evidence')
        small = json.loads(args.small_evidence.read_text())
        check_encrypted_small(small, builds, expected_source, reference_hashes)
    if args.preflight_only:
        print(json.dumps({'preflight_passed': True, 'size': args.size,
                          'build_audit': builds, 'native_gate': native,
                          'reference_evidence_sha256': reference_hashes,
                          'patched_source_hashes': expected_source}, indent=2))
        return
    work = SCRATCH / f'reproducible/converted-cache-analysis/ntt-generic-twiddle-fast-reference-matched-{args.size}-{job}'
    work.mkdir(parents=True, exist_ok=False)
    prepare_runtime(work)
    isolated = work / 'campaign'
    source = stage_source(isolated, original, expected_source)
    previous = (validation.CAMPAIGN, dict(validation.VARIANT_FAMILIES), sys.argv, validation.candidate_settings)
    reports = []
    try:
        validation.CAMPAIGN = isolated
        validation.candidate_settings = matched_settings
        for index, tag in enumerate(TAGS):
            measured_build = REBUILT if index == 1 else PARENT
            validation.VARIANT_FAMILIES['host_keys'] = str(measured_build.relative_to(SCRATCH / 'reproducible'))
            sys.argv = [str(Path(validation.__file__)), '--size', str(args.size), '--threads', '4',
                        '--variant', 'host_keys',
                        '--host-key-cache-gib', '24', '--run-tag', tag]
            validation.main()
            name = validation.validation_run_name(args.size, job, tag)
            measured = isolated / 'results' / name
            durable = HERE / 'results' / name
            durable.mkdir(parents=True, exist_ok=False)
            for filename in ('summary.json', 'provenance.json', 'results-1.json'):
                shutil.copyfile(measured / filename, durable / filename)
            reports.append({'tag': tag, 'directory': str(durable),
                            'assessment': json.loads((durable / 'summary.json').read_text()),
                            'provenance': json.loads((durable / 'provenance.json').read_text())})
            if (build_audit() != builds or validation.source_hashes(source) != expected_source
                    or validation.source_hashes(CAMPAIGN / 'source') != original):
                raise ValueError('Source or build changed during validation')
            if (validation.sha256(dataset) != expected_dataset or
                    {p.name: validation.sha256(p) for p in reference_gate.MODELS.glob('*.onnx')} != reference['model_sha256']):
                raise ValueError('Dataset or reference model inputs changed during validation')
    finally:
        validation.CAMPAIGN, families, sys.argv, validation.candidate_settings = previous
        validation.VARIANT_FAMILIES.clear()
        validation.VARIANT_FAMILIES.update(families)
    assessment = checked_matched(reports, args.size, builds, expected_source)
    if any(report['provenance']['input_store_sha256'] != expected_dataset for report in reports):
        raise ValueError('Encrypted dataset changed after admission')
    if small is not None:
        for current, prior in zip(reports, small['conditions']):
            if current['provenance']['input_store_sha256'] != prior['provenance']['input_store_sha256']:
                raise ValueError('Encrypted dataset changed since small')
    report = {'schema_version': 1, 'size': args.size, 'candidate_variant': VARIANT,
              'build_audit': builds, 'patched_source_hashes': expected_source,
              'reference_evidence_sha256': reference_hashes, 'native_gate': native, 'conditions': reports,
              'launcher_sha256': validation.sha256(Path(__file__)), **assessment}
    destination = HERE / f'results-ntt-generic-twiddle-fast-reference-{args.size}-{job}.json'
    with destination.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({'result': str(destination), **assessment}), flush=True)


if __name__ == '__main__':
    main()
