#!/usr/bin/env python3
"""Real-input gate for a frozen candidate; every working file stays in scratch."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from scratch_runtime import prepare_runtime

CAMPAIGN = Path(__file__).resolve().parent
SCRATCH = Path('/mnt/gs21/scratch/vishnu/cryptoface')
HISTORICAL = {1: 5.0779487266936485, 2: 4.905102923507911}
VARIANT_FAMILIES = {'tuning': 'hoisted-latency-v2', 'batched': 'batched-rotation-intt-v1',
                    'reclaim': 'reclaim-pool-v1', 'checked_plaintext': 'checked-plaintext-cache-v1',
                    'checked_converted': 'checked-converted-plaintext-v2',
                    'checked_compound': 'checked-converted-compound-v3',
                    'host_keys': 'scoped-host-key-cache-v1',
                    'parent': 'rotation-hoisting-v1'}


def candidate_settings(variant, threads, batch_intt=0, reclaim_threads=0, plaintext_cache_gib=0,
                       converted_cache_gib=0, host_key_cache_gib=0):
    if variant not in VARIANT_FAMILIES:
        raise ValueError('Unknown candidate variant')
    if variant == 'parent' and threads != 4:
        raise ValueError('Immutable parent uses four submission threads')
    if host_key_cache_gib not in [0, 24] or (host_key_cache_gib and variant != 'host_keys'):
        raise ValueError('Host key reuse requires its isolated build and a 24-GiB cap')
    if batch_intt and variant != 'batched':
        raise ValueError('Batched INTT requires its separate build')
    if reclaim_threads and variant != 'reclaim':
        raise ValueError('Reclaim workers require their separate build')
    if plaintext_cache_gib and variant not in ['checked_plaintext', 'checked_converted', 'checked_compound']:
        raise ValueError('Checked plaintext cache requires its separate build')
    if converted_cache_gib and (variant not in ['checked_converted', 'checked_compound'] or not plaintext_cache_gib):
        raise ValueError('Converted cache requires its separate build and checked raw cache')
    if converted_cache_gib not in [0, 2, 4, 8]:
        raise ValueError('Invalid converted cache cap')
    if threads not in [4, 8, 12, 16] or batch_intt not in [0, 1]:
        raise ValueError('Invalid execution configuration')
    if reclaim_threads not in [0, 2, 4] or plaintext_cache_gib not in [0, 4, 8, 16]:
        raise ValueError('Invalid reclaim thread count or cache cap')
    return {'LATTISENSE_GPU_SUBMISSION_THREADS': str(threads),
            'HEONGPU_ROTATE_MANY_BATCH_INTT': str(batch_intt),
            'LATTISENSE_GPU_RECLAIM_THREADS': str(reclaim_threads),
            'LATTISENSE_GPU_PLAINTEXT_CACHE_MAX_BYTES': str(plaintext_cache_gib << 30),
            'LATTISENSE_GPU_CONVERTED_PLAINTEXT_MAX_BYTES': str(converted_cache_gib << 30),
            'LATTISENSE_GPU_HOST_KEY_CACHE_MAX_BYTES': str(host_key_cache_gib << 30),
            'LATTISENSE_GPU_IMMUTABLE_EVAL_KEYS': '1' if host_key_cache_gib else '0',
            'LATTISENSE_GPU_ABI_CACHE_MAX_BYTES': '0',
            'LATTISENSE_GPU_KEY_EXPORT_PROFILE': '0'}


def validation_run_name(size, job, tag=''):
    if size not in [1, 2] or not str(job).isdecimal():
        raise ValueError('Require small/medium size and numeric Slurm job ID')
    if tag and (len(tag) > 64 or not tag.replace('-', '').replace('_', '').isalnum()):
        raise ValueError('Unsafe validation run tag')
    return f"four-gpu-{ {1: 'small', 2: 'medium'}[size]}-{job}" + (f'-{tag}' if tag else '')


def converted_cache_activity(log, expected_cap):
    matches = [tuple(map(int, fields)) for fields in re.findall(
        r'\[GPU Converted Plaintext Cache\] cap=(\d+) used=(\d+) hits=(\d+) misses=(\d+)', log)]
    if not matches or any(cap != expected_cap for cap, _, _, _ in matches):
        raise ValueError('Missing or mismatched converted-cache runtime marker')
    if any(used > cap for cap, used, _, _ in matches):
        raise ValueError('Converted cache exceeded its configured cap')
    used, hits, misses = [max(row[index] for row in matches) for index in [1, 2, 3]]
    if min(used, hits, misses) <= 0:
        raise ValueError('Converted cache did not demonstrate real reuse')
    return {'cap_bytes': expected_cap, 'used_bytes': used, 'hits': hits, 'misses': misses}


def rotation_hoisting_activity(log):
    markers = re.findall(
        r'\[GPU Rotation Hoisting\] enabled=(\d+) batches=(\d+) '
        r'grouped_rotations=(\d+) removed_compute_nodes=(\d+)', log)
    if not markers:
        raise ValueError('Missing rotation-hoisting runtime marker')
    enabled, batches, grouped, removed = map(int, markers[-1])
    if enabled != 1 or min(batches, grouped, removed) <= 0:
        raise ValueError('Rotation hoisting inactive or missing positive activity counters')
    return {'enabled': True, 'batches': batches, 'grouped_rotations': grouped,
            'removed_compute_nodes': removed}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def host_key_cache_activity(log, expected_cap):
    rows = [tuple(map(int, fields)) for fields in re.findall(
        r'\[Host Key Cache\] cap=(\d+) used=(\d+) entries=(\d+) generation=(\d+) '
        r'hits=(\d+) misses=(\d+) rejected=(\d+)', log)]
    if len(rows) < 2:
        raise ValueError('Missing host key cache reuse evidence')
    first = rows[0]
    if first[2] != 53 or first[4:6] != (0, 53):
        raise ValueError('Unexpected cold key-cache coverage')
    for index, row in enumerate(rows):
        cap, used, entries, generation, hits, misses, rejected = row
        if cap != expected_cap or not 0 < used <= cap or rejected != 0:
            raise ValueError('Invalid host key-cache capacity/admission')
        if row[1:4] != first[1:4] or (index and (hits, misses) != (53, 0)):
            raise ValueError('Host key cache did not retain/reuse all immutable keys')
    return {'cap_bytes': expected_cap, 'used_bytes': first[1], 'entries': first[2],
            'requests': len(rows), 'hot_hits': sum(row[4] for row in rows[1:]),
            'hot_misses': sum(row[5] for row in rows[1:])}


def source_hashes(source):
    files = sorted(p for p in source.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Application snapshot must contain only actual source/input files')
    return {str(p.relative_to(source)): sha256(p) for p in files}


def assess_result(result, size):
    if size not in HISTORICAL:
        raise ValueError('Only small and medium are supported')
    if result['Quality']['Comparison to ArcFace baseline'].get('passed') is not True:
        raise ValueError('ArcFace correctness gate failed')
    server = result['Server Reported']
    additional = server['additional_measurements']
    pairs = {1: 128, 2: 256}[size]
    if additional['Pair count'] != pairs or additional['workers']['gpu_count'] != 4:
        raise ValueError('Wrong pair count or GPU count')
    seconds = float(str(server['Encrypted computation']).removesuffix('s'))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid computation time')
    interval = seconds / pairs
    return {'correctness_passed': True, 'pair_count': pairs, 'gpu_count': 4,
            'encrypted_computation_seconds': seconds, 'compute_seconds_per_pair': interval,
            'historical_reference_seconds_per_pair': HISTORICAL[size],
            'historical_latency_reduction_fraction': 1 - interval / HISTORICAL[size],
            'comparison_caveat': 'Historical reference, not a same-allocation paired control.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, choices=[1, 2], required=True)
    parser.add_argument('--variant', choices=list(VARIANT_FAMILIES), required=True)
    parser.add_argument('--threads', type=int, choices=[4, 8, 12, 16], required=True)
    parser.add_argument('--batch-intt', type=int, choices=[0, 1], default=0)
    parser.add_argument('--reclaim-threads', type=int, choices=[0, 2, 4], default=0)
    parser.add_argument('--plaintext-cache-gib', type=int, choices=[0, 4, 8, 16], default=0)
    parser.add_argument('--converted-cache-gib', type=int, choices=[0, 2, 4, 8], default=0)
    parser.add_argument('--host-key-cache-gib', type=int, choices=[0, 24], default=0)
    parser.add_argument('--run-tag', default='', help='Unique label for sequential matched runs')
    args = parser.parse_args()
    try:
        candidate = candidate_settings(args.variant, args.threads, args.batch_intt,
                                       args.reclaim_threads, args.plaintext_cache_gib,
                                       args.converted_cache_gib, args.host_key_cache_gib)
    except ValueError as exc:
        parser.error(str(exc))
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdecimal():
        parser.error('Run inside a Slurm allocation with a numeric job ID')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or not all(devices):
        parser.error('Require exactly four allocated visible GPUs')
    size_name = {1: 'small', 2: 'medium'}[args.size]
    try:
        run_name = validation_run_name(args.size, job, args.run_tag)
    except ValueError as exc:
        parser.error(str(exc))
    work = SCRATCH / 'reproducible/hoisted-latency-v2' / run_name
    work.mkdir(parents=True, exist_ok=False)
    artifact = work / 'artifacts'
    artifact.mkdir()
    family = VARIANT_FAMILIES[args.variant]
    build = SCRATCH / 'reproducible' / family
    for name in ['.deps', 'submission-build']:
        (artifact / name).symlink_to((build / name).resolve(strict=True), target_is_directory=True)
    (artifact / 'datasets').mkdir()
    dataset = SCRATCH / 'CryptoFace-Latti-GPU/datasets/face_dataset.h5'
    (artifact / 'datasets/face_dataset.h5').symlink_to(dataset.resolve(strict=True))
    for directory in ['tmp', 'cache', 'cuda-cache', 'measurements']:
        (work / directory).mkdir()
    # Relative native writes must not reach the immutable home source tree.
    original_source = CAMPAIGN / 'source'
    expected_source_hashes = source_hashes(original_source)
    source = work / 'source'
    shutil.copytree(original_source, source)
    if source_hashes(source) != expected_source_hashes:
        raise RuntimeError('Scratch source snapshot differs from original')
    prepare_runtime(work)
    env = os.environ.copy()
    settings = {
        'CRYPTOFACE_ARTIFACT_ROOT': str(artifact),
        'CRYPTOFACE_BUILD_DIR': str(artifact / 'submission-build'),
        'CRYPTOFACE_GPUS': '0,1,2,3', 'CRYPTOFACE_GPU_HOIST_ROTATIONS': '1',
        'CRYPTOFACE_GPU_KEY_CACHE': '1', 'CRYPTOFACE_GPU_EVENT_CACHE': '0',
        'CRYPTOFACE_STREAM_SCORING': '1', 'CRYPTOFACE_KEEP_STAGE_INPUTS': '0',
        'CRYPTOFACE_FACE_CPU_THREADS': '8', 'CRYPTOFACE_RUN_OFFSET': '0',
        'CRYPTOFACE_MEASUREDIR': str(work / 'measurements'),
        'LATTISENSE_GPU_SUBMISSION_THREADS': str(args.threads),
        'HEONGPU_ROTATE_MANY_BATCH_INTT': str(args.batch_intt),
        'HEONGPU_ROTATE_MANY_SUFFIX_STREAMS': '1',
        'LATTISENSE_GPU_PROFILE_HOT': '0', 'LATTISENSE_GPU_TIMING': '1',
        'LATTISENSE_GPU_SYNC_DISABLE_TIMING': '0', 'LATTISENSE_GPU_SYNC_SHARE_BATCH_EVENT': '0',
        'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(work / 'tmp'),
        'XDG_CACHE_HOME': str(work / 'cache'), 'CUDA_CACHE_PATH': str(work / 'cuda-cache'),
    }
    settings.update(candidate)
    env.update(settings)
    runtime = artifact / '.deps/latti-ai/build-h200/inference/lattisense/liblattisense.so'
    manifest = artifact / 'submission-build/build_manifest.json'
    native = artifact / 'submission-build/latti_stage_runtime'
    provenance = {'job_id': job, 'host': os.uname().nodename,
                  'source': str(source), 'source_hashes': source_hashes(source),
                  'settings': settings, 'arguments': vars(args), 'seed': 42,
                  'binary_sha256': sha256(native), 'runtime_sha256': sha256(runtime),
                  'build_manifest': json.loads(manifest.read_text()),
                  'input_store': str(dataset), 'input_store_sha256': sha256(dataset),
                  'launcher_sha256': sha256(__file__)}
    provenance['original_source'] = str(original_source)
    provenance['runtime_guard_sha256'] = sha256(Path(__file__).with_name('scratch_runtime.py'))
    provenance['runtime_working_directory'] = str(source)
    provenance['core_dump_limit_bytes'] = 0
    provenance['gpu_inventory'] = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,power.limit', '--format=csv,noheader'], text=True)
    (work / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    command = [sys.executable, '-B', 'harness/run_submission.py', str(args.size), '--seed', '42', '--num_runs', '1']
    with (work / 'harness.log').open('w') as log:
        subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result_path = work / 'measurements/results-1.json'
    result = json.loads(result_path.read_text())
    assessment = assess_result(result, args.size)
    if (source_hashes(source) != provenance['source_hashes'] or
            source_hashes(original_source) != provenance['source_hashes']):
        raise RuntimeError('Application source changed during measurement')
    if sha256(native) != provenance['binary_sha256'] or sha256(runtime) != provenance['runtime_sha256']:
        raise RuntimeError('Native build changed during measurement')
    cache_activity = {}
    hoisting_activity = {}
    for gpu in range(4):
        log_path = artifact / f'io/{size_name}/logs/server_gpu_{gpu}.log'
        log_text = log_path.read_text()
        try:
            hoisting_activity[str(gpu)] = rotation_hoisting_activity(log_text)
        except ValueError as error:
            raise RuntimeError(f'Invalid rotation-hoisting evidence on worker {gpu}: {error}') from error
        if args.converted_cache_gib:
            cache_activity[str(gpu)] = converted_cache_activity(log_text, args.converted_cache_gib << 30)
    assessment['rotation_hoisting_activity'] = hoisting_activity
    if cache_activity:
        assessment['converted_cache_activity'] = cache_activity
    if args.host_key_cache_gib:
        assessment['host_key_cache'] = {
            str(gpu): host_key_cache_activity(
                (artifact / f'io/{size_name}/logs/server_gpu_{gpu}.log').read_text(),
                args.host_key_cache_gib << 30)
            for gpu in range(4)}
    # Only correctness-passing final evidence is retained in home. No logs/caches/I/O.
    destination = CAMPAIGN / 'results' / run_name
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'results-1.json').write_bytes(result_path.read_bytes())
    (destination / 'summary.json').write_text(json.dumps(assessment, indent=2) + '\n')
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'result': str(destination), **assessment}, indent=2), flush=True)


if __name__ == '__main__':
    main()
