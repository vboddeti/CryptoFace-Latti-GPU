#!/usr/bin/env python3
"""Verify published raw results against measured-source provenance; print means."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seconds(value):
    return float(value.removesuffix('s'))


def summarize(root):
    source = root
    release = load(source / 'release.json')
    release_hash = digest(source / 'release.json')
    for name, expected in release['files'].items():
        if digest(source / name) != expected:
            raise ValueError(f'Measured source changed: {name}')
    measured_path = root / 'measurements/small/release.json'
    measured = load(measured_path)
    measured_hash = digest(measured_path)
    for name, expected in measured['files'].items():
        if name not in ('README.md', 'GPU_MATCHING.md') and digest(root / name) != expected:
            raise ValueError(f'Measured executable/backend source differs: {name}')
    diagnostic = load(root / 'measurements/validation/provenance.json')
    check = diagnostic['matching_validation']
    if (not diagnostic['completed'] or not diagnostic['matching_diagnostic']
            or check['passed'] is not True or check['absolute_tolerance'] != 1e-6
            or check['pairs'] != 1 or check['max_absolute_error'] > 1e-6):
        raise ValueError('Missing matching correctness evidence')
    summary = {'publication_release_sha256': release_hash,
               'measured_release_sha256': measured_hash,
               'upstream_harness_revision': release['upstream_revision'],
               'matching_correctness': check, 'workloads': {}}
    for name, pairs in [('single', 1), ('small', 128)]:
        folder = root / 'measurements' / name
        provenance = load(folder / 'provenance.json')
        if (provenance['completed'] is not True or provenance['repetitions'] != 3
                or provenance['matching_diagnostic']
                or provenance['three_run_measurement_complete'] is not True
                or len(provenance['results']) != 3):
            raise ValueError(f'Incomplete official runset: {name}')
        if provenance['prepared']['release_sha256'] != measured_hash:
            raise ValueError('Results belong to another release')
        for key in ('release_sha256', 'native', 'dataset_sha256', 'settings'):
            if provenance['prepared'][key] != diagnostic['prepared'][key]:
                raise ValueError(f'Diagnostic identity differs: {key}')
        results = []
        for index, evidence in enumerate(provenance['results'], 1):
            path = folder / f'results-{index}.json'
            if evidence['path'] != f'measurements/{name}/results-{index}.json':
                raise ValueError('Unexpected raw-result mapping')
            if digest(path) != evidence['sha256']:
                raise ValueError(f'Raw result changed: {path}')
            result = load(path)
            details = result['Server Reported']['additional_measurements']
            if details['Matching backend'] != 'gpu' or details['Pair count'] != pairs:
                raise ValueError('Wrong matching backend or workload size')
            if name == 'small' and result['Quality']['Comparison to ArcFace baseline']['passed'] is not True:
                raise ValueError('Benchmark quality gate failed')
            results.append(result)
        compute = [seconds(r['Server Reported']['Encrypted computation']) for r in results]
        summary['workloads'][name] = {
            'pairs_per_run': pairs, 'num_runs': 3,
            'encrypted_compute_seconds_per_run': compute,
            'encrypted_compute_seconds_per_pair': [v / pairs for v in compute],
            'mean_encrypted_compute_seconds_per_run': mean(compute),
            'mean_encrypted_compute_seconds_per_pair': mean(compute) / pairs,
            'mean_harness_encrypted_stage_seconds_per_pair': mean(seconds(r['Timing']['Encrypted computation']) for r in results) / pairs,
            'mean_harness_total_seconds': mean(seconds(r['Timing']['Total']) for r in results),
            'all_small_quality_checks_passed': True if name == 'small' else None,
        }
    policy = load(root / 'measurements/small/provenance.json')['scheduling_policy']
    if digest(root / 'scripts/run_independent_small.py') != policy['runner_sha256']:
        raise ValueError('Independent-small orchestration source changed')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
