#!/usr/bin/env python3
"""Read-only publication audit; stdout is a final JSON summary, never runtime data."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
TARGET = 'https://github.com/vboddeti/CryptoFace-Latti-GPU'
BRANCH = 'main'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def seconds(value):
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)', value)
    if not match:
        raise ValueError(f'Unexpected time: {value}')
    result = float(match[1]) * {'ms': .001, 's': 1, 'm': 60, 'h': 3600}[match[2]]
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid timing')
    return result


def summarize(variant, size, count, pairs):
    directory = REPO / 'measurements' / variant
    evidence = load(directory / 'provenance.json')
    assert evidence['completed'] is True
    assert evidence['size'] == size and evidence['repetitions'] == count
    assert evidence['three_run_measurement_complete'] is (count == 3)
    assert evidence['command'][-4:] == ['--num_runs', str(count), '--seed', '42']
    assert evidence['prepared']['release_sha256'] == sha(REPO / 'release.json')
    assert sha(directory / 'release.json') == sha(REPO / 'release.json')
    assert len(evidence['results']) == count
    results = []
    identities = []
    for number, recorded in enumerate(evidence['results'], start=1):
        path = directory / f'results-{number}.json'
        assert recorded['path'] == f'measurements/{variant}/{path.name}'
        assert sha(path) == recorded['sha256']
        result = load(path)
        server = result['Server Reported']
        assert server['additional_measurements']['Pair count'] == pairs
        assert server['additional_measurements']['workers']['gpu_count'] == 4
        compute = seconds(server['Encrypted computation']) / pairs
        assert math.isclose(compute, recorded['compute_seconds_per_pair'], rel_tol=1e-12)
        assert math.isclose(seconds(result['Timing']['Total']), recorded['harness_total_seconds'], rel_tol=1e-12)
        if count == 3:
            assert result['Quality']['Comparison to ArcFace baseline']['passed'] is True
        else:
            assert math.isfinite(result['Quality']['Encrypted model quality']['score'])
        results.append(result)
        identities.append({'file': f'measurements/{variant}/{path.name}',
                           'sha256': sha(path), 'compute_seconds_per_pair': compute})
    assert all(result['Bandwidth'] == results[0]['Bandwidth'] for result in results)
    timing = {key: mean(seconds(r['Timing'][key]) for r in results)
              for key in results[0]['Timing']}
    server = {key: mean(seconds(r['Server Reported'][key]) for r in results)
              for key in ('Total', 'Encrypted computation')}
    quality = ({model: {key: mean(r['Quality'][model][key] for r in results)
                         for key in ('eer', 'tar_at_far_1pct', 'tar_at_far_01pct')}
                for model in ('Encrypted model quality', 'Harness model quality')}
               if count == 3 else results[0]['Quality'])
    return {'variant': variant, 'pairs_per_run': pairs, 'num_runs': count,
            'completed_utc': evidence['finished_utc'], 'slurm_job_id': evidence['slurm_job_id'],
            'release_sha256': evidence['prepared']['release_sha256'],
            'dataset_sha256': evidence['prepared']['dataset_sha256'],
            'native': evidence['prepared']['native'],
            'bandwidth_reported': results[0]['Bandwidth'],
            'mean_harness_seconds': timing, 'mean_server_seconds': server,
            'mean_compute_seconds_per_pair': server['Encrypted computation'] / pairs,
            'quality_mean_of_runs': quality, 'raw_results': identities}


def main():
    spec = importlib.util.spec_from_file_location('measured_benchmark', REPO / 'scripts/benchmark.py')
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    benchmark.verify_release(REPO)
    variants = {'small': summarize('small', 1, 3, 128)}
    output = {'schema_version': 1, 'publication_status': 'small-only submission; acceptance tracked by benchmark PR',
              'intended_repository': TARGET, 'intended_branch': BRANCH,
              'upstream_harness_commit': load(REPO / 'release.json')['upstream_revision'],
              'aggregation': 'arithmetic mean of per-run metrics; not pooled quality; no setup subtraction',
              'variants': variants,
              'not_reported': {'single': 'outside this submission',
                               'medium': 'outside this submission',
                               'large': 'not measured with this release'}}
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
