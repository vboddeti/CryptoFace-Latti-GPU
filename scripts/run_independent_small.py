#!/usr/bin/env python3
"""User-authorized parallel small run; numerical and source gates stay intact."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys


def validate_local_gate(validation, prepared, size):
    if size != 1:
        raise ValueError('Independent scheduling is authorized only for small')
    check = validation.get('matching_validation', {})
    if (validation.get('completed') is not True or validation.get('size') != 0
            or validation.get('matching_diagnostic') is not True
            or validation.get('repetitions') != 1 or check.get('passed') is not True
            or check.get('pairs') != 1 or check.get('absolute_tolerance') != 1e-6):
        raise ValueError('Require the successful real-pair local matching diagnostic')
    for key in ('release_sha256', 'native', 'dataset_sha256', 'settings'):
        if validation['prepared'][key] != prepared[key]:
            raise ValueError(f'Local correctness identity differs: {key}')


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--matching-validation', type=Path, required=True)
    parser.add_argument('--runner-sha256', required=True)
    args = parser.parse_args()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != args.runner_sha256:
        raise ValueError('Independent-small launcher changed since submission')
    sys.path.insert(0, str(args.source / 'scripts'))
    spec = importlib.util.spec_from_file_location('benchmark', args.source / 'scripts/benchmark.py')
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    work = benchmark.scratch_path(args.work, args.scratch_root)
    prepared = benchmark.load(work / 'prepared.json')
    validation = benchmark.load(args.matching_validation)
    validate_local_gate(validation, prepared, 1)
    policy = {
        'reason': 'User requested small not wait for the official single benchmark',
        'scope': 'small only; medium and large predecessor requirements unchanged',
        'local_correctness': str(args.matching_validation.resolve()),
        'local_correctness_sha256': benchmark.digest(args.matching_validation),
        'runner_sha256': runner_hash,
        'numerical_source_and_validation_unchanged': True,
    }
    benchmark.write_new(work / 'scheduling-policy.json', policy)
    shutil.copy2(__file__, work / 'independent-small-launcher.py')
    args.size, args.seed, args.smoke, args.matching_check = 1, 42, False, False
    args.previous = args.matching_validation
    original = benchmark.check_previous

    def require_local_check(path, current, size):
        if path != args.matching_validation.resolve():
            raise ValueError('Unexpected correctness prerequisite')
        validate_local_gate(benchmark.load(path), current, size)

    # Override only our orchestration's official-single ordering policy.
    # The sealed runner still verifies source, build, dataset, local numerical
    # validation, three repetitions, GPU execution and completed results.
    benchmark.check_previous = require_local_check
    try:
        benchmark.run(args)
    finally:
        benchmark.check_previous = original
    result_path = work / 'completed.json'
    result = benchmark.load(result_path)
    result['scheduling_policy'] = policy
    result_path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
