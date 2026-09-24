"""CPU-only contract/metadata tests; no synthetic inference inputs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark


class ContractTests(unittest.TestCase):
    def test_release_and_upstream_hashes(self):
        release = benchmark.verify_release()
        self.assertEqual(release['upstream_revision'], 'a7c12d34680bdd39198a349cf1ebeec89ae88061')

    def test_upstream_params_contract(self):
        spec = importlib.util.spec_from_file_location('upstream_params', ROOT / 'harness/params.py')
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        with tempfile.TemporaryDirectory() as directory:
            for size, pairs in enumerate((1, 128, 256, 1024)):
                instance = params.InstanceParams(size, directory)
                self.assertEqual(instance.get_batch_size(), pairs)
                self.assertEqual(instance.iodir().parent, Path(directory) / 'io')
                self.assertFalse(hasattr(instance, 'io_root'))

    def test_submission_has_no_private_harness_api_dependency(self):
        for filename in ('client_key_generation.py', 'server_preprocess_model.py'):
            self.assertNotIn('.io_root()', (ROOT / 'submission' / filename).read_text())

    def test_parameter_files_identical_to_validated_backend(self):
        for side in ('client', 'server'):
            relative = f'submission/latti/model/{side}/ckks_parameter.json'
            self.assertEqual((ROOT / relative).read_bytes(), (ROOT / 'backend/base/code' / relative).read_bytes())

    def test_only_expected_adapter_and_matching_files_differ(self):
        base = ROOT / 'backend/base/code/submission'
        changed = {p.relative_to(base).as_posix() for p in base.rglob('*')
                   if p.is_file() and p.read_bytes() != (ROOT / 'submission' / p.relative_to(base)).read_bytes()}
        self.assertEqual(changed, {'gpu_runtime.py', 'client_key_generation.py', 'server_preprocess_model.py',
                                  'client_decrypt_decode.py', 'server_encrypted_compute.py',
                                  'test_streamed_encrypted_compute.py', 'test_gpu_runtime.py',
                                  'native/build_runtime.sh', 'native/latti_stage_runtime.cpp'})

    def test_timing_units(self):
        self.assertAlmostEqual(benchmark.seconds('955.6222885912284s'), 955.6222885912284)
        self.assertEqual(benchmark.seconds('2m'), 120)
        self.assertEqual(benchmark.seconds('3ms'), .003)
        for value in ('nan', float('nan'), float('inf'), True, -1, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.seconds(value)

    def test_scratch_rejects_home_source_and_root(self):
        for root in (Path.home(), ROOT, Path('/')):
            with self.subTest(root=root), self.assertRaises(ValueError):
                benchmark.scratch_path(root / 'unsafe', root)

    def test_run_environment_removes_experiment_overrides(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'CRYPTOFACE_PAIR_LIMIT': '1', 'CRYPTOFACE_RUN_OFFSET': '2',
            'LATTISENSE_GPU_TWIDDLE_SHOUP_VERIFY': '1', 'PYTHONPATH': '/untrusted'}):
            env = benchmark.execution_environment(Path(directory), {'LATTISENSE_GPU_TWIDDLE_SHOUP_VERIFY': '0'})
            self.assertNotIn('CRYPTOFACE_PAIR_LIMIT', env)
            self.assertNotIn('CRYPTOFACE_RUN_OFFSET', env)
            self.assertNotIn('PYTHONPATH', env)
            self.assertEqual(env['LATTISENSE_GPU_TWIDDLE_SHOUP_VERIFY'], '0')
            self.assertEqual(env.get('HOME'), os.environ.get('HOME'))

    def test_medium_without_small_refused_before_launch(self):
        # Pure metadata fixture, never used as an inference input.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'incomplete.json'
            with path.open('w') as stream:
                json.dump({'completed': False, 'size': 1}, stream)
            with self.assertRaises(ValueError):
                benchmark.check_previous(path, {}, 2)


if __name__ == '__main__':
    unittest.main()
