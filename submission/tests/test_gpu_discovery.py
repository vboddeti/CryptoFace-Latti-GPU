"""GPU allocation discovery without invoking CUDA or inference."""
import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('submission_gpu_runtime', Path(__file__).resolve().parents[1] / 'gpu_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class GpuDiscoveryTests(unittest.TestCase):
    def test_auto_selection_respects_visible_device_allocation(self):
        config = {'minimum_gpu_free_memory_mib': 140000, 'maximum_gpu_utilization_percent': 0,
                  'required_gpu_compute_capability': '9.0'}
        inventory = '0, 143000, 9.0, 0\n1, 143000, 9.0, 0\n2, 143000, 9.0, 20\n'
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,2'}, clear=True), \
             patch.object(runtime.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, inventory)):
            self.assertEqual(runtime.gpu_ids(config), [1])

    def test_uuid_mask_is_mapped_to_physical_gpu(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-second'}, clear=True), \
             patch.object(runtime.subprocess, 'check_output', return_value='0, GPU-first-id\n2, GPU-second-id\n'):
            self.assertEqual(runtime._visible_gpu_ids(), {2})

    def test_explicit_override_cannot_escape_allocation(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1', 'CRYPTOFACE_GPUS': '0'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Invalid GPU list'):
                runtime.gpu_ids({})

    def test_hidden_gpus_are_not_treated_as_unrestricted(self):
        for mask in ('', '-1'):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}, clear=True):
                self.assertEqual(runtime._visible_gpu_ids(), set())


if __name__ == '__main__':
    unittest.main()
