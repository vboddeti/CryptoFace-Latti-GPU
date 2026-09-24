"""Static gates only: no synthetic images, ciphertexts or inference."""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MatchingContractTests(unittest.TestCase):
    def test_matching_scale_is_derived_from_ciphertext_metadata(self):
        patch = (ROOT / 'submission/native/latti-gpu-matching.patch').read_text()
        self.assertIn('left.data.front().get_scale()', patch)
        self.assertIn('right.data.front().get_scale()', patch)
        self.assertIn('left_scale * right_scale / divisor', patch)
        self.assertIn('get_parameter().get_q(2)', patch)
        self.assertIn('result.front().set_scale(output_scale)', patch)
        self.assertIn('requires uniform input scales', patch)
        self.assertIn('requires exactly one rescale', patch)

    def test_runtime_metadata_preserves_production_values_and_circuit(self):
        import copy
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'matching_generator', ROOT / 'scripts/generate_matching_task.py')
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        production = json.loads((ROOT / 'submission/latti/model/server/mega_ag.json').read_text())
        graph = copy.deepcopy(production)
        names = ('btp_eval_mod_q', 'btp_eval_mod_message_ratio')
        for name in names:
            del graph['parameter'][name]
        graph['parameter'].pop('btp_output_level')
        before = copy.deepcopy(graph)
        generator.add_runtime_metadata(graph, production)
        for name in names:
            self.assertEqual(graph['parameter'].pop(name), production['parameter'][name])
        self.assertEqual(graph, before)

    def test_model_layout_matches_specialized_circuit(self):
        task = json.loads((ROOT / 'submission/latti/model/server/task_config.json').read_text())
        out = task['task_output_param']['output']
        self.assertEqual((out['channel'], out['skip'], out['level'], out['pack_num']),
                         (256, 1024, 2, 32))

    def test_patch_counts_are_valid(self):
        import re
        lines = (ROOT / 'submission/native/latti-gpu-matching.patch').read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.startswith('@@ '):
                continue
            header = re.match(r'@@ -\d+,(\d+) \+\d+,(\d+) @@', line)
            self.assertIsNotNone(header)
            old = new = 0
            for following in lines[i + 1:]:
                if following.startswith(('@@ ', '--- a/')):
                    break
                old += following.startswith((' ', '-'))
                new += following.startswith((' ', '+'))
            self.assertEqual((old, new), tuple(map(int, header.groups())))

    def test_secret_and_parameter_paths_unchanged(self):
        import hashlib
        expected = {
            'client_key_generation.py': 'd1e1cb87e70a362d55e94a910fac9550e1f2e53f96b1270409aa4d784398066d',
            'client_encode_encrypt_input.py': '6d51201d4a094976cbea860b0eb64896c3751997cb8c0e7d0c486a5bc7bf7574',
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / 'submission' / name).read_bytes()).hexdigest(), digest)
        for side in ('client', 'server'):
            name = f'submission/latti/model/{side}/ckks_parameter.json'
            self.assertEqual((ROOT / name).read_bytes(), (ROOT / 'backend/base/code' / name).read_bytes())

    def test_gpu_is_required_and_reference_check_is_client_side(self):
        server = (ROOT / 'submission/server_encrypted_compute.py').read_text()
        self.assertIn('CPU fallback is forbidden', server)
        client = (ROOT / 'submission/client_decrypt_decode.py').read_text()
        self.assertIn('error > 1e-6', client)
        launcher = (ROOT / 'scripts/benchmark.py').read_text()
        self.assertIn('CPU comparison is diagnostic only, not a benchmark', launcher)
        self.assertIn('official three-run measurement', launcher)


if __name__ == '__main__':
    unittest.main()
