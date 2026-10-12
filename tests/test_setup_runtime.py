"""Setup guards without native inference or synthetic face inputs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import setup_runtime


class SetupTests(unittest.TestCase):
    def test_activation_removes_experimental_overrides(self):
        with tempfile.TemporaryDirectory(prefix='setup test ') as directory:
            work = Path(directory)
            prepared = work / 'run'
            (prepared / 'source').mkdir(parents=True)
            toolchain = work / 'toolchain.sh'
            toolchain.write_text('true\n')
            setup_runtime.activation(work, prepared, toolchain, Path(sys.executable), Path('/'))
            env = dict(os.environ, CRYPTOFACE_PAIR_LIMIT='1', LATTISENSE_UNREVIEWED='1', HEONGPU_UNREVIEWED='1')
            command = ['bash', '-c', 'source "$1"; "$PYTHON" -B -c '\
                       + "'import json, os; print(json.dumps(dict(os.environ)))'", '--', str(work / 'activate.sh')]
            result = json.loads(subprocess.check_output(command, env=env, text=True))
            for name in ('CRYPTOFACE_PAIR_LIMIT', 'LATTISENSE_UNREVIEWED', 'HEONGPU_UNREVIEWED'):
                self.assertNotIn(name, result)
            self.assertEqual(result['CRYPTOFACE_GPUS'], '0,1,2,3')
            self.assertEqual(result['PYTHON'], sys.executable)

    def test_corrupt_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'model.h5'
            source = Path(directory) / 'download'
            source.write_bytes(b'unit-test file, not inference data')
            with patch('urllib.request.urlopen', return_value=source.open('rb')):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    setup_runtime.checked_model(target, {'url': 'https://invalid.example/model', 'sha256': '0' * 64})
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.download').exists())

    def test_checked_local_input_is_copied_without_changing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input'
            content = b'unit-test file, not inference data'
            source.write_bytes(content)
            target = Path(directory) / 'model.h5'
            setup_runtime.checked_model(target, {'sha256': hashlib.sha256(content).hexdigest()}, source)
            self.assertEqual(source.read_bytes(), content)
            self.assertEqual(target.read_bytes(), content)

    def test_existing_reference_is_verified_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            model = home / '.insightface/models/buffalo_l'
            model.mkdir(parents=True)
            (model / 'reference').write_bytes(b'not an ONNX model')
            with patch.object(Path, 'home', return_value=home):
                with self.assertRaisesRegex(ValueError, 'reference model differs'):
                    setup_runtime.provision_reference(home / 'scratch', {'reference': '0' * 64})
            self.assertFalse(model.is_symlink())
            self.assertEqual((model / 'reference').read_bytes(), b'not an ONNX model')


if __name__ == '__main__':
    unittest.main()
