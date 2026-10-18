"""Setup contract checks; no native inference or synthetic face inputs."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('submission_bootstrap', Path(__file__).resolve().parents[1] / 'bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_corrupt_download_is_never_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'payload'
            with patch.object(bootstrap.urllib.request, 'urlopen', return_value=io.BytesIO(b'corrupt')):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    bootstrap.download({'url': 'https://invalid.example/payload', 'sha256': '0' * 64}, target)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.download').exists())

    def test_valid_cached_download_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'payload'
            target.write_bytes(b'unit test artifact')
            with patch.object(bootstrap.urllib.request, 'urlopen') as network:
                bootstrap.download({'url': 'https://invalid.example', 'sha256': bootstrap.digest(target)}, target)
            network.assert_not_called()

    def test_archive_cannot_escape_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, kind in (('../escape', tarfile.REGTYPE), ('recipe/link', tarfile.SYMTYPE)):
                archive = root / 'bad.tar'
                with tarfile.open(archive, 'w') as output:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = '/tmp'
                    output.addfile(member)
                with self.assertRaisesRegex(ValueError, 'Unsafe source archive'):
                    bootstrap.extract_recipe(archive, root / ('extract-' + str(kind)))
            self.assertFalse((root.parent / 'escape').exists())

    def test_cuda_discovery_checks_version_not_directory_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bin').mkdir()
            for name in ('nvcc', 'cuobjdump'):
                (root / 'bin' / name).touch()
            for directory, files in {'include': ('cuda_runtime.h', 'cuda_profiler_api.h', 'curand.h', 'thrust/version.h'),
                                     'lib64': ('libcudart.so', 'libcurand.so')}.items():
                (root / directory).mkdir()
                for name in files:
                    (root / directory / name).parent.mkdir(parents=True, exist_ok=True)
                    (root / directory / name).touch()
            with patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Cuda compilation tools, release 12.9, V12.9.86')):
                self.assertEqual(bootstrap.discover_cuda({'CUDA_HOME': str(root), 'PATH': ''}), root)
            with patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Cuda compilation tools, release 13.0, V13.0.0')):
                self.assertIsNone(bootstrap.discover_cuda({'CUDA_HOME': str(root), 'PATH': ''}))
            (root / 'include/curand.h').unlink()
            with patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Cuda compilation tools, release 12.9, V12.9.86')):
                self.assertIsNone(bootstrap.discover_cuda({'CUDA_HOME': str(root), 'PATH': ''}))

    def test_cache_rejects_shared_or_symlink_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'shared'
            shared.mkdir(mode=0o755)
            with self.assertRaisesRegex(RuntimeError, 'private directory'):
                bootstrap.private_directory(shared)
            linked = root / 'linked'
            linked.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, 'private directory'):
                bootstrap.private_directory(linked)

    def test_reexecution_keeps_same_cache_and_original_stage_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'identity'
            cache.mkdir()
            record = {key: str(cache / key) for key in ('prefix', 'cuda', 'artifact_root', 'build_dir', 'model_dir', 'sysroot', 'reference_root')}
            original_cwd = Path.cwd()
            stage = str(bootstrap.HERE / 'client_key_generation.py')
            try:
                with patch.object(bootstrap, 'cache_directory', return_value=cache), \
                     patch.object(bootstrap, 'ready_runtime', return_value=record), \
                     patch.object(bootstrap.resource, 'setrlimit'), \
                     patch.object(bootstrap.os, 'umask'), \
                     patch.object(bootstrap.sys, 'argv', [stage, '0']), \
                     patch.object(bootstrap.os, 'execve', side_effect=SystemExit) as execute:
                    with self.assertRaises(SystemExit):
                        bootstrap.ensure_environment()
                _, arguments, environment = execute.call_args.args
                self.assertEqual(arguments[-2:], [stage, '0'])
                self.assertEqual(environment['CRYPTOFACE_CACHE_DIR'], str(root))
                self.assertEqual(environment['CRYPTOFACE_MODEL_DIR'], record['model_dir'])
            finally:
                os.chdir(original_cwd)

    def test_cache_metadata_tampering_does_not_trigger_a_silent_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            record = {key: str(cache / key) for key in ('prefix', 'artifact_root', 'build_dir', 'model_dir', 'sysroot', 'recipe', 'reference_root')}
            record.update(schema_version=1, build_manifest_sha256='0' * 64)
            build = Path(record['build_dir'])
            build.mkdir()
            (build / 'build_manifest.json').write_text('{}')
            (cache / 'ready.json').write_text(json.dumps(record))
            with patch.object(bootstrap, 'build_runtime') as builder:
                with self.assertRaisesRegex(ValueError, 'manifest changed'):
                    bootstrap.ready_runtime(cache, {})
            builder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
