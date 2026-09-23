#!/usr/bin/env python3
"""Verify a host-key source bundle; optionally rebuild it in fresh scratch.

Default is read-only. --build requires --work and --ntl-root. It does not run
inference or claim reproduced latency; real small/medium replay remains required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess

SCRATCH = Path('/mnt/gs21/scratch/vishnu/cryptoface')
MODEL = 'submission/latti/model/server/model_parameters.h5'
COMMITS = {'latti-ai': '88b4bac84b938f74e6936b19580181f97542e175',
           'lattisense': '809b35830550370b4570c9d45ae9e33c78a97156'}
HEADERS = {'cxx_sdk_v2/key_export_profile.h', 'cxx_sdk_v2/scoped_key_export_cache.h'}

# HEonGPU's *.txt ignore rule hides these required, hand-written build sources.
# Explicitly bounded: never collect arbitrary ignored files or build directories.
CMAKE_SOURCES = {'backends/HEonGPU/' + directory + 'CMakeLists.txt' for directory in (
    '', 'src/', 'thirdparty/', 'test/', 'benchmark/', 'example/',
    'example/basic/', 'example/mpc/', 'example/bootstrapping/',
    'thirdparty/GPU-NTT/', 'thirdparty/GPU-NTT/src/',
    'thirdparty/GPU-NTT/example/', 'thirdparty/GPU-NTT/benchmark/',
    'thirdparty/GPU-FFT/', 'thirdparty/GPU-FFT/src/',
    'thirdparty/GPU-FFT/example/', 'thirdparty/GPU-FFT/benchmark/',
    'thirdparty/RNGonGPU/', 'thirdparty/RNGonGPU/src/',
    'thirdparty/RNGonGPU/test/', 'thirdparty/RNGonGPU/example/',
    'thirdparty/RNGonGPU/benchmark/', 'thirdparty/RNGonGPU/thirdparty/',
    'thirdparty/RNGonGPU/thirdparty/GPU-NTT/',
    'thirdparty/RNGonGPU/thirdparty/GPU-NTT/src/',
    'thirdparty/RNGonGPU/thirdparty/GPU-NTT/example/',
    'thirdparty/RNGonGPU/thirdparty/GPU-NTT/benchmark/',
)}
CMAKE_SOURCES.add('fhe_ops_lib/lattigo/go_sdk/strip_cgo_line.cmake')


def verify_build_sources(bundle, name, metadata):
    sources = metadata.get('ignored_build_source_sha256', {})
    if set(sources) != (CMAKE_SOURCES if name == 'lattisense' else set()):
        raise ValueError('Missing or unexpected ignored build sources')
    for filename, digest in sources.items():
        path = checked_file(bundle / 'ignored' / name, filename, digest)
        if path.stat().st_size > 128 << 10 or '\x00' in path.read_text():
            raise ValueError('Ignored build source is not bounded text')


def restore_build_sources(bundle, name, metadata, checkout):
    verify_build_sources(bundle, name, metadata)
    for filename, digest in metadata.get('ignored_build_source_sha256', {}).items():
        target = checkout / filename
        if target.exists() or target.is_symlink():
            raise ValueError('Ignored build source collides with pinned checkout')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checked_file(bundle / 'ignored' / name, filename, digest), target)
        checked_file(checkout, filename, digest)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_file(root, name, expected):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
        raise ValueError(f'Unsafe bundle path: {name}')
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError(f'Bundle path escapes root: {name}')
    if sha256(path) != expected:
        raise ValueError(f'Bundle hash mismatch: {name}')
    return path


def verify(bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('family') != 'scoped-host-key-cache-v1' or 'small' not in manifest['evidence']:
        raise ValueError('Require captured host-key bundle with successful small evidence')
    checked_file(bundle, 'rebuild_host_key_sources.py', manifest['rebuild_recipe']['sha256'])
    for name, digest in manifest['measured_build']['source_hashes'].items():
        if name != MODEL:
            checked_file(bundle / 'code', name, digest)
    for name, commit in COMMITS.items():
        metadata = manifest['dependencies'][name]
        verify_build_sources(bundle, name, metadata)
        if metadata['commit'] != commit:
            raise ValueError('Unexpected dependency revision')
        if set(metadata['untracked_source_sha256']) != (HEADERS if name == 'lattisense' else set()):
            raise ValueError('Missing or unexpected dependency headers')
        checked_file(bundle, f'{name}.patch', metadata['patch_sha256'])
        for header, digest in metadata['untracked_source_sha256'].items():
            checked_file(bundle / 'untracked' / name, header, digest)
    return manifest


def new_work(path):
    result = path.resolve()
    if result == SCRATCH.resolve() or not result.is_relative_to(SCRATCH.resolve()):
        raise ValueError('Build work must be a fresh directory strictly inside scratch')
    if path.exists() or path.is_symlink():
        raise ValueError('Refusing to reuse an existing build directory')
    return result


def clone_repositories(destination, mirror, run):
    repository = 'https://github.com/human-analysis/latti-ai.git'
    if mirror is not None:
        mirror = mirror.resolve(strict=True)
        submodule = (mirror / 'inference/lattisense').resolve(strict=True)
        for name, checkout in [('latti-ai', mirror), ('lattisense', submodule)]:
            run(['git', '-C', str(checkout), 'cat-file', '-e', COMMITS[name] + '^{commit}'])
        repository = str(mirror)
    run(['git', 'clone', '--no-hardlinks', '--no-checkout', repository, str(destination)])
    run(['git', '-C', str(destination), 'checkout', '--detach', COMMITS['latti-ai']])
    if mirror is not None:
        # Only the fresh clone's local config changes; tracked sources and the
        # original repositories remain untouched. No compiled files are copied.
        run(['git', '-C', str(destination), 'config', 'submodule.inference/lattisense.url', str(submodule)])
        run(['git', '-c', 'protocol.file.allow=always', '-C', str(destination),
             'submodule', 'update', '--init', '--recursive'])
    else:
        run(['git', '-C', str(destination), 'submodule', 'update', '--init', '--recursive'])


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--ntl-root', type=Path)
    parser.add_argument('--model-input', type=Path)
    parser.add_argument('--lattisense-extra-patch', type=Path,
                        help='Apply a source-only optimization patch in the new scratch build only')
    parser.add_argument('--repository-source', type=Path,
                        help='Existing local Git source; restore pinned commits without GitHub authentication')
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    manifest = verify(bundle)
    if not args.build:
        print('Bundle hashes verified; no files changed and no inference run.')
        return
    if not args.work or not args.ntl_root:
        parser.error('--build requires --work and --ntl-root')
    work = new_work(args.work)
    ntl = args.ntl_root.resolve(strict=True)
    if not (ntl / 'usr/include/NTL/RR.h').is_file():
        raise ValueError('Missing installed NTL prerequisite')
    model = (args.model_input or Path(manifest['required_external_inputs'][MODEL]['path'])).resolve(strict=True)
    if sha256(model) != manifest['required_external_inputs'][MODEL]['sha256']:
        raise ValueError('Model input differs from measured model')
    work.mkdir(parents=True, exist_ok=False)
    os.chdir(work)
    for directory in ('tmp', 'cache', '.deps', 'logs', 'submission-build'):
        (work / directory).mkdir()
    source = work / 'source'
    source.mkdir()
    for name, digest in manifest['measured_build']['source_hashes'].items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model if name == MODEL else checked_file(bundle / 'code', name, digest), target)
        if sha256(target) != digest:
            raise ValueError('Scratch source copy differs from captured source')
    (work / '.deps/ntl-sysroot').symlink_to(ntl, target_is_directory=True)
    (source / '.deps').symlink_to(work / '.deps', target_is_directory=True)
    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    env.update(PYTHONDONTWRITEBYTECODE='1', CRYPTOFACE_ARTIFACT_ROOT=str(work),
               CRYPTOFACE_BUILD_DIR=str(work / 'submission-build'), TMPDIR=str(work / 'tmp'),
               TMP=str(work / 'tmp'), TEMP=str(work / 'tmp'), JOBS='8')
    for key, leaf in {'XDG_CACHE_HOME': '', 'CUDA_CACHE_PATH': 'cuda', 'GOCACHE': 'go-build',
                      'GOMODCACHE': 'go-mod', 'CPM_SOURCE_CACHE': 'cpm', 'TORCH_HOME': 'torch',
                      'HF_HOME': 'huggingface', 'UV_CACHE_DIR': 'uv',
                      'PYTHONPYCACHEPREFIX': 'pycache', 'MPLCONFIGDIR': 'matplotlib'}.items():
        env[key] = str(work / 'cache' / leaf)
    dependency = work / '.deps/latti-ai'
    with (work / 'logs/rebuild.log').open('x') as log:
        def run(command):
            subprocess.run(command, check=True, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        clone_repositories(dependency, args.repository_source, run)
        for name, checkout in {'latti-ai': dependency, 'lattisense': dependency / 'inference/lattisense'}.items():
            revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            if revision != COMMITS[name]:
                raise ValueError('Fresh checkout revision mismatch')
            run(['git', '-C', str(checkout), 'apply', '--check', str(bundle / f'{name}.patch')])
            run(['git', '-C', str(checkout), 'apply', str(bundle / f'{name}.patch')])
            for header, digest in manifest['dependencies'][name]['untracked_source_sha256'].items():
                target = checkout / header
                if target.exists():
                    raise ValueError('Untracked header collides with pinned checkout')
                shutil.copy2(checked_file(bundle / 'untracked' / name, header, digest), target)
        for name, checkout in {'latti-ai': dependency, 'lattisense': dependency / 'inference/lattisense'}.items():
            restore_build_sources(bundle, name, manifest['dependencies'][name], checkout)
        if args.lattisense_extra_patch:
            extra = args.lattisense_extra_patch.resolve(strict=True)
            data = extra.read_bytes()
            if extra.suffix != '.patch' or len(data) > 1 << 20 or b'GIT binary patch' in data:
                raise ValueError('Require a bounded source-only optimization patch')
            captured = work / 'extra-lattisense.patch'
            captured.write_bytes(data)
            if sha256(captured) != sha256(extra):
                raise ValueError('Optimization patch changed during capture')
            checkout = dependency / 'inference/lattisense'
            run(['git', '-C', str(checkout), 'apply', '--check', str(captured)])
            run(['git', '-C', str(checkout), 'apply', str(captured)])
            (work / 'optimization-provenance.json').write_text(json.dumps({
                'base_bundle_manifest_sha256': sha256(bundle / 'manifest.json'),
                'extra_patch_sha256': sha256(captured),
                'recipe_sha256': sha256(Path(__file__).resolve()),
                'not_baseline_reproduction': True,
            }, indent=2) + '\n')
        env.update(LATTI_AI_SOURCE=str(dependency), LATTI_AI_BUILD=str(dependency / 'build-h200'),
                   BUILD_DIR=str(dependency / 'build-h200'), LATTISENSE_CUDA_ARCH='90',
                   LATTI_STAGE_RUNTIME_OUTPUT=str(work / 'submission-build/latti_stage_runtime'))
        run(['bash', '-lc', 'set -euo pipefail\nulimit -c 0\nsource "$1/scripts/load_h200_modules.sh"\n'
             'bash "$LATTI_AI_SOURCE/scripts/build_h200.sh"\n'
             'bash "$1/submission/native/build_runtime.sh"\n'
             'python3 -B "$1/scripts/write_build_manifest.py" --output "$CRYPTOFACE_BUILD_DIR/build_manifest.json" '
             '--latti-ai-source "$LATTI_AI_SOURCE" --native-binary "$LATTI_STAGE_RUNTIME_OUTPUT" '
             '--runtime "$LATTI_AI_BUILD/inference/lattisense/liblattisense.so" --cuda-architecture sm_90',
             '--', str(source)])
    print(f'Rebuild completed in {work}; real-image replay still required. No latency claim.')


if __name__ == '__main__':
    main()
