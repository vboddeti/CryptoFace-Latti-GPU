"""Automatic, user-local setup for the unchanged benchmark stage interface."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def configuration() -> dict:
    return json.loads((HERE / 'bootstrap.json').read_text())


def private_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if path.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f'Runtime cache must be a private directory owned by this user: {path}')
    return path.resolve()


def cache_directory() -> Path:
    selected = os.environ.get('CRYPTOFACE_CACHE_DIR')
    parent = Path(selected) if selected else Path(tempfile.gettempdir()) / f'cryptoface-{os.getuid()}'
    parent = private_directory(parent)
    identity = hashlib.sha256()
    for name in ('bootstrap.py', 'bootstrap.json', 'requirements.txt'):
        identity.update((HERE / name).read_bytes())
    identity.update(platform.node().encode())
    return private_directory(parent / identity.hexdigest()[:20])


def download(specification: dict, destination: Path) -> None:
    if destination.is_file() and digest(destination) == specification['sha256']:
        return
    temporary = destination.with_suffix(destination.suffix + '.download')
    try:
        request = urllib.request.Request(specification['url'], headers={'User-Agent': 'CryptoFace-bootstrap/1'})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        if digest(temporary) != specification['sha256']:
            raise ValueError(f'Download checksum mismatch: {destination.name}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_recipe(archive: Path, destination: Path) -> Path:
    destination.mkdir(mode=0o700)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            target = destination / member.name
            if not target.resolve().is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe source archive member: {member.name}')
        source.extractall(destination, members=members)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not (roots[0] / 'release.json').is_file():
        raise ValueError('Unexpected build recipe archive layout')
    return roots[0]


def discover_cuda(environment: dict[str, str] | None = None) -> Path | None:
    env = os.environ if environment is None else environment
    candidates = []
    for variable in ('CUDACXX', 'CUDA_HOME', 'CUDA_PATH'):
        if env.get(variable):
            path = Path(env[variable])
            candidates.append(path if variable == 'CUDACXX' else path / 'bin/nvcc')
    executable = shutil.which('nvcc', path=env.get('PATH', ''))
    if executable:
        candidates.append(Path(executable))
    for base in (Path('/usr/local'), Path('/opt')):
        candidates.extend(sorted(base.glob('cuda*/bin/nvcc'), reverse=True))
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            result = subprocess.run([str(candidate), '--version'], capture_output=True,
                                    text=True, check=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            continue
        # Preserve the tested toolkit family; do not silently switch CUDA majors.
        if re.search(r'release\s+12\.9\b', result.stdout):
            root = candidate.resolve().parent.parent
            includes = (root / 'include', root / 'include/cccl',
                        root / 'targets/x86_64-linux/include', root / 'targets/x86_64-linux/include/cccl')
            libraries = (root / 'lib64', root / 'lib', root / 'targets/x86_64-linux/lib')
            if ((root / 'bin/cuobjdump').is_file()
                    and all(any((path / header).is_file() for path in includes)
                            for header in ('cuda_runtime.h', 'cuda_profiler_api.h', 'curand.h', 'thrust/version.h'))
                    and all(any((path / library).is_file() for path in libraries)
                            for library in ('libcudart.so', 'libcurand.so'))):
                return root
    return None


def cache_environment(cache: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('CRYPTOFACE_', 'LATTISENSE_', 'HEONGPU_')) and key not in (
            'CRYPTOFACE_GPUS', 'CRYPTOFACE_VERIFY_GPU_MATCHING', 'CRYPTOFACE_CACHE_DIR'
        ):
            env.pop(key)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS',
                'CUDAFLAGS', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS',
                'MAMBARC', 'CONDARC'):
        env.pop(key, None)
    for key, leaf in {
        'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp', 'XDG_CACHE_HOME': 'cache',
        'PIP_CACHE_DIR': 'cache/pip', 'HF_HOME': 'cache/huggingface', 'TORCH_HOME': 'cache/torch',
        'CUDA_CACHE_PATH': 'cache/cuda', 'MPLCONFIGDIR': 'cache/matplotlib',
        'GOCACHE': 'cache/go', 'GOMODCACHE': 'cache/go-mod', 'MAMBA_ROOT_PREFIX': 'mamba',
    }.items():
        directory = cache / leaf
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(directory)
    env.update(PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1', GIT_TERMINAL_PROMPT='0')
    return env


def toolchain_environment(prefix: Path, cuda: Path, env: dict[str, str]) -> dict[str, str]:
    result = env.copy()
    result.setdefault('JOBS', str(min(16, os.cpu_count() or 1)))
    # The pinned backend uses std::tuple without including its standard header.
    # Supply the declaration without modifying frozen arithmetic sources.
    result['NVCC_PREPEND_FLAGS'] = '-include tuple'
    result['PATH'] = os.pathsep.join([str(prefix / 'bin'), str(cuda / 'bin'), env.get('PATH', '')])
    result.update(PYTHON=str(prefix / 'bin/python'), CUDA_HOME=str(cuda), CUDA_PATH=str(cuda),
                  CUDACXX=str(cuda / 'bin/nvcc'), CUDAToolkit_ROOT=str(cuda),
                  CC=str(prefix / 'bin/x86_64-conda-linux-gnu-cc'),
                  CXX=str(prefix / 'bin/x86_64-conda-linux-gnu-c++'),
                  CUDAHOSTCXX=str(prefix / 'bin/x86_64-conda-linux-gnu-c++'),
                  H5CXX=str(prefix / 'bin/h5c++'), CMAKE_PREFIX_PATH=str(prefix))
    for key, directories in {
        'CMAKE_INCLUDE_PATH': [prefix / 'include', cuda / 'include', cuda / 'include/cccl',
                               cuda / 'targets/x86_64-linux/include', cuda / 'targets/x86_64-linux/include/cccl'],
        'CPATH': [prefix / 'include', cuda / 'include', cuda / 'targets/x86_64-linux/include'],
        'LIBRARY_PATH': [prefix / 'lib', cuda / 'lib64', cuda / 'targets/x86_64-linux/lib'],
        'LD_LIBRARY_PATH': [prefix / 'lib', cuda / 'lib64', cuda / 'targets/x86_64-linux/lib'],
    }.items():
        result[key] = os.pathsep.join([str(path) for path in directories if path.is_dir()] +
                                     ([env[key]] if env.get(key) else []))
    return result


def check_payload(recipe: Path) -> None:
    """The frozen recipe is valid only for these unchanged numerical sources."""
    paths = [HERE / 'config.yml', *(HERE / 'native').rglob('*'), *(HERE / 'latti/model').rglob('*')]
    paths += list((HERE / 'utils').rglob('*.py'))
    for path in paths:
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        original = recipe / 'submission' / path.relative_to(HERE)
        if not original.is_file() or digest(path) != digest(original):
            raise ValueError(f'Numerical source differs from pinned build recipe: {path.relative_to(HERE)}')


def backend_link_environment(work: Path, environment: dict[str, str]) -> dict[str, str]:
    """Give the cross linker explicit paths to indirect native dependencies."""
    source = work / '.deps/latti-ai'
    directories = (source / 'build-h200/inference/lattisense/abi',
                   source / 'inference/lattisense/backends/HEonGPU/install/lib')
    flags = '-Wl,-rpath-link,' + os.pathsep.join(map(str, directories))
    return dict(environment, LDFLAGS=flags, HDF5_LDFLAGS=flags)


def build_runtime(cache: Path, specification: dict) -> dict:
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This submission requires Linux x86-64 and an NVIDIA H200 GPU.')
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('An NVIDIA GPU driver must be installed by the host administrator (nvidia-smi missing).')
    env = cache_environment(cache)
    attempt = Path(tempfile.mkdtemp(prefix='setup-', dir=cache))
    log_path = attempt / 'setup.log'
    print(f'[submission setup] First run: preparing dependencies and native runtime. Log: {log_path}', flush=True)
    with log_path.open('x') as log:
        def run(command: list[str], environment: dict[str, str] = env) -> None:
            subprocess.run(command, cwd=attempt, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

        try:
            download(specification['micromamba'], cache / 'micromamba.tar.bz2')
            mamba = cache / 'micromamba'
            with tarfile.open(cache / 'micromamba.tar.bz2') as archive:
                member = archive.getmember('bin/micromamba')
                if not member.isfile():
                    raise ValueError('Invalid micromamba archive')
                with archive.extractfile(member) as source, mamba.open('wb') as target:
                    shutil.copyfileobj(source, target)
            mamba.chmod(0o700)
            cuda = discover_cuda(env)
            packages = list(specification['toolchain_packages'])
            if cuda is None:
                print('[submission setup] No compatible CUDA toolkit found; installing CUDA 12.9 in the private cache.', flush=True)
                packages += specification['cuda_packages']
            else:
                print(f'[submission setup] Discovered CUDA toolkit: {cuda}', flush=True)
            prefix = attempt / 'environment'
            run([str(mamba), '--no-rc', 'create', '-y', '-p', str(prefix), '--override-channels',
                 '-c', 'conda-forge', '--strict-channel-priority', *packages])
            for alias, compiler in {'cc': 'cc', 'gcc': 'gcc', 'c++': 'c++', 'g++': 'g++'}.items():
                (prefix / 'bin' / alias).symlink_to('x86_64-conda-linux-gnu-' + compiler)
            if cuda is None:
                # CMake selects Conda's targets/.../bin/nvcc symlink. nvcc looks
                # for its profile and link stubs beside that invocation path.
                for name in ('nvcc.profile', 'crt'):
                    (prefix / 'targets/x86_64-linux/bin' / name).symlink_to('../../../bin/' + name)
            cuda = cuda or prefix
            environment = toolchain_environment(prefix, cuda, env)
            python = str(prefix / 'bin/python')
            print('[submission setup] Installing Python dependencies.', flush=True)
            run([python, '-m', 'pip', 'install', '--no-compile', '-r', str(HERE / 'requirements.txt')], environment)
            run([python, '-m', 'pip', 'check'], environment)
            download(specification['recipe'], cache / 'recipe.tar.gz')
            recipe = extract_recipe(cache / 'recipe.tar.gz', attempt / 'recipe')
            check_payload(recipe)
            reference = attempt / 'reference'
            expected_reference = json.loads((recipe / 'backend/manifest.json').read_text())['replay_environment']['model_sha256']
            existing_reference = Path.home() / '.insightface/models/buffalo_l'
            if all((existing_reference / name).is_file() and digest(existing_reference / name) == checksum
                   for name, checksum in expected_reference.items()):
                shutil.copytree(existing_reference, reference / 'models/buffalo_l')
            else:
                print('[submission setup] Downloading the pinned face preprocessing models.', flush=True)
                run([python, '-c', 'from insightface.utils.storage import ensure_available; '
                     'import sys; ensure_available("models", "buffalo_l", root=sys.argv[1])',
                     str(reference)], environment)
            for name, checksum in expected_reference.items():
                if digest(reference / 'models/buffalo_l' / name) != checksum:
                    raise ValueError(f'Face preprocessing model checksum mismatch: {name}')
            model = attempt / 'model_parameters.h5'
            download(specification['model'], model)
            sysroot = attempt / 'sysroot'
            sysroot.mkdir()
            (sysroot / 'usr').symlink_to(prefix, target_is_directory=True)
            environment['CRYPTOFACE_NTL_SYSROOT'] = str(sysroot)
            backend = attempt / 'reproducible/backend'
            gpu = attempt / 'reproducible/gpu'
            environment = backend_link_environment(backend, environment)
            print('[submission setup] Building pinned Latti-AI, LattiSense, and GPU matching sources.', flush=True)
            run([python, '-B', str(recipe / 'scripts/build_backend.py'), '--scratch-root', str(attempt),
                 '--work', str(backend), '--system-toolchain', '--ntl-root', str(sysroot),
                 '--model-input', str(model)], environment)
            run([python, '-B', str(recipe / 'scripts/build_gpu_matching.py'), '--scratch-root', str(attempt),
                 '--baseline-build', str(backend), '--work', str(gpu)], environment)
            # Save the exact resolved tool packages for reproducibility.
            with (attempt / 'toolchain-explicit.txt').open('x') as output:
                subprocess.run([str(mamba), '--no-rc', 'list', '-p', str(prefix), '--explicit'],
                               cwd=attempt, env=environment, stdout=output, stderr=log, check=True)
            result = {'schema_version': 1, 'prefix': str(prefix), 'cuda': str(cuda),
                      'artifact_root': str(gpu), 'build_dir': str(gpu / 'submission-build'),
                      'model_dir': str(gpu / 'source/submission/latti/model'),
                      'sysroot': str(sysroot), 'recipe': str(recipe), 'reference_root': str(reference),
                      'build_manifest_sha256': digest(gpu / 'submission-build/build_manifest.json')}
            temporary = cache / 'ready.json.tmp'
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(cache / 'ready.json')
            print('[submission setup] Runtime ready; subsequent stages and runs reuse this cache.', flush=True)
            return result
        except Exception as error:
            raise RuntimeError(f'Automatic submission setup failed; inspect {log_path}. '
                               'A later harness run will retry in a new build directory.') from error


def ready_runtime(cache: Path, specification: dict) -> dict:
    with (cache / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stamp = cache / 'ready.json'
        if not stamp.is_file():
            return build_runtime(cache, specification)
        record = json.loads(stamp.read_text())
        if record.get('schema_version') != 1:
            raise ValueError('Invalid automatic runtime cache metadata')
        for key in ('prefix', 'artifact_root', 'build_dir', 'model_dir', 'sysroot', 'recipe', 'reference_root'):
            if not Path(record[key]).resolve().is_relative_to(cache):
                raise ValueError(f'Runtime cache path escapes its private directory: {key}')
        if digest(Path(record['build_dir']) / 'build_manifest.json') != record['build_manifest_sha256']:
            raise ValueError('Runtime build manifest changed after setup')
        check_payload(Path(record['recipe']))
        return record


def ensure_environment() -> None:
    """Called before third-party imports in each benchmark submission stage."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    stage = str(Path(sys.argv[0]).resolve())
    specification = configuration()
    cache = cache_directory()
    record = ready_runtime(cache, specification)
    env = toolchain_environment(Path(record['prefix']), Path(record['cuda']), cache_environment(cache))
    env.update(specification['settings'])
    env.update(CRYPTOFACE_ARTIFACT_ROOT=record['artifact_root'], CRYPTOFACE_BUILD_DIR=record['build_dir'],
               CRYPTOFACE_MODEL_DIR=record['model_dir'], CRYPTOFACE_NTL_SYSROOT=record['sysroot'],
               CRYPTOFACE_REFERENCE_ROOT=record['reference_root'], CRYPTOFACE_CACHE_DIR=str(cache.parent))
    directory = cache / 'run'
    directory.mkdir(exist_ok=True, mode=0o700)
    os.chdir(directory)
    python = Path(record['prefix']) / 'bin/python'
    if Path(sys.prefix).resolve() != Path(record['prefix']).resolve():
        os.execve(str(python), [str(python), '-B', stage, *sys.argv[1:]], env)
    os.environ.clear()
    os.environ.update(env)
