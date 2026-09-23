"""Fail-closed working-directory/cache isolation for experiment processes."""
import os
from pathlib import Path
import resource
import shutil
import tempfile

SCRATCH = Path('/mnt/gs21/scratch/vishnu/cryptoface')


def resolve_executable(value):
    """Resolve a PATH command or explicit path before changing cwd."""
    executable = shutil.which(str(value))
    if executable is None:
        raise FileNotFoundError(f'Executable not found: {value}')
    return Path(executable).resolve(strict=True)


def prepare_runtime(root):
    """Confine relative writes and inherited library caches to a scratch run."""
    root = Path(root).resolve()
    scratch = SCRATCH.resolve()
    if root == scratch or not root.is_relative_to(scratch):
        raise ValueError('Runtime directory must be a dedicated CryptoFace scratch subdirectory')
    # Slurm nodes may inherit different core limits than the interactive shell.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    paths = {
        'TMPDIR': root / 'tmp', 'TMP': root / 'tmp', 'TEMP': root / 'tmp',
        'XDG_CACHE_HOME': root / 'cache', 'CUDA_CACHE_PATH': root / 'cuda-cache',
        'TORCH_HOME': root / 'cache/torch', 'HF_HOME': root / 'cache/huggingface',
        'NUMBA_CACHE_DIR': root / 'cache/numba',
        'MPLCONFIGDIR': root / 'cache/matplotlib',
        'GOCACHE': root / 'cache/go-build', 'GOMODCACHE': root / 'cache/go-mod',
    }
    for path in set(paths.values()):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Runtime cache escapes run directory: {path}')
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({name: str(path) for name, path in paths.items()})
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    tempfile.tempdir = None
    os.chdir(root)
    return root
