#!/usr/bin/env python3
"""Provision pinned inputs and build an isolated, harness-ready GPU submission."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import urllib.request

from benchmark import ROOT, digest, execution_environment, load, scratch_path, verify_inputs, verify_release


def checked_model(destination, specification, local=None):
    temporary = destination.with_suffix('.download')
    try:
        if local:
            shutil.copyfile(local, temporary)
        else:
            with urllib.request.urlopen(specification['url'], timeout=120) as response:
                with temporary.open('xb') as output:
                    shutil.copyfileobj(response, output)
        if digest(temporary) != specification['sha256']:
            raise ValueError('Packed model checksum mismatch; refusing this input')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def provision_reference(work, expected):
    # The unchanged upstream harness requires this standard InsightFace location.
    standard = Path.home() / '.insightface/models/buffalo_l'
    if not standard.exists() and not standard.is_symlink():
        from insightface.utils.storage import ensure_available
        downloaded = Path(ensure_available('models', 'buffalo_l', root=str(work / 'cache/insightface')))
        for name, sha in expected.items():
            if digest(downloaded / name) != sha:
                raise ValueError(f'Unexpected InsightFace reference model: {name}')
        standard.parent.mkdir(parents=True, exist_ok=True)
        standard.symlink_to(downloaded, target_is_directory=True)
    for name, sha in expected.items():
        if digest(standard / name) != sha:
            raise ValueError(f'Existing InsightFace reference model differs: {standard / name}')


def activation(work, prepared, toolchain, python, ntl):
    environment = execution_environment(prepared, load(ROOT / 'backend-settings.json'))
    selected = set(load(ROOT / 'backend-settings.json')) | {
        'CRYPTOFACE_ARTIFACT_ROOT', 'CRYPTOFACE_BUILD_DIR', 'PYTHONDONTWRITEBYTECODE',
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TMPDIR', 'TMP', 'TEMP',
        'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TORCH_HOME', 'HF_HOME',
        'MPLCONFIGDIR', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'GOCACHE', 'GOMODCACHE',
    }
    lines = ['# Source this file before invoking the unchanged benchmark harness.',
             'ulimit -c 0', 'umask 077', 'unset PYTHONPATH PYTHONSTARTUP',
             'while IFS= read -r name; do',
             '  case "$name" in CRYPTOFACE_*|LATTISENSE_*|HEONGPU_*) unset "$name" ;; esac',
             'done < <(compgen -v)',
             f'export CRYPTOFACE_NTL_SYSROOT={shlex.quote(str(ntl))}',
             f'source {shlex.quote(str(toolchain))} || return',
             f'export PYTHON={shlex.quote(str(python))}',
             f'export PATH={shlex.quote(str(python.parent))}:"$PATH"']
    lines += [f'export {key}={shlex.quote(environment[key])}' for key in sorted(selected)]
    lines += [f'cd -- {shlex.quote(str(prepared / "source"))}']
    (work / 'activate.sh').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-root', type=Path, required=True,
                        help='Existing non-home scratch directory; all build/runtime files go here')
    parser.add_argument('--work', type=Path, help='Fresh path under SCRATCH_ROOT/reproducible')
    parser.add_argument('--ntl-root', type=Path, default=Path('/'),
                        help='Sysroot with usr/include/NTL/RR.h; default is the installed system')
    parser.add_argument('--repository-source', type=Path, help='Optional offline Latti-AI Git checkout')
    parser.add_argument('--model-input', type=Path, help='Optional local packed weights instead of download')
    parser.add_argument('--dataset', type=Path, help='Optional existing pinned real face_dataset.h5')
    parser.add_argument('--cluster-toolchain', action='store_true',
                        help='Use the original H200 cluster modules instead of tools on PATH')
    args = parser.parse_args()
    for name in ('repository_source', 'model_input', 'dataset'):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve(strict=True))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    release = verify_release()
    for name, version in load(ROOT / 'backend/manifest.json')['replay_environment']['packages'].items():
        if importlib.metadata.version(name) != version:
            parser.error(f'Python dependency {name} must be {version}; install requirements.txt')
    scratch = args.scratch_root.resolve(strict=True)
    work = scratch_path(args.work or scratch / 'reproducible/standalone', scratch)
    if not work.is_relative_to(scratch / 'reproducible') or work.exists():
        parser.error('Use a fresh directory below SCRATCH_ROOT/reproducible')
    ntl = args.ntl_root.resolve(strict=True)
    if not (ntl / 'usr/include/NTL/RR.h').is_file():
        parser.error('Install NTL or specify --ntl-root containing usr/include/NTL/RR.h')
    for tool in ('git', 'cmake', 'go', 'nvcc', 'h5c++', 'patch'):
        if not args.cluster_toolchain and shutil.which(tool) is None:
            parser.error(f'Missing {tool}; install the prerequisites in README.md')
    python = Path(sys.executable).absolute()
    work.mkdir(parents=True, mode=0o700)
    source = work / 'source'
    source.mkdir(mode=0o700)
    for name in ('release.json', *release['files']):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    verify_release(source)
    os.chdir(work)
    environment = execution_environment(work, release['settings'])
    # Provisioning is online; benchmark execution uses the offline environment.
    environment.pop('HF_HUB_OFFLINE', None)
    environment.pop('TRANSFORMERS_OFFLINE', None)
    environment['CRYPTOFACE_NTL_SYSROOT'] = str(ntl)
    os.environ.clear()
    os.environ.update(environment)
    inputs = work / 'inputs'
    inputs.mkdir()
    model = inputs / 'model_parameters.h5'
    checked_model(model, load(source / 'external-inputs.json')['model'], args.model_input)
    backend = work / 'backend'
    gpu = work / 'gpu'
    command = [str(python), '-B', str(source / 'scripts/build_backend.py'),
               '--scratch-root', str(scratch), '--work', str(backend),
               '--ntl-root', str(ntl), '--model-input', str(model)]
    if args.repository_source:
        command += ['--repository-source', str(args.repository_source.resolve(strict=True))]
    if not args.cluster_toolchain:
        command += ['--system-toolchain']
    subprocess.run(command, cwd=work, env=environment, check=True)
    toolchain = (backend / 'source/scripts/load_h200_modules.sh' if args.cluster_toolchain
                 else source / 'scripts/load_system_toolchain.sh')
    subprocess.run(['bash', '-lc' if args.cluster_toolchain else '-c',
                    'set -euo pipefail; ulimit -c 0; source "$1"; shift; exec "$@"',
                    '--', str(toolchain), str(python), '-B', str(source / 'scripts/build_gpu_matching.py'),
                    '--scratch-root', str(scratch), '--baseline-build', str(backend), '--work', str(gpu)],
                   cwd=work, env=environment, check=True)
    provision_reference(work, load(source / 'backend/manifest.json')['replay_environment']['model_sha256'])
    dataset = args.dataset.resolve(strict=True) if args.dataset else inputs / 'datasets/face_dataset.h5'
    if not args.dataset:
        subprocess.run([str(python), '-B', str(source / 'harness/generate_dataset.py'),
                        str(dataset.with_name('face_dataset.npy'))], cwd=work, env=environment, check=True)
    verify_inputs(dataset, load(source / 'backend/manifest.json'))
    prepared = work / 'run'
    subprocess.run([str(python), '-B', str(source / 'scripts/benchmark.py'), 'prepare',
                    '--scratch-root', str(scratch), '--work', str(prepared),
                    '--backend-build', str(gpu), '--dataset', str(dataset)],
                   cwd=work, env=environment, check=True)
    activation(work, prepared, toolchain, python, ntl)
    print(f'Ready. Source {shlex.quote(str(work / "activate.sh"))}, then run the harness.')


if __name__ == '__main__':
    main()
