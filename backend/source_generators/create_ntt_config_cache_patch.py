#!/usr/bin/env python3
"""Generate a source-only NTT metadata-cache patch against hash-pinned sources."""
import argparse
import difflib
import hashlib
from pathlib import Path
import re

PREFIX = 'backends/HEonGPU/thirdparty/GPU-NTT/src/'
SOURCES = {
    PREFIX + 'include/gpuntt/ntt_merge/ntt.cuh': '9967be743feae32bb2d473d724d57eb7d96fce84630af5c289d2c3da96c2d4cf',
    PREFIX + 'lib/ntt_merge/ntt.cu': '01e392652ba3a497546376bfa3f7ab237bc895ac97c8cd5103832bf83b77b057',
}


def cached_getters(header):
    for direction in ('Forward', 'Inverse'):
        marker = f'    template <typename T> auto Create{direction}NTTKernel()'
        if header.count(marker) != 1:
            raise ValueError('Unexpected NTT factory definition')
        start = header.index(marker)
        end = header.index('\n    }\n', start) + len('\n    }\n')
        factory = header[start:end]
        if factory.count('return std::unordered_map<int, std::vector<KernelConfig>>') != 1:
            raise ValueError('NTT factory is not the expected fixed metadata table')
        addition = (
            f'\n    // Immutable host launch metadata only; no GPU/context-dependent state.\n'
            f'    template <typename T> const auto& Cached{direction}NTTKernel()\n'
            f'    {{\n'
            f'        static const auto parameters = Create{direction}NTTKernel<T>();\n'
            f'        return parameters;\n'
            f'    }}\n')
        header = header[:end] + addition + header[end:]
    return header


def cached_consumers(source):
    if source.count('auto kernel_parameters =') != 6:
        raise ValueError('Unexpected number of NTT table consumers')
    if re.search(r'current_kernel_params\.\w+\s*=(?!=)', source):
        raise ValueError('Cannot share mutable kernel configuration')
    source = source.replace('auto kernel_parameters =', 'const auto& kernel_parameters =')
    # Some callers reassign this cursor to the final launch configuration.
    # It must be a local value, never a reference into the shared immutable map.
    source = source.replace('auto& current_kernel_params', 'KernelConfig current_kernel_params')
    for direction in ('Forward', 'Inverse'):
        source = source.replace(f'Create{direction}NTTKernel<', f'Cached{direction}NTTKernel<')
    source, count = re.subn(r'kernel_parameters\s*\[\s*cfg\.n_power\s*\]',
                            'kernel_parameters.at(cfg.n_power)', source)
    if count < 12:
        raise ValueError('Expected all read-only NTT table lookups')
    return source


def make_patch(root):
    chunks = []
    for relative, expected in SOURCES.items():
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Unexpected baseline source: {relative}')
        original = data.decode()
        changed = cached_getters(original) if relative.endswith('.cuh') else cached_consumers(original)
        chunks.append(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                                  fromfile='a/' + relative, tofile='b/' + relative)))
    return ''.join(chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lattisense', type=Path, required=True)
    args = parser.parse_args()
    print(make_patch(args.lattisense), end='')
