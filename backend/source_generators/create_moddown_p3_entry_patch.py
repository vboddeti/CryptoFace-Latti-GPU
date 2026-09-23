#!/usr/bin/env python3
"""Generate a dedicated P3 entry-point patch from hash-pinned host-key sources."""

import argparse
import difflib
import hashlib
from pathlib import Path
import re

KERNEL = 'divide_round_lastq_permute_ckks_kernel'
SPECIAL = 'divide_round_lastq_permute_ckks_p3_kernel'
PREFIX = 'backends/HEonGPU/src/'
SOURCES = {
    PREFIX + 'lib/kernel/switchkey.cu': 'c7dc0197349e08e2fc228f15951c8bed021c3be1ed1b11d9708169fe6930404a',
    PREFIX + 'include/heongpu/kernel/switchkey.cuh': '3fbbdc9276c2320c98cbd6f9d595d05f843d68596847a903e4292a32a4083033',
    PREFIX + 'lib/host/ckks/operator.cu': 'db8de27ad06424c1efb82ff5c112d40ae37227c280e5fb63d37060af42235438',
}


def kernel_definition(source):
    marker = '    __global__ void ' + KERNEL + '('
    if source.count(marker) != 1:
        raise ValueError('Expected exactly one generic kernel')
    start = source.index(marker)
    end = source.index('\n    }', start) + len('\n    }')
    return source[start:end]


def specialize(definition):
    if definition.count(', int P_size)') != 1 or definition.count('Data64 last_ct[15];') != 1:
        raise ValueError('Unexpected P3 kernel signature or temporary array')
    result = definition.replace(KERNEL, SPECIAL).replace(', int P_size)', ')')
    result = result.replace('    {\n', '    {\n        constexpr int P_size = 3;\n', 1)
    result = result.replace('// Max P size is 15.', '// Dedicated three-prime entry point.')
    result = result.replace('Data64 last_ct[15];', 'Data64 last_ct[3];')
    result, count = re.subn(r'(?m)^( +)(for \(int [ij] = 0;.*)$', r'\1#pragma unroll\n\1\2', result)
    if count != 3:
        raise ValueError('Expected exactly three P loops')
    return result


def add_kernel(source):
    original = kernel_definition(source)
    return source.replace(original, specialize(original) + '\n\n' + original, 1)


def add_declaration(source):
    marker = '    __global__ void ' + KERNEL + '('
    if source.count(marker) != 1:
        raise ValueError('Expected exactly one generic declaration')
    start = source.index(marker)
    end = source.index(';', start) + 1
    declaration = source[start:end]
    if declaration.count(', int P_size)') != 1:
        raise ValueError('Unexpected generic declaration')
    addition = declaration.replace(KERNEL, SPECIAL).replace(', int P_size)', ')')
    return source[:start] + addition + '\n\n' + source[start:]


def dispatch_launches(source):
    pattern = re.compile(r'(?m)^(?P<indent> +)(?P<call>' + KERNEL + r'<<<[\s\S]*?\);)')

    def replace(match):
        indent, original = match.group('indent', 'call')
        if original.count('context_->P_size') != 1 or not original.endswith('context_->P_size);'):
            raise ValueError('Unexpected P_size launch argument')
        candidate = re.sub(r',\s*context_->P_size\);$', ');', original.replace(KERNEL, SPECIAL, 1))
        def nested(call):
            return indent + '    ' + call.replace('\n', '\n    ')
        return (indent + 'if (context_->P_size == 3)\n' + indent + '{\n' + nested(candidate)
                + '\n' + indent + '}\n' + indent + 'else\n' + indent + '{\n'
                + nested(original) + '\n' + indent + '}')

    result, count = pattern.subn(replace, source)
    if count != 11:
        raise ValueError('Expected exactly eleven mod-down launch sites')
    return result


def make_patch(root):
    chunks = []
    transforms = (add_kernel, add_declaration, dispatch_launches)
    for (name, digest), transform in zip(SOURCES.items(), transforms):
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Unexpected source hash: ' + name)
        original = data.decode()
        chunks.append('diff --git a/' + name + ' b/' + name + '\n')
        chunks.extend(difflib.unified_diff(original.splitlines(True), transform(original).splitlines(True),
                                         fromfile='a/' + name, tofile='b/' + name))
    return ''.join(chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lattisense', type=Path, required=True)
    args = parser.parse_args()
    print(make_patch(args.lattisense), end='')
