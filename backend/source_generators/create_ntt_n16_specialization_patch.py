#!/usr/bin/env python3
"""Hash-pinned P3+NTT compound plus guarded N16 ordered-forward specialization."""

import argparse
import difflib
import hashlib
from pathlib import Path
import re

import create_moddown_p3_entry_patch as p3
import create_ntt_config_cache_patch as ntt

KERNEL = 'ForwardCoreModulusOrdered'
SPECIAL = KERNEL + 'N16'
SOURCE = ntt.PREFIX + 'lib/ntt_merge/ntt.cu'
COMPOUND_SOURCE_SHA256 = 'fee8368f0585840f587a87763ce2e9ad21c8d71e50438e1f221a007c5928ab95'
FIXED = {'shared_index': '8', 'logm': 'LogM',
         'outer_iteration_count': 'OuterIterations', 'N_power': '16',
         'not_last_kernel': 'NotLast'}


def kernel_definition(source):
    marker = '    template <typename T>\n    __global__ void ' + KERNEL + '('
    if source.count(marker) != 1:
        raise ValueError('Require unique generic ordered-forward kernel')
    start = source.index(marker)
    end = source.index('\n    }\n', start) + len('\n    }\n')
    return source[start:end]


def specialize(definition):
    signature, body = definition.split('\n    {\n', 1)
    signature = signature.replace('template <typename T>',
                                  'template <typename T, int LogM, int OuterIterations, bool NotLast>')
    signature = signature.replace(KERNEL + '(', SPECIAL + '(')
    constants = []
    for name, value in FIXED.items():
        kind = 'bool' if name == 'not_last_kernel' else 'int'
        token = kind + ' ' + name
        if signature.count(token) != 1:
            raise ValueError('Unexpected fixed kernel parameter: ' + name)
        signature = signature.replace(token, kind + ' /* fixed ' + name + ' */')
        constants.append(f'        constexpr {kind} {name} = {value};\n')
    checks = ('        static_assert((LogM == 0 && OuterIterations == 7 && NotLast) ||\n'
              '                      (LogM == 7 && OuterIterations == 9 && !NotLast),\n'
              '                      "Unsupported ordered-forward N16 phase");\n')
    return signature + '\n    {\n' + checks + ''.join(constants) + body


def dispatch(source):
    pattern = re.compile(r'(?m)^(?P<indent> +)(?P<call>' + KERNEL + r'<<<[\s\S]*?\);)')
    matches = list(pattern.finditer(source))
    if len(matches) != 2:
        raise ValueError('Require two original ordered-forward launch sites')
    match = matches[0]  # Only the standard (n_power < 25) host branch.
    indent, call = match.group('indent'), match.group('call')
    if 'current_kernel_params.not_last_kernel' not in call:
        raise ValueError('Unexpected launch argument layout')

    def nested(text):
        return '\n'.join(indent + '    ' + line for line in text.splitlines())

    common = ('cfg.n_power == 16 && current_kernel_params.shared_index == 8 && '
              'current_kernel_params.shared_memory == 512 * sizeof(T)')
    phase7 = ('current_kernel_params.logm == 0 && current_kernel_params.outer_iteration_count == 7 && '
              'current_kernel_params.not_last_kernel && current_kernel_params.griddim_x == 128 && '
              'current_kernel_params.griddim_y == 1 && current_kernel_params.blockdim_x == 4 && '
              'current_kernel_params.blockdim_y == 64')
    phase9 = ('current_kernel_params.logm == 7 && current_kernel_params.outer_iteration_count == 9 && '
              '!current_kernel_params.not_last_kernel && current_kernel_params.griddim_x == 1 && '
              'current_kernel_params.griddim_y == 128 && current_kernel_params.blockdim_x == 256 && '
              'current_kernel_params.blockdim_y == 1')
    replacement = (
        indent + 'if (' + common + ' && ' + phase7 + ')\n' + indent + '{\n' +
        nested(call.replace(KERNEL, SPECIAL + '<T, 0, 7, true>', 1)) + '\n' + indent + '}\n' +
        indent + 'else if (' + common + ' && ' + phase9 + ')\n' + indent + '{\n' +
        nested(call.replace(KERNEL, SPECIAL + '<T, 7, 9, false>', 1)) + '\n' + indent + '}\n' +
        indent + 'else\n' + indent + '{\n' + nested(call) + '\n' + indent + '}')
    return source[:match.start()] + replacement + source[match.end():]


def transform(source):
    if hashlib.sha256(source.encode()).hexdigest() != COMPOUND_SOURCE_SHA256:
        raise ValueError('Unexpected compound NTT source')
    original = kernel_definition(source)
    changed = source.replace(original, original + '\n' + specialize(original), 1)
    return dispatch(changed)


def make_patch(root):
    # Emit one diff per file against the frozen host-key parent. NTT cache and
    # specialization share a file, so do not concatenate overlapping diffs.
    chunks = [p3.make_patch(root)]
    for relative, expected in ntt.SOURCES.items():
        original = (root / relative).read_text()
        if hashlib.sha256(original.encode()).hexdigest() != expected:
            raise ValueError('Unexpected base source: ' + relative)
        changed = ntt.cached_consumers(original) if relative == SOURCE else ntt.cached_getters(original)
        if relative == SOURCE:
            changed = transform(changed)
        chunks.append(''.join(difflib.unified_diff(
            original.splitlines(keepends=True), changed.splitlines(keepends=True),
            fromfile='a/' + relative, tofile='b/' + relative)))
    return ''.join(chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lattisense', type=Path, required=True)
    args = parser.parse_args()
    print(make_patch(args.lattisense), end='')
