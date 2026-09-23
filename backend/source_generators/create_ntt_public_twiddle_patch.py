#!/usr/bin/env python3
"""Emit an isolated public-twiddle experiment patch; never mutate input trees."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re

import create_moddown_p3_entry_patch as p3
import create_ntt_config_cache_patch as ntt
import create_ntt_n16_specialization_patch as n16
import ntt_public_twiddle_templates as template

PREFIX = 'backends/HEonGPU/'
CTX_H = PREFIX + 'src/include/heongpu/host/ckks/context.cuh'
CTX_C = PREFIX + 'src/lib/host/ckks/context.cu'
OPERATOR = PREFIX + 'src/lib/host/ckks/operator.cu'
NTT_H = PREFIX + 'thirdparty/GPU-NTT/src/include/gpuntt/ntt_merge/ntt.cuh'
NTT_C = PREFIX + 'thirdparty/GPU-NTT/src/lib/ntt_merge/ntt.cu'
EXPECTED_PARENT = {
    CTX_H: '68411fb8e09b86404ef225ba2099e662346923b3b8833035cdd7cd14fbbf5bce',
    CTX_C: 'e899c84d19b20aad02e9fe43ce40de793532d47db74ca4b0eb49134cd30c9143',
    OPERATOR: '8490f64bc250ca0dc210f8a5b20a8e526de95e28d058f276733bb9718cdadfad',
    NTT_H: '04e81edbb58d596bf6e11f619d7ac74be0d275747bf7f5eb0f0dea91b19970d2',
    NTT_C: '2bd1c1737f87ffa4e23683648ac6a122dd9c4f2e34668be33bad1408872f0c4a',
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Nonunique source anchor: ' + old[:100])
    return source.replace(old, new, 1)


def definition(source, name):
    # The pinned file uses four-space top-level function definitions.
    marker = name + '('
    candidates = [match.start() for match in re.finditer(re.escape(marker), source)
                  if source[max(0, match.start()-80):match.start()].rstrip().endswith('void')]
    if len(candidates) != 1:
        raise ValueError('Expected unique function definition: ' + name)
    pos = candidates[0]
    start = source.rfind('    template <', 0, pos)
    if start < 0 or pos - start > 300:
        raise ValueError('Unexpected template/function layout: ' + name)
    end = source.index('\n    }\n', pos) + len('\n    }\n')
    return source[start:end]


def transform_kernel(kernel):
    changed = once(kernel, 'bool NotLast>', 'bool NotLast, bool Verify>')
    changed = once(changed, 'ForwardCoreModulusOrderedN16(',
                   'ForwardCoreModulusOrderedN16PublicTwiddle(')
    changed, count = re.subn(r'(int\s*\*\s*order)\)',
                            r'\1, const Root<T>* __restrict__ twiddle_quotients)', changed, count=1)
    if count != 1:
        raise ValueError('Unexpected ordered N16 signature')
    changed = once(changed, '\n    {\n', '\n    {\n        static_assert(std::is_same<T, Data64>::value, "64-bit twiddle path only");\n')
    if changed.count('CooleyTukeyUnit(') != 3:
        raise ValueError('Expected exactly three butterfly call sites')
    changed = changed.replace('CooleyTukeyUnit(', 'CooleyTukeyUnitPublicTwiddle<Verify>(')
    changed, count = re.subn(
        r'(root_of_unity_table\[current_root_index\],\s*modulus\[prime_index\])\)',
        r'\1, twiddle_quotients[current_root_index])', changed)
    if count != 3:
        raise ValueError('Unexpected butterfly root/modulus pairing')
    return changed


def transform_dispatch(dispatch):
    _, body = dispatch.split('\n    {\n', 1)
    expression = re.compile(r'ForwardCoreModulusOrderedN16<T, (0, 7, true|7, 9, false)><<<[\s\S]*?\);')

    def replace_call(match):
        call = match.group()
        call = once(call, 'ForwardCoreModulusOrderedN16<T, ' + match.group(1) + '>',
                    'ForwardCoreModulusOrderedN16PublicTwiddle<T, ' + match.group(1) + ', Verify>')
        if not call.endswith('mod_count, order);'):
            raise ValueError('Unexpected specialized launch argument suffix')
        return call[:-2] + ', twiddle_quotients);'

    body, count = expression.subn(replace_call, body)
    if count != 2:
        raise ValueError('Expected the two guarded N16 launch sites')
    return template.DISPATCH_SIGNATURE + body


def transform_context(source):
    anchor = 'namespace heongpu\n{\n'
    source = once(source, anchor, anchor + template.HOST_HELPERS)
    allocation = re.compile(r'            ntt_table_ =\s*\n\s*std::make_shared<DeviceVector<Root64>>\(Qprime_ntt_table\);')
    matches = list(allocation.finditer(source))
    if len(matches) != 1:
        raise ValueError('Expected one original full-ring root allocation')
    match = matches[0]
    source = (source[:match.start()] + template.CONTEXT_RESET + match.group()
              + '\n' + template.CONTEXT_ADDITION + source[match.end():])
    return '#include <cstdlib>\n#include <cstring>\n#include <cstdio>\n#include <stdexcept>\n' + source


def transform_operator(source):
    pattern = re.compile(r'gpuntt::GPU_NTT_Modulus_Ordered_Inplace\([\s\S]*?\);')
    count = 0

    def replace_call(match):
        nonlocal count
        call = match.group()
        if 'context_->ntt_table_->data()' not in call:
            return call
        if 'context_->intt_table_' in call or 'context_->ntt_table_slot_' in call:
            raise ValueError('Ambiguous root table in candidate call')
        count += 1
        call = call.replace('GPU_NTT_Modulus_Ordered_Inplace(',
                            'GPU_NTT_Modulus_Ordered_PublicTwiddle_Inplace64(', 1)
        return call[:-2] + ',\n            ' + template.OPERATOR_ARGUMENTS + ');'

    changed = pattern.sub(replace_call, source)
    if count != 18:
        raise ValueError('Expected exactly 18 full-ring ordered NTT call sites')
    return changed, count


def transform_parent(sources):
    for name, expected in EXPECTED_PARENT.items():
        if sha(sources[name]) != expected:
            raise ValueError('Verified N16 parent differs: ' + name)
    changed = dict(sources)
    anchor = '        std::shared_ptr<DeviceVector<Root64>> ntt_table_;\n'
    changed[CTX_H] = once(sources[CTX_H], anchor, anchor + template.CONTEXT_FIELDS)
    changed[CTX_C] = transform_context(sources[CTX_C])
    changed[OPERATOR], count = transform_operator(sources[OPERATOR])
    changed[NTT_H] = once(sources[NTT_H], '} // namespace gpuntt',
                          template.PUBLIC_DECLARATION + '\n} // namespace gpuntt')
    kernel = definition(sources[NTT_C], 'ForwardCoreModulusOrderedN16')
    dispatch = definition(sources[NTT_C], 'GPU_NTT_Modulus_Ordered')
    # Keep CUDA intrinsics out of the public header, also consumed by host C++.
    changed[NTT_C] = once(sources[NTT_C], kernel,
                          kernel + '\n' + template.DEVICE_HELPER + transform_kernel(kernel))
    # Concrete Data64 uses must follow the parent's explicit configuration
    # specializations near EOF; leave those existing definitions untouched.
    changed[NTT_C] = once(changed[NTT_C], '} // namespace gpuntt',
                          transform_dispatch(dispatch) + '\n' + template.PUBLIC_WRAPPER
                          + '\n} // namespace gpuntt')
    changed[NTT_C] = '#include <cstdio>\n#include <mutex>\n' + changed[NTT_C]
    return changed, count


def reconstruct_parent(root):
    originals, parent = {}, {}
    for (name, expected), transform in zip(p3.SOURCES.items(),
                                           (p3.add_kernel, p3.add_declaration, p3.dispatch_launches)):
        source = (root / name).read_text()
        if sha(source) != expected:
            raise ValueError('Frozen host-key source changed: ' + name)
        originals[name], parent[name] = source, transform(source)
    for name, expected in ntt.SOURCES.items():
        source = (root / name).read_text()
        if sha(source) != expected:
            raise ValueError('Frozen NTT source changed: ' + name)
        value = ntt.cached_consumers(source) if name == n16.SOURCE else ntt.cached_getters(source)
        if name == n16.SOURCE:
            value = n16.transform(value)
        originals[name], parent[name] = source, value
    for name in (CTX_H, CTX_C):
        originals[name] = parent[name] = (root / name).read_text()
    return originals, parent


def prepare(root, cumulative=False):
    if cumulative:
        originals, parent = reconstruct_parent(root)
    else:
        parent = {name: (root / name).read_text() for name in EXPECTED_PARENT}
        originals = dict(parent)
    changed, count = transform_parent(parent)
    patch = ''.join(''.join(difflib.unified_diff(
        originals[name].splitlines(True), changed[name].splitlines(True),
        fromfile='a/' + name, tofile='b/' + name)) for name in sorted(changed))
    return patch, {'kind': 'public-twiddle-source-generation', 'cumulative': cumulative,
                   'selected_forward_ordered_calls': count, 'patch_sha256': sha(patch),
                   'changed_source_sha256': {name: sha(changed[name]) for name in changed
                                             if changed[name] != originals[name]},
                   'registry_promotion': False, 'inference_executed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--parent', type=Path)
    source.add_argument('--cumulative-from', type=Path)
    parser.add_argument('--metadata', action='store_true')
    args = parser.parse_args()
    patch, metadata = prepare(args.cumulative_from or args.parent, bool(args.cumulative_from))
    print(json.dumps(metadata, indent=2) if args.metadata else patch, end='\n' if args.metadata else '')
