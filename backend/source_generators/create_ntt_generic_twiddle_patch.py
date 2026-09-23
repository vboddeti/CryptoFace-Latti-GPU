#!/usr/bin/env python3
"""Generate an isolated generic-forward extension; no writes or inference.

The exact public-root multiplication and context ownership are inherited from
the pinned ordered-forward candidate. This generator changes only call routing
and adds an unsigned-64 RNS kernel/dispatch variant; it is not validation.
"""

import argparse
import difflib
from pathlib import Path
import re
import create_ntt_public_twiddle_patch as parent
import ntt_public_twiddle_templates as template

EXPECTED_PARENT = {
    parent.CTX_H: '885c9a7167211124d917f6217301b58a83c2f816821fcb24328be0c8d10a2667',
    parent.CTX_C: '763139907fcf7e04f3bf8b919c9928fcf9231edf6ed2a21e39c87864df364450',
    parent.OPERATOR: '1045f3d422d01e4330250d351570b6a39fe286221a36f4c9643cbcfc512b125c',
    parent.NTT_H: '2580d5767dc2aa95c3036d4efd6f99bf044c6cff81d22e3e35a480eca177c813',
    parent.NTT_C: '8546e89090dace1af2e00e4cb5ca61f36e1af9b850cfd022d6d3924756b62a3a',
}
EXPECTED_CALLS = 29


def rns_definition(source, name):
    definitions = []
    for match in re.finditer(re.escape(name) + r'\(', source):
        pos = match.start()
        if not source[max(0, pos - 80):pos].rstrip().endswith('void'):
            continue
        body = source.index('\n    {\n', pos)
        if 'int mod_count' not in source[pos:body]:
            continue
        start = source.rfind('    template <', 0, pos)
        if start < 0 or pos - start > 300:
            raise ValueError('Unexpected RNS template layout: ' + name)
        end = source.index('\n    }\n', body) + len('\n    }\n')
        definitions.append(source[start:end])
    if len(definitions) != 1:
        raise ValueError('Require one pinned RNS overload: ' + name)
    return definitions[0]


def transform_kernel(kernel):
    changed = parent.once(kernel, 'template <typename T>', 'template <typename T, bool Verify>')
    changed = parent.once(changed, 'ForwardCore(', 'ForwardCorePublicTwiddle(')
    changed = parent.once(changed, 'int mod_count)',
                          'int mod_count, const Root<T>* __restrict__ twiddle_quotients)')
    changed = parent.once(changed, '\n    {\n', '\n    {\n'
                          '        static_assert(std::is_same<T, Data64>::value, "64-bit RNS twiddle path only");\n')
    if changed.count('CooleyTukeyUnit(') != 3:
        raise ValueError('Require three generic butterfly call sites')
    changed = changed.replace('CooleyTukeyUnit(', 'CooleyTukeyUnitPublicTwiddle<Verify>(')
    changed, count = re.subn(
        r'(root_of_unity_table\[current_root_index\],\s*modulus_reg)\)',
        r'\1, twiddle_quotients[current_root_index])', changed)
    if count != 3:
        raise ValueError('Unexpected generic root/modulus pairing')
    return changed


def transform_dispatch(dispatch):
    _, body = dispatch.split('\n    {\n', 1)
    pattern = re.compile(r'ForwardCore<<<[\s\S]*?\);')

    def replace_call(match):
        call = parent.once(match.group(), 'ForwardCore<<<',
                           'ForwardCorePublicTwiddle<Data64, Verify><<<')
        if not call.endswith('mod_count);'):
            raise ValueError('Unexpected generic launch argument suffix')
        return call[:-2] + ', twiddle_quotients);'

    body, count = pattern.subn(replace_call, body)
    if count != 4:
        raise ValueError('Require four original generic launch expressions')
    signature = template.DISPATCH_SIGNATURE.replace('DispatchPublicTwiddleN16',
                                                    'DispatchPublicTwiddleGenericN16')
    signature = parent.once(signature, ', int* order', '')
    return signature + body


def public_entry_points():
    declaration = template.PUBLIC_DECLARATION.replace(
        'GPU_NTT_Modulus_Ordered_PublicTwiddle_Inplace64', 'GPU_NTT_PublicTwiddle_Inplace64')
    declaration = parent.once(declaration, ', int* order', '')
    wrapper = template.PUBLIC_WRAPPER.replace(
        'GPU_NTT_Modulus_Ordered_PublicTwiddle_Inplace64', 'GPU_NTT_PublicTwiddle_Inplace64')
    wrapper = parent.once(wrapper, ', int* order', '')
    wrapper = wrapper.replace(', order', '')
    wrapper = parent.once(wrapper, 'cfg.n_power != 16 || cfg.ntt_type != FORWARD ||',
                          'cfg.n_power != 16 || cfg.ntt_type != FORWARD ||\n'
                          '            cfg.ntt_layout != PerPolynomial ||')
    wrapper = wrapper.replace('GPU_NTT_Modulus_Ordered_Inplace', 'GPU_NTT_Inplace')
    wrapper = wrapper.replace('DispatchPublicTwiddleN16', 'DispatchPublicTwiddleGenericN16')
    wrapper = wrapper.replace('[GPU Public Twiddle]', '[GPU Public Twiddle Generic]')
    return declaration, wrapper


def transform_operator(source):
    pattern = re.compile(r'gpuntt::GPU_NTT_Inplace\([\s\S]*?\);')
    selected = []

    def replace_call(match):
        call = match.group()
        # Matching the following comma explicitly excludes offset root/modulus
        # pointers. Sparse and other-basis tables retain the original entry point.
        if not (re.search(r'context_->ntt_table_->data\(\)\s*,', call)
                and re.search(r'context_->modulus_->data\(\)\s*,', call)):
            return call
        selected.append(call)
        call = parent.once(call, 'GPU_NTT_Inplace(', 'GPU_NTT_PublicTwiddle_Inplace64(')
        return call[:-2] + ',\n            ' + template.OPERATOR_ARGUMENTS + ');'

    changed = pattern.sub(replace_call, source)
    if len(selected) != EXPECTED_CALLS:
        raise ValueError('Require exactly 29 full-table generic call sites')
    return changed, selected


def transform_parent(sources):
    for name, expected in EXPECTED_PARENT.items():
        if parent.sha(sources[name]) != expected:
            raise ValueError('Pinned public-twiddle parent changed: ' + name)
    changed = dict(sources)
    kernel = transform_kernel(rns_definition(sources[parent.NTT_C], 'ForwardCore'))
    dispatch = transform_dispatch(rns_definition(sources[parent.NTT_C], 'GPU_NTT'))
    # The existing exact device helper must be declared before the new kernel.
    anchor = parent.definition(sources[parent.NTT_C], 'ForwardCoreModulusOrderedN16PublicTwiddle')
    changed[parent.NTT_C] = parent.once(sources[parent.NTT_C], anchor, anchor + '\n' + kernel)
    declaration, wrapper = public_entry_points()
    changed[parent.NTT_C] = parent.once(changed[parent.NTT_C], '} // namespace gpuntt',
                                        dispatch + '\n' + wrapper + '\n} // namespace gpuntt')
    changed[parent.NTT_H] = parent.once(sources[parent.NTT_H], '} // namespace gpuntt',
                                        declaration + '\n} // namespace gpuntt')
    changed[parent.OPERATOR], selected = transform_operator(sources[parent.OPERATOR])
    return changed, selected


def prepare(root, cumulative=False):
    if cumulative:
        originals, n16_sources = parent.reconstruct_parent(root)
        sources, _ = parent.transform_parent(n16_sources)
    else:
        sources = {name: (root / name).read_text() for name in EXPECTED_PARENT}
        originals = dict(sources)
    changed, selected = transform_parent(sources)
    patch = ''.join(''.join(difflib.unified_diff(
        originals[name].splitlines(True), changed[name].splitlines(True),
        fromfile='a/' + name, tofile='b/' + name)) for name in sorted(changed))
    return patch, {
        'kind': 'generic-public-twiddle-source-prototype', 'cumulative': cumulative,
        'selected_generic_calls': len(selected), 'selected_call_sha256': [parent.sha(s) for s in selected],
        'patch_sha256': parent.sha(patch),
        'changed_source_sha256': {n: parent.sha(s) for n, s in changed.items() if s != originals[n]},
        'parent_source_sha256': EXPECTED_PARENT,
        'context_ownership_unchanged': all(changed[n] == sources[n] for n in (parent.CTX_H, parent.CTX_C)),
        'inference_executed': False, 'registry_promotion': False,
        'scope': 'Source generation only; source/compiled-code review, real-image verification and timing gates remain required.',
    }


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--parent', type=Path)
    source.add_argument('--cumulative-from', type=Path)
    parser.add_argument('--metadata', action='store_true')
    args = parser.parse_args()
    patch, metadata = prepare(args.cumulative_from or args.parent, bool(args.cumulative_from))
    print(json.dumps(metadata, indent=2) if args.metadata else patch, end='\n' if args.metadata else '')
