#!/usr/bin/env python3
"""Generate an isolated CPU reference patch against the frozen harness source."""

import argparse
import difflib
import hashlib
from pathlib import Path
import textwrap

SOURCE_SHA256 = '2a66f6fff05c844963a35045fdb42c2b1707d363b0137f14c31922f3ca794079'
RELATIVE = 'harness/cleartext_impl.py'
ORIGINAL = '''    app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition'],
        providers=['CPUExecutionProvider'],
    )
'''
PREFIX = '''    # This InsightFace version drops sess_options in get_model(). Scope the
    # actual session-constructor override to this single-process model load.
    import onnxruntime as ort
    from insightface.model_zoo import model_zoo
    original_session = model_zoo.PickableInferenceSession

    class ThreadLimitedSession(original_session):
        def __init__(self, *positional, **kwargs):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 8
            options.inter_op_num_threads = 1
            kwargs['sess_options'] = options
            super().__init__(*positional, **kwargs)

    model_zoo.PickableInferenceSession = ThreadLimitedSession
    try:
'''
SUFFIX = '''    finally:
        model_zoo.PickableInferenceSession = original_session
'''
VERIFY = '''
    if set(app.models) != {'detection', 'recognition'}:
        raise ValueError('Unexpected CPU reference models')
    for model in app.models.values():
        options = model.session.get_session_options()
        if (options.intra_op_num_threads, options.inter_op_num_threads) != (8, 1):
            raise ValueError('CPU reference thread limits were not applied')
        if model.session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('CPU reference provider changed')
'''


def transform(source):
    if source.count(ORIGINAL) != 1:
        raise ValueError('Unexpected ArcFace model construction')
    prepare = '    app.prepare(ctx_id=-1, det_size=ARCFACE_DET_SIZE)\n'
    if source.count(prepare) != 1:
        raise ValueError('Unexpected ArcFace preparation')
    changed = source.replace(ORIGINAL, PREFIX + textwrap.indent(ORIGINAL, '    ') + SUFFIX, 1)
    return changed.replace(prepare, prepare + VERIFY, 1)


def make_patch(source):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Reference source differs from frozen harness')
    original = data.decode()
    return ''.join(difflib.unified_diff(original.splitlines(True), transform(original).splitlines(True),
                                        fromfile='a/' + RELATIVE, tofile='b/' + RELATIVE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(make_patch(args.source), end='')
