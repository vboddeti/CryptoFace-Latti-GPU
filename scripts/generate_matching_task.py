#!/usr/bin/env python3
"""Compile the existing encrypted dot product, without generating any inputs."""
import argparse
import json
import os
from pathlib import Path
import random
import sys


def add_runtime_metadata(task, production_task):
    # The frozen GPU executor reads these scale fields for every CKKS op,
    # even when the circuit has no bootstrap. Preserve production values;
    # do not add btp_output_level, which would enable bootstrap setup.
    for name in ('btp_eval_mod_q', 'btp_eval_mod_message_ratio'):
        task['parameter'][name] = production_task['parameter'][name]


def generate(frontend_root, parameter_file, output):
    sys.path.insert(0, str(frontend_root))
    from frontend.custom_task import (CkksParam, CkksCiphertextNode, Argument,
                                      set_fhe_param, mult_relin, rescale,
                                      rotate_cols, add, process_custom_task)
    cfg = json.loads(parameter_file.read_text())['param0']
    # Reuse the entire Q/P chain. No keys, security parameters or inputs change.
    param = CkksParam(n=cfg['poly_modulus_degree'], scale=2.0 ** cfg['log_default_scale'])
    param.q = list(cfg['q'])
    param.p = list(cfg['p'])
    param.max_level = len(param.q) - 1
    random.seed(42)  # Graph identifiers only; never used for key generation.
    set_fhe_param(param)
    left = [CkksCiphertextNode(f'left_{i}', level=2) for i in range(8)]
    right = [CkksCiphertextNode(f'right_{i}', level=2) for i in range(8)]
    partials = []
    for i in range(8):
        partial = rescale(mult_relin(left[i], right[i]), f'product_{i}')
        for offset in (1, 2, 4, 8, 16):
            rotated = rotate_cols(partial, offset * 1024)[0]
            partial = add(partial, rotated, f'sum_{i}_{offset}')
        partials.append(partial)
    score = partials[0]
    for i in range(1, 8):
        score = add(score, partials[i], f'total_{i}')
    process_custom_task(input_args=[Argument('left', left), Argument('right', right)],
                        offline_input_args=[], output_args=[Argument('score', [score])],
                        output_instruction_path=str(output), fpga_acc=False)
    graph_path = output / 'mega_ag.json'
    graph = json.loads(graph_path.read_text())
    production = json.loads(parameter_file.with_name('mega_ag.json').read_text())
    add_runtime_metadata(graph, production)
    graph_path.write_text(json.dumps(graph, indent=2))


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-root', type=Path, required=True)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.frontend_root, args.parameters, args.output)
