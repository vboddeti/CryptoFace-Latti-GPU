#!/usr/bin/env python3
"""Reproduce the concrete LWE security estimate for the bundled CKKS parameters."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator-root", required=True, type=Path)
    parser.add_argument(
        "--parameter",
        type=Path,
        default=Path(__file__).parents[1] / "model/server/ckks_parameter.json",
    )
    args = parser.parse_args()

    estimator_root = args.estimator_root.resolve()
    sys.path.insert(0, str(estimator_root))

    from sage.all import RealField, log, oo  # pylint: disable=import-error
    from estimator import LWE, ND  # pylint: disable=import-error
    from estimator.conf import red_cost_model, red_shape_model  # pylint: disable=import-error

    config = json.loads(args.parameter.read_text(encoding="utf-8"))["param0"]
    n = int(config["poly_modulus_degree"])
    q_limbs = [int(value) for value in config["q"]]
    p_limbs = [int(value) for value in config["p"]]
    qp = 1
    for limb in q_limbs + p_limbs:
        qp *= limb

    params = LWE.Parameters(
        n=n,
        q=qp,
        Xe=ND.DiscreteGaussian(3.2),
        # Lattigo H=640 is total Hamming weight. SparseTernary's first two
        # arguments are the positive and negative counts independently.
        Xs=ND.SparseTernary(320, 320, n=n),
        m=oo,
        tag="CryptoFace-Q44-P61-H640",
    )
    estimates = LWE.estimate(
        params,
        deny_list=("arora-gb", "bkw"),
        jobs=6,
        quiet=True,
    )

    attacks = {}
    for name, result in estimates.items():
        rop = result.get("rop", oo)
        if rop == oo:
            continue
        attacks[name] = {
            "log2_rop": float(log(rop, 2)),
            "summary": repr(result),
        }

    limiting_name = min(attacks, key=lambda name: attacks[name]["log2_rop"])
    commit = subprocess.check_output(
        ["git", "-C", str(estimator_root), "rev-parse", "HEAD"], text=True
    ).strip()
    report = {
        "estimator": "malb/lattice-estimator",
        "estimator_commit": commit,
        "cost_model": repr(red_cost_model),
        "shape_model": repr(red_shape_model),
        "parameters": {
            "n": n,
            "q_limb_count": len(q_limbs),
            "p_limb_count": len(p_limbs),
            "log2_qp": float(RealField(200)(qp).log() / RealField(200)(2).log()),
            "error_sigma": 3.2,
            "secret_distribution": "SparseTernary",
            "secret_hamming_weight": 640,
            "samples": "unlimited",
        },
        "excluded_attacks": {
            "arora-gb": "not applicable to unbounded discrete-Gaussian error",
            "bkw": "estimator documentation classifies BKW as noncompetitive in this regime",
        },
        "attacks": attacks,
        "limiting_attack": limiting_name,
        "classical_security_bits": attacks[limiting_name]["log2_rop"],
        "passes_128_bit_gate": attacks[limiting_name]["log2_rop"] >= 128.0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
