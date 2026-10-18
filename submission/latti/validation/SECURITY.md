# Security estimate

The bundled Q44/P61/H640 configuration passes the project's 128-bit
classical-security gate at **128.31868162155314 bits**.

## Model

- Estimator: `malb/lattice-estimator`
- Estimator commit: `3e48ef421ec256afddb3e7d2249a77eab6e9ba12`
- Reduction cost: MATZOV default classical model
- Reduction shape: estimator default GSA model
- Ring/LWE dimension: 65,536
- Modulus: product of all 32 Q and 3 P limbs
- `log2(QP)`: 1737.9999663450142
- Error: discrete Gaussian sigma 3.2
- Secret: sparse ternary, total Hamming weight 640, represented as
  `ND.SparseTernary(320, 320, n=65536)`
- Samples: unlimited
- Bootstrap ephemeral secret weight: 32

## Complete relevant attack suite

- USVP: 128.500680 bits
- BDD: 128.496349 bits
- BDD hybrid: 128.490127 bits
- BDD MITM hybrid: **128.318682 bits** (limiting)
- Dual: 129.368604 bits
- Dual hybrid: 129.536312 bits

Arora-GB is excluded because the discrete-Gaussian error is unbounded. BKW is
excluded because the estimator documents it as noncompetitive in this regime.
P is included because evaluation and bootstrap keys are published modulo QP;
estimating Q alone would overstate security.

The executable estimator input is `security/estimate_q44p61.py`; the captured
result is `security/estimator_report.json`.
