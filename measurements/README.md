# Measured GPU results

See [PUBLICATION.md](../PUBLICATION.md) for hardware, exact source/build/input
identities, metric definitions, security limitations and reproduction prerequisites.

- `small/results-{1,2,3}.json`: three repetitions from one unmodified upstream
  `--num_runs 3 --seed 42` invocation, 128 real pairs each. All passed the EER gate.
- `publication-summary.json`: arithmetic means derived from these raw JSONs.
- Each result directory contains the successful run provenance and measured
  release manifest. Raw JSON files are not edited or reformatted for publication.

Only small results are submitted. Other locally retained measurements are excluded
from the published Git tree. No results from the modified optimization
harness are mixed into these measurements. Model weights, keys, ciphertexts,
profiles and other regenerable runtime artifacts are not included.

Recompute and verify the summary with:

```text
python3 -B scripts/summarize_publication.py
```
