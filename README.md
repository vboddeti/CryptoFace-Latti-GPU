# CryptoFace GPU

Encrypted face recognition with GPU feature extraction and pair matching, using
the unchanged [FHE face-recognition harness](https://github.com/fhe-benchmarking/face-recognition/tree/a7c12d34680bdd39198a349cf1ebeec89ae88061).
Raw results and provenance: [single](measurements/single/), [small](measurements/small/).

## Reproduce

Requires four H200 GPUs, the pinned Latti-AI/LattiSense sources (private access),
NTL, model weights, real face-pair data, InsightFace models, and a Python environment
matching `requirements.txt` and the backend manifest. This is not a self-contained
public clean-clone build. Use real images only; keep builds and runtime artifacts
in scratch.

Set `SOURCE` to this checkout, `PYTHON` to the validated Python executable, and
`SCRATCH`, `LATTI_SOURCE`, `NTL_ROOT`, `MODEL_INPUT`, `DATASET` to existing inputs.
The bundled module loader targets the measured H200 cluster.

```bash
set -e
ulimit -c 0
umask 077
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$SCRATCH/tmp" "$SCRATCH/cache"
export TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/cache"
cd "$SCRATCH"
"$PYTHON" -B "$SOURCE/scripts/benchmark.py" verify
"$PYTHON" -B "$SOURCE/scripts/build_backend.py" --scratch-root "$SCRATCH" \
  --work "$SCRATCH/reproducible/backend" --repository-source "$LATTI_SOURCE" \
  --ntl-root "$NTL_ROOT" --model-input "$MODEL_INPUT"
source "$SCRATCH/reproducible/backend/source/scripts/load_h200_modules.sh"
"$PYTHON" -B "$SOURCE/scripts/build_gpu_matching.py" --scratch-root "$SCRATCH" \
  --baseline-build "$SCRATCH/reproducible/backend" --work "$SCRATCH/reproducible/gpu"

prepare() {
  "$PYTHON" -B "$SOURCE/scripts/benchmark.py" prepare --scratch-root "$SCRATCH" \
    --work "$SCRATCH/reproducible/$1" --backend-build "$SCRATCH/reproducible/gpu" --dataset "$DATASET"
}
run() {
  local work="$1"; shift
  "$PYTHON" -B "$SOURCE/scripts/benchmark.py" run --scratch-root "$SCRATCH" \
    --work "$SCRATCH/reproducible/$work" "$@"
}
prepare check
run check --size 0 --smoke --matching-check
prepare single
run single --size 0 --matching-validation "$SCRATCH/reproducible/check/completed.json"
prepare small
run small --size 1 --matching-validation "$SCRATCH/reproducible/check/completed.json" \
  --previous "$SCRATCH/reproducible/single/completed.json"
```

Stop on any failed check. Single and small each use three repetitions. Use fresh
work directories. To run small independently after the local correctness gate,
use `scripts/run_independent_small.py --help`.

Verify source/result identities with `python3 -B scripts/summarize_publication.py`.
Documentation-only publication differences are recorded in
`measurements/source-equivalence.json`; measured executable code is unchanged.
See [security scope](submission/latti/validation/SECURITY.md). Numerical agreement
is not a security proof or benchmark-organizer acceptance.
