#!/usr/bin/env bash

set -euo pipefail

trial_job_id="${1:?usage: resubmit_large_on_timeout.sh TRIAL_JOB_ID}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

state="$(sacct -X -n -j "${trial_job_id}" --format=State | xargs)"
echo "trial_job_id=${trial_job_id}"
echo "trial_state=${state}"

if [[ "${state}" == TIMEOUT* ]]; then
  cd "${root}"
  env -u CRYPTOFACE_GPUS sbatch --parsable \
    --job-name=cf-large-r1-6h512 \
    --partition=general-long-gpu \
    --time=06:00:00 \
    --mem=512G \
    --gres=gpu:h200:4 \
    --export=ALL,CRYPTOFACE_SIZE=3,CRYPTOFACE_SEED=42,CRYPTOFACE_NUM_RUNS=1,CRYPTOFACE_RUN_OFFSET=0 \
    scripts/run_h200_slurm.sh
else
  echo "No retry submitted: the trial did not end in TIMEOUT."
fi
