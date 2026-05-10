#!/bin/bash

set -euo pipefail

JOB_REGEX="${JOB_REGEX:-^(mf_|mil_fact)}"
DRY_RUN="${DRY_RUN:-1}"

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue not found; run this on the SLURM login node."
  exit 1
fi

if ! command -v scancel >/dev/null 2>&1; then
  echo "scancel not found; run this on the SLURM login node."
  exit 1
fi

mapfile -t job_ids < <(squeue -h -u "${USER:-$LOGNAME}" -o "%A %j" | awk -v re="$JOB_REGEX" '$2 ~ re {print $1}')

if [ "${#job_ids[@]}" -eq 0 ]; then
  echo "No matching jobs found for regex: $JOB_REGEX"
  exit 0
fi

echo "Matching jobs:"
squeue -u "${USER:-$LOGNAME}" -o "%.18i %.9P %.28j %.8T %.10M %.9l %.6D %R" | awk -v re="$JOB_REGEX" 'NR == 1 || $3 ~ re'

echo
if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run only. To cancel these jobs, run:"
  echo "  DRY_RUN=0 bash experiments/mil_shihuaco_factorial/cancel_factorial_jobs.sh"
else
  echo "Cancelling ${#job_ids[@]} jobs..."
  scancel "${job_ids[@]}"
  echo "Cancel request sent."
fi
