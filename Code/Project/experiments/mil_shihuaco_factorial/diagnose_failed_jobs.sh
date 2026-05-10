#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

DEFAULT_SCRATCH_LOG_DIR="/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial/slurm_logs"
LOG_DIR="${LOG_DIR:-$DEFAULT_SCRATCH_LOG_DIR}"
TAIL_LINES="${TAIL_LINES:-80}"

if [ "$#" -eq 0 ]; then
  echo "Usage: bash experiments/mil_shihuaco_factorial/diagnose_failed_jobs.sh <job-id> [job-id ...]"
  echo
  echo "Example:"
  echo "  bash experiments/mil_shihuaco_factorial/diagnose_failed_jobs.sh 10152392 10152399"
  exit 1
fi

for job_id in "$@"; do
  echo "============================================================"
  echo "Job $job_id"
  echo "============================================================"

  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" --format=JobID,JobName%32,State,Elapsed,ExitCode,Submit,Start,End --parsable2 || true
  else
    echo "sacct not found."
  fi

  echo
  echo "Matching logs"
  matches=()
  while IFS= read -r path; do
    matches+=("$path")
  done < <(find "$LOG_DIR" logs/mil_factorial -maxdepth 1 -type f \( -name "*_${job_id}.out" -o -name "*_${job_id}.err" \) 2>/dev/null | sort)

  if [ "${#matches[@]}" -eq 0 ]; then
    echo "No logs found under $LOG_DIR or logs/mil_factorial for job $job_id."
    echo "If SLURM rejected the output path, check the directory exists on the submit node."
    continue
  fi

  for log_path in "${matches[@]}"; do
    echo
    echo "---- $log_path (last $TAIL_LINES lines) ----"
    tail -n "$TAIL_LINES" "$log_path" || true
  done
done
