#!/bin/bash

set -euo pipefail

DEFAULT_LOG_ROOT="/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial/slurm_logs"
LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
TAIL_LINES="${TAIL_LINES:-80}"

if [ "$#" -eq 0 ]; then
  echo "Usage: bash experiments/mil_shihuaco_factorial/monitor_job_ids.sh <job-id> [job-id ...]"
  echo
  echo "Example:"
  echo "  bash experiments/mil_shihuaco_factorial/monitor_job_ids.sh 10153829 10153830 10153831"
  exit 1
fi

job_csv="$(IFS=,; echo "$*")"

echo "============================================================"
echo "MIL factorial job monitor"
echo "Time     : $(date)"
echo "Job IDs  : $job_csv"
echo "Log root : $LOG_ROOT"
echo "============================================================"

if command -v squeue >/dev/null 2>&1; then
  echo
  echo "Active queue state"
  squeue -j "$job_csv" -o "%.18i %.9P %.28j %.8T %.10M %.9l %.6D %R" || true
else
  echo
  echo "squeue not found."
fi

if command -v sacct >/dev/null 2>&1; then
  echo
  echo "Accounting state"
  sacct -j "$job_csv" --format=JobID,JobName%32,State,Elapsed,ExitCode,Submit,Start,End --parsable2 || true
else
  echo
  echo "sacct not found."
fi

echo
echo "Log tails"
for job_id in "$@"; do
  echo
  echo "---- Job $job_id ----"
  matches=()
  while IFS= read -r path; do
    matches+=("$path")
  done < <(find "$LOG_ROOT" logs/mil_factorial -maxdepth 1 -type f \( -name "*_${job_id}.out" -o -name "*_${job_id}.err" \) 2>/dev/null | sort)

  if [ "${#matches[@]}" -eq 0 ]; then
    echo "No logs found yet for $job_id."
    continue
  fi

  for log_path in "${matches[@]}"; do
    echo
    echo "File: $log_path"
    tail -n "$TAIL_LINES" "$log_path" || true
  done
done
