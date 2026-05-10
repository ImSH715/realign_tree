#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

DEFAULT_SCRATCH_ROOT="/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial"
RESULTS_ROOT="${RESULTS_ROOT:-$DEFAULT_SCRATCH_ROOT}"
SUMMARY_CSV="${SUMMARY_CSV:-$SCRIPT_DIR/summary.csv}"
SUMMARY_MD="${SUMMARY_MD:-$SCRIPT_DIR/summary.md}"
JOB_MATCH="${JOB_MATCH:-mf_|mil_fact}"
SACCT_DAYS="${SACCT_DAYS:-3}"
FAILED_LOG_TAIL="${FAILED_LOG_TAIL:-1}"
TAIL_LINES="${TAIL_LINES:-40}"
LOG_ROOT="${LOG_ROOT:-$DEFAULT_SCRATCH_ROOT/slurm_logs}"

python_cmd() {
  if command -v python3 >/dev/null 2>&1; then
    printf "python3"
  elif command -v python >/dev/null 2>&1; then
    printf "python"
  else
    return 1
  fi
}

count_matching_files() {
  local pattern="$1"
  if [ ! -d "$RESULTS_ROOT" ]; then
    printf "0"
    return
  fi
  find "$RESULTS_ROOT" -mindepth 2 -maxdepth 2 -name "$pattern" | wc -l | tr -d " "
}

count_run_dirs() {
  if [ ! -d "$RESULTS_ROOT" ]; then
    printf "0"
    return
  fi
  find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d " "
}

echo "============================================================"
echo "MIL factorial monitor"
echo "Time         : $(date)"
echo "Project      : $PROJECT_DIR"
echo "Results root : $RESULTS_ROOT"
echo "============================================================"

if command -v squeue >/dev/null 2>&1; then
  echo
  echo "Active SLURM jobs"
  squeue -u "${USER:-$LOGNAME}" -o "%.18i %.9P %.28j %.8T %.10M %.9l %.6D %R" | grep -E "JOBID|$JOB_MATCH" || true
else
  echo
  echo "squeue not found; run this on the SLURM login node for active-job status."
fi

if command -v sacct >/dev/null 2>&1; then
  echo
  echo "Recent SLURM history"
  start_date="$(date -d "$SACCT_DAYS days ago" +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)"
  sacct -u "${USER:-$LOGNAME}" -S "$start_date" \
    --format=JobID,JobName%28,State,Elapsed,ExitCode,MaxRSS%12 \
    --parsable2 | grep -E "JobID|$JOB_MATCH" | tail -n 40 || true
else
  echo
  echo "sacct not found; skipping recent job history."
fi

if [ "$FAILED_LOG_TAIL" = "1" ] && command -v sacct >/dev/null 2>&1; then
  echo
  echo "Recent failed-job log tails"
  start_date="$(date -d "$SACCT_DAYS days ago" +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)"
  failed_ids="$(sacct -u "${USER:-$LOGNAME}" -S "$start_date" --format=JobIDRaw,JobName%28,State --parsable2 \
    | grep -E "$JOB_MATCH" \
    | awk -F'|' '$3 ~ /FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY/ {print $1}' \
    | sort -u \
    | tail -n 5)"
  if [ -z "$failed_ids" ]; then
    echo "No recent failed matching jobs found."
  else
    for job_id in $failed_ids; do
      echo
      echo "Job $job_id"
      find "$LOG_ROOT" logs/mil_factorial -maxdepth 1 -type f \( -name "*_${job_id}.out" -o -name "*_${job_id}.err" \) 2>/dev/null | sort | while read -r log_path; do
        echo "---- $log_path ----"
        tail -n "$TAIL_LINES" "$log_path" || true
      done
    done
  fi
fi

run_dirs="$(count_run_dirs)"
complete_reports="$(count_matching_files classification_report.json)"
diagnostics="$(count_matching_files binary_score_diagnostics.json)"
thresholds="$(count_matching_files threshold_tuning.csv)"

echo
echo "Result files"
echo "Run directories        : $run_dirs"
echo "Classification reports : $complete_reports"
echo "Score diagnostics      : $diagnostics"
echo "Threshold CSVs         : $thresholds"

if py="$(python_cmd)"; then
  echo
  echo "Refreshing summary"
  "$py" "$SCRIPT_DIR/collect_results.py" \
    --results_root "$RESULTS_ROOT" \
    --output_csv "$SUMMARY_CSV" \
    --output_md "$SUMMARY_MD"
else
  echo
  echo "No python executable found; cannot refresh summary."
fi

if [ -f "$SUMMARY_MD" ]; then
  echo
  echo "Top summary rows"
  sed -n "1,24p" "$SUMMARY_MD"
else
  echo
  echo "No summary markdown found yet."
fi

echo
echo "Useful follow-ups"
echo "Diagnose jobs : bash experiments/mil_shihuaco_factorial/diagnose_failed_jobs.sh <job-id> [...]"
echo "Tail live logs : tail -f $LOG_ROOT/<job-name>_<job-id>.out"
echo "Refresh watch  : watch -n 60 bash experiments/mil_shihuaco_factorial/monitor_factorial.sh"
echo "Summary CSV    : $SUMMARY_CSV"
