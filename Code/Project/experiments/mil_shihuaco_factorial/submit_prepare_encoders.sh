#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

ENCODERS="${ENCODERS:-lejepa dino3}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
SLURM_TIME="${SLURM_TIME:-90:00:00}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mil_shihuaco_factorial}"
SCRATCH_EXP_ROOT="${SCRATCH_EXP_ROOT:-/mnt/parscratch/users/aca21jo/realign_experiments/$EXPERIMENT_NAME}"
SCRATCH_LOG_ROOT="${SCRATCH_LOG_ROOT:-$SCRATCH_EXP_ROOT/slurm_logs}"
JOB_SCRIPT="$SCRIPT_DIR/prepare_encoder_checkpoint.sh"

mkdir -p "$SCRATCH_LOG_ROOT"

if [ "$DRY_RUN" != "1" ] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch was not found. Run on the SLURM login node, or set DRY_RUN=1 to preview."
  exit 1
fi

count=0
for encoder in $ENCODERS; do
  count=$((count + 1))
  job_name="prep_${encoder}"
  out_path="$SCRATCH_LOG_ROOT/${job_name}_%j.out"
  err_path="$SCRATCH_LOG_ROOT/${job_name}_%j.err"
  export_vars="ALL,PROJECT_DIR=${PROJECT_DIR},FACTORIAL_SCRIPT_DIR=${SCRIPT_DIR},ENCODER=${encoder},STAGE=${STAGE}"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[$count] sbatch --chdir $PROJECT_DIR --job-name $job_name --time $SLURM_TIME --output $out_path --error $err_path --export $export_vars $JOB_SCRIPT"
  else
    sbatch --chdir "$PROJECT_DIR" --job-name "$job_name" --time "$SLURM_TIME" --output "$out_path" --error "$err_path" --export="$export_vars" "$JOB_SCRIPT"
  fi
done

echo "Queued prep jobs : $count"
echo "Encoders         : $ENCODERS"
echo "Stage            : $STAGE"
echo "SLURM logs       : $SCRATCH_LOG_ROOT"
