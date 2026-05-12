#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-mil_shihuaco_factorial}"
SCRATCH_EXP_ROOT="${SCRATCH_EXP_ROOT:-/mnt/parscratch/users/aca21jo/realign_experiments/$EXPERIMENT_NAME}"
SCRATCH_LOG_ROOT="${SCRATCH_LOG_ROOT:-$SCRATCH_EXP_ROOT/slurm_logs}"
JOB_SCRIPT="$SCRIPT_DIR/job_one_mil.sh"

DRY_RUN="${DRY_RUN:-0}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-mf_dino2_shortcut_audit}"
RUN_NAME="${RUN_NAME:-dino2_rgb_white_boost_patch160_conv_lse_bag25_seed42_shortcut_audit}"

mkdir -p "$SCRATCH_LOG_ROOT"

export_vars="ALL"
export_vars="$export_vars,PROJECT_DIR=$PROJECT_DIR"
export_vars="$export_vars,FACTORIAL_SCRIPT_DIR=$SCRIPT_DIR"
export_vars="$export_vars,EXPERIMENT_NAME=$EXPERIMENT_NAME"
export_vars="$export_vars,SCRATCH_EXP_ROOT=$SCRATCH_EXP_ROOT"
export_vars="$export_vars,ENCODER=dino2"
export_vars="$export_vars,CONDA_ENV=${CONDA_ENV:-lejepa_gpu}"
export_vars="$export_vars,RUN_NAME=$RUN_NAME"
export_vars="$export_vars,TRAIN_IMAGE_MODE=${TRAIN_IMAGE_MODE:-rgb_white_boost}"
export_vars="$export_vars,EVAL_IMAGE_MODE=${EVAL_IMAGE_MODE:-${TRAIN_IMAGE_MODE:-rgb_white_boost}}"
export_vars="$export_vars,PATCH_SIZE_PX=${PATCH_SIZE_PX:-160}"
export_vars="$export_vars,POOLING=${POOLING:-conv_lse}"
export_vars="$export_vars,BAG_LAYOUT=${BAG_LAYOUT:-grid}"
export_vars="$export_vars,BAG_INSTANCES=${BAG_INSTANCES:-25}"
export_vars="$export_vars,CONV_KERNEL_SIZE=${CONV_KERNEL_SIZE:-3}"
export_vars="$export_vars,EPOCHS=${EPOCHS:-25}"
export_vars="$export_vars,MONITOR_METRIC=${MONITOR_METRIC:-val_average_precision}"
export_vars="$export_vars,PATIENCE=${PATIENCE:-0}"
export_vars="$export_vars,SEED=${SEED:-42}"
export_vars="$export_vars,BATCH_SIZE=${BATCH_SIZE:-2}"
export_vars="$export_vars,FREEZE_ENCODER_EPOCHS=${FREEZE_ENCODER_EPOCHS:-3}"
export_vars="$export_vars,LR_ENCODER=${LR_ENCODER:-5e-7}"
export_vars="$export_vars,LR_HEAD=${LR_HEAD:-1e-4}"
export_vars="$export_vars,TRAIN_REPEAT_FACTOR=${TRAIN_REPEAT_FACTOR:-2}"
export_vars="$export_vars,MAX_BLACK_FRACTION=${MAX_BLACK_FRACTION:-0.20}"
export_vars="$export_vars,MAX_BRIGHT_FRACTION=${MAX_BRIGHT_FRACTION:-0.35}"

out_path="$SCRATCH_LOG_ROOT/${JOB_NAME}_%j.out"
err_path="$SCRATCH_LOG_ROOT/${JOB_NAME}_%j.err"

if [ "$DRY_RUN" = "1" ]; then
  echo "sbatch --chdir $PROJECT_DIR --job-name $JOB_NAME --time $SLURM_TIME --output $out_path --error $err_path --export $export_vars $JOB_SCRIPT"
else
  sbatch --chdir "$PROJECT_DIR" \
    --job-name "$JOB_NAME" \
    --time "$SLURM_TIME" \
    --output "$out_path" \
    --error "$err_path" \
    --export="$export_vars" \
    "$JOB_SCRIPT"
fi

echo "Audit run name : $RUN_NAME"
echo "SLURM time     : $SLURM_TIME"
echo "Scratch root   : $SCRATCH_EXP_ROOT"
echo "SLURM logs     : $SCRATCH_LOG_ROOT"
