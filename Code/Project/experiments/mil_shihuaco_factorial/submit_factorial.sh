#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
export PROJECT_DIR
export FACTORIAL_SCRIPT_DIR="$SCRIPT_DIR"

JOB_SCRIPT="$SCRIPT_DIR/job_one_mil.sh"
SWEEP_SCRIPT="$SCRIPT_DIR/job_encoder_sweep.sh"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mil_shihuaco_factorial}"
SUBMIT_MODE="${SUBMIT_MODE:-model_pooling_sweep}"
SCRATCH_EXP_ROOT="${SCRATCH_EXP_ROOT:-/mnt/parscratch/users/aca21jo/realign_experiments/$EXPERIMENT_NAME}"
SCRATCH_LOG_ROOT="${SCRATCH_LOG_ROOT:-$SCRATCH_EXP_ROOT/slurm_logs}"

read -r -a ENCODER_LIST <<< "${ENCODERS:-dino2 dino3 lejepa}"
read -r -a IMAGE_MODE_LIST <<< "${IMAGE_MODES:-rgb_white_boost rgb_green_mean_white_boost}"
read -r -a PATCH_SIZE_LIST <<< "${PATCH_SIZES:-160}"
read -r -a POOLING_LIST <<< "${POOLINGS:-lse conv_lse}"
read -r -a SEED_LIST <<< "${SEEDS:-42}"

BAG_LAYOUT="${BAG_LAYOUT:-grid}"
BAG_INSTANCES="${BAG_INSTANCES:-25}"
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-3}"
DRY_RUN="${DRY_RUN:-0}"
SLURM_TIME="${SLURM_TIME:-90:00:00}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"

SBATCH_DEP_ARGS=()
SBATCH_DEP_TEXT=""
if [ -n "$SLURM_DEPENDENCY" ]; then
  SBATCH_DEP_ARGS=(--dependency "$SLURM_DEPENDENCY")
  SBATCH_DEP_TEXT="--dependency $SLURM_DEPENDENCY "
fi

mkdir -p "$SCRIPT_DIR/results" 2>/dev/null || true
if ! mkdir -p "$SCRATCH_LOG_ROOT"; then
  echo "Could not create scratch log root: $SCRATCH_LOG_ROOT"
  exit 1
fi

export IMAGE_MODES="${IMAGE_MODES:-rgb_white_boost rgb_green_mean_white_boost}"
export PATCH_SIZES="${PATCH_SIZES:-160}"
export SEEDS="${SEEDS:-42}"

if [ "$DRY_RUN" != "1" ] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch was not found. Run on the SLURM login node, or set DRY_RUN=1 to preview."
  exit 1
fi

count=0

if [ "$SUBMIT_MODE" = "model_pooling_sweep" ]; then
  for encoder in "${ENCODER_LIST[@]}"; do
    for pooling in "${POOLING_LIST[@]}"; do
      count=$((count + 1))
      job_name="mf_${encoder}_${pooling}"
      export POOLINGS="$pooling"
      export_vars="ALL,PROJECT_DIR=${PROJECT_DIR},FACTORIAL_SCRIPT_DIR=${SCRIPT_DIR},EXPERIMENT_NAME=${EXPERIMENT_NAME},ENCODER=${encoder},POOLINGS=${pooling},BAG_LAYOUT=${BAG_LAYOUT},BAG_INSTANCES=${BAG_INSTANCES},CONV_KERNEL_SIZE=${CONV_KERNEL_SIZE},SCRATCH_EXP_ROOT=${SCRATCH_EXP_ROOT}"
      out_path="$SCRATCH_LOG_ROOT/${job_name}_%j.out"
      err_path="$SCRATCH_LOG_ROOT/${job_name}_%j.err"

      if [ "$DRY_RUN" = "1" ]; then
        echo "[$count] sbatch --chdir $PROJECT_DIR ${SBATCH_DEP_TEXT}--job-name $job_name --time $SLURM_TIME --output $out_path --error $err_path --export $export_vars $SWEEP_SCRIPT"
      else
        sbatch --chdir "$PROJECT_DIR" "${SBATCH_DEP_ARGS[@]}" --job-name "$job_name" --time "$SLURM_TIME" --output "$out_path" --error "$err_path" --export="$export_vars" "$SWEEP_SCRIPT"
      fi
    done
  done
elif [ "$SUBMIT_MODE" = "encoder_sweep" ]; then
  for encoder in "${ENCODER_LIST[@]}"; do
    count=$((count + 1))
    job_name="mf_sweep_${encoder}"
    export_vars="ALL,PROJECT_DIR=${PROJECT_DIR},FACTORIAL_SCRIPT_DIR=${SCRIPT_DIR},EXPERIMENT_NAME=${EXPERIMENT_NAME},ENCODER=${encoder},BAG_LAYOUT=${BAG_LAYOUT},BAG_INSTANCES=${BAG_INSTANCES},CONV_KERNEL_SIZE=${CONV_KERNEL_SIZE},SCRATCH_EXP_ROOT=${SCRATCH_EXP_ROOT}"
    out_path="$SCRATCH_LOG_ROOT/${job_name}_%j.out"
    err_path="$SCRATCH_LOG_ROOT/${job_name}_%j.err"

    if [ "$DRY_RUN" = "1" ]; then
      echo "[$count] sbatch --chdir $PROJECT_DIR ${SBATCH_DEP_TEXT}--job-name $job_name --time $SLURM_TIME --output $out_path --error $err_path --export $export_vars $SWEEP_SCRIPT"
    else
      sbatch --chdir "$PROJECT_DIR" "${SBATCH_DEP_ARGS[@]}" --job-name "$job_name" --time "$SLURM_TIME" --output "$out_path" --error "$err_path" --export="$export_vars" "$SWEEP_SCRIPT"
    fi
  done
else
  for encoder in "${ENCODER_LIST[@]}"; do
    for image_mode in "${IMAGE_MODE_LIST[@]}"; do
      for patch_size in "${PATCH_SIZE_LIST[@]}"; do
        for pooling in "${POOLING_LIST[@]}"; do
          for seed in "${SEED_LIST[@]}"; do
            run_name="${encoder}_${image_mode}_patch${patch_size}_${pooling}_bag${BAG_INSTANCES}_seed${seed}"
            job_name="mf_${encoder}_${patch_size}_${pooling}"
            export_vars="ALL,PROJECT_DIR=${PROJECT_DIR},FACTORIAL_SCRIPT_DIR=${SCRIPT_DIR},EXPERIMENT_NAME=${EXPERIMENT_NAME},ENCODER=${encoder},TRAIN_IMAGE_MODE=${image_mode},EVAL_IMAGE_MODE=${image_mode},PATCH_SIZE_PX=${patch_size},POOLING=${pooling},BAG_LAYOUT=${BAG_LAYOUT},BAG_INSTANCES=${BAG_INSTANCES},CONV_KERNEL_SIZE=${CONV_KERNEL_SIZE},SEED=${seed},RUN_NAME=${run_name},SCRATCH_EXP_ROOT=${SCRATCH_EXP_ROOT}"
            out_path="$SCRATCH_LOG_ROOT/${job_name}_%j.out"
            err_path="$SCRATCH_LOG_ROOT/${job_name}_%j.err"

            count=$((count + 1))
            if [ "$DRY_RUN" = "1" ]; then
              echo "[$count] sbatch --chdir $PROJECT_DIR ${SBATCH_DEP_TEXT}--job-name $job_name --time $SLURM_TIME --output $out_path --error $err_path --export $export_vars $JOB_SCRIPT"
            else
              sbatch --chdir "$PROJECT_DIR" "${SBATCH_DEP_ARGS[@]}" --job-name "$job_name" --time "$SLURM_TIME" --output "$out_path" --error "$err_path" --export="$export_vars" "$JOB_SCRIPT"
            fi
          done
        done
      done
    done
  done
fi

echo "Submit mode     : $SUBMIT_MODE"
echo "Queued jobs     : $count"
echo "SLURM time      : $SLURM_TIME"
echo "Dependency      : ${SLURM_DEPENDENCY:-<none>}"
echo "Scratch root    : $SCRATCH_EXP_ROOT"
echo "SLURM logs      : $SCRATCH_LOG_ROOT"
echo "Local links     : $SCRIPT_DIR/results"
echo "Collect with    : python experiments/mil_shihuaco_factorial/collect_results.py"
