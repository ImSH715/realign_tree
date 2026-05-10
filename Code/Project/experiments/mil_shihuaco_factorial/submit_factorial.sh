#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

JOB_SCRIPT="$SCRIPT_DIR/job_one_mil.sh"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mil_shihuaco_factorial}"

read -r -a ENCODER_LIST <<< "${ENCODERS:-dino2 dino3 lejepa}"
read -r -a IMAGE_MODE_LIST <<< "${IMAGE_MODES:-rgb_white_boost rgb_green_mean_white_boost}"
read -r -a PATCH_SIZE_LIST <<< "${PATCH_SIZES:-160 224 320}"
read -r -a POOLING_LIST <<< "${POOLINGS:-lse conv_lse}"
read -r -a SEED_LIST <<< "${SEEDS:-42}"

BAG_LAYOUT="${BAG_LAYOUT:-grid}"
BAG_INSTANCES="${BAG_INSTANCES:-25}"
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-3}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p logs/mil_factorial
mkdir -p "$SCRIPT_DIR/results"

if [ "$DRY_RUN" != "1" ] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch was not found. Run on the SLURM login node, or set DRY_RUN=1 to preview."
  exit 1
fi

count=0

for encoder in "${ENCODER_LIST[@]}"; do
  for image_mode in "${IMAGE_MODE_LIST[@]}"; do
    for patch_size in "${PATCH_SIZE_LIST[@]}"; do
      for pooling in "${POOLING_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
          run_name="${encoder}_${image_mode}_patch${patch_size}_${pooling}_bag${BAG_INSTANCES}_seed${seed}"
          job_name="mf_${encoder}_${patch_size}_${pooling}"
          export_vars="ALL,EXPERIMENT_NAME=${EXPERIMENT_NAME},ENCODER=${encoder},TRAIN_IMAGE_MODE=${image_mode},EVAL_IMAGE_MODE=${image_mode},PATCH_SIZE_PX=${patch_size},POOLING=${pooling},BAG_LAYOUT=${BAG_LAYOUT},BAG_INSTANCES=${BAG_INSTANCES},CONV_KERNEL_SIZE=${CONV_KERNEL_SIZE},SEED=${seed},RUN_NAME=${run_name}"

          count=$((count + 1))
          if [ "$DRY_RUN" = "1" ]; then
            echo "[$count] sbatch --job-name $job_name --export $export_vars $JOB_SCRIPT"
          else
            sbatch --job-name "$job_name" --export="$export_vars" "$JOB_SCRIPT"
          fi
        done
      done
    done
  done
done

echo "Queued grid size: $count"
echo "Results root    : $SCRIPT_DIR/results"
echo "Collect with    : python experiments/mil_shihuaco_factorial/collect_results.py"
