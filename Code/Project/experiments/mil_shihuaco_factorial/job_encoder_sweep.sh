#!/bin/bash

#SBATCH --job-name=mil_sweep
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-mil_shihuaco_factorial}"
ENCODER="${ENCODER:-dino2}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
SCRATCH_EXP_ROOT="${SCRATCH_EXP_ROOT:-/mnt/parscratch/users/aca21jo/realign_experiments/$EXPERIMENT_NAME}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

read -r -a IMAGE_MODE_LIST <<< "${IMAGE_MODES:-rgb_white_boost rgb_green_mean_white_boost}"
read -r -a PATCH_SIZE_LIST <<< "${PATCH_SIZES:-160}"
read -r -a POOLING_LIST <<< "${POOLINGS:-lse conv_lse}"
read -r -a SEED_LIST <<< "${SEEDS:-42}"

failures=0
total=0

echo "============================================================"
echo "Encoder sweep started : $(date)"
echo "Project               : $PROJECT_DIR"
echo "Experiment            : $EXPERIMENT_NAME"
echo "Encoder               : $ENCODER"
echo "Image modes           : ${IMAGE_MODE_LIST[*]}"
echo "Patch sizes           : ${PATCH_SIZE_LIST[*]}"
echo "Poolings              : ${POOLING_LIST[*]}"
echo "Seeds                 : ${SEED_LIST[*]}"
echo "Continue on fail      : $CONTINUE_ON_FAIL"
echo "Skip completed        : $SKIP_COMPLETED"
echo "Scratch root          : $SCRATCH_EXP_ROOT"
echo "============================================================"

for image_mode in "${IMAGE_MODE_LIST[@]}"; do
  for patch_size in "${PATCH_SIZE_LIST[@]}"; do
    for pooling in "${POOLING_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        total=$((total + 1))
        run_name="${ENCODER}_${image_mode}_patch${patch_size}_${pooling}_bag${BAG_INSTANCES:-25}_seed${seed}"
        run_dir="$SCRATCH_EXP_ROOT/$run_name"

        echo
        echo "============================================================"
        echo "Sweep item $total"
        echo "Run name      : $run_name"
        echo "Image mode    : $image_mode"
        echo "Patch size px : $patch_size"
        echo "Pooling       : $pooling"
        echo "Seed          : $seed"
        echo "Started       : $(date)"
        echo "============================================================"

        if [ "$SKIP_COMPLETED" = "1" ] \
          && [ -f "$run_dir/classification_report.json" ] \
          && [ -f "$run_dir/binary_score_diagnostics.json" ] \
          && [ -f "$run_dir/threshold_tuning.csv" ]; then
          echo "Skipping completed run: $run_name"
          continue
        fi

        if ENCODER="$ENCODER" \
          TRAIN_IMAGE_MODE="$image_mode" \
          EVAL_IMAGE_MODE="$image_mode" \
          PATCH_SIZE_PX="$patch_size" \
          POOLING="$pooling" \
          SEED="$seed" \
          RUN_NAME="$run_name" \
          EXPERIMENT_NAME="$EXPERIMENT_NAME" \
          bash "$SCRIPT_DIR/job_one_mil.sh"; then
          echo "Completed run: $run_name"
        else
          failures=$((failures + 1))
          echo "FAILED run: $run_name"
          if [ "$CONTINUE_ON_FAIL" != "1" ]; then
            echo "Stopping encoder sweep after first failure. Set CONTINUE_ON_FAIL=1 to keep going."
            exit 1
          fi
        fi
      done
    done
  done
done

echo
echo "============================================================"
echo "Encoder sweep finished : $(date)"
echo "Total runs             : $total"
echo "Failures               : $failures"
echo "============================================================"

if [ "$failures" -gt 0 ]; then
  exit 1
fi
