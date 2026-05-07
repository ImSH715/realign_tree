#!/bin/bash

#SBATCH --job-name=mil_analysis
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --output=logs/mil/analysis/mil_analysis_%j.out
#SBATCH --error=logs/mil/analysis/mil_analysis_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(dirname "$0")"
fi

mkdir -p logs
mkdir -p logs/mil

module load Anaconda3
eval "$(conda shell.bash hook)"
export GEOTIFF_CSV=""
conda activate lejepa

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ------------------------------------------------------------------
# Set these from the command line when submitting.
# Example:
# MIL_OUTPUT_DIR=./outputs/mil_lejepa_shared_seasonal_20m_rgb_cuda \
# PCA_OUTPUT_DIR=./outputs/mil_lejepa_shared_seasonal_20m_rgb_cuda/pca_val_best \
# sbatch run_mil_analysis.sh
# ------------------------------------------------------------------
MIL_OUTPUT_DIR="${MIL_OUTPUT_DIR:-./outputs/mil_lejepa_shared_seasonal_20m_rgb_cuda}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best}"
SPLIT="${SPLIT:-val}"
PCA_OUTPUT_DIR="${PCA_OUTPUT_DIR:-$MIL_OUTPUT_DIR/pca_${SPLIT}_${CHECKPOINT_NAME}}"

echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "MIL output dir : $MIL_OUTPUT_DIR"
echo "PCA output dir : $PCA_OUTPUT_DIR"
echo "Python         : $(which python)"
python --version
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "============================================================"

test -d "$MIL_OUTPUT_DIR" || { echo "Missing MIL output dir: $MIL_OUTPUT_DIR"; exit 1; }

python analyze_mil_feature_space.py \
  --mil_output_dir "$MIL_OUTPUT_DIR" \
  --output_dir "$PCA_OUTPUT_DIR" \
  --split "$SPLIT" \
  --checkpoint_name "$CHECKPOINT_NAME" \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda

INSTANCE_CSV="$PCA_OUTPUT_DIR/mil_instance_pca.csv"

test -f "$INSTANCE_CSV" || { echo "Missing instance CSV: $INSTANCE_CSV"; exit 1; }

python make_mil_debug_patches.py \
  --instance_csv "$INSTANCE_CSV" \
  --output_dir "$PCA_OUTPUT_DIR/debug_top_confident" \
  --limit 80 \
  --sort_by bag_prob_1

python make_mil_debug_patches.py \
  --instance_csv "$INSTANCE_CSV" \
  --output_dir "$PCA_OUTPUT_DIR/debug_false_positives" \
  --limit 60 \
  --status FP \
  --sort_by bag_prob_1

python make_mil_debug_patches.py \
  --instance_csv "$INSTANCE_CSV" \
  --output_dir "$PCA_OUTPUT_DIR/debug_false_negatives" \
  --limit 60 \
  --status FN \
  --sort_by bag_prob_1 \
  --ascending

python make_mil_debug_patches.py \
  --instance_csv "$INSTANCE_CSV" \
  --output_dir "$PCA_OUTPUT_DIR/debug_largest_offsets" \
  --limit 80 \
  --sort_by offset_m

echo "============================================================"
echo "Analysis finished at: $(date)"
echo "Summary:"
if [ -f "$PCA_OUTPUT_DIR/mil_pca_summary.json" ]; then
  cat "$PCA_OUTPUT_DIR/mil_pca_summary.json"
  echo
fi
echo "PCA outputs      : $PCA_OUTPUT_DIR"
echo "Top confident    : $PCA_OUTPUT_DIR/debug_top_confident/contact_sheet.html"
echo "False positives  : $PCA_OUTPUT_DIR/debug_false_positives/contact_sheet.html"
echo "False negatives  : $PCA_OUTPUT_DIR/debug_false_negatives/contact_sheet.html"
echo "Largest offsets  : $PCA_OUTPUT_DIR/debug_largest_offsets/contact_sheet.html"
echo "============================================================"