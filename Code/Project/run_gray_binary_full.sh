#!/bin/bash

#SBATCH --job-name=gray_binary_full
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/gray_binary_full_%j.out
#SBATCH --error=logs/gray_binary_full_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

mkdir -p logs
mkdir -p outputs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python: $(which python)"
python --version
echo "============================================================"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

IMAGERY_ROOT="/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"

INIT_CKPT="./outputs/phase1_resnet50_cpu/phase1_encoder_best.pth"

TRAIN_SHP="./outputs/splits_binary/valid_points_train.shp"
VAL_SHP="./outputs/splits_binary/valid_points_val.shp"

TRAIN_OUT="./outputs/binary_resnet50_gray"
EVAL_OUT="./outputs/phase2_classifier_binary_gray"

RECOVERY_POINTS="./outputs/evaluation/valid_points_recovery_20m.csv"
REFINED_OUT="./outputs/evaluation/refined_classifier_binary_gray_20m.csv"

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

echo "Checking required inputs..."

test -f "$INIT_CKPT" || { echo "Missing init checkpoint: $INIT_CKPT"; exit 1; }
test -f "$TRAIN_SHP" || { echo "Missing train shapefile: $TRAIN_SHP"; exit 1; }
test -f "$VAL_SHP" || { echo "Missing val shapefile: $VAL_SHP"; exit 1; }
test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }

echo "All required inputs found."

# ------------------------------------------------------------
# Step 1: Train binary grayscale classifier
# ------------------------------------------------------------

echo "============================================================"
echo "Step 1: Training binary grayscale classifier"
echo "============================================================"

python train_supervised_encoder.py \
  --init_ckpt "$INIT_CKPT" \
  --train_shp "$TRAIN_SHP" \
  --val_shp "$VAL_SHP" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_dir "$TRAIN_OUT" \
  --label_field BinaryTree \
  --folder_field Folder \
  --file_field File \
  --fx_field fx \
  --fy_field fy \
  --coord_mode auto \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 8 \
  --epochs 40 \
  --lr_encoder 1e-5 \
  --lr_head 1e-4 \
  --weight_decay 1e-4 \
  --freeze_encoder_epochs 0 \
  --patience 0 \
  --debug_patches 64 \
  --print_val_dist \
  --num_workers 0 \
  --device cpu \
  --no_amp

# ------------------------------------------------------------
# Step 2: Evaluate classifier on validation set
# ------------------------------------------------------------

echo "============================================================"
echo "Step 2: Evaluating classifier"
echo "============================================================"

python eval_classifier_head.py \
  --encoder_ckpt "$TRAIN_OUT/phase1_encoder_best.pth" \
  --head_ckpt "$TRAIN_OUT/classifier_head_best.pth" \
  --gt_path "$VAL_SHP" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_dir "$EVAL_OUT" \
  --label_field BinaryTree \
  --folder_field Folder \
  --file_field File \
  --fx_field fx \
  --fy_field fy \
  --coord_mode auto \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 16 \
  --num_workers 0 \
  --device cpu

# ------------------------------------------------------------
# Step 3: Tune binary decision threshold
# ------------------------------------------------------------

echo "============================================================"
echo "Step 3: Tuning binary threshold"
echo "============================================================"

python tune_binary_threshold.py \
  --pred_csv "$EVAL_OUT/classifier_predictions.csv" \
  --positive_label 1 \
  --output_csv "$EVAL_OUT/threshold_tuning.csv"

# Extract best threshold by F1 from threshold_tuning.csv
BEST_THRESHOLD=$(python - <<'PY'
import pandas as pd
df = pd.read_csv("./outputs/phase2_classifier_binary_gray/threshold_tuning.csv")
best = df.sort_values("f1_shihuahuaco", ascending=False).iloc[0]
print(float(best["threshold"]))
PY
)

echo "Best threshold selected: $BEST_THRESHOLD"

# ------------------------------------------------------------
# Step 4: Optional bounded search refinement
# ------------------------------------------------------------

if [ -f "$RECOVERY_POINTS" ]; then
  echo "============================================================"
  echo "Step 4: Running classifier-based bounded search"
  echo "============================================================"

  python run_pipeline_classifier.py \
    --encoder_ckpt "$TRAIN_OUT/phase1_encoder_best.pth" \
    --head_ckpt "$TRAIN_OUT/classifier_head_best.pth" \
    --points_csv "$RECOVERY_POINTS" \
    --imagery_root "$IMAGERY_ROOT" \
    --output_csv "$REFINED_OUT" \
    --tile_column "matched_tif" \
    --point_id_column "point_id" \
    --x_column "original_east" \
    --y_column "original_north" \
    --target_label_column "label" \
    --coord_type world \
    --binary_positive_name "Shihuahuaco" \
    --decision_threshold "$BEST_THRESHOLD" \
    --search_radius_px 128 \
    --coarse_step_px 16 \
    --refine_radius_px 32 \
    --refine_step_px 8 \
    --beta 0.002 \
    --batch_size 32 \
    --device cpu

else
  echo "Skipping bounded search: recovery input not found:"
  echo "$RECOVERY_POINTS"
fi

echo "============================================================"
echo "Job finished at: $(date)"
echo "Training output: $TRAIN_OUT"
echo "Evaluation output: $EVAL_OUT"
echo "Refinement output: $REFINED_OUT"
echo "============================================================"