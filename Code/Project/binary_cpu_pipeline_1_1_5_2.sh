#!/bin/bash

#SBATCH --job-name=phase_1_15_2
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/preprocessing/p_lejepa_%j.out
#SBATCH --error=logs/preprocessing/p_lejepa_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

mkdir -p logs
mkdir -p logs/phase1
mkdir -p logs/phase_1_5
mkdir -p logs/phase2
mkdir -p outputs
mkdir -p outputs/evaluation

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

# binary split
TRAIN_SHP="./outputs/splits_binary/valid_points_train.shp"
VAL_SHP="./outputs/splits_binary/valid_points_val.shp"

# phase outputs
PHASE1_OUT="./outputs/phase1_lejepa_cpu_binary_preprocess"
PHASE1_5_OUT="./outputs/phase1_5_lejepa_cpu_binary_preprocess"
PHASE2_OUT="./outputs/phase2_binary_shihuahuaco_preprocess"
PHASE2_MULTI_OUT="./outputs/phase2_binary_shihuahuaco_multi_preprocess"
EVAL_OUT="./outputs/eval_binary_classifier_val_preprocess"

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

echo "Checking required inputs..."

test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }
test -f "$TRAIN_SHP" || { echo "Missing train shapefile: $TRAIN_SHP"; exit 1; }
test -f "$VAL_SHP" || { echo "Missing validation shapefile: $VAL_SHP"; exit 1; }

echo "All required inputs found."

# ------------------------------------------------------------
# Phase 1: SSL pretraining (CPU, LeJEPA backbone)
# ------------------------------------------------------------

echo "============================================================"
echo "PHASE 1: SSL encoder training (binary experiment base encoder)"
echo "============================================================"

python train_encoder.py \
  --train_root "$IMAGERY_ROOT" \
  --output_dir "$PHASE1_OUT" \
  --backbone_name "vit_base_patch16_224" \
  --ssl_epochs 30 \
  --batch_size_ssl 8 \
  --patches_per_image 10 \
  --num_workers 8 \
  --device cpu \
  --no_amp \
  --extract_stride_px 1024 \
  --extract_batch_size 16 \
  --image_size_global 224 \
  --image_size_local 224 \
  --max_extract_patches_per_image 20

echo "[DONE] Phase 1"

# ------------------------------------------------------------
# Phase 1.5: supervised fine-tuning for binary classification
# ------------------------------------------------------------

echo "============================================================"
echo "PHASE 1.5: supervised fine-tuning (BinaryTree)"
echo "============================================================"

python train_supervised_encoder.py \
  --init_ckpt "$PHASE1_OUT/phase1_encoder_best.pth" \
  --train_shp "$TRAIN_SHP" \
  --val_shp "$VAL_SHP" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_dir "$PHASE1_5_OUT" \
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
  --lr_encoder 1e-7 \
  --lr_head 1e-4 \
  --weight_decay 5e-4 \
  --num_workers 0 \
  --device cpu \
  --no_amp

echo "[DONE] Phase 1.5"

# ------------------------------------------------------------
# Optional quality check for binary classifier head
# ------------------------------------------------------------

echo "============================================================"
echo "CHECK: classifier head validation performance"
echo "============================================================"

python eval_classifier_head.py \
  --encoder_ckpt "$PHASE1_5_OUT/phase1_encoder_best.pth" \
  --head_ckpt "$PHASE1_5_OUT/classifier_head_best.pth" \
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

python tune_binary_threshold.py \
  --pred_csv "$EVAL_OUT/classifier_predictions.csv" \
  --positive_label 1 \
  --output_csv "$EVAL_OUT/threshold_tuning.csv"

echo "[DONE] classifier eval + threshold tuning"

# ------------------------------------------------------------
# Phase 2A: extract embeddings using the Phase 1.5 encoder
# ------------------------------------------------------------

echo "============================================================"
echo "PHASE 2A: extract GT embeddings (binary, train split)"
echo "============================================================"

python extract_gt_embeddings.py \
  --encoder_ckpt "$PHASE1_5_OUT/phase1_encoder_best.pth" \
  --gt_path "$TRAIN_SHP" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_csv "$PHASE1_5_OUT/phase1_embeddings.csv" \
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
  --device cpu \
  --no_amp

echo "[DONE] embedding extraction"

# ------------------------------------------------------------
# Phase 2B: build standard binary prototypes
# ------------------------------------------------------------

echo "============================================================"
echo "PHASE 2B: build standard binary prototypes"
echo "============================================================"

python build_prototypes.py \
  --phase1_ckpt "$PHASE1_5_OUT/phase1_encoder_best.pth" \
  --phase1_embedding_csv "$PHASE1_5_OUT/phase1_embeddings.csv" \
  --gt_path "$TRAIN_SHP" \
  --gt_type shp \
  --gt_label_field BinaryTree \
  --gt_folder_field Folder \
  --gt_file_field File \
  --gt_fx_field fx \
  --gt_fy_field fy \
  --imagery_root "$IMAGERY_ROOT" \
  --output_dir "$PHASE2_OUT" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 16 \
  --num_workers 0 \
  --device cpu \
  --similarity cosine \
  --no_amp

echo "[DONE] standard prototypes"

# ------------------------------------------------------------
# Phase 2C: optional multi-prototypes for binary experiment
# ------------------------------------------------------------

echo "============================================================"
echo "PHASE 2C: build multi-prototypes (optional binary ablation)"
echo "============================================================"

python build_multi_prototypes.py \
  --embedding_csv "$PHASE1_5_OUT/phase1_embeddings.csv" \
  --output_csv "$PHASE2_MULTI_OUT/multi_class_prototypes.csv" \
  --label_col label \
  --positive_label 1 \
  --k_other 5 \
  --k_positive 1

echo "[DONE] multi-prototypes"

# ------------------------------------------------------------
# Final quick summary
# ------------------------------------------------------------

echo "============================================================"
echo "PIPELINE COMPLETED"
echo "============================================================"
echo "Phase 1 output        : $PHASE1_OUT"
echo "Phase 1.5 output      : $PHASE1_5_OUT"
echo "Phase 2 output        : $PHASE2_OUT"
echo "Phase 2 multi output  : $PHASE2_MULTI_OUT"
echo "Classifier eval output: $EVAL_OUT"
echo "Finished at           : $(date)"
echo "============================================================"