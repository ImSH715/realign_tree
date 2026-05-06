#!/bin/bash

#SBATCH --job-name=ppv_2_phase1.5
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/preprocessing/phase_1_5_2/phase_1_5_ppv2_%j.out
#SBATCH --error=logs/preprocessing/phase_1_5_2/phase_1_5_ppv2_%j.err
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

IMAGERY_ROOT="/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"

TRAIN_SHP="./outputs/splits_binary/valid_points_train.shp"
VAL_SHP="./outputs/splits_binary/valid_points_val.shp"

PHASE1_OUT="./outputs/phase1_lejepa_cpu_binary_preprocess_new"
PHASE1_5_OUT="./outputs/phase1_5_lejepa_cpu_binary_preprocess_new"
PHASE2_OUT="./outputs/phase2_binary_shihuahuaco_preprocess_new"
PHASE2_MULTI_OUT="./outputs/phase2_binary_shihuahuaco_multi_preprocess_new"
EVAL_OUT="./outputs/eval_binary_classifier_val_preprocess_new"

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

echo "Checking required inputs..."
test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }
test -f "$TRAIN_SHP" || { echo "Missing train shp: $TRAIN_SHP"; exit 1; }
test -f "$VAL_SHP" || { echo "Missing val shp: $VAL_SHP"; exit 1; }
echo "All required inputs found."

# ------------------------------------------------------------
# Phase 1
# ------------------------------------------------------------

# ------------------------------------------------------------
# Phase 1.5
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
  --epochs 20 \
  --lr_encoder 1e-7 \
  --lr_head 1e-4 \
  --weight_decay 5e-4 \
  --num_workers 0 \
  --device cpu \
  --no_amp

echo "[DONE] Phase 1.5"

# ------------------------------------------------------------
# Optional classifier validation
# ------------------------------------------------------------

echo "============================================================"
echo "CHECK: classifier head validation"
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

echo "[DONE] classifier validation"

# ------------------------------------------------------------
# Phase 2A: extract embeddings
# ------------------------------------------------------------

echo "============================================================"
echo "PHASE 2A: extract GT embeddings"
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
# Phase 2B: standard prototypes
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
# Phase 2C: optional multi-prototypes
# ------------------------------------------------------------

echo "============================================================"
echo "PHASE 2C: build multi-prototypes"
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
# Final summary
# ------------------------------------------------------------

echo "============================================================"
echo "PHASE 1 + 1.5 + 2 COMPLETED"
echo "============================================================"
echo "Phase 1 output        : $PHASE1_OUT"
echo "Phase 1.5 output      : $PHASE1_5_OUT"
echo "Phase 2 output        : $PHASE2_OUT"
echo "Phase 2 multi output  : $PHASE2_MULTI_OUT"
echo "Classifier eval output: $EVAL_OUT"
echo "Finished at           : $(date)"
echo "============================================================"