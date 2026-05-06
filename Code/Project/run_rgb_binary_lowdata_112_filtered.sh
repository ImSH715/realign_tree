#!/bin/bash

#SBATCH --job-name=rgb112_lowdata
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/rgb112_lowdata_%j.out
#SBATCH --error=logs/rgb112_lowdata_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(dirname "$0")"
fi

mkdir -p logs
mkdir -p outputs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa_gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi
echo "============================================================"

IMAGERY_ROOT="/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"

INIT_CKPT="./outputs/phase1_resnet50_cpu/phase1_encoder_best.pth"
TRAIN_SHP="./outputs/splits_binary/valid_points_train.shp"
VAL_SHP="./outputs/splits_binary/valid_points_val.shp"

TRAIN_OUT="./outputs/binary_resnet50_rgb_lowdata_112_filtered_cuda"
EVAL_OUT="./outputs/phase2_classifier_binary_rgb_lowdata_112_filtered_cuda"

echo "Checking required inputs..."
test -f "$INIT_CKPT" || { echo "Missing init checkpoint: $INIT_CKPT"; exit 1; }
test -f "$TRAIN_SHP" || { echo "Missing train shapefile: $TRAIN_SHP"; exit 1; }
test -f "$VAL_SHP" || { echo "Missing validation shapefile: $VAL_SHP"; exit 1; }
test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }
echo "All required inputs found."

echo "============================================================"
echo "Step 1: Low-data RGB 112px training"
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
  --image_mode rgb \
  --image_size 224 \
  --patch_size_px 112 \
  --batch_size 8 \
  --epochs 30 \
  --lr_encoder 3e-6 \
  --lr_head 1e-4 \
  --weight_decay 5e-4 \
  --freeze_encoder_epochs 5 \
  --patience 0 \
  --save_every 0 \
  --balanced_sampler \
  --train_repeat_factor 4 \
  --label_smoothing 0.05 \
  --max_black_fraction 0.20 \
  --max_bright_fraction 0.35 \
  --debug_patches 80 \
  --print_val_dist \
  --num_workers 4 \
  --device cuda

echo "============================================================"
echo "Step 2: Evaluating low-data RGB classifier"
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
  --image_mode rgb \
  --image_size 224 \
  --patch_size_px 112 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda

echo "============================================================"
echo "Step 3: Tuning binary threshold"
echo "============================================================"

python tune_binary_threshold.py \
  --pred_csv "$EVAL_OUT/classifier_predictions.csv" \
  --positive_label 1 \
  --output_csv "$EVAL_OUT/threshold_tuning.csv"

echo "============================================================"
echo "Job finished at: $(date)"
echo "Training output   : $TRAIN_OUT"
echo "Evaluation output : $EVAL_OUT"
echo "Diagnostics       : $EVAL_OUT/binary_score_diagnostics.json"
echo "============================================================"
