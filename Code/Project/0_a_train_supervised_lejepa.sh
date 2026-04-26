#!/bin/bash

#SBATCH --job-name=lejepa_supervised
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=supervised_%j.out
#SBATCH --error=supervised_%j.err
#SBATCH --mail-type=END,FAIL

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

# 1. Split GT
python make_gt_splits.py \
  --input_shp "/mnt/parscratch/users/acb20si/realign_tree/Code/Project/data/valid_points.shp" \
  --output_dir "./outputs/splits_gt" \
  --label_field "Tree" \
  --group_field "File" \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42

# 2. Supervised fine-tune LeJEPA encoder
python train_supervised_encoder.py \
  --init_ckpt "./outputs/phase1_lejepa/phase1_encoder_best.pth" \
  --train_shp "./outputs/splits_gt/valid_points_train.shp" \
  --val_shp "./outputs/splits_gt/valid_points_val.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_lejepa_supervised" \
  --label_field "Tree" \
  --folder_field "Folder" \
  --file_field "File" \
  --fx_field "fx" \
  --fy_field "fy" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 32 \
  --epochs 30 \
  --lr_encoder 1e-5 \
  --lr_head 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --device cuda

echo "Job finished at $(date)"