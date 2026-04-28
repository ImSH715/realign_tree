#!/bin/bash

#SBATCH --job-name=Dino_short
#SBATCH --partition=gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/0_a_dino_ft_%j.out
#SBATCH --error=logs/0_a_dino_ft_%j.err
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi
# 1. Split GT
python make_gt_splits.py \
  --input_shp "./data/valid_points.shp" \
  --output_dir "./outputs/splits_gt" \
  --label_field "Tree" \
  --group_field "File" \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42

# 2. Supervised fine-tune LeJEPA encoder
python train_supervised_encoder.py \
  --init_ckpt "./outputs/phase1_dino/phase1_encoder_best.pth" \
  --train_shp "./outputs/splits_gt/valid_points_train.shp" \
  --val_shp "./outputs/splits_gt/valid_points_val.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_dino_supervised" \
  --label_field "Tree" \
  --folder_field "Folder" \
  --file_field "File" \
  --fx_field "fx" \
  --fy_field "fy" \
  --coord_mode auto \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 16 \
  --epochs 10 \
  --lr_encoder 1e-5 \
  --lr_head 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --device cuda

# Lejepa
: << 'COMMENT'
python train_supervised_encoder.py \
  --init_ckpt "./outputs/phase1_lejepa/phase1_encoder_best.pth" \
  --train_shp "./outputs/splits_gt/valid_points_train.shp" \
  --val_shp "./outputs/splits_gt/valid_points_val.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/2023" \
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
COMMENT

# Resnet50
: << 'COMMENT'
python train_supervised_encoder.py \
  --init_ckpt "./outputs/phase1_resnet50/phase1_encoder_best.pth" \
  --train_shp "./outputs/splits_gt/valid_points_train.shp" \
  --val_shp "./outputs/splits_gt/valid_points_val.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_resnet50_supervised" \
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
COMMENT

echo "Job finished at $(date)"