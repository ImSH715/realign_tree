#!/bin/bash

# --- 1. Slurm Resource Configuration ---
#SBATCH --job-name=lejepa_train
#SBATCH --partition=gpu-h100-nvl       # Partition: gpu
#SBATCH --qos=gpu                      # QOS: gpu
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=96G                      # Request 96GB RAM
#SBATCH --cpus-per-task=8              # 8 CPU cores for data loading
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks=1                     # Single task
#SBATCH --time=90:00:00                # Time limit (HH:MM:SS)
#SBATCH --output=logs/train_lejepa_%j.out         # Standard output log
#SBATCH --error=logs/train_lejepa_%j.err          # Error log

# --- 2. Email Notification Settings ---
#SBATCH --mail-type=END,FAIL           # Notify when finished or failed

# --- 3. Environment Setup (Stanage Optimized) ---
module load Anaconda3

eval "$(conda shell.bash hook)"

conda activate lejepa

echo "Using Python from: $(which python)"
python --version

conda activate lejepa

python train_encoder.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_lejepa" \
  --backbone_name "vit_base_patch16_224" \
  --pretrained_backbone \
  --ssl_epochs 100 \
  --batch_size_ssl 8 \
  --patches_per_image 10 \
  --num_workers 4 \
  --device cuda \
  --extract_stride_px 1024 \
  --extract_batch_size 16 \
  --image_size_global 224 \
  --image_size_local 224 \
  --max_extract_patches_per_image 20

# Lejepa
: << 'COMMENT'
python train_encoder.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_lejepa" \
  --backbone_name "vit_base_patch16_224" \
  --ssl_epochs 20 \
  --batch_size_ssl 8 \
  --patches_per_image 10 \
  --num_workers 4 \
  --device cuda \
  --extract_stride_px 1024 \
  --extract_batch_size 16 \
  --image_size_global 224 \
  --image_size_local 224 \
  --max_extract_patches_per_image 20
COMMENT

# Resnet
: << 'COMMENT'
python train_encoder.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_resnet50" \
  --backbone_name "resnet50" \
  --pretrained_backbone \
  --ssl_epochs 20 \
  --batch_size_ssl 16 \
  --patches_per_image 10 \
  --num_workers 4 \
  --device cuda \
  --extract_stride_px 1024 \
  --extract_batch_size 32 \
  --image_size_global 224 \
  --image_size_local 224 \
  --max_extract_patches_per_image 20
COMMENT

#DINO
: << 'COMMENT'
python train_encoder.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_dino" \
  --backbone_name "vit_small_patch14_dinov2.lvd142m" \
  --pretrained_backbone \
  --ssl_epochs 20 \
  --batch_size_ssl 8 \
  --patches_per_image 10 \
  --num_workers 4 \
  --device cuda \
  --extract_stride_px 1024 \
  --extract_batch_size 16 \
  --image_size_global 224 \
  --image_size_local 224 \
  --max_extract_patches_per_image 20
COMMENT


echo "Job finished at $(date)"