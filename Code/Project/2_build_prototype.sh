#!/bin/bash

# --- 1. Slurm Resource Configuration ---
#SBATCH --job-name=LeJEPA_train
#SBATCH --partition=gpu                # Partition: gpu
#SBATCH --qos=gpu                      # QOS: gpu
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=82G                      # Request 82GB RAM
#SBATCH --cpus-per-task=8              # 8 CPU cores for data loading
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks=1                     # Single task
#SBATCH --time=90:00:00                # Time limit (HH:MM:SS)
#SBATCH --output=result_%j.out         # Standard output log
#SBATCH --error=result_%j.err          # Error log

# --- 2. Email Notification Settings ---
#SBATCH --mail-type=END,FAIL           # Notify when finished or failed

# --- 3. Environment Setup (Stanage Optimized) ---
module load Anaconda3

eval "$(conda shell.bash hook)"

conda activate lejepa

echo "Using Python from: $(which python)"
python --version

conda activate lejepa

python build_prototypes.py \
  --phase1_ckpt "./outputs/phase1_dino/phase1_encoder_best.pth" \
  --phase1_embedding_csv "./outputs/phase1_dino/phase1_embeddings.csv" \
  --gt_path "/mnt/parscratch/users/acb20si/realign_tree/Code/Project/data/valid_points.shp" \
  --gt_type shp \
  --gt_label_field "Tree" \
  --gt_folder_field "Folder" \
  --gt_file_field "File" \
  --gt_fx_field "fx" \
  --gt_fy_field "fy" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_dino" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --similarity cosine
  
#LEJEPA
: << 'COMMENT'
python build_prototypes.py \
  --phase1_ckpt "./outputs/phase1_lejepa/phase1_encoder_best.pth" \
  --phase1_embedding_csv "./outputs/phase1_lejepa/phase1_embeddings.csv" \
  --gt_path "/mnt/parscratch/users/acb20si/realign_tree/Code/Project/data/valid_points.shp" \
  --gt_type shp \
  --gt_label_field "Tree" \
  --gt_folder_field "Folder" \
  --gt_file_field "File" \
  --gt_fx_field "fx" \
  --gt_fy_field "fy" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_lejepa" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --similarity cosine
COMMENT

#RESNET50
: << 'COMMENT'
python build_prototypes.py \
  --phase1_ckpt "./outputs/phase1_resnet50/phase1_encoder_best.pth" \
  --phase1_embedding_csv "./outputs/phase1_resnet50/phase1_embeddings.csv" \
  --gt_path "/mnt/parscratch/users/acb20si/realign_tree/Code/Project/data/valid_points.shp" \
  --gt_type shp \
  --gt_label_field "Tree" \
  --gt_folder_field "Folder" \
  --gt_file_field "File" \
  --gt_fx_field "fx" \
  --gt_fy_field "fy" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_resnet50" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --similarity cosine
COMMENT

#DINO
: << 'COMMENT'
python build_prototypes.py \
  --phase1_ckpt "./outputs/phase1_dino/phase1_encoder_best.pth" \
  --phase1_embedding_csv "./outputs/phase1_dino/phase1_embeddings.csv" \
  --gt_path "/mnt/parscratch/users/acb20si/realign_tree/Code/Project/data/valid_points.shp" \
  --gt_type shp \
  --gt_label_field "Tree" \
  --gt_folder_field "Folder" \
  --gt_file_field "File" \
  --gt_fx_field "fx" \
  --gt_fy_field "fy" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_dino" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --similarity cosine
COMMENT

echo "Job finished at $(date)"