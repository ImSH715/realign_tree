#!/bin/bash

#SBATCH --job-name=0_b_dino
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/phase_2/0_b_dino_ft_%j.out
#SBATCH --error=logs/phase_2/0_b_dino_ft_%j.err
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

python build_prototypes.py \
  --phase1_ckpt "./outputs/phase1_dino_supervised/phase1_encoder_best.pth" \
  --phase1_embedding_csv "./outputs/phase1_dino/phase1_embeddings.csv" \
  --gt_path "./outputs/splits_gt/valid_points_train.shp" \
  --gt_type shp \
  --gt_label_field "Tree" \
  --gt_folder_field "Folder" \
  --gt_file_field "File" \
  --gt_fx_field "fx" \
  --gt_fy_field "fy" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_dino_supervised" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --similarity cosine
#LeJEPA
: << 'COMMENT'
python build_prototypes.py \
  --phase1_ckpt "./outputs/phase1_lejepa_supervised/phase1_encoder_best.pth" \
  --phase1_embedding_csv "./outputs/phase1_lejepa/phase1_embeddings.csv" \
  --gt_path "./outputs/splits_gt/valid_points_train.shp" \
  --gt_type shp \
  --gt_label_field "Tree" \
  --gt_folder_field "Folder" \
  --gt_file_field "File" \
  --gt_fx_field "fx" \
  --gt_fy_field "fy" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_lejepa_supervised" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --similarity cosine
COMMENT
# Resnet50
: << 'COMMENT'
python build_prototypes.py \
  --phase1_ckpt "./outputs/phase1_resnet50_supervised/phase1_encoder_best.pth" \
  --phase1_embedding_csv "./outputs/phase1_resnet50/phase1_embeddings.csv" \
  --gt_path "./outputs/splits_gt/valid_points_train.shp" \
  --gt_type shp \
  --gt_label_field "Tree" \
  --gt_folder_field "Folder" \
  --gt_file_field "File" \
  --gt_fx_field "fx" \
  --gt_fy_field "fy" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_resnet50_supervised" \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --similarity cosine
COMMENT

echo "Job finished at $(date)"