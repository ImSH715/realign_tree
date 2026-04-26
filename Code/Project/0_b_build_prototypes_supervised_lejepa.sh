#!/bin/bash

#SBATCH --job-name=phase2_supervised_lejepa
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=phase2_sup_%j.out
#SBATCH --error=phase2_sup_%j.err
#SBATCH --mail-type=END,FAIL

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

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

echo "Job finished at $(date)"