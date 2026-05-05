#!/bin/bash

#SBATCH --job-name=phase3_recovery20
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=phase3_recovery20_%j.out
#SBATCH --error=phase3_recovery20_%j.err
#SBATCH --mail-type=END,FAIL

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

python run_pipeline.py \
  --encoder_ckpt "./outputs/phase1/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2/class_prototypes.csv" \
  --points_csv "./outputs/evaluation/valid_points_recovery_20m.csv" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_csv "./outputs/evaluation/refined_valid_points_recovery_20m.csv" \
  --tile_column "matched_tif" \
  --point_id_column "point_id" \
  --x_column "original_east" \
  --y_column "original_north" \
  --target_label_column "label" \
  --coord_type world \
  --search_radius_px 128 \
  --coarse_step_px 16 \
  --refine_radius_px 32 \
  --refine_step_px 8 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.002 \
  --batch_size 32 \
  --device cuda

: << 'COMMENT'
python run_pipeline.py \
  --points_csv "./outputs/phase2/corrected_labels.csv" \
  --tile_column "image_path" \
  --x_column "x" \
  --y_column "y" \
  --target_label_column "corrected_label" \
  --coord_type pixel
COMMENT
echo "Job finished at $(date)"