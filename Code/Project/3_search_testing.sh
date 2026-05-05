#!/bin/bash

#SBATCH --job-name=search_testing
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/testing/phase1/testing_search_%j.out
#SBATCH --error=logs/testing/phase1/testing_search_%j.err

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

IMAGERY_ROOT="/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
POINTS_CSV="./outputs/evaluation/valid_points_recovery_20m.csv"

ENCODER_CKPT="./outputs/phase1/phase1_encoder_best.pth"
PROTOTYPES_CSV="./outputs/phase2/class_prototypes.csv"

# -------------------------------
# EXP 1: beta = 0.0002
# -------------------------------
python run_pipeline.py \
  --encoder_ckpt "$ENCODER_CKPT" \
  --prototypes_csv "$PROTOTYPES_CSV" \
  --points_csv "$POINTS_CSV" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_csv "./outputs/evaluation/recovery20_largeSearch_beta0002_refined.csv" \
  --tile_column "matched_tif" \
  --point_id_column "point_id" \
  --x_column "original_east" \
  --y_column "original_north" \
  --target_label_column "label" \
  --coord_type world \
  --search_radius_px 560 \
  --coarse_step_px 32 \
  --refine_radius_px 96 \
  --refine_step_px 8 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.0002 \
  --batch_size 32 \
  --device cuda

python eval_direct_gt.py \
  --input_csv "./outputs/evaluation/recovery20_largeSearch_beta0002_refined.csv" \
  --output_csv "./outputs/evaluation/recovery20_largeSearch_beta0002_evaluated.csv"

# -------------------------------
# EXP 2: beta = 0.002
# -------------------------------
python run_pipeline.py \
  --encoder_ckpt "$ENCODER_CKPT" \
  --prototypes_csv "$PROTOTYPES_CSV" \
  --points_csv "$POINTS_CSV" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_csv "./outputs/evaluation/recovery20_largeSearch_beta0002b_refined.csv" \
  --tile_column "matched_tif" \
  --point_id_column "point_id" \
  --x_column "original_east" \
  --y_column "original_north" \
  --target_label_column "label" \
  --coord_type world \
  --search_radius_px 560 \
  --coarse_step_px 32 \
  --refine_radius_px 96 \
  --refine_step_px 8 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.002 \
  --batch_size 32 \
  --device cuda

python eval_direct_gt.py \
  --input_csv "./outputs/evaluation/recovery20_largeSearch_beta0002b_refined.csv" \
  --output_csv "./outputs/evaluation/recovery20_largeSearch_beta0002b_evaluated.csv"

echo "Done"