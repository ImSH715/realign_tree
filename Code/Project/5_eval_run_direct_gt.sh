#!/bin/bash

#SBATCH --job-name=direct_gt_eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=direct_gt_%j.out
#SBATCH --error=direct_gt_%j.err
#SBATCH --mail-type=END,FAIL

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

# 1. Phase 3 refinement
python run_pipeline.py \
  --encoder_ckpt "./outputs/phase1/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2/class_prototypes.csv" \
  --points_csv "./outputs/evaluation/valid_points_direct.csv" \
  --imagery_root "/mnt/parscratch/users/acc21pf/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_csv "./outputs/evaluation/direct_gt_refined.csv" \
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

# 2. GT evaluation
python eval_direct_gt.py \
  --input_csv "./outputs/evaluation/direct_gt_refined.csv" \
  --output_csv "./outputs/evaluation/direct_gt_refined_evaluated.csv"

# 3. Feature-space analysis
python analyze_feature_space.py \
  --encoder_ckpt "./outputs/phase1/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2/class_prototypes.csv" \
  --input_csv "./outputs/evaluation/direct_gt_refined.csv" \
  --output_dir "./outputs/analysis/direct_gt" \
  --label_column "label" \
  --image_column "image_path" \
  --mode refined \
  --dbscan_eps 0.20 \
  --dbscan_min_samples 5 \
  --batch_size 32 \
  --device cuda

echo "Job finished at $(date)"
