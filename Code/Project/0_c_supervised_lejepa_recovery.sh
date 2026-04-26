#!/bin/bash

#SBATCH --job-name=phase3_sup_lejepa
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=phase3_sup_%j.out
#SBATCH --error=phase3_sup_%j.err
#SBATCH --mail-type=END,FAIL

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

# Choose recovery input
INPUT_CSV="./outputs/evaluation/valid_points_recovery_20m.csv"
OUT_PREFIX="supervised_lejepa_recovery_20m"

python run_pipeline.py \
  --encoder_ckpt "./outputs/phase1_lejepa_supervised/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2_lejepa_supervised/class_prototypes.csv" \
  --points_csv "$INPUT_CSV" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_csv "./outputs/evaluation/${OUT_PREFIX}_refined.csv" \
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
  --input_csv "./outputs/evaluation/${OUT_PREFIX}_refined.csv" \
  --output_csv "./outputs/evaluation/${OUT_PREFIX}_evaluated.csv"

python analyze_feature_space.py \
  --encoder_ckpt "./outputs/phase1_lejepa_supervised/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2_lejepa_supervised/class_prototypes.csv" \
  --input_csv "./outputs/evaluation/${OUT_PREFIX}_refined.csv" \
  --output_dir "./outputs/analysis/${OUT_PREFIX}" \
  --label_column "label" \
  --image_column "image_path" \
  --mode refined \
  --dbscan_eps 0.20 \
  --dbscan_min_samples 5 \
  --batch_size 32 \
  --device cuda

echo "Job finished at $(date)"