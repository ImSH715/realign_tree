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

python run_pipeline.py \
  --encoder_ckpt "./outputs/phase1/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2/class_prototypes.csv" \
  --points_csv "./outputs/evaluation/valid_points_recovery_20m.csv" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_csv "./outputs/evaluation/refined_recovery_5m.csv" \
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