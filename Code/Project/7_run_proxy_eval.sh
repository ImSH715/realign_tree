#!/bin/bash#!/bin/bash

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

python evaluate_pipeline.py \
  --encoder_ckpt "./outputs/phase1/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2/class_prototypes.csv" \
  --input_csv "./outputs/evaluation/Censo_Forestal_overlap_all.csv" \
  --output_csv "./outputs/evaluation/censo_refined_run1.csv" \
  --point_id_column "point_id" \
  --label_column "label" \
  --x_column "original_east" \
  --y_column "original_north" \
  --image_column "matched_tif" \
  --search_radius_px 192 \
  --coarse_step_px 8 \
  --refine_radius_px 48 \
  --refine_step_px 4 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.0002 \
  --batch_size 32 \
  --device cuda

python eval_proxy_no_gt.py \
  --input_csv "./outputs/evaluation/censo_refined_run1.csv" \
  --single_output_csv "./outputs/evaluation/censo_refined_run1_proxy.csv" \
  --stability_output_csv "./outputs/evaluation/censo_refined_run1_stability.csv"

echo "Job finished at $(date)"