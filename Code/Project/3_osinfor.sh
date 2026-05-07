#!/bin/bash

# --- 1. Slurm Resource Configuration ---
#SBATCH --job-name=cpu_osinfor
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/osinfor/cpu_phase3_1200px_censo_%j.out
#SBATCH --error=logs/osinfor/cpu_phase3_1200px_censo_%j.err

# --- 2. Email Notification Settings ---
#SBATCH --mail-type=END,FAIL

# --- 3. Environment Setup ---
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

python run_pipeline.py \
  --encoder_ckpt "./outputs/phase1_5_lejepa_cpu_binary_preprocess/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2_binary_shihuahuaco/class_prototypes_named.csv" \
  --points_csv "Project/data/Censo_Forestal_shihuahuaco_phase3.csv" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Turing_L/datasets/Osinfor/Ortomosaicos" \
  --output_csv "./outputs/phase3/censo_shihuahuaco_refined_cpu.csv" \
  --tile_column "matched_tif" \
  --point_id_column "point_id" \
  --x_column "original_east" \
  --y_column "original_north" \
  --target_label_column "label" \
  --coord_type world \
  --search_radius_px 900 \
  --coarse_step_px 48 \
  --refine_radius_px 128 \
  --refine_step_px 16 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.0002 \
  --batch_size 32 \
  --device cpu

echo "Job finished at $(date)"