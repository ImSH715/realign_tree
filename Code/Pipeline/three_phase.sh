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


python phase3_point_realign_dbscan.py \
  --phase1_ckpt ./phase1_output/phase1_final.pth \
  --prototypes_csv ./phase2_output/class_prototypes.csv \
  --corrected_labels_csv ./phase2_output/corrected_labels.csv \
  --target_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --points_csv ./points.csv \
  --output_dir ./phase3_output \
  --search_radius_px 128 \
  --grid_step_px 16 \
  --refinement_radius_px 32 \
  --refinement_step_px 8 \
  --dbscan_eps 48 \
  --dbscan_min_samples 2

echo "Job finished at $(date)"