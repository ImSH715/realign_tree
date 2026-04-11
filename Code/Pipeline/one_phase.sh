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

python one_phase_training.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./phase1_output" \
  --ssl_epochs 20 \
  --batch_size_ssl 8 \
  --patches_per_image 100 \
  --extract_stride_px 224 \
  --extract_batch_size 32 \
  --num_workers 0 \

echo "Job finished at $(date)"