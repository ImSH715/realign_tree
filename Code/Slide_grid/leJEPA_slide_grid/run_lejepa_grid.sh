#!/bin/bash

# --- 1. Slurm Resource Configuration (Matched to your srun command) ---
#SBATCH --job-name=LeJEPA_Grid_Scan
#SBATCH --partition=gpu                # Partition: gpu
#SBATCH --qos=gpu                      # QOS: gpu (Crucial!)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=82G                      # Request 82GB RAM
#SBATCH --cpus-per-task=8              # Increased to 8 for faster data loading
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks=1                     # Single task
#SBATCH --time=12:00:00                # Time limit (Adjust if needed)
#SBATCH --output=result_%j.out         # Log for standard output
#SBATCH --error=result_%j.err          # Log for errors

# --- 2. Email Notification Settings ---
#SBATCH --mail-user=seunghyunim2@gmail.com
#SBATCH --mail-type=END,FAIL

# --- 3. Environment Setup ---
# No need for --pty bash in sbatch; it runs in the background.
source activate lejepa

# --- 4. Execution ---
# Added -u to python for real-time unbuffered logging to the .out file
python -u leJEPA_slide_grid.py