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

python build_eval_inputs.py direct_gt_from_shp \
  --shp_path "/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_csv "./outputs/evaluation/valid_points_direct.csv" \
  --label_field "Tree" \
  --point_id_field "Tree" \
  --tolerance_m 10.0 \
  --target_crs "EPSG:32718"

python build_eval_inputs.py recovery_from_direct_gt \
  --input_csv "./outputs/evaluation/valid_points_direct.csv" \
  --output_csv "./outputs/evaluation/valid_points_recovery_5m.csv" \
  --offset_m 5 \
  --seed 42

python build_eval_inputs.py recovery_from_direct_gt \
  --input_csv "./outputs/evaluation/valid_points_direct.csv" \
  --output_csv "./outputs/evaluation/valid_points_recovery_20m.csv" \
  --offset_m 20 \
  --seed 42

python build_eval_inputs.py censo_overlap \
  --input_csv "/mnt/parscratch/users/acb20si/realign_tree/Code/Project/data/Censo_Forestal.csv" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Turing_L/datasets/Osinfor/Ortomosaicos" \
  --output_csv "./outputs/evaluation/Censo_Forestal_overlap_all.csv" \
  --x_column "COORDENADA_ESTE" \
  --y_column "COORDENADA_NORTE" \
  --label_column "NOMBRE_COMUN" \

echo "Job finished at $(date)"