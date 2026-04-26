#!/bin/bash

# --- 1. Slurm Resource Configuration ---
#SBATCH --job-name=phase3_old_censo
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=phase3_old_%j.out
#SBATCH --error=phase3_old_%j.err

# --- 2. Email Notification Settings ---
#SBATCH --mail-type=END,FAIL

# --- 3. Environment Setup ---
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

python old_data_run_pipeline.py \
  --encoder_ckpt "./outputs/phase1/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2/class_prototypes.csv" \
  --points_csv "/mnt/parscratch/users/acb20si/realign_tree/Code/Project/data/Censo_Forestal_overlap_all_strict.csv" \
  --output_csv "./outputs/phase3/refined_all_overlap_strict.csv" \
  --label_column "NOMBRE_COMUN" \
  --x_column "COORDENADA_ESTE" \
  --y_column "COORDENADA_NORTE" \
  --image_column "matched_tif" \
  --search_radius_px 192 \
  --coarse_step_px 8 \
  --refine_radius_px 48 \
  --refine_step_px 4 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.0002 \
  --batch_size 32 \
  --filter_label "shihuahuaco"\
  --device cuda

echo "Job finished at $(date)"