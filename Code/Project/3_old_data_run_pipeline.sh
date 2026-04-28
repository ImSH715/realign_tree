#!/bin/bash

#SBATCH --job-name=0_a_dino_ft
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/0_a_dino_ft_%j.out
#SBATCH --error=logs/0_a_dino_ft_%j.err
#SBATCH --mail-type=END,FAIL

mkdir -p logs

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
  --output_csv "./outputs/phase3/refined_all_overlap_strict_shihuahuaco_20m.csv" \
  --label_column "NOMBRE_COMUN" \
  --x_column "COORDENADA_ESTE" \
  --y_column "COORDENADA_NORTE" \
  --image_column "matched_tif" \
  --search_radius_px 560 \
  --coarse_step_px 32 \
  --refine_radius_px 96 \
  --refine_step_px 8 \
  --similarity cosine \
  --filter_label "Shihuahuaco"\
  --alpha 1.0 \
  --beta 0.0002 \
  --batch_size 32 \
  --device cuda

echo "Job finished at $(date)"