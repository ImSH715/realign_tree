#!/bin/bash

#SBATCH --job-name=gpu_run_multi_prototype_pipeline
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/gpu_run_multi_prototype_pipeline_%j.out
#SBATCH --error=logs/gpu_run_multi_prototype_pipeline_%j.err
#SBATCH --mail-type=END,FAIL

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"


python run_pipeline_classifier.py \
  --encoder_ckpt "./outputs/binary_shihuahuaco_classweights_check/phase1_encoder_best.pth" \
  --head_ckpt "./outputs/binary_shihuahuaco_classweights_check/classifier_head_best.pth" \
  --points_csv "./outputs/evaluation/valid_points_recovery_20m.csv" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_csv "./outputs/evaluation/refined_classifier_binary_20m.csv" \
  --tile_column "matched_tif" \
  --point_id_column "point_id" \
  --x_column "original_east" \
  --y_column "original_north" \
  --target_label_column "label" \
  --coord_type world \
  --binary_positive_name "Shihuahuaco" \
  --search_radius_px 128 \
  --coarse_step_px 16 \
  --refine_radius_px 32 \
  --refine_step_px 8 \
  --beta 0.002 \
  --batch_size 32 \
  --device cuda