#!/bin/bash
#SBATCH --job-name=phase3_cls_20m
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=cls_%j.out
#SBATCH --error=cls_%j.err

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "=== Classifier Phase3 ==="

python run_pipeline_classifier.py \
  --encoder_ckpt "./outputs/binary_shihuahuaco_classweights_check/phase1_encoder_best.pth" \
  --head_ckpt "./outputs/binary_shihuahuaco_classweights_check/classifier_head_best.pth" \
  --points_csv "./outputs/evaluation/valid_points_recovery_20m.csv" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_csv "./outputs/evaluation/refined_classifier_20m.csv" \
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

echo "Done"