#!/bin/bash

#SBATCH --job-name=classifier_phase3
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/phase_3/cls_%j.out
#SBATCH --error=logs/phase_3/cls_%j.err

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "=== Classifier Phase3 ==="

python eval_classifier_head.py \
  --encoder_ckpt "./outputs/binary_shihuahuaco_classweights_check/phase1_encoder_best.pth" \
  --head_ckpt "./outputs/binary_shihuahuaco_classweights_check/classifier_head_best.pth" \
  --gt_path "./outputs/splits_binary/valid_points_val.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_classifier_binary" \
  --label_field BinaryTree \
  --device cpu

python tune_binary_threshold.py \
  --pred_csv "./outputs/phase2_classifier_binary/classifier_predictions.csv" \
  --positive_label 1 \
  --output_csv "./outputs/phase2_classifier_binary/threshold_tuning.csv"

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
  --decision_threshold 0.42 \
  --search_radius_px 128 \
  --coarse_step_px 16 \
  --refine_radius_px 32 \
  --refine_step_px 8 \
  --beta 0.002 \
  --batch_size 32 \
  --device cuda

echo "Done"