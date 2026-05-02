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

python run_slide_grid_classifier.py \
  --input_csv "./outputs/evaluation/valid_points_direct.csv" \
  --output_shp "./outputs/evaluation/slide_grid_classifier_binary_th018.shp" \
  --encoder_ckpt "./outputs/binary_shihuahuaco_classweights_check/phase1_encoder_best.pth" \
  --head_ckpt "./outputs/binary_shihuahuaco_classweights_check/classifier_head_best.pth" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --tile_column "matched_tif" \
  --label_column "label" \
  --target_label "Shihuahuaco" \
  --x_column "gt_east" \
  --y_column "gt_north" \
  --grid_sizes "30,20,10" \
  --threshold 0.18 \
  --min_realigned_boxes 3 \
  --max_iterations 10 \
  --positive_class 1 \
  --device cpu \
  --no_amp

echo "Done"