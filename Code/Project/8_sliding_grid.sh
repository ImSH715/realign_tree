#!/bin/bash

#SBATCH --job-name=phase4_slide_mil
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/mil/phase4_sliding_grid/phase4_slide_mil_%j.out
#SBATCH --error=logs/mil/phase4_sliding_grid/phase4_slide_mil_%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail
export GEOTIFF_CSV=""

cd /mnt/parscratch/users/acb20si/realign_tree/Code/Project

mkdir -p logs
mkdir -p logs/mil
mkdir -p logs/mil/phase4_sliding_grid
mkdir -p outputs/evaluation

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

python run_slide_grid_classifier.py \
  --input_csv "./outputs/evaluation/mil_phase4_input.csv" \
  --output_shp "./outputs/evaluation/mil_phase4_slidegrid.shp" \
  --encoder_ckpt "./outputs/binary_dino/phase1_encoder_best.pth" \
  --head_ckpt "./outputs/binary_dino/classifier_head_best.pth" \
  --tile_column "matched_tif" \
  --label_column "label" \
  --target_label "Shihuahuaco" \
  --x_column "original_east" \
  --y_column "original_north" \
  --crs "EPSG:32718" \
  --grid_sizes "20,10,5" \
  --threshold 0.40 \
  --min_realigned_boxes 3 \
  --final_refine_radius_m 3 \
  --final_refine_step_m 0.5 \
  --max_iterations 6 \
  --positive_class 1 \
  --patch_size_px 224 \
  --image_size 224 \
  --device cpu \
  --path_rewrite_from "/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --path_rewrite_to "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"

echo "Job finished at $(date)"