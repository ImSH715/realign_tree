#!/bin/bash

#SBATCH --job-name=LeJEPA_eval_full
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=result_%j.out
#SBATCH --error=result_%j.err
#SBATCH --mail-type=END,FAIL

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python: $(which python)"
python --version
echo "Job started at $(date)"

INPUT_CSV="./outputs/phase3/refined_shihuahuaco_overlap_strict_world.csv"

python eval_proxy_no_gt.py \
  --input_csv "$INPUT_CSV" \
  --single_output_csv "./outputs/evaluation/refined_shihuahuaco_proxy.csv" \
  --stability_output_csv "./outputs/evaluation/refined_shihuahuaco_stability.csv"

python analyze_feature_space.py \
  --encoder_ckpt "./outputs/phase1/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2/class_prototypes.csv" \
  --input_csv "$INPUT_CSV" \
  --output_dir "./outputs/analysis/shihuahuaco_strict_world" \
  --label_column "target_label" \
  --image_column "image_path" \
  --mode refined \
  --dbscan_eps 0.20 \
  --dbscan_min_samples 5 \
  --batch_size 32 \
  --device cuda

echo "Job finished at $(date)"