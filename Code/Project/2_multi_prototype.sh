#!/bin/bash

#SBATCH --job-name=multi_prototype
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/multi_prototype_%j.out
#SBATCH --error=logs/multi_prototype_%j.err
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

python eval_classifier_head.py \
  --encoder_ckpt "./outputs/binary_shihuahuaco_classweights_check/phase1_encoder_best.pth" \
  --head_ckpt "./outputs/binary_shihuahuaco_classweights_check/classifier_head_best.pth" \
  --gt_path "./outputs/splits_binary/valid_points_val.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/eval_binary_classifier_val" \
  --label_field BinaryTree \
  --device cpu

python tune_binary_threshold.py \
  --pred_csv "./outputs/eval_binary_classifier_val/classifier_predictions.csv" \
  --positive_label 1 \
  --output_csv "./outputs/eval_binary_classifier_val/threshold_tuning.csv"


python build_multi_prototypes.py \
  --embedding_csv "./outputs/binary_shihuahuaco_classweights_check/phase1_embeddings.csv" \
  --output_csv "./outputs/phase2_binary_shihuahuaco/multi_class_prototypes.csv" \
  --label_col label \
  --positive_label 1 \
  --k_other 5 \
  --k_positive 1
  
echo "Job finished at $(date)"