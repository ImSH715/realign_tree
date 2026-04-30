#!/bin/bash

#SBATCH --job-name=lejepa_pro_binary
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/pro_binary_lejepa_%j.out
#SBATCH --error=logs/pro_binary_lejepa_%j.err
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

python build_prototypes.py \
  --phase1_ckpt "./outputs/binary_shihuahuaco_classweights_check/phase1_encoder_best.pth" \
  --phase1_embedding_csv "./outputs/binary_shihuahuaco_classweights_check/phase1_embeddings.csv" \
  --gt_path "./outputs/splits_binary/valid_points_train.shp" \
  --gt_label_field BinaryTree \
  --gt_folder_field Folder \
  --gt_file_field File \
  --gt_fx_field fx \
  --gt_fy_field fy \
  --output_dir "./outputs/phase2_binary_shihuahuaco"

echo "Job finished at $(date)"