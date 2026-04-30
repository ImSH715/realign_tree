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
  --encoder_ckpt "./outputs/binary_shihuahuaco_classweights_check/phase1_encoder_best.pth" \
  --train_shp "./outputs/splits_binary/valid_points_train.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase2_binary_lejepa" \
  --label_field Tree \
  --folder_field Folder \
  --file_field File \
  --fx_field fx \
  --fy_field fy \
  --coord_mode auto \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 16 \
  --num_workers 0 \
  --device cpu

echo "Job finished at $(date)"