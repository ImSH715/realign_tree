#!/bin/bash

#SBATCH --job-name=cpu_phase1_test
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpu_phase1_test_%j.out
#SBATCH --error=logs/cpu_phase1_test_%j.err
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

python train_encoder.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_cpu_test" \
  --backbone_name "resnet50" \
  --ssl_epochs 1 \
  --batch_size_ssl 1 \
  --patches_per_image 1 \
  --num_global_views 1 \
  --num_local_views 1 \
  --num_workers 2 \
  --device cpu \
  --no_amp \
  --extract_stride_px 4096 \
  --extract_batch_size 1 \
  --max_extract_patches_per_image 1 \
  --save_every 1

echo "Job finished at $(date)"