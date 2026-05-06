#!/bin/bash
#SBATCH --job-name=resnet50_phase1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/resnet50_phase1_%j.out
#SBATCH --error=logs/resnet50_phase1_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa_gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python train_encoder.py \
  --train_root "/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_resnet50_cpu" \
  --backbone_name "resnet50" \
  --pretrained_backbone \
  --ssl_epochs 20 \
  --batch_size_ssl 16 \
  --patches_per_image 10 \
  --num_workers 4 \
  --device cuda \
  --extract_stride_px 1024 \
  --extract_batch_size 32 \
  --image_size_global 224 \
  --image_size_local 224 \
  --max_extract_patches_per_image 20
