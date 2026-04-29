#!/bin/bash

#SBATCH --job-name=resnet_train_cpu
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/cpu_resnet_phase1_%j.out
#SBATCH --error=logs/cpu_resnet_phase1_%j.err
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

python train_encoder.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_resnet50_cpu" \
  --backbone_name "resnet50" \
  --pretrained_backbone \
  --ssl_epochs 100 \
  --batch_size_ssl 8 \
  --patches_per_image 10 \
  --num_workers 8 \
  --device cpu \
  --no_amp \
  --extract_stride_px 1024 \
  --extract_batch_size 16 \
  --image_size_global 224 \
  --image_size_local 224 \
  --max_extract_patches_per_image 20

echo "Job finished at $(date)"