#!/bin/bash

#SBATCH --job-name=11_dino
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/phase_1_5/11_dino_%j.out
#SBATCH --error=logs/phase_1_5/11_dino_%j.err
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

python train_supervised_encoder.py \
  --init_ckpt "./outputs/phase1_dino_cpu/phase1_encoder_best.pth" \
  --train_shp "./outputs/splits_gt_multi_reduced40/valid_points_train.shp" \
  --val_shp "./outputs/splits_gt_multi_reduced40/valid_points_val.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_5_dino_11" \
  --label_field Tree \
  --folder_field Folder \
  --file_field File \
  --fx_field fx \
  --fy_field fy \
  --coord_mode auto \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 8 \
  --epochs 40 \
  --lr_encoder 1e-7 \
  --lr_head 1e-4 \
  --weight_decay 5e-4 \
  --patience 10 \
  --debug_patches 64 \
  --print_val_dist \
  --num_workers 0 \
  --device cpu \
  --no_amp
echo "Job finished at $(date)"