#!/bin/bash

#SBATCH --job-name=cpu_0_a_lejepa
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/0a_cpu_lejepa_%j.out
#SBATCH --error=logs/0a_cpu_lejepa_%j.err
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

python make_gt_splits.py \
  --input_shp "./data/valid_points.shp" \
  --output_dir "./outputs/splits_gt" \
  --label_field "Tree" \
  --group_field "File" \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42

python train_supervised_encoder.py \
  --init_ckpt "./outputs/phase1_lejepa/phase1_encoder_best.pth" \
  --train_shp "./outputs/splits_gt/valid_points_train.shp" \
  --val_shp "./outputs/splits_gt/valid_points_val.shp" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_lejepa_supervised_cpu" \
  --label_field "Tree" \
  --folder_field "Folder" \
  --file_field "File" \
  --fx_field "fx" \
  --fy_field "fy" \
  --coord_mode auto \
  --image_size 224 \
  --patch_size_px 224 \
  --batch_size 8 \
  --epochs 30 \
  --lr_encoder 1e-6 \
  --lr_head 1e-4 \
  --weight_decay 1e-4 \
  --freeze_encoder_epochs 3 \
  --patience 8 \
  --debug_patches 64 \
  --num_workers 8 \
  --device cpu \
  --no_amp

echo "Job finished at $(date)"