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

python train_encoder.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1_lejepa_cpu" \
  --backbone_name "vit_base_patch16_224" \
  --ssl_epochs 20 \
  --batch_size_ssl 8 \
  --ssl_lr 5e-4 \
  --weight_decay 5e-2 \
  --warmup_epochs_ssl 3 \
  --min_lr_ratio 1e-3 \
  --patch_size_px 224 \
  --patches_per_image 10 \
  --num_global_views 2 \
  --num_local_views 4 \
  --image_size_global 224 \
  --image_size_local 224 \
  --projector_hidden_dim 2048 \
  --projector_out_dim 512 \
  --align_weight 1.0 \
  --var_weight 25.0 \
  --cov_weight 1.0 \
  --slice_weight 1.0 \
  --num_slices 256 \
  --extract_stride_px 1024 \
  --extract_batch_size 16 \
  --max_extract_patches_per_image 20 \
  --num_workers 8 \
  --tile_cache_size 0 \
  --save_every 1 \
  --device cpu \
  --no_amp

echo "Job finished at $(date)"