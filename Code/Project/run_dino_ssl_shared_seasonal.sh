#!/bin/bash

#SBATCH --job-name=dino_ssl_season
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/dino_ssl_season_%j.out
#SBATCH --error=logs/dino_ssl_season_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(dirname "$0")"
fi

mkdir -p logs
mkdir -p outputs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa_gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi
echo "============================================================"

SCRATCH_OUT_ROOT="/mnt/parscratch/users/aca21jo/realign_outputs"
SUBSET_ROOT="$SCRATCH_OUT_ROOT/tif_subsets/shared_ortho_months_08_11"
PHASE1_OUT_ABS="$SCRATCH_OUT_ROOT/phase1_dino_ssl_shared_seasonal"
PHASE1_OUT_LINK="./outputs/phase1_dino_ssl_shared_seasonal"

mkdir -p "$SCRATCH_OUT_ROOT"
mkdir -p "$PHASE1_OUT_ABS"

if [ -e "$PHASE1_OUT_LINK" ] && [ ! -L "$PHASE1_OUT_LINK" ]; then
  echo "Refusing to overwrite non-symlink output path: $PHASE1_OUT_LINK"
  echo "Move it to scratch or remove it, then rerun."
  exit 1
fi
ln -sfn "$PHASE1_OUT_ABS" "$PHASE1_OUT_LINK"

CANDIDATE_ROOTS=(
  "/mnt/parscratch/users/aca21jo/ai4eo_shared/Shared/2025_Forge/OSINFOR_data/2023"
  "/mnt/parscratch/users/aca21jo/ai4eo_shared/Shared/2025_Turing_L/datasets/Osinfor/Ortomosaicos"
  "/shared/ai4eo/Shared/2025_Forge/OSINFOR_data/2023"
  "/shared/ai4eo/Shared/2025_Turing_L/datasets/Osinfor/Ortomosaicos"
)

SOURCE_ROOTS=()
for root in "${CANDIDATE_ROOTS[@]}"; do
  if [ -d "$root" ]; then
    SOURCE_ROOTS+=("$root")
  fi
done

if [ "${#SOURCE_ROOTS[@]}" -eq 0 ]; then
  echo "No source roots found. Candidate roots checked:"
  printf '  %s\n' "${CANDIDATE_ROOTS[@]}"
  exit 1
fi

echo "Source roots:"
printf '  %s\n' "${SOURCE_ROOTS[@]}"

echo "============================================================"
echo "Building seasonal TIFF symlink subset"
echo "============================================================"

python make_tif_subset.py \
  --roots "${SOURCE_ROOTS[@]}" \
  --output_root "$SUBSET_ROOT" \
  --include_months "08,09,10,11"

echo "Subset TIFF count:"
find "$SUBSET_ROOT/tifs" -type l -o -type f | wc -l

echo "============================================================"
echo "Phase 1: DINOv2 SSL pretraining on seasonal shared orthomosaics"
echo "============================================================"

python train_encoder.py \
  --train_root "$SUBSET_ROOT/tifs" \
  --output_dir "$PHASE1_OUT_LINK" \
  --backbone_name "vit_small_patch14_dinov2.lvd142m" \
  --pretrained_backbone \
  --ssl_epochs 20 \
  --batch_size_ssl 8 \
  --ssl_lr 1e-5 \
  --weight_decay 5e-2 \
  --warmup_epochs_ssl 3 \
  --patch_size_px 224 \
  --patches_per_image 96 \
  --num_global_views 2 \
  --num_local_views 2 \
  --image_size_global 224 \
  --image_size_local 224 \
  --eval_batches 10 \
  --num_workers 4 \
  --tile_cache_size 16 \
  --save_every 0 \
  --debug_patches 32 \
  --skip_extract \
  --cudnn_benchmark \
  --device cuda

echo "============================================================"
echo "Job finished at: $(date)"
echo "DINO SSL output: $PHASE1_OUT_LINK -> $PHASE1_OUT_ABS"
echo "Seasonal subset : $SUBSET_ROOT"
echo "============================================================"
