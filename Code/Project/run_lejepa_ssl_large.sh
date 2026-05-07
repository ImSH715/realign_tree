#!/bin/bash

#SBATCH --job-name=lejepa_ssl_large
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/mil/phase1/lejepa_ssl_large_%j.out
#SBATCH --error=logs/mil/phase1/lejepa_ssl_large_%j.err
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
conda activate lejepa

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python: $(which python)"
python --version
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "============================================================"

SCRATCH_OUT_ROOT="/mnt/parscratch/users/acb20si/realign_outputs"
PHASE1_OUT_ABS="$SCRATCH_OUT_ROOT/phase1_lejepa_ssl_large"
PHASE1_OUT_LINK="./outputs/phase1_lejepa_ssl_large"

mkdir -p "$SCRATCH_OUT_ROOT"
mkdir -p "$PHASE1_OUT_ABS"

if [ -e "$PHASE1_OUT_LINK" ] && [ ! -L "$PHASE1_OUT_LINK" ]; then
  echo "Refusing to overwrite non-symlink output path: $PHASE1_OUT_LINK"
  echo "Move it to scratch or remove it, then rerun."
  exit 1
fi
ln -sfn "$PHASE1_OUT_ABS" "$PHASE1_OUT_LINK"

TRAIN_ROOTS=(
  "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
)

OPTIONAL_ROOTS=(
  "/mnt/parscratch/users/acb20si/ai4eo/Shared/2025_Forge/OSINFOR_data/2023"
  "/mnt/parscratch/users/acb20si/ai4eo/Shared/2025_Turing_L/datasets/Osinfor/Ortomosaicos"
  "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/02. Non-curated"
  "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/non_curated"
  "/mnt/parscratch/users/acb20si/2025_Forge/non_curated"
)

for root in "${OPTIONAL_ROOTS[@]}"; do
  if [ -d "$root" ]; then
    TRAIN_ROOTS+=("$root")
  fi
done

echo "Training roots:"
printf '  %s\n' "${TRAIN_ROOTS[@]}"

echo "============================================================"
echo "Phase 1: LEJEPA SSL pretraining on expanded imagery"
echo "============================================================"

python train_encoder.py \
  --train_root "${TRAIN_ROOTS[@]}" \
  --output_dir "$PHASE1_OUT_LINK" \
  --backbone_name "vit_base_patch16_224" \
  --pretrained_backbone \
  --ssl_epochs 20 \
  --batch_size_ssl 8 \
  --ssl_lr 1e-5 \
  --weight_decay 5e-2 \
  --warmup_epochs_ssl 3 \
  --patch_size_px 224 \
  --patches_per_image 64 \
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
  --device cpu

echo "============================================================"
echo "LEJEPA SSL phase finished at: $(date)"
echo "Output dir: $PHASE1_OUT_LINK"
echo "============================================================"