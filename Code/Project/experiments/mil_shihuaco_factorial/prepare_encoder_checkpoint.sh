#!/bin/bash

#SBATCH --job-name=prep_encoder
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [ -n "${PROJECT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  SCRIPT_DIR_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SCRIPT_DIR_FALLBACK/../.." && pwd)"
fi
SCRIPT_DIR="${FACTORIAL_SCRIPT_DIR:-$PROJECT_DIR/experiments/mil_shihuaco_factorial}"
cd "$PROJECT_DIR"

ENCODER="${ENCODER:-lejepa}"
STAGE="${STAGE:-all}"
SCRATCH_OUT_ROOT="${SCRATCH_OUT_ROOT:-/mnt/parscratch/users/aca21jo/realign_outputs}"
SUBSET_ROOT="${SUBSET_ROOT:-$SCRATCH_OUT_ROOT/tif_subsets/shared_ortho_months_08_11}"
IMAGERY_ROOT="${IMAGERY_ROOT:-/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023}"
TRAIN_SHP="${TRAIN_SHP:-./outputs/splits_binary/valid_points_train.shp}"
VAL_SHP="${VAL_SHP:-./outputs/splits_binary/valid_points_val.shp}"

SSL_EPOCHS="${SSL_EPOCHS:-20}"
BINARY_EPOCHS="${BINARY_EPOCHS:-40}"
BINARY_FREEZE_EPOCHS="${BINARY_FREEZE_EPOCHS:-10}"
TRAIN_REPEAT_FACTOR="${TRAIN_REPEAT_FACTOR:-2}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
LR_ENCODER="${LR_ENCODER:-1e-6}"
LR_HEAD="${LR_HEAD:-1e-4}"
TRAIN_IMAGE_MODE="${TRAIN_IMAGE_MODE:-rgb}"
EVAL_IMAGE_MODE="${EVAL_IMAGE_MODE:-$TRAIN_IMAGE_MODE}"

mkdir -p "$SCRATCH_OUT_ROOT"
mkdir -p "$SCRATCH_OUT_ROOT/../realign_experiments/mil_shihuaco_factorial/slurm_logs" 2>/dev/null || true
mkdir -p ./outputs 2>/dev/null || true

normalise_name() {
  printf "%s" "$1" | tr "[:upper:]" "[:lower:]"
}

ensure_output_link() {
  local name="$1"
  local abs="$SCRATCH_OUT_ROOT/$name"
  local link="./outputs/$name"
  mkdir -p "$abs"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    echo "Refusing to overwrite non-symlink output path: $link"
    exit 1
  fi
  ln -sfn "$abs" "$link"
}

first_existing() {
  for candidate in "$@"; do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
      printf "%s\n" "$candidate"
      return 0
    fi
  done
  return 1
}

conda_env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -Fxq "$env_name"
}

pick_conda_env() {
  local explicit_env="$1"
  local primary_env="$2"
  local fallback_env="$3"

  if [ -n "$explicit_env" ]; then
    printf "%s\n" "$explicit_env"
  elif conda_env_exists "$primary_env"; then
    printf "%s\n" "$primary_env"
  elif conda_env_exists "$fallback_env"; then
    printf "%s\n" "$fallback_env"
  else
    printf "%s\n" "$primary_env"
  fi
}

build_seasonal_subset() {
  if [ -d "$SUBSET_ROOT/tifs" ]; then
    echo "Using existing seasonal subset: $SUBSET_ROOT/tifs"
    return
  fi

  CANDIDATE_ROOTS=(
    "/mnt/parscratch/users/aca21jo/ai4eo_shared/Shared/2025_Forge/OSINFOR_data/2023"
    "/mnt/parscratch/users/aca21jo/ai4eo_shared/Shared/2025_Turing_L/datasets/Osinfor/Ortomosaicos"
    "/shared/ai4eo/Shared/2025_Forge/OSINFOR_data/2023"
    "/shared/ai4eo/Shared/2025_Turing_L/datasets/Osinfor/Ortomosaicos"
    "$IMAGERY_ROOT"
  )

  SOURCE_ROOTS=()
  for root in "${CANDIDATE_ROOTS[@]}"; do
    if [ -d "$root" ]; then
      SOURCE_ROOTS+=("$root")
    fi
  done

  if [ "${#SOURCE_ROOTS[@]}" -eq 0 ]; then
    echo "No source roots found for seasonal TIFF subset."
    exit 1
  fi

  python make_tif_subset.py \
    --roots "${SOURCE_ROOTS[@]}" \
    --output_root "$SUBSET_ROOT" \
    --include_months "08,09,10,11"
}

setup_modules() {
  if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
      . /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
      . /usr/share/Modules/init/bash
    fi
  fi
  if command -v module >/dev/null 2>&1; then
    module load Anaconda3
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found."
    exit 1
  fi
  eval "$(conda shell.bash hook)"
}

run_ssl() {
  local backbone_name="$1"
  local phase1_name="$2"
  local pretrained_flag="$3"

  ensure_output_link "$phase1_name"
  build_seasonal_subset

  cmd=(
    python train_encoder.py
    --train_root "$SUBSET_ROOT/tifs"
    --output_dir "./outputs/$phase1_name"
    --backbone_name "$backbone_name"
    --ssl_epochs "$SSL_EPOCHS"
    --batch_size_ssl 8
    --ssl_lr 1e-5
    --weight_decay 5e-2
    --warmup_epochs_ssl 3
    --patch_size_px 224
    --patches_per_image 96
    --num_global_views 2
    --num_local_views 2
    --image_size_global 224
    --image_size_local 224
    --eval_batches 10
    --num_workers 4
    --tile_cache_size 16
    --save_every 0
    --debug_patches 32
    --skip_extract
    --cudnn_benchmark
    --device cuda
  )

  if [ "$pretrained_flag" = "1" ]; then
    cmd+=(--pretrained_backbone)
  fi

  "${cmd[@]}"
}

run_binary() {
  local phase1_name="$1"
  local binary_name="$2"

  ensure_output_link "$binary_name"

  local init_ckpt
  init_ckpt="$(first_existing \
    "./outputs/$phase1_name/phase1_encoder_best.pth" \
    "./outputs/$phase1_name/phase1_encoder_last.pth")" || {
      echo "Missing phase-1 checkpoint for binary adaptation: ./outputs/$phase1_name"
      exit 1
    }

  test -f "$TRAIN_SHP" || { echo "Missing train shapefile: $TRAIN_SHP"; exit 1; }
  test -f "$VAL_SHP" || { echo "Missing validation shapefile: $VAL_SHP"; exit 1; }
  test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }

  python train_supervised_encoder.py \
    --init_ckpt "$init_ckpt" \
    --train_shp "$TRAIN_SHP" \
    --val_shp "$VAL_SHP" \
    --imagery_root "$IMAGERY_ROOT" \
    --output_dir "./outputs/$binary_name" \
    --label_field BinaryTree \
    --folder_field Folder \
    --file_field File \
    --fx_field fx \
    --fy_field fy \
    --coord_mode auto \
    --image_mode "$TRAIN_IMAGE_MODE" \
    --image_size 224 \
    --patch_size_px 224 \
    --batch_size 8 \
    --epochs "$BINARY_EPOCHS" \
    --lr_encoder "$LR_ENCODER" \
    --lr_head "$LR_HEAD" \
    --weight_decay 5e-4 \
    --freeze_encoder_epochs "$BINARY_FREEZE_EPOCHS" \
    --patience 0 \
    --save_every 0 \
    --balanced_sampler \
    --train_repeat_factor "$TRAIN_REPEAT_FACTOR" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --max_black_fraction 0.20 \
    --max_bright_fraction 0.35 \
    --debug_patches 80 \
    --print_val_dist \
    --num_workers 4 \
    --device cuda
}

setup_modules

ENCODER="$(normalise_name "$ENCODER")"
case "$ENCODER" in
  lejepa|le-jepa)
    CONDA_ENV="$(pick_conda_env "${LEJEPA_CONDA_ENV:-}" "lejepa" "lejepa_gpu")"
    conda activate "$CONDA_ENV"
    PHASE1_NAME="${LEJEPA_PHASE1_NAME:-phase1_lejepa_ssl_large_gpu}"
    BINARY_NAME="${LEJEPA_BINARY_NAME:-binary_lejepa_shared_seasonal_rgb_balanced_cuda}"
    BACKBONE_NAME="${LEJEPA_BACKBONE_NAME:-vit_base_patch16_224}"
    PRETRAINED="${LEJEPA_PRETRAINED:-1}"
    ;;
  dino3|dinov3)
    CONDA_ENV="$(pick_conda_env "${DINO_CONDA_ENV:-}" "lejepa_gpu" "lejepa")"
    conda activate "$CONDA_ENV"
    PHASE1_NAME="${DINO3_PHASE1_NAME:-phase1_dino3_ssl_shared_seasonal}"
    BINARY_NAME="${DINO3_BINARY_NAME:-binary_dino3_shared_seasonal_rgb_balanced_cuda}"
    BACKBONE_NAME="${DINO3_BACKBONE_NAME:-}"
    PRETRAINED="${DINO3_PRETRAINED:-1}"
    if [ -z "$BACKBONE_NAME" ]; then
      echo "DINO3_BACKBONE_NAME is required."
      echo "List available candidates with:"
      echo "  python -c \"import timm; print('\\n'.join(timm.list_models('*dino*3*')))\""
      exit 1
    fi
    ;;
  *)
    echo "Unknown ENCODER=$ENCODER. Expected lejepa or dino3."
    exit 1
    ;;
esac

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================================"
echo "Preparing encoder checkpoint"
echo "Encoder       : $ENCODER"
echo "Stage         : $STAGE"
echo "Backbone      : $BACKBONE_NAME"
echo "Pretrained    : $PRETRAINED"
echo "Conda env     : $CONDA_ENV"
echo "Phase1 output : ./outputs/$PHASE1_NAME"
echo "Binary output : ./outputs/$BINARY_NAME"
echo "Python        : $(which python)"
python --version
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "============================================================"

case "$STAGE" in
  ssl)
    run_ssl "$BACKBONE_NAME" "$PHASE1_NAME" "$PRETRAINED"
    ;;
  binary)
    run_binary "$PHASE1_NAME" "$BINARY_NAME"
    ;;
  all)
    if [ -f "./outputs/$PHASE1_NAME/phase1_encoder_best.pth" ]; then
      echo "Skipping SSL; phase-1 checkpoint already exists."
    else
      run_ssl "$BACKBONE_NAME" "$PHASE1_NAME" "$PRETRAINED"
    fi
    run_binary "$PHASE1_NAME" "$BINARY_NAME"
    ;;
  *)
    echo "Unknown STAGE=$STAGE. Expected ssl, binary, or all."
    exit 1
    ;;
esac

echo "============================================================"
echo "Encoder preparation finished at: $(date)"
echo "MIL init checkpoint:"
echo "  ./outputs/$BINARY_NAME/phase1_encoder_best.pth"
echo "============================================================"
