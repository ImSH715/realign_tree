#!/bin/bash

#SBATCH --job-name=mil_fact
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/mil_factorial/%x_%j.out
#SBATCH --error=logs/mil_factorial/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs/mil_factorial
mkdir -p outputs

normalise_name() {
  printf "%s" "$1" | tr "[:upper:]" "[:lower:]"
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

EXPERIMENT_NAME="${EXPERIMENT_NAME:-mil_shihuaco_factorial}"
ENCODER="$(normalise_name "${ENCODER:-dino2}")"
TRAIN_IMAGE_MODE="${TRAIN_IMAGE_MODE:-rgb_white_boost}"
EVAL_IMAGE_MODE="${EVAL_IMAGE_MODE:-$TRAIN_IMAGE_MODE}"
PATCH_SIZE_PX="${PATCH_SIZE_PX:-224}"
POOLING="${POOLING:-lse}"
RUN_TAG="${RUN_TAG:-}"

BAG_RADIUS_M="${BAG_RADIUS_M:-20}"
NEGATIVE_BAG_RADIUS_M="${NEGATIVE_BAG_RADIUS_M:-0}"
BAG_INSTANCES="${BAG_INSTANCES:-25}"
BAG_LAYOUT="${BAG_LAYOUT:-grid}"
LSE_TAU="${LSE_TAU:-1.0}"
TOPK="${TOPK:-3}"
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-3}"
TRAIN_REPEAT_FACTOR="${TRAIN_REPEAT_FACTOR:-2}"
LR_ENCODER="${LR_ENCODER:-5e-7}"
LR_HEAD="${LR_HEAD:-1e-4}"
MONITOR_METRIC="${MONITOR_METRIC:-val_macro_f1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

IMAGERY_ROOT="${IMAGERY_ROOT:-/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023}"
TRAIN_SHP="${TRAIN_SHP:-./outputs/splits_binary/valid_points_train.shp}"
VAL_SHP="${VAL_SHP:-./outputs/splits_binary/valid_points_val.shp}"
SCRATCH_EXP_ROOT="${SCRATCH_EXP_ROOT:-/mnt/parscratch/users/aca21jo/realign_experiments/$EXPERIMENT_NAME}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-./experiments/mil_shihuaco_factorial/results}"

case "$ENCODER" in
  dino2|dinov2)
    ENCODER="dino2"
    CONDA_ENV="${CONDA_ENV:-${DINO_CONDA_ENV:-lejepa_gpu}}"
    if [ -n "${INIT_CKPT:-}" ]; then
      INIT_ENCODER_CKPT="$INIT_CKPT"
    elif [ -n "${DINO2_INIT_CKPT:-}" ]; then
      INIT_ENCODER_CKPT="$DINO2_INIT_CKPT"
    else
      INIT_ENCODER_CKPT="$(first_existing \
        "./outputs/binary_dino_shared_seasonal_rgb_balanced_cuda_80ep/phase1_encoder_best.pth" \
        "./outputs/binary_dino_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth" \
        "./outputs/phase1_dino_ssl_shared_seasonal/phase1_encoder_best.pth")" || {
          echo "Could not find a DINOv2 init checkpoint. Set DINO2_INIT_CKPT or INIT_CKPT."
          exit 1
        }
    fi
    ;;
  dino3|dinov3)
    ENCODER="dino3"
    CONDA_ENV="${CONDA_ENV:-${DINO_CONDA_ENV:-lejepa_gpu}}"
    if [ -n "${INIT_CKPT:-}" ]; then
      INIT_ENCODER_CKPT="$INIT_CKPT"
    elif [ -n "${DINO3_INIT_CKPT:-}" ]; then
      INIT_ENCODER_CKPT="$DINO3_INIT_CKPT"
    else
      INIT_ENCODER_CKPT="$(first_existing \
        "./outputs/binary_dino3_shared_seasonal_rgb_balanced_cuda_80ep/phase1_encoder_best.pth" \
        "./outputs/binary_dino3_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth" \
        "./outputs/phase1_dino3_ssl_shared_seasonal/phase1_encoder_best.pth")" || {
          echo "Could not find a DINOv3 init checkpoint."
          echo "Set DINO3_INIT_CKPT or INIT_CKPT to the DINOv3 phase1_encoder_best.pth."
          exit 1
        }
    fi
    ;;
  lejepa|le-jepa)
    ENCODER="lejepa"
    CONDA_ENV="${CONDA_ENV:-${LEJEPA_CONDA_ENV:-lejepa}}"
    if [ -n "${INIT_CKPT:-}" ]; then
      INIT_ENCODER_CKPT="$INIT_CKPT"
    elif [ -n "${LEJEPA_INIT_CKPT:-}" ]; then
      INIT_ENCODER_CKPT="$LEJEPA_INIT_CKPT"
    else
      INIT_ENCODER_CKPT="$(first_existing \
        "./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda_80ep/phase1_encoder_best.pth" \
        "./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth" \
        "./outputs/phase1_lejepa_ssl_large_gpu/phase1_encoder_best.pth")" || {
          echo "Could not find a LeJEPA init checkpoint. Set LEJEPA_INIT_CKPT or INIT_CKPT."
          exit 1
        }
    fi
    ;;
  *)
    echo "Unknown ENCODER=$ENCODER. Expected dino2, dino3, or lejepa."
    exit 1
    ;;
esac

if [ "$BAG_LAYOUT" != "grid" ] && [[ "$POOLING" == conv_* ]]; then
  echo "Convolutional MIL pooling requires BAG_LAYOUT=grid."
  exit 1
fi

if [ "$BAG_LAYOUT" = "grid" ]; then
  case "$BAG_INSTANCES" in
    1|9|25|49|81|121|169) ;;
    *)
      echo "Grid layout requires BAG_INSTANCES to be an odd square, for example 9, 25, or 49."
      exit 1
      ;;
  esac
fi

if [ "${SKIP_MODULE_LOAD:-0}" != "1" ]; then
  if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
      # Some SLURM batch shells do not initialise Environment Modules.
      # Source the standard init script before loading Anaconda.
      . /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
      . /usr/share/Modules/init/bash
    fi
  fi

  if command -v module >/dev/null 2>&1; then
    module load Anaconda3
  else
    echo "module command not found; assuming conda is already available."
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found after module setup."
  echo "Set SKIP_MODULE_LOAD=1 only if conda is already on PATH, or set CONDA_ENV for the cluster environment."
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export GEOTIFF_CSV="${GEOTIFF_CSV:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$SCRATCH_EXP_ROOT"
mkdir -p "$LOCAL_RESULTS_ROOT"

BASE_RUN_NAME="${ENCODER}_${TRAIN_IMAGE_MODE}_patch${PATCH_SIZE_PX}_${POOLING}_bag${BAG_INSTANCES}_seed${SEED}${RUN_TAG}"
RUN_NAME="${RUN_NAME:-$BASE_RUN_NAME}"
SCRATCH_OUT_DIR="$SCRATCH_EXP_ROOT/$RUN_NAME"
OUT_DIR="$LOCAL_RESULTS_ROOT/$RUN_NAME"

mkdir -p "$SCRATCH_OUT_DIR"
if [ -e "$OUT_DIR" ] && [ ! -L "$OUT_DIR" ]; then
  echo "Refusing to overwrite non-symlink output path: $OUT_DIR"
  exit 1
fi
ln -sfn "$SCRATCH_OUT_DIR" "$OUT_DIR"

{
  echo "experiment_name=$EXPERIMENT_NAME"
  echo "run_name=$RUN_NAME"
  echo "encoder=$ENCODER"
  echo "conda_env=$CONDA_ENV"
  echo "train_image_mode=$TRAIN_IMAGE_MODE"
  echo "eval_image_mode=$EVAL_IMAGE_MODE"
  echo "patch_size_px=$PATCH_SIZE_PX"
  echo "pooling=$POOLING"
  echo "bag_layout=$BAG_LAYOUT"
  echo "bag_instances=$BAG_INSTANCES"
  echo "conv_kernel_size=$CONV_KERNEL_SIZE"
  echo "seed=$SEED"
  echo "init_ckpt=$INIT_ENCODER_CKPT"
  echo "scratch_out_dir=$SCRATCH_OUT_DIR"
  echo "out_dir=$OUT_DIR"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
} > "$OUT_DIR/run_metadata.txt"

echo "============================================================"
echo "Job started at       : $(date)"
echo "Running on node      : $(hostname)"
echo "Python               : $(which python)"
python --version
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "Experiment           : $EXPERIMENT_NAME"
echo "Run name             : $RUN_NAME"
echo "Output dir           : $OUT_DIR"
echo "Scratch output dir   : $SCRATCH_OUT_DIR"
echo "Encoder              : $ENCODER"
echo "Init encoder         : $INIT_ENCODER_CKPT"
echo "Train image mode     : $TRAIN_IMAGE_MODE"
echo "Eval image mode      : $EVAL_IMAGE_MODE"
echo "Patch size px        : $PATCH_SIZE_PX"
echo "Bag layout/instances : $BAG_LAYOUT / $BAG_INSTANCES"
echo "Pooling              : $POOLING"
echo "Conv kernel size     : $CONV_KERNEL_SIZE"
echo "Epochs               : $EPOCHS"
echo "Batch size           : $BATCH_SIZE"
echo "============================================================"

test -f "$INIT_ENCODER_CKPT" || { echo "Missing init checkpoint: $INIT_ENCODER_CKPT"; exit 1; }
test -f "$TRAIN_SHP" || { echo "Missing train shapefile: $TRAIN_SHP"; exit 1; }
test -f "$VAL_SHP" || { echo "Missing validation shapefile: $VAL_SHP"; exit 1; }
test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }

python train_mil_classifier.py \
  --init_ckpt "$INIT_ENCODER_CKPT" \
  --train_shp "$TRAIN_SHP" \
  --val_shp "$VAL_SHP" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_dir "$OUT_DIR" \
  --label_field BinaryTree \
  --folder_field Folder \
  --file_field File \
  --fx_field fx \
  --fy_field fy \
  --coord_mode auto \
  --positive_class 1 \
  --image_mode "$TRAIN_IMAGE_MODE" \
  --eval_image_mode "$EVAL_IMAGE_MODE" \
  --image_size "$IMAGE_SIZE" \
  --patch_size_px "$PATCH_SIZE_PX" \
  --bag_radius_m "$BAG_RADIUS_M" \
  --negative_bag_radius_m "$NEGATIVE_BAG_RADIUS_M" \
  --bag_instances "$BAG_INSTANCES" \
  --bag_layout "$BAG_LAYOUT" \
  --pooling "$POOLING" \
  --lse_tau "$LSE_TAU" \
  --topk "$TOPK" \
  --conv_kernel_size "$CONV_KERNEL_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr_encoder "$LR_ENCODER" \
  --lr_head "$LR_HEAD" \
  --weight_decay 5e-4 \
  --freeze_encoder_epochs "$FREEZE_ENCODER_EPOCHS" \
  --patience 0 \
  --save_every 0 \
  --balanced_sampler \
  --train_repeat_factor "$TRAIN_REPEAT_FACTOR" \
  --max_black_fraction 0.20 \
  --max_bright_fraction 0.35 \
  --monitor_metric "$MONITOR_METRIC" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device cuda

python tune_binary_threshold.py \
  --pred_csv "$OUT_DIR/classifier_predictions.csv" \
  --positive_label 1 \
  --output_csv "$OUT_DIR/threshold_tuning.csv"

echo "============================================================"
echo "Job finished at      : $(date)"
echo "MIL output           : $OUT_DIR"
echo "Diagnostics          : $OUT_DIR/binary_score_diagnostics.json"
echo "Thresholds           : $OUT_DIR/threshold_tuning.csv"
echo "============================================================"
