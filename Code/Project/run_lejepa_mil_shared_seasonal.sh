#!/bin/bash

#SBATCH --job-name=lejepa_mil_20m
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --output=logs/mil/lejepa/phase2/lejepa_mil_20m_%j.out
#SBATCH --error=logs/mil/lejepa/phase2/lejepa_mil_20m_%j.err
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
export GEOTIFF_CSV=""
conda activate lejepa

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python: $(which python)"
python --version
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "============================================================"

SCRATCH_OUT_ROOT="/mnt/parscratch/users/aca21jo/realign_outputs"
mkdir -p "$SCRATCH_OUT_ROOT"

RUN_TAG="${RUN_TAG:-}"
TRAIN_IMAGE_MODE="${TRAIN_IMAGE_MODE:-rgb}"
if [ -z "${EVAL_IMAGE_MODE+x}" ]; then
  if [ "$TRAIN_IMAGE_MODE" = "rgb_green_dropout" ]; then
    EVAL_IMAGE_MODE="rgb"
  else
    EVAL_IMAGE_MODE="$TRAIN_IMAGE_MODE"
  fi
fi

BAG_RADIUS_M="${BAG_RADIUS_M:-20}"
NEGATIVE_BAG_RADIUS_M="${NEGATIVE_BAG_RADIUS_M:-0}"
BAG_INSTANCES="${BAG_INSTANCES:-17}"
BAG_LAYOUT="${BAG_LAYOUT:-rings}"
POOLING="${POOLING:-lse}"
LSE_TAU="${LSE_TAU:-1.0}"
TOPK="${TOPK:-3}"
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
PATCH_SIZE_PX="${PATCH_SIZE_PX:-224}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-3}"
TRAIN_REPEAT_FACTOR="${TRAIN_REPEAT_FACTOR:-2}"
LR_ENCODER="${LR_ENCODER:-5e-7}"
LR_HEAD="${LR_HEAD:-1e-4}"
MONITOR_METRIC="${MONITOR_METRIC:-val_macro_f1}"

OUT_NAME="${OUT_NAME:-mil_lejepa_shared_seasonal_20m_${TRAIN_IMAGE_MODE}${RUN_TAG}_cuda}"

ensure_output_link() {
  local name="$1"
  local abs="$SCRATCH_OUT_ROOT/$name"
  local link="./outputs/$name"
  mkdir -p "$abs"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    echo "Refusing to overwrite non-symlink output path: $link"
    echo "Move it to scratch or remove it, then rerun."
    exit 1
  fi
  ln -sfn "$abs" "$link"
}

ensure_output_link "$OUT_NAME"

if [ -n "${INIT_CKPT:-}" ]; then
  INIT_ENCODER_CKPT="$INIT_CKPT"
elif [ -f "./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda_80ep/phase1_encoder_best.pth" ]; then
  INIT_ENCODER_CKPT="./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda_80ep/phase1_encoder_best.pth"
elif [ -f "./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth" ]; then
  INIT_ENCODER_CKPT="./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth"
elif [ -f "./outputs/phase1_lejepa_ssl_large_gpu/phase1_encoder_best.pth" ]; then
  INIT_ENCODER_CKPT="./outputs/phase1_lejepa_ssl_large_gpu/phase1_encoder_best.pth"
else
  echo "Could not find an init LEJEPA encoder checkpoint."
  echo "Set INIT_CKPT explicitly, or run seasonal LEJEPA binary / SSL first."
  exit 1
fi

IMAGERY_ROOT="/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
TRAIN_SHP="./outputs/splits_binary/valid_points_train.shp"
VAL_SHP="./outputs/splits_binary/valid_points_val.shp"
OUT_DIR="./outputs/$OUT_NAME"

echo "MIL output dir        : $OUT_DIR"
echo "Init encoder          : $INIT_ENCODER_CKPT"
echo "Train image mode      : $TRAIN_IMAGE_MODE"
echo "Eval image mode       : $EVAL_IMAGE_MODE"
echo "Positive bag radius m : $BAG_RADIUS_M"
echo "Negative bag radius m : $NEGATIVE_BAG_RADIUS_M"
echo "Bag instances         : $BAG_INSTANCES"
echo "Bag layout            : $BAG_LAYOUT"
echo "Patch size px         : $PATCH_SIZE_PX"
echo "Model image size      : $IMAGE_SIZE"
echo "Pooling               : $POOLING"
echo "Conv kernel size      : $CONV_KERNEL_SIZE"
echo "Epochs                : $EPOCHS"
echo "Batch size            : $BATCH_SIZE"

test -f "$INIT_ENCODER_CKPT" || { echo "Missing init checkpoint: $INIT_ENCODER_CKPT"; exit 1; }
test -f "$TRAIN_SHP" || { echo "Missing train shapefile: $TRAIN_SHP"; exit 1; }
test -f "$VAL_SHP" || { echo "Missing validation shapefile: $VAL_SHP"; exit 1; }
test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }

echo "============================================================"
echo "Step 1: Training 20 m MIL classifier"
echo "============================================================"

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
  --num_workers 4 \
  --device cuda

echo "============================================================"
echo "Step 2: Tuning MIL bag threshold"
echo "============================================================"

python tune_binary_threshold.py \
  --pred_csv "$OUT_DIR/classifier_predictions.csv" \
  --positive_label 1 \
  --output_csv "$OUT_DIR/threshold_tuning.csv"

echo "============================================================"
echo "Job finished at: $(date)"
echo "MIL output      : $OUT_DIR"
echo "Diagnostics     : $OUT_DIR/binary_score_diagnostics.json"
echo "Thresholds      : $OUT_DIR/threshold_tuning.csv"
echo "============================================================"
