#!/bin/bash

#SBATCH --job-name=resnet50_bin_season
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/mil/resnet/phase_1_5/resnet50_bin_season_%j.out
#SBATCH --error=logs/mil/resnet/phase_1_5/resnet50_bin_season_%j.err
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
EPOCHS="${EPOCHS:-40}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-0}"
TRAIN_REPEAT_FACTOR="${TRAIN_REPEAT_FACTOR:-2}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
LR_ENCODER="${LR_ENCODER:-1e-5}"
LR_HEAD="${LR_HEAD:-1e-4}"
SUPERVISED_INIT_DIR="${SUPERVISED_INIT_DIR:-}"
SUPERVISED_INIT_NAME="${SUPERVISED_INIT_NAME:-best}"
TRAIN_IMAGE_MODE="${TRAIN_IMAGE_MODE:-rgb}"

if [ -z "${EVAL_IMAGE_MODE+x}" ]; then
  if [ "$TRAIN_IMAGE_MODE" = "rgb_green_dropout" ]; then
    EVAL_IMAGE_MODE="rgb"
  else
    EVAL_IMAGE_MODE="$TRAIN_IMAGE_MODE"
  fi
fi

TRAIN_OUT_NAME="binary_resnet50_shared_seasonal_rgb_balanced_cuda${RUN_TAG}"
EVAL_OUT_NAME="phase2_classifier_binary_resnet50_shared_seasonal_rgb_balanced_cuda${RUN_TAG}"

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

ensure_output_link "$TRAIN_OUT_NAME"
ensure_output_link "$EVAL_OUT_NAME"

INIT_HEAD_ARGS=()

if [ -n "$SUPERVISED_INIT_DIR" ]; then
  if [ ! -f "$SUPERVISED_INIT_DIR/phase1_encoder_${SUPERVISED_INIT_NAME}.pth" ]; then
    echo "Missing supervised encoder checkpoint: $SUPERVISED_INIT_DIR/phase1_encoder_${SUPERVISED_INIT_NAME}.pth"
    exit 1
  fi
  if [ ! -f "$SUPERVISED_INIT_DIR/classifier_head_${SUPERVISED_INIT_NAME}.pth" ]; then
    echo "Missing supervised classifier head checkpoint: $SUPERVISED_INIT_DIR/classifier_head_${SUPERVISED_INIT_NAME}.pth"
    exit 1
  fi
  INIT_CKPT="$SUPERVISED_INIT_DIR/phase1_encoder_${SUPERVISED_INIT_NAME}.pth"
  INIT_HEAD_ARGS=(--init_head_ckpt "$SUPERVISED_INIT_DIR/classifier_head_${SUPERVISED_INIT_NAME}.pth")
else
  PHASE1_DIR="./outputs/phase1_resnet50_cpu"
  if [ -f "$PHASE1_DIR/phase1_encoder_best.pth" ]; then
    INIT_CKPT="$PHASE1_DIR/phase1_encoder_best.pth"
  elif [ -f "$PHASE1_DIR/phase1_encoder_last.pth" ]; then
    INIT_CKPT="$PHASE1_DIR/phase1_encoder_last.pth"
  else
    echo "Missing ResNet50 phase1 checkpoint under: $PHASE1_DIR"
    exit 1
  fi
fi

IMAGERY_ROOT="/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
TRAIN_SHP="./outputs/splits_binary/valid_points_train.shp"
VAL_SHP="./outputs/splits_binary/valid_points_val.shp"

TRAIN_OUT="./outputs/$TRAIN_OUT_NAME"
EVAL_OUT="./outputs/$EVAL_OUT_NAME"

echo "Using ResNet50 init checkpoint: $INIT_CKPT"
echo "Run tag               : ${RUN_TAG:-<none>}"
echo "Fine-tune epochs      : $EPOCHS"
echo "Freeze encoder epochs : $FREEZE_ENCODER_EPOCHS"
echo "Train repeat factor   : $TRAIN_REPEAT_FACTOR"
echo "Label smoothing       : $LABEL_SMOOTHING"
echo "Train image mode      : $TRAIN_IMAGE_MODE"
echo "Eval image mode       : $EVAL_IMAGE_MODE"

test -f "$TRAIN_SHP" || { echo "Missing train shapefile: $TRAIN_SHP"; exit 1; }
test -f "$VAL_SHP" || { echo "Missing validation shapefile: $VAL_SHP"; exit 1; }
test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }

echo "============================================================"
echo "Step 1: Fine-tuning binary classifier from ResNet50 encoder"
echo "============================================================"

python train_supervised_encoder.py \
  --init_ckpt "$INIT_CKPT" \
  "${INIT_HEAD_ARGS[@]}" \
  --train_shp "$TRAIN_SHP" \
  --val_shp "$VAL_SHP" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_dir "$TRAIN_OUT" \
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
  --epochs "$EPOCHS" \
  --lr_encoder "$LR_ENCODER" \
  --lr_head "$LR_HEAD" \
  --weight_decay 5e-4 \
  --freeze_encoder_epochs "$FREEZE_ENCODER_EPOCHS" \
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

echo "============================================================"
echo "Step 2: Evaluating seasonal ResNet50 binary classifier"
echo "============================================================"

python eval_classifier_head.py \
  --encoder_ckpt "$TRAIN_OUT/phase1_encoder_best.pth" \
  --head_ckpt "$TRAIN_OUT/classifier_head_best.pth" \
  --gt_path "$VAL_SHP" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_dir "$EVAL_OUT" \
  --label_field BinaryTree \
  --folder_field Folder \
  --file_field File \
  --fx_field fx \
  --fy_field fy \
  --coord_mode auto \
  --image_mode "$EVAL_IMAGE_MODE" \
  --image_size 224 \
  --patch_size_px 224 \
  --max_black_fraction 0.20 \
  --max_bright_fraction 0.35 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda

echo "============================================================"
echo "Step 3: Tuning binary threshold"
echo "============================================================"

python tune_binary_threshold.py \
  --pred_csv "$EVAL_OUT/classifier_predictions.csv" \
  --positive_label 1 \
  --output_csv "$EVAL_OUT/threshold_tuning.csv"

echo "============================================================"
echo "Job finished at: $(date)"
echo "Training output   : $TRAIN_OUT"
echo "Evaluation output : $EVAL_OUT"
echo "Diagnostics       : $EVAL_OUT/binary_score_diagnostics.json"
echo "============================================================"