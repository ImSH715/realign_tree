#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

read -r -a ENCODER_LIST <<< "${ENCODERS:-dino2 dino3 lejepa}"
read -r -a POOLING_LIST <<< "${POOLINGS:-lse conv_lse}"

BAG_LAYOUT="${BAG_LAYOUT:-grid}"
BAG_INSTANCES="${BAG_INSTANCES:-25}"
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-3}"

IMAGERY_ROOT="${IMAGERY_ROOT:-/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023}"
TRAIN_SHP="${TRAIN_SHP:-./outputs/splits_binary/valid_points_train.shp}"
VAL_SHP="${VAL_SHP:-./outputs/splits_binary/valid_points_val.shp}"
SCRATCH_EXP_ROOT="${SCRATCH_EXP_ROOT:-/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial}"
SCRATCH_LOG_ROOT="${SCRATCH_LOG_ROOT:-$SCRATCH_EXP_ROOT/slurm_logs}"

status=0

check_file() {
  local label="$1"
  local path="$2"
  if [ -f "$path" ]; then
    echo "OK   $label: $path"
  else
    echo "MISS $label: $path"
    status=1
  fi
}

check_dir() {
  local label="$1"
  local path="$2"
  if [ -d "$path" ]; then
    echo "OK   $label: $path"
  else
    echo "MISS $label: $path"
    status=1
  fi
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

echo "============================================================"
echo "MIL factorial preflight"
echo "Project: $PROJECT_DIR"
echo "============================================================"

check_file "train shapefile" "$TRAIN_SHP"
check_file "val shapefile" "$VAL_SHP"
check_dir "imagery root" "$IMAGERY_ROOT"
check_file "MIL trainer" "$PROJECT_DIR/train_mil_classifier.py"
check_file "single-run job script" "$SCRIPT_DIR/job_one_mil.sh"
check_file "sweep job script" "$SCRIPT_DIR/job_encoder_sweep.sh"

if [ "$BAG_LAYOUT" = "grid" ]; then
  case "$BAG_INSTANCES" in
    9) grid_side=3; echo "OK   grid bag instances: $BAG_INSTANCES" ;;
    25) grid_side=5; echo "OK   grid bag instances: $BAG_INSTANCES" ;;
    49) grid_side=7; echo "OK   grid bag instances: $BAG_INSTANCES" ;;
    81) grid_side=9; echo "OK   grid bag instances: $BAG_INSTANCES" ;;
    121) grid_side=11; echo "OK   grid bag instances: $BAG_INSTANCES" ;;
    169) grid_side=13; echo "OK   grid bag instances: $BAG_INSTANCES" ;;
    *)
      grid_side=0
      echo "MISS grid BAG_INSTANCES must be an odd square >= 9, for example 9, 25, or 49: $BAG_INSTANCES"
      status=1
      ;;
  esac
else
  grid_side=0
  for pooling in "${POOLING_LIST[@]}"; do
    case "$pooling" in
      conv_*)
        echo "MISS convolutional pooling requires BAG_LAYOUT=grid: $pooling with BAG_LAYOUT=$BAG_LAYOUT"
        status=1
        ;;
    esac
  done
fi

case "$CONV_KERNEL_SIZE" in
  ''|*[!0-9]*)
    echo "MISS CONV_KERNEL_SIZE must be a positive odd integer: $CONV_KERNEL_SIZE"
    status=1
    ;;
  *)
    if [ "$CONV_KERNEL_SIZE" -le 0 ] || [ $((CONV_KERNEL_SIZE % 2)) -eq 0 ]; then
      echo "MISS CONV_KERNEL_SIZE must be a positive odd integer: $CONV_KERNEL_SIZE"
      status=1
    elif [ "$grid_side" -gt 0 ] && [ "$CONV_KERNEL_SIZE" -gt "$grid_side" ]; then
      echo "MISS CONV_KERNEL_SIZE=$CONV_KERNEL_SIZE is larger than the ${grid_side}x${grid_side} bag grid"
      status=1
    else
      echo "OK   conv kernel size: $CONV_KERNEL_SIZE"
    fi
    ;;
esac

if [ "$BAG_LAYOUT" = "grid" ] && [ "$grid_side" -gt 0 ]; then
  for pooling in "${POOLING_LIST[@]}"; do
    case "$pooling" in
      conv_*)
        # Already validated above.
        ;;
      max|lse|topk)
        ;;
      *)
        echo "WARN pooling will be validated by train_mil_classifier.py: $pooling"
        ;;
    esac
  done
fi

if mkdir -p "$SCRATCH_EXP_ROOT" >/dev/null 2>&1; then
  echo "OK   scratch root writable or creatable: $SCRATCH_EXP_ROOT"
else
  echo "MISS scratch root not writable/creatable: $SCRATCH_EXP_ROOT"
  status=1
fi

if mkdir -p "$SCRATCH_LOG_ROOT" >/dev/null 2>&1; then
  echo "OK   scratch log root writable or creatable: $SCRATCH_LOG_ROOT"
else
  echo "MISS scratch log root not writable/creatable: $SCRATCH_LOG_ROOT"
  status=1
fi

if command -v sbatch >/dev/null 2>&1; then
  echo "OK   sbatch: $(command -v sbatch)"
else
  echo "MISS sbatch not found"
  status=1
fi

if command -v module >/dev/null 2>&1; then
  echo "OK   module command available"
else
  echo "WARN module command not currently available in this shell"
fi

if command -v conda >/dev/null 2>&1; then
  echo "OK   conda: $(command -v conda)"
else
  echo "WARN conda not currently on PATH before module loading"
fi

echo
echo "Checkpoint fallbacks"

for encoder in "${ENCODER_LIST[@]}"; do
  case "$encoder" in
    dino2|dinov2)
      dino2_ckpt="${DINO2_INIT_CKPT:-}"
      if [ -z "$dino2_ckpt" ]; then
        dino2_ckpt="$(first_existing \
          "./outputs/binary_dino_shared_seasonal_rgb_balanced_cuda_80ep/phase1_encoder_best.pth" \
          "./outputs/binary_dino_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth" \
          "./outputs/phase1_dino_ssl_shared_seasonal/phase1_encoder_best.pth")" || true
      fi
      if [ -n "$dino2_ckpt" ]; then
        echo "OK   DINOv2 checkpoint: $dino2_ckpt"
      else
        echo "MISS DINOv2 checkpoint; set DINO2_INIT_CKPT"
        status=1
      fi
      ;;
    dino3|dinov3)
      dino3_ckpt="${DINO3_INIT_CKPT:-}"
      if [ -z "$dino3_ckpt" ]; then
        dino3_ckpt="$(first_existing \
          "./outputs/binary_dino3_shared_seasonal_rgb_balanced_cuda_80ep/phase1_encoder_best.pth" \
          "./outputs/binary_dino3_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth" \
          "./outputs/phase1_dino3_ssl_shared_seasonal/phase1_encoder_best.pth")" || true
      fi
      if [ -n "$dino3_ckpt" ]; then
        echo "OK   DINOv3 checkpoint: $dino3_ckpt"
      else
        echo "MISS DINOv3 checkpoint; set DINO3_INIT_CKPT or exclude dino3 with ENCODERS=\"dino2 lejepa\""
        status=1
      fi
      ;;
    lejepa|le-jepa)
      lejepa_ckpt="${LEJEPA_INIT_CKPT:-}"
      if [ -z "$lejepa_ckpt" ]; then
        lejepa_ckpt="$(first_existing \
          "./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda_80ep/phase1_encoder_best.pth" \
          "./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth" \
          "./outputs/phase1_lejepa_ssl_large_gpu/phase1_encoder_best.pth")" || true
      fi
      if [ -n "$lejepa_ckpt" ]; then
        echo "OK   LeJEPA checkpoint: $lejepa_ckpt"
      else
        echo "MISS LeJEPA checkpoint; set LEJEPA_INIT_CKPT"
        status=1
      fi
      ;;
    *)
      echo "MISS unknown encoder in ENCODERS: $encoder"
      status=1
      ;;
  esac
done

echo
if [ "$status" -eq 0 ]; then
  echo "Preflight passed."
else
  echo "Preflight found missing requirements."
fi
exit "$status"
