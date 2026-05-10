#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

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

echo
if [ "$status" -eq 0 ]; then
  echo "Preflight passed."
else
  echo "Preflight found missing requirements."
fi
exit "$status"
