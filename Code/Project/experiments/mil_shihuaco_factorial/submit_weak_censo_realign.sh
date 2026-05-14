#!/bin/bash

#SBATCH --job-name=weak_censo_realign
#SBATCH --partition=gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_curated_factorial/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_curated_factorial/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [ -n "${PROJECT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  SCRIPT_DIR_FALLBACK="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
  PROJECT_DIR="$(cd "$SCRIPT_DIR_FALLBACK/../.." && pwd)"
fi
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/apply_mil_realign.py" ]; then
  echo "Project directory does not contain apply_mil_realign.py: $PROJECT_DIR"
  exit 1
fi

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_curated_factorial}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT_ROOT/weak_censo_500_realign}"
LOG_ROOT="${LOG_ROOT:-$EXPERIMENT_ROOT/slurm_logs}"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export OUTPUT_ROOT

CENSO_CSV="${CENSO_CSV:-/mnt/parscratch/users/aca21jo/curated/censo_forestal_datos.csv}"
CURATED_SHP="${CURATED_SHP:-/mnt/parscratch/users/aca21jo/curated/copas-2023/copas-2023/copas_2023_condatos_vs2.shp}"
IMAGERY_ROOT="${IMAGERY_ROOT:-/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023}"

SUBSET_GPKG="${SUBSET_GPKG:-$OUTPUT_ROOT/weak_shihuahuaco_500_not_curated.gpkg}"
SUBSET_CSV="${SUBSET_CSV:-$OUTPUT_ROOT/weak_shihuahuaco_500_not_curated.csv}"
SUBSET_SUMMARY="${SUBSET_SUMMARY:-$OUTPUT_ROOT/weak_shihuahuaco_subset_summary.json}"

LIMIT="${LIMIT:-500}"
SEED="${SEED:-42}"
EXCLUDE_DISTANCE_M="${EXCLUDE_DISTANCE_M:-30}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best}"
SELECTION="${SELECTION:-raw}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CONDA_ENV="${CONDA_ENV:-lejepa_gpu}"
REFRESH_SUBSET="${REFRESH_SUBSET:-0}"

if [ "${SKIP_MODULE_LOAD:-0}" != "1" ]; then
  if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
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
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================================"
echo "Weak census Shihuahuaco realignment"
echo "Time            : $(date)"
echo "Project         : $PROJECT_DIR"
echo "Experiment root : $EXPERIMENT_ROOT"
echo "Output root     : $OUTPUT_ROOT"
echo "Partition       : ${SLURM_JOB_PARTITION:-unknown}"
echo "GPU             : ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "============================================================"

if [ "$REFRESH_SUBSET" = "1" ] || [ ! -f "$SUBSET_GPKG" ]; then
  echo
  echo "[INFO] Preparing weak census subset..."
  python prepare_weak_shihuahuaco_subset.py \
    --censo_csv "$CENSO_CSV" \
    --curated "$CURATED_SHP" \
    --imagery_root "$IMAGERY_ROOT" \
    --output_gpkg "$SUBSET_GPKG" \
    --output_csv "$SUBSET_CSV" \
    --summary_json "$SUBSET_SUMMARY" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --exclude_distance_m "$EXCLUDE_DISTANCE_M"
else
  echo
  echo "[INFO] Reusing existing subset: $SUBSET_GPKG"
fi

if [ -n "${RUN_NAMES:-}" ]; then
  run_dirs=()
  for name in $RUN_NAMES; do
    run_dirs+=("$EXPERIMENT_ROOT/$name")
  done
elif [ -n "${RUN_DIRS:-}" ]; then
  # shellcheck disable=SC2206
  run_dirs=($RUN_DIRS)
else
  run_dirs=()
  while IFS= read -r run_dir; do
    run_dirs+=("$run_dir")
  done < <(
    find "$EXPERIMENT_ROOT" -mindepth 1 -maxdepth 1 -type d \
      ! -name "slurm_logs" ! -name "weak_censo_500_realign" \
      -exec test -f "{}/mil_config.json" \; \
      -exec test -f "{}/phase1_encoder_${CHECKPOINT_NAME}.pth" \; \
      -exec test -f "{}/mil_head_${CHECKPOINT_NAME}.pth" \; \
      -print | sort
  )
fi

if [ "${#run_dirs[@]}" -eq 0 ]; then
  echo "No run directories found. Set RUN_NAMES or RUN_DIRS."
  exit 1
fi

echo
echo "[INFO] Models to apply: ${#run_dirs[@]}"
printf '  %s\n' "${run_dirs[@]}"

MODEL_ROOT="$OUTPUT_ROOT/realigned_by_model"
mkdir -p "$MODEL_ROOT"

for run_dir in "${run_dirs[@]}"; do
  if [ ! -d "$run_dir" ]; then
    echo "[WARN] Missing run directory: $run_dir"
    continue
  fi
  run_name="$(basename "$run_dir")"
  out_dir="$MODEL_ROOT/$run_name"
  mkdir -p "$out_dir"

  echo
  echo "------------------------------------------------------------"
  echo "[INFO] Applying model: $run_name"
  echo "------------------------------------------------------------"
  python apply_mil_realign.py \
    --mil_output_dir "$run_dir" \
    --input_points "$SUBSET_GPKG" \
    --output_csv "$out_dir/realigned_points.csv" \
    --output_gpkg "$out_dir/realigned_points.gpkg" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --selection "$SELECTION" \
    --device cuda \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --coord_mode world
done

echo
echo "[INFO] Combining per-model CSVs..."
python - <<'PY'
from pathlib import Path
import pandas as pd
import os

output_root = Path(os.environ["OUTPUT_ROOT"])
model_root = output_root / "realigned_by_model"
frames = []
for csv_path in sorted(model_root.glob("*/realigned_points.csv")):
    df = pd.read_csv(csv_path)
    df["realign_csv"] = str(csv_path)
    frames.append(df)
if frames:
    out = pd.concat(frames, ignore_index=True)
else:
    out = pd.DataFrame()
out_path = output_root / "realigned_points_all_models.csv"
out.to_csv(out_path, index=False)
print(out_path)
print(f"rows={len(out)}")
PY

echo
echo "Done."
echo "Subset                 : $SUBSET_GPKG"
echo "Per-model outputs      : $MODEL_ROOT"
echo "Combined realignments  : $OUTPUT_ROOT/realigned_points_all_models.csv"
