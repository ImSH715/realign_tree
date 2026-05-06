#!/bin/bash

#SBATCH --job-name=bin_ppv1_p3
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/phase3/bin_ppv1_p3_%j.out
#SBATCH --error=logs/phase3/bin_ppv1_p3_%j.err
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

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

IMAGERY_ROOT="/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"

POINTS_CSV="./outputs/evaluation/valid_points_recovery_20m_shihuahuaco_only.csv"

ENCODER_CKPT="./outputs/phase1_5_lejepa_cpu_binary_preprocess/phase1_encoder_best.pth"
PROTOTYPES_CSV="./outputs/phase2_binary_shihuahuaco_preprocess/class_prototypes.csv"

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

echo "Checking required inputs..."
test -d "$IMAGERY_ROOT" || { echo "Missing imagery root: $IMAGERY_ROOT"; exit 1; }
test -f "$POINTS_CSV" || { echo "Missing points csv: $POINTS_CSV"; exit 1; }
test -f "$ENCODER_CKPT" || { echo "Missing encoder ckpt: $ENCODER_CKPT"; exit 1; }
test -f "$PROTOTYPES_CSV" || { echo "Missing prototypes csv: $PROTOTYPES_CSV"; exit 1; }
echo "All required inputs found."

# ------------------------------------------------------------
# EXP 1: beta = 0.0002
# ------------------------------------------------------------

echo "============================================================"
echo "EXP 1: prototype phase3 (preprocess v1) | beta=0.0002"
echo "============================================================"

python run_pipeline.py \
  --encoder_ckpt "$ENCODER_CKPT" \
  --prototypes_csv "$PROTOTYPES_CSV" \
  --points_csv "$POINTS_CSV" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_csv "./outputs/evaluation/phase3_binary_cpu_proto_ppv1_beta0002_refined.csv" \
  --tile_column "matched_tif" \
  --point_id_column "point_id" \
  --x_column "original_east" \
  --y_column "original_north" \
  --target_label_column "label" \
  --coord_type world \
  --search_radius_px 560 \
  --coarse_step_px 32 \
  --refine_radius_px 96 \
  --refine_step_px 8 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.0002 \
  --batch_size 32 \
  --device cpu \
  --no_amp

python eval_direct_gt.py \
  --input_csv "./outputs/evaluation/phase3_binary_cpu_proto_ppv1_beta0002_refined.csv" \
  --output_csv "./outputs/evaluation/phase3_binary_cpu_proto_ppv1_beta0002_evaluated.csv"

# ------------------------------------------------------------
# EXP 2: beta = 0.002
# ------------------------------------------------------------

echo "============================================================"
echo "EXP 2: prototype phase3 (preprocess v1) | beta=0.002"
echo "============================================================"

python run_pipeline.py \
  --encoder_ckpt "$ENCODER_CKPT" \
  --prototypes_csv "$PROTOTYPES_CSV" \
  --points_csv "$POINTS_CSV" \
  --imagery_root "$IMAGERY_ROOT" \
  --output_csv "./outputs/evaluation/phase3_binary_cpu_proto_ppv1_beta0002b_refined.csv" \
  --tile_column "matched_tif" \
  --point_id_column "point_id" \
  --x_column "original_east" \
  --y_column "original_north" \
  --target_label_column "label" \
  --coord_type world \
  --search_radius_px 560 \
  --coarse_step_px 32 \
  --refine_radius_px 96 \
  --refine_step_px 8 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.002 \
  --batch_size 32 \
  --device cpu \
  --no_amp

python eval_direct_gt.py \
  --input_csv "./outputs/evaluation/phase3_binary_cpu_proto_ppv1_beta0002b_refined.csv" \
  --output_csv "./outputs/evaluation/phase3_binary_cpu_proto_ppv1_beta0002b_evaluated.csv"

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

python - <<'PY'
import pandas as pd

files = [
    "./outputs/evaluation/phase3_binary_cpu_proto_ppv1_beta0002_evaluated.csv",
    "./outputs/evaluation/phase3_binary_cpu_proto_ppv1_beta0002b_evaluated.csv",
]

print("=" * 100)
print("SUMMARY")
print("=" * 100)

for path in files:
    df = pd.read_csv(path)
    print(path)
    print("mean_before_m   :", df["distance_before_m"].mean())
    print("mean_after_m    :", df["distance_after_m"].mean())
    print("median_before_m :", df["distance_before_m"].median())
    print("median_after_m  :", df["distance_after_m"].median())
    print("mean_movement_m :", df["movement_m"].mean())
    print("improved :", int((df["evaluation"] == "improved").sum()))
    print("unchanged:", int((df["evaluation"] == "unchanged").sum()))
    print("worse    :", int((df["evaluation"] == "worse").sum()))
    print("-" * 80)
PY

echo "============================================================"
echo "PHASE 3 COMPLETED"
echo "Finished at: $(date)"
echo "============================================================"