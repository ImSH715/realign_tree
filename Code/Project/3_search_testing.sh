#!/bin/bash

#SBATCH --job-name=L_11_class
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=logs/testing/lejepa/phase3/11_class_%j.out
#SBATCH --error=logs/testing/lejepa/phase3/11_class_%j.err

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

IMAGERY_ROOT="/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
POINTS_CSV="./outputs/evaluation/valid_points_recovery_20m_shihuahuaco_top1tif.csv"

ENCODER_CKPT="./outputs/phase1_5_lejepa_11/phase1_encoder_best.pth"
PROTOTYPES_CSV="./outputs/phase2_lejepa_11/class_prototypes.csv"

# beta 0.0002
python run_pipeline.py \
  --encoder_ckpt "./outputs/phase1_5_lejepa_11/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2_lejepa_11/class_prototypes.csv" \
  --points_csv "./outputs/evaluation/valid_points_recovery_20m_lejepa11.csv" \
  --imagery_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_csv "./outputs/evaluation/valid_points_recovery_20m_lejepa11_refined.csv" \
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
  --input_csv "./outputs/evaluation/valid_points_recovery_20m_lejepa11_refined.csv" \
  --output_csv "./outputs/evaluation/valid_points_recovery_20m_lejepa11_evaluated.csv"

python - <<'PY'
import pandas as pd

df = pd.read_csv("./outputs/evaluation/valid_points_recovery_20m_lejepa11_evaluated.csv")

print("rows:", len(df))
print("unique point_id:", df["point_id"].nunique())
print("mean_before_m:", df["distance_before_m"].mean())
print("mean_after_m :", df["distance_after_m"].mean())
print("median_before_m:", df["distance_before_m"].median())
print("median_after_m :", df["distance_after_m"].median())
print("mean_movement_m:", df["movement_m"].mean())
print("\nevaluation counts:")
print(df["evaluation"].value_counts())
print("\nlabel counts:")
print(df["label"].value_counts())
PY