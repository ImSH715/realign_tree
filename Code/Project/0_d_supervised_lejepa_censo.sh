#!/bin/bash

#SBATCH --job-name=phase3_sup_censo
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --output=phase3_sup_censo_%j.out
#SBATCH --error=phase3_sup_censo_%j.err
#SBATCH --mail-type=END,FAIL

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lejepa

echo "Using Python from: $(which python)"
python --version
echo "Job started at $(date)"

python old_data_run_pipeline.py \
  --encoder_ckpt "./outputs/phase1_lejepa_supervised/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2_lejepa_supervised/class_prototypes.csv" \
  --points_csv "/mnt/parscratch/users/acb20si/realign_tree/Code/Project/data/Censo_Forestal_overlap_all_strict.csv" \
  --output_csv "./outputs/phase3/refined_censo_supervised_lejepa_20m.csv" \
  --label_column "NOMBRE_COMUN" \
  --x_column "COORDENADA_ESTE" \
  --y_column "COORDENADA_NORTE" \
  --image_column "matched_tif" \
  --search_radius_px 560 \
  --coarse_step_px 32 \
  --refine_radius_px 96 \
  --refine_step_px 8 \
  --similarity cosine \
  --alpha 1.0 \
  --beta 0.0002 \
  --batch_size 32 \
  --device cuda

python eval_proxy_no_gt.py \
  --input_csv "./outputs/phase3/refined_censo_supervised_lejepa_20m.csv" \
  --single_output_csv "./outputs/evaluation/refined_censo_supervised_lejepa_proxy.csv" \
  --stability_output_csv "./outputs/evaluation/refined_censo_supervised_lejepa_stability.csv"

python analyze_feature_space.py \
  --encoder_ckpt "./outputs/phase1_lejepa_supervised/phase1_encoder_best.pth" \
  --prototypes_csv "./outputs/phase2_lejepa_supervised/class_prototypes.csv" \
  --input_csv "./outputs/phase3/refined_censo_supervised_lejepa_20m.csv" \
  --output_dir "./outputs/analysis/censo_supervised_lejepa_20m" \
  --label_column "target_label" \
  --image_column "image_path" \
  --mode refined \
  --dbscan_eps 0.20 \
  --dbscan_min_samples 5 \
  --batch_size 32 \
  --device cuda

echo "Job finished at $(date)"