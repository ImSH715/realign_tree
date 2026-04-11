python train_encoder.py \
  --train_root "/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023" \
  --output_dir "./outputs/phase1" \
  --ssl_epochs 1 \
  --batch_size_ssl 1 \
  --patches_per_image 1 \
  --extract_stride_px 1024 \
  --extract_batch_size 1 \
  --num_workers 0 \
  --device cpu