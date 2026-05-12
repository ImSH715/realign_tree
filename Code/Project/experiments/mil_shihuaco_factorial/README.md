# MIL Shihuahuaco Factorial Experiments

This folder queues and collects the MIL runs needed to compare:

- RGB with white boost vs green-mean with white boost.
- Standard MIL pooling vs neighbourhood convolution over the candidate grid.
- DINOv2, DINOv3, and LeJEPA encoder checkpoints.

The white boost used by these runs is the guarded version in
`train_supervised_encoder.py`: bright low-saturation pixels are boosted only
when the crop has vegetation context and the local pixel is not strongly
yellow-cast. This keeps the silvery-bark cue while reducing false positives on
yellow dirt paths.

## Default Grid

`submit_factorial.sh` expands this default run grid:

- `ENCODERS="dino2 dino3 lejepa"`
- `IMAGE_MODES="rgb_white_boost rgb_green_mean_white_boost"`
- `PATCH_SIZES="160"`
- `POOLINGS="lse conv_lse"`

Both `lse` and `conv_lse` use `BAG_LAYOUT=grid` and `BAG_INSTANCES=25`, which
means each bag is a 5 by 5 neighbourhood. The non-convolution run therefore
uses the same candidate crops as the convolution run.

Default total: 12 training runs grouped into 6 SLURM jobs, one job per
`encoder x pooling` pair. Each job runs the RGB-vs-green comparison
sequentially. Patch size is fixed at 160 px because the pulled MIL logs show
the strongest previous run at `p160`: green-mean plus white boost reached
about 0.711 validation/selected PR-AUC, ahead of the available 224 px runs.

## Queue Runs

From `realign_tree/Code/Project`:

```bash
bash experiments/mil_shihuaco_factorial/preflight_factorial.sh
bash experiments/mil_shihuaco_factorial/submit_factorial.sh
```

Use `DRY_RUN=1` to preview the queue:

```bash
DRY_RUN=1 bash experiments/mil_shihuaco_factorial/submit_factorial.sh
```

Useful overrides:

```bash
EPOCHS=30 \
  bash experiments/mil_shihuaco_factorial/submit_factorial.sh
```

The sweeps skip already completed run folders by default, so if a sweep hits
the walltime you can submit the same command again and it will continue from
the missing combinations.

To force the older one-job-per-configuration behaviour, set
`SUBMIT_MODE=grid`. To merge all pooling/image-mode runs by encoder, set
`SUBMIT_MODE=encoder_sweep`. The recommended mode is the default
`model_pooling_sweep`, which submits 6 jobs.

For DINOv3, set `DINO3_INIT_CKPT` unless the checkpoint exists in one of the
default fallback paths listed in `job_one_mil.sh`.

```bash
DINO3_INIT_CKPT=./outputs/my_dino3_run/phase1_encoder_best.pth \
  bash experiments/mil_shihuaco_factorial/submit_factorial.sh
```

## Shortcut Audit Run

After visual inspection, the convolutional MIL head should be treated as a
bag-level context classifier rather than a direct coordinate selector. The
focused shortcut audit run retrains only the strongest DINOv2 configuration
with the stricter dirt-path suppression in `train_supervised_encoder.py` and
the raw-instance coordinate selection in `train_mil_classifier.py`.

```bash
DRY_RUN=1 bash experiments/mil_shihuaco_factorial/submit_dino2_shortcut_audit.sh
bash experiments/mil_shihuaco_factorial/submit_dino2_shortcut_audit.sh
```

Defaults:

- `RUN_NAME=dino2_rgb_white_boost_patch160_conv_lse_bag25_seed42_shortcut_audit`
- `EPOCHS=25`
- `MONITOR_METRIC=val_average_precision`
- `POOLING=conv_lse`
- `BAG_LAYOUT=grid`
- `BAG_INSTANCES=25`

## Prepare Missing Encoders

The MIL jobs need compatible encoder checkpoints. If LeJEPA or DINOv3 are
missing from `./outputs`, prepare them before requeueing those MIL runs.

LeJEPA can use the existing ViT recipe:

```bash
ENCODERS="lejepa" DRY_RUN=1 bash experiments/mil_shihuaco_factorial/submit_prepare_encoders.sh
ENCODERS="lejepa" bash experiments/mil_shihuaco_factorial/submit_prepare_encoders.sh
```

For DINOv3, first list the DINOv3 model names available in the active `timm`
environment:

```bash
module load Anaconda3
conda activate lejepa_gpu
python -c "import timm; print('\n'.join(timm.list_models('*dino*3*')))"
```

Then pass the chosen name:

```bash
DINO3_BACKBONE_NAME="<timm-dino3-model-name>" \
ENCODERS="dino3" \
bash experiments/mil_shihuaco_factorial/submit_prepare_encoders.sh
```

The prep job runs phase-1 SSL if needed, then binary Shihuahuaco adaptation.
The MIL init checkpoints it creates are:

- `./outputs/binary_lejepa_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth`
- `./outputs/binary_dino3_shared_seasonal_rgb_balanced_cuda/phase1_encoder_best.pth`

## Collect Results

After jobs finish:

```bash
python experiments/mil_shihuaco_factorial/collect_results.py
```

This writes:

- `experiments/mil_shihuaco_factorial/summary.csv`
- `experiments/mil_shihuaco_factorial/summary.md`

The summary table includes default-threshold metrics, PR-AUC, ROC-AUC, and the
best Shihuahuaco F1 found by `tune_binary_threshold.py`.

By default, result collection scans the scratch root
`/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial`.

## Monitor Runs

From the cluster login node:

```bash
bash experiments/mil_shihuaco_factorial/monitor_factorial.sh
```

To keep it refreshing:

```bash
watch -n 60 bash experiments/mil_shihuaco_factorial/monitor_factorial.sh
```

The monitor shows active SLURM jobs, recent job history, completed result-file
counts, and the current top rows from `summary.md`.

For instant failures, inspect the first failing line from the SLURM logs:

```bash
bash experiments/mil_shihuaco_factorial/diagnose_failed_jobs.sh 10152392 10152399
```

To cancel the queued/running factorial jobs only:

```bash
bash experiments/mil_shihuaco_factorial/cancel_factorial_jobs.sh
DRY_RUN=0 bash experiments/mil_shihuaco_factorial/cancel_factorial_jobs.sh
```
