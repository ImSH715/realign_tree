# MIL Shihuahuaco Factorial Experiments

This folder queues and collects the MIL runs needed to compare:

- RGB with white boost vs green-mean with white boost.
- Small/current/large spatial crops.
- Standard MIL pooling vs neighbourhood convolution over the candidate grid.
- DINOv2, DINOv3, and LeJEPA encoder checkpoints.

The white boost used by these runs is the guarded version in
`train_supervised_encoder.py`: bright low-saturation pixels are boosted only
when the crop has vegetation context and the local pixel is not strongly
yellow-cast. This keeps the silvery-bark cue while reducing false positives on
yellow dirt paths.

## Default Grid

`submit_factorial.sh` expands this default grid:

- `ENCODERS="dino2 dino3 lejepa"`
- `IMAGE_MODES="rgb_white_boost rgb_green_mean_white_boost"`
- `PATCH_SIZES="160 224 320"`
- `POOLINGS="lse conv_lse"`

Both `lse` and `conv_lse` use `BAG_LAYOUT=grid` and `BAG_INSTANCES=25`, which
means each bag is a 5 by 5 neighbourhood. The non-convolution run therefore
uses the same candidate crops as the convolution run.

Default total: 36 SLURM jobs.

## Queue Runs

From `realign_tree/Code/Project`:

```bash
bash experiments/mil_shihuaco_factorial/submit_factorial.sh
```

Use `DRY_RUN=1` to preview the queue:

```bash
DRY_RUN=1 bash experiments/mil_shihuaco_factorial/submit_factorial.sh
```

Useful overrides:

```bash
PATCH_SIZES="160 224" EPOCHS=30 \
  bash experiments/mil_shihuaco_factorial/submit_factorial.sh
```

For DINOv3, set `DINO3_INIT_CKPT` unless the checkpoint exists in one of the
default fallback paths listed in `job_one_mil.sh`.

```bash
DINO3_INIT_CKPT=./outputs/my_dino3_run/phase1_encoder_best.pth \
  bash experiments/mil_shihuaco_factorial/submit_factorial.sh
```

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
