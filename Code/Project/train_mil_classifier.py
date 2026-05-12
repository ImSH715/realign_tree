"""
Multiple-instance binary training for noisy tree-point labels.

Positive rows are treated as bags of candidate crops around the supplied point:
at least one crop in the bag should contain the target crown. Negative rows are
centered by default, because a non-target tree point does not imply a whole 20 m
neighbourhood is target-free.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.models.checkpoint import load_encoder_from_checkpoint
from train_supervised_encoder import (
    VALID_IMAGE_MODES,
    build_eval_transform,
    build_tif_index,
    build_train_transform,
    convert_to_pixel,
    forward_features,
    infer_feature_dim,
    patch_quality_fractions,
    resolve_tif_path_fast,
    safe_mkdir,
    save_compatible_checkpoint,
    set_encoder_trainable,
    set_seed,
)


@dataclass
class Config:
    init_ckpt: str
    train_shp: str
    val_shp: str
    imagery_root: str
    output_dir: str

    label_field: str = "BinaryTree"
    folder_field: str = "Folder"
    file_field: str = "File"
    fx_field: str = "fx"
    fy_field: str = "fy"
    coord_mode: str = "auto"
    positive_class: str = "1"

    image_size: int = 224
    patch_size_px: int = 224
    image_mode: str = "rgb"
    eval_image_mode: str = "rgb"

    bag_radius_m: float = 20.0
    negative_bag_radius_m: float = 0.0
    bag_instances: int = 17
    bag_layout: str = "rings"
    pooling: str = "lse"
    lse_tau: float = 1.0
    topk: int = 3
    conv_kernel_size: int = 3

    batch_size: int = 2
    epochs: int = 50
    lr_encoder: float = 5e-7
    lr_head: float = 1e-4
    weight_decay: float = 5e-4
    freeze_encoder_epochs: int = 3
    patience: int = 0

    max_black_fraction: float = 1.0
    max_bright_fraction: float = 1.0
    train_repeat_factor: int = 1

    use_amp: bool = True
    use_pos_weight: bool = True
    use_balanced_sampler: bool = False
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 42
    monitor_metric: str = "val_macro_f1"
    save_every: int = 0


def read_patch_from_src(src, px: float, py: float, patch_size: int) -> Image.Image:
    half = patch_size // 2
    col0 = int(round(px)) - half
    row0 = int(round(py)) - half
    window = rasterio.windows.Window(col0, row0, patch_size, patch_size)
    arr = src.read(window=window, boundless=True, fill_value=0)

    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    else:
        raise ValueError(f"Invalid band count: {arr.shape}")

    arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.nanpercentile(arr, [1, 99])
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr)


def raster_pixel_size(src) -> Tuple[float, float]:
    x_size = abs(float(src.transform.a)) if src.transform is not None else 1.0
    y_size = abs(float(src.transform.e)) if src.transform is not None else 1.0
    if not np.isfinite(x_size) or x_size <= 0:
        x_size = 1.0
    if not np.isfinite(y_size) or y_size <= 0:
        y_size = x_size
    return x_size, y_size


def fixed_offsets(n: int, radius_px_x: float, radius_px_y: float) -> np.ndarray:
    if n <= 0:
        raise ValueError("bag_instances must be positive")
    if radius_px_x <= 0 or radius_px_y <= 0:
        return np.zeros((n, 2), dtype=np.float32)

    offsets = [(0.0, 0.0)]
    directions = [
        (math.cos(theta), math.sin(theta))
        for theta in np.linspace(0, 2 * math.pi, 8, endpoint=False)
    ]
    for ring in [0.5, 1.0, 0.25, 0.75, 0.9]:
        for dx, dy in directions:
            offsets.append((ring * radius_px_x * dx, ring * radius_px_y * dy))
            if len(offsets) >= n:
                return np.asarray(offsets, dtype=np.float32)

    return np.asarray(offsets[:n], dtype=np.float32)


def grid_side_for_instances(n: int) -> int:
    side = int(round(math.sqrt(int(n))))
    if side * side != int(n):
        raise ValueError(
            f"Grid bag layout requires bag_instances to be a square number, got {n}. "
            "Use 9, 25, 49, ..."
        )
    if side < 3 or side % 2 == 0:
        raise ValueError(
            f"Grid bag layout requires an odd side length >= 3, got {side}x{side}."
        )
    return side


def grid_offsets(n: int, radius_px_x: float, radius_px_y: float) -> np.ndarray:
    if n <= 0:
        raise ValueError("bag_instances must be positive")
    side = grid_side_for_instances(n)
    if radius_px_x <= 0 or radius_px_y <= 0:
        return np.zeros((n, 2), dtype=np.float32)

    xs = np.linspace(-radius_px_x, radius_px_x, side, dtype=np.float32)
    ys = np.linspace(-radius_px_y, radius_px_y, side, dtype=np.float32)
    offsets = [(float(x), float(y)) for y in ys for x in xs]
    return np.asarray(offsets, dtype=np.float32)


def random_offsets(n: int, radius_px_x: float, radius_px_y: float) -> np.ndarray:
    if n <= 0:
        raise ValueError("bag_instances must be positive")
    offsets = [(0.0, 0.0)]
    if radius_px_x <= 0 or radius_px_y <= 0:
        return np.zeros((n, 2), dtype=np.float32)

    while len(offsets) < n:
        r = math.sqrt(float(np.random.rand()))
        theta = 2 * math.pi * float(np.random.rand())
        offsets.append((radius_px_x * r * math.cos(theta), radius_px_y * r * math.sin(theta)))

    return np.asarray(offsets, dtype=np.float32)


class MILPointBagDataset(Dataset):
    def __init__(
        self,
        shp_path: str,
        imagery_root: str,
        label_field: str,
        folder_field: str,
        file_field: str,
        fx_field: str,
        fy_field: str,
        coord_mode: str,
        positive_class: str,
        patch_size_px: int,
        transform,
        bag_radius_m: float,
        negative_bag_radius_m: float,
        bag_instances: int,
        max_black_fraction: float,
        max_bright_fraction: float,
        bag_layout: str = "rings",
        folder_to_paths: Dict[str, List[str]] = None,
        repeat_factor: int = 1,
        train: bool = True,
    ):
        self.gdf = gpd.read_file(shp_path)
        self.gdf = self.gdf[self.gdf[label_field].notna()].copy()
        self.gdf[label_field] = self.gdf[label_field].astype(str).str.strip()

        self.imagery_root = imagery_root
        self.label_field = label_field
        self.folder_field = folder_field
        self.file_field = file_field
        self.fx_field = fx_field
        self.fy_field = fy_field
        self.coord_mode = coord_mode
        self.positive_class = str(positive_class)
        self.patch_size_px = patch_size_px
        self.transform = transform
        self.bag_radius_m = float(bag_radius_m)
        self.negative_bag_radius_m = float(negative_bag_radius_m)
        self.bag_instances = int(bag_instances)
        self.bag_layout = str(bag_layout)
        self.max_black_fraction = float(max_black_fraction)
        self.max_bright_fraction = float(max_bright_fraction)
        self.repeat_factor = max(1, int(repeat_factor))
        self.train = bool(train)

        if self.bag_layout not in {"rings", "grid"}:
            raise ValueError("bag_layout must be one of: rings, grid")
        if self.bag_layout == "grid":
            grid_side_for_instances(self.bag_instances)

        required = [label_field, folder_field, file_field, fx_field, fy_field]
        for c in required:
            if c not in self.gdf.columns:
                raise ValueError(f"Missing required field '{c}'. Available: {self.gdf.columns.tolist()}")

        if folder_to_paths is None:
            folder_to_paths = build_tif_index(imagery_root)
        self.folder_to_paths = folder_to_paths

        self.samples = []
        self.failed_rows = []

        print(f"[INFO] Resolving MIL bags for {os.path.basename(shp_path)}...")
        iterator = tqdm(self.gdf.iterrows(), total=len(self.gdf), dynamic_ncols=True, desc="Resolving bags")
        for i, (_, row) in enumerate(iterator):
            label = str(row[label_field]).strip()
            try:
                image_path = resolve_tif_path_fast(
                    self.folder_to_paths,
                    row[folder_field],
                    row[file_field],
                )
                with rasterio.open(image_path) as src:
                    px, py, used_mode = convert_to_pixel(
                        src,
                        float(row[fx_field]),
                        float(row[fy_field]),
                        coord_mode,
                    )
                    pixel_size_x, pixel_size_y = raster_pixel_size(src)

                self.samples.append({
                    "source_index": int(i),
                    "label": label,
                    "bag_label": int(label == self.positive_class),
                    "image_path": image_path,
                    "folder": str(row[folder_field]),
                    "file": str(row[file_field]),
                    "raw_x": float(row[fx_field]),
                    "raw_y": float(row[fy_field]),
                    "center_px": float(px),
                    "center_py": float(py),
                    "coord_mode_used": used_mode,
                    "pixel_size_x": float(pixel_size_x),
                    "pixel_size_y": float(pixel_size_y),
                })
            except Exception as e:
                self.failed_rows.append((i, str(e)))
                if len(self.failed_rows) <= 10:
                    print(f"[WARN] Failed to resolve row {i}: {e}")

        print(f"[INFO] Resolved bags: {len(self.samples)}")
        print(f"[INFO] Failed rows  : {len(self.failed_rows)}")
        print(f"[INFO] Bag labels   : {pd.Series([s['bag_label'] for s in self.samples]).value_counts().to_dict()}")
        if not self.samples:
            raise RuntimeError("No usable MIL bags after TIFF path resolution.")

    def __len__(self):
        return len(self.samples) * self.repeat_factor

    def targets(self) -> List[int]:
        labels = [s["bag_label"] for s in self.samples]
        return labels * self.repeat_factor

    def _offsets_for_sample(self, sample):
        radius_m = self.bag_radius_m if sample["bag_label"] == 1 else self.negative_bag_radius_m
        radius_px_x = radius_m / max(sample["pixel_size_x"], 1e-6)
        radius_px_y = radius_m / max(sample["pixel_size_y"], 1e-6)
        if self.bag_layout == "grid":
            return grid_offsets(self.bag_instances, radius_px_x, radius_px_y)
        if self.train:
            return random_offsets(self.bag_instances, radius_px_x, radius_px_y)
        return fixed_offsets(self.bag_instances, radius_px_x, radius_px_y)

    def __getitem__(self, idx):
        idx = idx % len(self.samples)
        sample = self.samples[idx]
        offsets = self._offsets_for_sample(sample)

        tensors = []
        kept_offsets = []

        with rasterio.open(sample["image_path"]) as src:
            for dx, dy in offsets:
                img = read_patch_from_src(
                    src,
                    sample["center_px"] + float(dx),
                    sample["center_py"] + float(dy),
                    self.patch_size_px,
                )
                if self.max_black_fraction < 1.0 or self.max_bright_fraction < 1.0:
                    black_fraction, bright_fraction = patch_quality_fractions(img)
                    is_low_quality = (
                        black_fraction > self.max_black_fraction
                        or bright_fraction > self.max_bright_fraction
                    )
                    # Grid bags are spatial tensors for convolutional MIL, so
                    # dropping candidates would scramble the 2D neighbourhood.
                    if is_low_quality and self.bag_layout != "grid":
                        continue
                tensors.append(self.transform(img))
                kept_offsets.append((float(dx), float(dy)))

            if not tensors:
                img = read_patch_from_src(src, sample["center_px"], sample["center_py"], self.patch_size_px)
                tensors.append(self.transform(img))
                kept_offsets.append((0.0, 0.0))

        while len(tensors) < self.bag_instances:
            tensors.append(tensors[-1].clone())
            kept_offsets.append(kept_offsets[-1])

        tensors = tensors[: self.bag_instances]
        kept_offsets = kept_offsets[: self.bag_instances]

        bag = torch.stack(tensors, dim=0)
        y = torch.tensor(sample["bag_label"], dtype=torch.float32)
        offset_tensor = torch.tensor(kept_offsets, dtype=torch.float32)
        return bag, y, torch.tensor(idx, dtype=torch.long), offset_tensor


def build_balanced_sampler(dataset: MILPointBagDataset):
    targets = np.asarray(dataset.targets(), dtype=np.int64)
    counts = np.bincount(targets, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[targets]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def is_conv_pooling(pooling: str) -> bool:
    return str(pooling).startswith("conv_")


def base_pooling_name(pooling: str) -> str:
    pooling = str(pooling)
    return pooling[5:] if is_conv_pooling(pooling) else pooling


class ConvolutionalMILHead(nn.Module):
    def __init__(self, feat_dim: int, bag_instances: int, kernel_size: int = 3):
        super().__init__()
        side = grid_side_for_instances(bag_instances)
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be a positive odd integer")
        if kernel_size > side:
            raise ValueError(
                f"conv_kernel_size={kernel_size} is larger than the {side}x{side} bag grid"
            )

        self.instance_head = nn.Linear(feat_dim, 1)
        self.side = side
        self.bag_instances = int(bag_instances)
        self.context_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="replicate",
            bias=True,
        )
        self.reset_context_conv()

    def reset_context_conv(self):
        nn.init.zeros_(self.context_conv.weight)
        nn.init.zeros_(self.context_conv.bias)
        center = self.context_conv.kernel_size[0] // 2
        with torch.no_grad():
            self.context_conv.weight[0, 0, center, center] = 1.0

    def forward(self, z, batch_size: int, bag_instances: int):
        if int(bag_instances) != self.bag_instances:
            raise ValueError(
                f"ConvolutionalMILHead was built for {self.bag_instances} instances, "
                f"but received {bag_instances}"
            )
        raw_logits = self.instance_head(z).view(batch_size, bag_instances)
        grid_logits = raw_logits.view(batch_size, 1, self.side, self.side)
        context_logits = self.context_conv(grid_logits).view(batch_size, bag_instances)
        return raw_logits, context_logits


def build_mil_head(feat_dim: int, cfg: Config, device):
    if is_conv_pooling(cfg.pooling):
        if cfg.bag_layout != "grid":
            raise ValueError("Convolutional pooling requires --bag_layout grid")
        head = ConvolutionalMILHead(
            feat_dim=feat_dim,
            bag_instances=cfg.bag_instances,
            kernel_size=cfg.conv_kernel_size,
        )
    else:
        head = nn.Linear(feat_dim, 1)
    return head.to(device)


def head_logits(head, z, batch_size: int, bag_instances: int):
    if isinstance(head, ConvolutionalMILHead):
        return head(z, batch_size, bag_instances)
    raw_logits = head(z).view(batch_size, bag_instances)
    return raw_logits, raw_logits


def pool_instance_logits(instance_logits, pooling: str, lse_tau: float, topk: int):
    pooling = base_pooling_name(pooling)
    if pooling == "max":
        return instance_logits.max(dim=1).values
    if pooling == "lse":
        tau = max(float(lse_tau), 1e-6)
        n = instance_logits.shape[1]
        return tau * torch.logsumexp(instance_logits / tau, dim=1) - tau * math.log(max(n, 1))
    if pooling == "topk":
        k = max(1, min(int(topk), instance_logits.shape[1]))
        return instance_logits.topk(k, dim=1).values.mean(dim=1)
    raise ValueError(f"Unknown pooling mode: {pooling}")


def forward_bags(model, head, bags, pooling: str, lse_tau: float, topk: int):
    b, n, c, h, w = bags.shape
    flat = bags.reshape(b * n, c, h, w)
    z = forward_features(model, flat)
    raw_instance_logits, scoring_instance_logits = head_logits(head, z, b, n)
    bag_logits = pool_instance_logits(scoring_instance_logits, pooling, lse_tau, topk)
    return bag_logits, scoring_instance_logits, raw_instance_logits


def metric_from_row(row, name):
    if name == "neg_val_loss":
        return -row["val_loss"]
    return row[name]


def binary_score_diagnostics(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    out = {
        "positive_class": "1",
        "positive_count": int(y_true.sum()),
        "negative_count": int((1 - y_true).sum()),
        "positive_prevalence": float(y_true.mean()) if len(y_true) else 0.0,
        "probability_column": "prob_1",
    }
    if len(np.unique(y_true)) == 2:
        out["average_precision"] = float(average_precision_score(y_true, y_score))
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["average_precision"] = None
        out["roc_auc"] = None

    df = pd.DataFrame({"y_true": y_true, "prob_1": y_score})
    grouped = df.groupby("y_true")["prob_1"].describe()
    grouped.index = grouped.index.map({0: "negative", 1: "positive"})
    out["score_summary_by_true_class"] = json.loads(grouped.to_json(orient="index"))
    if y_true.sum() and (1 - y_true).sum():
        out["positive_mean_minus_negative_mean"] = float(y_score[y_true == 1].mean() - y_score[y_true == 0].mean())
    return out


def run_epoch(
    model,
    head,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    cfg: Config,
    train: bool,
):
    model.train(train)
    head.train(train)

    losses = []
    y_true, y_pred, y_score = [], [], []

    desc = "Train" if train else "Val"
    iterator = tqdm(loader, desc=desc, dynamic_ncols=True)
    for bags, y, _, _ in iterator:
        bags = bags.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=(cfg.use_amp and device.type == "cuda")):
                bag_logits, _, _ = forward_bags(model, head, bags, cfg.pooling, cfg.lse_tau, cfg.topk)
                loss = criterion(bag_logits, y)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(bag_logits.detach())
        preds = (probs >= 0.5).long()

        losses.append(float(loss.detach().cpu().item()) * y.numel())
        y_true.extend(y.detach().cpu().long().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_score.extend(probs.detach().cpu().tolist())
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    total = max(1, len(y_true))
    metrics = {
        "loss": float(sum(losses) / total),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "pos_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["average_precision"] = float("nan")
        metrics["roc_auc"] = float("nan")
    return metrics


@torch.no_grad()
def evaluate_to_files(model, head, dataset, loader, device, cfg: Config):
    model.eval()
    head.eval()

    y_true, y_pred, y_score = [], [], []
    rows = []

    for bags, y, idxs, offsets in tqdm(loader, desc="Predict", dynamic_ncols=True):
        bags = bags.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bag_logits, instance_logits, raw_instance_logits = forward_bags(
            model, head, bags, cfg.pooling, cfg.lse_tau, cfg.topk
        )
        probs = torch.sigmoid(bag_logits)
        instance_probs = torch.sigmoid(instance_logits)
        raw_instance_probs = torch.sigmoid(raw_instance_logits)
        preds = (probs >= 0.5).long()
        best_instance = raw_instance_logits.argmax(dim=1).detach().cpu()
        best_context_instance = instance_logits.argmax(dim=1).detach().cpu()

        y_true_batch = y.detach().cpu().long()
        probs_cpu = probs.detach().cpu()
        preds_cpu = preds.detach().cpu()
        idxs_cpu = idxs.detach().cpu()
        offsets_cpu = offsets.detach().cpu()
        instance_probs_cpu = instance_probs.detach().cpu()
        raw_instance_probs_cpu = raw_instance_probs.detach().cpu()

        for j in range(len(y_true_batch)):
            sample = dataset.samples[int(idxs_cpu[j])]
            best_i = int(best_instance[j])
            best_context_i = int(best_context_instance[j])
            dx_px = float(offsets_cpu[j, best_i, 0])
            dy_px = float(offsets_cpu[j, best_i, 1])
            prob_1 = float(probs_cpu[j])
            best_instance_prob_1 = float(instance_probs_cpu[j, best_i])
            best_raw_instance_prob_1 = float(raw_instance_probs_cpu[j, best_i])
            true_label = int(y_true_batch[j])
            pred_label = int(preds_cpu[j])
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_score.append(prob_1)
            rows.append({
                "y_true": true_label,
                "y_pred": pred_label,
                "prob_0": 1.0 - prob_1,
                "prob_1": prob_1,
                "best_instance_prob_1": best_instance_prob_1,
                "best_raw_instance_prob_1": best_raw_instance_prob_1,
                "best_instance": best_i,
                "best_context_instance": best_context_i,
                "best_context_instance_prob_1": float(instance_probs_cpu[j, best_context_i]),
                "best_context_raw_instance_prob_1": float(raw_instance_probs_cpu[j, best_context_i]),
                "best_context_dx_px": float(offsets_cpu[j, best_context_i, 0]),
                "best_context_dy_px": float(offsets_cpu[j, best_context_i, 1]),
                "best_context_dx_m": float(offsets_cpu[j, best_context_i, 0]) * sample["pixel_size_x"],
                "best_context_dy_m": float(offsets_cpu[j, best_context_i, 1]) * sample["pixel_size_y"],
                "best_context_px": sample["center_px"] + float(offsets_cpu[j, best_context_i, 0]),
                "best_context_py": sample["center_py"] + float(offsets_cpu[j, best_context_i, 1]),
                "best_dx_px": dx_px,
                "best_dy_px": dy_px,
                "best_dx_m": dx_px * sample["pixel_size_x"],
                "best_dy_m": dy_px * sample["pixel_size_y"],
                "best_px": sample["center_px"] + dx_px,
                "best_py": sample["center_py"] + dy_px,
                "center_px": sample["center_px"],
                "center_py": sample["center_py"],
                "label": sample["label"],
                "folder": sample["folder"],
                "file": sample["file"],
                "image_path": sample["image_path"],
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(cfg.output_dir, "classifier_predictions.csv"), index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["0", "1"],
        output_dict=True,
        zero_division=0,
    )
    diagnostics = binary_score_diagnostics(y_true, y_score)

    with open(os.path.join(cfg.output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(cfg.output_dir, "binary_score_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("Binary score diagnostics:")
    print(json.dumps(diagnostics, indent=2))
    print(json.dumps(report, indent=2))


def save_mil_checkpoint(cfg, model, head, feat_dim, epoch, metric_value, name):
    encoder_path = os.path.join(cfg.output_dir, f"phase1_encoder_{name}.pth")
    head_path = os.path.join(cfg.output_dir, f"mil_head_{name}.pth")
    extra = {
        "epoch": int(epoch),
        "metric_name": cfg.monitor_metric,
        "metric_value": float(metric_value),
        "mil": True,
        "pooling": cfg.pooling,
        "bag_radius_m": cfg.bag_radius_m,
        "negative_bag_radius_m": cfg.negative_bag_radius_m,
        "bag_instances": cfg.bag_instances,
        "bag_layout": cfg.bag_layout,
        "conv_kernel_size": cfg.conv_kernel_size,
        "image_mode": cfg.image_mode,
        "eval_image_mode": cfg.eval_image_mode,
    }
    save_compatible_checkpoint(cfg.init_ckpt, model, encoder_path, extra)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "feat_dim": int(feat_dim),
            "classes": ["0", "1"],
            "class_to_idx": {"0": 0, "1": 1},
            "positive_class": "1",
            "epoch": int(epoch),
            "metric_name": cfg.monitor_metric,
            "metric_value": float(metric_value),
            "mil": True,
            "pooling": cfg.pooling,
            "bag_radius_m": cfg.bag_radius_m,
            "negative_bag_radius_m": cfg.negative_bag_radius_m,
            "bag_instances": cfg.bag_instances,
            "bag_layout": cfg.bag_layout,
            "conv_kernel_size": cfg.conv_kernel_size,
        },
        head_path,
    )


def parse_args():
    p = argparse.ArgumentParser(description="20 m multiple-instance binary classifier")
    p.add_argument("--init_ckpt", required=True)
    p.add_argument("--train_shp", required=True)
    p.add_argument("--val_shp", required=True)
    p.add_argument("--imagery_root", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--label_field", default="BinaryTree")
    p.add_argument("--folder_field", default="Folder")
    p.add_argument("--file_field", default="File")
    p.add_argument("--fx_field", default="fx")
    p.add_argument("--fy_field", default="fy")
    p.add_argument("--coord_mode", default="auto", choices=["auto", "normalized", "pixel", "world"])
    p.add_argument("--positive_class", default="1")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size_px", type=int, default=224)
    p.add_argument("--image_mode", default="rgb", choices=VALID_IMAGE_MODES)
    p.add_argument("--eval_image_mode", default="", choices=[""] + VALID_IMAGE_MODES)

    p.add_argument("--bag_radius_m", type=float, default=20.0)
    p.add_argument("--negative_bag_radius_m", type=float, default=0.0)
    p.add_argument("--bag_instances", type=int, default=17)
    p.add_argument("--bag_layout", default="rings", choices=["rings", "grid"])
    p.add_argument("--pooling", default="lse", choices=["max", "lse", "topk", "conv_max", "conv_lse", "conv_topk"])
    p.add_argument("--lse_tau", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--conv_kernel_size", type=int, default=3)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr_encoder", type=float, default=5e-7)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--freeze_encoder_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=0)

    p.add_argument("--max_black_fraction", type=float, default=1.0)
    p.add_argument("--max_bright_fraction", type=float, default=1.0)
    p.add_argument("--train_repeat_factor", type=int, default=1)
    p.add_argument("--balanced_sampler", action="store_true")
    p.add_argument("--no_pos_weight", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--monitor_metric",
        default="val_macro_f1",
        choices=["val_macro_f1", "val_weighted_f1", "val_pos_f1", "val_average_precision", "val_roc_auc", "neg_val_loss"],
    )
    p.add_argument("--save_every", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    eval_image_mode = args.eval_image_mode
    if not eval_image_mode:
        eval_image_mode = "rgb" if args.image_mode == "rgb_green_dropout" else args.image_mode

    cfg = Config(
        init_ckpt=args.init_ckpt,
        train_shp=args.train_shp,
        val_shp=args.val_shp,
        imagery_root=args.imagery_root,
        output_dir=args.output_dir,
        label_field=args.label_field,
        folder_field=args.folder_field,
        file_field=args.file_field,
        fx_field=args.fx_field,
        fy_field=args.fy_field,
        coord_mode=args.coord_mode,
        positive_class=args.positive_class,
        image_size=args.image_size,
        patch_size_px=args.patch_size_px,
        image_mode=args.image_mode,
        eval_image_mode=eval_image_mode,
        bag_radius_m=args.bag_radius_m,
        negative_bag_radius_m=args.negative_bag_radius_m,
        bag_instances=args.bag_instances,
        bag_layout=args.bag_layout,
        pooling=args.pooling,
        lse_tau=args.lse_tau,
        topk=args.topk,
        conv_kernel_size=args.conv_kernel_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        patience=args.patience,
        max_black_fraction=args.max_black_fraction,
        max_bright_fraction=args.max_bright_fraction,
        train_repeat_factor=args.train_repeat_factor,
        use_amp=not args.no_amp,
        use_pos_weight=not args.no_pos_weight,
        use_balanced_sampler=args.balanced_sampler,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        monitor_metric=args.monitor_metric,
        save_every=args.save_every,
    )

    safe_mkdir(cfg.output_dir)
    set_seed(cfg.seed)
    if is_conv_pooling(cfg.pooling):
        if cfg.bag_layout != "grid":
            raise ValueError("Convolutional MIL pooling requires --bag_layout grid")
        grid_side_for_instances(cfg.bag_instances)
        if cfg.conv_kernel_size <= 0 or cfg.conv_kernel_size % 2 == 0:
            raise ValueError("--conv_kernel_size must be a positive odd integer")
    with open(os.path.join(cfg.output_dir, "mil_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    folder_to_paths = build_tif_index(cfg.imagery_root)
    model, _ = load_encoder_from_checkpoint(cfg.init_ckpt, device)

    train_ds = MILPointBagDataset(
        shp_path=cfg.train_shp,
        imagery_root=cfg.imagery_root,
        label_field=cfg.label_field,
        folder_field=cfg.folder_field,
        file_field=cfg.file_field,
        fx_field=cfg.fx_field,
        fy_field=cfg.fy_field,
        coord_mode=cfg.coord_mode,
        positive_class=cfg.positive_class,
        patch_size_px=cfg.patch_size_px,
        transform=build_train_transform(cfg.image_size, cfg.image_mode),
        bag_radius_m=cfg.bag_radius_m,
        negative_bag_radius_m=cfg.negative_bag_radius_m,
        bag_instances=cfg.bag_instances,
        bag_layout=cfg.bag_layout,
        max_black_fraction=cfg.max_black_fraction,
        max_bright_fraction=cfg.max_bright_fraction,
        folder_to_paths=folder_to_paths,
        repeat_factor=cfg.train_repeat_factor,
        train=True,
    )

    val_ds = MILPointBagDataset(
        shp_path=cfg.val_shp,
        imagery_root=cfg.imagery_root,
        label_field=cfg.label_field,
        folder_field=cfg.folder_field,
        file_field=cfg.file_field,
        fx_field=cfg.fx_field,
        fy_field=cfg.fy_field,
        coord_mode=cfg.coord_mode,
        positive_class=cfg.positive_class,
        patch_size_px=cfg.patch_size_px,
        transform=build_eval_transform(cfg.image_size, cfg.eval_image_mode),
        bag_radius_m=cfg.bag_radius_m,
        negative_bag_radius_m=cfg.negative_bag_radius_m,
        bag_instances=cfg.bag_instances,
        bag_layout=cfg.bag_layout,
        max_black_fraction=cfg.max_black_fraction,
        max_bright_fraction=cfg.max_bright_fraction,
        folder_to_paths=folder_to_paths,
        repeat_factor=1,
        train=False,
    )

    train_sampler = build_balanced_sampler(train_ds) if cfg.use_balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    feat_dim = infer_feature_dim(model, device, cfg.image_size)
    head = build_mil_head(feat_dim, cfg, device)

    targets = np.asarray(train_ds.targets(), dtype=np.float32)
    pos_count = max(1.0, float(targets.sum()))
    neg_count = max(1.0, float(len(targets) - targets.sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device) if cfg.use_pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr_encoder},
            {"params": head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda")) if device.type == "cuda" else None

    print("=" * 100)
    print("MIL binary training setup")
    print(f"Init checkpoint       : {cfg.init_ckpt}")
    print(f"Train bags            : {len(train_ds)}")
    print(f"Val bags              : {len(val_ds)}")
    print(f"Positive radius (m)   : {cfg.bag_radius_m}")
    print(f"Negative radius (m)   : {cfg.negative_bag_radius_m}")
    print(f"Bag instances         : {cfg.bag_instances}")
    print(f"Bag layout            : {cfg.bag_layout}")
    print(f"Pooling               : {cfg.pooling}")
    if is_conv_pooling(cfg.pooling):
        print(f"Conv kernel size      : {cfg.conv_kernel_size}")
    print(f"Train image mode      : {cfg.image_mode}")
    print(f"Eval image mode       : {cfg.eval_image_mode}")
    print(f"Balanced sampler      : {cfg.use_balanced_sampler}")
    print(f"Positive loss weight  : {pos_weight.detach().cpu().tolist() if pos_weight is not None else None}")
    print("=" * 100)

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        set_encoder_trainable(model, epoch > cfg.freeze_encoder_epochs)

        train_metrics = run_epoch(model, head, train_loader, criterion, optimizer, scaler, device, cfg, train=True)
        val_metrics = run_epoch(model, head, val_loader, criterion, None, None, device, cfg, train=False)

        row = {
            "epoch": epoch,
            "encoder_frozen": int(epoch <= cfg.freeze_encoder_epochs),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_pos_f1": train_metrics["pos_f1"],
            "train_average_precision": train_metrics["average_precision"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_pos_f1": val_metrics["pos_f1"],
            "val_average_precision": val_metrics["average_precision"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(row)
        current_metric = metric_from_row(row, cfg.monitor_metric)
        scheduler.step(current_metric)

        print(
            f"Epoch {epoch:03d} | frozen={row['encoder_frozen']} | "
            f"train loss {row['train_loss']:.4f} macro_f1 {row['train_macro_f1']:.4f} "
            f"pos_f1 {row['train_pos_f1']:.4f} pr_auc {row['train_average_precision']:.4f} | "
            f"val loss {row['val_loss']:.4f} macro_f1 {row['val_macro_f1']:.4f} "
            f"pos_f1 {row['val_pos_f1']:.4f} pr_auc {row['val_average_precision']:.4f} "
            f"roc_auc {row['val_roc_auc']:.4f}"
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0
            save_mil_checkpoint(cfg, model, head, feat_dim, epoch, current_metric, "best")
            print(f"[INFO] Saved best MIL checkpoint at epoch {epoch}: {cfg.monitor_metric}={current_metric:.6f}")
        else:
            bad_epochs += 1

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_mil_checkpoint(cfg, model, head, feat_dim, epoch, current_metric, f"epoch_{epoch:03d}")

        pd.DataFrame(history).to_csv(os.path.join(cfg.output_dir, "training_history.csv"), index=False)

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[INFO] Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    last_metric = metric_from_row(history[-1], cfg.monitor_metric)
    save_mil_checkpoint(cfg, model, head, feat_dim, history[-1]["epoch"], last_metric, "last")

    best_encoder_path = os.path.join(cfg.output_dir, "phase1_encoder_best.pth")
    best_head_path = os.path.join(cfg.output_dir, "mil_head_best.pth")
    best_model, _ = load_encoder_from_checkpoint(best_encoder_path, device)
    best_head = build_mil_head(feat_dim, cfg, device)
    best_head.load_state_dict(torch.load(best_head_path, map_location=device)["head_state_dict"])
    evaluate_to_files(best_model, best_head, val_ds, val_loader, device, cfg)

    elapsed = time.time() - start
    print("=" * 100)
    print("MIL training completed")
    print(f"Best epoch      : {best_epoch}")
    print(f"Best metric     : {best_metric:.6f}")
    print(f"Total elapsed   : {elapsed / 3600:.2f} hours")
    print(f"Predictions CSV : {os.path.join(cfg.output_dir, 'classifier_predictions.csv')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
