"""
Phase 1.5: supervised fine-tuning of Phase 1 encoder using GT tree labels.
"""

import argparse
import os
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.models.checkpoint import load_encoder_from_checkpoint


@dataclass
class Config:
    init_ckpt: str
    train_shp: str
    val_shp: str
    imagery_root: str
    output_dir: str

    label_field: str = "Tree"
    folder_field: str = "Folder"
    file_field: str = "File"
    fx_field: str = "fx"
    fy_field: str = "fy"
    coord_mode: str = "auto"

    image_size: int = 224
    patch_size_px: int = 224
    batch_size: int = 16
    epochs: int = 50

    lr_encoder: float = 1e-6
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    freeze_encoder_epochs: int = 3
    patience: int = 10

    num_workers: int = 0
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True

    use_class_weights: bool = True
    use_balanced_sampler: bool = False

    save_every: int = 1
    monitor_metric: str = "val_macro_f1"
    debug_patches: int = 32


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name).strip()))[0].lower()


def build_tif_index(imagery_root: str) -> Dict[str, List[str]]:
    folder_to_paths: Dict[str, List[str]] = {}

    print("[INFO] Building TIFF index...")
    for root, _, files in os.walk(imagery_root):
        tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        if not tif_files:
            continue

        rel = os.path.relpath(root, imagery_root)
        parts = rel.split(os.sep)

        folder_key = None
        for p in parts:
            if p.startswith("2023-"):
                folder_key = p
                break

        if folder_key is None:
            folder_key = parts[0] if parts and parts[0] != "." else ""

        folder_to_paths.setdefault(folder_key, [])
        for f in tif_files:
            folder_to_paths[folder_key].append(os.path.join(root, f))

    total = sum(len(v) for v in folder_to_paths.values())
    print(f"[INFO] Indexed TIFF folders: {len(folder_to_paths)}")
    print(f"[INFO] Indexed TIFF files  : {total}")

    if total == 0:
        raise RuntimeError(f"No TIFF files found under imagery_root: {imagery_root}")

    return folder_to_paths


def resolve_tif_path_fast(folder_to_paths: Dict[str, List[str]], folder, filename) -> str:
    folder = str(folder).strip()
    stem = normalize_stem(filename)

    if folder not in folder_to_paths:
        available = sorted(folder_to_paths.keys())[:30]
        raise FileNotFoundError(
            f"Folder key not found: {folder}. Example indexed folders: {available}"
        )

    paths = folder_to_paths[folder]
    exact, contains, reverse_contains = [], [], []

    for p in paths:
        tif_stem = normalize_stem(p)

        if tif_stem == stem:
            exact.append(p)
        elif stem in tif_stem:
            contains.append(p)
        elif tif_stem in stem:
            reverse_contains.append(p)

    if exact:
        return sorted(exact, key=len)[0]
    if contains:
        return sorted(contains, key=len)[0]
    if reverse_contains:
        return sorted(reverse_contains, key=len)[0]

    raise FileNotFoundError(
        f"No matching TIFF in folder={folder} for file stem={stem}. "
        f"Folder TIFF count={len(paths)}. "
        f"First files={[os.path.basename(p) for p in paths[:10]]}"
    )


def convert_to_pixel(src, x, y, coord_mode: str) -> Tuple[float, float, str]:
    x = float(x)
    y = float(y)

    if coord_mode == "pixel":
        return x, y, "pixel"

    if coord_mode == "normalized":
        return x * src.width, y * src.height, "normalized"

    if coord_mode == "world":
        row, col = src.index(x, y)
        return float(col), float(row), "world"

    if coord_mode == "auto":
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * src.width, y * src.height, "normalized"

        if 0.0 <= x < src.width and 0.0 <= y < src.height:
            return x, y, "pixel"

        row, col = src.index(x, y)
        return float(col), float(row), "world"

    raise ValueError("coord_mode must be one of: auto, normalized, pixel, world")


def read_patch(image_path, x, y, patch_size, coord_mode="auto", return_debug=False):
    half = patch_size // 2

    with rasterio.open(image_path) as src:
        px, py, used_mode = convert_to_pixel(src, x, y, coord_mode)

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

    img = Image.fromarray(arr)

    if return_debug:
        return img, {
            "pixel_x": float(px),
            "pixel_y": float(py),
            "coord_mode_used": used_mode,
            "col0": int(col0),
            "row0": int(row0),
        }

    return img


class GTPointDataset(Dataset):
    def __init__(
        self,
        shp_path,
        imagery_root,
        label_field,
        folder_field,
        file_field,
        fx_field,
        fy_field,
        patch_size_px,
        transform,
        coord_mode="auto",
        class_to_idx=None,
        folder_to_paths=None,
        debug_dir=None,
        debug_patches=0,
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
        self.patch_size_px = patch_size_px
        self.transform = transform
        self.coord_mode = coord_mode
        self.debug_dir = debug_dir
        self.debug_patches = debug_patches

        required = [label_field, folder_field, file_field, fx_field, fy_field]
        for c in required:
            if c not in self.gdf.columns:
                raise ValueError(f"Missing required field '{c}'. Available: {self.gdf.columns.tolist()}")

        if class_to_idx is None:
            classes = sorted(self.gdf[label_field].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]

        if folder_to_paths is None:
            folder_to_paths = build_tif_index(imagery_root)

        self.folder_to_paths = folder_to_paths
        self.rows = []
        self.failed_rows = []

        print(f"[INFO] Resolving TIFF paths for {os.path.basename(shp_path)}...")
        iterator = tqdm(self.gdf.iterrows(), total=len(self.gdf), dynamic_ncols=True, desc="Resolving")

        for i, (_, row) in enumerate(iterator):
            label = str(row[label_field]).strip()
            if label not in self.class_to_idx:
                continue

            try:
                image_path = resolve_tif_path_fast(
                    self.folder_to_paths,
                    row[self.folder_field],
                    row[self.file_field],
                )

                rec = row.copy()
                rec["_image_path"] = image_path
                self.rows.append(rec)

            except Exception as e:
                self.failed_rows.append((i, str(e)))
                if len(self.failed_rows) <= 10:
                    print(f"[WARN] Failed to resolve row {i}: {e}")

        print(f"[INFO] Resolved samples: {len(self.rows)}")
        print(f"[INFO] Failed samples  : {len(self.failed_rows)}")

        if len(self.rows) == 0:
            raise RuntimeError("No usable samples after TIFF path resolution.")

        if self.debug_dir and self.debug_patches > 0:
            self.save_debug_patches()

    def __len__(self):
        return len(self.rows)

    def label_counts(self):
        labels = [str(r[self.label_field]).strip() for r in self.rows]
        return pd.Series(labels).value_counts().to_dict()

    def targets(self):
        return [self.class_to_idx[str(r[self.label_field]).strip()] for r in self.rows]

    def save_debug_patches(self):
        safe_mkdir(self.debug_dir)
        rows = []

        n = min(self.debug_patches, len(self.rows))
        print(f"[INFO] Saving {n} debug patches to {self.debug_dir}")

        for idx in range(n):
            row = self.rows[idx]
            label = str(row[self.label_field]).strip()
            image_path = row["_image_path"]

            try:
                patch, info = read_patch(
                    image_path=image_path,
                    x=float(row[self.fx_field]),
                    y=float(row[self.fy_field]),
                    patch_size=self.patch_size_px,
                    coord_mode=self.coord_mode,
                    return_debug=True,
                )

                safe_label = label.replace(" ", "_").replace("/", "_")
                out_name = f"{idx:04d}_{safe_label}.png"
                out_path = os.path.join(self.debug_dir, out_name)
                patch.save(out_path)

                rows.append({
                    "idx": idx,
                    "label": label,
                    "folder": str(row[self.folder_field]),
                    "file": str(row[self.file_field]),
                    "image_path": image_path,
                    "raw_x": float(row[self.fx_field]),
                    "raw_y": float(row[self.fy_field]),
                    **info,
                    "debug_patch": out_path,
                })

            except Exception as e:
                rows.append({
                    "idx": idx,
                    "label": label,
                    "error": str(e),
                })

        pd.DataFrame(rows).to_csv(
            os.path.join(self.debug_dir, "debug_patches.csv"),
            index=False,
        )

    def __getitem__(self, idx):
        row = self.rows[idx]

        patch = read_patch(
            image_path=row["_image_path"],
            x=float(row[self.fx_field]),
            y=float(row[self.fy_field]),
            patch_size=self.patch_size_px,
            coord_mode=self.coord_mode,
        )

        x = self.transform(patch)
        label = str(row[self.label_field]).strip()
        y = self.class_to_idx[label]

        return x, torch.tensor(y, dtype=torch.long)


def build_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def forward_features(model, x):
    if hasattr(model, "encode"):
        out = model.encode(x)
    else:
        out = model(x)

    if isinstance(out, dict):
        for k in ["features", "embedding", "embeddings", "x", "last_hidden_state"]:
            if k in out:
                out = out[k]
                break

    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.ndim == 4:
        out = out.mean(dim=(2, 3))
    elif out.ndim == 3:
        out = out[:, 0]

    return out


def infer_feature_dim(model, device, image_size):
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size).to(device)
        z = forward_features(model, x)
    return int(z.shape[1])


def save_compatible_checkpoint(init_ckpt, model, output_path, extra):
    try:
        ckpt = torch.load(init_ckpt, map_location="cpu")
    except Exception:
        ckpt = {}

    state = model.state_dict()

    if isinstance(ckpt, dict):
        updated = dict(ckpt)

        replaced = False
        for key in ["model_state", "model", "model_state_dict", "encoder", "encoder_state_dict", "state_dict"]:
            if key in updated and isinstance(updated[key], dict):
                updated[key] = state
                replaced = True
                break

        if not replaced:
            updated["model_state"] = state

        updated["supervised_finetune"] = extra
        torch.save(updated, output_path)
    else:
        torch.save({"model_state": state, "supervised_finetune": extra}, output_path)


def save_checkpoint_bundle(cfg, model, head, feat_dim, epoch, metric_value, classes, class_to_idx, name):
    encoder_path = os.path.join(cfg.output_dir, f"phase1_encoder_{name}.pth")
    head_path = os.path.join(cfg.output_dir, f"classifier_head_{name}.pth")

    extra = {
        "epoch": int(epoch),
        "metric_name": cfg.monitor_metric,
        "metric_value": float(metric_value),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "coord_mode": cfg.coord_mode,
        "supervised": True,
    }

    save_compatible_checkpoint(
        init_ckpt=cfg.init_ckpt,
        model=model,
        output_path=encoder_path,
        extra=extra,
    )

    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "classes": classes,
            "class_to_idx": class_to_idx,
            "feat_dim": feat_dim,
            "epoch": int(epoch),
            "metric_name": cfg.monitor_metric,
            "metric_value": float(metric_value),
        },
        head_path,
    )


def compute_class_weights(dataset, num_classes, device):
    targets = np.array(dataset.targets(), dtype=np.int64)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0

    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_balanced_sampler(dataset):
    targets = np.array(dataset.targets(), dtype=np.int64)
    counts = np.bincount(targets)
    counts[counts == 0] = 1

    sample_weights = 1.0 / counts[targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def set_encoder_trainable(model, trainable: bool):
    for p in model.parameters():
        p.requires_grad = trainable


def print_prediction_distribution(y_true, y_pred, classes, prefix):
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    print(f"[{prefix}] True counts by index:", dict(true_counts))
    print(f"[{prefix}] Pred counts by index:", dict(pred_counts))

    print(f"[{prefix}] True counts by class:")
    for idx, count in sorted(true_counts.items()):
        cls = classes[idx] if idx < len(classes) else str(idx)
        print(f"  {idx:02d} {cls}: {count}")

    print(f"[{prefix}] Pred counts by class:")
    for idx, count in sorted(pred_counts.items()):
        cls = classes[idx] if idx < len(classes) else str(idx)
        print(f"  {idx:02d} {cls}: {count}")


def run_epoch(
    model,
    head,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
    classes,
    train=True,
    epoch=0,
    print_dist=False,
):
    model.train(train)
    head.train(train)

    losses = []
    y_true = []
    y_pred = []

    desc = f"Train {epoch:03d}" if train else f"Val {epoch:03d}"
    iterator = tqdm(loader, desc=desc, dynamic_ncols=True)

    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z = forward_features(model, x)
                logits = head(z)
                loss = criterion(logits, y)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(float(loss.detach().cpu()))

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())

        running_acc = accuracy_score(y_true, y_pred)
        iterator.set_postfix(loss=f"{np.mean(losses):.4f}", acc=f"{running_acc:.4f}")

    if print_dist:
        prefix = "TRAIN" if train else "VAL"
        print_prediction_distribution(y_true, y_pred, classes, prefix=f"{prefix} epoch {epoch}")

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1.5 supervised fine-tuning")

    parser.add_argument("--init_ckpt", required=True)
    parser.add_argument("--train_shp", required=True)
    parser.add_argument("--val_shp", required=True)
    parser.add_argument("--imagery_root", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--label_field", default="Tree")
    parser.add_argument("--folder_field", default="Folder")
    parser.add_argument("--file_field", default="File")
    parser.add_argument("--fx_field", default="fx")
    parser.add_argument("--fy_field", default="fy")
    parser.add_argument("--coord_mode", default="auto", choices=["auto", "normalized", "pixel", "world"])

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size_px", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--lr_encoder", type=float, default=1e-6)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--freeze_encoder_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--monitor_metric",
        default="val_macro_f1",
        choices=["val_macro_f1", "val_weighted_f1", "val_accuracy", "neg_val_loss"],
    )

    parser.add_argument("--debug_patches", type=int, default=32)

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--balanced_sampler", action="store_true")

    parser.add_argument("--print_train_dist", action="store_true")
    parser.add_argument("--print_val_dist", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    safe_mkdir(args.output_dir)

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
        image_size=args.image_size,
        patch_size_px=args.patch_size_px,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        use_amp=not args.no_amp,
        use_class_weights=not args.no_class_weights,
        use_balanced_sampler=args.balanced_sampler,
        save_every=args.save_every,
        monitor_metric=args.monitor_metric,
        debug_patches=args.debug_patches,
    )

    set_seed(cfg.seed)

    with open(os.path.join(cfg.output_dir, "supervised_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"

    folder_to_paths = build_tif_index(cfg.imagery_root)
    model, _ = load_encoder_from_checkpoint(cfg.init_ckpt, device)

    debug_root = os.path.join(cfg.output_dir, "debug_patches")

    train_ds = GTPointDataset(
        shp_path=cfg.train_shp,
        imagery_root=cfg.imagery_root,
        label_field=cfg.label_field,
        folder_field=cfg.folder_field,
        file_field=cfg.file_field,
        fx_field=cfg.fx_field,
        fy_field=cfg.fy_field,
        patch_size_px=cfg.patch_size_px,
        transform=build_train_transform(cfg.image_size),
        coord_mode=cfg.coord_mode,
        folder_to_paths=folder_to_paths,
        debug_dir=os.path.join(debug_root, "train"),
        debug_patches=cfg.debug_patches,
    )

    val_ds = GTPointDataset(
        shp_path=cfg.val_shp,
        imagery_root=cfg.imagery_root,
        label_field=cfg.label_field,
        folder_field=cfg.folder_field,
        file_field=cfg.file_field,
        fx_field=cfg.fx_field,
        fy_field=cfg.fy_field,
        patch_size_px=cfg.patch_size_px,
        transform=build_eval_transform(cfg.image_size),
        coord_mode=cfg.coord_mode,
        class_to_idx=train_ds.class_to_idx,
        folder_to_paths=folder_to_paths,
        debug_dir=os.path.join(debug_root, "val"),
        debug_patches=cfg.debug_patches,
    )

    with open(os.path.join(cfg.output_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(train_ds.class_to_idx, f, indent=2, ensure_ascii=False)

    with open(os.path.join(cfg.output_dir, "label_counts.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_ds.label_counts(),
                "val": val_ds.label_counts(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("Phase 1.5 supervised fine-tuning setup")
    print("Classes:", train_ds.classes)
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Coord mode:", cfg.coord_mode)
    print("AMP enabled:", use_amp)
    print("Class weights:", cfg.use_class_weights)
    print("Balanced sampler:", cfg.use_balanced_sampler)
    print("Debug patches:", debug_root)
    print("=" * 80)

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
    head = nn.Linear(feat_dim, len(train_ds.classes)).to(device)

    if cfg.use_class_weights:
        class_weights = compute_class_weights(train_ds, len(train_ds.classes), device)
        print("Class weights:", class_weights.detach().cpu().numpy().tolist())
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr_encoder},
            {"params": head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if epoch <= cfg.freeze_encoder_epochs:
            set_encoder_trainable(model, False)
        else:
            set_encoder_trainable(model, True)

        train_metrics = run_epoch(
            model=model,
            head=head,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            classes=train_ds.classes,
            train=True,
            epoch=epoch,
            print_dist=args.print_train_dist,
        )

        val_metrics = run_epoch(
            model=model,
            head=head,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            use_amp=False,
            classes=train_ds.classes,
            train=False,
            epoch=epoch,
            print_dist=args.print_val_dist,
        )

        row = {
            "epoch": epoch,
            "encoder_frozen": int(epoch <= cfg.freeze_encoder_epochs),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }

        history.append(row)

        if cfg.monitor_metric == "neg_val_loss":
            current_metric = -row["val_loss"]
        else:
            current_metric = row[cfg.monitor_metric]

        scheduler.step(current_metric)

        print(
            f"Epoch {epoch:03d} | frozen={row['encoder_frozen']} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} "
            f"macro_f1 {row['train_macro_f1']:.4f} weighted_f1 {row['train_weighted_f1']:.4f} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.4f} "
            f"macro_f1 {row['val_macro_f1']:.4f} weighted_f1 {row['val_weighted_f1']:.4f}"
        )

        improved = current_metric > best_metric

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0

            save_checkpoint_bundle(
                cfg=cfg,
                model=model,
                head=head,
                feat_dim=feat_dim,
                epoch=epoch,
                metric_value=current_metric,
                classes=train_ds.classes,
                class_to_idx=train_ds.class_to_idx,
                name="best",
            )

            print(f"[INFO] Saved best checkpoint at epoch {epoch}: {cfg.monitor_metric}={current_metric:.6f}")
        else:
            bad_epochs += 1

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint_bundle(
                cfg=cfg,
                model=model,
                head=head,
                feat_dim=feat_dim,
                epoch=epoch,
                metric_value=current_metric,
                classes=train_ds.classes,
                class_to_idx=train_ds.class_to_idx,
                name=f"epoch_{epoch:03d}",
            )

        pd.DataFrame(history).to_csv(
            os.path.join(cfg.output_dir, "training_history.csv"),
            index=False,
        )

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[INFO] Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    save_checkpoint_bundle(
        cfg=cfg,
        model=model,
        head=head,
        feat_dim=feat_dim,
        epoch=history[-1]["epoch"],
        metric_value=history[-1][cfg.monitor_metric] if cfg.monitor_metric != "neg_val_loss" else -history[-1]["val_loss"],
        classes=train_ds.classes,
        class_to_idx=train_ds.class_to_idx,
        name="last",
    )

    val_final = run_epoch(
        model=model,
        head=head,
        loader=val_loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        device=device,
        use_amp=False,
        classes=train_ds.classes,
        train=False,
        epoch=history[-1]["epoch"],
        print_dist=True,
    )

    report = classification_report(
        val_final["y_true"],
        val_final["y_pred"],
        labels=list(range(len(train_ds.classes))),
        target_names=train_ds.classes,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(
        val_final["y_true"],
        val_final["y_pred"],
        labels=list(range(len(train_ds.classes))),
    )

    with open(os.path.join(cfg.output_dir, "val_classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    pd.DataFrame(cm, index=train_ds.classes, columns=train_ds.classes).to_csv(
        os.path.join(cfg.output_dir, "val_confusion_matrix.csv")
    )

    print("=" * 80)
    print("Phase 1.5 supervised fine-tuning completed")
    print(f"Best monitored metric: {cfg.monitor_metric} = {best_metric:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved best encoder: {os.path.join(cfg.output_dir, 'phase1_encoder_best.pth')}")
    print(f"Saved last encoder: {os.path.join(cfg.output_dir, 'phase1_encoder_last.pth')}")
    print(f"Elapsed: {time.time() - start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()