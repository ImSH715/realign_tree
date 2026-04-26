"""
Supervised fine-tuning for the LeJEPA encoder using GT tree labels.

This script:
- loads a pretrained/self-supervised encoder checkpoint,
- reads GT points from SHP,
- crops patches centered at fx/fy,
- trains a classifier head with cross-entropy,
- updates the encoder so the feature space becomes species-aware,
- saves the best encoder checkpoint for Phase 2 / Phase 3.

Use this when self-supervised embeddings collapse and class prototypes are not
separable.
"""

import argparse
import os
import json
import time
from dataclasses import asdict, dataclass

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

    image_size: int = 224
    patch_size_px: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr_encoder: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_tif_path(imagery_root, folder, filename):
    candidates = [
        os.path.join(imagery_root, str(folder), str(filename)),
        os.path.join(imagery_root, str(filename)),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    basename = os.path.basename(str(filename))
    for root, _, files in os.walk(imagery_root):
        if basename in files:
            return os.path.join(root, basename)

    raise FileNotFoundError(f"Could not find TIFF: folder={folder}, file={filename}")


def read_patch(image_path, cx, cy, patch_size):
    half = patch_size // 2

    with rasterio.open(image_path) as src:
        col0 = int(round(cx)) - half
        row0 = int(round(cy)) - half

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
        class_to_idx=None,
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

        if class_to_idx is None:
            classes = sorted(self.gdf[label_field].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]

        self.rows = []
        for _, row in self.gdf.iterrows():
            label = str(row[label_field]).strip()
            if label not in self.class_to_idx:
                continue

            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        image_path = resolve_tif_path(
            self.imagery_root,
            row[self.folder_field],
            row[self.file_field],
        )

        patch = read_patch(
            image_path=image_path,
            cx=float(row[self.fx_field]),
            cy=float(row[self.fy_field]),
            patch_size=self.patch_size_px,
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
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
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
        for key in ["model", "model_state_dict", "encoder", "encoder_state_dict", "state_dict"]:
            if key in updated and isinstance(updated[key], dict):
                updated[key] = state
                replaced = True
                break

        if not replaced:
            updated["model_state_dict"] = state

        updated["supervised_finetune"] = extra
        torch.save(updated, output_path)
    else:
        torch.save(
            {"model_state_dict": state, "supervised_finetune": extra},
            output_path,
        )


def run_epoch(model, head, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    head.train(train)

    losses = []
    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
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

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def parse_args():
    parser = argparse.ArgumentParser()

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

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size_px", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
        image_size=args.image_size,
        patch_size_px=args.patch_size_px,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    set_seed(cfg.seed)

    with open(os.path.join(cfg.output_dir, "supervised_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model, _ = load_encoder_from_checkpoint(cfg.init_ckpt, device)

    train_tf = build_train_transform(cfg.image_size)
    val_tf = build_eval_transform(cfg.image_size)

    train_ds = GTPointDataset(
        shp_path=cfg.train_shp,
        imagery_root=cfg.imagery_root,
        label_field=cfg.label_field,
        folder_field=cfg.folder_field,
        file_field=cfg.file_field,
        fx_field=cfg.fx_field,
        fy_field=cfg.fy_field,
        patch_size_px=cfg.patch_size_px,
        transform=train_tf,
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
        transform=val_tf,
        class_to_idx=train_ds.class_to_idx,
    )

    print("Classes:", train_ds.classes)
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    with open(os.path.join(cfg.output_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(train_ds.class_to_idx, f, indent=2, ensure_ascii=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feat_dim = infer_feature_dim(model, device, cfg.image_size)
    head = nn.Linear(feat_dim, len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr_encoder},
            {"params": head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_f1 = -1
    history = []

    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(
            model, head, train_loader, criterion, optimizer, scaler, device, train=True
        )
        val_metrics = run_epoch(
            model, head, val_loader, criterion, optimizer, None, device, train=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} f1 {row['train_macro_f1']:.4f} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.4f} f1 {row['val_macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]

            save_compatible_checkpoint(
                init_ckpt=cfg.init_ckpt,
                model=model,
                output_path=os.path.join(cfg.output_dir, "phase1_encoder_best.pth"),
                extra={
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "classes": train_ds.classes,
                    "class_to_idx": train_ds.class_to_idx,
                },
            )

            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "classes": train_ds.classes,
                    "class_to_idx": train_ds.class_to_idx,
                    "feat_dim": feat_dim,
                },
                os.path.join(cfg.output_dir, "classifier_head_best.pth"),
            )

    import pandas as pd
    pd.DataFrame(history).to_csv(os.path.join(cfg.output_dir, "training_history.csv"), index=False)

    val_final = run_epoch(model, head, val_loader, criterion, None, device, train=False)

    report = classification_report(
        val_final["y_true"],
        val_final["y_pred"],
        target_names=train_ds.classes,
        zero_division=0,
        output_dict=True,
    )

    with open(os.path.join(cfg.output_dir, "val_classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Supervised fine-tuning completed")
    print(f"Best val macro F1: {best_f1:.4f}")
    print(f"Saved encoder: {os.path.join(cfg.output_dir, 'phase1_encoder_best.pth')}")
    print(f"Elapsed: {time.time() - start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()