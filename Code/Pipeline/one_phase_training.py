import os
import csv
import math
import time
import json
import glob
import copy
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm with: pip install timm") from e

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n


def recursive_find_tif_files(root_dir: str) -> List[str]:
    patterns = ["**/*.tif", "**/*.TIF", "**/*.tiff", "**/*.TIFF"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    files = sorted(list(set(files)))
    return files


def safe_open_image(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise RuntimeError(f"Failed to open image: {path} | {e}") from e


def infer_label_from_parent_folder(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def load_label_map_csv(label_csv: str) -> Dict[str, str]:
    mapping = {}
    with open(label_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Label CSV must contain columns: path,label")
        for row in reader:
            mapping[os.path.abspath(row["path"])] = str(row["label"])
    return mapping


class RecursiveTifDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        label_csv: Optional[str] = None,
        unlabeled_ok: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.unlabeled_ok = unlabeled_ok

        self.paths = recursive_find_tif_files(root_dir)
        if len(self.paths) == 0:
            raise RuntimeError(f"No .tif or .tiff files found under: {root_dir}")

        self.paths = [os.path.abspath(p) for p in self.paths]

        self.label_map = None
        if label_csv is not None:
            self.label_map = load_label_map_csv(label_csv)

        label_names = []
        samples = []

        for path in self.paths:
            if self.label_map is not None:
                label_name = self.label_map.get(path, None)
                if label_name is None:
                    if unlabeled_ok:
                        label_name = "unlabeled"
                    else:
                        continue
            else:
                label_name = infer_label_from_parent_folder(path)

            label_names.append(label_name)
            samples.append((path, label_name))

        if len(samples) == 0:
            raise RuntimeError("No valid samples were collected after label assignment.")

        unique_labels = sorted(list(set(label_names)))
        self.class_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        self.samples = [(path, self.class_to_idx[label_name]) for path, label_name in samples]
        self.classes = unique_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = safe_open_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path


class RecursiveTifMultiCropDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        label_csv: Optional[str] = None,
        image_size_global: int = 224,
        image_size_local: int = 96,
        num_global_views: int = 2,
        num_local_views: int = 4,
    ) -> None:
        self.base_dataset = RecursiveTifDataset(
            root_dir=root_dir,
            transform=None,
            label_csv=label_csv,
            unlabeled_ok=False,
        )
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        common_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
            normalize,
        ]

        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size_global,
                    scale=(0.30, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                *common_aug,
            ]
        )

        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size_local,
                    scale=(0.05, 0.30),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                *common_aug,
            ]
        )

        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        self.idx_to_class = self.base_dataset.idx_to_class

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        _, target, path = self.base_dataset.samples[idx]
        image = safe_open_image(path)

        views = []
        for _ in range(self.num_global_views):
            views.append(self.global_transform(image))
        for _ in range(self.num_local_views):
            views.append(self.local_transform(image))

        return views, target, path


def multicrop_collate_fn(batch):
    all_views = list(zip(*[item[0] for item in batch]))
    stacked_views = [torch.stack(v, dim=0) for v in all_views]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    paths = [item[2] for item in batch]
    return stacked_views, labels, paths


def build_supervised_transform(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.6, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ViTBackbone(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        if hasattr(self.model, "num_features"):
            self.embed_dim = self.model.num_features
        elif hasattr(self.model, "embed_dim"):
            self.embed_dim = self.model.embed_dim
        else:
            raise ValueError("Could not infer embedding dimension from the backbone.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.forward_features(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if feat.ndim == 3:
            if hasattr(self.model, "num_prefix_tokens") and self.model.num_prefix_tokens > 0:
                return feat[:, 0]
            return feat.mean(dim=1)

        return feat


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeJEPALikeModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 512,
        backbone_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ViTBackbone(backbone_name, pretrained=backbone_pretrained)
        self.projector = MLPProjector(
            in_dim=self.backbone.embed_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
        )
        self.predictor = nn.Sequential(
            nn.Linear(projector_out_dim, projector_out_dim),
            nn.GELU(),
            nn.Linear(projector_out_dim, projector_out_dim),
        )
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project(self, feat: torch.Tensor) -> torch.Tensor:
        return self.projector(feat)

    def predict(self, proj: torch.Tensor) -> torch.Tensor:
        return self.predictor(proj)

    def classify(self, feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(feat)

    def forward(self, x: torch.Tensor):
        feat = self.encode(x)
        proj = self.project(feat)
        pred = self.predict(proj)
        logits = self.classify(feat)
        return feat, proj, pred, logits


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("Input must be a square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SliceRegularization(nn.Module):
    def __init__(self, num_slices: int = 256):
        super().__init__()
        self.num_slices = num_slices

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        device = z.device
        d = z.size(1)

        directions = torch.randn(self.num_slices, d, device=device)
        directions = F.normalize(directions, dim=1)

        projections = z @ directions.t()
        mean_term = projections.mean(dim=0).pow(2).mean()
        std_term = F.relu(1.0 - projections.std(dim=0)).pow(2).mean()

        return mean_term + std_term


class LeJEPALikeLoss(nn.Module):
    def __init__(
        self,
        align_weight: float = 1.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        slice_weight: float = 1.0,
        num_slices: int = 256,
    ) -> None:
        super().__init__()
        self.align_weight = align_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.slice_weight = slice_weight
        self.slice_reg = SliceRegularization(num_slices=num_slices)

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.size(0) - 1)
        return off_diagonal(cov).pow(2).sum() / z.size(1)

    def forward(self, predicted_views: List[torch.Tensor], target_views: List[torch.Tensor]):
        align_losses = []
        var_losses = []
        cov_losses = []
        slice_losses = []

        anchor_target = target_views[0].detach()

        for pv in predicted_views[1:]:
            align_losses.append(F.mse_loss(F.normalize(pv, dim=-1), F.normalize(anchor_target, dim=-1)))

        for zv in predicted_views:
            var_losses.append(self.variance_loss(zv))
            cov_losses.append(self.covariance_loss(zv))
            slice_losses.append(self.slice_reg(zv))

        for zv in target_views:
            var_losses.append(self.variance_loss(zv))
            cov_losses.append(self.covariance_loss(zv))
            slice_losses.append(self.slice_reg(zv))

        align_loss = torch.stack(align_losses).mean()
        var_loss = torch.stack(var_losses).mean()
        cov_loss = torch.stack(cov_losses).mean()
        slice_loss = torch.stack(slice_losses).mean()

        total = (
            self.align_weight * align_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
            + self.slice_weight * slice_loss
        )

        metrics = {
            "ssl_total": total.item(),
            "align_loss": align_loss.item(),
            "var_loss": var_loss.item(),
            "cov_loss": cov_loss.item(),
            "slice_loss": slice_loss.item(),
        }
        return total, metrics


@dataclass
class Config:
    train_root: str
    val_root: Optional[str]
    output_dir: str
    label_csv_train: Optional[str]
    label_csv_val: Optional[str]

    backbone_name: str = "vit_base_patch16_224"
    backbone_pretrained: bool = False
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 8

    ssl_epochs: int = 20
    ft_epochs: int = 10
    batch_size_ssl: int = 16
    batch_size_ft: int = 32
    batch_size_extract: int = 32

    ssl_lr: float = 5e-4
    ft_lr: float = 1e-4
    weight_decay: float = 5e-2
    warmup_epochs_ssl: int = 3
    warmup_epochs_ft: int = 2
    min_lr_ratio: float = 1e-3

    image_size: int = 224
    local_image_size: int = 96
    num_global_views: int = 2
    num_local_views: int = 4

    projector_hidden_dim: int = 2048
    projector_out_dim: int = 512

    align_weight: float = 1.0
    var_weight: float = 25.0
    cov_weight: float = 1.0
    slice_weight: float = 1.0
    num_slices: int = 256

    mixed_precision: bool = True
    save_every: int = 1


def create_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def cosine_scheduler(optimizer, base_lr, min_lr, total_epochs, warmup_epochs):
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    path: str,
    config: Config,
    class_to_idx: Dict[str, int],
):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": asdict(config),
        "class_to_idx": class_to_idx,
    }
    torch.save(state, path)


def train_ssl_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_fn,
    device,
    epoch,
    total_epochs,
    global_start_time,
    estimated_total_epochs,
    use_amp,
):
    model.train()

    meters = {
        "ssl_total": AverageMeter(),
        "align_loss": AverageMeter(),
        "var_loss": AverageMeter(),
        "cov_loss": AverageMeter(),
        "slice_loss": AverageMeter(),
    }

    epoch_start = time.time()
    pbar = tqdm(loader, desc=f"SSL Epoch {epoch+1}/{total_epochs}", dynamic_ncols=True)

    for step, (views, _, _) in enumerate(pbar):
        views = [v.to(device, non_blocking=True) for v in views]
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            feats = [model.encode(v) for v in views]
            projs = [model.project(f) for f in feats]
            preds = [model.predict(p) for p in projs]

            loss, metrics = loss_fn(preds, projs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = views[0].size(0)
        for k, v in metrics.items():
            meters[k].update(v, bs)

        elapsed_epoch = time.time() - epoch_start
        elapsed_global = time.time() - global_start_time
        avg_step = elapsed_epoch / max(1, step + 1)
        epoch_eta = avg_step * (len(loader) - step - 1)

        approx_done_epochs = epoch + (step + 1) / len(loader)
        avg_epoch_time = elapsed_global / max(1e-6, approx_done_epochs)
        total_eta = avg_epoch_time * estimated_total_epochs - elapsed_global

        pbar.set_postfix(
            loss=f"{meters['ssl_total'].avg:.4f}",
            align=f"{meters['align_loss'].avg:.4f}",
            slice=f"{meters['slice_loss'].avg:.4f}",
            epoch_eta=format_seconds(epoch_eta),
            total_eta=format_seconds(total_eta),
        )

    return {k: v.avg for k, v in meters.items()}


def train_ft_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    total_epochs,
    global_start_time,
    completed_ssl_epochs,
    estimated_total_epochs,
    use_amp,
):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    epoch_start = time.time()
    pbar = tqdm(loader, desc=f"FT Epoch {epoch+1}/{total_epochs}", dynamic_ncols=True)

    for step, (images, targets, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            feat = model.encode(images)
            logits = model.classify(feat)
            loss = F.cross_entropy(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_top1(logits, targets), bs)

        elapsed_epoch = time.time() - epoch_start
        elapsed_global = time.time() - global_start_time
        approx_done_epochs = completed_ssl_epochs + epoch + (step + 1) / len(loader)
        avg_epoch_time = elapsed_global / max(1e-6, approx_done_epochs)
        total_eta = avg_epoch_time * estimated_total_epochs - elapsed_global
        epoch_eta = (elapsed_epoch / max(1, step + 1)) * (len(loader) - step - 1)

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.4f}",
            epoch_eta=format_seconds(epoch_eta),
            total_eta=format_seconds(total_eta),
        )

    return {"train_loss": loss_meter.avg, "train_acc": acc_meter.avg}


@torch.no_grad()
def evaluate_classifier(model, loader, device, use_amp):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets, _ in tqdm(loader, desc="Validation", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            feat = model.encode(images)
            logits = model.classify(feat)
            loss = F.cross_entropy(logits, targets)

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_top1(logits, targets), bs)

    return {"val_loss": loss_meter.avg, "val_acc": acc_meter.avg}


@torch.no_grad()
def extract_embeddings_to_csv(model, loader, device, csv_path, class_names, use_amp):
    model.eval()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    first_batch = True
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = None

        pbar = tqdm(loader, desc=f"Extracting -> {os.path.basename(csv_path)}", dynamic_ncols=True)
        for images, targets, paths in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                feat = model.encode(images)
                logits = model.classify(feat)
                probs = torch.softmax(logits, dim=1)
                confs, preds = probs.max(dim=1)

            feat_np = feat.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            confs_np = confs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            if first_batch:
                dim = feat_np.shape[1]
                header = [
                    "image_path",
                    "target_idx",
                    "target_name",
                    "pred_idx",
                    "pred_name",
                    "confidence",
                ] + [f"emb_{i}" for i in range(dim)]
                writer = csv.writer(f)
                writer.writerow(header)
                first_batch = False

            for i in range(len(paths)):
                writer.writerow(
                    [
                        paths[i],
                        int(targets_np[i]),
                        class_names[int(targets_np[i])],
                        int(preds_np[i]),
                        class_names[int(preds_np[i])],
                        float(confs_np[i]),
                    ] + feat_np[i].astype(np.float32).tolist()
                )


def build_datasets_and_loaders(config: Config):
    ssl_dataset = RecursiveTifMultiCropDataset(
        root_dir=config.train_root,
        label_csv=config.label_csv_train,
        image_size_global=config.image_size,
        image_size_local=config.local_image_size,
        num_global_views=config.num_global_views,
        num_local_views=config.num_local_views,
    )

    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=config.batch_size_ssl,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=multicrop_collate_fn,
    )

    train_tf = build_supervised_transform(config.image_size, train=True)
    eval_tf = build_supervised_transform(config.image_size, train=False)

    ft_train_dataset = RecursiveTifDataset(
        root_dir=config.train_root,
        transform=train_tf,
        label_csv=config.label_csv_train,
        unlabeled_ok=False,
    )
    ft_train_loader = DataLoader(
        ft_train_dataset,
        batch_size=config.batch_size_ft,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    extract_train_dataset = RecursiveTifDataset(
        root_dir=config.train_root,
        transform=eval_tf,
        label_csv=config.label_csv_train,
        unlabeled_ok=False,
    )
    extract_train_loader = DataLoader(
        extract_train_dataset,
        batch_size=config.batch_size_extract,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    val_loader = None
    extract_val_loader = None

    if config.val_root is not None and os.path.isdir(config.val_root):
        val_dataset = RecursiveTifDataset(
            root_dir=config.val_root,
            transform=eval_tf,
            label_csv=config.label_csv_val,
            unlabeled_ok=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size_ft,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )
        extract_val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size_extract,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )

    return (
        ssl_dataset,
        ssl_loader,
        ft_train_dataset,
        ft_train_loader,
        extract_train_loader,
        val_loader,
        extract_val_loader,
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 1: recursive TIFF training with self-contained LeJEPA-style objective")

    parser.add_argument(
        "--train_root",
        type=str,
        default="/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023",
        help="Root directory to recursively search TIFF files",
    )
    parser.add_argument("--val_root", type=str, default=None)
    parser.add_argument("--label_csv_train", type=str, default=None)
    parser.add_argument("--label_csv_val", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--backbone_pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--ssl_epochs", type=int, default=20)
    parser.add_argument("--ft_epochs", type=int, default=10)
    parser.add_argument("--batch_size_ssl", type=int, default=16)
    parser.add_argument("--batch_size_ft", type=int, default=32)
    parser.add_argument("--batch_size_extract", type=int, default=32)

    parser.add_argument("--ssl_lr", type=float, default=5e-4)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--warmup_epochs_ssl", type=int, default=3)
    parser.add_argument("--warmup_epochs_ft", type=int, default=2)
    parser.add_argument("--min_lr_ratio", type=float, default=1e-3)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--local_image_size", type=int, default=96)
    parser.add_argument("--num_global_views", type=int, default=2)
    parser.add_argument("--num_local_views", type=int, default=4)

    parser.add_argument("--projector_hidden_dim", type=int, default=2048)
    parser.add_argument("--projector_out_dim", type=int, default=512)

    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--var_weight", type=float, default=25.0)
    parser.add_argument("--cov_weight", type=float, default=1.0)
    parser.add_argument("--slice_weight", type=float, default=1.0)
    parser.add_argument("--num_slices", type=int, default=256)

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()

    config = Config(
        train_root=args.train_root,
        val_root=args.val_root,
        output_dir=args.output_dir,
        label_csv_train=args.label_csv_train,
        label_csv_val=args.label_csv_val,
        backbone_name=args.backbone_name,
        backbone_pretrained=args.backbone_pretrained,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        ssl_epochs=args.ssl_epochs,
        ft_epochs=args.ft_epochs,
        batch_size_ssl=args.batch_size_ssl,
        batch_size_ft=args.batch_size_ft,
        batch_size_extract=args.batch_size_extract,
        ssl_lr=args.ssl_lr,
        ft_lr=args.ft_lr,
        weight_decay=args.weight_decay,
        warmup_epochs_ssl=args.warmup_epochs_ssl,
        warmup_epochs_ft=args.warmup_epochs_ft,
        min_lr_ratio=args.min_lr_ratio,
        image_size=args.image_size,
        local_image_size=args.local_image_size,
        num_global_views=args.num_global_views,
        num_local_views=args.num_local_views,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_out_dim=args.projector_out_dim,
        align_weight=args.align_weight,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        slice_weight=args.slice_weight,
        num_slices=args.num_slices,
        mixed_precision=not args.no_amp,
        save_every=args.save_every,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = config.mixed_precision and device.type == "cuda"

    found_files = recursive_find_tif_files(config.train_root)
    print("=" * 100)
    print("Recursive TIFF scan")
    print(f"Train root: {config.train_root}")
    print(f"Found TIFF files: {len(found_files)}")
    if len(found_files) > 0:
        print("First 5 files:")
        for p in found_files[:5]:
            print(f"  {p}")
    print("=" * 100)

    (
        ssl_dataset,
        ssl_loader,
        ft_train_dataset,
        ft_train_loader,
        extract_train_loader,
        val_loader,
        extract_val_loader,
    ) = build_datasets_and_loaders(config)

    num_classes = len(ft_train_dataset.classes)
    class_to_idx = ft_train_dataset.class_to_idx
    class_names = ft_train_dataset.classes

    with open(os.path.join(config.output_dir, "phase1_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    with open(os.path.join(config.output_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    model = LeJEPALikeModel(
        backbone_name=config.backbone_name,
        num_classes=num_classes,
        projector_hidden_dim=config.projector_hidden_dim,
        projector_out_dim=config.projector_out_dim,
        backbone_pretrained=config.backbone_pretrained,
    ).to(device)

    ssl_loss_fn = LeJEPALikeLoss(
        align_weight=config.align_weight,
        var_weight=config.var_weight,
        cov_weight=config.cov_weight,
        slice_weight=config.slice_weight,
        num_slices=config.num_slices,
    )

    ssl_optimizer = create_optimizer(model, lr=config.ssl_lr, weight_decay=config.weight_decay)
    ssl_scheduler = cosine_scheduler(
        ssl_optimizer,
        base_lr=config.ssl_lr,
        min_lr=config.ssl_lr * config.min_lr_ratio,
        total_epochs=config.ssl_epochs,
        warmup_epochs=config.warmup_epochs_ssl,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = -1.0
    total_pipeline_epochs = config.ssl_epochs + config.ft_epochs
    global_start_time = time.time()

    print("=" * 100)
    print("Phase 1 started")
    print(f"Num classes            : {num_classes}")
    print(f"Classes                : {class_names}")
    print(f"Backbone               : {config.backbone_name}")
    print(f"Train samples          : {len(ft_train_dataset)}")
    print(f"SSL epochs             : {config.ssl_epochs}")
    print(f"Fine-tuning epochs     : {config.ft_epochs}")
    print(f"Device                 : {device}")
    print(f"AMP enabled            : {use_amp}")
    print("=" * 100)

    print("\n[Stage 1/3] Self-supervised LeJEPA-style pretraining")
    for epoch in range(config.ssl_epochs):
        metrics = train_ssl_one_epoch(
            model=model,
            loader=ssl_loader,
            optimizer=ssl_optimizer,
            scaler=scaler,
            loss_fn=ssl_loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=config.ssl_epochs,
            global_start_time=global_start_time,
            estimated_total_epochs=total_pipeline_epochs,
            use_amp=use_amp,
        )
        ssl_scheduler.step()

        print(
            f"[SSL][Epoch {epoch+1}/{config.ssl_epochs}] "
            f"total={metrics['ssl_total']:.4f} "
            f"align={metrics['align_loss']:.4f} "
            f"var={metrics['var_loss']:.4f} "
            f"cov={metrics['cov_loss']:.4f} "
            f"slice={metrics['slice_loss']:.4f}"
        )

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=ssl_optimizer,
                scheduler=ssl_scheduler,
                epoch=epoch,
                best_metric=best_val_acc,
                path=os.path.join(config.output_dir, f"ssl_epoch_{epoch+1:03d}.pth"),
                config=config,
                class_to_idx=class_to_idx,
            )

    save_checkpoint(
        model=model,
        optimizer=ssl_optimizer,
        scheduler=ssl_scheduler,
        epoch=config.ssl_epochs - 1,
        best_metric=best_val_acc,
        path=os.path.join(config.output_dir, "phase1_ssl_last.pth"),
        config=config,
        class_to_idx=class_to_idx,
    )

    print("\n[Stage 2/3] Supervised fine-tuning")
    ft_optimizer = create_optimizer(model, lr=config.ft_lr, weight_decay=config.weight_decay)
    ft_scheduler = cosine_scheduler(
        ft_optimizer,
        base_lr=config.ft_lr,
        min_lr=config.ft_lr * config.min_lr_ratio,
        total_epochs=config.ft_epochs,
        warmup_epochs=config.warmup_epochs_ft,
    )

    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(config.ft_epochs):
        train_metrics = train_ft_one_epoch(
            model=model,
            loader=ft_train_loader,
            optimizer=ft_optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=config.ft_epochs,
            global_start_time=global_start_time,
            completed_ssl_epochs=config.ssl_epochs,
            estimated_total_epochs=total_pipeline_epochs,
            use_amp=use_amp,
        )
        ft_scheduler.step()

        msg = (
            f"[FT][Epoch {epoch+1}/{config.ft_epochs}] "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"train_acc={train_metrics['train_acc']:.4f}"
        )

        if val_loader is not None:
            val_metrics = evaluate_classifier(model, val_loader, device, use_amp)
            msg += (
                f" val_loss={val_metrics['val_loss']:.4f} "
                f"val_acc={val_metrics['val_acc']:.4f}"
            )
            metric_to_compare = val_metrics["val_acc"]
        else:
            metric_to_compare = train_metrics["train_acc"]

        if metric_to_compare > best_val_acc:
            best_val_acc = metric_to_compare
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model=model,
                optimizer=ft_optimizer,
                scheduler=ft_scheduler,
                epoch=epoch,
                best_metric=best_val_acc,
                path=os.path.join(config.output_dir, "phase1_best.pth"),
                config=config,
                class_to_idx=class_to_idx,
            )

        print(msg)

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=ft_optimizer,
                scheduler=ft_scheduler,
                epoch=epoch,
                best_metric=best_val_acc,
                path=os.path.join(config.output_dir, f"ft_epoch_{epoch+1:03d}.pth"),
                config=config,
                class_to_idx=class_to_idx,
            )

    model.load_state_dict(best_model_state)

    save_checkpoint(
        model=model,
        optimizer=ft_optimizer,
        scheduler=ft_scheduler,
        epoch=config.ft_epochs - 1,
        best_metric=best_val_acc,
        path=os.path.join(config.output_dir, "phase1_final.pth"),
        config=config,
        class_to_idx=class_to_idx,
    )

    print("\n[Stage 3/3] Embedding extraction")
    train_csv = os.path.join(config.output_dir, "train_embeddings.csv")
    extract_embeddings_to_csv(
        model=model,
        loader=extract_train_loader,
        device=device,
        csv_path=train_csv,
        class_names=class_names,
        use_amp=use_amp,
    )

    if extract_val_loader is not None:
        val_csv = os.path.join(config.output_dir, "val_embeddings.csv")
        extract_embeddings_to_csv(
            model=model,
            loader=extract_val_loader,
            device=device,
            csv_path=val_csv,
            class_names=class_names,
            use_amp=use_amp,
        )

    total_elapsed = time.time() - global_start_time

    print("\n" + "=" * 100)
    print("Phase 1 completed")
    print(f"Best metric           : {best_val_acc:.4f}")
    print(f"Final model           : {os.path.join(config.output_dir, 'phase1_final.pth')}")
    print(f"Best model            : {os.path.join(config.output_dir, 'phase1_best.pth')}")
    print(f"Train embeddings CSV  : {train_csv}")
    if extract_val_loader is not None:
        print(f"Val embeddings CSV    : {os.path.join(config.output_dir, 'val_embeddings.csv')}")
    print(f"Total elapsed         : {format_seconds(total_elapsed)}")
    print("=" * 100)


if __name__ == "__main__":
    main()