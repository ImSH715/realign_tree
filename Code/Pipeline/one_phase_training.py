import os
import csv
import json
import math
import time
import glob
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

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
    return sorted(list(set(os.path.abspath(p) for p in files)))


def safe_open_image(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise RuntimeError(f"Failed to open image: {path} | {e}") from e


class TileCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max_items
        self.cache: Dict[str, Image.Image] = {}
        self.order: List[str] = []

    def get(self, path: str) -> Image.Image:
        if path in self.cache:
            if path in self.order:
                self.order.remove(path)
            self.order.append(path)
            return self.cache[path]

        image = safe_open_image(path)
        self.cache[path] = image
        self.order.append(path)

        if len(self.order) > self.max_items:
            old_path = self.order.pop(0)
            if old_path in self.cache:
                del self.cache[old_path]

        return image


def collate_multiview_with_meta(batch):
    all_views = list(zip(*[item[0] for item in batch]))
    stacked_views = [torch.stack(v, dim=0) for v in all_views]
    metas = [item[1] for item in batch]
    return stacked_views, metas


class RandomPatchTifDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size_px: int = 224,
        patches_per_image: int = 100,
        num_global_views: int = 2,
        num_local_views: int = 4,
        image_size_global: int = 224,
        image_size_local: int = 96,
        tile_cache_size: int = 8,
    ) -> None:
        self.tif_paths = recursive_find_tif_files(root_dir)
        if len(self.tif_paths) == 0:
            raise RuntimeError(f"No TIFF files found under: {root_dir}")

        self.patch_size_px = patch_size_px
        self.patches_per_image = patches_per_image
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.tile_cache = TileCache(max_items=tile_cache_size)

        self.index_map: List[str] = []
        for path in self.tif_paths:
            for _ in range(self.patches_per_image):
                self.index_map.append(path)

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
                    scale=(0.6, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                *common_aug,
            ]
        )

        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size_local,
                    scale=(0.3, 0.7),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                *common_aug,
            ]
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def random_crop_patch(self, image: Image.Image) -> Tuple[Image.Image, float, float]:
        w, h = image.size

        if w < self.patch_size_px or h < self.patch_size_px:
            new_w = max(w, self.patch_size_px)
            new_h = max(h, self.patch_size_px)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
            w, h = image.size

        x = random.randint(0, w - self.patch_size_px)
        y = random.randint(0, h - self.patch_size_px)

        patch = image.crop((x, y, x + self.patch_size_px, y + self.patch_size_px))
        center_x = x + self.patch_size_px / 2.0
        center_y = y + self.patch_size_px / 2.0

        return patch, center_x, center_y

    def __getitem__(self, idx: int):
        path = self.index_map[idx]
        image = self.tile_cache.get(path)
        patch, center_x, center_y = self.random_crop_patch(image)

        views = []
        for _ in range(self.num_global_views):
            views.append(self.global_transform(patch))
        for _ in range(self.num_local_views):
            views.append(self.local_transform(patch))

        meta = {
            "image_path": path,
            "x": float(center_x),
            "y": float(center_y),
        }
        return views, meta


class GridPatchTifDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size_px: int = 224,
        stride_px: int = 224,
        image_size: int = 224,
        tile_cache_size: int = 8,
        max_patches_per_image: Optional[int] = None,
    ) -> None:
        self.tif_paths = recursive_find_tif_files(root_dir)
        if len(self.tif_paths) == 0:
            raise RuntimeError(f"No TIFF files found under: {root_dir}")

        self.patch_size_px = patch_size_px
        self.stride_px = stride_px
        self.image_size = image_size
        self.tile_cache = TileCache(max_items=tile_cache_size)
        self.samples: List[Tuple[str, float, float]] = []

        for path in self.tif_paths:
            image = self.tile_cache.get(path)
            w, h = image.size

            xs = list(range(patch_size_px // 2, max(patch_size_px // 2 + 1, w - patch_size_px // 2 + 1), stride_px))
            ys = list(range(patch_size_px // 2, max(patch_size_px // 2 + 1, h - patch_size_px // 2 + 1), stride_px))

            image_samples = [(path, float(x), float(y)) for y in ys for x in xs]

            if len(image_samples) == 0:
                cx = max(w / 2.0, patch_size_px / 2.0)
                cy = max(h / 2.0, patch_size_px / 2.0)
                image_samples = [(path, float(cx), float(cy))]

            if max_patches_per_image is not None and len(image_samples) > max_patches_per_image:
                step = len(image_samples) / max_patches_per_image
                reduced = []
                for i in range(max_patches_per_image):
                    reduced.append(image_samples[int(i * step)])
                image_samples = reduced

            self.samples.extend(image_samples)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def crop_patch_center(self, image: Image.Image, center_x: float, center_y: float) -> Image.Image:
        w, h = image.size
        half = self.patch_size_px // 2

        cx = int(round(center_x))
        cy = int(round(center_y))

        left = cx - half
        top = cy - half
        right = left + self.patch_size_px
        bottom = top + self.patch_size_px

        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - w)
        pad_bottom = max(0, bottom - h)

        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        patch = image.crop((left, top, right, bottom))

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            canvas = Image.new("RGB", (self.patch_size_px, self.patch_size_px))
            canvas.paste(patch, (pad_left, pad_top))
            patch = canvas

        if patch.size != (self.patch_size_px, self.patch_size_px):
            patch = patch.resize((self.patch_size_px, self.patch_size_px), resample=Image.BICUBIC)

        return patch

    def __getitem__(self, idx: int):
        path, x, y = self.samples[idx]
        image = self.tile_cache.get(path)
        patch = self.crop_patch_center(image, x, y)
        patch = self.transform(patch)

        meta = {
            "image_path": path,
            "x": float(x),
            "y": float(y),
        }
        return patch, meta


def collate_patch_with_meta(batch):
    patches = torch.stack([item[0] for item in batch], dim=0)
    metas = [item[1] for item in batch]
    return patches, metas


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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project(self, feat: torch.Tensor) -> torch.Tensor:
        return self.projector(feat)

    def predict(self, proj: torch.Tensor) -> torch.Tensor:
        return self.predictor(proj)


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
        d = z.size(1)
        directions = torch.randn(self.num_slices, d, device=z.device, dtype=z.dtype)
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
        cov = (z.T @ z) / max(1, (z.size(0) - 1))
        return off_diagonal(cov).pow(2).sum() / z.size(1)

    def forward(self, predicted_views: List[torch.Tensor], target_views: List[torch.Tensor]):
        align_losses = []
        var_losses = []
        cov_losses = []
        slice_losses = []

        anchor_target = target_views[0].detach()

        for pv in predicted_views[1:]:
            align_losses.append(
                F.mse_loss(F.normalize(pv, dim=-1), F.normalize(anchor_target, dim=-1))
            )

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
            "ssl_total": float(total.item()),
            "align_loss": float(align_loss.item()),
            "var_loss": float(var_loss.item()),
            "cov_loss": float(cov_loss.item()),
            "slice_loss": float(slice_loss.item()),
        }
        return total, metrics


@dataclass
class Config:
    train_root: str
    output_dir: str

    backbone_name: str = "vit_base_patch16_224"
    backbone_pretrained: bool = False

    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0

    ssl_epochs: int = 20
    batch_size_ssl: int = 8
    ssl_lr: float = 5e-4
    weight_decay: float = 5e-2
    warmup_epochs_ssl: int = 3
    min_lr_ratio: float = 1e-3

    patch_size_px: int = 224
    patches_per_image: int = 100
    num_global_views: int = 2
    num_local_views: int = 4
    image_size_global: int = 224
    image_size_local: int = 96

    projector_hidden_dim: int = 2048
    projector_out_dim: int = 512

    align_weight: float = 1.0
    var_weight: float = 25.0
    cov_weight: float = 1.0
    slice_weight: float = 1.0
    num_slices: int = 256

    extract_stride_px: int = 224
    extract_batch_size: int = 32
    max_extract_patches_per_image: Optional[int] = None

    mixed_precision: bool = True
    tile_cache_size: int = 8
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


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    path: str,
    config: Config,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": asdict(config),
    }
    torch.save(state, path)


def train_ssl_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    global_start_time: float,
    use_amp: bool,
) -> Dict[str, float]:
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

    for step, (views, _) in enumerate(pbar):
        views = [v.to(device, non_blocking=True) for v in views]
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
        elapsed_total = time.time() - global_start_time

        avg_step_time = elapsed_epoch / max(1, step + 1)
        epoch_eta = avg_step_time * (len(loader) - step - 1)

        avg_epoch_time = elapsed_total / max(1e-6, epoch + (step + 1) / len(loader))
        total_eta = avg_epoch_time * total_epochs - elapsed_total

        pbar.set_postfix(
            loss=f"{meters['ssl_total'].avg:.4f}",
            align=f"{meters['align_loss'].avg:.4f}",
            slice=f"{meters['slice_loss'].avg:.4f}",
            epoch_eta=format_seconds(epoch_eta),
            total_eta=format_seconds(total_eta),
        )

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate_ssl_proxy_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    num_batches: int = 10,
) -> float:
    model.eval()
    losses = []

    pbar = tqdm(loader, desc="Evaluating proxy loss", dynamic_ncols=True)
    for i, (views, _) in enumerate(pbar):
        if i >= num_batches:
            break

        views = [v.to(device, non_blocking=True) for v in views]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            feats = [model.encode(v) for v in views]
            projs = [model.project(f) for f in feats]
            preds = [model.predict(p) for p in projs]
            loss, _ = loss_fn(preds, projs)

        losses.append(float(loss.item()))

    if len(losses) == 0:
        return float("inf")
    return float(np.mean(losses))


@torch.no_grad()
def extract_embeddings_to_csv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_csv: str,
    use_amp: bool,
) -> None:
    model.eval()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    first_batch = True
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = None

        pbar = tqdm(loader, desc="Extracting embeddings", dynamic_ncols=True)
        for patches, metas in pbar:
            patches = patches.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                feat = model.encode(patches)

            feat_np = feat.detach().cpu().numpy()

            if first_batch:
                dim = feat_np.shape[1]
                header = ["image_path", "point_id", "x", "y"] + [f"emb_{i}" for i in range(dim)]
                writer = csv.writer(f)
                writer.writerow(header)
                first_batch = False

            for i, meta in enumerate(metas):
                point_id = f"{os.path.basename(meta['image_path'])}_{int(round(meta['x']))}_{int(round(meta['y']))}"
                row = [
                    meta["image_path"],
                    point_id,
                    float(meta["x"]),
                    float(meta["y"]),
                ] + feat_np[i].astype(np.float32).tolist()
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: unlabeled LeJEPA-style training and embedding extraction from recursive TIFF dataset"
    )

    parser.add_argument(
        "--train_root",
        type=str,
        default="/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023",
        help="Root directory to recursively scan TIFF files",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--backbone_pretrained", action="store_true")

    parser.add_argument("--ssl_epochs", type=int, default=20)
    parser.add_argument("--batch_size_ssl", type=int, default=8)
    parser.add_argument("--ssl_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--warmup_epochs_ssl", type=int, default=3)
    parser.add_argument("--min_lr_ratio", type=float, default=1e-3)

    parser.add_argument("--patch_size_px", type=int, default=224)
    parser.add_argument("--patches_per_image", type=int, default=100)
    parser.add_argument("--num_global_views", type=int, default=2)
    parser.add_argument("--num_local_views", type=int, default=4)
    parser.add_argument("--image_size_global", type=int, default=224)
    parser.add_argument("--image_size_local", type=int, default=96)

    parser.add_argument("--projector_hidden_dim", type=int, default=2048)
    parser.add_argument("--projector_out_dim", type=int, default=512)

    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--var_weight", type=float, default=25.0)
    parser.add_argument("--cov_weight", type=float, default=1.0)
    parser.add_argument("--slice_weight", type=float, default=1.0)
    parser.add_argument("--num_slices", type=int, default=256)

    parser.add_argument("--extract_stride_px", type=int, default=224)
    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--max_extract_patches_per_image", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tile_cache_size", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    config = Config(
        train_root=args.train_root,
        output_dir=args.output_dir,
        backbone_name=args.backbone_name,
        backbone_pretrained=args.backbone_pretrained,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        ssl_epochs=args.ssl_epochs,
        batch_size_ssl=args.batch_size_ssl,
        ssl_lr=args.ssl_lr,
        weight_decay=args.weight_decay,
        warmup_epochs_ssl=args.warmup_epochs_ssl,
        min_lr_ratio=args.min_lr_ratio,
        patch_size_px=args.patch_size_px,
        patches_per_image=args.patches_per_image,
        num_global_views=args.num_global_views,
        num_local_views=args.num_local_views,
        image_size_global=args.image_size_global,
        image_size_local=args.image_size_local,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_out_dim=args.projector_out_dim,
        align_weight=args.align_weight,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        slice_weight=args.slice_weight,
        num_slices=args.num_slices,
        extract_stride_px=args.extract_stride_px,
        extract_batch_size=args.extract_batch_size,
        max_extract_patches_per_image=args.max_extract_patches_per_image,
        mixed_precision=not args.no_amp,
        tile_cache_size=args.tile_cache_size,
        save_every=args.save_every,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = config.mixed_precision and device.type == "cuda"

    tif_files = recursive_find_tif_files(config.train_root)

    print("=" * 100)
    print("Recursive TIFF scan")
    print(f"Train root: {config.train_root}")
    print(f"Found TIFF files: {len(tif_files)}")
    if len(tif_files) > 0:
        print("First 5 files:")
        for p in tif_files[:5]:
            print(f"  {p}")
    print("=" * 100)

    if len(tif_files) == 0:
        raise RuntimeError("No TIFF files found. Please check --train_root.")

    with open(os.path.join(config.output_dir, "phase1_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    train_dataset = RandomPatchTifDataset(
        root_dir=config.train_root,
        patch_size_px=config.patch_size_px,
        patches_per_image=config.patches_per_image,
        num_global_views=config.num_global_views,
        num_local_views=config.num_local_views,
        image_size_global=config.image_size_global,
        image_size_local=config.image_size_local,
        tile_cache_size=config.tile_cache_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_ssl,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(config.num_workers > 0),
        collate_fn=collate_multiview_with_meta,
    )

    eval_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_ssl,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(config.num_workers > 0),
        collate_fn=collate_multiview_with_meta,
    )

    extract_dataset = GridPatchTifDataset(
        root_dir=config.train_root,
        patch_size_px=config.patch_size_px,
        stride_px=config.extract_stride_px,
        image_size=config.image_size_global,
        tile_cache_size=config.tile_cache_size,
        max_patches_per_image=config.max_extract_patches_per_image,
    )

    extract_loader = DataLoader(
        extract_dataset,
        batch_size=config.extract_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(config.num_workers > 0),
        collate_fn=collate_patch_with_meta,
    )

    model = LeJEPALikeModel(
        backbone_name=config.backbone_name,
        projector_hidden_dim=config.projector_hidden_dim,
        projector_out_dim=config.projector_out_dim,
        backbone_pretrained=config.backbone_pretrained,
    ).to(device)

    loss_fn = LeJEPALikeLoss(
        align_weight=config.align_weight,
        var_weight=config.var_weight,
        cov_weight=config.cov_weight,
        slice_weight=config.slice_weight,
        num_slices=config.num_slices,
    )

    optimizer = create_optimizer(model, lr=config.ssl_lr, weight_decay=config.weight_decay)
    scheduler = cosine_scheduler(
        optimizer=optimizer,
        base_lr=config.ssl_lr,
        min_lr=config.ssl_lr * config.min_lr_ratio,
        total_epochs=config.ssl_epochs,
        warmup_epochs=config.warmup_epochs_ssl,
    )

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    global_start_time = time.time()
    best_proxy_loss = float("inf")

    print("=" * 100)
    print("Phase 1 started")
    print(f"Backbone                  : {config.backbone_name}")
    print(f"Device                    : {device}")
    print(f"AMP enabled               : {use_amp}")
    print(f"Train TIFF files          : {len(tif_files)}")
    print(f"Patches per image         : {config.patches_per_image}")
    print(f"Train samples             : {len(train_dataset)}")
    print(f"SSL epochs                : {config.ssl_epochs}")
    print(f"Patch size                : {config.patch_size_px}")
    print(f"Extraction stride         : {config.extract_stride_px}")
    print(f"Extraction samples        : {len(extract_dataset)}")
    print("=" * 100)

    for epoch in range(config.ssl_epochs):
        train_metrics = train_ssl_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=config.ssl_epochs,
            global_start_time=global_start_time,
            use_amp=use_amp,
        )
        scheduler.step()

        proxy_loss = evaluate_ssl_proxy_loss(
            model=model,
            loader=eval_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            num_batches=min(10, len(eval_loader)),
        )

        msg = (
            f"[SSL][Epoch {epoch+1}/{config.ssl_epochs}] "
            f"total={train_metrics['ssl_total']:.4f} "
            f"align={train_metrics['align_loss']:.4f} "
            f"var={train_metrics['var_loss']:.4f} "
            f"cov={train_metrics['cov_loss']:.4f} "
            f"slice={train_metrics['slice_loss']:.4f} "
            f"proxy_eval={proxy_loss:.4f}"
        )
        print(msg)

        if proxy_loss < best_proxy_loss:
            best_proxy_loss = proxy_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_proxy_loss,
                path=os.path.join(config.output_dir, "phase1_encoder_best.pth"),
                config=config,
            )

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_proxy_loss,
                path=os.path.join(config.output_dir, f"phase1_epoch_{epoch+1:03d}.pth"),
                config=config,
            )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config.ssl_epochs - 1,
        best_metric=best_proxy_loss,
        path=os.path.join(config.output_dir, "phase1_encoder_last.pth"),
        config=config,
    )

    print("\n[Stage 2/2] Embedding extraction")
    output_csv = os.path.join(config.output_dir, "phase1_embeddings.csv")
    extract_embeddings_to_csv(
        model=model,
        loader=extract_loader,
        device=device,
        output_csv=output_csv,
        use_amp=use_amp,
    )

    total_elapsed = time.time() - global_start_time

    print("\n" + "=" * 100)
    print("Phase 1 completed")
    print(f"Best checkpoint            : {os.path.join(config.output_dir, 'phase1_encoder_best.pth')}")
    print(f"Last checkpoint            : {os.path.join(config.output_dir, 'phase1_encoder_last.pth')}")
    print(f"Embedding CSV              : {output_csv}")
    print(f"Best proxy loss            : {best_proxy_loss:.4f}")
    print(f"Total elapsed              : {format_seconds(total_elapsed)}")
    print("=" * 100)


if __name__ == "__main__":
    main()