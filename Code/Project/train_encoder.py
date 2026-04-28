"""
Phase 1: self-supervised encoder pretraining on recursive TIFF patches.

This stage does NOT use labels.
It learns general image representations from random TIFF patches.

Outputs:
- phase1_encoder_best.pth
- phase1_encoder_last.pth
- phase1_epoch_XXX.pth
- phase1_embeddings.csv
- phase1_training_history.csv
- debug_patches/*.png

Downstream Phase 2 / Phase 3 remain compatible as long as model.encode(...) works.
"""

import os
import csv
import math
import time
import json
import argparse
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.models.encoder import SSLImageEncoder, LeJEPALikeLoss
from src.models.checkpoint import save_checkpoint
from src.data.tif_io import recursive_find_tif_files
from src.data.patches import (
    RandomPatchTifDataset,
    GridPatchTifDataset,
    collate_multiview_with_meta,
    collate_patch_with_meta,
)


def set_seed(seed: int) -> None:
    import random
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
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n


def create_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def cosine_scheduler(optimizer, base_lr, min_lr, total_epochs, warmup_epochs):
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)

        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def backward_step(loss, optimizer, scaler, grad_clip_norm):
    if scaler is not None:
        scaler.scale(loss).backward()

        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                grad_clip_norm,
            )

        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                grad_clip_norm,
            )

        optimizer.step()


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
    use_amp,
    grad_clip_norm,
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

    for step, (views, _) in enumerate(pbar):
        views = [v.to(device, non_blocking=True) for v in views]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            feats = [model.encode(v) for v in views]
            projs = [model.project(f) for f in feats]
            preds = [model.predict(p) for p in projs]
            loss, metrics = loss_fn(preds, projs)

        backward_step(
            loss=loss,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
        )

        batch_size = views[0].size(0)

        for key, value in metrics.items():
            meters[key].update(value, batch_size)

        elapsed_epoch = time.time() - epoch_start
        elapsed_total = time.time() - global_start_time

        avg_step_time = elapsed_epoch / max(1, step + 1)
        epoch_eta = avg_step_time * (len(loader) - step - 1)

        avg_epoch_time = elapsed_total / max(1e-6, epoch + (step + 1) / len(loader))
        total_eta = avg_epoch_time * total_epochs - elapsed_total

        pbar.set_postfix(
            loss=f"{meters['ssl_total'].avg:.4f}",
            align=f"{meters['align_loss'].avg:.4f}",
            var=f"{meters['var_loss'].avg:.4f}",
            slice=f"{meters['slice_loss'].avg:.4f}",
            lr=f"{get_lr(optimizer):.2e}",
            epoch_eta=format_seconds(epoch_eta),
            total_eta=format_seconds(total_eta),
        )

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate_ssl_proxy_loss(
    model,
    loader,
    loss_fn,
    device,
    use_amp,
    num_batches=10,
):
    model.eval()

    losses = []
    pbar = tqdm(loader, desc="Evaluating SSL proxy", dynamic_ncols=True)

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
        pbar.set_postfix(proxy=f"{np.mean(losses):.4f}")

    if len(losses) == 0:
        return float("inf")

    return float(np.mean(losses))


def denormalize_for_debug(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


@torch.no_grad()
def save_debug_patches(loader, output_dir, max_images=32):
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    for patches, metas in loader:
        patches = denormalize_for_debug(patches)

        for i in range(patches.size(0)):
            if saved >= max_images:
                return

            meta = metas[i]
            base = os.path.basename(meta["image_path"])
            x = int(round(float(meta["x"])))
            y = int(round(float(meta["y"])))

            out_path = os.path.join(output_dir, f"{saved:04d}_{base}_{x}_{y}.png")
            save_image(patches[i], out_path)
            saved += 1


@torch.no_grad()
def extract_embeddings_to_csv(
    model,
    loader,
    device,
    output_csv,
    use_amp,
):
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
                point_id = (
                    f"{os.path.basename(meta['image_path'])}_"
                    f"{int(round(meta['x']))}_"
                    f"{int(round(meta['y']))}"
                )

                row = [
                    meta["image_path"],
                    point_id,
                    float(meta["x"]),
                    float(meta["y"]),
                ] + feat_np[i].astype(np.float32).tolist()

                writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: self-supervised encoder training on recursive TIFF patches"
    )

    parser.add_argument("--train_root", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--backbone_name",
        default="vit_base_patch16_224",
        choices=[
            "vit_base_patch16_224",
            "resnet50",
            "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m",
        ],
    )

    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--backbone_pretrained", action="store_true")

    parser.add_argument("--ssl_epochs", type=int, default=20)
    parser.add_argument("--batch_size_ssl", type=int, default=8)
    parser.add_argument("--ssl_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--warmup_epochs_ssl", type=int, default=3)
    parser.add_argument("--min_lr_ratio", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

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

    parser.add_argument("--eval_batches", type=int, default=10)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tile_cache_size", type=int, default=0)

    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--debug_patches", type=int, default=32)

    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    use_pretrained_backbone = args.pretrained_backbone or args.backbone_pretrained

    config = vars(args).copy()
    config["pretrained_backbone"] = use_pretrained_backbone
    config["mixed_precision"] = not args.no_amp

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    tif_files = recursive_find_tif_files(args.train_root)

    print("=" * 100)
    print("Recursive TIFF scan")
    print(f"Train root        : {args.train_root}")
    print(f"Found TIFF files  : {len(tif_files)}")
    print("=" * 100)

    if len(tif_files) == 0:
        raise RuntimeError("No TIFF files found. Please check --train_root.")

    print("First 10 TIFF files:")
    for p in tif_files[:10]:
        print(f"  {p}")
    print("=" * 100)

    with open(os.path.join(args.output_dir, "phase1_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_dataset = RandomPatchTifDataset(
        root_dir=args.train_root,
        patch_size_px=args.patch_size_px,
        patches_per_image=args.patches_per_image,
        num_global_views=args.num_global_views,
        num_local_views=args.num_local_views,
        image_size_global=args.image_size_global,
        image_size_local=args.image_size_local,
        tile_cache_size=args.tile_cache_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_ssl,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_multiview_with_meta,
    )

    eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_ssl,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_multiview_with_meta,
    )

    extract_dataset = None
    extract_loader = None

    if not args.skip_extract:
        extract_dataset = GridPatchTifDataset(
            root_dir=args.train_root,
            patch_size_px=args.patch_size_px,
            stride_px=args.extract_stride_px,
            image_size=args.image_size_global,
            tile_cache_size=args.tile_cache_size,
            max_patches_per_image=args.max_extract_patches_per_image,
        )

        extract_loader = DataLoader(
            extract_dataset,
            batch_size=args.extract_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_patch_with_meta,
        )

        save_debug_patches(
            loader=extract_loader,
            output_dir=os.path.join(args.output_dir, "debug_patches"),
            max_images=args.debug_patches,
        )

    model = SSLImageEncoder(
        backbone_name=args.backbone_name,
        pretrained_backbone=use_pretrained_backbone,
        image_size=args.image_size_global,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_out_dim=args.projector_out_dim,
    ).to(device)

    loss_fn = LeJEPALikeLoss(
        align_weight=args.align_weight,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        slice_weight=args.slice_weight,
        num_slices=args.num_slices,
    )

    optimizer = create_optimizer(
        model=model,
        lr=args.ssl_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = cosine_scheduler(
        optimizer=optimizer,
        base_lr=args.ssl_lr,
        min_lr=args.ssl_lr * args.min_lr_ratio,
        total_epochs=args.ssl_epochs,
        warmup_epochs=args.warmup_epochs_ssl,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    print("=" * 100)
    print("Phase 1 started")
    print(f"Backbone              : {args.backbone_name}")
    print(f"Pretrained backbone   : {use_pretrained_backbone}")
    print(f"Device                : {device}")
    print(f"AMP enabled           : {use_amp}")
    print(f"Train TIFF files      : {len(tif_files)}")
    print(f"Patches per image     : {args.patches_per_image}")
    print(f"Train samples         : {len(train_dataset)}")
    print(f"SSL epochs            : {args.ssl_epochs}")
    print(f"Patch size            : {args.patch_size_px}")
    print(f"Global image size     : {args.image_size_global}")
    print(f"Local image size      : {args.image_size_local}")
    print(f"Extraction enabled    : {not args.skip_extract}")
    if extract_dataset is not None:
        print(f"Extraction stride     : {args.extract_stride_px}")
        print(f"Extraction samples    : {len(extract_dataset)}")
    print("=" * 100)

    global_start_time = time.time()
    best_proxy_loss = float("inf")
    history = []

    for epoch in range(args.ssl_epochs):
        train_metrics = train_ssl_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=args.ssl_epochs,
            global_start_time=global_start_time,
            use_amp=use_amp,
            grad_clip_norm=args.grad_clip_norm,
        )

        scheduler.step()

        proxy_loss = evaluate_ssl_proxy_loss(
            model=model,
            loader=eval_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            num_batches=min(args.eval_batches, len(eval_loader)),
        )

        row = {
            "epoch": epoch + 1,
            "lr": get_lr(optimizer),
            "proxy_loss": proxy_loss,
            **train_metrics,
        }
        history.append(row)

        print(
            f"[SSL][Epoch {epoch+1}/{args.ssl_epochs}] "
            f"total={train_metrics['ssl_total']:.4f} "
            f"align={train_metrics['align_loss']:.4f} "
            f"var={train_metrics['var_loss']:.4f} "
            f"cov={train_metrics['cov_loss']:.4f} "
            f"slice={train_metrics['slice_loss']:.4f} "
            f"proxy_eval={proxy_loss:.4f} "
            f"lr={get_lr(optimizer):.2e}"
        )

        with open(os.path.join(args.output_dir, "phase1_training_history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if proxy_loss < best_proxy_loss:
            best_proxy_loss = proxy_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_proxy_loss,
                path=os.path.join(args.output_dir, "phase1_encoder_best.pth"),
                config_dict=config,
            )

            print(f"[INFO] Saved best checkpoint: proxy_loss={best_proxy_loss:.6f}")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_proxy_loss,
                path=os.path.join(args.output_dir, f"phase1_epoch_{epoch+1:03d}.pth"),
                config_dict=config,
            )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.ssl_epochs - 1,
        best_metric=best_proxy_loss,
        path=os.path.join(args.output_dir, "phase1_encoder_last.pth"),
        config_dict=config,
    )

    if not args.skip_extract and extract_loader is not None:
        print("\n[Stage 2/2] Embedding extraction")

        output_csv = os.path.join(args.output_dir, "phase1_embeddings.csv")

        extract_embeddings_to_csv(
            model=model,
            loader=extract_loader,
            device=device,
            output_csv=output_csv,
            use_amp=use_amp,
        )
    else:
        output_csv = None

    total_elapsed = time.time() - global_start_time

    print("\n" + "=" * 100)
    print("Phase 1 completed")
    print(f"Best checkpoint      : {os.path.join(args.output_dir, 'phase1_encoder_best.pth')}")
    print(f"Last checkpoint      : {os.path.join(args.output_dir, 'phase1_encoder_last.pth')}")
    if output_csv:
        print(f"Embedding CSV        : {output_csv}")
    else:
        print("Embedding CSV        : skipped")
    print(f"Best proxy loss      : {best_proxy_loss:.4f}")
    print(f"Total elapsed        : {format_seconds(total_elapsed)}")
    print("=" * 100)


if __name__ == "__main__":
    main()