import os
import csv
import math
import time
import argparse
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.encoder import LeJEPALikeEncoder, LeJEPALikeLoss
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


def create_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
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


def train_ssl_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    global_start_time: float,
    use_amp: bool,
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

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
            slice=f"{meters['slice_loss'].avg:.4f}",
            epoch_eta=format_seconds(epoch_eta),
            total_eta=format_seconds(total_eta),
        )

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate_ssl_proxy_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
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
    model: torch.nn.Module,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: unlabeled LeJEPA-style encoder training on recursive TIFF patches"
    )

    parser.add_argument("--train_root", type=str, required=True)
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
    parser.add_argument("--tile_cache_size", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = vars(args).copy()
    config["mixed_precision"] = not args.no_amp

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    tif_files = recursive_find_tif_files(args.train_root)

    print("=" * 100)
    print("Recursive TIFF scan")
    print(f"Train root: {args.train_root}")
    print(f"Found TIFF files: {len(tif_files)}")
    if len(tif_files) > 0:
        print("First 5 files:")
        for p in tif_files[:5]:
            print(f"  {p}")
    print("=" * 100)

    if len(tif_files) == 0:
        raise RuntimeError("No TIFF files found. Please check --train_root.")

    with open(os.path.join(args.output_dir, "phase1_config.json"), "w", encoding="utf-8") as f:
        import json
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

    model = LeJEPALikeEncoder(
        backbone_name=args.backbone_name,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_out_dim=args.projector_out_dim,
        backbone_pretrained=args.backbone_pretrained,
    ).to(device)

    loss_fn = LeJEPALikeLoss(
        align_weight=args.align_weight,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        slice_weight=args.slice_weight,
        num_slices=args.num_slices,
    )

    optimizer = create_optimizer(model, lr=args.ssl_lr, weight_decay=args.weight_decay)
    scheduler = cosine_scheduler(
        optimizer=optimizer,
        base_lr=args.ssl_lr,
        min_lr=args.ssl_lr * args.min_lr_ratio,
        total_epochs=args.ssl_epochs,
        warmup_epochs=args.warmup_epochs_ssl,
    )

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    global_start_time = time.time()
    best_proxy_loss = float("inf")

    print("=" * 100)
    print("Phase 1 started")
    print(f"Backbone                  : {args.backbone_name}")
    print(f"Device                    : {device}")
    print(f"AMP enabled               : {use_amp}")
    print(f"Train TIFF files          : {len(tif_files)}")
    print(f"Patches per image         : {args.patches_per_image}")
    print(f"Train samples             : {len(train_dataset)}")
    print(f"SSL epochs                : {args.ssl_epochs}")
    print(f"Patch size                : {args.patch_size_px}")
    print(f"Extraction stride         : {args.extract_stride_px}")
    print(f"Extraction samples        : {len(extract_dataset)}")
    print("=" * 100)

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

        print(
            f"[SSL][Epoch {epoch+1}/{args.ssl_epochs}] "
            f"total={train_metrics['ssl_total']:.4f} "
            f"align={train_metrics['align_loss']:.4f} "
            f"var={train_metrics['var_loss']:.4f} "
            f"cov={train_metrics['cov_loss']:.4f} "
            f"slice={train_metrics['slice_loss']:.4f} "
            f"proxy_eval={proxy_loss:.4f}"
        )

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

        if (epoch + 1) % args.save_every == 0:
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

    print("\n[Stage 2/2] Embedding extraction")
    output_csv = os.path.join(args.output_dir, "phase1_embeddings.csv")
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
    print(f"Best checkpoint            : {os.path.join(args.output_dir, 'phase1_encoder_best.pth')}")
    print(f"Last checkpoint            : {os.path.join(args.output_dir, 'phase1_encoder_last.pth')}")
    print(f"Embedding CSV              : {output_csv}")
    print(f"Best proxy loss            : {best_proxy_loss:.4f}")
    print(f"Total elapsed              : {format_seconds(total_elapsed)}")
    print("=" * 100)


if __name__ == "__main__":
    main()