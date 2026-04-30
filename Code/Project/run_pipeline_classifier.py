"""
Phase 3 classifier-head based bounded search.

For each input point:
- generate candidate patches around original coordinate
- encode patch
- classifier head predicts class probability
- choose coordinate with highest target-class probability minus distance penalty

Supports:
- binary: target class can be mapped to 1/0
- multiclass: target label must match classifier classes
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.tif_io import recursive_find_tif_files


def format_seconds(seconds):
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def world_to_pixel(image_path, east, north):
    with rasterio.open(image_path) as src:
        row, col = src.index(float(east), float(north))
    return float(col), float(row)


def pixel_to_world(image_path, x, y):
    with rasterio.open(image_path) as src:
        east, north = src.xy(float(y), float(x))
    return float(east), float(north)


class TileResolver:
    def __init__(self, imagery_root):
        self.lookup = {}
        for p in recursive_find_tif_files(imagery_root):
            ap = os.path.abspath(p)
            self.lookup[ap] = ap
            self.lookup[os.path.basename(ap)] = ap

    def resolve(self, value):
        if value in self.lookup:
            return self.lookup[value]
        abs_value = os.path.abspath(value)
        if abs_value in self.lookup:
            return self.lookup[abs_value]
        base = os.path.basename(value)
        if base in self.lookup:
            return self.lookup[base]
        raise FileNotFoundError(f"Could not resolve tile: {value}")


def read_patch(image_path, x, y, patch_size):
    half = patch_size // 2
    with rasterio.open(image_path) as src:
        window = rasterio.windows.Window(
            int(round(x)) - half,
            int(round(y)) - half,
            patch_size,
            patch_size,
        )
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


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def forward_features(model, x):
    if hasattr(model, "encode"):
        z = model.encode(x)
    else:
        z = model(x)

    if isinstance(z, dict):
        for k in ["features", "embedding", "embeddings", "x", "last_hidden_state"]:
            if k in z:
                z = z[k]
                break

    if isinstance(z, (tuple, list)):
        z = z[0]

    if z.ndim == 4:
        z = z.mean(dim=(2, 3))
    elif z.ndim == 3:
        z = z[:, 0]

    return z


def infer_feature_dim(model, device, image_size):
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size).to(device)
        z = forward_features(model, x)
    return int(z.shape[1])


def make_grid(cx, cy, search_radius, coarse_step, refine_radius, refine_step):
    coarse = []
    for dy in range(-search_radius, search_radius + 1, coarse_step):
        for dx in range(-search_radius, search_radius + 1, coarse_step):
            coarse.append((cx + dx, cy + dy, "coarse"))

    return coarse


def make_refine_grid(cx, cy, refine_radius, refine_step):
    pts = []
    for dy in range(-refine_radius, refine_radius + 1, refine_step):
        for dx in range(-refine_radius, refine_radius + 1, refine_step):
            pts.append((cx + dx, cy + dy, "refine"))
    return pts


@torch.no_grad()
def score_candidates(
    model,
    head,
    transform,
    image_path,
    candidates,
    target_idx,
    origin_x,
    origin_y,
    patch_size,
    beta,
    batch_size,
    device,
    use_amp,
):
    rows = []

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        imgs = []

        for x, y, stage in batch:
            patch = read_patch(image_path, x, y, patch_size)
            imgs.append(transform(patch))

        x_tensor = torch.stack(imgs).to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            z = forward_features(model, x_tensor)
            logits = head(z)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs.detach().cpu().numpy()

        for i, (x, y, stage) in enumerate(batch):
            prob = float(probs_np[i, target_idx])
            dist = float(np.sqrt((x - origin_x) ** 2 + (y - origin_y) ** 2))
            score = prob - beta * dist

            rows.append({
                "x": float(x),
                "y": float(y),
                "stage": stage,
                "target_prob": prob,
                "distance_px": dist,
                "score": score,
            })

    return rows


def map_target_label(raw_label, class_to_idx, positive_name=None, positive_class="1", negative_class="0"):
    raw = str(raw_label).strip()

    if raw in class_to_idx:
        return raw

    if positive_name is not None:
        if raw == str(positive_name):
            return str(positive_class)
        return str(negative_class)

    raise ValueError(
        f"Target label '{raw}' not found in class_to_idx. "
        f"Available: {list(class_to_idx.keys())}. "
        f"For binary, pass --binary_positive_name Shihuahuaco"
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--head_ckpt", required=True)
    p.add_argument("--points_csv", required=True)
    p.add_argument("--imagery_root", required=True)
    p.add_argument("--output_csv", required=True)

    p.add_argument("--tile_column", default="matched_tif")
    p.add_argument("--point_id_column", default="point_id")
    p.add_argument("--x_column", default="original_east")
    p.add_argument("--y_column", default="original_north")
    p.add_argument("--target_label_column", default="label")
    p.add_argument("--coord_type", choices=["world", "pixel"], default="world")

    p.add_argument("--binary_positive_name", default=None)
    p.add_argument("--binary_positive_class", default="1")
    p.add_argument("--binary_negative_class", default="0")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size_px", type=int, default=224)

    p.add_argument("--search_radius_px", type=int, default=128)
    p.add_argument("--coarse_step_px", type=int, default=16)
    p.add_argument("--refine_radius_px", type=int, default=32)
    p.add_argument("--refine_step_px", type=int, default=8)

    p.add_argument("--beta", type=float, default=0.002)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--decision_threshold", type = float, default=0.5)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    model, _ = load_encoder_from_checkpoint(args.encoder_ckpt, device)
    model.eval()

    head_ckpt = torch.load(args.head_ckpt, map_location=device)
    classes = [str(c) for c in head_ckpt["classes"]]
    class_to_idx = {str(k): int(v) for k, v in head_ckpt["class_to_idx"].items()}

    feat_dim = infer_feature_dim(model, device, args.image_size)
    head = nn.Linear(feat_dim, len(classes)).to(device)
    head.load_state_dict(head_ckpt["head_state_dict"])
    head.eval()

    transform = build_transform(args.image_size)
    resolver = TileResolver(args.imagery_root)

    df = pd.read_csv(args.points_csv)

    required = [
        args.tile_column,
        args.point_id_column,
        args.x_column,
        args.y_column,
        args.target_label_column,
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}. Available: {df.columns.tolist()}")

    results = []
    start_time = time.time()

    print("=" * 100)
    print("Classifier Phase 3 started")
    print("Encoder:", args.encoder_ckpt)
    print("Head   :", args.head_ckpt)
    print("Input  :", args.points_csv)
    print("Output :", args.output_csv)
    print("Classes:", classes)
    print("=" * 100)

    for _, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True):
        point_id = str(row[args.point_id_column])
        image_path = resolver.resolve(str(row[args.tile_column]))

        raw_x = float(row[args.x_column])
        raw_y = float(row[args.y_column])

        if args.coord_type == "world":
            origin_x, origin_y = world_to_pixel(image_path, raw_x, raw_y)
        else:
            origin_x, origin_y = raw_x, raw_y

        target_class = map_target_label(
            row[args.target_label_column],
            class_to_idx=class_to_idx,
            positive_name=args.binary_positive_name,
            positive_class=args.binary_positive_class,
            negative_class=args.binary_negative_class,
        )
        target_idx = class_to_idx[target_class]

        # Original score
        original_rows = score_candidates(
            model=model,
            head=head,
            transform=transform,
            image_path=image_path,
            candidates=[(origin_x, origin_y, "original")],
            target_idx=target_idx,
            origin_x=origin_x,
            origin_y=origin_y,
            patch_size=args.patch_size_px,
            beta=args.beta,
            batch_size=1,
            device=device,
            use_amp=use_amp,
        )
        original_score = original_rows[0]["score"]
        original_prob = original_rows[0]["target_prob"]

        # Coarse search
        coarse_candidates = make_grid(
            origin_x,
            origin_y,
            args.search_radius_px,
            args.coarse_step_px,
            args.refine_radius_px,
            args.refine_step_px,
        )

        coarse_rows = score_candidates(
            model=model,
            head=head,
            transform=transform,
            image_path=image_path,
            candidates=coarse_candidates,
            target_idx=target_idx,
            origin_x=origin_x,
            origin_y=origin_y,
            patch_size=args.patch_size_px,
            beta=args.beta,
            batch_size=args.batch_size,
            device=device,
            use_amp=use_amp,
        )

        best_coarse = max(coarse_rows, key=lambda r: r["score"])

        # Refine search
        refine_candidates = make_refine_grid(
            best_coarse["x"],
            best_coarse["y"],
            args.refine_radius_px,
            args.refine_step_px,
        )

        refine_rows = score_candidates(
            model=model,
            head=head,
            transform=transform,
            image_path=image_path,
            candidates=refine_candidates,
            target_idx=target_idx,
            origin_x=origin_x,
            origin_y=origin_y,
            patch_size=args.patch_size_px,
            beta=args.beta,
            batch_size=args.batch_size,
            device=device,
            use_amp=use_amp,
        )

        best_refine = max(refine_rows, key=lambda r: r["score"])

        refined_east, refined_north = pixel_to_world(image_path, best_refine["x"], best_refine["y"])
        coarse_east, coarse_north = pixel_to_world(image_path, best_coarse["x"], best_coarse["y"])

        out = row.to_dict()
        out.update({
            "image_path": image_path,
            "target_class_used": target_class,
            "target_idx": target_idx,

            "original_x": origin_x,
            "original_y": origin_y,
            "original_prob": original_prob,
            "original_score": original_score,

            "coarse_x": best_coarse["x"],
            "coarse_y": best_coarse["y"],
            "coarse_prob": best_coarse["target_prob"],
            "coarse_score": best_coarse["score"],
            "coarse_distance_px": best_coarse["distance_px"],
            "coarse_east": coarse_east,
            "coarse_north": coarse_north,

            "refined_x": best_refine["x"],
            "refined_y": best_refine["y"],
            "refined_prob": best_refine["target_prob"],
            "refined_score": best_refine["score"],
            "refined_distance_px": best_refine["distance_px"],
            "refined_east": refined_east,
            "refined_north": refined_north,

            "is_positive_original": int(original_prob >= args.decision_threshold),
            "is_positive_refined": int(best_refine["target_prob"] >= args.decision_threshold),
            "decision_threshold": args.decision_threshold,

            "score_gain": best_refine["score"] - original_score,
            "prob_gain": best_refine["target_prob"] - original_prob,
        })

        results.append(out)

    pd.DataFrame(results).to_csv(args.output_csv, index=False)

    print("=" * 100)
    print("Classifier Phase 3 completed")
    print("Saved:", args.output_csv)
    print("Elapsed:", format_seconds(time.time() - start_time))
    print("=" * 100)


if __name__ == "__main__":
    main()