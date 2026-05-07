import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from shapely.geometry import Point
from torchvision import transforms

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.preprocess import preprocess


@dataclass
class Candidate:
    east: float
    north: float
    prob: float


def rewrite_path(path: str, from_prefix: str, to_prefix: str) -> str:
    if from_prefix and to_prefix:
        return str(path).replace(from_prefix, to_prefix)
    return str(path)


def build_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def read_patch_world(image_path: str, east: float, north: float, patch_size_px: int) -> Image.Image:
    with rasterio.open(image_path) as src:
        row, col = src.index(float(east), float(north))
        half = patch_size_px // 2
        window = rasterio.windows.Window(
            int(round(col)) - half,
            int(round(row)) - half,
            patch_size_px,
            patch_size_px,
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

    img = Image.fromarray(arr)
    img = preprocess(img)
    return img


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


class ClassifierWrapper:
    def __init__(self, encoder_ckpt: str, head_ckpt: str, image_size: int, device: str):
        self.device = torch.device(device)
        self.encoder, _ = load_encoder_from_checkpoint(encoder_ckpt, self.device)
        self.encoder.eval()

        ckpt = torch.load(head_ckpt, map_location="cpu")
        self.classes = ckpt["classes"]
        self.class_to_idx = ckpt["class_to_idx"]
        feat_dim = ckpt["feat_dim"]

        self.head = torch.nn.Linear(feat_dim, len(self.classes))
        self.head.load_state_dict(ckpt["head_state_dict"])
        self.head.to(self.device)
        self.head.eval()

        self.transform = build_eval_transform(image_size)

    def predict_prob(self, img: Image.Image, positive_class: int) -> float:
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = forward_features(self.encoder, x)
            logits = self.head(z)
            prob = torch.softmax(logits, dim=1)[0, positive_class].item()
        return float(prob)


def make_3x3_grid(center_e: float, center_n: float, grid_size_m: float) -> List[Tuple[float, float]]:
    offsets = [-grid_size_m, 0.0, grid_size_m]
    coords = []
    for dy in offsets:
        for dx in offsets:
            coords.append((center_e + dx, center_n + dy))
    return coords


def evaluate_grid(
    model: ClassifierWrapper,
    image_path: str,
    center_e: float,
    center_n: float,
    grid_size_m: float,
    patch_size_px: int,
    positive_class: int,
) -> List[Candidate]:
    coords = make_3x3_grid(center_e, center_n, grid_size_m)
    out = []
    for e, n in coords:
        img = read_patch_world(image_path, e, n, patch_size_px)
        prob = model.predict_prob(img, positive_class)
        out.append(Candidate(east=e, north=n, prob=prob))
    return out


def slide_once(
    model: ClassifierWrapper,
    image_path: str,
    center_e: float,
    center_n: float,
    grid_size_m: float,
    threshold: float,
    min_realigned_boxes: int,
    patch_size_px: int,
    positive_class: int,
):
    cand = evaluate_grid(
        model=model,
        image_path=image_path,
        center_e=center_e,
        center_n=center_n,
        grid_size_m=grid_size_m,
        patch_size_px=patch_size_px,
        positive_class=positive_class,
    )

    selected = [c for c in cand if c.prob >= threshold]

    if len(selected) >= min_realigned_boxes:
        new_e = float(np.mean([c.east for c in selected]))
        new_n = float(np.mean([c.north for c in selected]))
        moved = True
    else:
        best = max(cand, key=lambda x: x.prob)
        new_e = best.east
        new_n = best.north
        moved = False

    return cand, new_e, new_n, moved


def final_local_refine(
    model: ClassifierWrapper,
    image_path: str,
    center_e: float,
    center_n: float,
    radius_m: float,
    step_m: float,
    patch_size_px: int,
    positive_class: int,
):
    best_prob = -1.0
    best_e = center_e
    best_n = center_n

    ys = np.arange(-radius_m, radius_m + 1e-9, step_m)
    xs = np.arange(-radius_m, radius_m + 1e-9, step_m)

    for dy in ys:
        for dx in xs:
            e = center_e + dx
            n = center_n + dy
            img = read_patch_world(image_path, e, n, patch_size_px)
            prob = model.predict_prob(img, positive_class)
            if prob > best_prob:
                best_prob = prob
                best_e = e
                best_n = n

    return best_e, best_n, best_prob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_shp", required=True)
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--head_ckpt", required=True)
    p.add_argument("--imagery_root", required=False, default="")
    p.add_argument("--tile_column", default="matched_tif")
    p.add_argument("--label_column", default="label")
    p.add_argument("--target_label", required=True)
    p.add_argument("--x_column", default="original_east")
    p.add_argument("--y_column", default="original_north")
    p.add_argument("--crs", default="EPSG:32718")
    p.add_argument("--grid_sizes", default="20,10,5")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--min_realigned_boxes", type=int, default=3)
    p.add_argument("--final_refine_radius_m", type=float, default=3.0)
    p.add_argument("--final_refine_step_m", type=float, default=0.5)
    p.add_argument("--max_iterations", type=int, default=6)
    p.add_argument("--positive_class", type=int, default=1)
    p.add_argument("--patch_size_px", type=int, default=224)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--path_rewrite_from", default="")
    p.add_argument("--path_rewrite_to", default="")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_shp), exist_ok=True)

    df = pd.read_csv(args.input_csv).copy()
    df[args.label_column] = df[args.label_column].astype(str).str.strip()

    df = df[df[args.label_column] == str(args.target_label).strip()].copy()
    print(f"[INFO] Filtered rows for target_label={args.target_label}: {len(df)}")

    if len(df) == 0:
        raise RuntimeError("No rows left after target_label filtering.")

    model = ClassifierWrapper(
        encoder_ckpt=args.encoder_ckpt,
        head_ckpt=args.head_ckpt,
        image_size=args.image_size,
        device=args.device,
    )

    grid_sizes = [float(x) for x in args.grid_sizes.split(",") if str(x).strip()]

    rows = []

    for i, row in df.iterrows():
        image_path = str(row[args.tile_column])
        image_path = rewrite_path(image_path, args.path_rewrite_from, args.path_rewrite_to)

        if not os.path.exists(image_path):
            rows.append({
                "point_id": row.get("point_id", f"row_{i}"),
                "status": f"missing_image: {image_path}",
            })
            continue

        cur_e = float(row[args.x_column])
        cur_n = float(row[args.y_column])

        start_e = cur_e
        start_n = cur_n

        all_stage_max_probs = []

        for _ in range(args.max_iterations):
            moved_any = False
            for g in grid_sizes:
                cand, new_e, new_n, moved = slide_once(
                    model=model,
                    image_path=image_path,
                    center_e=cur_e,
                    center_n=cur_n,
                    grid_size_m=g,
                    threshold=args.threshold,
                    min_realigned_boxes=args.min_realigned_boxes,
                    patch_size_px=args.patch_size_px,
                    positive_class=args.positive_class,
                )
                cur_e = new_e
                cur_n = new_n
                all_stage_max_probs.append(max(c.prob for c in cand))
                moved_any = moved_any or moved

            if not moved_any:
                break

        final_e, final_n, final_prob = final_local_refine(
            model=model,
            image_path=image_path,
            center_e=cur_e,
            center_n=cur_n,
            radius_m=args.final_refine_radius_m,
            step_m=args.final_refine_step_m,
            patch_size_px=args.patch_size_px,
            positive_class=args.positive_class,
        )

        rows.append({
            "point_id": row.get("point_id", f"row_{i}"),
            "label": row[args.label_column],
            "image_path": image_path,
            "start_east": start_e,
            "start_north": start_n,
            "coarse_final_east": cur_e,
            "coarse_final_north": cur_n,
            "final_east": final_e,
            "final_north": final_n,
            "final_refined_prob": final_prob,
            "max_stage_prob": float(np.max(all_stage_max_probs)) if all_stage_max_probs else np.nan,
            "mean_stage_prob": float(np.mean(all_stage_max_probs)) if all_stage_max_probs else np.nan,
            "status": "ok",
        })

    out = pd.DataFrame(rows)
    out_csv = os.path.splitext(args.output_shp)[0] + ".csv"
    out.to_csv(out_csv, index=False)
    print(f"[INFO] Saved CSV: {out_csv}")

    ok = out[out["status"] == "ok"].copy()
    if len(ok) > 0:
        gdf = gpd.GeoDataFrame(
            ok,
            geometry=gpd.points_from_xy(ok["final_east"], ok["final_north"]),
            crs=args.crs,
        )
        gdf.to_file(args.output_shp)
        print(f"[INFO] Saved SHP: {args.output_shp}")
    else:
        print("[WARN] No successful rows. SHP not written.")


if __name__ == "__main__":
    main()