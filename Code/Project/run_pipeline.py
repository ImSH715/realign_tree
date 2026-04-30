"""
Phase 3: feature-guided bounded search for coordinate refinement.

Supports:
1. normal class_prototypes.csv
2. multi_class_prototypes.csv with columns:
   prototype_id,label,cluster_id,n_source,emb_0...
"""

import argparse
import os
import time
from dataclasses import dataclass

import pandas as pd
import rasterio
import torch

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.points import InputPoint
from src.data.tif_io import recursive_find_tif_files
from src.data.patches import PatchExtractor, EncoderWrapper
from pipeline import FeatureGuidedBoundedSearchPipeline
from src.outputs.export_csv import save_refinement_results_csv


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def world_to_pixel(image_path: str, east: float, north: float):
    with rasterio.open(image_path) as src:
        row, col = src.index(float(east), float(north))
    return float(col), float(row)


def pixel_to_world(image_path: str, x_col: float, y_row: float):
    with rasterio.open(image_path) as src:
        east, north = src.xy(float(y_row), float(x_col))
    return float(east), float(north)


def load_prototypes_flexible(prototypes_csv: str):
    """
    Supports both:
    A) class_prototypes.csv:
       class, emb_0, emb_1, ...

    B) multi_class_prototypes.csv:
       prototype_id, label, cluster_id, n_source, emb_0, emb_1, ...

    Output format:
       dict[label_or_proto_id] = tensor/list embedding
    """
    df = pd.read_csv(prototypes_csv)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError(
            f"No embedding columns found in {prototypes_csv}. "
            f"Expected columns like emb_0, emb_1, ..."
        )

    prototypes = {}

    if "prototype_id" in df.columns and "label" in df.columns:
        print("[INFO] Loading multi-prototype CSV")
        for _, row in df.iterrows():
            proto_id = str(row["prototype_id"])
            label = str(row["label"])
            key = proto_id

            prototypes[key] = {
                "label": label,
                "embedding": row[emb_cols].values.astype("float32"),
            }

    else:
        print("[INFO] Loading standard class prototype CSV")
        label_col = df.columns[0]
        for _, row in df.iterrows():
            label = str(row[label_col]).strip()
            prototypes[label] = row[emb_cols].values.astype("float32")

    print(f"[INFO] Loaded prototypes: {len(prototypes)}")
    return prototypes


class TileResolver:
    def __init__(self, imagery_root: str):
        self.lookup = {}
        tif_paths = recursive_find_tif_files(imagery_root)

        for p in tif_paths:
            ap = os.path.abspath(p)
            self.lookup[ap] = ap
            self.lookup[os.path.basename(ap)] = ap

    def resolve(self, value: str) -> str:
        if value in self.lookup:
            return self.lookup[value]

        abs_value = os.path.abspath(value)
        if abs_value in self.lookup:
            return self.lookup[abs_value]

        base = os.path.basename(value)
        if base in self.lookup:
            return self.lookup[base]

        raise FileNotFoundError(f"Could not resolve imagery path: {value}")


@dataclass
class Phase3Config:
    encoder_ckpt: str
    prototypes_csv: str
    points_csv: str
    imagery_root: str
    output_csv: str

    tile_column: str = "image_path"
    point_id_column: str = "point_id"
    x_column: str = "x"
    y_column: str = "y"
    target_label_column: str = "target_label"
    coord_type: str = "pixel"

    image_size: int = 224
    patch_size_px: int = 224

    search_radius_px: int = 128
    coarse_step_px: int = 16
    refine_radius_px: int = 32
    refine_step_px: int = 8

    similarity: str = "cosine"
    alpha: float = 1.0
    beta: float = 0.002

    batch_size: int = 64
    device: str = "cuda"
    mixed_precision: bool = True


def load_points_csv_flexible(
    csv_path,
    tile_column,
    point_id_column,
    x_column,
    y_column,
    target_label_column,
):
    df = pd.read_csv(csv_path)

    required = [tile_column, point_id_column, x_column, y_column, target_label_column]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")

    points = []

    for _, row in df.iterrows():
        points.append(
            InputPoint(
                point_id=str(row[point_id_column]),
                image_path=str(row[tile_column]),
                x=float(row[x_column]),
                y=float(row[y_column]),
                target_label=str(row[target_label_column]).strip(),
            )
        )

    return points, df


def convert_points_world_to_pixel(points):
    converted = []

    for p in points:
        x_px, y_px = world_to_pixel(p.image_path, p.x, p.y)
        converted.append(
            InputPoint(
                point_id=p.point_id,
                image_path=p.image_path,
                x=x_px,
                y=y_px,
                target_label=p.target_label,
            )
        )

    return converted


def augment_output_with_world_coords(output_csv):
    df = pd.read_csv(output_csv)

    refined_east, refined_north = [], []
    coarse_east, coarse_north = [], []

    for _, row in df.iterrows():
        try:
            re, rn = pixel_to_world(row["image_path"], row["refined_x"], row["refined_y"])
        except Exception:
            re, rn = float("nan"), float("nan")

        try:
            ce, cn = pixel_to_world(row["image_path"], row["coarse_x"], row["coarse_y"])
        except Exception:
            ce, cn = float("nan"), float("nan")

        refined_east.append(re)
        refined_north.append(rn)
        coarse_east.append(ce)
        coarse_north.append(cn)

    df["refined_east"] = refined_east
    df["refined_north"] = refined_north
    df["coarse_east"] = coarse_east
    df["coarse_north"] = coarse_north

    df.to_csv(output_csv, index=False)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--prototypes_csv", required=True)
    p.add_argument("--points_csv", required=True)
    p.add_argument("--imagery_root", required=True)
    p.add_argument("--output_csv", required=True)

    p.add_argument("--tile_column", default="image_path")
    p.add_argument("--point_id_column", default="point_id")
    p.add_argument("--x_column", default="x")
    p.add_argument("--y_column", default="y")
    p.add_argument("--target_label_column", default="corrected_label")
    p.add_argument("--coord_type", default="pixel", choices=["pixel", "world"])

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size_px", type=int, default=224)

    p.add_argument("--search_radius_px", type=int, default=128)
    p.add_argument("--coarse_step_px", type=int, default=16)
    p.add_argument("--refine_radius_px", type=int, default=32)
    p.add_argument("--refine_step_px", type=int, default=8)

    p.add_argument("--similarity", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.002)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Phase3Config(
        encoder_ckpt=args.encoder_ckpt,
        prototypes_csv=args.prototypes_csv,
        points_csv=args.points_csv,
        imagery_root=args.imagery_root,
        output_csv=args.output_csv,
        tile_column=args.tile_column,
        point_id_column=args.point_id_column,
        x_column=args.x_column,
        y_column=args.y_column,
        target_label_column=args.target_label_column,
        coord_type=args.coord_type,
        image_size=args.image_size,
        patch_size_px=args.patch_size_px,
        search_radius_px=args.search_radius_px,
        coarse_step_px=args.coarse_step_px,
        refine_radius_px=args.refine_radius_px,
        refine_step_px=args.refine_step_px,
        similarity=args.similarity,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
        device=args.device,
        mixed_precision=not args.no_amp,
    )

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = cfg.mixed_precision and device.type == "cuda"

    start = time.time()

    model, _ = load_encoder_from_checkpoint(cfg.encoder_ckpt, device)

    encoder = EncoderWrapper(
        model=model,
        device=device,
        image_size=cfg.image_size,
        use_amp=use_amp,
    )

    prototypes = load_prototypes_flexible(cfg.prototypes_csv)

    tile_resolver = TileResolver(cfg.imagery_root)

    points, raw_df = load_points_csv_flexible(
        csv_path=cfg.points_csv,
        tile_column=cfg.tile_column,
        point_id_column=cfg.point_id_column,
        x_column=cfg.x_column,
        y_column=cfg.y_column,
        target_label_column=cfg.target_label_column,
    )

    for p in points:
        p.image_path = tile_resolver.resolve(p.image_path)

    if cfg.coord_type == "world":
        points = convert_points_world_to_pixel(points)

    patch_extractor = PatchExtractor(patch_size_px=cfg.patch_size_px)

    pipeline = FeatureGuidedBoundedSearchPipeline(
        encoder=encoder,
        prototypes=prototypes,
        patch_extractor=patch_extractor,
        similarity_mode=cfg.similarity,
        search_radius_px=cfg.search_radius_px,
        coarse_step_px=cfg.coarse_step_px,
        refine_radius_px=cfg.refine_radius_px,
        refine_step_px=cfg.refine_step_px,
        alpha=cfg.alpha,
        beta=cfg.beta,
        batch_size=cfg.batch_size,
    )

    print("=" * 100)
    print("Phase 3 started")
    print(f"Encoder checkpoint : {cfg.encoder_ckpt}")
    print(f"Prototypes CSV     : {cfg.prototypes_csv}")
    print(f"Points CSV         : {cfg.points_csv}")
    print(f"Output CSV         : {cfg.output_csv}")
    print(f"Coordinate type    : {cfg.coord_type}")
    print(f"Points loaded      : {len(points)}")
    print("=" * 100)

    results = pipeline.run(points)

    save_refinement_results_csv(results, cfg.output_csv)
    augment_output_with_world_coords(cfg.output_csv)

    out_df = pd.read_csv(cfg.output_csv)

    if "point_id" in raw_df.columns:
        raw_df["point_id"] = raw_df["point_id"].astype(str)
        out_df["point_id"] = out_df["point_id"].astype(str)

        merged = out_df.merge(
            raw_df,
            on="point_id",
            how="left",
            suffixes=("", "_input"),
        )
        merged.to_csv(cfg.output_csv, index=False)

    elapsed = time.time() - start

    print("=" * 100)
    print("Phase 3 completed")
    print(f"Saved results to: {cfg.output_csv}")
    print(f"Elapsed         : {format_seconds(elapsed)}")
    print("=" * 100)


if __name__ == "__main__":
    main()