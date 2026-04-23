"""
Phase 3: feature-guided bounded search for coordinate refinement.

This script supports two kinds of Phase 3 inputs:

1. Original pipeline input format
   - tile path already provided in a CSV column
   - coordinates are pixel coordinates
   - typical for corrected_labels.csv or old phase2 outputs

2. Evaluation input format
   - tile path stored in a matched TIFF column
   - coordinates are world coordinates (easting/northing)
   - typical for valid_points_direct.csv and recovery CSVs

The script keeps the refinement logic unchanged, and only adds:
- coordinate type handling (pixel or world)
- flexible tile column naming
- output augmentation with refined world coordinates
"""

import argparse
import os
import time
from dataclasses import dataclass

import pandas as pd
import rasterio
import torch

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.points import InputPoint, load_points_csv
from src.data.tif_io import recursive_find_tif_files
from src.data.patches import PatchExtractor, EncoderWrapper
from src.scoring.prototypes import load_prototypes_csv
from src.pipeline import FeatureGuidedBoundedSearchPipeline
from src.outputs.export_csv import save_refinement_results_csv


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def world_to_pixel(image_path: str, east: float, north: float):
    with rasterio.open(image_path) as src:
        row, col = src.index(float(east), float(north))
    return float(col), float(row)


def pixel_to_world(image_path: str, x_col: float, y_row: float):
    with rasterio.open(image_path) as src:
        east, north = src.xy(float(y_row), float(x_col))
    return float(east), float(north)


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
    coord_type: str = "pixel"   # pixel or world

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


def load_points_csv_flexible(
    csv_path: str,
    tile_column: str,
    point_id_column: str,
    x_column: str,
    y_column: str,
    target_label_column: str,
    coord_type: str,
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


def augment_output_with_world_coords(output_csv: str):
    df = pd.read_csv(output_csv)

    refined_east = []
    refined_north = []
    coarse_east = []
    coarse_north = []

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
    parser = argparse.ArgumentParser(
        description="Phase 3: feature-guided bounded search for coordinate refinement"
    )

    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--prototypes_csv", type=str, required=True)
    parser.add_argument("--points_csv", type=str, required=True)
    parser.add_argument("--imagery_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--tile_column", type=str, default="image_path")
    parser.add_argument("--point_id_column", type=str, default="point_id")
    parser.add_argument("--x_column", type=str, default="x")
    parser.add_argument("--y_column", type=str, default="y")
    parser.add_argument("--target_label_column", type=str, default="corrected_label")
    parser.add_argument("--coord_type", type=str, default="pixel", choices=["pixel", "world"])

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size_px", type=int, default=224)

    parser.add_argument("--search_radius_px", type=int, default=128)
    parser.add_argument("--coarse_step_px", type=int, default=16)
    parser.add_argument("--refine_radius_px", type=int, default=32)
    parser.add_argument("--refine_step_px", type=int, default=8)

    parser.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.002)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = Phase3Config(
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

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = config.mixed_precision and device.type == "cuda"

    start_time = time.time()

    model, _ = load_encoder_from_checkpoint(config.encoder_ckpt, device)
    encoder = EncoderWrapper(
        model=model,
        device=device,
        image_size=config.image_size,
        use_amp=use_amp,
    )

    prototypes = load_prototypes_csv(config.prototypes_csv)
    tile_resolver = TileResolver(config.imagery_root)

    # Flexible CSV loading
    points, raw_df = load_points_csv_flexible(
        csv_path=config.points_csv,
        tile_column=config.tile_column,
        point_id_column=config.point_id_column,
        x_column=config.x_column,
        y_column=config.y_column,
        target_label_column=config.target_label_column,
        coord_type=config.coord_type,
    )

    for p in points:
        p.image_path = tile_resolver.resolve(p.image_path)

    # If input coordinates are in world CRS, convert them to pixel before refinement
    if config.coord_type == "world":
        points = convert_points_world_to_pixel(points)

    patch_extractor = PatchExtractor(patch_size_px=config.patch_size_px)

    pipeline = FeatureGuidedBoundedSearchPipeline(
        encoder=encoder,
        prototypes=prototypes,
        patch_extractor=patch_extractor,
        similarity_mode=config.similarity,
        search_radius_px=config.search_radius_px,
        coarse_step_px=config.coarse_step_px,
        refine_radius_px=config.refine_radius_px,
        refine_step_px=config.refine_step_px,
        alpha=config.alpha,
        beta=config.beta,
        batch_size=config.batch_size,
    )

    print("=" * 100)
    print("Phase 3 started")
    print(f"Encoder checkpoint : {config.encoder_ckpt}")
    print(f"Prototypes CSV     : {config.prototypes_csv}")
    print(f"Points CSV         : {config.points_csv}")
    print(f"Imagery root       : {config.imagery_root}")
    print(f"Output CSV         : {config.output_csv}")
    print(f"Coordinate type    : {config.coord_type}")
    print(f"Points loaded      : {len(points)}")
    print("=" * 100)

    results = pipeline.run(points)

    save_refinement_results_csv(results, config.output_csv)
    augment_output_with_world_coords(config.output_csv)

    # Merge original input columns back in if useful
    out_df = pd.read_csv(config.output_csv)
    if "point_id" in raw_df.columns:
        raw_df["point_id"] = raw_df["point_id"].astype(str)
        out_df["point_id"] = out_df["point_id"].astype(str)
        merged = out_df.merge(raw_df, on="point_id", how="left", suffixes=("", "_input"))
        merged.to_csv(config.output_csv, index=False)

    elapsed = time.time() - start_time
    print("=" * 100)
    print("Phase 3 completed")
    print(f"Saved results to   : {config.output_csv}")
    print(f"Elapsed            : {format_seconds(elapsed)}")
    print("=" * 100)


if __name__ == "__main__":
    main()