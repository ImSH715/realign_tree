"""
Phase 3 refinement pipeline for old/Turing_L Censo data.

This script is designed for Censo overlap CSV files where each row already has:
- a world-coordinate point, e.g. COORDENADA_ESTE / COORDENADA_NORTE
- a label, e.g. NOMBRE_COMUN
- a matched TIFF path, e.g. matched_tif

Mechanism:
1. Read world coordinates from CSV.
2. Open the matched TIFF for each point.
3. Convert world coordinates to pixel coordinates with rasterio.index().
4. Run Phase 3 feature-guided bounded search in pixel space.
5. Convert original/coarse/refined pixel coordinates back to world coordinates.
6. Save output CSV with both pixel and world coordinates.

This prevents the previous bug where world coordinates such as 458000/9150000
were accidentally treated as pixel coordinates.
"""

import argparse
import time
from dataclasses import dataclass

import pandas as pd
import rasterio
import torch

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.points import InputPoint
from src.data.patches import PatchExtractor, EncoderWrapper
from src.scoring.prototypes import load_prototypes_csv
from src.pipeline import FeatureGuidedBoundedSearchPipeline
from src.outputs.export_csv import save_refinement_results_csv


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def pixel_to_world(image_path: str, col: float, row: float):
    with rasterio.open(image_path) as src:
        east, north = src.xy(float(row), float(col))
    return float(east), float(north)


def world_to_pixel_checked(image_path: str, east: float, north: float):
    with rasterio.open(image_path) as src:
        row, col = src.index(float(east), float(north))

        inside = (0 <= col < src.width) and (0 <= row < src.height)
        if not inside:
            raise ValueError(
                f"world point outside TIFF after conversion: "
                f"east={east}, north={north}, col={col}, row={row}, "
                f"width={src.width}, height={src.height}, bounds={src.bounds}"
            )

    return float(col), float(row)


@dataclass
class Phase3Config:
    encoder_ckpt: str
    prototypes_csv: str
    points_csv: str
    output_csv: str

    label_column: str
    x_column: str
    y_column: str
    image_column: str
    point_id_column: str = ""

    image_size: int = 224
    patch_size_px: int = 224

    search_radius_px: int = 192
    coarse_step_px: int = 8
    refine_radius_px: int = 48
    refine_step_px: int = 4

    similarity: str = "cosine"
    alpha: float = 1.0
    beta: float = 0.0002

    batch_size: int = 32
    device: str = "cuda"
    mixed_precision: bool = True
    filter_label: str = ""


def load_target_points(
    csv_path: str,
    label_column: str,
    x_column: str,
    y_column: str,
    image_column: str,
    point_id_column: str = "",
    filter_label: str = "",
):
    df = pd.read_csv(csv_path)

    required = [label_column, x_column, y_column, image_column]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Available columns: {df.columns.tolist()}")

    df[label_column] = df[label_column].astype(str).str.strip()
    df[image_column] = df[image_column].astype(str).str.strip()

    if filter_label:
        df = df[df[label_column].str.lower() == filter_label.strip().lower()].copy()

    points = []
    kept_rows = []
    skipped = 0

    for i, row in df.iterrows():
        try:
            image_path = str(row[image_column]).strip()
            east = float(row[x_column])
            north = float(row[y_column])
            label = str(row[label_column]).strip()

            col, pix_row = world_to_pixel_checked(image_path, east, north)

            if point_id_column and point_id_column in df.columns:
                point_id = str(row[point_id_column])
            elif "point_id" in df.columns:
                point_id = str(row["point_id"])
            elif "index" in df.columns:
                point_id = str(row["index"])
            else:
                point_id = f"pt_{i}"

            points.append(
                InputPoint(
                    point_id=point_id,
                    image_path=image_path,
                    x=col,
                    y=pix_row,
                    target_label=label,
                )
            )

            kept = row.copy()
            kept["_phase3_point_id"] = point_id
            kept["_input_pixel_x"] = col
            kept["_input_pixel_y"] = pix_row
            kept_rows.append(kept)

        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping row {i}: {e}")

    meta_df = pd.DataFrame(kept_rows)

    print(f"[INFO] Loaded points: {len(points)}")
    print(f"[INFO] Skipped rows : {skipped}")
    return points, meta_df


def add_world_coords_to_output(output_csv: str):
    df = pd.read_csv(output_csv)

    original_east, original_north = [], []
    refined_east, refined_north = [], []
    coarse_east, coarse_north = [], []

    for _, row in df.iterrows():
        image_path = row["image_path"]

        try:
            xw, yw = pixel_to_world(image_path, row["original_x"], row["original_y"])
        except Exception:
            xw, yw = float("nan"), float("nan")
        original_east.append(xw)
        original_north.append(yw)

        try:
            xw, yw = pixel_to_world(image_path, row["refined_x"], row["refined_y"])
        except Exception:
            xw, yw = float("nan"), float("nan")
        refined_east.append(xw)
        refined_north.append(yw)

        try:
            xw, yw = pixel_to_world(image_path, row["coarse_x"], row["coarse_y"])
        except Exception:
            xw, yw = float("nan"), float("nan")
        coarse_east.append(xw)
        coarse_north.append(yw)

    df["original_east"] = original_east
    df["original_north"] = original_north
    df["refined_east"] = refined_east
    df["refined_north"] = refined_north
    df["coarse_east"] = coarse_east
    df["coarse_north"] = coarse_north

    df.to_csv(output_csv, index=False)


def merge_original_csv_columns(output_csv: str, meta_df: pd.DataFrame):
    if meta_df.empty:
        return

    out = pd.read_csv(output_csv)

    meta = meta_df.copy()
    meta["_phase3_point_id"] = meta["_phase3_point_id"].astype(str)
    out["point_id"] = out["point_id"].astype(str)

    merged = out.merge(
        meta,
        left_on="point_id",
        right_on="_phase3_point_id",
        how="left",
        suffixes=("", "_input"),
    )

    merged.to_csv(output_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 refinement for old/Turing_L Censo overlap data"
    )

    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--prototypes_csv", required=True)
    parser.add_argument("--points_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--label_column", required=True)
    parser.add_argument("--x_column", required=True)
    parser.add_argument("--y_column", required=True)
    parser.add_argument("--image_column", default="matched_tif")
    parser.add_argument("--point_id_column", default="")
    parser.add_argument("--filter_label", default="")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size_px", type=int, default=224)

    parser.add_argument("--search_radius_px", type=int, default=192)
    parser.add_argument("--coarse_step_px", type=int, default=8)
    parser.add_argument("--refine_radius_px", type=int, default=48)
    parser.add_argument("--refine_step_px", type=int, default=4)

    parser.add_argument("--similarity", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0002)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = Phase3Config(
        encoder_ckpt=args.encoder_ckpt,
        prototypes_csv=args.prototypes_csv,
        points_csv=args.points_csv,
        output_csv=args.output_csv,
        label_column=args.label_column,
        x_column=args.x_column,
        y_column=args.y_column,
        image_column=args.image_column,
        point_id_column=args.point_id_column,
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
        filter_label=args.filter_label,
    )

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = config.mixed_precision and device.type == "cuda"

    model, _ = load_encoder_from_checkpoint(config.encoder_ckpt, device)

    encoder = EncoderWrapper(
        model=model,
        device=device,
        image_size=config.image_size,
        use_amp=use_amp,
    )

    prototypes = load_prototypes_csv(config.prototypes_csv)

    points, meta_df = load_target_points(
        csv_path=config.points_csv,
        label_column=config.label_column,
        x_column=config.x_column,
        y_column=config.y_column,
        image_column=config.image_column,
        point_id_column=config.point_id_column,
        filter_label=config.filter_label,
    )

    extractor = PatchExtractor(config.patch_size_px)

    pipeline = FeatureGuidedBoundedSearchPipeline(
        encoder=encoder,
        prototypes=prototypes,
        patch_extractor=extractor,
        similarity_mode=config.similarity,
        search_radius_px=config.search_radius_px,
        coarse_step_px=config.coarse_step_px,
        refine_radius_px=config.refine_radius_px,
        refine_step_px=config.refine_step_px,
        alpha=config.alpha,
        beta=config.beta,
        batch_size=config.batch_size,
    )

    print("=" * 80)
    print("Phase 3 started")
    print(f"Encoder checkpoint : {config.encoder_ckpt}")
    print(f"Prototypes CSV     : {config.prototypes_csv}")
    print(f"Points CSV         : {config.points_csv}")
    print(f"Output CSV         : {config.output_csv}")
    print(f"Points loaded      : {len(points)}")
    print(f"Filter label       : {config.filter_label}")
    print("=" * 80)

    start = time.time()

    results = pipeline.run(points)

    save_refinement_results_csv(results, config.output_csv)
    add_world_coords_to_output(config.output_csv)
    merge_original_csv_columns(config.output_csv, meta_df)

    print("=" * 80)
    print("Phase 3 completed")
    print(f"Saved: {config.output_csv}")
    print(f"Time: {format_seconds(time.time() - start)}")
    print("=" * 80)


if __name__ == "__main__":
    main()