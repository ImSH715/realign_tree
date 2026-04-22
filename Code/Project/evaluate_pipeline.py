"""
Run the existing Phase 3 refinement logic on evaluation CSV files.

This script is evaluation-oriented and does not retrain the model. It:
- loads an already trained encoder checkpoint,
- loads class prototypes,
- reads evaluation points from a CSV,
- converts world coordinates to image pixel coordinates,
- runs bounded local search refinement,
- saves refined outputs,
- and augments the output with world coordinates for QGIS / evaluation use.

It supports direct GT evaluation inputs, recovery-test inputs, and no-GT overlap inputs,
as long as the input CSV follows the expected schema.
"""

import argparse
import time
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


def load_eval_points(
    csv_path: str,
    label_column: str,
    x_column: str,
    y_column: str,
    image_column: str,
    point_id_column: str,
    filter_label: str = "",
):
    df = pd.read_csv(csv_path)

    required = [point_id_column, label_column, x_column, y_column, image_column]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in input CSV.")

    df[label_column] = df[label_column].astype(str).str.strip()

    if filter_label:
        df = df[df[label_column].str.lower() == filter_label.strip().lower()].copy()

    points = []
    rows = []

    for _, row in df.iterrows():
        point_id = str(row[point_id_column])
        label = str(row[label_column]).strip()
        image_path = str(row[image_column]).strip()
        east = float(row[x_column])
        north = float(row[y_column])

        try:
            x_px, y_px = world_to_pixel(image_path, east, north)
        except Exception as e:
            print(f"[WARN] Could not convert world->pixel for {point_id}: {e}")
            continue

        points.append(
            InputPoint(
                point_id=point_id,
                image_path=image_path,
                x=x_px,
                y=y_px,
                target_label=label,
            )
        )
        rows.append(row)

    meta_df = pd.DataFrame(rows)
    return points, meta_df


def augment_output_with_world_coords(output_csv: str, input_meta_df: pd.DataFrame):
    df = pd.read_csv(output_csv)

    input_meta_df = input_meta_df.copy()
    input_meta_df["point_id"] = input_meta_df["point_id"].astype(str)

    merged = df.merge(
        input_meta_df,
        on="point_id",
        how="left",
        suffixes=("", "_input"),
    )

    refined_east = []
    refined_north = []
    coarse_east = []
    coarse_north = []

    for _, row in merged.iterrows():
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

    merged["refined_east"] = refined_east
    merged["refined_north"] = refined_north
    merged["coarse_east"] = coarse_east
    merged["coarse_north"] = coarse_north

    merged.to_csv(output_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 3 evaluation pipeline on prepared evaluation CSV.")

    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--prototypes_csv", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--point_id_column", default="point_id")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--x_column", default="original_east")
    parser.add_argument("--y_column", default="original_north")
    parser.add_argument("--image_column", default="matched_tif")
    parser.add_argument("--filter_label", default="")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size_px", type=int, default=224)
    parser.add_argument("--search_radius_px", type=int, default=128)
    parser.add_argument("--coarse_step_px", type=int, default=16)
    parser.add_argument("--refine_radius_px", type=int, default=32)
    parser.add_argument("--refine_step_px", type=int, default=8)

    parser.add_argument("--similarity", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.002)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    model, _ = load_encoder_from_checkpoint(args.encoder_ckpt, device)

    encoder = EncoderWrapper(
        model=model,
        device=device,
        image_size=args.image_size,
        use_amp=use_amp,
    )

    prototypes = load_prototypes_csv(args.prototypes_csv)

    points, meta_df = load_eval_points(
        csv_path=args.input_csv,
        point_id_column=args.point_id_column,
        label_column=args.label_column,
        x_column=args.x_column,
        y_column=args.y_column,
        image_column=args.image_column,
        filter_label=args.filter_label,
    )

    patch_extractor = PatchExtractor(args.patch_size_px)

    pipeline = FeatureGuidedBoundedSearchPipeline(
        encoder=encoder,
        prototypes=prototypes,
        patch_extractor=patch_extractor,
        similarity_mode=args.similarity,
        search_radius_px=args.search_radius_px,
        coarse_step_px=args.coarse_step_px,
        refine_radius_px=args.refine_radius_px,
        refine_step_px=args.refine_step_px,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
    )

    print("=" * 100)
    print("Evaluation Phase 3 started")
    print(f"Input CSV          : {args.input_csv}")
    print(f"Output CSV         : {args.output_csv}")
    print(f"Points loaded      : {len(points)}")
    print(f"Filter label       : {args.filter_label}")
    print("=" * 100)

    start = time.time()
    results = pipeline.run(points)
    save_refinement_results_csv(results, args.output_csv)
    augment_output_with_world_coords(args.output_csv, meta_df)

    elapsed = time.time() - start

    print("=" * 100)
    print("Evaluation Phase 3 completed")
    print(f"Saved output CSV   : {args.output_csv}")
    print(f"Elapsed            : {format_seconds(elapsed)}")
    print("=" * 100)


if __name__ == "__main__":
    main()