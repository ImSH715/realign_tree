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
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


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

    filter_label: str = ""


def load_censo_points(
    csv_path: str,
    label_column: str,
    x_column: str,
    y_column: str,
    image_column: str,
    filter_label: str = "",
):
    df = pd.read_csv(csv_path)

    required = [label_column, x_column, y_column, image_column]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}' in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )

    df[label_column] = df[label_column].astype(str).str.strip()

    if filter_label:
        df = df[df[label_column].str.lower() == filter_label.strip().lower()].copy()

    points = []

    for i, row in df.iterrows():
        try:
            image_path = str(row[image_column]).strip()
            x_world = float(row[x_column])
            y_world = float(row[y_column])

            with rasterio.open(image_path) as src:
                r, c = src.index(x_world, y_world)

            points.append(
                InputPoint(
                    point_id=f"pt_{i}",
                    image_path=image_path,
                    x=float(c),
                    y=float(r),
                    target_label=str(row[label_column]).strip(),
                )
            )

        except Exception as e:
            print(f"[WARN] Skipping row {i}: {e}")

    print(f"[INFO] Loaded points: {len(points)}")
    return points


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3: bounded-search refinement on overlap-only census points"
    )

    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--prototypes_csv", required=True)
    parser.add_argument("--points_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--label_column", required=True)
    parser.add_argument("--x_column", required=True)
    parser.add_argument("--y_column", required=True)
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

    config = Phase3Config(
        encoder_ckpt=args.encoder_ckpt,
        prototypes_csv=args.prototypes_csv,
        points_csv=args.points_csv,
        output_csv=args.output_csv,
        label_column=args.label_column,
        x_column=args.x_column,
        y_column=args.y_column,
        image_column=args.image_column,
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

    points = load_censo_points(
        csv_path=config.points_csv,
        label_column=config.label_column,
        x_column=config.x_column,
        y_column=config.y_column,
        image_column=config.image_column,
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

    print("=" * 80)
    print("Phase 3 completed")
    print(f"Saved: {config.output_csv}")
    print(f"Time: {format_seconds(time.time() - start)}")
    print("=" * 80)


if __name__ == "__main__":
    main()