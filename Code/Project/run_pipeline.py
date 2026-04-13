import argparse
import os
import time
from dataclasses import asdict, dataclass

import torch

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.points import load_points_csv
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

    points = load_points_csv(
        csv_path=config.points_csv,
        tile_column=config.tile_column,
        point_id_column=config.point_id_column,
        x_column=config.x_column,
        y_column=config.y_column,
        target_label_column=config.target_label_column,
    )

    for p in points:
        p.image_path = tile_resolver.resolve(p.image_path)

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
    print(f"Points loaded      : {len(points)}")
    print("=" * 100)

    results = pipeline.run(points)

    save_refinement_results_csv(results, config.output_csv)

    elapsed = time.time() - start_time
    print("=" * 100)
    print("Phase 3 completed")
    print(f"Saved results to   : {config.output_csv}")
    print(f"Elapsed            : {format_seconds(elapsed)}")
    print("=" * 100)


if __name__ == "__main__":
    main()