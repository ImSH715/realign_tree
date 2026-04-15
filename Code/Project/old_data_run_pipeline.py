import argparse
import os
import re
import time
from dataclasses import dataclass

import torch
import pandas as pd
import rasterio

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.points import InputPoint
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


class CensusTileResolver:
    def __init__(self, imagery_root: str):
        self.tif_paths = recursive_find_tif_files(imagery_root)
        self.index = {}

        for p in self.tif_paths:
            key = self._extract_key(os.path.basename(p))
            if key is not None and key not in self.index:
                self.index[key] = p

        print(f"[INFO] Indexed TIFF files: {len(self.tif_paths)}")
        print(f"[INFO] Indexed FAJA/PCA keys: {len(self.index)}")

    def _extract_key(self, filename: str):
        name = os.path.basename(filename).lower()
        stem = os.path.splitext(name)[0]

        m = re.search(r"pv\s*(\d+)\s*[-_]\s*0*(\d+)", stem, re.IGNORECASE)
        if m:
            faja = int(m.group(1))
            pca = int(m.group(2))
            return (faja, pca)

        return None

    def resolve(self, faja, pca):
        faja_num = int(str(faja).strip())
        pca_str = str(pca).lower().replace("pc", "").strip()
        pca_num = int(pca_str)

        key = (faja_num, pca_num)

        if key in self.index:
            return self.index[key]

        raise FileNotFoundError(f"No TIFF found for FAJA={faja_num}, PCA={pca_num}")


def load_censo_points(
    csv_path,
    imagery_root,
    label_column,
    x_column,
    y_column,
    faja_column,
    pca_column,
    filter_label=None,
):
    df = pd.read_csv(csv_path)
    df[label_column] = df[label_column].astype(str).str.strip()

    if filter_label:
        df = df[df[label_column].str.lower() == filter_label.strip().lower()].copy()

    resolver = CensusTileResolver(imagery_root)
    points = []

    for i, row in df.iterrows():
        try:
            image_path = resolver.resolve(row[faja_column], row[pca_column])

            with rasterio.open(image_path) as src:
                r, c = src.index(float(row[x_column]), float(row[y_column]))

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
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--prototypes_csv", required=True)
    parser.add_argument("--points_csv", required=True)
    parser.add_argument("--imagery_root", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--label_column", required=True)
    parser.add_argument("--x_column", required=True)
    parser.add_argument("--y_column", required=True)
    parser.add_argument("--faja_column", required=True)
    parser.add_argument("--pca_column", required=True)

    parser.add_argument("--filter_label", default="")

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
    use_amp = (not args.no_amp and device.type == "cuda")

    model, _ = load_encoder_from_checkpoint(args.encoder_ckpt, device)

    encoder = EncoderWrapper(
        model=model,
        device=device,
        image_size=224,
        use_amp=use_amp,
    )

    prototypes = load_prototypes_csv(args.prototypes_csv)

    points = load_censo_points(
        csv_path=args.points_csv,
        imagery_root=args.imagery_root,
        label_column=args.label_column,
        x_column=args.x_column,
        y_column=args.y_column,
        faja_column=args.faja_column,
        pca_column=args.pca_column,
        filter_label=args.filter_label,
    )

    extractor = PatchExtractor(224)

    pipeline = FeatureGuidedBoundedSearchPipeline(
        encoder=encoder,
        prototypes=prototypes,
        patch_extractor=extractor,
        similarity_mode=args.similarity,
        search_radius_px=args.search_radius_px,
        coarse_step_px=args.coarse_step_px,
        refine_radius_px=args.refine_radius_px,
        refine_step_px=args.refine_step_px,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
    )

    print("=" * 80)
    print("Phase 3 started")
    print(f"Encoder checkpoint : {args.encoder_ckpt}")
    print(f"Prototypes CSV     : {args.prototypes_csv}")
    print(f"Points CSV         : {args.points_csv}")
    print(f"Imagery root       : {args.imagery_root}")
    print(f"Output CSV         : {args.output_csv}")
    print(f"Points loaded      : {len(points)}")
    print("=" * 80)

    start = time.time()
    results = pipeline.run(points)
    save_refinement_results_csv(results, args.output_csv)

    print("=" * 80)
    print("Phase 3 completed")
    print(f"Saved: {args.output_csv}")
    print(f"Time: {format_seconds(time.time() - start)}")
    print("=" * 80)


if __name__ == "__main__":
    main()