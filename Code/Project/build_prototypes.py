import os
import json
import time
import argparse
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.gt_shp_dataset import ShapefilePointDataset
from src.scoring.prototypes import (
    extract_ground_truth_embeddings,
    read_embedding_csv_rows,
    compute_class_prototypes,
    save_prototypes_csv,
    save_internal_semantic_mapping,
    save_corrected_rows_csv,
)
from src.scoring.ranker import (
    build_internal_to_semantic_mapping,
    correct_labels,
)


def set_seed(seed: int) -> None:
    import random
    import numpy as np
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


@dataclass
class Phase2Config:
    phase1_ckpt: str
    phase1_embedding_csv: str
    gt_path: str
    gt_type: str
    gt_label_field: str
    gt_folder_field: str
    gt_file_field: str
    gt_fx_field: str
    gt_fy_field: str
    imagery_root: str
    output_dir: str

    image_size: int = 224
    patch_size_px: int = 224
    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    seed: int = 42
    mixed_precision: bool = True

    similarity: str = "cosine"
    confidence_threshold: float = 0.0
    correction_margin: float = 0.0


def build_eval_transform(image_size: int) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: build semantic prototypes and correct labels from SHP ground truth"
    )

    parser.add_argument("--phase1_ckpt", type=str, required=True)
    parser.add_argument("--phase1_embedding_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--gt_type", type=str, default="shp", choices=["shp"])
    parser.add_argument("--gt_label_field", type=str, required=True)
    parser.add_argument("--gt_folder_field", type=str, required=True)
    parser.add_argument("--gt_file_field", type=str, required=True)
    parser.add_argument("--gt_fx_field", type=str, required=True)
    parser.add_argument("--gt_fy_field", type=str, required=True)
    parser.add_argument("--imagery_root", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size_px", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--correction_margin", type=float, default=0.0)
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = Phase2Config(
        phase1_ckpt=args.phase1_ckpt,
        phase1_embedding_csv=args.phase1_embedding_csv,
        gt_path=args.gt_path,
        gt_type=args.gt_type,
        gt_label_field=args.gt_label_field,
        gt_folder_field=args.gt_folder_field,
        gt_file_field=args.gt_file_field,
        gt_fx_field=args.gt_fx_field,
        gt_fy_field=args.gt_fy_field,
        imagery_root=args.imagery_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        patch_size_px=args.patch_size_px,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        similarity=args.similarity,
        confidence_threshold=args.confidence_threshold,
        correction_margin=args.correction_margin,
        mixed_precision=not args.no_amp,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = config.mixed_precision and device.type == "cuda"

    with open(os.path.join(config.output_dir, "phase2_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("Phase 2 started")
    print(f"Phase 1 checkpoint     : {config.phase1_ckpt}")
    print(f"Phase 1 embedding CSV  : {config.phase1_embedding_csv}")
    print(f"Ground truth path      : {config.gt_path}")
    print(f"Ground truth type      : {config.gt_type}")
    print(f"GT label field         : {config.gt_label_field}")
    print(f"GT folder field        : {config.gt_folder_field}")
    print(f"GT file field          : {config.gt_file_field}")
    print(f"GT fx field            : {config.gt_fx_field}")
    print(f"GT fy field            : {config.gt_fy_field}")
    print(f"Imagery root           : {config.imagery_root}")
    print(f"Output dir             : {config.output_dir}")
    print(f"Similarity             : {config.similarity}")
    print("=" * 100)

    start_time = time.time()

    model, _ = load_encoder_from_checkpoint(config.phase1_ckpt, device)
    gt_transform = build_eval_transform(config.image_size)

    gt_dataset = ShapefilePointDataset(
        shp_path=config.gt_path,
        imagery_root=config.imagery_root,
        label_field=config.gt_label_field,
        folder_field=config.gt_folder_field,
        file_field=config.gt_file_field,
        fx_field=config.gt_fx_field,
        fy_field=config.gt_fy_field,
        patch_size_px=config.patch_size_px,
        transform=gt_transform,
    )

    gt_loader = DataLoader(
        gt_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )

    gt_embeddings_csv = os.path.join(config.output_dir, "ground_truth_embeddings.csv")
    extract_ground_truth_embeddings(
        model=model,
        loader=gt_loader,
        device=device,
        output_csv=gt_embeddings_csv,
        class_names=gt_dataset.classes,
        use_amp=use_amp,
    )

    gt_rows, gt_embs, _ = read_embedding_csv_rows(gt_embeddings_csv)
    prototypes = compute_class_prototypes(gt_rows, gt_embs)

    prototypes_csv = os.path.join(config.output_dir, "class_prototypes.csv")
    save_prototypes_csv(prototypes, prototypes_csv)

    phase1_rows, phase1_embs, _ = read_embedding_csv_rows(config.phase1_embedding_csv)

    internal_to_semantic = build_internal_to_semantic_mapping(
        phase1_rows=phase1_rows,
        phase1_embeddings=phase1_embs,
        prototypes=prototypes,
        similarity=config.similarity,
    )

    mapping_json = os.path.join(config.output_dir, "internal_to_semantic_mapping.json")
    save_internal_semantic_mapping(internal_to_semantic, mapping_json)

    corrected_rows, summary = correct_labels(
        phase1_rows=phase1_rows,
        phase1_embeddings=phase1_embs,
        prototypes=prototypes,
        similarity=config.similarity,
        confidence_threshold=config.confidence_threshold,
        correction_margin=config.correction_margin,
    )

    corrected_csv = os.path.join(config.output_dir, "corrected_labels.csv")
    save_corrected_rows_csv(corrected_rows, corrected_csv)

    summary["num_gt_samples"] = float(len(gt_rows))
    summary["num_prototypes"] = float(len(prototypes))
    summary["elapsed_seconds"] = float(time.time() - start_time)

    summary_json = os.path.join(config.output_dir, "correction_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("Phase 2 completed")
    print(f"Ground truth embeddings : {gt_embeddings_csv}")
    print(f"Prototype CSV           : {prototypes_csv}")
    print(f"Corrected labels CSV    : {corrected_csv}")
    print(f"Summary JSON            : {summary_json}")
    print(f"Elapsed                 : {format_seconds(summary['elapsed_seconds'])}")
    print("=" * 100)


if __name__ == "__main__":
    main()