import os
import json
import time
import argparse
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.checkpoint import load_encoder_from_checkpoint
from src.scoring.prototypes import (
    GroundTruthTifDataset,
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
    gt_root: str
    gt_label_csv: str
    output_dir: str

    image_size: int = 224
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
            transforms.Resize(int(image_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: build semantic prototypes and correct labels from ground truth"
    )

    parser.add_argument("--phase1_ckpt", type=str, required=True)
    parser.add_argument("--phase1_embedding_csv", type=str, required=True)
    parser.add_argument("--gt_root", type=str, required=True)
    parser.add_argument("--gt_label_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=224)
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
        gt_root=args.gt_root,
        gt_label_csv=args.gt_label_csv,
        output_dir=args.output_dir,
        image_size=args.image_size,
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
    print(f"Ground truth root      : {config.gt_root}")
    print(f"Ground truth label CSV : {config.gt_label_csv}")
    print(f"Output dir             : {config.output_dir}")
    print(f"Similarity             : {config.similarity}")
    print("=" * 100)

    start_time = time.time()

    model, _ = load_encoder_from_checkpoint(config.phase1_ckpt, device)

    gt_transform = build_eval_transform(config.image_size)
    gt_dataset = GroundTruthTifDataset(
        root_dir=config.gt_root,
        label_csv=config.gt_label_csv,
        transform=gt_transform,
        strict=True,
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