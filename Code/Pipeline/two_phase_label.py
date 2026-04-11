import os
import csv
import json
import math
import time
import glob
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm with: pip install timm") from e


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
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


def safe_open_image(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise RuntimeError(f"Failed to open image: {path} | {e}") from e


def recursive_find_tif_files(root_dir: str) -> List[str]:
    patterns = ["**/*.tif", "**/*.TIF", "**/*.tiff", "**/*.TIFF"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    return sorted(list(set(os.path.abspath(p) for p in files)))


def load_label_map_csv(label_csv: str) -> Dict[str, str]:
    mapping = {}
    with open(label_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Label CSV must contain columns: path,label")
        for row in reader:
            mapping[os.path.abspath(row["path"])] = str(row["label"])
    return mapping


class GroundTruthTifDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        label_csv: str,
        transform=None,
        strict: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = load_label_map_csv(label_csv)

        all_paths = recursive_find_tif_files(root_dir)
        self.samples = []

        label_names = []
        for path in all_paths:
            label = self.label_map.get(path)
            if label is None:
                if strict:
                    continue
                label = "unlabeled"
            self.samples.append((path, label))
            label_names.append(label)

        if len(self.samples) == 0:
            raise RuntimeError("No valid ground-truth TIFF samples were found.")

        self.classes = sorted(list(set(label_names)))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        self.samples = [(p, self.class_to_idx[label]) for p, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = safe_open_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path


class EmbeddingCSVDataset(Dataset):
    def __init__(self, csv_path: str) -> None:
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            emb_cols = [c for c in fieldnames if c.startswith("emb_")]
            required = {"image_path", "target_idx", "target_name", "pred_idx", "pred_name", "confidence"}
            if not required.issubset(set(fieldnames)):
                raise ValueError(
                    "Embedding CSV must contain columns: "
                    "image_path,target_idx,target_name,pred_idx,pred_name,confidence,emb_*"
                )
            if len(emb_cols) == 0:
                raise ValueError("No embedding columns found in CSV.")

            self.emb_cols = sorted(emb_cols, key=lambda x: int(x.split("_")[1]))
            for row in reader:
                self.rows.append(row)

        if len(self.rows) == 0:
            raise RuntimeError(f"No rows found in embedding CSV: {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        emb = np.array([float(row[c]) for c in self.emb_cols], dtype=np.float32)
        return row, emb


class ViTBackbone(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        if hasattr(self.model, "num_features"):
            self.embed_dim = self.model.num_features
        elif hasattr(self.model, "embed_dim"):
            self.embed_dim = self.model.embed_dim
        else:
            raise ValueError("Could not infer embedding dimension from the backbone.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.forward_features(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if feat.ndim == 3:
            if hasattr(self.model, "num_prefix_tokens") and self.model.num_prefix_tokens > 0:
                return feat[:, 0]
            return feat.mean(dim=1)

        return feat


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeJEPALikeModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 512,
        backbone_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ViTBackbone(backbone_name, pretrained=backbone_pretrained)
        self.projector = MLPProjector(
            in_dim=self.backbone.embed_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
        )
        self.predictor = nn.Sequential(
            nn.Linear(projector_out_dim, projector_out_dim),
            nn.GELU(),
            nn.Linear(projector_out_dim, projector_out_dim),
        )
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project(self, feat: torch.Tensor) -> torch.Tensor:
        return self.projector(feat)

    def predict(self, proj: torch.Tensor) -> torch.Tensor:
        return self.predictor(proj)

    def classify(self, feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(feat)


@dataclass
class Config:
    phase1_ckpt: str
    phase1_embedding_csv: str
    gt_root: str
    gt_label_csv: str
    output_dir: str

    backbone_name: str = "vit_base_patch16_224"
    backbone_pretrained: bool = False
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 512

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


def load_phase1_checkpoint_metadata(ckpt_path: str, device: torch.device) -> dict:
    return torch.load(ckpt_path, map_location=device)


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    state = load_phase1_checkpoint_metadata(ckpt_path, device)

    cfg = state.get("config", {})
    class_to_idx = state.get("class_to_idx", {})
    num_classes = len(class_to_idx)

    backbone_name = cfg.get("backbone_name", "vit_base_patch16_224")
    backbone_pretrained = cfg.get("backbone_pretrained", False)
    projector_hidden_dim = cfg.get("projector_hidden_dim", 2048)
    projector_out_dim = cfg.get("projector_out_dim", 512)

    model = LeJEPALikeModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        projector_hidden_dim=projector_hidden_dim,
        projector_out_dim=projector_out_dim,
        backbone_pretrained=backbone_pretrained,
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    return model, state


@torch.no_grad()
def extract_ground_truth_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_csv: str,
    class_names: List[str],
    use_amp: bool,
) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    first_batch = True
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = None

        pbar = tqdm(loader, desc="Extracting GT embeddings", dynamic_ncols=True)
        for images, targets, paths in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                feat = model.encode(images)
                logits = model.classify(feat)
                probs = torch.softmax(logits, dim=1)
                confs, preds = probs.max(dim=1)

            feat_np = feat.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            confs_np = confs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            if first_batch:
                dim = feat_np.shape[1]
                header = [
                    "image_path",
                    "target_idx",
                    "target_name",
                    "pred_idx",
                    "pred_name",
                    "confidence",
                ] + [f"emb_{i}" for i in range(dim)]
                writer = csv.writer(f)
                writer.writerow(header)
                first_batch = False

            for i in range(len(paths)):
                writer.writerow(
                    [
                        paths[i],
                        int(targets_np[i]),
                        class_names[int(targets_np[i])],
                        int(preds_np[i]) if 0 <= int(preds_np[i]) < len(class_names) else -1,
                        class_names[int(preds_np[i])] if 0 <= int(preds_np[i]) < len(class_names) else "unknown",
                        float(confs_np[i]),
                    ] + feat_np[i].astype(np.float32).tolist()
                )


def read_embedding_csv_rows(csv_path: str) -> Tuple[List[dict], np.ndarray, List[str]]:
    rows = []
    emb_matrix = []
    emb_cols = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        emb_cols = sorted([c for c in fieldnames if c.startswith("emb_")], key=lambda x: int(x.split("_")[1]))
        if len(emb_cols) == 0:
            raise ValueError(f"No embedding columns found in {csv_path}")

        for row in reader:
            rows.append(row)
            emb_matrix.append([float(row[c]) for c in emb_cols])

    return rows, np.asarray(emb_matrix, dtype=np.float32), emb_cols


def compute_class_prototypes(gt_rows: List[dict], gt_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    buckets: Dict[str, List[np.ndarray]] = {}

    for row, emb in zip(gt_rows, gt_embeddings):
        label = row["target_name"]
        buckets.setdefault(label, []).append(emb)

    prototypes = {}
    for label, vecs in buckets.items():
        mat = np.stack(vecs, axis=0)
        proto = mat.mean(axis=0)
        prototypes[label] = proto.astype(np.float32)

    return prototypes


def save_prototypes_csv(prototypes: Dict[str, np.ndarray], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    labels = sorted(prototypes.keys())
    dim = len(next(iter(prototypes.values())))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name"] + [f"emb_{i}" for i in range(dim)])
        for label in labels:
            writer.writerow([label] + prototypes[label].tolist())


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    return x_norm @ y_norm.T


def euclidean_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x ** 2, axis=1, keepdims=True)
    y2 = np.sum(y ** 2, axis=1, keepdims=True).T
    dist2 = x2 + y2 - 2 * (x @ y.T)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)
    return -dist


def build_internal_to_semantic_mapping(
    phase1_rows: List[dict],
    phase1_embeddings: np.ndarray,
    prototypes: Dict[str, np.ndarray],
    similarity: str = "cosine",
) -> Dict[str, str]:
    proto_labels = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[l] for l in proto_labels], axis=0)

    if similarity == "cosine":
        sims = cosine_similarity_matrix(phase1_embeddings, proto_matrix)
    elif similarity == "euclidean":
        sims = euclidean_similarity_matrix(phase1_embeddings, proto_matrix)
    else:
        raise ValueError("similarity must be one of: cosine, euclidean")

    internal_votes: Dict[str, List[str]] = {}

    for row, sim_vec in zip(phase1_rows, sims):
        internal_pred = row["pred_name"]
        best_idx = int(np.argmax(sim_vec))
        semantic_label = proto_labels[best_idx]
        internal_votes.setdefault(internal_pred, []).append(semantic_label)

    mapping = {}
    for internal_pred, labels in internal_votes.items():
        counts: Dict[str, int] = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        best_label = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        mapping[internal_pred] = best_label

    return mapping


def correct_labels(
    phase1_rows: List[dict],
    phase1_embeddings: np.ndarray,
    prototypes: Dict[str, np.ndarray],
    similarity: str = "cosine",
    confidence_threshold: float = 0.0,
    correction_margin: float = 0.0,
) -> Tuple[List[dict], Dict[str, float]]:
    proto_labels = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[l] for l in proto_labels], axis=0)

    if similarity == "cosine":
        sim_matrix = cosine_similarity_matrix(phase1_embeddings, proto_matrix)
    elif similarity == "euclidean":
        sim_matrix = euclidean_similarity_matrix(phase1_embeddings, proto_matrix)
    else:
        raise ValueError("similarity must be one of: cosine, euclidean")

    corrected_rows = []
    num_changed = 0

    for row, sim_vec in zip(phase1_rows, sim_matrix):
        best_idx = int(np.argmax(sim_vec))
        best_label = proto_labels[best_idx]
        best_score = float(sim_vec[best_idx])

        second_score = float(np.partition(sim_vec, -2)[-2]) if len(sim_vec) > 1 else best_score
        margin = best_score - second_score

        original_pred = row["pred_name"]
        phase1_conf = float(row["confidence"])

        should_correct = True
        if phase1_conf < confidence_threshold:
            should_correct = False
        if margin < correction_margin:
            should_correct = False

        corrected_label = best_label if should_correct else original_pred

        if corrected_label != original_pred:
            num_changed += 1

        new_row = dict(row)
        new_row["prototype_best_label"] = best_label
        new_row["prototype_best_score"] = best_score
        new_row["prototype_margin"] = margin
        new_row["corrected_label"] = corrected_label
        new_row["correction_applied"] = int(corrected_label != original_pred)
        corrected_rows.append(new_row)

    summary = {
        "total_samples": float(len(phase1_rows)),
        "num_changed": float(num_changed),
        "change_ratio": float(num_changed / max(1, len(phase1_rows))),
    }
    return corrected_rows, summary


def save_corrected_rows_csv(corrected_rows: List[dict], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if len(corrected_rows) == 0:
        raise RuntimeError("No corrected rows to save.")

    fieldnames = list(corrected_rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in corrected_rows:
            writer.writerow(row)


def save_internal_semantic_mapping(mapping: Dict[str, str], output_json: str) -> None:
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: ground-truth feature extraction and label correction")

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

    args = parser.parse_args()

    config = Config(
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

    model, ckpt_state = build_model_from_checkpoint(config.phase1_ckpt, device)

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
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
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

    print(f"Saved ground-truth embeddings to: {gt_embeddings_csv}")

    gt_rows, gt_embs, _ = read_embedding_csv_rows(gt_embeddings_csv)
    prototypes = compute_class_prototypes(gt_rows, gt_embs)

    prototypes_csv = os.path.join(config.output_dir, "class_prototypes.csv")
    save_prototypes_csv(prototypes, prototypes_csv)
    print(f"Saved prototypes to: {prototypes_csv}")

    phase1_rows, phase1_embs, _ = read_embedding_csv_rows(config.phase1_embedding_csv)

    internal_to_semantic = build_internal_to_semantic_mapping(
        phase1_rows=phase1_rows,
        phase1_embeddings=phase1_embs,
        prototypes=prototypes,
        similarity=config.similarity,
    )

    mapping_json = os.path.join(config.output_dir, "internal_to_semantic_mapping.json")
    save_internal_semantic_mapping(internal_to_semantic, mapping_json)
    print(f"Saved internal-to-semantic mapping to: {mapping_json}")

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