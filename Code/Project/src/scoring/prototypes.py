import os
import csv
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.tif_io import safe_open_image, recursive_find_tif_files


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


@torch.no_grad()
def extract_ground_truth_embeddings(
    model: torch.nn.Module,
    loader,
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

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                feat = model.encode(images)

            feat_np = feat.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            if first_batch:
                dim = feat_np.shape[1]
                header = [
                    "image_path",
                    "target_idx",
                    "target_name",
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


def save_internal_semantic_mapping(mapping: Dict[str, str], output_json: str) -> None:
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


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
def load_prototypes_csv(csv_path: str) -> Dict[str, np.ndarray]:
    prototypes = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        emb_cols = sorted([c for c in fieldnames if c.startswith("emb_")], key=lambda x: int(x.split("_")[1]))
        if "class_name" not in fieldnames:
            raise ValueError("Prototype CSV must contain class_name column.")
        for row in reader:
            label = row["class_name"]
            emb = np.array([float(row[c]) for c in emb_cols], dtype=np.float32)
            prototypes[label] = emb
    return prototypes