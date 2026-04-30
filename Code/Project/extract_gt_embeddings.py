"""
Extract embeddings from GT point SHP using a trained/fine-tuned encoder.

Output:
- CSV with image_path, point_id, label, x, y, emb_0...
"""

import argparse
import os
import csv
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.models.checkpoint import load_encoder_from_checkpoint


def normalize_stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name).strip()))[0].lower()


def build_tif_index(imagery_root: str) -> Dict[str, List[str]]:
    folder_to_paths: Dict[str, List[str]] = {}

    print("[INFO] Building TIFF index...")
    for root, _, files in os.walk(imagery_root):
        tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        if not tif_files:
            continue

        rel = os.path.relpath(root, imagery_root)
        parts = rel.split(os.sep)

        folder_key = None
        for p in parts:
            if p.startswith("2023-"):
                folder_key = p
                break

        if folder_key is None:
            folder_key = parts[0] if parts and parts[0] != "." else ""

        folder_to_paths.setdefault(folder_key, [])

        for f in tif_files:
            folder_to_paths[folder_key].append(os.path.join(root, f))

    total = sum(len(v) for v in folder_to_paths.values())
    print(f"[INFO] Indexed TIFF folders: {len(folder_to_paths)}")
    print(f"[INFO] Indexed TIFF files  : {total}")

    if total == 0:
        raise RuntimeError(f"No TIFF files found under imagery_root: {imagery_root}")

    return folder_to_paths


def resolve_tif_path_fast(folder_to_paths: Dict[str, List[str]], folder, filename) -> str:
    folder = str(folder).strip()
    stem = normalize_stem(filename)

    if folder not in folder_to_paths:
        available = sorted(folder_to_paths.keys())[:30]
        raise FileNotFoundError(
            f"Folder key not found: {folder}. Example indexed folders: {available}"
        )

    paths = folder_to_paths[folder]
    exact, contains, reverse_contains = [], [], []

    for p in paths:
        tif_stem = normalize_stem(p)

        if tif_stem == stem:
            exact.append(p)
        elif stem in tif_stem:
            contains.append(p)
        elif tif_stem in stem:
            reverse_contains.append(p)

    if exact:
        return sorted(exact, key=len)[0]
    if contains:
        return sorted(contains, key=len)[0]
    if reverse_contains:
        return sorted(reverse_contains, key=len)[0]

    raise FileNotFoundError(
        f"No matching TIFF in folder={folder} for file stem={stem}. "
        f"Folder TIFF count={len(paths)}. "
        f"First files={[os.path.basename(p) for p in paths[:10]]}"
    )


def convert_to_pixel(src, x, y, coord_mode: str) -> Tuple[float, float, str]:
    x = float(x)
    y = float(y)

    if coord_mode == "pixel":
        return x, y, "pixel"

    if coord_mode == "normalized":
        return x * src.width, y * src.height, "normalized"

    if coord_mode == "world":
        row, col = src.index(x, y)
        return float(col), float(row), "world"

    if coord_mode == "auto":
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * src.width, y * src.height, "normalized"

        if 0.0 <= x < src.width and 0.0 <= y < src.height:
            return x, y, "pixel"

        row, col = src.index(x, y)
        return float(col), float(row), "world"

    raise ValueError("coord_mode must be one of: auto, normalized, pixel, world")


def read_patch(image_path, x, y, patch_size, coord_mode="auto"):
    half = patch_size // 2

    with rasterio.open(image_path) as src:
        px, py, used_mode = convert_to_pixel(src, x, y, coord_mode)

        col0 = int(round(px)) - half
        row0 = int(round(py)) - half

        window = rasterio.windows.Window(col0, row0, patch_size, patch_size)
        arr = src.read(window=window, boundless=True, fill_value=0)

    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    else:
        raise ValueError(f"Invalid band count: {arr.shape}")

    arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.nanpercentile(arr, [1, 99])
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr), float(px), float(py), used_mode


def build_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class GTEmbeddingDataset(Dataset):
    def __init__(
        self,
        gt_path,
        imagery_root,
        label_field,
        folder_field,
        file_field,
        fx_field,
        fy_field,
        patch_size_px,
        image_size,
        coord_mode="auto",
    ):
        self.gdf = gpd.read_file(gt_path)
        self.gdf = self.gdf[self.gdf[label_field].notna()].copy()
        self.gdf[label_field] = self.gdf[label_field].astype(str).str.strip()

        self.label_field = label_field
        self.folder_field = folder_field
        self.file_field = file_field
        self.fx_field = fx_field
        self.fy_field = fy_field
        self.patch_size_px = patch_size_px
        self.coord_mode = coord_mode
        self.transform = build_eval_transform(image_size)

        required = [label_field, folder_field, file_field, fx_field, fy_field]
        for c in required:
            if c not in self.gdf.columns:
                raise ValueError(f"Missing required field '{c}'. Available: {self.gdf.columns.tolist()}")

        folder_to_paths = build_tif_index(imagery_root)

        self.rows = []
        self.failed_rows = []

        print(f"[INFO] Resolving TIFF paths for {os.path.basename(gt_path)}...")
        for i, (_, row) in enumerate(tqdm(self.gdf.iterrows(), total=len(self.gdf), dynamic_ncols=True)):
            try:
                image_path = resolve_tif_path_fast(
                    folder_to_paths,
                    row[folder_field],
                    row[file_field],
                )

                rec = row.copy()
                rec["_image_path"] = image_path
                self.rows.append(rec)

            except Exception as e:
                self.failed_rows.append((i, str(e)))
                if len(self.failed_rows) <= 10:
                    print(f"[WARN] Failed row {i}: {e}")

        print(f"[INFO] Resolved samples: {len(self.rows)}")
        print(f"[INFO] Failed samples  : {len(self.failed_rows)}")

        if len(self.rows) == 0:
            raise RuntimeError("No usable samples after TIFF path resolution.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = row["_image_path"]

        patch, px, py, used_mode = read_patch(
            image_path=image_path,
            x=float(row[self.fx_field]),
            y=float(row[self.fy_field]),
            patch_size=self.patch_size_px,
            coord_mode=self.coord_mode,
        )

        x = self.transform(patch)

        label = str(row[self.label_field]).strip()
        folder = str(row[self.folder_field])
        file_value = str(row[self.file_field])
        raw_x = float(row[self.fx_field])
        raw_y = float(row[self.fy_field])

        point_id = f"{os.path.basename(image_path)}_{int(round(px))}_{int(round(py))}"

        meta = {
            "image_path": image_path,
            "point_id": point_id,
            "label": label,
            "folder": folder,
            "file": file_value,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "x": px,
            "y": py,
            "coord_mode_used": used_mode,
        }

        return x, meta


def collate_patch_with_meta(batch):
    xs, metas = zip(*batch)
    xs = torch.stack(xs, dim=0)
    metas = list(metas)
    return xs, metas


def forward_encode(model, x):
    if hasattr(model, "encode"):
        z = model.encode(x)
    else:
        z = model(x)

    if isinstance(z, dict):
        for k in ["features", "embedding", "embeddings", "x", "last_hidden_state"]:
            if k in z:
                z = z[k]
                break

    if isinstance(z, (tuple, list)):
        z = z[0]

    if z.ndim == 4:
        z = z.mean(dim=(2, 3))
    elif z.ndim == 3:
        z = z[:, 0]

    return z


@torch.no_grad()
def extract_embeddings(model, loader, device, output_csv, use_amp):
    model.eval()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    first_batch = True

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = None

        for patches, metas in tqdm(loader, desc="Extracting GT embeddings", dynamic_ncols=True):
            patches = patches.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                feat = forward_encode(model, patches)

            feat_np = feat.detach().cpu().numpy()

            if first_batch:
                dim = feat_np.shape[1]
                header = [
                    "image_path",
                    "point_id",
                    "label",
                    "folder",
                    "file",
                    "raw_x",
                    "raw_y",
                    "x",
                    "y",
                    "coord_mode_used",
                ] + [f"emb_{i}" for i in range(dim)]

                writer = csv.writer(f)
                writer.writerow(header)
                first_batch = False

            for i, meta in enumerate(metas):
                row = [
                    meta["image_path"],
                    meta["point_id"],
                    meta["label"],
                    meta["folder"],
                    meta["file"],
                    float(meta["raw_x"]),
                    float(meta["raw_y"]),
                    float(meta["x"]),
                    float(meta["y"]),
                    meta["coord_mode_used"],
                ] + feat_np[i].astype(np.float32).tolist()

                writer.writerow(row)

    print(f"[INFO] Saved embeddings: {output_csv}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--gt_path", required=True)
    p.add_argument("--imagery_root", required=True)
    p.add_argument("--output_csv", required=True)

    p.add_argument("--label_field", default="Tree")
    p.add_argument("--folder_field", default="Folder")
    p.add_argument("--file_field", default="File")
    p.add_argument("--fx_field", default="fx")
    p.add_argument("--fy_field", default="fy")
    p.add_argument("--coord_mode", default="auto", choices=["auto", "normalized", "pixel", "world"])

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size_px", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    print("=" * 80)
    print("Extract GT embeddings")
    print(f"Checkpoint : {args.encoder_ckpt}")
    print(f"GT path    : {args.gt_path}")
    print(f"Output CSV : {args.output_csv}")
    print(f"Device     : {device}")
    print(f"AMP        : {use_amp}")
    print("=" * 80)

    model, _ = load_encoder_from_checkpoint(args.encoder_ckpt, device)

    dataset = GTEmbeddingDataset(
        gt_path=args.gt_path,
        imagery_root=args.imagery_root,
        label_field=args.label_field,
        folder_field=args.folder_field,
        file_field=args.file_field,
        fx_field=args.fx_field,
        fy_field=args.fy_field,
        patch_size_px=args.patch_size_px,
        image_size=args.image_size,
        coord_mode=args.coord_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_patch_with_meta,
    )

    extract_embeddings(
        model=model,
        loader=loader,
        device=device,
        output_csv=args.output_csv,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()