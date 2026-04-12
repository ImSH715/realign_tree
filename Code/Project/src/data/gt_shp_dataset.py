import os
import glob
from typing import List, Dict

import geopandas as gpd
from torch.utils.data import Dataset

from src.data.tif_io import read_patch_as_pil, get_image_size, recursive_find_tif_files


def normalize_name(name: str) -> str:
    name = str(name).strip()
    return os.path.splitext(name)[0].lower()


def build_tif_index(imagery_root: str) -> List[str]:
    tif_paths = recursive_find_tif_files(imagery_root)
    return tif_paths


def resolve_image_path(imagery_root: str, folder: str, file_name: str, tif_paths: List[str]) -> str:
    folder = str(folder).strip()
    file_name = str(file_name).strip()

    target_norm = normalize_name(file_name)

    # 1. Direct folder/file path
    direct_candidates = [
        os.path.join(imagery_root, folder, file_name),
        os.path.join(imagery_root, folder, file_name + ".tif"),
        os.path.join(imagery_root, folder, file_name + ".TIF"),
        os.path.join(imagery_root, folder, file_name + ".tiff"),
        os.path.join(imagery_root, folder, file_name + ".TIFF"),
    ]
    for c in direct_candidates:
        if os.path.exists(c):
            return c

    # 2. Exact basename match
    exact_matches = []
    for p in tif_paths:
        base_norm = normalize_name(os.path.basename(p))
        if base_norm == target_norm:
            exact_matches.append(p)
    if len(exact_matches) > 0:
        return exact_matches[0]

    # 3. Prefix basename match
    prefix_matches = []
    for p in tif_paths:
        base_norm = normalize_name(os.path.basename(p))
        if base_norm.startswith(target_norm):
            prefix_matches.append(p)
    if len(prefix_matches) > 0:
        return prefix_matches[0]

    # 4. Substring basename match
    substring_matches = []
    for p in tif_paths:
        base_norm = normalize_name(os.path.basename(p))
        if target_norm in base_norm:
            substring_matches.append(p)
    if len(substring_matches) > 0:
        return substring_matches[0]

    # 5. Restrict to folder then retry
    folder_filtered = [p for p in tif_paths if f"/{folder}/" in p]
    if len(folder_filtered) > 0:
        for p in folder_filtered:
            base_norm = normalize_name(os.path.basename(p))
            if base_norm == target_norm:
                return p
        for p in folder_filtered:
            base_norm = normalize_name(os.path.basename(p))
            if base_norm.startswith(target_norm):
                return p
        for p in folder_filtered:
            base_norm = normalize_name(os.path.basename(p))
            if target_norm in base_norm:
                return p

    return direct_candidates[0]


class ShapefilePointDataset(Dataset):
    def __init__(
        self,
        shp_path: str,
        imagery_root: str,
        label_field: str,
        folder_field: str,
        file_field: str,
        fx_field: str,
        fy_field: str,
        patch_size_px: int = 224,
        transform=None,
    ):
        self.gdf = gpd.read_file(shp_path)
        self.imagery_root = imagery_root
        self.label_field = label_field
        self.folder_field = folder_field
        self.file_field = file_field
        self.fx_field = fx_field
        self.fy_field = fy_field
        self.patch_size_px = patch_size_px
        self.transform = transform

        available_cols = self.gdf.columns.tolist()
        for required_col in [label_field, folder_field, file_field, fx_field, fy_field]:
            if required_col not in available_cols:
                raise ValueError(
                    f"Required field '{required_col}' not found in shapefile. "
                    f"Available columns: {available_cols}"
                )

        self.tif_paths = build_tif_index(imagery_root)
        print(f"[INFO] Indexed TIFF files for SHP matching: {len(self.tif_paths)}")

        self.samples: List[Dict] = []
        label_names: List[str] = []

        for i, (_, row) in enumerate(self.gdf.iterrows()):
            try:
                label = str(row[label_field]).strip()
                folder = str(row[folder_field]).strip()
                file_name = str(row[file_field]).strip()

                fx_val = row[fx_field]
                fy_val = row[fy_field]

                if fx_val is None or fy_val is None:
                    if i < 20:
                        print(f"[DEBUG] Skipping row {i}: fx/fy is None")
                    continue

                fx = float(fx_val)
                fy = float(fy_val)

                image_path = resolve_image_path(
                    imagery_root=self.imagery_root,
                    folder=folder,
                    file_name=file_name,
                    tif_paths=self.tif_paths,
                )

                if i < 20:
                    print(f"[DEBUG] row={i}")
                    print(f"[DEBUG] folder={folder}")
                    print(f"[DEBUG] file={file_name}")
                    print(f"[DEBUG] resolved_path={image_path}")
                    print(f"[DEBUG] exists={os.path.exists(image_path)}")
                    print(f"[DEBUG] fx={fx}, fy={fy}")

                if not os.path.exists(image_path):
                    print(f"[WARN] Missing TIFF: {image_path}")
                    continue

                try:
                    w, h = get_image_size(image_path)
                except Exception as e:
                    print(f"[WARN] Failed to read image size: {image_path} | {e}")
                    continue

                x_px = fx * w
                y_px = fy * h

                if not (0 <= x_px <= w and 0 <= y_px <= h):
                    if i < 20:
                        print(
                            f"[DEBUG] Skipping row {i}: pixel coordinates out of range "
                            f"(x={x_px}, y={y_px}, w={w}, h={h})"
                        )
                    continue

                self.samples.append(
                    {
                        "image_path": image_path,
                        "x": float(x_px),
                        "y": float(y_px),
                        "label": label,
                    }
                )
                label_names.append(label)

            except Exception as e:
                print(f"[WARN] Skipping SHP row {i} due to error: {e}")
                continue

        print(f"[INFO] Loaded valid SHP samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("No valid SHP samples loaded.")

        self.classes = sorted(list(set(label_names)))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        left = int(round(s["x"] - self.patch_size_px / 2))
        top = int(round(s["y"] - self.patch_size_px / 2))

        patch = read_patch_as_pil(
            path=s["image_path"],
            left=left,
            top=top,
            width=self.patch_size_px,
            height=self.patch_size_px,
        )

        if self.transform is not None:
            patch = self.transform(patch)

        target_idx = self.class_to_idx[s["label"]]
        return patch, target_idx, s["image_path"]