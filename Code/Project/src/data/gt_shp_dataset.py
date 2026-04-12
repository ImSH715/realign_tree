import os
from typing import List, Dict

import geopandas as gpd
from torch.utils.data import Dataset

from src.data.tif_io import read_patch_as_pil, get_image_size


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

        self.samples: List[Dict] = []
        label_names: List[str] = []

        for _, row in self.gdf.iterrows():
            label = str(row[label_field])
            folder = str(row[folder_field])
            file_name = str(row[file_field])
            fx = float(row[fx_field])
            fy = float(row[fy_field])

            image_path = os.path.join(self.imagery_root, folder, file_name)

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

            self.samples.append(
                {
                    "image_path": image_path,
                    "x": float(x_px),
                    "y": float(y_px),
                    "label": label,
                }
            )
            label_names.append(label)

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