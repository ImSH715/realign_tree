import os
from typing import List, Dict

import geopandas as gpd
import rasterio
from torch.utils.data import Dataset

from src.data.tif_io import read_patch_as_pil


class ShapefilePointDataset(Dataset):
    def __init__(
        self,
        shp_path: str,
        imagery_root: str,
        label_field: str,
        tile_field: str,
        patch_size_px: int = 224,
        transform=None,
    ):
        self.gdf = gpd.read_file(shp_path)
        self.imagery_root = imagery_root
        self.label_field = label_field
        self.tile_field = tile_field
        self.patch_size_px = patch_size_px
        self.transform = transform

        self.samples: List[Dict] = []
        label_names: List[str] = []

        for _, row in self.gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type != "Point":
                continue

            x_world, y_world = geom.x, geom.y
            label = str(row[label_field])
            tile_name = str(row[tile_field])

            image_path = tile_name
            if not os.path.isabs(image_path):
                image_path = os.path.join(imagery_root, tile_name)

            if not os.path.exists(image_path):
                print(f"[WARN] Missing TIFF: {image_path}")
                continue

            try:
                with rasterio.open(image_path) as src:
                    row_px, col_px = src.index(x_world, y_world)
            except Exception as e:
                print(f"[WARN] Coordinate transform failed: {image_path} | {e}")
                continue

            self.samples.append(
                {
                    "image_path": image_path,
                    "x": float(col_px),
                    "y": float(row_px),
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