import os
import glob
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from PIL import Image
from preprocess import preprocess

def recursive_find_tif_files(root_dir: str) -> List[str]:
    patterns = ["**/*.tif", "**/*.TIF", "**/*.tiff", "**/*.TIFF"]
    files = []
    exclude_tokens = [
        "/2. Raw_photographs/",
        ".files/",
        "/orthomosaic/tile-",
    ]

    for pattern in patterns:
        found = glob.glob(os.path.join(root_dir, pattern), recursive=True)
        for p in found:
            ap = os.path.abspath(p)
            if any(token in ap for token in exclude_tokens):
                continue
            files.append(ap)

    return sorted(list(set(files)))


def is_readable_tif(path: str) -> bool:
    try:
        with rasterio.open(path) as src:
            _ = src.width
            _ = src.height
        return True
    except Exception:
        return False


def get_image_size(path: str) -> Tuple[int, int]:
    with rasterio.open(path) as src:
        return src.width, src.height


def read_patch_as_pil(path: str, left: int, top: int, width: int, height: int) -> Image.Image:
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(left, top, width, height)
        data = src.read(window=window, boundless=True, fill_value=0)

    if data.ndim != 3:
        raise RuntimeError(f"Unexpected raster shape for {path}: {data.shape}")

    bands, h, w = data.shape

    if bands >= 3:
        rgb = data[:3]
    elif bands == 1:
        rgb = np.repeat(data, 3, axis=0)
    else:
        raise RuntimeError(f"Unsupported number of bands in {path}: {bands}")

    rgb = np.transpose(rgb, (1, 2, 0))

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        min_val = rgb.min()
        max_val = rgb.max()
        if max_val > min_val:
            rgb = (rgb - min_val) / (max_val - min_val)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    return preprocess(img)


class TileCache:
    def __init__(self, max_items: int = 0) -> None:
        self.max_items = max_items
        self.cache: Dict[str, Tuple[int, int]] = {}
        self.order: List[str] = []

    def get_size(self, path: str) -> Tuple[int, int]:
        if self.max_items <= 0:
            return get_image_size(path)

        if path in self.cache:
            if path in self.order:
                self.order.remove(path)
            self.order.append(path)
            return self.cache[path]

        size = get_image_size(path)
        self.cache[path] = size
        self.order.append(path)

        if len(self.order) > self.max_items:
            old_path = self.order.pop(0)
            if old_path in self.cache:
                del self.cache[old_path]

        return size
