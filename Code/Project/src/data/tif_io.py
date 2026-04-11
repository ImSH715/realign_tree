import os
import glob
from typing import Dict, List

from PIL import Image, ImageFile, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def recursive_find_tif_files(root_dir: str) -> List[str]:
    patterns = ["**/*.tif", "**/*.TIF", "**/*.tiff", "**/*.TIFF"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    return sorted(list(set(os.path.abspath(p) for p in files)))


def safe_open_image(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise RuntimeError(f"Failed to open image: {path} | {e}") from e


class TileCache:
    def __init__(self, max_items: int = 0) -> None:
        self.max_items = max_items
        self.cache: Dict[str, Image.Image] = {}
        self.order: List[str] = []

    def get(self, path: str) -> Image.Image:
        if self.max_items <= 0:
            return safe_open_image(path)

        if path in self.cache:
            if path in self.order:
                self.order.remove(path)
            self.order.append(path)
            return self.cache[path]

        image = safe_open_image(path)
        self.cache[path] = image
        self.order.append(path)

        if len(self.order) > self.max_items:
            old_path = self.order.pop(0)
            if old_path in self.cache:
                del self.cache[old_path]

        return image