from PIL import Image
import numpy as np


def _center_emphasis(img_np: np.ndarray, strength: float = 0.15) -> np.ndarray:
    h, w = img_np.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    dy = (yy - cy) / max(cy, 1.0)
    dx = (xx - cx) / max(cx, 1.0)
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.clip(dist, 0.0, 1.0)

    weight = 1.0 - strength * dist
    weight = weight[..., None]

    out = img_np.astype(np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(
    img: Image.Image,
    use_center_emphasis: bool = True,
    center_strength: float = 0.15,
) -> Image.Image:
    if not isinstance(img, Image.Image):
        raise TypeError("preprocess expects a PIL.Image")

    img = img.convert("RGB")
    arr = np.array(img)

    if use_center_emphasis:
        arr = _center_emphasis(arr, strength=center_strength)

    return Image.fromarray(arr)