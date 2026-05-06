from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


def _center_emphasis(img_np: np.ndarray, strength: float = 0.30) -> np.ndarray:
    h, w = img_np.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    dy = (yy - cy) / max(cy, 1.0)
    dx = (xx - cx) / max(cx, 1.0)
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.clip(dist, 0.0, 1.0)

    # 중심 1.0, 가장자리 1-strength
    weight = 1.0 - strength * dist
    weight = weight[..., None]

    out = img_np.astype(np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def _percentile_stretch(img_np: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    out = img_np.astype(np.float32).copy()

    for c in range(out.shape[2]):
        lo = np.percentile(out[..., c], low)
        hi = np.percentile(out[..., c], high)
        if hi > lo:
            out[..., c] = (out[..., c] - lo) / (hi - lo)
        else:
            out[..., c] = 0.0

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def preprocess(
    img: Image.Image,
    use_center_emphasis: bool = True,
    center_strength: float = 0.30,
    use_percentile_stretch: bool = True,
    sharpen: bool = True,
    contrast_factor: float = 1.10,
    color_factor: float = 1.05,
) -> Image.Image:
    if not isinstance(img, Image.Image):
        raise TypeError("preprocess expects a PIL.Image")

    img = img.convert("RGB")
    arr = np.array(img)

    if use_percentile_stretch:
        arr = _percentile_stretch(arr, low=2.0, high=98.0)

    if use_center_emphasis:
        arr = _center_emphasis(arr, strength=center_strength)

    img = Image.fromarray(arr)

    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

    if contrast_factor != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    if color_factor != 1.0:
        img = ImageEnhance.Color(img).enhance(color_factor)

    return img