import PIL
import PIL.ImageEnhance
from PIL import Image

def preprocess(img: Image.Image) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Denoise
    img = img.filter(PIL.ImageFilter.MedianFilter(size=3))

    # Green curve
    rr, gg, bb = img.split()
    gg = gg.point(lambda p: 0 if p < 128 else p-128)
    img = Image.merge('RGB', (rr, gg, bb))

    img = PIL.ImageEnhance.Contrast(img).enhance(1.2)

    img = PIL.ImageEnhance.Color(img).enhance(1.2)

    return img
