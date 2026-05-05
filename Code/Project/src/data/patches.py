import os
import random
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.tif_io import (
    recursive_find_tif_files,
    TileCache,
    is_readable_tif,
    get_image_size,
    read_patch_as_pil,
)


def collate_multiview_with_meta(batch):
    all_views = list(zip(*[item[0] for item in batch]))
    stacked_views = [torch.stack(v, dim=0) for v in all_views]
    metas = [item[1] for item in batch]
    return stacked_views, metas


def collate_patch_with_meta(batch):
    patches = torch.stack([item[0] for item in batch], dim=0)
    metas = [item[1] for item in batch]
    return patches, metas


class RandomPatchTifDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size_px: int = 224,
        patches_per_image: int = 100,
        num_global_views: int = 2,
        num_local_views: int = 4,
        image_size_global: int = 224,
        image_size_local: int = 96,
        tile_cache_size: int = 0,
    ) -> None:
        tif_paths = recursive_find_tif_files(root_dir)
        if len(tif_paths) == 0:
            raise RuntimeError(f"No TIFF files found under: {root_dir}")

        valid_tif_paths = []
        for path in tif_paths:
            if is_readable_tif(path):
                valid_tif_paths.append(path)
            else:
                print(f"[WARN] Skipping unreadable TIFF in training set: {path}")

        self.tif_paths = valid_tif_paths
        if len(self.tif_paths) == 0:
            raise RuntimeError("No readable TIFF files found for training.")

        self.patch_size_px = patch_size_px
        self.patches_per_image = patches_per_image
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.tile_cache = TileCache(max_items=tile_cache_size)

        self.index_map: List[str] = []
        for path in self.tif_paths:
            for _ in range(self.patches_per_image):
                self.index_map.append(path)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        common_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]

        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size_global,
                    scale=(0.6, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                *common_aug,
            ]
        )

        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size_local,
                    scale=(0.3, 0.7),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                *common_aug,
            ]
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        path = self.index_map[idx]
        w, h = self.tile_cache.get_size(path)

        if w < self.patch_size_px or h < self.patch_size_px:
            cx = w / 2.0
            cy = h / 2.0
            left = max(0, int(cx - self.patch_size_px / 2))
            top = max(0, int(cy - self.patch_size_px / 2))
        else:
            left = random.randint(0, w - self.patch_size_px)
            top = random.randint(0, h - self.patch_size_px)

        patch = read_patch_as_pil(
            path=path,
            left=left,
            top=top,
            width=self.patch_size_px,
            height=self.patch_size_px,
        )

        center_x = left + self.patch_size_px / 2.0
        center_y = top + self.patch_size_px / 2.0

        views = []
        for _ in range(self.num_global_views):
            views.append(self.global_transform(patch))
        for _ in range(self.num_local_views):
            views.append(self.local_transform(patch))

        meta = {
            "image_path": path,
            "x": float(center_x),
            "y": float(center_y),
        }
        return views, meta


class GridPatchTifDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size_px: int = 224,
        stride_px: int = 224,
        image_size: int = 224,
        tile_cache_size: int = 0,
        max_patches_per_image: Optional[int] = None,
    ) -> None:
        tif_paths = recursive_find_tif_files(root_dir)
        if len(tif_paths) == 0:
            raise RuntimeError(f"No TIFF files found under: {root_dir}")

        self.patch_size_px = patch_size_px
        self.stride_px = stride_px
        self.image_size = image_size
        self.tile_cache = TileCache(max_items=tile_cache_size)
        self.samples: List[Tuple[str, float, float]] = []
        self.tif_paths: List[str] = []

        for path in tif_paths:
            try:
                w, h = get_image_size(path)
            except Exception as e:
                print(f"[WARN] Skipping unreadable TIFF in extraction set: {path} | {e}")
                continue

            self.tif_paths.append(path)

            xs = list(range(
                patch_size_px // 2,
                max(patch_size_px // 2 + 1, w - patch_size_px // 2 + 1),
                stride_px,
            ))
            ys = list(range(
                patch_size_px // 2,
                max(patch_size_px // 2 + 1, h - patch_size_px // 2 + 1),
                stride_px,
            ))

            image_samples = [(path, float(x), float(y)) for y in ys for x in xs]

            if len(image_samples) == 0:
                cx = max(w / 2.0, patch_size_px / 2.0)
                cy = max(h / 2.0, patch_size_px / 2.0)
                image_samples = [(path, float(cx), float(cy))]

            if max_patches_per_image is not None and len(image_samples) > max_patches_per_image:
                step = len(image_samples) / max_patches_per_image
                reduced = []
                for i in range(max_patches_per_image):
                    reduced.append(image_samples[int(i * step)])
                image_samples = reduced

            self.samples.extend(image_samples)

        if len(self.tif_paths) == 0:
            raise RuntimeError("No readable TIFF files found for extraction.")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, x, y = self.samples[idx]
        left = int(round(x - self.patch_size_px / 2))
        top = int(round(y - self.patch_size_px / 2))

        patch = read_patch_as_pil(
            path=path,
            left=left,
            top=top,
            width=self.patch_size_px,
            height=self.patch_size_px,
        )
        patch = self.transform(patch)

        meta = {
            "image_path": path,
            "x": float(x),
            "y": float(y),
        }
        return patch, meta


class PatchExtractor:
    def __init__(self, patch_size_px: int, tile_cache_size: int = 0):
        self.patch_size_px = patch_size_px
        self.tile_cache = TileCache(max_items=tile_cache_size)

    def extract(self, image_path: str, center_x: float, center_y: float):
        left = int(round(center_x - self.patch_size_px / 2))
        top = int(round(center_y - self.patch_size_px / 2))

        return read_patch_as_pil(
            path=image_path,
            left=left,
            top=top,
            width=self.patch_size_px,
            height=self.patch_size_px,
        )


class EncoderWrapper:
    def __init__(self, model, device, image_size: int, use_amp: bool):
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def encode_batch(self, patches, batch_size: int):
        import numpy as np

        outputs = []
        for start in range(0, len(patches), batch_size):
            batch_imgs = patches[start:start + batch_size]
            batch_tensor = torch.stack([self.transform(p) for p in batch_imgs], dim=0).to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                feats = self.model.encode(batch_tensor)

            outputs.append(feats.detach().cpu().numpy())

        return np.concatenate(outputs, axis=0)