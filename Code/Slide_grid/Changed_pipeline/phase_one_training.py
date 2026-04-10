import os
import glob
import time
import random
import copy
import warnings
import gc

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS & HYPERPARAMETERS
# =========================================================
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"

MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)


# image settings
IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER   # 896
HALF_CROP = CROP_SIZE // 2               # 448

# training
BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-5
WEIGHT_DECAY = 1e-4
SEED = 42
EMA_MOMENTUM = 0.998

# dataset sampling
SAMPLES_PER_EPOCH = 8000        # total random samples per epoch
MAX_CENTER_JITTER_M = 3.0        # view-to-view center jitter
MIN_VALID_PIXEL_RATIO = 0.20     # reject patches that are mostly empty/zero
MAX_SAMPLE_RETRIES = 70          # retries to find valid crop


ENCODER_SAVE_PATH = os.path.join(MODEL_DIR, f"encoder_phase1_large_epoch{EPOCHS}.pth")
# dataloader
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (device.type == "cuda")


# =========================================================
# 2. MODEL DEFINITION
# =========================================================
class LeJepaEncoder(nn.Module):
    def __init__(
        self,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=3,
        embed_dim=128,
        depth=4,
        num_heads=4
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation="gelu",
                batch_first=True
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]
        x = x + self.pos_embed

        if ids_keep is not None:
            D = x.shape[-1]
            x = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)


class LeJepaPredictor(nn.Module):
    def __init__(self, embed_dim=128, depth=2, num_heads=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation="gelu",
                batch_first=True
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context, mask_pos):
        B = context.shape[0]
        N = mask_pos.shape[1]

        mask_tokens = self.mask_token.repeat(B, N, 1) + mask_pos
        x = torch.cat([context, mask_tokens], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, -N:, :]


# =========================================================
# 3. UTILS
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_target(student, target, momentum):
    with torch.no_grad():
        for s, t in zip(student.parameters(), target.parameters()):
            t.data = momentum * t.data + (1.0 - momentum) * s.data


def build_tif_index(tif_dir):
    tif_files = glob.glob(os.path.join(tif_dir, "2023-*", "*.tif"))
    index = []

    print("Indexing TIF boundaries...")
    for f in tqdm(tif_files, leave=False):
        try:
            with rasterio.open(f) as src:
                bounds = src.bounds
                width = bounds.right - bounds.left
                height = bounds.top - bounds.bottom

                if width <= 0 or height <= 0:
                    continue

                index.append({
                    "path": f,
                    "bounds": bounds,
                    "area": float(width * height)
                })
        except Exception:
            continue

    if len(index) == 0:
        raise RuntimeError("No valid TIF files found.")

    total_area = sum(item["area"] for item in index)
    for item in index:
        item["sampling_weight"] = item["area"] / total_area

    print(f"Indexed TIF count: {len(index)}")
    return index


def choose_random_tif(tif_index):
    weights = [item["sampling_weight"] for item in tif_index]
    return random.choices(tif_index, weights=weights, k=1)[0]


def sample_random_xy_with_margin(bounds, margin):
    left = bounds.left + margin
    right = bounds.right - margin
    bottom = bounds.bottom + margin
    top = bounds.top - margin

    if left >= right or bottom >= top:
        return None

    x = random.uniform(left, right)
    y = random.uniform(bottom, top)
    return x, y


def valid_pixel_ratio(tile):
    """
    tile: np.ndarray [3, H, W]
    zero-filled border/empty patches are common with boundless reads.
    We reject mostly-empty patches.
    """
    if tile is None or tile.size == 0:
        return 0.0

    nonzero = np.any(tile > 0, axis=0)
    return float(nonzero.mean())


def normalize_tile_to_tensor(tile):
    img_t = torch.from_numpy(tile).float().unsqueeze(0) / 255.0
    img_t = F.interpolate(
        img_t,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_t = (img_t - mean_t) / std_t

    return img_t.squeeze(0)  # [3, 448, 448]


def read_crop_as_tensor(src, x, y):
    """
    Extract 896x896 around (x, y), resize to 448x448, normalize.
    """
    try:
        py, px = src.index(x, y)

        window = Window(
            col_off=px - HALF_CROP,
            row_off=py - HALF_CROP,
            width=CROP_SIZE,
            height=CROP_SIZE
        )

        tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)

        if tile.shape != (3, CROP_SIZE, CROP_SIZE):
            return None, 0.0

        ratio = valid_pixel_ratio(tile)
        img_t = normalize_tile_to_tensor(tile)
        return img_t, ratio

    except Exception:
        return None, 0.0


# =========================================================
# 4. DATASET
# =========================================================
class RandomForestJEPADataset(Dataset):
    """
    Phase 1 dataset:
    - samples random locations across the full orthomosaic collection
    - creates two jittered views around the same latent object/region
    - uses no labels
    """
    def __init__(
        self,
        tif_index,
        samples_per_epoch=20000,
        max_center_jitter_m=8.0,
        min_valid_pixel_ratio=0.20,
        max_sample_retries=20
    ):
        self.tif_index = tif_index
        self.samples_per_epoch = samples_per_epoch
        self.max_center_jitter_m = max_center_jitter_m
        self.min_valid_pixel_ratio = min_valid_pixel_ratio
        self.max_sample_retries = max_sample_retries

    def __len__(self):
        return self.samples_per_epoch

    def _sample_valid_pair(self):
        for _ in range(self.max_sample_retries):
            tif_item = choose_random_tif(self.tif_index)
            tif_path = tif_item["path"]
            bounds = tif_item["bounds"]

            # avoid choosing centers too close to borders
            # if CRS is meter-based, HALF_CROP approximately works as spatial buffer
            # if not exact, boundless read still protects us
            sampled = sample_random_xy_with_margin(bounds, margin=1.0)
            if sampled is None:
                continue

            x0, y0 = sampled

            jx1 = x0 + random.uniform(-self.max_center_jitter_m, self.max_center_jitter_m)
            jy1 = y0 + random.uniform(-self.max_center_jitter_m, self.max_center_jitter_m)

            jx2 = x0 + random.uniform(-self.max_center_jitter_m, self.max_center_jitter_m)
            jy2 = y0 + random.uniform(-self.max_center_jitter_m, self.max_center_jitter_m)

            try:
                with rasterio.open(tif_path) as src:
                    img1, ratio1 = read_crop_as_tensor(src, jx1, jy1)
                    img2, ratio2 = read_crop_as_tensor(src, jx2, jy2)

                if img1 is None or img2 is None:
                    continue

                if ratio1 < self.min_valid_pixel_ratio or ratio2 < self.min_valid_pixel_ratio:
                    continue

                return img1, img2

            except Exception:
                continue

        # fallback if repeated failures
        z = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        return z, z

    def __getitem__(self, idx):
        img1, img2 = self._sample_valid_pair()
        return img1, img2


# =========================================================
# 5. TRAIN LEJEPA
# =========================================================
def train_lejepa(train_loader):
    student = LeJepaEncoder().to(device)
    predictor = LeJepaPredictor().to(device)
    target = copy.deepcopy(student).to(device)

    for p in target.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    criterion = nn.MSELoss()

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    keep = int(num_patches * 0.5)

    best_loss = float("inf")

    print("\n[Phase 1] Training LeJEPA on full orthomosaic random samples...")
    for epoch in range(EPOCHS):
        student.train()
        predictor.train()

        total_loss = 0.0
        num_steps = 0

        for img1, img2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    with torch.no_grad():
                        tgt = target(img2)

                    ids = torch.argsort(
                        torch.rand(img1.shape[0], num_patches, device=device), dim=1
                    )
                    keep_ids = ids[:, :keep]
                    mask_ids = ids[:, keep:]

                    ctx = student(img1, keep_ids)
                    pos = student.pos_embed.expand(img1.shape[0], -1, -1)

                    mask_pos = torch.gather(
                        pos,
                        1,
                        mask_ids.unsqueeze(-1).expand(-1, -1, pos.shape[-1])
                    )

                    pred = predictor(ctx, mask_pos)

                    tgt_mask = torch.gather(
                        tgt,
                        1,
                        mask_ids.unsqueeze(-1).expand(-1, -1, tgt.shape[-1])
                    )

                    loss = criterion(pred, tgt_mask)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                with torch.no_grad():
                    tgt = target(img2)

                ids = torch.argsort(
                    torch.rand(img1.shape[0], num_patches, device=device), dim=1
                )
                keep_ids = ids[:, :keep]
                mask_ids = ids[:, keep:]

                ctx = student(img1, keep_ids)
                pos = student.pos_embed.expand(img1.shape[0], -1, -1)

                mask_pos = torch.gather(
                    pos,
                    1,
                    mask_ids.unsqueeze(-1).expand(-1, -1, pos.shape[-1])
                )

                pred = predictor(ctx, mask_pos)

                tgt_mask = torch.gather(
                    tgt,
                    1,
                    mask_ids.unsqueeze(-1).expand(-1, -1, tgt.shape[-1])
                )

                loss = criterion(pred, tgt_mask)
                loss.backward()
                optimizer.step()

            update_target(student, target, EMA_MOMENTUM)

            total_loss += loss.item()
            num_steps += 1

        mean_loss = total_loss / max(1, num_steps)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {mean_loss:.6f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(student.state_dict(), ENCODER_SAVE_PATH)
            print(f"[BEST SAVED] epoch={epoch+1}, loss={best_loss:.6f}")

    print(f"\nFinal best loss: {best_loss:.6f}")
    print(f"[SAVED] Best encoder -> {ENCODER_SAVE_PATH}")

    del predictor, target, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return student


# =========================================================
# 6. MAIN
# =========================================================
def main():
    start_time = time.time()
    set_seed(SEED)

    print(f"Device: {device}")

    tif_index = build_tif_index(BASE_DIR)

    train_dataset = RandomForestJEPADataset(
        tif_index=tif_index,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        max_center_jitter_m=MAX_CENTER_JITTER_M,
        min_valid_pixel_ratio=MIN_VALID_PIXEL_RATIO,
        max_sample_retries=MAX_SAMPLE_RETRIES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    train_lejepa(train_loader)

    print("\n[PHASE 1 DONE]")
    print(f"Total time: {(time.time() - start_time) / 60:.2f} min")


if __name__ == "__main__":
    main()