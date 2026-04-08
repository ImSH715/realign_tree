import os
import glob
import time
import random
import copy
import warnings
import gc

import numpy as np
import pandas as pd
import geopandas as gpd
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
ANNOTATED_SHP = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"

LABEL_COL = "Tree"

MODEL_DIR = "data/models"
EXPORT_DIR = "data/exports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

ENCODER_SAVE_PATH = os.path.join(MODEL_DIR, "encoder.pth")

# image settings: must match small_tif
IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER   # 896
HALF_CROP = CROP_SIZE // 2               # 448

# training
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
EMA_MOMENTUM = 0.996

# tree-centered self-supervised augmentation
VIEWS_PER_POINT = 4          # how many training samples per annotated tree
MAX_JITTER_METERS = 8.0      # random crop center shift around each labeled tree point

# export
EXPORT_BATCH_SIZE = 64

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
                index.append({
                    "path": f,
                    "bounds": src.bounds
                })
        except Exception:
            continue
    return index


def find_tif_for_point(x, y, tif_index):
    for item in tif_index:
        b = item["bounds"]
        if b.left <= x <= b.right and b.bottom <= y <= b.top:
            return item["path"]
    return None


def read_crop_as_tensor(src, x, y):
    """
    Extract 896x896 around (x, y), resize to 448x448, normalize.
    This MUST match small_tif preprocessing exactly.
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
            return None

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

    except Exception:
        return None


def load_annotations(shp_path, label_col):
    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf[label_col] = gdf[label_col].astype(str).str.strip().str.upper()
    return gdf


# =========================================================
# 4. DATASET
# =========================================================
class TreeCrownJEPADataset(Dataset):
    """
    For each annotated tree point, generate two jittered views around the same crown.
    This makes pretraining much more aligned with small_tif's realignment purpose.
    """
    def __init__(self, records, views_per_point=4, max_jitter_m=8.0):
        self.records = records
        self.views_per_point = views_per_point
        self.max_jitter_m = max_jitter_m

    def __len__(self):
        return len(self.records) * self.views_per_point

    def __getitem__(self, idx):
        base_idx = idx % len(self.records)
        rec = self.records[base_idx]

        tif_path = rec["tif_path"]
        x0 = rec["x"]
        y0 = rec["y"]

        with rasterio.open(tif_path) as src:
            # view A
            jx1 = x0 + random.uniform(-self.max_jitter_m, self.max_jitter_m)
            jy1 = y0 + random.uniform(-self.max_jitter_m, self.max_jitter_m)
            img1 = read_crop_as_tensor(src, jx1, jy1)

            # view B
            jx2 = x0 + random.uniform(-self.max_jitter_m, self.max_jitter_m)
            jy2 = y0 + random.uniform(-self.max_jitter_m, self.max_jitter_m)
            img2 = read_crop_as_tensor(src, jx2, jy2)

            # fallback to exact center if jitter crop failed
            if img1 is None:
                img1 = read_crop_as_tensor(src, x0, y0)
            if img2 is None:
                img2 = read_crop_as_tensor(src, x0, y0)

        # extremely rare fallback
        if img1 is None:
            img1 = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        if img2 is None:
            img2 = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

        return img1, img2


class TreePointExportDataset(Dataset):
    """
    Exact-center dataset for final embedding export.
    """
    def __init__(self, records, label_to_idx):
        self.records = records
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tif_path = rec["tif_path"]
        x = rec["x"]
        y = rec["y"]
        label = rec["label"]

        with rasterio.open(tif_path) as src:
            img = read_crop_as_tensor(src, x, y)

        if img is None:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

        label_idx = self.label_to_idx[label]
        coord = torch.tensor([x, y], dtype=torch.float32)

        return img, coord, label_idx


# =========================================================
# 5. PREPARE RECORDS
# =========================================================
def build_records(gdf, tif_index, label_col):
    records = []
    skipped = 0

    print("Matching annotation points to TIFs...")
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), leave=False):
        x = float(row.geometry.x)
        y = float(row.geometry.y)
        label = str(row[label_col]).strip().upper()

        tif_path = find_tif_for_point(x, y, tif_index)
        if tif_path is None:
            skipped += 1
            continue

        records.append({
            "x": x,
            "y": y,
            "label": label,
            "tif_path": tif_path
        })

    print(f"Usable annotated points: {len(records)}")
    print(f"Skipped annotated points: {skipped}")
    return records


# =========================================================
# 6. TRAIN LEJEPA
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
    keep = int(num_patches * 0.25)

    print("\n[Phase 1] Training tree-crown-centered LeJEPA...")
    for epoch in range(EPOCHS):
        student.train()
        predictor.train()

        total_loss = 0.0
        num_steps = 0

        for img1, img2 in tqdm(train_loader, leave=False):
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

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / max(1, num_steps):.6f}")

    torch.save(student.state_dict(), ENCODER_SAVE_PATH)
    print(f"\n[SAVED] Encoder -> {ENCODER_SAVE_PATH}")

    del predictor, target, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return student


# =========================================================
# 7. EXPORT ANNOTATED EMBEDDINGS TO NPY
# =========================================================
def export_embeddings(encoder, export_loader, class_names):
    encoder.eval()

    all_embeds = []
    all_coords = []
    all_labels = []

    print("\n[Phase 2] Exporting annotated tree embeddings to .npy...")
    with torch.no_grad():
        for imgs, coords, labels in tqdm(export_loader, leave=False):
            imgs = imgs.to(device, non_blocking=True)

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    emb = encoder(imgs).mean(dim=1)
            else:
                emb = encoder(imgs).mean(dim=1)

            all_embeds.append(emb.cpu().numpy().astype(np.float32))
            all_coords.append(coords.numpy().astype(np.float32))
            all_labels.append(labels.numpy().astype(np.int64))

    embeddings = np.concatenate(all_embeds, axis=0)
    coords = np.concatenate(all_coords, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    class_names = np.array(class_names, dtype=object)

    np.save(os.path.join(EXPORT_DIR, "annot_embeddings.npy"), embeddings)
    np.save(os.path.join(EXPORT_DIR, "annot_coords.npy"), coords)
    np.save(os.path.join(EXPORT_DIR, "annot_labels.npy"), labels)
    np.save(os.path.join(EXPORT_DIR, "class_names.npy"), class_names)

    print(f"[SAVED] {os.path.join(EXPORT_DIR, 'annot_embeddings.npy')}")
    print(f"[SAVED] {os.path.join(EXPORT_DIR, 'annot_coords.npy')}")
    print(f"[SAVED] {os.path.join(EXPORT_DIR, 'annot_labels.npy')}")
    print(f"[SAVED] {os.path.join(EXPORT_DIR, 'class_names.npy')}")

    print("\nExport shapes:")
    print("embeddings:", embeddings.shape)
    print("coords:", coords.shape)
    print("labels:", labels.shape)
    print("class_names:", class_names.shape)


# =========================================================
# 8. MAIN
# =========================================================
def main():
    start_time = time.time()
    set_seed(SEED)

    print(f"Device: {device}")
    print("\nLoading annotations...")
    gdf = load_annotations(ANNOTATED_SHP, LABEL_COL)

    tif_index = build_tif_index(BASE_DIR)
    if len(tif_index) == 0:
        raise RuntimeError("No valid TIF files found.")

    records = build_records(gdf, tif_index, LABEL_COL)
    if len(records) == 0:
        raise RuntimeError("No usable annotation records found.")

    class_names = sorted(list({rec["label"] for rec in records}))
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    # -------------------------
    # Train loader (tree-centered JEPA views)
    # -------------------------
    train_dataset = TreeCrownJEPADataset(
        records=records,
        views_per_point=VIEWS_PER_POINT,
        max_jitter_m=MAX_JITTER_METERS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    encoder = train_lejepa(train_loader)

    del train_loader, train_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------
    # Export exact-center embeddings
    # -------------------------
    export_dataset = TreePointExportDataset(records, label_to_idx)
    export_loader = DataLoader(
        export_dataset,
        batch_size=EXPORT_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    export_embeddings(encoder, export_loader, class_names)

    print("\n[ALL DONE]")
    print(f"Total time: {(time.time() - start_time) / 60:.2f} min")


if __name__ == "__main__":
    main()