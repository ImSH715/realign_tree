import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
import rasterio
import geopandas as gpd
import glob
import random
import copy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import cKDTree
import gc

# --------------------------
# Hyperparameters & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"

SAVE_DIR = "data/chunks" # Temporary chunk storage to prevent OOM
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("data/models", exist_ok=True)

IMG_SIZE = 448
PATCH_SIZE = 16

CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER
HALF_CROP = CROP_SIZE // 2

BATCH_SIZE = 64
INFERENCE_BATCH_SIZE = 128
EPOCHS = 15
SEED = 42
EMA_MOMENTUM = 0.996

MAX_TRAIN_SAMPLES = 8000
GRID_STRIDE_PIXELS = CROP_SIZE // 2  # ~50% overlap
CHUNK_SIZE = 128  # Disk save unit to prevent OOM

# --------------------------
# Model Definitions
# --------------------------
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
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
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
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

# --------------------------
# Dataset & Utils
# --------------------------
class InMemoryDataset(Dataset):
    def __init__(self, images, coords, transform=None):
        self.images = images
        self.coords = coords
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        x, y = self.coords[idx]
        return img, x, y

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_target(student, target, momentum):
    with torch.no_grad():
        for s, t in zip(student.parameters(), target.parameters()):
            t.data = momentum * t.data + (1 - momentum) * s.data

# OOM-safe chunk processing function for Phase 2
def process_chunk(images, coords, chunk_id, encoder, device, mean, std):
    if len(images) == 0:
        return chunk_id

    imgs = torch.from_numpy(np.stack(images)).permute(0,3,1,2).float().to(device) / 255.0
    imgs = (imgs - mean) / std  # Fast normalization on GPU

    all_embeds = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, imgs.shape[0], INFERENCE_BATCH_SIZE):
            batch = imgs[i:i+INFERENCE_BATCH_SIZE]
            emb = encoder(batch).mean(dim=1).cpu().numpy().astype(np.float16)
            all_embeds.append(emb)

    embeds = np.concatenate(all_embeds)
    coords_np = np.array(coords).astype(np.float32)

    # Save to disk in chunks (frees up RAM)
    np.save(f"{SAVE_DIR}/emb_{chunk_id}.npy", embeds)
    np.save(f"{SAVE_DIR}/coord_{chunk_id}.npy", coords_np)

    return chunk_id + 1


# --------------------------
# MAIN
# --------------------------
def main():
    start_time = time.time()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))

    # =========================================================
    # Phase 1: Training (JEPA Encoder Training)
    # =========================================================
    print("\n[Phase 1] Sampling patches for training...")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_imgs, train_coords = [], []
    per_tif = max(1, MAX_TRAIN_SAMPLES // len(tif_files)) if tif_files else 0

    for tif in tqdm(tif_files):
        try:
            with rasterio.open(tif) as src:
                img = src.read([1,2,3])
                img = np.moveaxis(img, 0, -1)
                h, w = img.shape[:2]

                for _ in range(per_tif):
                    px = random.randint(HALF_CROP, w - HALF_CROP - 1)
                    py = random.randint(HALF_CROP, h - HALF_CROP - 1)
                    tile = img[py-HALF_CROP:py+HALF_CROP, px-HALF_CROP:px+HALF_CROP]
                    x, y = src.xy(py, px)

                    train_imgs.append(tile)
                    train_coords.append((x,y))
        except Exception as e:
            continue

    train_loader = DataLoader(
        InMemoryDataset(train_imgs, train_coords, transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    student = LeJepaEncoder().to(device)
    predictor = LeJepaPredictor().to(device)
    target = copy.deepcopy(student)

    for p in target.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(predictor.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.MSELoss()

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    keep = int(num_patches * 0.25)

    print("\n[Phase 1] Training Model...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, leave=False):
            imgs = batch[0].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    tgt = target(imgs)

                ids = torch.argsort(torch.rand(imgs.shape[0], num_patches, device=device), dim=1)
                keep_ids = ids[:, :keep]
                mask_ids = ids[:, keep:]

                ctx = student(imgs, keep_ids)
                pos = student.pos_embed.expand(imgs.shape[0], -1, -1)
                mask_pos = torch.gather(pos, 1, mask_ids.unsqueeze(-1).expand(-1, -1, student.pos_embed.shape[-1]))

                pred = predictor(ctx, mask_pos)
                tgt_mask = torch.gather(tgt, 1, mask_ids.unsqueeze(-1).expand(-1, -1, student.pos_embed.shape[-1]))

                loss = criterion(pred, tgt_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_target(student, target, EMA_MOMENTUM)
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: {total_loss/len(train_loader):.4f}")

    torch.save(student.state_dict(), "data/models/encoder.pth")

    # [IMPORTANT] Free Phase 1 memory
    del train_imgs, train_coords, train_loader, predictor, target, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # Phase 2: OOM-Safe Dense Extraction
    # =========================================================
    print("\n[Phase 2] Fast OOM-safe dense extraction...")
    
    student.eval()
    
    # Tensorization for normalization (Phase 2 only)
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    chunk_id = 0

    for tif in tqdm(tif_files, desc="Processing TIFs for Extraction"):
        try:
            with rasterio.open(tif) as src:
                img = src.read([1,2,3])
                img = np.moveaxis(img, 0, -1)
                h, w = img.shape[:2]

                batch_imgs, batch_coords = [], []

                for py in range(HALF_CROP, h - HALF_CROP, GRID_STRIDE_PIXELS):
                    for px in range(HALF_CROP, w - HALF_CROP, GRID_STRIDE_PIXELS):
                        tile = img[py-HALF_CROP:py+HALF_CROP, px-HALF_CROP:px+HALF_CROP]
                        x, y = src.xy(py, px)

                        batch_imgs.append(tile)
                        batch_coords.append((x, y))

                        if len(batch_imgs) >= CHUNK_SIZE:
                            chunk_id = process_chunk(batch_imgs, batch_coords, chunk_id, student, device, mean_t, std_t)
                            batch_imgs.clear()
                            batch_coords.clear()
                            gc.collect()

                # Process remaining data
                if batch_imgs:
                    chunk_id = process_chunk(batch_imgs, batch_coords, chunk_id, student, device, mean_t, std_t)
                    batch_imgs.clear()
                    batch_coords.clear()
                    gc.collect()
        except Exception as e:
            print(f"[ERROR SKIPPED] {tif}: {e}")
            continue

    # [IMPORTANT] Merge embedding memory
    print("\n[Phase 2] Merging chunks...")
    emb_files = sorted(glob.glob(f"{SAVE_DIR}/emb_*.npy"))
    coord_files = sorted(glob.glob(f"{SAVE_DIR}/coord_*.npy"))

    dense_embeddings = np.concatenate([np.load(f) for f in emb_files])
    dense_coords = np.concatenate([np.load(f) for f in coord_files])

    # Delete temporary chunk files (frees disk space, can be commented out if not desired)
    for f in emb_files + coord_files:
        os.remove(f)

    # [IMPORTANT] Free Phase 2 memory
    del student
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # Phase 3: Classification (Random Forest)
    # =========================================================
    print("\n[Phase 3] Training classifier & Predicting...")

    gdf = gpd.read_file(ANNOTATED_COR)
    tree = cKDTree(dense_coords)

    X, y = [], []

    # Match ground truth labels
    for _, row in gdf.iterrows():
        pt = [row.geometry.x, row.geometry.y]
        dist, idx = tree.query(pt)

        if dist < 50:
            X.append(dense_embeddings[idx])
            y.append(str(row[0])) # Ground truth label (modify according to the file, e.g., species, 1/0)

    # Train classifier and predict on all data
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(X, y)
    
    preds = clf.predict(dense_embeddings)

    # Save final outputs
    np.save("data/embeddings.npy", dense_embeddings)
    np.save("data/coords.npy", dense_coords)
    np.save("data/labels.npy", preds)

    print("\n[All Phases DONE]")
    print(f"Total time: {(time.time()-start_time)/60:.2f} min")

if __name__ == "__main__":
    main()