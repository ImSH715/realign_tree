import torch
import torch.nn as nn
import numpy as np
import rasterio
import glob
import os
from tqdm import tqdm
import gc

from lejepa_fixed import LeJepaEncoder

# --------------------------
# CONFIG
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"

SAVE_DIR = "data/chunks"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 448
CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER
HALF_CROP = CROP_SIZE // 2

GRID_STRIDE = CROP_SIZE // 2  # 50% overlap
CHUNK_SIZE = 128
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# NORMALIZATION (IMPORTANT)
# --------------------------
mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

# --------------------------
# LOAD MODEL (FIXED)
# --------------------------
encoder = LeJepaEncoder().to(device)

encoder.load_state_dict(
    torch.load("data/models/encoder.pth", map_location=device)
)

encoder.eval()

# --------------------------
# HELPER: PROCESS CHUNK
# --------------------------
def process_chunk(images, coords, chunk_id):
    if len(images) == 0:
        return chunk_id

    imgs = torch.from_numpy(np.stack(images)).permute(0,3,1,2).float().to(device) / 255.0

    # Normalize (CRITICAL)
    imgs = (imgs - mean) / std

    all_embeds = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, imgs.shape[0], BATCH_SIZE):
            batch = imgs[i:i+BATCH_SIZE]
            emb = encoder(batch).mean(dim=1).cpu().numpy().astype(np.float16)
            all_embeds.append(emb)

    embeds = np.concatenate(all_embeds)
    coords = np.array(coords).astype(np.float32)

    # 🔥 SAVE TO DISK (KEY FIX)
    np.save(f"{SAVE_DIR}/emb_{chunk_id}.npy", embeds)
    np.save(f"{SAVE_DIR}/coord_{chunk_id}.npy", coords)

    return chunk_id + 1

# --------------------------
# MAIN
# --------------------------
def main():
    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))

    chunk_id = 0

    print("\n[Phase 2] OOM-safe dense extraction...")

    for tif in tqdm(tif_files, desc="Processing TIFs"):
        try:
            with rasterio.open(tif) as src:
                
                # 🔥 READ FULL IMAGE ONCE (FAST)
                img = src.read([1,2,3])
                img = np.moveaxis(img, 0, -1)

                h, w = img.shape[:2]

                batch_imgs = []
                batch_coords = []

                for py in range(HALF_CROP, h - HALF_CROP, GRID_STRIDE):
                    for px in range(HALF_CROP, w - HALF_CROP, GRID_STRIDE):

                        tile = img[
                            py-HALF_CROP:py+HALF_CROP,
                            px-HALF_CROP:px+HALF_CROP
                        ]

                        x, y = src.xy(py, px)

                        batch_imgs.append(tile)
                        batch_coords.append((x, y))

                        # 🔥 PROCESS CHUNK
                        if len(batch_imgs) >= CHUNK_SIZE:
                            chunk_id = process_chunk(batch_imgs, batch_coords, chunk_id)

                            batch_imgs.clear()
                            batch_coords.clear()
                            gc.collect()

                # Process remaining
                if batch_imgs:
                    chunk_id = process_chunk(batch_imgs, batch_coords, chunk_id)

                    batch_imgs.clear()
                    batch_coords.clear()
                    gc.collect()

        except Exception as e:
            print(f"[ERROR SKIPPED] {tif}: {e}")
            continue

    print("\n[Phase 3] Merging chunks...")

    emb_files = sorted(glob.glob(f"{SAVE_DIR}/emb_*.npy"))
    coord_files = sorted(glob.glob(f"{SAVE_DIR}/coord_*.npy"))

    # 🔥 SAFE MERGE (still may use RAM, but controlled)
    embeddings = np.concatenate([np.load(f) for f in emb_files])
    coords = np.concatenate([np.load(f) for f in coord_files])

    np.save("data/embeddings.npy", embeddings)
    np.save("data/coords.npy", coords)

    print("\nDONE: embeddings + coords saved")

# --------------------------
# RUN
# --------------------------
if __name__ == "__main__":
    main()
