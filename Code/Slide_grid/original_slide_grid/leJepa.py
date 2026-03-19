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
from shapely.geometry import box
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time

# --------------------------
# Hyperparameters & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp"

IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 4
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER
HALF_CROP = CROP_SIZE // 2

BATCH_SIZE = 16
EPOCHS = 300

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# --------------------------
# Model Definitions
# --------------------------
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

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
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class LeJepaPredictor(nn.Module):
    def __init__(self, embed_dim=128, predictor_depth=2, num_heads=4):
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation='gelu',
                batch_first=True
            ) for _ in range(predictor_depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_embeds, mask_pos_embeds):
        B = context_embeds.shape[0]
        N_mask = mask_pos_embeds.shape[1]

        mask_tokens = self.mask_token.repeat(B, N_mask, 1) + mask_pos_embeds
        x = torch.cat([context_embeds, mask_tokens], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, -N_mask:, :]

# --------------------------
# Utility Functions
# --------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_arrays(embeddings, labels, train_ratio, val_ratio, test_ratio, seed):
    n = len(embeddings)
    indices = np.arange(n)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        "train_embeddings": embeddings[train_idx],
        "val_embeddings": embeddings[val_idx],
        "test_embeddings": embeddings[test_idx],
        "train_labels": labels[train_idx],
        "val_labels": labels[val_idx],
        "test_labels": labels[test_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

# --------------------------
# Main Execution
# --------------------------
def main():
    start_time = time.time()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = "data/models"
    embedding_dir = "data/embeddings"
    label_dir = "data/label"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    encoder = LeJepaEncoder().to(device)
    predictor = LeJepaPredictor().to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-4
    )

    criterion = nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("Loading shapefile...")
    gdf = gpd.read_file(ANNOTATED_COR)
    original_crs = gdf.crs

    if 'temp_id' not in gdf.columns:
        gdf['temp_id'] = range(len(gdf))

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))

    patches = []
    patch_labels = []
    successful_rows = []
    extracted_ids = set()

    print("Extracting patches...")

    for tif_path in tqdm(tif_files):
        with rasterio.open(tif_path) as src:

            bounds = src.bounds
            img_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

            bbox_gdf = gpd.GeoDataFrame({'geometry': [img_box]}, crs=src.crs)
            if bbox_gdf.crs != gdf.crs:
                bbox_gdf = bbox_gdf.to_crs(gdf.crs)

            intersecting = gdf[gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]

            if intersecting.empty:
                continue

            if intersecting.crs != src.crs:
                intersecting = intersecting.to_crs(src.crs)

            for idx, row in intersecting.iterrows():
                temp_id = row['temp_id']
                if temp_id in extracted_ids:
                    continue

                x, y = row.geometry.x, row.geometry.y
                py, px = src.index(x, y)

                window = rasterio.windows.Window(
                    px - HALF_CROP,
                    py - HALF_CROP,
                    CROP_SIZE,
                    CROP_SIZE
                )

                tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                tile = np.moveaxis(tile, 0, -1)

                if tile.shape[0] == CROP_SIZE and tile.shape[1] == CROP_SIZE:
                    img = Image.fromarray(tile.astype('uint8'))
                    patches.append(transform(img))

                    label_val = row.get('Tree', f"id_{temp_id}")
                    patch_labels.append(str(label_val))

                    successful_rows.append(row)
                    extracted_ids.add(temp_id)

    if not patches:
        print("No patches extracted.")
        return

    print(f"Total patches: {len(patches)}")

    valid_gdf = gpd.GeoDataFrame(successful_rows, crs=original_crs)

    # Extract coordinates (CRITICAL PART)
    coords = np.array([[geom.x, geom.y] for geom in valid_gdf.geometry])

    dataset = TensorDataset(torch.stack(patches))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder.train()
    predictor.train()

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    keep_num = int(num_patches * 0.25)

    print("Training LeJEPA...")

    for epoch in tqdm(range(EPOCHS)):
        for batch in dataloader:
            imgs = batch[0].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                target = encoder(imgs)

            ids_shuffle = torch.argsort(torch.rand(imgs.shape[0], num_patches, device=device), dim=1)
            ids_keep = ids_shuffle[:, :keep_num]
            ids_mask = ids_shuffle[:, keep_num:]

            context = encoder(imgs, ids_keep)
            pos_embed = encoder.pos_embed.expand(imgs.shape[0], -1, -1)

            mask_pos = torch.gather(
                pos_embed,
                1,
                ids_mask.unsqueeze(-1).expand(-1, -1, encoder.embed_dim)
            )

            pred = predictor(context, mask_pos)

            target_mask = torch.gather(
                target,
                1,
                ids_mask.unsqueeze(-1).expand(-1, -1, encoder.embed_dim)
            )

            loss = criterion(pred, target_mask)
            loss.backward()
            optimizer.step()

    print("Extracting embeddings...")

    encoder.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE):
            imgs = batch[0].to(device)
            emb = encoder(imgs).mean(dim=1).cpu().numpy()
            all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings)
    labels = np.array(patch_labels)

    assert len(embeddings) == len(labels) == len(coords)

    # Save EVERYTHING needed for sliding grid
    np.save(f"{embedding_dir}/embeddings.npy", embeddings)
    np.save(f"{label_dir}/labels.npy", labels)
    np.save(f"{embedding_dir}/coords.npy", coords)

    print("Saved embeddings, labels, and coordinates")

    total_time = (time.time() - start_time) / 60
    print(f"Done in {total_time:.2f} minutes")


if __name__ == "__main__":
    main()