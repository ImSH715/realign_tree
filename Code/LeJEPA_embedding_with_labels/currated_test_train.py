import os
import time
import glob
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import rasterio
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

# --------------------------
# Hyperparameters & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/aca21jo/label_tree_shp/Project/Annotated tree centroids/trees_32718.shp"

IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 4
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER
HALF_CROP = CROP_SIZE // 2
BATCH_SIZE = 16
EPOCHS = 300

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducibility
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.blocks = nn.ModuleList([encoder_layer for _ in range(depth)])
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
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.blocks = nn.ModuleList([predictor_layer for _ in range(predictor_depth)])
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


def make_split_indices(n_samples, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx


def extract_embeddings_for_subset(encoder, dataset_subset, device, batch_size=BATCH_SIZE):
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Embeddings", leave=False):
            batch_imgs = batch[0].to(device)
            embeds = encoder(batch_imgs).mean(dim=1).cpu().numpy()
            all_embeddings.append(embeds)

    if len(all_embeddings) == 0:
        return np.empty((0, encoder.embed_dim), dtype=np.float32)

    return np.concatenate(all_embeddings, axis=0)


# --------------------------
# Main Execution
# --------------------------
def main():
    start_time = time.time()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. Setup output directories
    model_dir = "data/models"
    embedding_dir = "data/embeddings"
    label_dir = "data/label"
    split_dir = "data/splits"

    for d in [model_dir, embedding_dir, label_dir, split_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Initialize Model & Tools
    encoder = LeJepaEncoder(img_size=IMG_SIZE).to(device)
    predictor = LeJepaPredictor().to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-4
    )
    criterion = nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Shapefile & Prep IDs
    print("Loading Original Shapefile...")
    gdf = gpd.read_file(ANNOTATED_COR)
    original_crs = gdf.crs
    if 'temp_id' not in gdf.columns:
        gdf['temp_id'] = range(len(gdf))

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    if not tif_files:
        print("No .tif files found.")
        return

    patches = []
    patch_labels = []
    successful_rows = []
    extracted_temp_ids = set()

    # 3. Optimized Extraction Mechanism
    print(f"\nStarting extraction for {len(tif_files)} TIF files...")

    for tif_path in tqdm(tif_files, desc="Processing TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                current_gdf = gdf.copy()
                if current_gdf.crs != src.crs:
                    current_gdf = current_gdf.to_crs(src.crs)

                b = src.bounds
                img_box = box(b.left, b.bottom, b.right, b.top)
                contained = current_gdf[current_gdf.geometry.intersects(img_box)]

                for idx, row in contained.iterrows():
                    temp_id = row['temp_id']

                    if temp_id in extracted_temp_ids:
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
                        img_pil = Image.fromarray(tile.astype('uint8'))
                        patches.append(transform(img_pil))

                        label_val = row.get('Tree') or row.get('tree') or row.get('id') or f"tree_{temp_id}"
                        patch_labels.append(str(label_val))

                        successful_rows.append(gdf.loc[idx])
                        extracted_temp_ids.add(temp_id)

        except Exception as e:
            tqdm.write(f"Error reading {tif_path}: {e}")

    if not patches:
        print("No patches extracted. Check file paths and CRS.")
        return

    print(f"\nTotal Successfully Extracted Patches: {len(patches)}")

    # 4. Internal validation count
    valid_points_gdf = gpd.GeoDataFrame(successful_rows, crs=original_crs)
    print(f"Kept {len(valid_points_gdf)} valid points for internal validation.")

    # 5. Build full dataset
    all_patches_tensor = torch.stack(patches)
    final_labels = np.array(patch_labels)

    assert len(all_patches_tensor) == len(final_labels) == len(valid_points_gdf), "Count mismatch before split!"

    full_dataset = TensorDataset(all_patches_tensor)

    # 6. Train / Val / Test split
    n_samples = len(full_dataset)
    train_idx, val_idx, test_idx = make_split_indices(
        n_samples,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print("\nDataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Save split indices for reproducibility
    np.save(os.path.join(split_dir, "train_indices.npy"), train_idx)
    np.save(os.path.join(split_dir, "val_indices.npy"), val_idx)
    np.save(os.path.join(split_dir, "test_indices.npy"), test_idx)

    # 7. LeJEPA Training Loop (train split only)
    encoder.train()
    predictor.train()

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    keep_num = int(num_patches * 0.25)

    print("\nStarting LeJEPA Training on TRAIN split only...")
    epoch_pbar = tqdm(range(EPOCHS), desc="Training Model")
    for epoch in epoch_pbar:
        epoch_loss = 0.0

        for batch in train_loader:
            batch_imgs = batch[0].to(device)
            curr_batch_size = batch_imgs.shape[0]

            optimizer.zero_grad()

            with torch.no_grad():
                target = encoder(batch_imgs).detach()

            ids_shuffle = torch.argsort(
                torch.rand(curr_batch_size, num_patches, device=device),
                dim=1
            )
            ids_keep = ids_shuffle[:, :keep_num]
            ids_mask = ids_shuffle[:, keep_num:]

            context_embeds = encoder(batch_imgs, ids_keep=ids_keep)

            mask_pos_embeds = encoder.pos_embed.expand(curr_batch_size, -1, -1)
            mask_pos_embeds = torch.gather(
                mask_pos_embeds,
                dim=1,
                index=ids_mask.unsqueeze(-1).expand(-1, -1, encoder.embed_dim)
            )

            pred_features = predictor(context_embeds, mask_pos_embeds)
            actual_target_features = torch.gather(
                target,
                dim=1,
                index=ids_mask.unsqueeze(-1).expand(-1, -1, encoder.embed_dim)
            )

            loss = criterion(pred_features, actual_target_features)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_pbar.set_postfix(Loss=f"{avg_loss:.6f}")

    torch.save(encoder.state_dict(), os.path.join(model_dir, "lejepa_encoder.pth"))
    print("\nTraining Complete. Extracting Final Embeddings by split...")

    # 8. Extract embeddings for each split
    train_embeddings = extract_embeddings_for_subset(encoder, train_dataset, device)
    val_embeddings = extract_embeddings_for_subset(encoder, val_dataset, device)
    test_embeddings = extract_embeddings_for_subset(encoder, test_dataset, device)

    train_labels = final_labels[train_idx]
    val_labels = final_labels[val_idx]
    test_labels = final_labels[test_idx]

    # Final checks
    assert len(train_embeddings) == len(train_labels), "Train count mismatch!"
    assert len(val_embeddings) == len(val_labels), "Val count mismatch!"
    assert len(test_embeddings) == len(test_labels), "Test count mismatch!"

    # 9. Save outputs
    np.save(os.path.join(embedding_dir, "train_embeddings.npy"), train_embeddings)
    np.save(os.path.join(embedding_dir, "val_embeddings.npy"), val_embeddings)
    np.save(os.path.join(embedding_dir, "test_embeddings.npy"), test_embeddings)

    np.save(os.path.join(label_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(label_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(label_dir, "test_labels.npy"), test_labels)

    # Optional: save full labels too
    np.save(os.path.join(label_dir, "labels_all.npy"), final_labels)

    total_time = (time.time() - start_time) / 60
    print(f"\nSuccess! Total samples: {n_samples}")
    print(f"Train samples: {len(train_embeddings)}")
    print(f"Val samples:   {len(val_embeddings)}")
    print(f"Test samples:  {len(test_embeddings)}")

    print(f"\nSaved model to: {os.path.join(model_dir, 'lejepa_encoder.pth')}")
    print(f"Saved train embeddings to: {os.path.join(embedding_dir, 'train_embeddings.npy')}")
    print(f"Saved val embeddings to:   {os.path.join(embedding_dir, 'val_embeddings.npy')}")
    print(f"Saved test embeddings to:  {os.path.join(embedding_dir, 'test_embeddings.npy')}")
    print(f"Saved train labels to:     {os.path.join(label_dir, 'train_labels.npy')}")
    print(f"Saved val labels to:       {os.path.join(label_dir, 'val_labels.npy')}")
    print(f"Saved test labels to:      {os.path.join(label_dir, 'test_labels.npy')}")
    print(f"Saved split indices to:    {split_dir}")
    print(f"Total Execution Time: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()