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
from shapely.geometry import box
from torch.utils.data import Dataset, DataLoader
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

EMA_MOMENTUM = 0.996 # Momentum for updating the target encoder

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
# Custom Dataset (Lazy Loading for Memory Efficiency)
# --------------------------
class TreePatchDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Open the image from disk and crop it only when requested (saves RAM)
        with rasterio.open(row['tif_path']) as src:
            py, px = src.index(row['x'], row['y'])
            window = rasterio.windows.Window(
                px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE
            )
            tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            tile = np.moveaxis(tile, 0, -1)

        img = Image.fromarray(tile.astype('uint8'))
        if self.transform:
            img = self.transform(img)

        label = row['label']
        coords = np.array([row['x'], row['y']], dtype=np.float32)
        
        return img, label, coords

# --------------------------
# Utility Functions
# --------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_target_encoder(student, target, momentum):
    """Update target encoder weights using Exponential Moving Average (EMA)"""
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), target.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

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

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading shapefile...")
    gdf = gpd.read_file(ANNOTATED_COR)
    if 'temp_id' not in gdf.columns:
        gdf['temp_id'] = range(len(gdf))

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    
    # 1. Collect metadata only (Do not load images into memory to prevent OOM)
    print("Extracting metadata...")
    metadata = []
    extracted_ids = set()

    for tif_path in tqdm(tif_files):
        try:
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

                for _, row in intersecting.iterrows():
                    temp_id = row['temp_id']
                    if temp_id in extracted_ids:
                        continue
                    
                    label_val = row.get('Tree', f"id_{temp_id}")
                    metadata.append({
                        'temp_id': temp_id,
                        'tif_path': tif_path,
                        'x': row.geometry.x,
                        'y': row.geometry.y,
                        'label': str(label_val)
                    })
                    extracted_ids.add(temp_id)
        except Exception as e:
            print(f"\nSkipping {os.path.basename(tif_path)} due to error: {e}")
            continue

    if not metadata:
        print("No patches extracted.")
        return

    df_meta = pd.DataFrame(metadata)
    print(f"Total points found: {len(df_meta)}")

    # 2. Tree ID-based split to prevent spatial data leakage
    unique_ids = df_meta['temp_id'].unique()
    np.random.shuffle(unique_ids)
    
    n_total = len(unique_ids)
    train_end = int(n_total * TRAIN_RATIO)
    val_end = train_end + int(n_total * VAL_RATIO)

    train_ids = unique_ids[:train_end]
    val_ids = unique_ids[train_end:val_end]
    # test_ids = unique_ids[val_end:] # Can be used for testing logic if needed

    train_df = df_meta[df_meta['temp_id'].isin(train_ids)]
    val_df = df_meta[df_meta['temp_id'].isin(val_ids)]

    print(f"Split: Train {len(train_df)} patches, Val {len(val_df)} patches.")

    # 3. Prepare DataLoaders
    train_dataset = TreePatchDataset(train_df, transform=transform)
    val_dataset = TreePatchDataset(val_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Initialize Models (Separate Student & Target Encoders)
    student_encoder = LeJepaEncoder().to(device)
    predictor = LeJepaPredictor().to(device)
    
    target_encoder = copy.deepcopy(student_encoder) # Duplicate student for EMA target
    for param in target_encoder.parameters():
        param.requires_grad = False # Freeze target encoder gradients

    optimizer = torch.optim.AdamW(list(student_encoder.parameters()) + list(predictor.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    keep_num = int(num_patches * 0.25)

    print("Training LeJEPA...")
    best_val_loss = float('inf')

    # 5. Training & Validation Loop
    for epoch in range(EPOCHS):
        # -- Train Phase --
        student_encoder.train()
        predictor.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            imgs = batch[0].to(device)
            optimizer.zero_grad()

            # Get target outputs without updating the target encoder
            with torch.no_grad():
                target = target_encoder(imgs)

            ids_shuffle = torch.argsort(torch.rand(imgs.shape[0], num_patches, device=device), dim=1)
            ids_keep = ids_shuffle[:, :keep_num]
            ids_mask = ids_shuffle[:, keep_num:]

            context = student_encoder(imgs, ids_keep)
            pos_embed = student_encoder.pos_embed.expand(imgs.shape[0], -1, -1)

            mask_pos = torch.gather(pos_embed, 1, ids_mask.unsqueeze(-1).expand(-1, -1, student_encoder.embed_dim))
            pred = predictor(context, mask_pos)

            target_mask = torch.gather(target, 1, ids_mask.unsqueeze(-1).expand(-1, -1, student_encoder.embed_dim))

            loss = criterion(pred, target_mask)
            loss.backward()
            optimizer.step()
            
            # EMA weight update for the target encoder
            update_target_encoder(student_encoder, target_encoder, EMA_MOMENTUM)
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -- Validation Phase --
        student_encoder.eval()
        predictor.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch[0].to(device)
                
                target = target_encoder(imgs)
                
                ids_shuffle = torch.argsort(torch.rand(imgs.shape[0], num_patches, device=device), dim=1)
                ids_keep = ids_shuffle[:, :keep_num]
                ids_mask = ids_shuffle[:, keep_num:]

                context = student_encoder(imgs, ids_keep)
                pos_embed = student_encoder.pos_embed.expand(imgs.shape[0], -1, -1)
                mask_pos = torch.gather(pos_embed, 1, ids_mask.unsqueeze(-1).expand(-1, -1, student_encoder.embed_dim))
                
                pred = predictor(context, mask_pos)
                target_mask = torch.gather(target, 1, ids_mask.unsqueeze(-1).expand(-1, -1, student_encoder.embed_dim))
                
                loss = criterion(pred, target_mask)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the best model based on Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student_encoder.state_dict(), f"{model_dir}/lejepa_encoder_best.pth")
            print(f"  -> Best model saved! (Val Loss: {best_val_loss:.4f})")

    # 6. Extract Embeddings (For the entire dataset to build the full map)
    print("\nLoading best model for embedding extraction...")
    student_encoder.load_state_dict(torch.load(f"{model_dir}/lejepa_encoder_best.pth"))
    student_encoder.eval()
    
    # Use the full metadata (df_meta) before the split to generate the complete map
    full_dataset = TreePatchDataset(df_meta, transform=transform)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_embeddings = []
    all_labels = []
    all_coords = []

    print("Extracting full map embeddings...")
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Extracting"):
            imgs, labels, coords = batch
            imgs = imgs.to(device)
            
            # Pass through the model and average to get a [Batch, 128] vector
            emb = student_encoder(imgs).mean(dim=1).cpu().numpy()
            
            all_embeddings.append(emb)
            all_labels.extend(labels)
            all_coords.append(coords.numpy())

    embeddings = np.concatenate(all_embeddings)
    labels = np.array(all_labels)
    coords = np.concatenate(all_coords)

    assert len(embeddings) == len(labels) == len(coords)

    np.save(f"{embedding_dir}/embeddings.npy", embeddings)
    np.save(f"{label_dir}/labels.npy", labels)
    np.save(f"{embedding_dir}/coords.npy", coords)

    print("\nSaved final embeddings, labels, and coordinates!")
    total_time = (time.time() - start_time) / 60
    print(f"Total execution time: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()