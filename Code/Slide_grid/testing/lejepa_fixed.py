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

# --------------------------
# Hyperparameters & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"

IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 4
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER
HALF_CROP = CROP_SIZE // 2

BATCH_SIZE = 64 
INFERENCE_BATCH_SIZE = 128 
EPOCHS = 15 
SEED = 42
EMA_MOMENTUM = 0.996

NUM_RANDOM_CROPS_PER_TIF = 200   
GRID_STRIDE_PIXELS = 256         

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
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                activation='gelu', batch_first=True
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
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                activation='gelu', batch_first=True
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
# Custom Datasets
# --------------------------
class InMemoryDataset(Dataset):
    """
    OPTIMIZATION: Instead of opening TIF files during training, this dataset 
    reads directly from pre-loaded arrays in RAM. Zero disk I/O bottleneck!
    """
    def __init__(self, images_list, coords_list, transform=None):
        self.images = images_list
        self.coords = coords_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        x, y = self.coords[idx]
        return img, x, y

# --------------------------
# Utility Functions
# --------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_target_encoder(student, target, momentum):
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

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    if not tif_files:
        print("Error: No TIF files found. Check your BASE_DIR.")
        return

    # =========================================================================
    # Phase 1: Build a Pure Self-Supervised Dataset IN MEMORY
    # =========================================================================
    print("\n[Phase 1] Pre-extracting Random Crops into RAM to eliminate Disk I/O...")
    train_images = []
    train_coords = []
    
    for tif_path in tqdm(tif_files, desc="Extracting Training Patches"):
        try:
            with rasterio.open(tif_path) as src:
                width, height = src.width, src.height
                valid_x_range = range(HALF_CROP, width - HALF_CROP)
                valid_y_range = range(HALF_CROP, height - HALF_CROP)
                
                if not valid_x_range or not valid_y_range:
                    continue

                for _ in range(NUM_RANDOM_CROPS_PER_TIF):
                    px = random.choice(valid_x_range)
                    py = random.choice(valid_y_range)
                    
                    # Read only the small window needed
                    window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
                    tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                    tile = np.moveaxis(tile, 0, -1).astype('uint8')
                    
                    x, y = src.xy(py, px) 
                    train_images.append(tile)
                    train_coords.append((x, y))
        except Exception as e:
            print(f"Skipping {tif_path} due to error: {e}")
            continue

    print(f"Loaded {len(train_images)} random patches directly into Memory.")

    # Create the purely in-memory dataset
    train_dataset = InMemoryDataset(train_images, train_coords, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    student_encoder = LeJepaEncoder().to(device)
    predictor = LeJepaPredictor().to(device)
    target_encoder = copy.deepcopy(student_encoder)
    
    for param in target_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(list(student_encoder.parameters()) + list(predictor.parameters()), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    keep_num = int(num_patches * 0.25)

    print("\nStarting Pure Self-Supervised Training...")
    student_encoder.train()
    predictor.train()
    
    for epoch in range(EPOCHS):
        train_loss = 0.0
        # This loop will now fly because it doesn't touch the hard drive
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            imgs = batch[0].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
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
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            update_target_encoder(student_encoder, target_encoder, EMA_MOMENTUM)
            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss / len(train_loader):.4f}")

    torch.save(student_encoder.state_dict(), f"{model_dir}/lejepa_encoder_final.pth")
    print("Model training complete and saved.")

    # Free up RAM used by training images
    del train_images 
    del train_dataset
    del train_loader

    # =========================================================================
    # Phase 2: Extract Dense Grid TIF-by-TIF (Massive Speedup)
    # =========================================================================
    print("\n[Phase 2] Extracting Dense Grid (Processing TIF-by-TIF to avoid I/O bottlenecks)...")
    student_encoder.eval()
    
    all_embeddings = []
    all_coords = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for tif_path in tqdm(tif_files, desc="Processing TIFs"):
            tif_images = []
            tif_coords = []
            try:
                # Open TIF once, extract all grid points, close TIF. 
                with rasterio.open(tif_path) as src:
                    width, height = src.width, src.height
                    for py in range(HALF_CROP, height - HALF_CROP, GRID_STRIDE_PIXELS):
                        for px in range(HALF_CROP, width - HALF_CROP, GRID_STRIDE_PIXELS):
                            window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
                            tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                            tile = np.moveaxis(tile, 0, -1).astype('uint8')
                            x, y = src.xy(py, px)
                            
                            tif_images.append(tile)
                            tif_coords.append((x, y))
                            
                # Batch process this specific TIF's patches through the model
                if len(tif_images) > 0:
                    temp_dataset = InMemoryDataset(tif_images, tif_coords, transform=transform)
                    temp_loader = DataLoader(temp_dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, num_workers=4)
                    
                    for batch in temp_loader:
                        imgs, xs, ys = batch
                        imgs = imgs.to(device)
                        emb = student_encoder(imgs).mean(dim=1).cpu().numpy()
                        
                        all_embeddings.append(emb)
                        coords = np.stack((xs.numpy(), ys.numpy()), axis=1)
                        all_coords.append(coords)
                        
            except Exception as e:
                print(f"Failed to process {tif_path}: {e}")
                continue

    dense_embeddings = np.concatenate(all_embeddings)
    dense_coords = np.concatenate(all_coords)

    # =========================================================================
    # Phase 3: Downstream Task (Label Mapping)
    # =========================================================================
    print("\n[Phase 3] Mapping Labels to Dense Map using shapefile...")
    gdf = gpd.read_file(ANNOTATED_COR)
    
    print("Extracting features at labeled locations to train classifier...")
    spatial_tree = cKDTree(dense_coords)
    
    X_train = []
    y_train = []
    label_col = 'Tree' if 'Tree' in gdf.columns else gdf.columns[0]
    
    for _, row in gdf.iterrows():
        point = [row.geometry.x, row.geometry.y]
        dist, idx = spatial_tree.query(point)
        
        if dist < 50.0:
            X_train.append(dense_embeddings[idx])
            y_train.append(str(row[label_col]))
            
    if not X_train:
        print("ERROR: No coordinates matched within 50.0 distance. Check CRS projection!")
        return

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED) 
    clf.fit(X_train, y_train)
    print(f"Classifier trained on {len(X_train)} labeled points.")

    print("Predicting labels for the entire dense map...")
    dense_labels = clf.predict(dense_embeddings)

    np.save(f"{embedding_dir}/embeddings.npy", dense_embeddings)
    np.save(f"{label_dir}/labels.npy", dense_labels)
    np.save(f"{embedding_dir}/coords.npy", dense_coords)

    print("\nSaved final embeddings, predictions, and coordinates!")
    total_time = (time.time() - start_time) / 60
    print(f"Total execution time: {total_time:.2f} minutes")
    print("=> You can now run your `sliding_grid.py` script!")

if __name__ == "__main__":
    main()