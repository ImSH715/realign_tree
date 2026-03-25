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

BATCH_SIZE = 16
EPOCHS = 30
SEED = 42
EMA_MOMENTUM = 0.996

NUM_RANDOM_CROPS_PER_TIF = 1000 
GRID_STRIDE_PIXELS = 128 

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

class UnsupervisedPatchDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            with rasterio.open(row['tif_path']) as src:
                window = rasterio.windows.Window(
                    row['px'] - HALF_CROP, row['py'] - HALF_CROP, CROP_SIZE, CROP_SIZE
                )
                tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                tile = np.moveaxis(tile, 0, -1)
            img = Image.fromarray(tile.astype('uint8'))
        except Exception as e:
            # Fallback for broken files during DataLoader iteration
            img = Image.fromarray(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype='uint8'))
        
        if self.transform:
            img = self.transform(img)
        return img, row['x'], row['y']

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

    model_dir, embedding_dir, label_dir = "data/models", "data/embeddings", "data/label"
    for d in [model_dir, embedding_dir, label_dir]: os.makedirs(d, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))

    # =========================================================================
    # Phase 1: Self-Supervised Training (With Error Handling)
    # =========================================================================
    print("\n[Phase 1] Sampling TIFs for Training...")
    train_metadata = []
    for tif_path in tqdm(tif_files, desc="Processing TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                w, h = src.width, src.height
                if w < CROP_SIZE or h < CROP_SIZE: continue
                for _ in range(NUM_RANDOM_CROPS_PER_TIF):
                    px, py = random.randint(HALF_CROP, w-HALF_CROP), random.randint(HALF_CROP, h-HALF_CROP)
                    x, y = src.xy(py, px)
                    train_metadata.append({'tif_path': tif_path, 'px': px, 'py': py, 'x': x, 'y': y})
        except Exception as e:
            print(f"Skipping broken file: {os.path.basename(tif_path)}")
            continue

    student_encoder = LeJepaEncoder().to(device)
    predictor = LeJepaPredictor().to(device)
    target_encoder = copy.deepcopy(student_encoder)
    for p in target_encoder.parameters(): p.requires_grad = False
    
    optimizer = torch.optim.AdamW(list(student_encoder.parameters()) + list(predictor.parameters()), lr=1e-4)
    criterion = nn.MSELoss()
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    keep_num = int(num_patches * 0.25)

    train_loader = DataLoader(UnsupervisedPatchDataset(pd.DataFrame(train_metadata), transform=transform), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Starting Training...")
    for epoch in range(EPOCHS):
        student_encoder.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs = batch[0].to(device)
            optimizer.zero_grad()
            with torch.no_grad(): target = target_encoder(imgs)
            
            ids_shuffle = torch.argsort(torch.rand(imgs.shape[0], num_patches, device=device), dim=1)
            ids_keep, ids_mask = ids_shuffle[:, :keep_num], ids_shuffle[:, keep_num:]
            
            context = student_encoder(imgs, ids_keep)
            pos_embed = student_encoder.pos_embed.expand(imgs.shape[0], -1, -1)
            mask_pos = torch.gather(pos_embed, 1, ids_mask.unsqueeze(-1).expand(-1, -1, 128))
            pred = predictor(context, mask_pos)
            target_mask = torch.gather(target, 1, ids_mask.unsqueeze(-1).expand(-1, -1, 128))
            
            loss = criterion(pred, target_mask)
            loss.backward()
            optimizer.step()
            update_target_encoder(student_encoder, target_encoder, EMA_MOMENTUM)
            total_loss += loss.item()
        print(f"Loss: {total_loss/len(train_loader):.4f}")

    # =========================================================================
    # Phase 2: Extract Spatial Grid (With Error Handling)
    # =========================================================================
    print("\n[Phase 2] Generating Dense Spatial Grid...")
    grid_metadata = []
    for tif_path in tqdm(tif_files):
        try:
            with rasterio.open(tif_path) as src:
                for py in range(HALF_CROP, src.height - HALF_CROP, GRID_STRIDE_PIXELS):
                    for px in range(HALF_CROP, src.width - HALF_CROP, GRID_STRIDE_PIXELS):
                        x, y = src.xy(py, px)
                        grid_metadata.append({'tif_path': tif_path, 'px': px, 'py': py, 'x': x, 'y': y})
        except Exception: continue

    grid_loader = DataLoader(UnsupervisedPatchDataset(pd.DataFrame(grid_metadata), transform=transform), batch_size=BATCH_SIZE)
    student_encoder.eval()
    all_spatial_embeddings, all_coords = [], []

    with torch.no_grad():
        for batch in tqdm(grid_loader, desc="Inference"):
            imgs, xs, ys = batch
            tokens = student_encoder(imgs.to(device))
            spatial_emb = tokens.view(-1, 28, 28, 128)
            all_spatial_embeddings.append(spatial_emb.cpu().numpy())
            all_coords.append(np.stack((xs.numpy(), ys.numpy()), axis=1))

    dense_spatial = np.concatenate(all_spatial_embeddings, axis=0)
    dense_coords = np.concatenate(all_coords, axis=0)

    # =========================================================================
    # Phase 3: Classifier (Centroid Sampling)
    # =========================================================================
    print("\n[Phase 3] Training Classifier...")
    gdf = gpd.read_file(ANNOTATED_COR)
    tree = cKDTree(dense_coords)
    X_train, y_train = [], []
    label_col = 'Tree' if 'Tree' in gdf.columns else gdf.columns[0]

    for _, row in tqdm(gdf.iterrows(), total=len(gdf)):
        dist, idx = tree.query([row.geometry.x, row.geometry.y])
        if dist < 50.0:
            # Core sampling: center of the 28x28 grid
            X_train.append(dense_spatial[idx, 12:16, 12:16, :].mean(axis=(0,1)))
            y_train.append(str(row[label_col]))

    clf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_train, y_train)
    global_avg = dense_spatial.mean(axis=(1,2))
    dense_labels = clf.predict(global_avg)

    np.save(f"{embedding_dir}/embeddings.npy", global_avg)
    np.save(f"{label_dir}/labels.npy", dense_labels)
    np.save(f"{embedding_dir}/coords.npy", dense_coords)

    print(f"\nDone! Total time: {(time.time() - start_time) / 60:.2f} min")

if __name__ == "__main__":
    main()