import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import glob
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. PATHS & HYPERPARAMETERS
# ---------------------------------------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_SHP = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"

# IMPORTANT: Path to the trained brain from Phase 1
SAVED_MODEL_PATH = "data/models/encoder.pth" 

LABEL_COL = "Tree"
JITTER_MAX = 25.0       # Maximum synthetic error to inject (meters)
SEARCH_RADIUS = 30.0    # Radius to search around the jittered point
GRID_STRIDE = 4         # Grid density for searching
IMG_SIZE = 448
PATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. MODEL DEFINITION
# ---------------------------------------------------------
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks: 
            x = block(x)
        return self.norm(x)

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_embedding(encoder, tif_path, x, y):
    """Extracts a 128D embedding from a specific coordinate."""
    try:
        with rasterio.open(tif_path) as src:
            py, px = src.index(x, y)
            half = IMG_SIZE // 2
            window = Window(col_off=px - half, row_off=py - half, width=IMG_SIZE, height=IMG_SIZE)
            tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            
            if tile.shape != (3, IMG_SIZE, IMG_SIZE): return None
            
            img_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                emb = encoder(img_tensor).mean(dim=1).cpu().numpy()[0]
            return emb
    except:
        return None

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("\n--- Starting Verification (Small TIF) ---")
    
    # [Step A] Load the Trained Encoder
    encoder = LeJepaEncoder().to(device)
    if os.path.exists(SAVED_MODEL_PATH):
        encoder.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
        print(f"[SUCCESS] Loaded trained model from {SAVED_MODEL_PATH}")
    else:
        print(f"[WARNING] Model {SAVED_MODEL_PATH} not found. Using untrained weights!")
    encoder.eval()

    # [Step B] Load Ground Truth and Split Data
    gdf = gpd.read_file(ANNOTATED_SHP)
    # Clean string labels (strip spaces, make uppercase)
    gdf[LABEL_COL] = gdf[LABEL_COL].astype(str).str.strip().str.upper()
    
    train_df, test_df = train_test_split(gdf, test_size=0.2, random_state=42)
    print(f"Train Points: {len(train_df)} | Test Points: {len(test_df)}")

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    default_tif = tif_files[0] if tif_files else None

    # [Step C] Train Random Forest on Clean Coordinates
    print("\nExtracting features for Random Forest...")
    X_train, y_train = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        emb = extract_embedding(encoder, default_tif, row.geometry.x, row.geometry.y)
        if emb is not None:
            X_train.append(emb)
            y_train.append(row[LABEL_COL])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print(f"Random Forest trained on {len(X_train)} points.")

    # [Step D] Test on Jittered (Noisy) Coordinates
    print("\nRunning realignment test on Jittered points...")
    results = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        true_x, true_y = row.geometry.x, row.geometry.y
        target_species = row[LABEL_COL]
        
        # 1. Inject Noise
        jitter_x = true_x + random.uniform(-JITTER_MAX, JITTER_MAX)
        jitter_y = true_y + random.uniform(-JITTER_MAX, JITTER_MAX)
        initial_error = np.sqrt((true_x - jitter_x)**2 + (true_y - jitter_y)**2)
        
        # 2. Search around jittered coordinate
        search_coords = []
        search_embeds = []
        
        for oy in np.arange(-SEARCH_RADIUS, SEARCH_RADIUS, GRID_STRIDE):
            for ox in np.arange(-SEARCH_RADIUS, SEARCH_RADIUS, GRID_STRIDE):
                cx, cy = jitter_x + ox, jitter_y + oy
                emb = extract_embedding(encoder, default_tif, cx, cy)
                if emb is not None:
                    search_coords.append([cx, cy])
                    search_embeds.append(emb)
                    
        if not search_embeds: continue
        
        # 3. Predict & Cluster
        preds = clf.predict(search_embeds)
        search_coords = np.array(search_coords)
        mask = (preds == target_species)
        target_points = search_coords[mask]
        
        if len(target_points) > 0:
            # DBSCAN Tweaks: eps increased to 4.0, min_samples lowered to 2 to capture sparse canopy pixels
            db = DBSCAN(eps=4.0, min_samples=2).fit(target_points)
            labels = db.labels_
            
            if len(set(labels)) - (1 if -1 in labels else 0) > 0:
                # Find largest cluster
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                best_cluster = unique_labels[np.argmax(counts)]
                cluster_pts = target_points[labels == best_cluster]
                
                # Calculate new centroid
                realigned_x, realigned_y = cluster_pts.mean(axis=0)
                final_error = np.sqrt((true_x - realigned_x)**2 + (true_y - realigned_y)**2)
                improvement = initial_error - final_error
            else:
                final_error = initial_error
                improvement = 0.0
        else:
            final_error = initial_error
            improvement = 0.0
            
        results.append({
            "Species": target_species,
            "Initial_Error": initial_error,
            "Final_Error": final_error,
            "Improvement": improvement
        })

    # [Step E] Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv("verification_metrics.csv", index=False)
    print(f"\nVerification Complete. Mean Improvement: {res_df['Improvement'].mean():.2f}m")
    print(res_df.groupby("Species")["Improvement"].mean())

if __name__ == "__main__":
    main()