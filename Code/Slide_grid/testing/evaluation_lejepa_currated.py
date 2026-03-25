import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- INPUT PATHS ---
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_SHP = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"
LABEL_COL = "Tree"  # Your species column

# --- SETTINGS ---
JITTER_MAX = 25.0    # Random noise radius (meters)
SEARCH_RADIUS = 30.0 # Search window for realignment
IMG_SIZE = 448
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. FILE DISCOVERY & SPATIAL INDEXING
# ==========================================
print("Indexing TIF files...")
# This finds all TIFs in 2023-01, 2023-02, etc.
tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))

def get_tif_for_point(point, tif_list):
    """Finds which TIF file contains the coordinate."""
    for tif in tif_list:
        with rasterio.open(tif) as src:
            bounds = src.bounds
            if (bounds.left <= point.x <= bounds.right and 
                bounds.bottom <= point.y <= bounds.top):
                return tif
    return None

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class LeJepaEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.randn(1, (448//16)**2, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True) for _ in range(4)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks: x = block(x)
        return self.norm(x)

# Initialize and assume weights are loaded from your Phase 1 training
encoder = LeJepaEncoder().to(device)
encoder.eval()

# ==========================================
# 3. STRATIFIED JITTER VALIDATION
# ==========================================
def run_verification():
    # Load and Split Labels
    gdf = gpd.read_file(ANNOTATED_SHP)
    print(f"Loaded {len(gdf)} labels with {gdf[LABEL_COL].nunique()} species.")

    # Stratified Split (80% Train, 20% Jitter-Test)
    train_gdf, test_gdf = train_test_split(
        gdf, test_size=0.20, stratify=gdf[LABEL_COL], random_state=42
    )

    # A. TRAIN CLASSIFIER
    X_train, y_train = [], []
    print("\n[Step 1] Extracting Clean Training Features...")
    for _, row in tqdm(train_gdf.iterrows(), total=len(train_gdf)):
        tif_path = get_tif_for_point(row.geometry, tif_files)
        if tif_path:
            with rasterio.open(tif_path) as src:
                py, px = src.index(row.geometry.x, row.geometry.y)
                window = rasterio.windows.Window(px-224, py-224, 448, 448)
                img = src.read([1,2,3], window=window, boundless=True, fill_value=0)
                
                img_t = torch.from_numpy(img).float().unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    tokens = encoder(img_t).view(1, 28, 28, 128)
                    # Average the center of the crown
                    feat = tokens[:, 12:16, 12:16, :].mean(dim=(1,2)).cpu().numpy().squeeze()
                    X_train.append(feat)
                    y_train.append(row[LABEL_COL])

    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

    # B. GENERATE JITTERED TEST SET
    print("\n[Step 2] Applying 25m Jitter to Test Set...")
    results = []
    for _, row in tqdm(test_gdf.iterrows(), total=len(test_gdf)):
        true_x, true_y = row.geometry.x, row.geometry.y
        # Add random offset
        off_x, off_y = (np.random.rand(2) - 0.5) * 2 * JITTER_MAX
        noisy_x, noisy_y = true_x + off_x, true_y + off_y
        
        # FIND REALIGNMENT
        tif_path = get_tif_for_point(row.geometry, tif_files)
        if not tif_path: continue
        
        with rasterio.open(tif_path) as src:
            # Create a local dense grid around the noisy point
            py, px = src.index(noisy_x, noisy_y)
            dense_coords, dense_preds = [], []
            
            # Scan 60m area around noisy point
            for oy in range(-30, 30, 4): 
                for ox in range(-30, 30, 4):
                    win = rasterio.windows.Window(px+ox-224, py+oy-224, 448, 448)
                    img = src.read([1,2,3], window=win, boundless=True, fill_value=0)
                    img_t = torch.from_numpy(img).float().unsqueeze(0).to(device) / 255.0
                    with torch.no_grad():
                        tokens = encoder(img_t).mean(dim=1).cpu().numpy()
                        dense_preds.append(clf.predict(tokens)[0])
                        mx, my = src.xy(py+oy, px+ox)
                        dense_coords.append([mx, my])

            # Apply DBSCAN to find the cluster of the target species
            coords_arr = np.array(dense_coords)
            preds_arr = np.array(dense_preds)
            mask = (preds_arr == row[LABEL_COL])
            
            if mask.sum() >= 3:
                db = DBSCAN(eps=3.0, min_samples=3).fit(coords_arr[mask])
                if len(set(db.labels_)) > (1 if -1 in db.labels_ else 0):
                    # Pick cluster closest to noisy start
                    u, counts = np.unique(db.labels_[db.labels_!=-1], return_counts=True)
                    best_c = u[np.argmax(counts)]
                    realigned_pos = coords_arr[mask][db.labels_ == best_c].mean(axis=0)
                    
                    # Score
                    dist_init = np.linalg.norm([noisy_x - true_x, noisy_y - true_y])
                    dist_final = np.linalg.norm([realigned_pos[0] - true_x, realigned_pos[1] - true_y])
                    
                    results.append({
                        'Species': row[LABEL_COL],
                        'Initial_Error': dist_init,
                        'Final_Error': dist_final,
                        'Improvement': dist_init - dist_final
                    })

    # C. REPORT
    res_df = pd.DataFrame(results)
    print("\n" + "="*30)
    print("VERIFICATION COMPLETE")
    print(f"Avg Jitter Distance: {res_df['Initial_Error'].mean():.2f}m")
    print(f"Avg Recovered Distance: {res_df['Final_Error'].mean():.2f}m")
    print(f"Mean Precision Gain: {res_df['Improvement'].mean():.2f}m")
    print("="*30)
    
    res_df.to_csv("verification_metrics.csv", index=False)

if __name__ == "__main__":
    run_verification()