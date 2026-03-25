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
from tqdm import tqdm

# --- INPUT PATHS ---
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_SHP = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"
LABEL_COL = "Tree" 

# --- SETTINGS ---
JITTER_MAX = 25.0    
SEARCH_RADIUS = 30.0 
IMG_SIZE = 448
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. FILE DISCOVERY & ERROR-RESISTANT INDEXING
# ==========================================
print("Indexing TIF files and checking for corruption...")
all_tif_paths = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
valid_tif_index = []

for path in tqdm(all_tif_paths, desc="Validating TIFs"):
    try:
        with rasterio.open(path) as src:
            # Store the path and bounds so we don't have to re-open to check coordinates
            valid_tif_index.append({
                'path': path,
                'bounds': src.bounds
            })
    except Exception as e:
        print(f"\n[SKIP] Skipping broken file {os.path.basename(path)}: {e}")

def get_tif_for_point(point, tif_index):
    """Efficiently finds which TIF contains the point using pre-indexed bounds."""
    for item in tif_index:
        b = item['bounds']
        if (b.left <= point.x <= b.right and b.bottom <= point.y <= b.top):
            return item['path']
    return None

# ==========================================
# 2. MODEL DEFINITION (Ensure weights are loaded)
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

encoder = LeJepaEncoder().to(device)
encoder.eval()

# ==========================================
# 3. STRATIFIED JITTER VALIDATION
# ==========================================
def run_verification():
    gdf = gpd.read_file(ANNOTATED_SHP)
    print(f"Loaded {len(gdf)} labels with {gdf[LABEL_COL].nunique()} species.")

    train_gdf, test_gdf = train_test_split(
        gdf, test_size=0.20, stratify=gdf[LABEL_COL], random_state=42
    )

    # A. TRAIN CLASSIFIER
    X_train, y_train = [], []
    print("\n[Step 1] Extracting Clean Training Features...")
    for _, row in tqdm(train_gdf.iterrows(), total=len(train_gdf)):
        tif_path = get_tif_for_point(row.geometry, valid_tif_index)
        if tif_path:
            try:
                with rasterio.open(tif_path) as src:
                    py, px = src.index(row.geometry.x, row.geometry.y)
                    window = rasterio.windows.Window(px-224, py-224, 448, 448)
                    img = src.read([1,2,3], window=window, boundless=True, fill_value=0)
                    
                    img_t = torch.from_numpy(img).float().unsqueeze(0).to(device) / 255.0
                    with torch.no_grad():
                        tokens = encoder(img_t).view(1, 28, 28, 128)
                        feat = tokens[:, 12:16, 12:16, :].mean(dim=(1,2)).cpu().numpy().squeeze()
                        X_train.append(feat)
                        y_train.append(row[LABEL_COL])
            except Exception:
                continue # Skip if file fails during read

    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

    # B. GENERATE JITTERED TEST SET & RECOVER
    print("\n[Step 2] Applying 25m Jitter and Recovering...")
    results = []
    for _, row in tqdm(test_gdf.iterrows(), total=len(test_gdf)):
        true_x, true_y = row.geometry.x, row.geometry.y
        off_x, off_y = (np.random.rand(2) - 0.5) * 2 * JITTER_MAX
        noisy_x, noisy_y = true_x + off_x, true_y + off_y
        
        tif_path = get_tif_for_point(row.geometry, valid_tif_index)
        if not tif_path: continue
        
        try:
            with rasterio.open(tif_path) as src:
                py, px = src.index(noisy_x, noisy_y)
                dense_coords, dense_preds = [], []
                
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

                coords_arr = np.array(dense_coords)
                preds_arr = np.array(dense_preds)
                mask = (preds_arr == row[LABEL_COL])
                
                if mask.sum() >= 3:
                    db = DBSCAN(eps=3.0, min_samples=3).fit(coords_arr[mask])
                    if len(set(db.labels_)) > (1 if -1 in db.labels_ else 0):
                        u, counts = np.unique(db.labels_[db.labels_!=-1], return_counts=True)
                        best_c = u[np.argmax(counts)]
                        realigned_pos = coords_arr[mask][db.labels_ == best_c].mean(axis=0)
                        
                        dist_init = np.linalg.norm([noisy_x - true_x, noisy_y - true_y])
                        dist_final = np.linalg.norm([realigned_pos[0] - true_x, realigned_pos[1] - true_y])
                        
                        results.append({
                            'Species': row[LABEL_COL],
                            'Initial_Error': dist_init,
                            'Final_Error': dist_final,
                            'Improvement': dist_init - dist_final
                        })
        except Exception:
            continue

    # C. REPORT
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print("\n" + "="*30)
        print("VERIFICATION COMPLETE")
        print(f"Avg Jitter Distance: {res_df['Initial_Error'].mean():.2f}m")
        print(f"Avg Recovered Distance: {res_df['Final_Error'].mean():.2f}m")
        print(f"Mean Precision Gain: {res_df['Improvement'].mean():.2f}m")
        print("="*30)
        res_df.to_csv("verification_metrics.csv", index=False)
    else:
        print("No results were successfully processed.")

if __name__ == "__main__":
    run_verification()