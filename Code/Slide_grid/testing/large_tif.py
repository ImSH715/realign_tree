import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. PATHS & HYPERPARAMETERS
# ---------------------------------------------------------
TIF_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_SHP = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"
LARGE_CENSUS_CSV = "data/tree_label_rdn/Censo_Forestal.csv"
OUTPUT_CSV = "OSINFOR_2023_Realigned_Final.csv"

# IMPORTANT: Path to the trained brain from Phase 1
SAVED_MODEL_PATH = "data/models/encoder.pth" 

LABEL_COL_SHP = "Tree"
LABEL_COL_CSV = "NOMBRE_COMUN" # Change to "NOMBRE_CIENTIFICO" if needed

SEARCH_RADIUS = 30.0 
GRID_STRIDE = 4 

# Image dimensions MUST match the LeJepa training script exactly
IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER  # 896
HALF_CROP = CROP_SIZE // 2              # 448

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
def build_tif_index(tif_dir):
    """Creates a spatial bounding box index for all TIF files."""
    tif_files = glob.glob(os.path.join(tif_dir, "2023-*", "*.tif"))
    index = []
    print("Indexing TIF boundaries...")
    for f in tqdm(tif_files, leave=False):
        try:
            with rasterio.open(f) as src:
                index.append({'path': f, 'bounds': src.bounds})
        except:
            continue
    return index

def find_tif_for_point(x, y, index):
    """Returns the correct TIF path for a given coordinate."""
    for item in index:
        b = item['bounds']
        if b.left <= x <= b.right and b.bottom <= y <= b.top:
            return item['path']
    return None

def extract_embedding(encoder, src, x, y):
    """Extracts a 128D embedding matching the exact preprocessing of the training phase."""
    try:
        py, px = src.index(x, y)
        window = Window(col_off=px - HALF_CROP, row_off=py - HALF_CROP, width=CROP_SIZE, height=CROP_SIZE)
        tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        
        if tile.shape != (3, CROP_SIZE, CROP_SIZE): return None
        
        img_t = torch.from_numpy(tile).float().unsqueeze(0).to(device) / 255.0
        
        # Apply interpolation and normalization to match the trained encoder
        img_t = F.interpolate(img_t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        img_t = (img_t - mean_t) / std_t
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            emb = encoder(img_t).mean(dim=1).cpu().numpy()[0]
        return emb
    except:
        return None

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("\n--- Starting Massive Realignment (Large TIF) ---")
    
    # [Step A] Load Trained Encoder
    encoder = LeJepaEncoder().to(device)
    if os.path.exists(SAVED_MODEL_PATH):
        encoder.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
        print(f"[SUCCESS] Loaded trained model from {SAVED_MODEL_PATH}")
    else:
        print(f"[WARNING] Model {SAVED_MODEL_PATH} not found. Using untrained weights!")
    encoder.eval()

    tif_index = build_tif_index(TIF_DIR)

    # [Step B] Train Classifier on Ground Truth Data
    print("\nTraining Classifier on Ground Truth Data...")
    gdf = gpd.read_file(ANNOTATED_SHP)
    gdf[LABEL_COL_SHP] = gdf[LABEL_COL_SHP].astype(str).str.strip().str.upper() # Clean labels
    
    X_train, y_train = [], []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf)):
        tif_path = find_tif_for_point(row.geometry.x, row.geometry.y, tif_index)
        if tif_path:
            with rasterio.open(tif_path) as src:
                emb = extract_embedding(encoder, src, row.geometry.x, row.geometry.y)
                if emb is not None:
                    X_train.append(emb)
                    y_train.append(row[LABEL_COL_SHP])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    known_species = set(y_train)
    print(f"Classifier trained. Known Species count: {len(known_species)}")

    # [Step C] Process Large Census CSV
    print("\nProcessing Massive Census CSV...")
    df = pd.read_csv(LARGE_CENSUS_CSV)
    
    # Clean CSV labels to perfectly match Shapefile labels
    df[LABEL_COL_CSV] = df[LABEL_COL_CSV].astype(str).str.strip().str.upper()
    
    df["ORIGINAL_X"] = df["COORDENADA_ESTE"]
    df["ORIGINAL_Y"] = df["COORDENADA_NORTE"]
    df["REALIGNED_X"] = df["ORIGINAL_X"]
    df["REALIGNED_Y"] = df["ORIGINAL_Y"]
    df["SHIFT_DISTANCE"] = 0.0
    df["STATUS"] = "PENDING"

    # NOTE: Remove '[:100]' here if you want to process all 17,000 rows
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        orig_x, orig_y = row["ORIGINAL_X"], row["ORIGINAL_Y"]
        target_species = row[LABEL_COL_CSV]
        
        # Early skip if species is unknown to the classifier
        if target_species not in known_species:
            df.at[idx, "STATUS"] = "KEEP_ORIGINAL_UNKNOWN_SPECIES"
            continue

        tif_path = find_tif_for_point(orig_x, orig_y, tif_index)
        if not tif_path:
            df.at[idx, "STATUS"] = "KEEP_ORIGINAL_OUT_OF_BOUNDS"
            continue

        search_coords = []
        search_embeds = []
        
        with rasterio.open(tif_path) as src:
            for oy in np.arange(-SEARCH_RADIUS, SEARCH_RADIUS, GRID_STRIDE):
                for ox in np.arange(-SEARCH_RADIUS, SEARCH_RADIUS, GRID_STRIDE):
                    cx, cy = orig_x + ox, orig_y + oy
                    emb = extract_embedding(encoder, src, cx, cy)
                    if emb is not None:
                        search_coords.append([cx, cy])
                        search_embeds.append(emb)

        if not search_embeds:
            df.at[idx, "STATUS"] = "KEEP_ORIGINAL_NO_VALID_PATCHES"
            continue

        # Predict & Cluster
        preds = clf.predict(search_embeds)
        search_coords = np.array(search_coords)
        mask = (preds == target_species)
        target_points = search_coords[mask]

        if len(target_points) > 0:
            # DBSCAN: eps=4.0, min_samples=2 optimized for tree crowns
            db = DBSCAN(eps=4.0, min_samples=2).fit(target_points)
            labels = db.labels_
            
            if len(set(labels)) - (1 if -1 in labels else 0) > 0:
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                best_cluster = unique_labels[np.argmax(counts)]
                cluster_pts = target_points[labels == best_cluster]
                
                new_x, new_y = cluster_pts.mean(axis=0)
                shift = np.sqrt((orig_x - new_x)**2 + (orig_y - new_y)**2)
                
                df.at[idx, "REALIGNED_X"] = new_x
                df.at[idx, "REALIGNED_Y"] = new_y
                df.at[idx, "SHIFT_DISTANCE"] = shift
                df.at[idx, "STATUS"] = "SUCCESSFULLY_MOVED"
            else:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_NO_CLUSTER"
        else:
            df.at[idx, "STATUS"] = "KEEP_ORIGINAL_SPECIES_NOT_FOUND_IN_RADIUS"

        # Checkpoint every 500 rows to prevent data loss
        if idx % 500 == 0:
            df.to_csv(OUTPUT_CSV, index=False)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRealignment saved successfully to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()