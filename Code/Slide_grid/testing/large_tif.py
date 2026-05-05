import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PATHS & HYPERPARAMETERS
# ---------------------------------------------------------
TIF_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_SHP = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"
LARGE_CENSUS_CSV = "data/tree_label_rdn/Censo_Forestal.csv"
OUTPUT_CSV = "OSINFOR_2023_Realigned_Final.csv"

SAVED_MODEL_PATH = "data/models/encoder.pth"

LABEL_COL_SHP = "Tree"
LABEL_COL_CSV = "NOMBRE_COMUN"   # 필요 시 "NOMBRE_CIENTIFICO"로 변경

SEARCH_RADIUS = 30.0
GRID_STRIDE = 4.0

IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER   # 896
HALF_CROP = CROP_SIZE // 2               # 448

RF_TREES = 100
BATCH_SIZE_INFER = 64
CHECKPOINT_EVERY = 500
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (device.type == "cuda")

# ---------------------------------------------------------
# 2. MODEL DEFINITION
# ---------------------------------------------------------
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3,
                 embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation="gelu",
                batch_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# ---------------------------------------------------------
# 3. UTILS
# ---------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_tif_index(tif_dir):
    """Creates a spatial bounding box index for all TIF files."""
    tif_files = glob.glob(os.path.join(tif_dir, "2023-*", "*.tif"))
    index = []
    print("Indexing TIF boundaries...")
    for f in tqdm(tif_files, leave=False):
        try:
            with rasterio.open(f) as src:
                index.append({
                    "path": f,
                    "bounds": src.bounds
                })
        except Exception:
            continue
    print(f"Indexed TIF count: {len(index)}")
    return index

def find_tif_for_point(x, y, index):
    """Returns the correct TIF path for a given coordinate."""
    for item in index:
        b = item["bounds"]
        if b.left <= x <= b.right and b.bottom <= y <= b.top:
            return item["path"]
    return None

def build_search_grid(center_x, center_y, radius, stride):
    coords = []
    for oy in np.arange(-radius, radius, stride):
        for ox in np.arange(-radius, radius, stride):
            coords.append((center_x + ox, center_y + oy))
    return coords

def extract_embedding_batch(encoder, src, coords, batch_size=BATCH_SIZE_INFER):
    """
    Extract embeddings for many (x, y) points in one batched forward pass.
    Returns:
        valid_coords_np: (N,2)
        embeds_np:       (N,D)
    """
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    tiles = []
    valid_coords = []

    for x, y in coords:
        try:
            py, px = src.index(x, y)
            window = Window(
                col_off=px - HALF_CROP,
                row_off=py - HALF_CROP,
                width=CROP_SIZE,
                height=CROP_SIZE
            )
            tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)

            if tile.shape != (3, CROP_SIZE, CROP_SIZE):
                continue

            tiles.append(tile)
            valid_coords.append((x, y))

        except Exception:
            continue

    if len(tiles) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    imgs = torch.from_numpy(np.stack(tiles)).float().to(device) / 255.0
    imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    imgs = (imgs - mean_t) / std_t

    emb_list = []

    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch_size):
            batch = imgs[i:i + batch_size]

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    emb = encoder(batch).mean(dim=1)
            else:
                emb = encoder(batch).mean(dim=1)

            emb_list.append(emb.cpu().numpy())

    embeds_np = np.concatenate(emb_list, axis=0).astype(np.float32)
    valid_coords_np = np.array(valid_coords, dtype=np.float32)

    return valid_coords_np, embeds_np

def extract_single_embedding(encoder, src, x, y):
    coords_np, embeds_np = extract_embedding_batch(encoder, src, [(x, y)], batch_size=1)
    if len(embeds_np) == 0:
        return None
    return embeds_np[0]

def initialize_or_resume_csv():
    """
    If OUTPUT_CSV exists, resume from it.
    Otherwise initialize from LARGE_CENSUS_CSV.
    """
    if os.path.exists(OUTPUT_CSV):
        print(f"[RESUME] Loading existing output: {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)

        required_cols = [
            "ORIGINAL_X", "ORIGINAL_Y",
            "REALIGNED_X", "REALIGNED_Y",
            "SHIFT_DISTANCE", "STATUS"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Existing OUTPUT_CSV missing columns: {missing}")

        if LABEL_COL_CSV in df.columns:
            df[LABEL_COL_CSV] = df[LABEL_COL_CSV].astype(str).str.strip().str.upper()

        return df

    print(f"[NEW RUN] Loading source CSV: {LARGE_CENSUS_CSV}")
    df = pd.read_csv(LARGE_CENSUS_CSV)

    df[LABEL_COL_CSV] = df[LABEL_COL_CSV].astype(str).str.strip().str.upper()

    df["ORIGINAL_X"] = df["COORDENADA_ESTE"]
    df["ORIGINAL_Y"] = df["COORDENADA_NORTE"]
    df["REALIGNED_X"] = df["ORIGINAL_X"]
    df["REALIGNED_Y"] = df["ORIGINAL_Y"]
    df["SHIFT_DISTANCE"] = 0.0
    df["STATUS"] = "PENDING"

    return df

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    set_seed(SEED)
    print("\n--- Starting Massive Realignment (Large TIF) ---")
    print(f"Device: {device}")

    # [Step A] Load Trained Encoder
    encoder = LeJepaEncoder().to(device)
    if os.path.exists(SAVED_MODEL_PATH):
        encoder.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
        print(f"[SUCCESS] Loaded trained model from {SAVED_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found: {SAVED_MODEL_PATH}")
    encoder.eval()

    tif_index = build_tif_index(TIF_DIR)
    if len(tif_index) == 0:
        raise RuntimeError("No valid TIFs found in TIF_DIR.")

    # [Step B] Train Classifier on Ground Truth Data
    print("\nTraining Classifier on Ground Truth Data...")
    gdf = gpd.read_file(ANNOTATED_SHP)
    gdf[LABEL_COL_SHP] = gdf[LABEL_COL_SHP].astype(str).str.strip().str.upper()

    X_train, y_train = [], []

    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extracting GT embeddings"):
        x = row.geometry.x
        y = row.geometry.y
        tif_path = find_tif_for_point(x, y, tif_index)

        if tif_path is None:
            continue

        try:
            with rasterio.open(tif_path) as src:
                emb = extract_single_embedding(encoder, src, x, y)
                if emb is not None:
                    X_train.append(emb)
                    y_train.append(row[LABEL_COL_SHP])
        except Exception:
            continue

    if len(X_train) == 0:
        raise RuntimeError("No training embeddings extracted from ground truth points.")

    clf = RandomForestClassifier(
        n_estimators=RF_TREES,
        random_state=SEED,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    known_species = set(y_train)
    print(f"Classifier trained on {len(X_train)} points.")
    print(f"Known Species count: {len(known_species)}")

    # [Step C] Load or Resume Census CSV
    print("\nProcessing Massive Census CSV...")
    df = initialize_or_resume_csv()

    pending_idx = df.index[df["STATUS"] == "PENDING"].tolist()
    print(f"Pending rows to process: {len(pending_idx)} / {len(df)}")

    processed_since_save = 0

    for idx in tqdm(pending_idx, total=len(pending_idx), desc="Realigning rows"):
        row = df.loc[idx]

        try:
            orig_x = float(row["ORIGINAL_X"])
            orig_y = float(row["ORIGINAL_Y"])
            target_species = str(row[LABEL_COL_CSV]).strip().upper()

            if target_species not in known_species:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_UNKNOWN_SPECIES"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            tif_path = find_tif_for_point(orig_x, orig_y, tif_index)
            if tif_path is None:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_OUT_OF_BOUNDS"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            candidate_coords = build_search_grid(
                center_x=orig_x,
                center_y=orig_y,
                radius=SEARCH_RADIUS,
                stride=GRID_STRIDE
            )

            with rasterio.open(tif_path) as src:
                search_coords, search_embeds = extract_embedding_batch(
                    encoder, src, candidate_coords, batch_size=BATCH_SIZE_INFER
                )

            if len(search_embeds) == 0:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_NO_VALID_PATCHES"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            preds = clf.predict(search_embeds)

            mask = (preds == target_species)
            target_points = search_coords[mask]

            if len(target_points) == 0:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_SPECIES_NOT_FOUND_IN_RADIUS"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            db = DBSCAN(eps=4.0, min_samples=2).fit(target_points)
            labels = db.labels_

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if num_clusters <= 0:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_NO_CLUSTER"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            best_cluster = unique_labels[np.argmax(counts)]
            cluster_pts = target_points[labels == best_cluster]

            new_x, new_y = cluster_pts.mean(axis=0)
            shift = float(np.sqrt((orig_x - new_x) ** 2 + (orig_y - new_y) ** 2))

            df.at[idx, "REALIGNED_X"] = float(new_x)
            df.at[idx, "REALIGNED_Y"] = float(new_y)
            df.at[idx, "SHIFT_DISTANCE"] = shift
            df.at[idx, "STATUS"] = "SUCCESSFULLY_MOVED"

        except Exception as e:
            df.at[idx, "STATUS"] = f"ERROR_{type(e).__name__}"

        processed_since_save += 1

        if processed_since_save >= CHECKPOINT_EVERY:
            df.to_csv(OUTPUT_CSV, index=False)
            processed_since_save = 0

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nRealignment saved successfully to {OUTPUT_CSV}")
    print("\nSTATUS summary:")
    print(df["STATUS"].value_counts(dropna=False))

    moved_mask = df["STATUS"] == "SUCCESSFULLY_MOVED"
    if moved_mask.any():
        print(f"\nMean shift distance (moved only): {df.loc[moved_mask, 'SHIFT_DISTANCE'].mean():.3f}")

if __name__ == "__main__":
    main()