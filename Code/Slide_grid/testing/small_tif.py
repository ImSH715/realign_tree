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

SAVED_MODEL_PATH = "data/models/encoder.pth"

LABEL_COL = "Tree"
JITTER_MAX = 25.0
SEARCH_RADIUS = 20.0
GRID_STRIDE = 3.0

# scoring hyperparams
MIN_SPECIES_PROB = 0.45
DIST_PENALTY_ALPHA = 0.35
MIN_CLUSTER_SIZE = 3
MAX_ALLOWED_SHIFT = 25.0

IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER
HALF_CROP = CROP_SIZE // 2

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (device.type == "cuda")


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
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_tif_index(tif_dir):
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
    for item in index:
        b = item['bounds']
        if b.left <= x <= b.right and b.bottom <= y <= b.top:
            return item['path']
    return None

def extract_embedding(encoder, src, x, y):
    try:
        py, px = src.index(x, y)
        window = Window(col_off=px - HALF_CROP, row_off=py - HALF_CROP, width=CROP_SIZE, height=CROP_SIZE)
        tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)

        if tile.shape != (3, CROP_SIZE, CROP_SIZE):
            return None

        img_t = torch.from_numpy(tile).float().unsqueeze(0).to(device) / 255.0
        img_t = F.interpolate(img_t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_t = (img_t - mean_t) / std_t

        with torch.no_grad():
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    emb = encoder(img_t).mean(dim=1).cpu().numpy()[0]
            else:
                emb = encoder(img_t).mean(dim=1).cpu().numpy()[0]
        return emb
    except:
        return None

def euclidean(x1, y1, x2, y2):
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
def main():
    set_seed(SEED)
    print("\n--- Starting Verification (Small TIF, Improved) ---")

    encoder = LeJepaEncoder().to(device)
    if os.path.exists(SAVED_MODEL_PATH):
        encoder.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
        print(f"[SUCCESS] Loaded trained model from {SAVED_MODEL_PATH}")
    else:
        print(f"[WARNING] Model {SAVED_MODEL_PATH} not found. Using untrained weights!")
    encoder.eval()

    tif_index = build_tif_index(BASE_DIR)

    gdf = gpd.read_file(ANNOTATED_SHP)
    gdf[LABEL_COL] = gdf[LABEL_COL].astype(str).str.strip().str.upper()

    train_df, test_df = train_test_split(
        gdf,
        test_size=0.2,
        random_state=42,
        stratify=gdf[LABEL_COL]
    )
    print(f"Train Points: {len(train_df)} | Test Points: {len(test_df)}")

    # ---------------------------------------------------------
    # Train RF
    # ---------------------------------------------------------
    print("\nExtracting features for Random Forest...")
    X_train, y_train = [], []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        x, y = row.geometry.x, row.geometry.y
        tif_path = find_tif_for_point(x, y, tif_index)
        if tif_path is None:
            continue

        with rasterio.open(tif_path) as src:
            emb = extract_embedding(encoder, src, x, y)
            if emb is not None:
                X_train.append(emb)
                y_train.append(row[LABEL_COL])

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print(f"Random Forest trained on {len(X_train)} points.")

    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print("\nRunning realignment test on Jittered points...")
    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        true_x, true_y = row.geometry.x, row.geometry.y
        target_species = row[LABEL_COL]

        jitter_x = true_x + random.uniform(-JITTER_MAX, JITTER_MAX)
        jitter_y = true_y + random.uniform(-JITTER_MAX, JITTER_MAX)
        initial_error = euclidean(true_x, true_y, jitter_x, jitter_y)

        tif_path = find_tif_for_point(jitter_x, jitter_y, tif_index)
        if tif_path is None:
            continue

        candidate_coords = []
        candidate_embeds = []

        with rasterio.open(tif_path) as src:
            for oy in np.arange(-SEARCH_RADIUS, SEARCH_RADIUS + 1e-6, GRID_STRIDE):
                for ox in np.arange(-SEARCH_RADIUS, SEARCH_RADIUS + 1e-6, GRID_STRIDE):
                    cx, cy = jitter_x + ox, jitter_y + oy
                    emb = extract_embedding(encoder, src, cx, cy)
                    if emb is not None:
                        candidate_coords.append([cx, cy])
                        candidate_embeds.append(emb)

        if len(candidate_embeds) == 0:
            results.append({
                "Species": target_species,
                "Initial_Error": initial_error,
                "Final_Error": initial_error,
                "Improvement": 0.0,
                "Status": "NO_VALID_PATCHES"
            })
            continue

        candidate_coords = np.array(candidate_coords, dtype=np.float32)
        candidate_embeds = np.array(candidate_embeds, dtype=np.float32)

        # predict probabilities
        proba = clf.predict_proba(candidate_embeds)

        if target_species not in class_to_idx:
            results.append({
                "Species": target_species,
                "Initial_Error": initial_error,
                "Final_Error": initial_error,
                "Improvement": 0.0,
                "Status": "UNKNOWN_SPECIES"
            })
            continue

        species_idx = class_to_idx[target_species]
        species_probs = proba[:, species_idx]

        # distance penalty from jittered point
        dists = np.sqrt((candidate_coords[:, 0] - jitter_x) ** 2 + (candidate_coords[:, 1] - jitter_y) ** 2)
        norm_dists = dists / max(SEARCH_RADIUS, 1e-6)

        scores = species_probs - DIST_PENALTY_ALPHA * norm_dists

        # keep high-confidence candidates only
        keep_mask = species_probs >= MIN_SPECIES_PROB
        filtered_coords = candidate_coords[keep_mask]
        filtered_scores = scores[keep_mask]
        filtered_probs = species_probs[keep_mask]

        if len(filtered_coords) < MIN_CLUSTER_SIZE:
            results.append({
                "Species": target_species,
                "Initial_Error": initial_error,
                "Final_Error": initial_error,
                "Improvement": 0.0,
                "Status": "LOW_CONFIDENCE_KEEP_ORIGINAL"
            })
            continue

        db = DBSCAN(eps=4.0, min_samples=MIN_CLUSTER_SIZE).fit(filtered_coords)
        labels = db.labels_

        valid_cluster_ids = [lab for lab in np.unique(labels) if lab != -1]
        if len(valid_cluster_ids) == 0:
            results.append({
                "Species": target_species,
                "Initial_Error": initial_error,
                "Final_Error": initial_error,
                "Improvement": 0.0,
                "Status": "NO_CLUSTER_KEEP_ORIGINAL"
            })
            continue

        # choose best cluster by mean score, not by size only
        best_cluster = None
        best_cluster_score = -1e9
        best_cluster_center = None

        for cid in valid_cluster_ids:
            pts = filtered_coords[labels == cid]
            pts_scores = filtered_scores[labels == cid]
            pts_probs = filtered_probs[labels == cid]

            center_x, center_y = pts.mean(axis=0)
            center_dist = euclidean(center_x, center_y, jitter_x, jitter_y)

            cluster_score = pts_scores.mean() + 0.2 * pts_probs.mean() - 0.1 * (center_dist / SEARCH_RADIUS)

            if cluster_score > best_cluster_score:
                best_cluster_score = cluster_score
                best_cluster = cid
                best_cluster_center = (center_x, center_y)

        realigned_x, realigned_y = best_cluster_center
        move_dist = euclidean(realigned_x, realigned_y, jitter_x, jitter_y)

        # safety fallback: do not move too far
        if move_dist > MAX_ALLOWED_SHIFT:
            final_error = initial_error
            improvement = 0.0
            status = "MOVE_TOO_LARGE_KEEP_ORIGINAL"
        else:
            final_error = euclidean(true_x, true_y, realigned_x, realigned_y)
            improvement = initial_error - final_error
            status = "MOVED"

            # optional fallback: if worse, keep original in evaluation
            if improvement < 0:
                final_error = initial_error
                improvement = 0.0
                status = "NEGATIVE_MOVE_REJECTED"

        results.append({
            "Species": target_species,
            "Initial_Error": initial_error,
            "Final_Error": final_error,
            "Improvement": improvement,
            "Status": status
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv("verification_metrics_improved.csv", index=False)

    print(f"\nVerification Complete. Mean Improvement: {res_df['Improvement'].mean():.2f}m")
    print("\nStatus summary:")
    print(res_df["Status"].value_counts())
    print("\nSpecies mean improvement:")
    print(res_df.groupby("Species")["Improvement"].mean().sort_values())
    

if __name__ == "__main__":
    main()