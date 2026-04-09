"""
Phase two is to use encoder from phase one and labelling results from phase three to realign the points.
- Phase two but run phase three first.
Pipeline order
- Phase one -> Phase three -> Phase two
"""
import os
import glob
import random
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import DBSCAN
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PATHS & HYPERPARAMETERS
# ---------------------------------------------------------
TIF_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
LARGE_CENSUS_CSV = "data/tree_label_rdn/Censo_Forestal.csv"
OUTPUT_CSV = "OSINFOR_2023_Realigned_Final.csv"

MODEL_DIR = "data/models"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_phase1_large_epoch1.pth")
RF_BUNDLE_PATH = os.path.join(MODEL_DIR, "species_model_bundle.joblib")

# CSV column names
LABEL_COL_CSV = "NOMBRE_COMUN"   # 필요 시 "NOMBRE_CIENTIFICO" 로 교체
X_COL_CSV = "COORDENADA_ESTE"
Y_COL_CSV = "COORDENADA_NORTE"

# image settings: must match Phase 1 / Phase 3 exactly
IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER   # 896
HALF_CROP = CROP_SIZE // 2               # 448

# inference speed / checkpointing
BATCH_SIZE_INFER = 64
CHECKPOINT_EVERY = 500
SEED = 42

# defaults, overwritten by bundle if available
SEARCH_RADIUS = 20.0
GRID_STRIDE = 3.0
MIN_SPECIES_PROB = 0.45
DIST_PENALTY_ALPHA = 0.35
MIN_CLUSTER_SIZE = 3
MAX_ALLOWED_SHIFT = 25.0
DBSCAN_EPS = 4.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (device.type == "cuda")


# ---------------------------------------------------------
# 2. MODEL DEFINITION
# ---------------------------------------------------------
class LeJepaEncoder(nn.Module):
    def __init__(
        self,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=3,
        embed_dim=128,
        depth=4,
        num_heads=4
    ):
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
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        if ids_keep is not None:
            D = x.shape[-1]
            x = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        for block in self.blocks:
            x = block(x)

        return self.norm(x)


# ---------------------------------------------------------
# 3. UTILS
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_species_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def build_tif_index(tif_dir):
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


def find_tif_for_point(x, y, tif_index):
    for item in tif_index:
        b = item["bounds"]
        if b.left <= x <= b.right and b.bottom <= y <= b.top:
            return item["path"]
    return None


def valid_pixel_ratio(tile):
    if tile is None or tile.size == 0:
        return 0.0
    nonzero = np.any(tile > 0, axis=0)
    return float(nonzero.mean())


def build_search_grid(center_x, center_y, radius, stride, circular=True):
    coords = []
    for oy in np.arange(-radius, radius + 1e-6, stride):
        for ox in np.arange(-radius, radius + 1e-6, stride):
            if circular:
                if (ox ** 2 + oy ** 2) > (radius ** 2):
                    continue
            coords.append((center_x + ox, center_y + oy))
    return coords


def extract_embedding_batch(encoder, src, coords, batch_size=BATCH_SIZE_INFER):
    """
    Extract embeddings for many (x, y) points in one batched forward pass.
    Returns:
        valid_coords_np: (N,2)
        embeds_np:       (N,D)
        ratios_np:       (N,)
    """
    if len(coords) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 128), dtype=np.float32),
            np.empty((0,), dtype=np.float32)
        )

    tiles = []
    valid_coords = []
    ratios = []

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

            ratio = valid_pixel_ratio(tile)

            tiles.append(tile)
            valid_coords.append((x, y))
            ratios.append(ratio)

        except Exception:
            continue

    if len(tiles) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 128), dtype=np.float32),
            np.empty((0,), dtype=np.float32)
        )

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
    ratios_np = np.array(ratios, dtype=np.float32)

    return valid_coords_np, embeds_np, ratios_np


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
            df[LABEL_COL_CSV] = df[LABEL_COL_CSV].apply(normalize_species_name)

        return df

    print(f"[NEW RUN] Loading source CSV: {LARGE_CENSUS_CSV}")
    df = pd.read_csv(LARGE_CENSUS_CSV)

    if LABEL_COL_CSV not in df.columns:
        raise ValueError(f"Missing label column in CSV: {LABEL_COL_CSV}")
    if X_COL_CSV not in df.columns or Y_COL_CSV not in df.columns:
        raise ValueError(f"Missing coordinate columns: {X_COL_CSV}, {Y_COL_CSV}")

    df[LABEL_COL_CSV] = df[LABEL_COL_CSV].apply(normalize_species_name)

    df["ORIGINAL_X"] = df[X_COL_CSV]
    df["ORIGINAL_Y"] = df[Y_COL_CSV]
    df["REALIGNED_X"] = df["ORIGINAL_X"]
    df["REALIGNED_Y"] = df["ORIGINAL_Y"]
    df["SHIFT_DISTANCE"] = 0.0
    df["STATUS"] = "PENDING"

    return df


def choose_best_cluster(filtered_coords, filtered_scores, filtered_probs, origin_x, origin_y, search_radius, eps, min_cluster_size):
    if len(filtered_coords) < min_cluster_size:
        return None

    db = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(filtered_coords)
    labels = db.labels_

    valid_cluster_ids = [lab for lab in np.unique(labels) if lab != -1]
    if len(valid_cluster_ids) == 0:
        return None

    best_cluster_score = -1e9
    best_cluster_center = None
    best_cluster_size = 0

    for cid in valid_cluster_ids:
        pts = filtered_coords[labels == cid]
        pts_scores = filtered_scores[labels == cid]
        pts_probs = filtered_probs[labels == cid]

        center_x, center_y = pts.mean(axis=0)
        center_dist = float(np.sqrt((center_x - origin_x) ** 2 + (center_y - origin_y) ** 2))

        cluster_score = (
            float(pts_scores.mean())
            + 0.2 * float(pts_probs.mean())
            - 0.1 * (center_dist / max(search_radius, 1e-6))
        )

        if cluster_score > best_cluster_score:
            best_cluster_score = cluster_score
            best_cluster_center = (float(center_x), float(center_y))
            best_cluster_size = int(len(pts))

    if best_cluster_center is None:
        return None

    return {
        "center_x": best_cluster_center[0],
        "center_y": best_cluster_center[1],
        "cluster_score": best_cluster_score,
        "cluster_size": best_cluster_size
    }


# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    global SEARCH_RADIUS, GRID_STRIDE, MIN_SPECIES_PROB
    global DIST_PENALTY_ALPHA, MIN_CLUSTER_SIZE, MAX_ALLOWED_SHIFT, DBSCAN_EPS

    set_seed(SEED)
    print("\n--- Phase 2: Local Grid Search Realignment ---")
    print(f"Device: {device}")

    # -----------------------------------------------------
    # Load encoder
    # -----------------------------------------------------
    encoder = LeJepaEncoder().to(device)

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.eval()
    print(f"[SUCCESS] Loaded encoder from {ENCODER_PATH}")

    # -----------------------------------------------------
    # Load species classifier bundle
    # -----------------------------------------------------
    if not os.path.exists(RF_BUNDLE_PATH):
        raise FileNotFoundError(f"RF bundle not found: {RF_BUNDLE_PATH}")

    bundle = joblib.load(RF_BUNDLE_PATH)
    clf = bundle["classifier"]
    known_species = set(bundle["classes"])

    if "phase3_params" in bundle:
        params = bundle["phase3_params"]
        SEARCH_RADIUS = float(params.get("SEARCH_RADIUS", SEARCH_RADIUS))
        GRID_STRIDE = float(params.get("GRID_STRIDE", GRID_STRIDE))
        MIN_SPECIES_PROB = float(params.get("MIN_SPECIES_PROB", MIN_SPECIES_PROB))
        DIST_PENALTY_ALPHA = float(params.get("DIST_PENALTY_ALPHA", DIST_PENALTY_ALPHA))
        MIN_CLUSTER_SIZE = int(params.get("MIN_CLUSTER_SIZE", MIN_CLUSTER_SIZE))
        MAX_ALLOWED_SHIFT = float(params.get("MAX_ALLOWED_SHIFT", MAX_ALLOWED_SHIFT))
        DBSCAN_EPS = float(params.get("DBSCAN_EPS", DBSCAN_EPS))

    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    print(f"[SUCCESS] Loaded RF bundle from {RF_BUNDLE_PATH}")
    print(f"Known species count: {len(known_species)}")
    print("Using parameters:")
    print(f" - SEARCH_RADIUS      : {SEARCH_RADIUS}")
    print(f" - GRID_STRIDE        : {GRID_STRIDE}")
    print(f" - MIN_SPECIES_PROB   : {MIN_SPECIES_PROB}")
    print(f" - DIST_PENALTY_ALPHA : {DIST_PENALTY_ALPHA}")
    print(f" - MIN_CLUSTER_SIZE   : {MIN_CLUSTER_SIZE}")
    print(f" - MAX_ALLOWED_SHIFT  : {MAX_ALLOWED_SHIFT}")
    print(f" - DBSCAN_EPS         : {DBSCAN_EPS}")

    # -----------------------------------------------------
    # Build tif index
    # -----------------------------------------------------
    tif_index = build_tif_index(TIF_DIR)
    if len(tif_index) == 0:
        raise RuntimeError("No valid TIFs found in TIF_DIR.")

    # -----------------------------------------------------
    # Load census CSV
    # -----------------------------------------------------
    df = initialize_or_resume_csv()
    pending_idx = df.index[df["STATUS"] == "PENDING"].tolist()

    print(f"\nPending rows to process: {len(pending_idx)} / {len(df)}")

    processed_since_save = 0

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------
    for idx in tqdm(pending_idx, total=len(pending_idx), desc="Realigning rows"):
        row = df.loc[idx]

        try:
            orig_x = float(row["ORIGINAL_X"])
            orig_y = float(row["ORIGINAL_Y"])
            target_species = normalize_species_name(row[LABEL_COL_CSV])

            # -----------------------------
            # Basic validation
            # -----------------------------
            if target_species == "":
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_EMPTY_SPECIES"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

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

            # -----------------------------
            # Build local search grid
            # -----------------------------
            candidate_coords = build_search_grid(
                center_x=orig_x,
                center_y=orig_y,
                radius=SEARCH_RADIUS,
                stride=GRID_STRIDE,
                circular=True
            )

            with rasterio.open(tif_path) as src:
                search_coords, search_embeds, search_ratios = extract_embedding_batch(
                    encoder, src, candidate_coords, batch_size=BATCH_SIZE_INFER
                )

            if len(search_embeds) == 0:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_NO_VALID_PATCHES"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            # -----------------------------
            # Predict probabilities
            # -----------------------------
            proba = clf.predict_proba(search_embeds)

            if target_species not in class_to_idx:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_UNKNOWN_SPECIES_INDEX"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            species_idx = class_to_idx[target_species]
            species_probs = proba[:, species_idx]

            # -----------------------------
            # Distance-aware scoring
            # -----------------------------
            dists = np.sqrt(
                (search_coords[:, 0] - orig_x) ** 2 +
                (search_coords[:, 1] - orig_y) ** 2
            )
            norm_dists = dists / max(SEARCH_RADIUS, 1e-6)

            scores = species_probs - DIST_PENALTY_ALPHA * norm_dists

            # -----------------------------
            # Keep confident candidates only
            # -----------------------------
            keep_mask = species_probs >= MIN_SPECIES_PROB

            filtered_coords = search_coords[keep_mask]
            filtered_scores = scores[keep_mask]
            filtered_probs = species_probs[keep_mask]

            if len(filtered_coords) < MIN_CLUSTER_SIZE:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_LOW_CONFIDENCE"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            # -----------------------------
            # Cluster and choose best centroid
            # -----------------------------
            best_cluster = choose_best_cluster(
                filtered_coords=filtered_coords,
                filtered_scores=filtered_scores,
                filtered_probs=filtered_probs,
                origin_x=orig_x,
                origin_y=orig_y,
                search_radius=SEARCH_RADIUS,
                eps=DBSCAN_EPS,
                min_cluster_size=MIN_CLUSTER_SIZE
            )

            if best_cluster is None:
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_NO_CLUSTER"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            new_x = float(best_cluster["center_x"])
            new_y = float(best_cluster["center_y"])
            shift = float(np.sqrt((orig_x - new_x) ** 2 + (orig_y - new_y) ** 2))

            # -----------------------------
            # Safety guard: do not move too far
            # -----------------------------
            if shift > MAX_ALLOWED_SHIFT:
                df.at[idx, "REALIGNED_X"] = float(orig_x)
                df.at[idx, "REALIGNED_Y"] = float(orig_y)
                df.at[idx, "SHIFT_DISTANCE"] = 0.0
                df.at[idx, "STATUS"] = "KEEP_ORIGINAL_MOVE_TOO_LARGE"
                processed_since_save += 1
                if processed_since_save >= CHECKPOINT_EVERY:
                    df.to_csv(OUTPUT_CSV, index=False)
                    processed_since_save = 0
                continue

            # -----------------------------
            # Accept move
            # -----------------------------
            df.at[idx, "REALIGNED_X"] = new_x
            df.at[idx, "REALIGNED_Y"] = new_y
            df.at[idx, "SHIFT_DISTANCE"] = shift
            df.at[idx, "STATUS"] = "SUCCESSFULLY_MOVED"

        except Exception as e:
            df.at[idx, "STATUS"] = f"ERROR_{type(e).__name__}"

        processed_since_save += 1

        if processed_since_save >= CHECKPOINT_EVERY:
            df.to_csv(OUTPUT_CSV, index=False)
            processed_since_save = 0

    # -----------------------------------------------------
    # Save final output
    # -----------------------------------------------------
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nRealignment saved successfully to {OUTPUT_CSV}")
    print("\nSTATUS summary:")
    print(df["STATUS"].value_counts(dropna=False))

    moved_mask = df["STATUS"] == "SUCCESSFULLY_MOVED"
    if moved_mask.any():
        print(f"\nMean shift distance (moved only): {df.loc[moved_mask, 'SHIFT_DISTANCE'].mean():.3f}")

    print("\n[PHASE 2 DONE]")


if __name__ == "__main__":
    main()