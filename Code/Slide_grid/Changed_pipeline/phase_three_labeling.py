import os
import glob
import json
import random
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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PATHS & HYPERPARAMETERS
# ---------------------------------------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_SHP = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"

LABEL_COL = "Tree"

MODEL_DIR = "data/models"
EXPORT_DIR = "data/exports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

"""
EPOCH CONTROLL
"""
EPOCHS = 7

ENCODER_PATH = os.path.join(MODEL_DIR, f"encoder_phase1_large_epoch{EPOCHS}.pth")
RF_BUNDLE_PATH = os.path.join(MODEL_DIR, "species_model_bundle.joblib")
RF_ONLY_PATH = os.path.join(MODEL_DIR, "rf_classifier.joblib")
CLASS_NAMES_JSON = os.path.join(EXPORT_DIR, "class_names_phase3.json")
TRAIN_METRICS_CSV = os.path.join(EXPORT_DIR, "phase3_test_predictions.csv")
CONFUSION_MATRIX_CSV = os.path.join(EXPORT_DIR, "phase3_confusion_matrix.csv")
CLASSIFICATION_REPORT_TXT = os.path.join(EXPORT_DIR, "phase3_classification_report.txt")

# image settings: must match phase 1 + inference
IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 2
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER   # 896
HALF_CROP = CROP_SIZE // 2               # 448

# data / model
TEST_SIZE = 0.2
SEED = 42
RF_TREES = 400
MIN_SAMPLES_PER_CLASS = 2  # too-rare classes below this are dropped

# these will be reused by Phase 2 / Final
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
# 3. HELPER FUNCTIONS
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


def read_crop_as_tensor(src, x, y):
    """
    Must match Phase 1 preprocessing exactly.
    Returns:
        img_t: torch.Tensor [3, IMG_SIZE, IMG_SIZE] or None
        ratio: valid pixel ratio
    """
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
            return None, 0.0

        ratio = valid_pixel_ratio(tile)

        img_t = torch.from_numpy(tile).float().unsqueeze(0) / 255.0
        img_t = F.interpolate(
            img_t,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_t = (img_t - mean_t) / std_t

        return img_t.squeeze(0), ratio

    except Exception:
        return None, 0.0


def extract_embedding(encoder, src, x, y):
    img_t, ratio = read_crop_as_tensor(src, x, y)
    if img_t is None:
        return None, ratio

    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        if USE_AMP:
            with torch.amp.autocast("cuda"):
                emb = encoder(img_t).mean(dim=1).cpu().numpy()[0]
        else:
            emb = encoder(img_t).mean(dim=1).cpu().numpy()[0]

    return emb.astype(np.float32), ratio


def load_annotations(shp_path, label_col):
    gdf = gpd.read_file(shp_path).copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf[label_col] = gdf[label_col].apply(normalize_species_name)
    gdf = gdf[gdf[label_col] != ""].copy()
    return gdf


def drop_rare_classes_for_split(gdf, label_col, min_count=2):
    counts = gdf[label_col].value_counts()
    keep_classes = counts[counts >= min_count].index.tolist()

    dropped = sorted(set(counts.index) - set(keep_classes))
    if dropped:
        print(f"Dropping rare classes (<{min_count} samples): {dropped}")

    gdf2 = gdf[gdf[label_col].isin(keep_classes)].copy()
    return gdf2


def build_embedding_table(gdf, tif_index, encoder, label_col):
    rows = []

    print("\nExtracting exact-center embeddings from labeled points...")
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), leave=False):
        x = float(row.geometry.x)
        y = float(row.geometry.y)
        label = normalize_species_name(row[label_col])

        tif_path = find_tif_for_point(x, y, tif_index)
        if tif_path is None:
            rows.append({
                "x": x,
                "y": y,
                "label": label,
                "status": "NO_TIF",
                "valid_ratio": 0.0,
                "embedding": None
            })
            continue

        try:
            with rasterio.open(tif_path) as src:
                emb, ratio = extract_embedding(encoder, src, x, y)

            if emb is None:
                rows.append({
                    "x": x,
                    "y": y,
                    "label": label,
                    "status": "INVALID_CROP",
                    "valid_ratio": ratio,
                    "embedding": None
                })
                continue

            rows.append({
                "x": x,
                "y": y,
                "label": label,
                "status": "OK",
                "valid_ratio": ratio,
                "embedding": emb
            })

        except Exception:
            rows.append({
                "x": x,
                "y": y,
                "label": label,
                "status": "ERROR",
                "valid_ratio": 0.0,
                "embedding": None
            })

    return pd.DataFrame(rows)


def save_bundle(clf):
    bundle = {
        "classifier": clf,
        "classes": clf.classes_.tolist(),
        "phase3_params": {
            "IMG_SIZE": IMG_SIZE,
            "PATCH_SIZE": PATCH_SIZE,
            "CROP_SIZE": CROP_SIZE,
            "SEARCH_RADIUS": SEARCH_RADIUS,
            "GRID_STRIDE": GRID_STRIDE,
            "MIN_SPECIES_PROB": MIN_SPECIES_PROB,
            "DIST_PENALTY_ALPHA": DIST_PENALTY_ALPHA,
            "MIN_CLUSTER_SIZE": MIN_CLUSTER_SIZE,
            "MAX_ALLOWED_SHIFT": MAX_ALLOWED_SHIFT,
            "DBSCAN_EPS": DBSCAN_EPS
        }
    }

    joblib.dump(clf, RF_ONLY_PATH)
    joblib.dump(bundle, RF_BUNDLE_PATH)

    with open(CLASS_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(clf.classes_.tolist(), f, ensure_ascii=False, indent=2)

    print(f"[SAVED] RF only -> {RF_ONLY_PATH}")
    print(f"[SAVED] RF bundle -> {RF_BUNDLE_PATH}")
    print(f"[SAVED] Class names -> {CLASS_NAMES_JSON}")


# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
def main():
    set_seed(SEED)
    print("\n--- Phase 3: Species Definition / Teacher Training ---")
    print(f"Device: {device}")

    # -----------------------------------------------------
    # Load encoder
    # -----------------------------------------------------
    encoder = LeJepaEncoder().to(device)

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            f"Encoder not found: {ENCODER_PATH}\n"
            f"Run Phase 1 first."
        )

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.eval()
    print(f"[SUCCESS] Loaded encoder from {ENCODER_PATH}")

    # -----------------------------------------------------
    # Build tif index
    # -----------------------------------------------------
    tif_index = build_tif_index(BASE_DIR)
    if len(tif_index) == 0:
        raise RuntimeError("No valid TIF files found.")

    # -----------------------------------------------------
    # Load labels
    # -----------------------------------------------------
    print("\nLoading labeled annotations...")
    gdf = load_annotations(ANNOTATED_SHP, LABEL_COL)
    print(f"Loaded labeled points: {len(gdf)}")

    if len(gdf) == 0:
        raise RuntimeError("No labeled points found.")

    gdf = drop_rare_classes_for_split(gdf, LABEL_COL, MIN_SAMPLES_PER_CLASS)
    print(f"Remaining labeled points after rare-class filter: {len(gdf)}")

    if len(gdf) == 0:
        raise RuntimeError("No valid labeled points remain after filtering.")

    # -----------------------------------------------------
    # Extract embeddings
    # -----------------------------------------------------
    emb_df = build_embedding_table(gdf, tif_index, encoder, LABEL_COL)

    print("\nEmbedding extraction status:")
    print(emb_df["status"].value_counts(dropna=False))

    emb_df = emb_df[emb_df["status"] == "OK"].copy()
    if len(emb_df) == 0:
        raise RuntimeError("No valid embeddings were extracted.")

    print(f"\nUsable labeled embeddings: {len(emb_df)}")
    print("Class distribution:")
    print(emb_df["label"].value_counts())

    # -----------------------------------------------------
    # Prepare arrays
    # -----------------------------------------------------
    X = np.stack(emb_df["embedding"].values).astype(np.float32)
    y = emb_df["label"].values
    coords = emb_df[["x", "y"]].values.astype(np.float32)

    # -----------------------------------------------------
    # Train / test split
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size : {len(X_test)}")

    # -----------------------------------------------------
    # Train Random Forest
    # -----------------------------------------------------
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=RF_TREES,
        random_state=SEED,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # -----------------------------------------------------
    # Save trained classifier bundle
    # -----------------------------------------------------
    save_bundle(clf)

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------
    print("\nEvaluating classifier on held-out test set...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:")
    print(report)

    labels_sorted = list(clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    cm_df.to_csv(CONFUSION_MATRIX_CSV)

    with open(CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {acc:.6f}\n\n")
        f.write(report)

    # per-sample outputs
    pred_rows = []
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    for i in range(len(X_test)):
        true_label = y_test[i]
        pred_label = y_pred[i]
        prob_true = float(y_proba[i, class_to_idx[true_label]]) if true_label in class_to_idx else np.nan
        prob_pred = float(np.max(y_proba[i]))

        pred_rows.append({
            "x": float(coords_test[i, 0]),
            "y": float(coords_test[i, 1]),
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": int(true_label == pred_label),
            "prob_true_label": prob_true,
            "prob_pred_label": prob_pred
        })

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(TRAIN_METRICS_CSV, index=False)

    print(f"\n[SAVED] Test predictions -> {TRAIN_METRICS_CSV}")
    print(f"[SAVED] Confusion matrix -> {CONFUSION_MATRIX_CSV}")
    print(f"[SAVED] Classification report -> {CLASSIFICATION_REPORT_TXT}")

    print("\n[PHASE 3 DONE]")
    print("Artifacts produced:")
    print(f" - Encoder used        : {ENCODER_PATH}")
    print(f" - RF bundle           : {RF_BUNDLE_PATH}")
    print(f" - RF only             : {RF_ONLY_PATH}")
    print(f" - Class names         : {CLASS_NAMES_JSON}")
    print(f" - Test predictions    : {TRAIN_METRICS_CSV}")
    print(f" - Confusion matrix    : {CONFUSION_MATRIX_CSV}")
    print(f" - Classification text : {CLASSIFICATION_REPORT_TXT}")


if __name__ == "__main__":
    main()