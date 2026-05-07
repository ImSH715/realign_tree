import os
import numpy as np
import pandas as pd
import rasterio
import torch
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.preprocess import preprocess

REFINED_CSV = "./outputs/evaluation/single_tif_beta0002_refined.csv"
ENCODER_CKPT = "./outputs/phase1_5_lejepa_cpu_binary_preprocess/phase1_encoder_best.pth"
PROTOTYPES_CSV = "./outputs/phase2_binary_shihuahuaco/class_prototypes_named.csv"
OUTPUT_DIR = "./outputs/evaluation/one_tif_analysis"

PATCH_SIZE = 224
IMAGE_SIZE = 224
DEVICE = "cpu"
USE_TSNE = True
TARGET_POSITIVE = "Shihuahuaco"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_patch_world(image_path, east, north, patch_size):
    with rasterio.open(image_path) as src:
        row, col = src.index(float(east), float(north))
        half = patch_size // 2
        window = rasterio.windows.Window(
            int(round(col)) - half,
            int(round(row)) - half,
            patch_size,
            patch_size,
        )
        arr = src.read(window=window, boundless=True, fill_value=0)

    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    else:
        raise ValueError(f"Invalid band shape: {arr.shape}")

    arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.nanpercentile(arr, [1, 99])
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr)
    img = preprocess(img)
    return img


def pil_to_tensor(img):
    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std
    return torch.from_numpy(arr)


def forward_features(model, x):
    if hasattr(model, "encode"):
        out = model.encode(x)
    else:
        out = model(x)

    if isinstance(out, dict):
        for k in ["features", "embedding", "embeddings", "x", "last_hidden_state"]:
            if k in out:
                out = out[k]
                break

    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.ndim == 4:
        out = out.mean(dim=(2, 3))
    elif out.ndim == 3:
        out = out[:, 0]

    return out


def l2_normalize(x, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def nearest_prototype_labels(embeddings, proto_labels, proto_emb):
    emb_n = l2_normalize(embeddings, axis=1)
    proto_n = l2_normalize(proto_emb, axis=1)
    sim = emb_n @ proto_n.T
    idx = sim.argmax(axis=1)
    pred = [proto_labels[i] for i in idx]
    best_sim = sim[np.arange(len(idx)), idx]
    return pred, best_sim


def to_binary_label(x):
    return TARGET_POSITIVE if str(x).strip() == TARGET_POSITIVE else "Other"


df = pd.read_csv(REFINED_CSV).copy()

required = [
    "image_path",
    "original_east", "original_north",
    "refined_east", "refined_north",
    "label",
]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

proto = pd.read_csv(PROTOTYPES_CSV).copy()
proto_label_col = proto.columns[0]
emb_cols = [c for c in proto.columns if c.startswith("emb_")]
proto_labels = proto[proto_label_col].astype(str).str.strip().tolist()
proto_emb = proto[emb_cols].values.astype(np.float32)

device = torch.device(DEVICE)
model, _ = load_encoder_from_checkpoint(ENCODER_CKPT, device)
model.eval()

orig_embs = []
ref_embs = []

with torch.no_grad():
    for _, row in df.iterrows():
        img_path = row["image_path"]

        orig_img = read_patch_world(
            img_path, row["original_east"], row["original_north"], PATCH_SIZE
        )
        ref_img = read_patch_world(
            img_path, row["refined_east"], row["refined_north"], PATCH_SIZE
        )

        x_orig = pil_to_tensor(orig_img).unsqueeze(0).to(device)
        x_ref = pil_to_tensor(ref_img).unsqueeze(0).to(device)

        z_orig = forward_features(model, x_orig).cpu().numpy()[0]
        z_ref = forward_features(model, x_ref).cpu().numpy()[0]

        orig_embs.append(z_orig)
        ref_embs.append(z_ref)

orig_embs = np.stack(orig_embs, axis=0)
ref_embs = np.stack(ref_embs, axis=0)

if "gt_east" in df.columns and "gt_north" in df.columns:
    before = np.sqrt(
        (df["original_east"] - df["gt_east"]) ** 2
        + (df["original_north"] - df["gt_north"]) ** 2
    )
    after = np.sqrt(
        (df["refined_east"] - df["gt_east"]) ** 2
        + (df["refined_north"] - df["gt_north"]) ** 2
    )
    move = np.sqrt(
        (df["refined_east"] - df["original_east"]) ** 2
        + (df["refined_north"] - df["original_north"]) ** 2
    )

    spatial_summary = {
        "n": len(df),
        "mean_before_m": float(before.mean()),
        "mean_after_m": float(after.mean()),
        "median_before_m": float(before.median()),
        "median_after_m": float(after.median()),
        "mean_movement_m": float(move.mean()),
        "improved": int((after < before).sum()),
        "unchanged": int((after == before).sum()),
        "worse": int((after > before).sum()),
        "acc_before_1m": float((before <= 1).mean()),
        "acc_after_1m": float((after <= 1).mean()),
        "acc_before_2m": float((before <= 2).mean()),
        "acc_after_2m": float((after <= 2).mean()),
        "acc_before_5m": float((before <= 5).mean()),
        "acc_after_5m": float((after <= 5).mean()),
    }
    pd.DataFrame([spatial_summary]).to_csv(
        os.path.join(OUTPUT_DIR, "spatial_summary.csv"), index=False
    )

orig_pred, orig_sim = nearest_prototype_labels(orig_embs, proto_labels, proto_emb)
ref_pred, ref_sim = nearest_prototype_labels(ref_embs, proto_labels, proto_emb)

df["orig_pred_label"] = orig_pred
df["ref_pred_label"] = ref_pred
df["orig_pred_sim"] = orig_sim
df["ref_pred_sim"] = ref_sim

true_mc = df["label"].astype(str).str.strip().tolist()
orig_acc_mc = np.mean([t == p for t, p in zip(true_mc, orig_pred)])
ref_acc_mc = np.mean([t == p for t, p in zip(true_mc, ref_pred)])

true_bin = [to_binary_label(x) for x in true_mc]
orig_pred_bin = [to_binary_label(x) for x in orig_pred]
ref_pred_bin = [to_binary_label(x) for x in ref_pred]

orig_acc_bin = np.mean([t == p for t, p in zip(true_bin, orig_pred_bin)])
ref_acc_bin = np.mean([t == p for t, p in zip(true_bin, ref_pred_bin)])

classification_summary = {
    "multiclass_acc_original": float(orig_acc_mc),
    "multiclass_acc_refined": float(ref_acc_mc),
    "binary_acc_original": float(orig_acc_bin),
    "binary_acc_refined": float(ref_acc_bin),
    "mean_orig_proto_sim": float(np.mean(orig_sim)),
    "mean_ref_proto_sim": float(np.mean(ref_sim)),
}
pd.DataFrame([classification_summary]).to_csv(
    os.path.join(OUTPUT_DIR, "classification_summary.csv"), index=False
)

df.to_csv(os.path.join(OUTPUT_DIR, "pointwise_analysis.csv"), index=False)

all_embs = np.concatenate([orig_embs, ref_embs], axis=0)
all_stage = np.array(["original"] * len(orig_embs) + ["refined"] * len(ref_embs))
all_true_bin = np.array(true_bin + true_bin)

pca = PCA(n_components=2, random_state=42)
xy_pca = pca.fit_transform(all_embs)

plt.figure(figsize=(8, 6))
for cls in ["Shihuahuaco", "Other"]:
    m = all_true_bin == cls
    plt.scatter(xy_pca[m, 0], xy_pca[m, 1], s=18, label=cls, alpha=0.7)
plt.title("PCA: Shihuahuaco vs Others")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_binary.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 6))
for st in ["original", "refined"]:
    m = all_stage == st
    plt.scatter(xy_pca[m, 0], xy_pca[m, 1], s=18, label=st, alpha=0.7)
plt.title("PCA: Original vs Refined")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_stage.png"), dpi=200)
plt.close()

if USE_TSNE and len(all_embs) >= 6:
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(all_embs) // 4)),
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    xy_tsne = tsne.fit_transform(all_embs)

    plt.figure(figsize=(8, 6))
    for cls in ["Shihuahuaco", "Other"]:
        m = all_true_bin == cls
        plt.scatter(xy_tsne[m, 0], xy_tsne[m, 1], s=18, label=cls, alpha=0.7)
    plt.title("t-SNE: Shihuahuaco vs Others")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_binary.png"), dpi=200)
    plt.close()

print("Saved outputs to:", OUTPUT_DIR)
print(" - spatial_summary.csv")
print(" - classification_summary.csv")
print(" - pointwise_analysis.csv")
print(" - pca_binary.png")
print(" - pca_stage.png")
if USE_TSNE and len(all_embs) >= 6:
    print(" - tsne_binary.png")