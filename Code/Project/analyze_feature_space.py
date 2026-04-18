"""
Analyze the learned feature space used by the refinement pipeline.

This script:
- loads an already trained encoder,
- extracts embeddings for original or refined points,
- assigns nearest prototype labels,
- projects embeddings to 2D using PCA,
- clusters embeddings using DBSCAN,
- saves embeddings / summaries,
- and generates visualization plots.

This is intended to help inspect representation quality:
- whether classes are separable,
- whether clusters are meaningful,
- whether prototype assignments agree with expected labels,
- and whether the feature space is collapsed or discriminative.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import rasterio
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from src.models.checkpoint import load_encoder_from_checkpoint
from src.data.patches import PatchExtractor, EncoderWrapper
from src.scoring.prototypes import load_prototypes_csv


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def world_to_pixel(image_path: str, east: float, north: float):
    with rasterio.open(image_path) as src:
        row, col = src.index(float(east), float(north))
    return float(col), float(row)


def nearest_prototype_label(embedding: np.ndarray, prototypes: dict):
    best_label = None
    best_sim = -1e9
    for label, proto in prototypes.items():
        sim = cosine_similarity(embedding, proto)
        if sim > best_sim:
            best_sim = sim
            best_label = label
    return best_label, float(best_sim)


def load_points_for_analysis(
    csv_path: str,
    label_column: str,
    image_column: str,
    mode: str,
    filter_label: str = "",
):
    df = pd.read_csv(csv_path)

    if label_column not in df.columns:
        raise ValueError(f"Missing label column '{label_column}'")

    if image_column not in df.columns:
        raise ValueError(f"Missing image column '{image_column}'")

    df[label_column] = df[label_column].astype(str).str.strip()

    if filter_label:
        df = df[df[label_column].str.lower() == filter_label.strip().lower()].copy()

    records = []

    for _, row in df.iterrows():
        image_path = str(row[image_column]).strip()

        if mode == "original":
            if {"original_x", "original_y"}.issubset(df.columns):
                x_px = float(row["original_x"])
                y_px = float(row["original_y"])
            elif {"original_east", "original_north"}.issubset(df.columns):
                x_px, y_px = world_to_pixel(image_path, row["original_east"], row["original_north"])
            else:
                raise ValueError("Could not find original pixel/world columns.")
        elif mode == "refined":
            if {"refined_x", "refined_y"}.issubset(df.columns):
                x_px = float(row["refined_x"])
                y_px = float(row["refined_y"])
            elif {"refined_east", "refined_north"}.issubset(df.columns):
                x_px, y_px = world_to_pixel(image_path, row["refined_east"], row["refined_north"])
            else:
                raise ValueError("Could not find refined pixel/world columns.")
        else:
            raise ValueError("mode must be original or refined")

        records.append(
            {
                "point_id": str(row["point_id"]) if "point_id" in df.columns else None,
                "image_path": image_path,
                "label": str(row[label_column]).strip(),
                "x_px": x_px,
                "y_px": y_px,
            }
        )

    return pd.DataFrame(records)


def encode_points(df_points, encoder, patch_extractor, batch_size: int):
    patches = []
    metas = []

    for _, row in df_points.iterrows():
        try:
            patch = patch_extractor.extract(
                image_path=row["image_path"],
                center_x=row["x_px"],
                center_y=row["y_px"],
            )
            patches.append(patch)
            metas.append(row.to_dict())
        except Exception as e:
            print(f"[WARN] Patch extraction failed: {e}")

    feats = encoder.encode_batch(patches, batch_size=batch_size)

    rows = []
    for meta, feat in zip(metas, feats):
        rec = dict(meta)
        for i, v in enumerate(feat):
            rec[f"emb_{i}"] = float(v)
        rows.append(rec)

    return pd.DataFrame(rows)


def assign_prototypes(df_emb: pd.DataFrame, prototypes: dict):
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]

    pred_labels = []
    pred_sims = []

    for _, row in df_emb.iterrows():
        emb = row[emb_cols].to_numpy(dtype=np.float32)
        label, sim = nearest_prototype_label(emb, prototypes)
        pred_labels.append(label)
        pred_sims.append(sim)

    df_emb["nearest_prototype_label"] = pred_labels
    df_emb["nearest_prototype_similarity"] = pred_sims
    df_emb["prototype_match"] = (df_emb["label"] == df_emb["nearest_prototype_label"])
    return df_emb


def run_dbscan(df_emb: pd.DataFrame, eps: float, min_samples: int):
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    X = df_emb[emb_cols].to_numpy(dtype=np.float32)

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_ids = db.fit_predict(Xn)
    df_emb["dbscan_cluster"] = cluster_ids

    summary = {
        "num_clusters_excluding_noise": int(len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)),
        "num_noise_points": int((cluster_ids == -1).sum()),
    }

    mask = cluster_ids != -1
    if mask.sum() > 2 and len(set(cluster_ids[mask])) > 1:
        try:
            summary["silhouette_score"] = float(silhouette_score(Xn[mask], cluster_ids[mask]))
        except Exception:
            summary["silhouette_score"] = None
    else:
        summary["silhouette_score"] = None

    return df_emb, summary


def add_pca(df_emb: pd.DataFrame):
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    X = df_emb[emb_cols].to_numpy(dtype=np.float32)

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)

    df_emb["pca_x"] = xy[:, 0]
    df_emb["pca_y"] = xy[:, 1]

    return df_emb, {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def plot_scatter(df_emb: pd.DataFrame, color_col: str, output_png: str, title: str):
    plt.figure(figsize=(10, 8))

    values = sorted(df_emb[color_col].astype(str).unique().tolist())
    for val in values:
        sub = df_emb[df_emb[color_col].astype(str) == str(val)]
        plt.scatter(sub["pca_x"], sub["pca_y"], s=18, alpha=0.8, label=str(val))

    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze feature space / clustering.")
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--prototypes_csv", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--label_column", default="target_label")
    parser.add_argument("--image_column", default="image_path")
    parser.add_argument("--mode", choices=["original", "refined"], default="refined")
    parser.add_argument("--filter_label", default="")

    parser.add_argument("--patch_size_px", type=int, default=224)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--dbscan_eps", type=float, default=0.20)
    parser.add_argument("--dbscan_min_samples", type=int, default=5)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    safe_mkdir(args.output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    model, _ = load_encoder_from_checkpoint(args.encoder_ckpt, device)
    encoder = EncoderWrapper(
        model=model,
        device=device,
        image_size=args.image_size,
        use_amp=use_amp,
    )
    patch_extractor = PatchExtractor(args.patch_size_px)
    prototypes = load_prototypes_csv(args.prototypes_csv)

    df_points = load_points_for_analysis(
        csv_path=args.input_csv,
        label_column=args.label_column,
        image_column=args.image_column,
        mode=args.mode,
        filter_label=args.filter_label,
    )

    print(f"[INFO] Points loaded for feature analysis: {len(df_points)}")

    df_emb = encode_points(df_points, encoder, patch_extractor, args.batch_size)
    df_emb = assign_prototypes(df_emb, prototypes)
    df_emb, dbscan_summary = run_dbscan(df_emb, args.dbscan_eps, args.dbscan_min_samples)
    df_emb, pca_summary = add_pca(df_emb)

    emb_csv = os.path.join(args.output_dir, "feature_space_embeddings.csv")
    df_emb.to_csv(emb_csv, index=False)

    plot_scatter(
        df_emb,
        color_col="label",
        output_png=os.path.join(args.output_dir, "pca_by_true_label.png"),
        title="PCA by true label",
    )
    plot_scatter(
        df_emb,
        color_col="nearest_prototype_label",
        output_png=os.path.join(args.output_dir, "pca_by_nearest_prototype.png"),
        title="PCA by nearest prototype label",
    )
    plot_scatter(
        df_emb,
        color_col="dbscan_cluster",
        output_png=os.path.join(args.output_dir, "pca_by_dbscan_cluster.png"),
        title="PCA by DBSCAN cluster",
    )

    summary = {
        "num_points": int(len(df_emb)),
        "prototype_match_rate": float(df_emb["prototype_match"].mean()),
        "dbscan": dbscan_summary,
        "pca": pca_summary,
    }

    with open(os.path.join(args.output_dir, "feature_space_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("Feature space analysis completed")
    print(f"Embeddings CSV        : {emb_csv}")
    print(f"Prototype match rate  : {summary['prototype_match_rate']:.4f}")
    print(f"DBSCAN clusters       : {dbscan_summary['num_clusters_excluding_noise']}")
    print(f"DBSCAN noise points   : {dbscan_summary['num_noise_points']}")
    print(f"PCA variance (2D sum) : {pca_summary['explained_variance_ratio_sum']:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()