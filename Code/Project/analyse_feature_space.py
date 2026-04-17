import os
import json
import math
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import rasterio
from PIL import Image
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


def nearest_prototype_label(embedding: np.ndarray, prototypes: dict):
    best_label = None
    best_sim = -1e9
    for label, proto in prototypes.items():
        sim = cosine_similarity(embedding, proto)
        if sim > best_sim:
            best_sim = sim
            best_label = label
    return best_label, float(best_sim)


def world_to_pixel(image_path: str, x_world: float, y_world: float):
    with rasterio.open(image_path) as src:
        row, col = src.index(float(x_world), float(y_world))
    return float(col), float(row)


def load_points_from_censo_csv(
    csv_path: str,
    label_column: str,
    x_column: str,
    y_column: str,
    image_column: str,
    filter_label: str = "",
):
    df = pd.read_csv(csv_path)

    required = [label_column, x_column, y_column, image_column]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'. Available: {df.columns.tolist()}")

    df[label_column] = df[label_column].astype(str).str.strip()

    if filter_label:
        df = df[df[label_column].str.lower() == filter_label.strip().lower()].copy()

    rows = []
    for i, row in df.iterrows():
        image_path = str(row[image_column]).strip()
        x_world = float(row[x_column])
        y_world = float(row[y_column])

        try:
            x_px, y_px = world_to_pixel(image_path, x_world, y_world)
            rows.append(
                {
                    "point_id": f"pt_{i}",
                    "image_path": image_path,
                    "label": str(row[label_column]).strip(),
                    "x_world": x_world,
                    "y_world": y_world,
                    "x_px": x_px,
                    "y_px": y_px,
                    "source": "censo",
                }
            )
        except Exception as e:
            print(f"[WARN] Skipping row {i}: {e}")

    return pd.DataFrame(rows)


def load_points_from_refined_csv(
    csv_path: str,
    label_column: str = "target_label",
):
    df = pd.read_csv(csv_path)

    required = ["point_id", "image_path", "original_x", "original_y", "refined_x", "refined_y"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'. Available: {df.columns.tolist()}")

    if label_column not in df.columns:
        raise ValueError(f"Missing label column '{label_column}'. Available: {df.columns.tolist()}")

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "point_id": str(row["point_id"]),
                "image_path": str(row["image_path"]),
                "label": str(row[label_column]).strip(),
                "x_world": np.nan,
                "y_world": np.nan,
                "x_px": float(row["refined_x"]),
                "y_px": float(row["refined_y"]),
                "source": "refined",
            }
        )

    return pd.DataFrame(rows)


def encode_points(df_points, encoder, patch_extractor, batch_size: int):
    patches = []
    metas = []

    for _, row in df_points.iterrows():
        try:
            patch = patch_extractor.extract(
                image_path=row["image_path"],
                center_x=float(row["x_px"]),
                center_y=float(row["y_px"]),
            )
            patches.append(patch)
            metas.append(row.to_dict())
        except Exception as e:
            print(f"[WARN] Patch extraction failed for {row.get('point_id', 'unknown')}: {e}")

    if len(patches) == 0:
        raise RuntimeError("No valid patches extracted.")

    feats = encoder.encode_batch(patches, batch_size=batch_size)

    out_rows = []
    for meta, emb in zip(metas, feats):
        record = dict(meta)
        for j, v in enumerate(emb):
            record[f"emb_{j}"] = float(v)
        out_rows.append(record)

    return pd.DataFrame(out_rows)


def assign_prototypes(df_emb: pd.DataFrame, prototypes: dict):
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    pred_labels = []
    pred_sims = []

    for _, row in df_emb.iterrows():
        emb = row[emb_cols].to_numpy(dtype=np.float32)
        pred_label, pred_sim = nearest_prototype_label(emb, prototypes)
        pred_labels.append(pred_label)
        pred_sims.append(pred_sim)

    df_emb["nearest_prototype_label"] = pred_labels
    df_emb["nearest_prototype_similarity"] = pred_sims
    df_emb["prototype_match"] = (df_emb["label"].astype(str) == df_emb["nearest_prototype_label"].astype(str))
    return df_emb


def run_dbscan(df_emb: pd.DataFrame, eps: float, min_samples: int):
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    X = df_emb[emb_cols].to_numpy(dtype=np.float32)

    # Normalize embeddings for cosine-like geometry
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_ids = db.fit_predict(Xn)

    df_emb["dbscan_cluster"] = cluster_ids

    summary = {
        "num_points": int(len(df_emb)),
        "num_clusters_excluding_noise": int(len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)),
        "num_noise_points": int((cluster_ids == -1).sum()),
        "cluster_sizes": dict(Counter(cluster_ids.tolist())),
    }

    valid_mask = cluster_ids != -1
    if valid_mask.sum() > 2 and len(set(cluster_ids[valid_mask])) > 1:
        try:
            summary["silhouette_score"] = float(silhouette_score(Xn[valid_mask], cluster_ids[valid_mask]))
        except Exception:
            summary["silhouette_score"] = None
    else:
        summary["silhouette_score"] = None

    return df_emb, summary


def add_pca_projection(df_emb: pd.DataFrame):
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    X = df_emb[emb_cols].to_numpy(dtype=np.float32)

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)

    df_emb["pca_x"] = xy[:, 0]
    df_emb["pca_y"] = xy[:, 1]

    pca_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    return df_emb, pca_info


def plot_scatter(df_emb: pd.DataFrame, color_col: str, output_png: str, title: str):
    plt.figure(figsize=(10, 8))

    unique_vals = sorted(df_emb[color_col].astype(str).unique().tolist())
    for val in unique_vals:
        sub = df_emb[df_emb[color_col].astype(str) == str(val)]
        plt.scatter(
            sub["pca_x"],
            sub["pca_y"],
            s=18,
            alpha=0.8,
            label=str(val),
        )

    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(markerscale=1.5, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def save_summary_json(path: str, summary: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze feature-space clustering and prototype decisions")

    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--prototypes_csv", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--input_type", choices=["censo", "refined"], default="censo")

    parser.add_argument("--label_column", default="NOMBRE_COMUN")
    parser.add_argument("--x_column", default="COORDENADA_ESTE")
    parser.add_argument("--y_column", default="COORDENADA_NORTE")
    parser.add_argument("--image_column", default="matched_tif")
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

    if args.input_type == "censo":
        df_points = load_points_from_censo_csv(
            csv_path=args.input_csv,
            label_column=args.label_column,
            x_column=args.x_column,
            y_column=args.y_column,
            image_column=args.image_column,
            filter_label=args.filter_label,
        )
    else:
        df_points = load_points_from_refined_csv(
            csv_path=args.input_csv,
            label_column="target_label",
        )
        if args.filter_label:
            df_points = df_points[df_points["label"].str.lower() == args.filter_label.strip().lower()].copy()

    print(f"[INFO] Points loaded for analysis: {len(df_points)}")
    if len(df_points) == 0:
        raise RuntimeError("No points available after filtering.")

    df_emb = encode_points(df_points, encoder, patch_extractor, args.batch_size)
    df_emb = assign_prototypes(df_emb, prototypes)
    df_emb, dbscan_summary = run_dbscan(df_emb, args.dbscan_eps, args.dbscan_min_samples)
    df_emb, pca_info = add_pca_projection(df_emb)

    emb_csv = os.path.join(args.output_dir, "feature_space_embeddings.csv")
    df_emb.to_csv(emb_csv, index=False)

    plot_scatter(
        df_emb,
        color_col="label",
        output_png=os.path.join(args.output_dir, "pca_by_true_label.png"),
        title="Feature-space PCA colored by true label",
    )
    plot_scatter(
        df_emb,
        color_col="nearest_prototype_label",
        output_png=os.path.join(args.output_dir, "pca_by_nearest_prototype.png"),
        title="Feature-space PCA colored by nearest prototype label",
    )
    plot_scatter(
        df_emb,
        color_col="dbscan_cluster",
        output_png=os.path.join(args.output_dir, "pca_by_dbscan_cluster.png"),
        title="Feature-space PCA colored by DBSCAN cluster",
    )

    summary = {
        "num_points": int(len(df_emb)),
        "true_label_counts": dict(df_emb["label"].value_counts()),
        "nearest_prototype_label_counts": dict(df_emb["nearest_prototype_label"].value_counts()),
        "prototype_match_rate": float(df_emb["prototype_match"].mean()),
        "dbscan": dbscan_summary,
        "pca": pca_info,
    }

    save_summary_json(os.path.join(args.output_dir, "feature_space_summary.json"), summary)

    print("=" * 100)
    print("Feature-space analysis completed")
    print(f"Saved embeddings CSV : {emb_csv}")
    print(f"Prototype match rate : {summary['prototype_match_rate']:.4f}")
    print(f"DBSCAN clusters      : {dbscan_summary['num_clusters_excluding_noise']}")
    print(f"Noise points         : {dbscan_summary['num_noise_points']}")
    print(f"PCA variance (2D)    : {pca_info['explained_variance_ratio_sum']:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()