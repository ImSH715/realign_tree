"""
Feature-space clustering and prototype-assignment utilities.

This module contains reusable logic for:
- cosine similarity against class prototypes,
- nearest prototype assignment,
- DBSCAN clustering in normalized embedding space,
- PCA projection to 2D,
- and summary statistics about cluster structure.

It is intended for evaluation and interpretation of learned feature representations.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def nearest_prototype_label(embedding: np.ndarray, prototypes: Dict[str, np.ndarray]):
    best_label = None
    best_sim = -1e9
    for label, proto in prototypes.items():
        sim = cosine_similarity(embedding, proto)
        if sim > best_sim:
            best_sim = sim
            best_label = label
    return best_label, float(best_sim)


def assign_prototypes(df_emb: pd.DataFrame, prototypes: Dict[str, np.ndarray]) -> pd.DataFrame:
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]

    pred_labels = []
    pred_sims = []

    for _, row in df_emb.iterrows():
        emb = row[emb_cols].to_numpy(dtype=np.float32)
        label, sim = nearest_prototype_label(emb, prototypes)
        pred_labels.append(label)
        pred_sims.append(sim)

    out = df_emb.copy()
    out["nearest_prototype_label"] = pred_labels
    out["nearest_prototype_similarity"] = pred_sims
    if "label" in out.columns:
        out["prototype_match"] = (out["label"] == out["nearest_prototype_label"])
    return out


def run_dbscan(df_emb: pd.DataFrame, eps: float, min_samples: int) -> Tuple[pd.DataFrame, Dict]:
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    X = df_emb[emb_cols].to_numpy(dtype=np.float32)

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_ids = db.fit_predict(Xn)

    out = df_emb.copy()
    out["dbscan_cluster"] = cluster_ids

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

    return out, summary


def add_pca_projection(df_emb: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    X = df_emb[emb_cols].to_numpy(dtype=np.float32)

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)

    out = df_emb.copy()
    out["pca_x"] = xy[:, 0]
    out["pca_y"] = xy[:, 1]

    summary = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }

    return out, summary