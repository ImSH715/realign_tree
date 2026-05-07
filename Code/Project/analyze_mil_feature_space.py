"""
PCA analysis for the MIL feature space.

This script loads a trained MIL output directory, replays the validation or
training bags, extracts one embedding per candidate crop, projects the
candidate features to PCA space, and writes plots/CSVs for inspection.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.checkpoint import load_encoder_from_checkpoint
from train_mil_classifier import (
    MILPointBagDataset,
    Config,
    forward_bags,
    pool_instance_logits,
)
from train_supervised_encoder import (
    build_eval_transform,
    build_tif_index,
    forward_features,
    infer_feature_dim,
    safe_mkdir,
)


def load_config(output_dir: Path) -> dict:
    cfg_path = output_dir / "mil_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing MIL config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_runtime_config(raw: dict, output_dir: Path, split: str) -> Config:
    cfg = Config(**raw)
    cfg.output_dir = str(output_dir)
    if split == "train":
        cfg.eval_image_mode = cfg.image_mode
    return cfg


def load_head(head_path: Path, feat_dim: int, device):
    state = torch.load(head_path, map_location=device)
    head = nn.Linear(feat_dim, 1).to(device)
    head.load_state_dict(state["head_state_dict"])
    head.eval()
    return head, state


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def selected_status(row) -> str:
    if int(row["y_true"]) == 1 and int(row["y_pred"]) == 1:
        return "TP"
    if int(row["y_true"]) == 0 and int(row["y_pred"]) == 0:
        return "TN"
    if int(row["y_true"]) == 0 and int(row["y_pred"]) == 1:
        return "FP"
    return "FN"


def scatter_by_category(df, color_col, output_png, title, alpha=0.8, size=18):
    plt.figure(figsize=(9, 7))
    values = sorted(df[color_col].astype(str).unique().tolist())
    for val in values:
        sub = df[df[color_col].astype(str) == val]
        plt.scatter(sub["pca_x"], sub["pca_y"], s=size, alpha=alpha, label=val)
    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def scatter_by_value(df, color_col, output_png, title, size=18):
    plt.figure(figsize=(9, 7))
    points = plt.scatter(
        df["pca_x"],
        df["pca_y"],
        c=df[color_col].astype(float),
        s=size,
        alpha=0.85,
        cmap="viridis",
    )
    plt.colorbar(points, label=color_col)
    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_selected_overlay(df, output_png):
    plt.figure(figsize=(9, 7))
    rest = df[df["is_selected"] == 0]
    selected = df[df["is_selected"] == 1]
    plt.scatter(rest["pca_x"], rest["pca_y"], s=8, alpha=0.18, color="#98a2b3", label="candidate")
    for label, color in [("0", "#2563eb"), ("1", "#dc2626")]:
        sub = selected[selected["y_true"].astype(str) == label]
        plt.scatter(sub["pca_x"], sub["pca_y"], s=35, alpha=0.9, label=f"selected true {label}", color=color)
    plt.title("MIL Candidate PCA With Selected Corrections Highlighted")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


@torch.no_grad()
def extract_rows(model, head, dataset, loader, device, cfg, save_embeddings=False):
    rows = []
    embeddings = []

    model.eval()
    head.eval()

    for bags, y, idxs, offsets in tqdm(loader, desc="Extract MIL features", dynamic_ncols=True):
        bags = bags.to(device, non_blocking=True)
        b, n, c, h, w = bags.shape
        flat = bags.reshape(b * n, c, h, w)
        feats = forward_features(model, flat)
        instance_logits = head(feats).view(b, n)
        bag_logits = pool_instance_logits(instance_logits, cfg.pooling, cfg.lse_tau, cfg.topk)

        instance_probs = torch.sigmoid(instance_logits).detach().cpu().numpy()
        bag_probs = torch.sigmoid(bag_logits).detach().cpu().numpy()
        bag_preds = (bag_probs >= 0.5).astype(np.int64)
        best_instances = instance_logits.argmax(dim=1).detach().cpu().numpy()
        feat_np = feats.detach().cpu().numpy().reshape(b, n, -1)

        y_np = y.detach().cpu().numpy().astype(np.int64)
        idx_np = idxs.detach().cpu().numpy().astype(np.int64)
        offsets_np = offsets.detach().cpu().numpy()

        for j in range(b):
            sample = dataset.samples[int(idx_np[j])]
            best_i = int(best_instances[j])
            for i in range(n):
                dx_px = float(offsets_np[j, i, 0])
                dy_px = float(offsets_np[j, i, 1])
                dx_m = dx_px * sample["pixel_size_x"]
                dy_m = dy_px * sample["pixel_size_y"]
                row = {
                    "bag_index": int(idx_np[j]),
                    "source_index": sample["source_index"],
                    "instance_index": int(i),
                    "is_selected": int(i == best_i),
                    "y_true": int(y_np[j]),
                    "y_pred": int(bag_preds[j]),
                    "status": selected_status({"y_true": int(y_np[j]), "y_pred": int(bag_preds[j])}),
                    "bag_prob_1": float(bag_probs[j]),
                    "instance_prob_1": float(instance_probs[j, i]),
                    "selected_instance_prob_1": float(instance_probs[j, best_i]),
                    "selected_instance": best_i,
                    "dx_px": dx_px,
                    "dy_px": dy_px,
                    "dx_m": dx_m,
                    "dy_m": dy_m,
                    "offset_m": float((dx_m ** 2 + dy_m ** 2) ** 0.5),
                    "px": sample["center_px"] + dx_px,
                    "py": sample["center_py"] + dy_px,
                    "center_px": sample["center_px"],
                    "center_py": sample["center_py"],
                    "folder": sample["folder"],
                    "file": sample["file"],
                    "image_path": sample["image_path"],
                    "label": sample["label"],
                    "patch_size_px": int(cfg.patch_size_px),
                    "bag_radius_m": float(cfg.bag_radius_m),
                    "image_mode": str(cfg.eval_image_mode),
                }
                if save_embeddings:
                    for k, v in enumerate(feat_np[j, i]):
                        row[f"emb_{k}"] = float(v)
                rows.append(row)
                embeddings.append(feat_np[j, i])

    return pd.DataFrame(rows), np.asarray(embeddings, dtype=np.float32)


def summarize(df_instances: pd.DataFrame, pca, output_dir: Path):
    selected = df_instances[df_instances["is_selected"] == 1].copy()
    summary = {
        "num_instances": int(len(df_instances)),
        "num_bags": int(len(selected)),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    if len(selected) and len(selected["y_true"].unique()) == 2:
        summary["selected_average_precision"] = float(
            average_precision_score(selected["y_true"], selected["bag_prob_1"])
        )
        summary["selected_roc_auc"] = float(roc_auc_score(selected["y_true"], selected["bag_prob_1"]))
        summary["selected_instance_average_precision"] = float(
            average_precision_score(selected["y_true"], selected["selected_instance_prob_1"])
        )
        summary["selected_instance_roc_auc"] = float(
            roc_auc_score(selected["y_true"], selected["selected_instance_prob_1"])
        )

    summary["selected_status_counts"] = selected["status"].value_counts().to_dict()
    summary["selected_offset_m_by_true_class"] = json.loads(
        selected.groupby("y_true")["offset_m"].describe().to_json(orient="index")
    )

    with (output_dir / "mil_pca_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="PCA analysis for MIL candidate features.")
    p.add_argument("--mil_output_dir", required=True)
    p.add_argument("--output_dir", default="")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--checkpoint_name", default="best", choices=["best", "last"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_embeddings", action="store_true")
    p.add_argument("--no_normalize_features", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    mil_output_dir = Path(args.mil_output_dir)
    output_dir = Path(args.output_dir) if args.output_dir else mil_output_dir / f"pca_{args.split}_{args.checkpoint_name}"
    safe_mkdir(output_dir)

    raw_cfg = load_config(mil_output_dir)
    cfg = make_runtime_config(raw_cfg, output_dir, args.split)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    encoder_path = mil_output_dir / f"phase1_encoder_{args.checkpoint_name}.pth"
    head_path = mil_output_dir / f"mil_head_{args.checkpoint_name}.pth"
    if not encoder_path.exists():
        raise FileNotFoundError(encoder_path)
    if not head_path.exists():
        raise FileNotFoundError(head_path)

    model, _ = load_encoder_from_checkpoint(str(encoder_path), device)
    feat_dim = infer_feature_dim(model, device, cfg.image_size)
    head, _ = load_head(head_path, feat_dim, device)

    folder_to_paths = build_tif_index(cfg.imagery_root)
    shp_path = cfg.val_shp if args.split == "val" else cfg.train_shp
    image_mode = cfg.eval_image_mode if args.split == "val" else cfg.image_mode

    dataset = MILPointBagDataset(
        shp_path=shp_path,
        imagery_root=cfg.imagery_root,
        label_field=cfg.label_field,
        folder_field=cfg.folder_field,
        file_field=cfg.file_field,
        fx_field=cfg.fx_field,
        fy_field=cfg.fy_field,
        coord_mode=cfg.coord_mode,
        positive_class=cfg.positive_class,
        patch_size_px=cfg.patch_size_px,
        transform=build_eval_transform(cfg.image_size, image_mode),
        bag_radius_m=cfg.bag_radius_m,
        negative_bag_radius_m=cfg.negative_bag_radius_m,
        bag_instances=cfg.bag_instances,
        max_black_fraction=cfg.max_black_fraction,
        max_bright_fraction=cfg.max_bright_fraction,
        folder_to_paths=folder_to_paths,
        repeat_factor=1,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    df_instances, feats = extract_rows(
        model=model,
        head=head,
        dataset=dataset,
        loader=loader,
        device=device,
        cfg=cfg,
        save_embeddings=args.save_embeddings,
    )

    feats_for_pca = feats if args.no_normalize_features else l2_normalize(feats)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(feats_for_pca)
    df_instances["pca_x"] = xy[:, 0]
    df_instances["pca_y"] = xy[:, 1]
    df_instances["is_positive_instance"] = df_instances["instance_prob_1"] >= 0.5

    selected = df_instances[df_instances["is_selected"] == 1].copy()

    instance_csv = output_dir / "mil_instance_pca.csv"
    selected_csv = output_dir / "mil_selected_pca.csv"
    df_instances.to_csv(instance_csv, index=False)
    selected.to_csv(selected_csv, index=False)

    scatter_by_category(
        selected,
        "y_true",
        output_dir / "pca_selected_by_true_label.png",
        "Selected MIL Corrections By True Bag Label",
        size=35,
    )
    scatter_by_category(
        selected,
        "status",
        output_dir / "pca_selected_by_status.png",
        "Selected MIL Corrections By Prediction Status",
        size=35,
    )
    scatter_by_value(
        selected,
        "bag_prob_1",
        output_dir / "pca_selected_by_bag_prob.png",
        "Selected MIL Corrections By Bag Probability",
        size=35,
    )
    scatter_by_value(
        selected,
        "selected_instance_prob_1",
        output_dir / "pca_selected_by_instance_prob.png",
        "Selected MIL Corrections By Selected Instance Probability",
        size=35,
    )
    scatter_by_category(
        df_instances,
        "y_true",
        output_dir / "pca_all_instances_by_true_label.png",
        "All MIL Candidate Instances By True Bag Label",
        alpha=0.35,
        size=8,
    )
    plot_selected_overlay(df_instances, output_dir / "pca_all_instances_selected_overlay.png")

    summary = summarize(df_instances, pca, output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Instance PCA CSV : {instance_csv}")
    print(f"Selected PCA CSV : {selected_csv}")
    print(f"Output directory : {output_dir}")


if __name__ == "__main__":
    main()
