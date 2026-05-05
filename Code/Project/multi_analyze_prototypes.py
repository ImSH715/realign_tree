"""
Analyze multi-prototype structure before Phase 3.

Works with:
- multi_class_prototypes.csv
  columns: prototype_id,label,cluster_id,n_source,emb_0...
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def label_name(x):
    x = str(x)
    if x == "0":
        return "Other"
    if x == "1":
        return "Shihuahuaco"
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prototypes_csv",
        default="./outputs/phase2_binary_shihuahuaco/multi_class_prototypes.csv",
    )
    p.add_argument(
        "--output_dir",
        default="./outputs/phase2_binary_shihuahuaco/analysis_multi_prototypes",
    )
    p.add_argument("--positive_label", default="1")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.prototypes_csv)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    if not emb_cols:
        raise ValueError("No emb_* columns found.")

    X = df[emb_cols].values.astype(np.float32)
    labels = df["label"].astype(str).values
    proto_ids = df["prototype_id"].astype(str).values

    sim = cosine_similarity(X)

    print("=" * 80)
    print("Multi-prototype analysis")
    print("Prototype CSV:", args.prototypes_csv)
    print("Number of prototypes:", len(df))
    print("=" * 80)

    print("\nPrototype counts by label:")
    print(df.groupby("label")["prototype_id"].count())

    print("\nCluster counts by label:")
    print(df.groupby("label")["cluster_id"].nunique())

    # Save similarity matrix
    display_ids = [f"{pid}({label_name(lbl)})" for pid, lbl in zip(proto_ids, labels)]
    sim_df = pd.DataFrame(sim, index=display_ids, columns=display_ids)
    sim_df.to_csv(os.path.join(args.output_dir, "prototype_similarity_matrix.csv"))

    print("\nTop nearest prototype for each prototype:")
    nearest_rows = []

    for i in range(len(proto_ids)):
        sims = sim[i].copy()
        sims[i] = -1

        j = int(np.argmax(sims))

        row = {
            "prototype_id": proto_ids[i],
            "label": labels[i],
            "label_name": label_name(labels[i]),
            "nearest_prototype_id": proto_ids[j],
            "nearest_label": labels[j],
            "nearest_label_name": label_name(labels[j]),
            "similarity": float(sims[j]),
            "same_label": labels[i] == labels[j],
        }
        nearest_rows.append(row)

        print(
            f"{proto_ids[i]} ({label_name(labels[i])}) "
            f"-> {proto_ids[j]} ({label_name(labels[j])}) "
            f"sim={sims[j]:.4f}"
        )

    nearest_df = pd.DataFrame(nearest_rows)
    nearest_df.to_csv(os.path.join(args.output_dir, "nearest_prototypes.csv"), index=False)

    # Cross-class separation
    pos_mask = labels == str(args.positive_label)
    neg_mask = ~pos_mask

    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        cross_sim = cosine_similarity(X[pos_mask], X[neg_mask])

        print("\n" + "=" * 80)
        print("Positive vs Other separation")
        print("Positive label:", args.positive_label, "=", label_name(args.positive_label))
        print(f"Max similarity : {cross_sim.max():.4f}")
        print(f"Mean similarity: {cross_sim.mean():.4f}")
        print(f"Min similarity : {cross_sim.min():.4f}")
        print(f"Distance max-based: {1.0 - cross_sim.max():.4f}")
        print(f"Distance mean-based: {1.0 - cross_sim.mean():.4f}")

        pd.DataFrame(cross_sim).to_csv(
            os.path.join(args.output_dir, "positive_vs_other_similarity.csv"),
            index=False,
        )

        if cross_sim.max() > 0.95:
            print("Interpretation: very weak separation. Prototype Phase 3 may be unreliable.")
        elif cross_sim.max() > 0.85:
            print("Interpretation: moderate separation. Some confusion expected.")
        else:
            print("Interpretation: reasonably separated.")

    # PCA plot
    if len(df) >= 2:
        X2 = PCA(n_components=2).fit_transform(X)

        plot_df = pd.DataFrame({
            "pc1": X2[:, 0],
            "pc2": X2[:, 1],
            "label": labels,
            "label_name": [label_name(x) for x in labels],
            "prototype_id": proto_ids,
        })
        plot_df.to_csv(os.path.join(args.output_dir, "prototype_pca_points.csv"), index=False)

        plt.figure(figsize=(8, 6))
        for lab in sorted(set(labels)):
            idx = labels == lab
            plt.scatter(X2[idx, 0], X2[idx, 1], label=f"{lab}={label_name(lab)}")

            for x, y, pid in zip(X2[idx, 0], X2[idx, 1], proto_ids[idx]):
                plt.text(x, y, pid, fontsize=8)

        plt.title("Multi-prototype PCA")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(args.output_dir, "prototype_pca.png")
        plt.savefig(out_png, dpi=200)
        print("\nSaved PCA plot:", out_png)

    print("\nSaved analysis to:", args.output_dir)


if __name__ == "__main__":
    main()