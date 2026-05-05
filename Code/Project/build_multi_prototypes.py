import argparse
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--label_col", default="label")
    p.add_argument("--k_other", type=int, default=5)
    p.add_argument("--k_positive", type=int, default=1)
    p.add_argument("--positive_label", default="1")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = pd.read_csv(args.embedding_csv)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    rows = []

    for label, group in df.groupby(args.label_col):
        X = group[emb_cols].values.astype(np.float32)

        if str(label) == str(args.positive_label):
            k = args.k_positive
        else:
            k = args.k_other

        k = min(k, len(group))

        if k <= 1:
            centers = X.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
            km.fit(X)
            centers = km.cluster_centers_

        for i, c in enumerate(centers):
            row = {
                "prototype_id": f"{label}_proto_{i}",
                "label": label,
                "cluster_id": i,
                "n_source": len(group),
            }
            for j, v in enumerate(c):
                row[f"emb_{j}"] = float(v)
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)

    print("Saved:", args.output_csv)
    print(out[["prototype_id", "label", "cluster_id", "n_source"]])


if __name__ == "__main__":
    main()