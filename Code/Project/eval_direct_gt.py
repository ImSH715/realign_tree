"""
Evaluate refinement outputs when ground truth is available.

This script is designed for:
1. direct GT evaluation, where the input points are already ground-truth points,
2. recovery evaluation, where the input points are perturbed versions of ground truth.

It computes:
- distance before refinement,
- distance after refinement,
- movement distance,
- improved / unchanged / worse counts,
- and threshold accuracies.

The input CSV must contain world-coordinate columns for original, GT, and refined points.
"""

import os
import argparse
import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame):
    required = [
        "original_east",
        "original_north",
        "gt_east",
        "gt_north",
        "refined_east",
        "refined_north",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    before = np.sqrt(
        (df["original_east"] - df["gt_east"]) ** 2 +
        (df["original_north"] - df["gt_north"]) ** 2
    )

    after = np.sqrt(
        (df["refined_east"] - df["gt_east"]) ** 2 +
        (df["refined_north"] - df["gt_north"]) ** 2
    )

    moved = np.sqrt(
        (df["refined_east"] - df["original_east"]) ** 2 +
        (df["refined_north"] - df["original_north"]) ** 2
    )

    improved = (after < before).sum()
    unchanged = (after == before).sum()
    worse = (after > before).sum()

    summary = {
        "n": int(len(df)),
        "mean_before_m": float(before.mean()),
        "mean_after_m": float(after.mean()),
        "median_before_m": float(np.median(before)),
        "median_after_m": float(np.median(after)),
        "mean_movement_m": float(moved.mean()),
        "improved": int(improved),
        "unchanged": int(unchanged),
        "worse": int(worse),
    }

    for th in [0.5, 1, 2, 5, 10]:
        summary[f"acc_before_{th}m"] = float((before <= th).mean())
        summary[f"acc_after_{th}m"] = float((after <= th).mean())

    df["distance_before_m"] = before
    df["distance_after_m"] = after
    df["movement_m"] = moved
    df["evaluation"] = np.where(after < before, "improved",
                         np.where(after > before, "worse", "unchanged"))

    return df, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate direct GT / recovery refinement output.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(args.input_csv)

    df = pd.read_csv(args.input_csv)
    df_out, summary = summarize(df)
    df_out.to_csv(args.output_csv, index=False)

    print("=" * 100)
    print("Direct GT / Recovery evaluation summary")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=" * 100)
    print(f"Saved evaluated CSV: {args.output_csv}")


if __name__ == "__main__":
    main()