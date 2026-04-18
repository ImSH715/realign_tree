"""
Evaluate refinement outputs when no ground truth is available.

This script computes proxy metrics that can be used to assess model behavior
without direct answer labels. It supports:

1. Single-run proxy evaluation:
   - mean / median best similarity,
   - mean / median movement distance,
   - high-confidence ratios.

2. Multi-run stability evaluation:
   - spatial dispersion of refined outputs across repeated runs,
   - per-point refined coordinate standard deviation,
   - mean radial dispersion.

If only one CSV is provided, only single-run proxy metrics are computed.
If multiple CSV files are provided, stability metrics are also computed.
"""

import argparse
import numpy as np
import pandas as pd


def evaluate_single(df: pd.DataFrame):
    required = ["point_id", "best_similarity", "refined_east", "refined_north", "original_east", "original_north"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    movement = np.sqrt(
        (df["refined_east"] - df["original_east"]) ** 2 +
        (df["refined_north"] - df["original_north"]) ** 2
    )

    df["movement_m"] = movement

    summary = {
        "n": int(len(df)),
        "mean_best_similarity": float(df["best_similarity"].mean()),
        "median_best_similarity": float(df["best_similarity"].median()),
        "mean_movement_m": float(df["movement_m"].mean()),
        "median_movement_m": float(df["movement_m"].median()),
        "high_confidence_ratio_0.8": float((df["best_similarity"] >= 0.8).mean()),
        "high_confidence_ratio_0.9": float((df["best_similarity"] >= 0.9).mean()),
    }

    return df, summary


def evaluate_stability(csv_paths):
    runs = []
    for i, p in enumerate(csv_paths):
        df = pd.read_csv(p)
        df["run_id"] = i
        runs.append(df)

    all_df = pd.concat(runs, ignore_index=True)

    required = ["point_id", "refined_east", "refined_north"]
    for c in required:
        if c not in all_df.columns:
            raise ValueError(f"Missing required column '{c}' for stability analysis")

    grouped = all_df.groupby("point_id")
    stability_rows = []

    for pid, g in grouped:
        if len(g) < 2:
            continue

        x_std = g["refined_east"].std()
        y_std = g["refined_north"].std()
        radial_std = np.sqrt(
            (g["refined_east"] - g["refined_east"].mean()) ** 2 +
            (g["refined_north"] - g["refined_north"].mean()) ** 2
        ).mean()

        stability_rows.append(
            {
                "point_id": pid,
                "num_runs": len(g),
                "refined_east_std": float(x_std),
                "refined_north_std": float(y_std),
                "mean_radial_dispersion_m": float(radial_std),
            }
        )

    out = pd.DataFrame(stability_rows)

    summary = {
        "num_points_with_repeats": int(len(out)),
        "mean_refined_east_std": float(out["refined_east_std"].mean()) if len(out) > 0 else None,
        "mean_refined_north_std": float(out["refined_north_std"].mean()) if len(out) > 0 else None,
        "mean_radial_dispersion_m": float(out["mean_radial_dispersion_m"].mean()) if len(out) > 0 else None,
    }

    return out, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Proxy evaluation for no-GT refinement outputs.")
    parser.add_argument("--input_csv", nargs="+", required=True)
    parser.add_argument("--single_output_csv", required=True)
    parser.add_argument("--stability_output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    base_df = pd.read_csv(args.input_csv[0])
    df_single, summary_single = evaluate_single(base_df)
    df_single.to_csv(args.single_output_csv, index=False)

    print("=" * 100)
    print("Proxy evaluation summary (single run)")
    for k, v in summary_single.items():
        print(f"{k}: {v}")
    print("=" * 100)

    if len(args.input_csv) > 1:
        df_stab, summary_stab = evaluate_stability(args.input_csv)
        df_stab.to_csv(args.stability_output_csv, index=False)

        print("=" * 100)
        print("Stability evaluation summary (multi-run)")
        for k, v in summary_stab.items():
            print(f"{k}: {v}")
        print("=" * 100)
    else:
        pd.DataFrame().to_csv(args.stability_output_csv, index=False)
        print("[INFO] Only one input CSV provided. Stability analysis skipped.")


if __name__ == "__main__":
    main()