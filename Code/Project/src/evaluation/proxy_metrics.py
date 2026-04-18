"""
Proxy evaluation utilities for no-ground-truth data.

This module computes indirect evaluation metrics when exact answers are unavailable.
It supports:
- single-run confidence and movement analysis,
- repeated-run stability analysis,
- and summary statistics for model behavior.

These metrics do not measure true accuracy, but they help assess reliability,
consistency, and confidence of the refinement outputs.
"""

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


def evaluate_single_run(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    required = ["point_id", "best_similarity", "refined_east", "refined_north", "original_east", "original_north"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    movement = np.sqrt(
        (df["refined_east"] - df["original_east"]) ** 2 +
        (df["refined_north"] - df["original_north"]) ** 2
    )

    out = df.copy()
    out["movement_m"] = movement

    summary = {
        "n": int(len(out)),
        "mean_best_similarity": float(out["best_similarity"].mean()),
        "median_best_similarity": float(out["best_similarity"].median()),
        "mean_movement_m": float(out["movement_m"].mean()),
        "median_movement_m": float(out["movement_m"].median()),
        "high_confidence_ratio_0.8": float((out["best_similarity"] >= 0.8).mean()),
        "high_confidence_ratio_0.9": float((out["best_similarity"] >= 0.9).mean()),
    }

    return out, summary


def evaluate_stability(csv_paths: List[str]) -> Tuple[pd.DataFrame, Dict]:
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