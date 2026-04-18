"""
Direct ground-truth evaluation utilities.

This module evaluates refinement outputs when exact ground-truth coordinates are available.
It supports:
- direct GT evaluation, where original points are already ground truth,
- recovery evaluation, where original points are perturbed from GT.

It computes:
- distance before refinement,
- distance after refinement,
- movement distance,
- improved / unchanged / worse classification,
- and threshold-based accuracies.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def evaluate_against_gt(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
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

    out = df.copy()
    out["distance_before_m"] = before
    out["distance_after_m"] = after
    out["movement_m"] = moved
    out["evaluation"] = np.where(
        after < before,
        "improved",
        np.where(after > before, "worse", "unchanged"),
    )

    summary = {
        "n": int(len(out)),
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

    return out, summary