"""
Localization recovery evaluation utilities.

This module focuses on recovery-style evaluation:
- ground-truth points are perturbed,
- the model refines them,
- and recovery success is measured against the original GT coordinates.

It provides helpers for:
- summarizing recovery accuracy,
- grouping performance by perturbation radius,
- and computing recovery rates under distance thresholds.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def summarize_recovery(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    required = ["gt_east", "gt_north", "original_east", "original_north", "refined_east", "refined_north"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    initial_offset = np.sqrt(
        (df["original_east"] - df["gt_east"]) ** 2 +
        (df["original_north"] - df["gt_north"]) ** 2
    )

    final_error = np.sqrt(
        (df["refined_east"] - df["gt_east"]) ** 2 +
        (df["refined_north"] - df["gt_north"]) ** 2
    )

    recovered = final_error < initial_offset

    out = df.copy()
    out["initial_offset_m"] = initial_offset
    out["final_error_m"] = final_error
    out["recovered"] = recovered

    summary = {
        "n": int(len(out)),
        "mean_initial_offset_m": float(initial_offset.mean()),
        "mean_final_error_m": float(final_error.mean()),
        "median_initial_offset_m": float(np.median(initial_offset)),
        "median_final_error_m": float(np.median(final_error)),
        "recovery_rate": float(recovered.mean()),
    }

    for th in [0.5, 1, 2, 5, 10]:
        summary[f"recovered_within_{th}m"] = float((final_error <= th).mean())

    return out, summary


def summarize_recovery_by_offset_bin(df: pd.DataFrame, bins=None) -> pd.DataFrame:
    if bins is None:
        bins = [0, 1, 2, 5, 10, 20, 50, 100]

    required = ["initial_offset_m", "final_error_m"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    out = df.copy()
    out["offset_bin"] = pd.cut(out["initial_offset_m"], bins=bins, include_lowest=True)

    grouped = out.groupby("offset_bin", observed=False).agg(
        count=("initial_offset_m", "size"),
        mean_initial_offset_m=("initial_offset_m", "mean"),
        mean_final_error_m=("final_error_m", "mean"),
        median_final_error_m=("final_error_m", "median"),
    ).reset_index()

    return grouped