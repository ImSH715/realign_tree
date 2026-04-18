"""
Visualization helpers for evaluation and feature-space inspection.

This module provides plotting utilities for:
- PCA scatter plots colored by label, prototype assignment, or cluster ID,
- saving figures to disk,
- and lightweight visual diagnostics for representation analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(df_emb: pd.DataFrame, color_col: str, output_png: str, title: str):
    plt.figure(figsize=(10, 8))

    values = sorted(df_emb[color_col].astype(str).unique().tolist())
    for val in values:
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
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()