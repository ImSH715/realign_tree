import os
import numpy as np
import pandas as pd
import rasterio

INPUT_CSV = "outputs/phase3/refined_points_world.csv"
OUTPUT_CSV = "outputs/phase3/refined_points_world_evaluated.csv"


def pixel_to_world(image_path, x_col, y_row):
    with rasterio.open(image_path) as src:
        xw, yw = src.xy(float(y_row), float(x_col))
    return float(xw), float(yw)


def ensure_world_columns(df):
    cols = set(df.columns)

    # Reconstruct original world coordinates if missing
    if not {"original_east", "original_north"}.issubset(cols):
        if {"image_path", "original_x", "original_y"}.issubset(cols):
            original_east = []
            original_north = []
            for _, row in df.iterrows():
                xw, yw = pixel_to_world(row["image_path"], row["original_x"], row["original_y"])
                original_east.append(xw)
                original_north.append(yw)
            df["original_east"] = original_east
            df["original_north"] = original_north
        else:
            raise ValueError(
                "Missing original world coordinates and cannot reconstruct them. "
                "Need either original_east/original_north or image_path+original_x+original_y."
            )

    # Reconstruct refined world coordinates if missing
    if not {"refined_east", "refined_north"}.issubset(cols):
        if {"image_path", "refined_x", "refined_y"}.issubset(cols):
            refined_east = []
            refined_north = []
            for _, row in df.iterrows():
                xw, yw = pixel_to_world(row["image_path"], row["refined_x"], row["refined_y"])
                refined_east.append(xw)
                refined_north.append(yw)
            df["refined_east"] = refined_east
            df["refined_north"] = refined_north
        else:
            raise ValueError(
                "Missing refined world coordinates and cannot reconstruct them. "
                "Need either refined_east/refined_north or image_path+refined_x+refined_y."
            )

    return df


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print("Columns found:")
    print(df.columns.tolist())

    df = ensure_world_columns(df)

    # For the valid_points.shp experiment:
    # the original point is the ground-truth answer
    df["gt_east"] = df["original_east"]
    df["gt_north"] = df["original_north"]

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

    print("=" * 80)
    print("Evaluation summary")
    print(f"N: {len(df)}")
    print(f"Mean distance before: {before.mean():.4f}")
    print(f"Mean distance after : {after.mean():.4f}")
    print(f"Median distance before: {np.median(before):.4f}")
    print(f"Median distance after : {np.median(after):.4f}")
    print(f"Mean movement from original point: {moved.mean():.4f}")
    print(f"Improved: {int(improved)}")
    print(f"Unchanged: {int(unchanged)}")
    print(f"Worse: {int(worse)}")
    print("=" * 80)

    for th in [0.5, 1, 2, 5, 10]:
        acc_before = (before <= th).mean()
        acc_after = (after <= th).mean()
        print(f"Accuracy within {th} m - before: {acc_before:.3f}, after: {acc_after:.3f}")

    df["distance_before_m"] = before
    df["distance_after_m"] = after
    df["movement_m"] = moved
    df["evaluation"] = np.where(after < before, "improved",
                         np.where(after > before, "worse", "unchanged"))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved evaluated CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()