"""
Split ground-truth SHP into train/val/test sets.

Recommended split is group-based by File or Folder, so nearby points from the
same orthomosaic tile do not leak across train and test.
"""

import argparse
import os

import geopandas as gpd
from sklearn.model_selection import GroupShuffleSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shp", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_field", default="Tree")
    parser.add_argument("--group_field", default="File")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gdf = gpd.read_file(args.input_shp)

    if args.label_field not in gdf.columns:
        raise ValueError(f"Missing label field: {args.label_field}")

    if args.group_field not in gdf.columns:
        raise ValueError(f"Missing group field: {args.group_field}")

    gdf = gdf[gdf[args.label_field].notna()].copy()
    gdf[args.label_field] = gdf[args.label_field].astype(str).str.strip()
    gdf[args.group_field] = gdf[args.group_field].astype(str).str.strip()

    groups = gdf[args.group_field].values

    splitter1 = GroupShuffleSplit(
        n_splits=1,
        train_size=args.train_ratio,
        random_state=args.seed,
    )

    train_idx, temp_idx = next(splitter1.split(gdf, groups=groups))

    train_gdf = gdf.iloc[train_idx].copy()
    temp_gdf = gdf.iloc[temp_idx].copy()

    temp_groups = temp_gdf[args.group_field].values
    val_fraction_of_temp = args.val_ratio / (args.val_ratio + args.test_ratio)

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_fraction_of_temp,
        random_state=args.seed,
    )

    val_idx, test_idx = next(splitter2.split(temp_gdf, groups=temp_groups))

    val_gdf = temp_gdf.iloc[val_idx].copy()
    test_gdf = temp_gdf.iloc[test_idx].copy()

    train_path = os.path.join(args.output_dir, "valid_points_train.shp")
    val_path = os.path.join(args.output_dir, "valid_points_val.shp")
    test_path = os.path.join(args.output_dir, "valid_points_test.shp")

    train_gdf.to_file(train_path)
    val_gdf.to_file(val_path)
    test_gdf.to_file(test_path)

    print("=" * 80)
    print("GT split completed")
    print(f"Train: {len(train_gdf)} -> {train_path}")
    print(f"Val  : {len(val_gdf)} -> {val_path}")
    print(f"Test : {len(test_gdf)} -> {test_path}")
    print("=" * 80)

    print("\nTrain label counts:")
    print(train_gdf[args.label_field].value_counts())

    print("\nVal label counts:")
    print(val_gdf[args.label_field].value_counts())

    print("\nTest label counts:")
    print(test_gdf[args.label_field].value_counts())


if __name__ == "__main__":
    main()