import argparse
import os
import geopandas as gpd
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_shp", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label_field", default="Tree")
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--min_class_count", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gdf = gpd.read_file(args.input_shp)
    gdf = gdf[gdf[args.label_field].notna()].copy()
    gdf[args.label_field] = gdf[args.label_field].astype(str).str.strip()

    counts = gdf[args.label_field].value_counts()
    print("Original class counts:")
    print(counts)

    valid_classes = counts[counts >= args.min_class_count].index.tolist()
    dropped = counts[counts < args.min_class_count]

    if len(dropped) > 0:
        print("Dropping rare classes:")
        print(dropped)

    gdf = gdf[gdf[args.label_field].isin(valid_classes)].copy()

    train_gdf, temp_gdf = train_test_split(
        gdf,
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=gdf[args.label_field],
    )

    val_fraction = args.val_ratio / (args.val_ratio + args.test_ratio)

    temp_counts = temp_gdf[args.label_field].value_counts()
    if (temp_counts < 2).any():
        print("Some classes have <2 samples in temp split. Falling back to non-stratified val/test split.")
        val_gdf, test_gdf = train_test_split(
            temp_gdf,
            train_size=val_fraction,
            random_state=args.seed,
            stratify=None,
        )
    else:
        val_gdf, test_gdf = train_test_split(
            temp_gdf,
            train_size=val_fraction,
            random_state=args.seed,
            stratify=temp_gdf[args.label_field],
        )

    base = os.path.splitext(os.path.basename(args.input_shp))[0]

    train_gdf.to_file(os.path.join(args.output_dir, f"{base}_train.shp"))
    val_gdf.to_file(os.path.join(args.output_dir, f"{base}_val.shp"))
    test_gdf.to_file(os.path.join(args.output_dir, f"{base}_test.shp"))

    print("Saved to:", args.output_dir)
    print("Train:", len(train_gdf))
    print("Val:", len(val_gdf))
    print("Test:", len(test_gdf))

    print("Train counts:")
    print(train_gdf[args.label_field].value_counts())

    print("Val counts:")
    print(val_gdf[args.label_field].value_counts())

    print("Test counts:")
    print(test_gdf[args.label_field].value_counts())


if __name__ == "__main__":
    main()
