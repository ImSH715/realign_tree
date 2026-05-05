import argparse
import os
import geopandas as gpd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_shp", required=True)
    p.add_argument("--output_shp", required=True)
    p.add_argument("--label_field", default="Tree")
    p.add_argument("--target_label", default="Shihuahuaco")
    p.add_argument("--binary_field", default="BinaryTree")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_shp), exist_ok=True)

    gdf = gpd.read_file(args.input_shp)
    gdf = gdf[gdf[args.label_field].notna()].copy()
    gdf[args.label_field] = gdf[args.label_field].astype(str).str.strip()

    target = args.target_label.strip().lower()

    gdf[args.binary_field] = gdf[args.label_field].apply(
        lambda x: args.target_label if str(x).strip().lower() == target else "Other"
    )

    print(gdf[args.binary_field].value_counts())

    gdf.to_file(args.output_shp)
    print("Saved:", args.output_shp)


if __name__ == "__main__":
    main()
