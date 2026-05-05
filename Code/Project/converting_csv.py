import argparse
import os
import geopandas as gpd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_shp", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--target_crs", default="EPSG:32718")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    gdf = gpd.read_file(args.input_shp)

    print("Original CRS:", gdf.crs)

    if gdf.crs is None:
        raise ValueError(
            "Input SHP has no CRS. You must define the source CRS first before converting."
        )

    gdf = gdf.to_crs(args.target_crs)

    print("Converted CRS:", gdf.crs)

    gdf["x_epsg32718"] = gdf.geometry.x
    gdf["y_epsg32718"] = gdf.geometry.y

    df = gdf.drop(columns="geometry")

    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("Saved CSV:", args.output_csv)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df[["x_epsg32718", "y_epsg32718"]].head())


if __name__ == "__main__":
    main()
