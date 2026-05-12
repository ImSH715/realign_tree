"""
Compare the full forest census CSV against curated crown and pipeline splits.

The full census table stores species and UTM coordinates. The curated crown
layer appears to be a spatial subset of that table with crown geometries. This
script compares records using UTM zone, easting, and northing so we can separate
"not in the original census" from "in the census but not in the image-backed
pipeline subset".
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def normalize_text(value):
    return str(value).strip()


def normalize_key_text(value):
    return normalize_text(value).lower()


def read_csv_flexible(path):
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def numeric_coord(series):
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def add_censo_fields(df, species_col, zone_col, east_col, north_col, prefix):
    out = df.copy()
    out[f"{prefix}_species"] = out[species_col].map(normalize_text)
    out[f"{prefix}_species_norm"] = out[species_col].map(normalize_key_text)
    out[f"{prefix}_zone"] = out[zone_col].map(normalize_text)
    out[f"{prefix}_east"] = numeric_coord(out[east_col])
    out[f"{prefix}_north"] = numeric_coord(out[north_col])
    out[f"{prefix}_coord_key"] = (
        out[f"{prefix}_zone"].astype(str)
        + "|"
        + out[f"{prefix}_east"].astype(str)
        + "|"
        + out[f"{prefix}_north"].astype(str)
    )
    return out


def read_curated(path, species_col, zone_col, east_col, north_col):
    gdf = gpd.read_file(path)
    return add_censo_fields(gdf, species_col, zone_col, east_col, north_col, "curated")


def read_pipeline_splits(pipeline_dir, target_crs, zone):
    frames = []
    for split in ["train", "val", "test"]:
        path = Path(pipeline_dir) / f"valid_points_{split}.shp"
        if not path.exists():
            continue
        gdf = gpd.read_file(path).to_crs(target_crs)
        gdf["split"] = split
        gdf["pipeline_zone"] = zone
        gdf["pipeline_east"] = gdf.geometry.x.round().astype("Int64")
        gdf["pipeline_north"] = gdf.geometry.y.round().astype("Int64")
        gdf["pipeline_coord_key"] = (
            gdf["pipeline_zone"].astype(str)
            + "|"
            + gdf["pipeline_east"].astype(str)
            + "|"
            + gdf["pipeline_north"].astype(str)
        )
        frames.append(gdf)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def count_dict(series):
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()}


def parse_args():
    p = argparse.ArgumentParser(description="Compare full census CSV with curated crowns and pipeline splits.")
    p.add_argument("--censo_csv", required=True)
    p.add_argument("--curated", required=True)
    p.add_argument("--pipeline_dir", default="./outputs/splits_binary_curated")
    p.add_argument("--output_dir", default="./outputs/censo_curated_comparison")

    p.add_argument("--censo_species_col", default="NOMBRE_COMUN")
    p.add_argument("--censo_zone_col", default="ZONA_UTM")
    p.add_argument("--censo_east_col", default="COORDENADA_ESTE")
    p.add_argument("--censo_north_col", default="COORDENADA_NORTE")

    p.add_argument("--curated_species_col", default="NOMBRE_COM")
    p.add_argument("--curated_zone_col", default="ZONA_UTM")
    p.add_argument("--curated_east_col", default="COORDENADA")
    p.add_argument("--curated_north_col", default="COORDENA_1")

    p.add_argument("--target_label", default="Shihuahuaco")
    p.add_argument("--target_crs", default="EPSG:32718")
    p.add_argument("--pipeline_zone", default="18S")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    censo_raw = read_csv_flexible(args.censo_csv)
    censo = add_censo_fields(
        censo_raw,
        args.censo_species_col,
        args.censo_zone_col,
        args.censo_east_col,
        args.censo_north_col,
        "censo",
    )
    curated = read_curated(
        args.curated,
        args.curated_species_col,
        args.curated_zone_col,
        args.curated_east_col,
        args.curated_north_col,
    )
    pipeline = read_pipeline_splits(args.pipeline_dir, args.target_crs, args.pipeline_zone)

    target_norm = normalize_key_text(args.target_label)
    censo_keys = set(censo["censo_coord_key"])
    curated_keys = set(curated["curated_coord_key"])
    censo_target_keys = set(censo.loc[censo["censo_species_norm"] == target_norm, "censo_coord_key"])
    curated_target_keys = set(curated.loc[curated["curated_species_norm"] == target_norm, "curated_coord_key"])
    censo_duplicate_key_rows = int(censo["censo_coord_key"].duplicated(keep=False).sum())
    censo_lookup = censo.drop_duplicates("censo_coord_key", keep="first").copy()

    curated_match = curated.merge(
        censo_lookup[
            [
                "censo_coord_key",
                "censo_species",
                "censo_species_norm",
                args.censo_species_col,
                args.censo_zone_col,
                args.censo_east_col,
                args.censo_north_col,
            ]
        ],
        left_on="curated_coord_key",
        right_on="censo_coord_key",
        how="left",
        suffixes=("_curated", "_censo"),
    )
    curated_match["species_agree"] = curated_match["curated_species_norm"] == curated_match["censo_species_norm"]
    curated_match.drop(columns="geometry", errors="ignore").to_csv(output_dir / "curated_to_censo_matches.csv", index=False)

    summary = {
        "censo_csv": str(args.censo_csv),
        "curated": str(args.curated),
        "pipeline_dir": str(args.pipeline_dir) if pipeline is not None else None,
        "target_label": args.target_label,
        "censo_rows": int(len(censo)),
        "censo_unique_coordinate_keys": int(len(censo_keys)),
        "censo_duplicate_coordinate_key_rows": censo_duplicate_key_rows,
        "censo_species_counts_top": count_dict(censo["censo_species"].head(0)),
        "censo_target_rows": int((censo["censo_species_norm"] == target_norm).sum()),
        "censo_target_unique_coordinate_keys": int(len(censo_target_keys)),
        "curated_rows": int(len(curated)),
        "curated_unique_coordinate_keys": int(len(curated_keys)),
        "curated_rows_matching_censo_key": int(curated["curated_coord_key"].isin(censo_keys).sum()),
        "curated_rows_species_agreeing_with_censo": int(curated_match["species_agree"].sum()),
        "curated_target_rows": int((curated["curated_species_norm"] == target_norm).sum()),
        "curated_target_unique_coordinate_keys": int(len(curated_target_keys)),
        "curated_target_keys_in_censo_target": int(len(curated_target_keys & censo_target_keys)),
        "censo_target_keys_in_curated_target": int(len(censo_target_keys & curated_target_keys)),
        "censo_target_keys_missing_from_curated": int(len(censo_target_keys - curated_target_keys)),
    }

    species_counts = censo["censo_species"].value_counts(dropna=False).rename_axis("species").reset_index(name="count")
    species_counts.to_csv(output_dir / "censo_species_counts.csv", index=False)
    summary["censo_species_counts_top"] = {
        str(row["species"]): int(row["count"])
        for _, row in species_counts.head(50).iterrows()
    }

    curated_species_counts = (
        curated["curated_species"].value_counts(dropna=False).rename_axis("species").reset_index(name="count")
    )
    curated_species_counts.to_csv(output_dir / "curated_species_counts.csv", index=False)
    summary["curated_species_counts_top"] = {
        str(row["species"]): int(row["count"])
        for _, row in curated_species_counts.head(50).iterrows()
    }

    if pipeline is not None:
        pipeline_keys = set(pipeline["pipeline_coord_key"])
        pipeline_target = pipeline[pipeline["BinaryTree"].astype(str).isin(["1", args.target_label])].copy()
        pipeline_target_keys = set(pipeline_target["pipeline_coord_key"])
        summary.update(
            {
                "pipeline_rows": int(len(pipeline)),
                "pipeline_unique_coordinate_keys": int(len(pipeline_keys)),
                "pipeline_target_rows": int(len(pipeline_target)),
                "pipeline_target_unique_coordinate_keys": int(len(pipeline_target_keys)),
                "pipeline_target_keys_in_curated_target": int(len(pipeline_target_keys & curated_target_keys)),
                "curated_target_keys_in_pipeline": int(len(curated_target_keys & pipeline_keys)),
                "curated_target_keys_missing_from_pipeline": int(len(curated_target_keys - pipeline_keys)),
                "censo_target_keys_in_pipeline": int(len(censo_target_keys & pipeline_keys)),
            }
        )
        pipeline.drop(columns="geometry", errors="ignore").to_csv(output_dir / "pipeline_coordinate_keys.csv", index=False)

    summary_path = output_dir / "censo_curated_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary: {summary_path}")
    print(f"Species: {output_dir / 'censo_species_counts.csv'}")
    print(f"Matches: {output_dir / 'curated_to_censo_matches.csv'}")


if __name__ == "__main__":
    main()
