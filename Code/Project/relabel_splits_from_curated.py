"""
Relabel existing pipeline split shapefiles from a curated crown layer.

The current split files already contain the fields needed by the training
pipeline, including Folder/File/fx/fy. This script preserves those rows and
train/val/test memberships, nearest-joins them to curated crown centroids, and
rewrites the species/binary labels from the curated species field.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def normalize_label(value):
    return str(value).strip().lower()


def read_curated_centroids(path, target_crs):
    gdf = gpd.read_file(path).to_crs(target_crs)
    source_geom = gdf.geometry.copy()
    gdf["geometry"] = source_geom.centroid
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=target_crs)


def split_path(input_dir, split):
    return Path(input_dir) / f"valid_points_{split}.shp"


def output_split_path(output_dir, split):
    return Path(output_dir) / f"valid_points_{split}.shp"


def value_counts(series):
    return {str(k): int(v) for k, v in series.astype(str).value_counts(dropna=False).to_dict().items()}


def parse_args():
    p = argparse.ArgumentParser(description="Correct split labels from a curated crown shapefile.")
    p.add_argument("--curated", required=True, help="Curated SHP/GPKG path.")
    p.add_argument("--input_dir", default="./outputs/splits_binary")
    p.add_argument("--output_dir", default="./outputs/splits_binary_curated")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--target_crs", default="EPSG:32718")
    p.add_argument("--curated_species_field", default="NOMBRE_COM")
    p.add_argument("--tree_field", default="Tree")
    p.add_argument("--binary_field", default="BinaryTree")
    p.add_argument("--target_label", default="Shihuahuaco")
    p.add_argument("--positive_value", default="1")
    p.add_argument("--negative_value", default="0")
    p.add_argument("--max_distance_m", type=float, default=5.0)
    p.add_argument(
        "--allow_unmatched",
        action="store_true",
        help="Write outputs even when a split point is farther than max_distance_m from curated data.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curated = read_curated_centroids(args.curated, args.target_crs)
    if args.curated_species_field not in curated.columns:
        raise ValueError(
            f"Missing curated species field '{args.curated_species_field}'. "
            f"Available columns: {list(curated.columns)}"
        )

    curated_cols = [args.curated_species_field, "geometry"]
    for optional in ["NOMBRE_CIE", "cod_match", "cod_superv", "ZONA_UTM", "PCA"]:
        if optional in curated.columns:
            curated_cols.insert(-1, optional)

    audit_frames = []
    summaries = []
    target_norm = normalize_label(args.target_label)

    for split in args.splits:
        in_path = split_path(args.input_dir, split)
        if not in_path.exists():
            raise FileNotFoundError(in_path)

        original = gpd.read_file(in_path)
        working = original.to_crs(args.target_crs)

        joined = gpd.sjoin_nearest(
            working,
            curated[curated_cols],
            how="left",
            distance_col="curated_dist_m",
        )
        joined = (
            joined.sort_values("curated_dist_m", na_position="last")
            .groupby(level=0, sort=False)
            .first()
            .reindex(working.index)
        )

        unmatched = joined["curated_dist_m"].isna() | (joined["curated_dist_m"] > args.max_distance_m)
        if unmatched.any() and not args.allow_unmatched:
            examples = joined.loc[unmatched, ["curated_dist_m"]].head(10)
            raise RuntimeError(
                f"{split}: {int(unmatched.sum())} rows were farther than "
                f"{args.max_distance_m:g} m from curated centroids. Examples:\n{examples}"
            )

        corrected = original.copy()
        curated_species = joined[args.curated_species_field].astype(str).str.strip()
        corrected[args.tree_field] = curated_species.to_numpy()
        corrected[args.binary_field] = [
            args.positive_value if normalize_label(x) == target_norm else args.negative_value
            for x in curated_species
        ]

        out_path = output_split_path(output_dir, split)
        corrected.to_file(out_path)

        audit = pd.DataFrame(
            {
                "split": split,
                "source_index": range(len(corrected)),
                "old_tree": original[args.tree_field].astype(str).to_numpy()
                if args.tree_field in original.columns
                else "",
                "old_binary": original[args.binary_field].astype(str).to_numpy()
                if args.binary_field in original.columns
                else "",
                "curated_species": curated_species.to_numpy(),
                "new_tree": corrected[args.tree_field].astype(str).to_numpy(),
                "new_binary": corrected[args.binary_field].astype(str).to_numpy(),
                "curated_dist_m": joined["curated_dist_m"].to_numpy(),
            }
        )
        for optional in ["NOMBRE_CIE", "cod_match", "cod_superv", "ZONA_UTM", "PCA"]:
            if optional in joined.columns:
                audit[f"curated_{optional}"] = joined[optional].astype(str).to_numpy()
        audit_frames.append(audit)

        summaries.append(
            {
                "split": split,
                "input": str(in_path),
                "output": str(out_path),
                "rows": int(len(corrected)),
                "old_tree_counts": value_counts(original[args.tree_field])
                if args.tree_field in original.columns
                else {},
                "old_binary_counts": value_counts(original[args.binary_field])
                if args.binary_field in original.columns
                else {},
                "new_tree_counts": value_counts(corrected[args.tree_field]),
                "new_binary_counts": value_counts(corrected[args.binary_field]),
                "max_curated_dist_m": float(joined["curated_dist_m"].max()),
                "rows_over_max_distance": int(unmatched.sum()),
            }
        )

    audit_df = pd.concat(audit_frames, ignore_index=True)
    audit_csv = output_dir / "curated_relabel_audit.csv"
    audit_df.to_csv(audit_csv, index=False)

    summary = {
        "curated": str(args.curated),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "target_label": args.target_label,
        "curated_species_field": args.curated_species_field,
        "max_distance_m": args.max_distance_m,
        "splits": summaries,
        "overall_old_binary_counts": value_counts(audit_df["old_binary"]),
        "overall_new_binary_counts": value_counts(audit_df["new_binary"]),
        "overall_curated_species_counts": value_counts(audit_df["curated_species"]),
    }
    summary_json = output_dir / "curated_relabel_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Audit CSV   : {audit_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
