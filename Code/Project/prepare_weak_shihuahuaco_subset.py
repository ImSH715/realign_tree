"""
Prepare a weakly labelled Shihuahuaco subset from the full census table.

The full census CSV has species and UTM coordinates, but the MIL pipeline also
needs each point to be assigned to an orthomosaic TIFF. This script:

1. filters the census to Shihuahuaco rows;
2. removes records that appear to be in/near the curated Shihuahuaco crown set;
3. assigns each remaining point to a covering TIFF footprint;
4. samples a reproducible subset and writes a GeoPackage/CSV with the fields
   expected by the MIL point-bag dataset.
"""

import argparse
import json
import math
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

from src.data.tif_io import recursive_find_tif_files


def read_csv_flexible(path):
    last_error = None
    for encoding in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def normalize_text(value):
    return str(value).strip()


def normalize_key(value):
    return normalize_text(value).lower()


def numeric_coord(series):
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def coord_key(df, zone_col, east_col, north_col):
    zone = df[zone_col].map(normalize_text)
    east = numeric_coord(df[east_col])
    north = numeric_coord(df[north_col])
    return zone.astype(str) + "|" + east.astype(str) + "|" + north.astype(str)


def derive_folder_key(path, imagery_root):
    rel = os.path.relpath(path, imagery_root)
    parts = rel.split(os.sep)
    for part in parts:
        if part.startswith("2023-"):
            return part
    return parts[0] if parts and parts[0] != "." else ""


def load_raster_footprints(imagery_root, target_crs):
    rows = []
    unreadable = []
    for tif in recursive_find_tif_files(imagery_root):
        try:
            with rasterio.open(tif) as src:
                geom = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                one = gpd.GeoDataFrame(
                    {
                        "matched_tif": [tif],
                        "File": [os.path.basename(tif)],
                        "Folder": [derive_folder_key(tif, imagery_root)],
                        "tif_width": [int(src.width)],
                        "tif_height": [int(src.height)],
                        "geometry": [geom],
                    },
                    crs=src.crs,
                )
                if one.crs is not None and str(one.crs) != str(target_crs):
                    one = one.to_crs(target_crs)
                rows.append(one)
        except Exception as exc:
            unreadable.append((tif, str(exc)))

    if not rows:
        raise RuntimeError(f"No readable TIFFs found under {imagery_root}")

    footprints = pd.concat(rows, ignore_index=True)
    footprints = gpd.GeoDataFrame(footprints, geometry="geometry", crs=target_crs)
    return footprints, unreadable


def centroid_distance(point_geom, poly_geom):
    c = poly_geom.centroid
    return math.hypot(point_geom.x - c.x, point_geom.y - c.y)


def assign_footprints(points, footprints):
    joined = gpd.sjoin(
        points,
        footprints[["Folder", "File", "matched_tif", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return joined

    footprint_geoms = footprints.geometry
    joined["match_center_distance_m"] = [
        centroid_distance(row.geometry, footprint_geoms.loc[row["index_right"]])
        for _, row in joined.iterrows()
    ]
    joined = (
        joined.sort_values(["match_center_distance_m", "matched_tif"])
        .groupby(level=0, sort=False)
        .first()
    )
    return joined


def parse_args():
    p = argparse.ArgumentParser(description="Prepare weak Shihuahuaco census subset for MIL realignment.")
    p.add_argument("--censo_csv", required=True)
    p.add_argument("--curated", required=True)
    p.add_argument("--imagery_root", required=True)
    p.add_argument("--output_gpkg", required=True)
    p.add_argument("--output_csv", default="")
    p.add_argument("--summary_json", default="")

    p.add_argument("--target_label", default="Shihuahuaco")
    p.add_argument("--target_crs", default="EPSG:32718")
    p.add_argument("--zone_value", default="18S")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--censo_species_col", default="NOMBRE_COMUN")
    p.add_argument("--censo_zone_col", default="ZONA_UTM")
    p.add_argument("--censo_east_col", default="COORDENADA_ESTE")
    p.add_argument("--censo_north_col", default="COORDENADA_NORTE")

    p.add_argument("--curated_species_col", default="NOMBRE_COM")
    p.add_argument("--curated_zone_col", default="ZONA_UTM")
    p.add_argument("--curated_east_col", default="COORDENADA")
    p.add_argument("--curated_north_col", default="COORDENA_1")
    p.add_argument(
        "--exclude_distance_m",
        type=float,
        default=30.0,
        help="Also exclude census points within this distance of a curated target crown centroid.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else output_gpkg.with_suffix(".csv")
    summary_json = Path(args.summary_json) if args.summary_json else output_gpkg.with_name("weak_shihuahuaco_subset_summary.json")

    target_norm = normalize_key(args.target_label)

    censo = read_csv_flexible(args.censo_csv)
    censo["__species_norm"] = censo[args.censo_species_col].map(normalize_key)
    censo["__zone"] = censo[args.censo_zone_col].map(normalize_text)
    censo["__east"] = pd.to_numeric(censo[args.censo_east_col], errors="coerce")
    censo["__north"] = pd.to_numeric(censo[args.censo_north_col], errors="coerce")
    censo["__coord_key"] = coord_key(censo, args.censo_zone_col, args.censo_east_col, args.censo_north_col)
    censo["censo_source_index"] = range(len(censo))

    candidates = censo[
        (censo["__species_norm"] == target_norm)
        & censo["__east"].notna()
        & censo["__north"].notna()
    ].copy()
    if args.zone_value:
        candidates = candidates[candidates["__zone"].str.upper() == args.zone_value.upper()].copy()

    points = gpd.GeoDataFrame(
        candidates,
        geometry=gpd.points_from_xy(candidates["__east"], candidates["__north"]),
        crs=args.target_crs,
    )

    curated = gpd.read_file(args.curated).to_crs(args.target_crs)
    curated["__species_norm"] = curated[args.curated_species_col].map(normalize_key)
    curated_target = curated[curated["__species_norm"] == target_norm].copy()
    curated_target_keys = set()
    for col in [args.curated_zone_col, args.curated_east_col, args.curated_north_col]:
        if col not in curated_target.columns:
            curated_target_keys = set()
            break
    else:
        curated_target["__coord_key"] = coord_key(
            curated_target,
            args.curated_zone_col,
            args.curated_east_col,
            args.curated_north_col,
        )
        curated_target_keys = set(curated_target["__coord_key"])

    exclude_key = points["__coord_key"].isin(curated_target_keys)

    if len(curated_target):
        inside = gpd.sjoin(
            points[["censo_source_index", "geometry"]],
            curated_target[["geometry"]],
            how="left",
            predicate="within",
        )
        inside_mask = inside["index_right"].notna().groupby(level=0).any().reindex(points.index, fill_value=False)

        curated_centroids = curated_target.copy()
        curated_centroids["geometry"] = curated_centroids.geometry.centroid
        nearest = gpd.sjoin_nearest(
            points[["censo_source_index", "geometry"]],
            curated_centroids[["geometry"]],
            how="left",
            distance_col="nearest_curated_target_m",
        )
        nearest_dist = nearest.groupby(level=0)["nearest_curated_target_m"].min().reindex(points.index)
        near_mask = nearest_dist <= float(args.exclude_distance_m)
    else:
        inside_mask = pd.Series(False, index=points.index)
        nearest_dist = pd.Series(pd.NA, index=points.index)
        near_mask = pd.Series(False, index=points.index)

    points["nearest_curated_target_m"] = nearest_dist
    points["excluded_exact_curated_key"] = exclude_key.astype(int)
    points["excluded_inside_curated_target"] = inside_mask.astype(int)
    points["excluded_near_curated_target"] = near_mask.fillna(False).astype(int)

    not_curated = points[~(exclude_key | inside_mask | near_mask.fillna(False))].copy()

    footprints, unreadable = load_raster_footprints(args.imagery_root, args.target_crs)
    matched = assign_footprints(not_curated, footprints)
    matched = gpd.GeoDataFrame(matched, geometry="geometry", crs=args.target_crs)

    if len(matched) > args.limit:
        matched = matched.sample(n=args.limit, random_state=args.seed).sort_index()

    matched["Tree"] = args.target_label
    matched["BinaryTree"] = "1"
    matched["weak_label_source"] = "censo_shihuahuaco_not_curated"
    matched["fx"] = matched.geometry.x.astype(float)
    matched["fy"] = matched.geometry.y.astype(float)
    matched["original_east"] = matched.geometry.x.astype(float)
    matched["original_north"] = matched.geometry.y.astype(float)
    matched["point_id"] = matched["censo_source_index"].map(lambda x: f"censo_{int(x)}")

    # Keep common pipeline fields early in the table.
    preferred = [
        "point_id",
        "Tree",
        "BinaryTree",
        "Folder",
        "File",
        "fx",
        "fy",
        "original_east",
        "original_north",
        "matched_tif",
        "match_center_distance_m",
        "nearest_curated_target_m",
        "weak_label_source",
        "censo_source_index",
    ]
    remaining = [c for c in matched.columns if c not in preferred and c != "geometry"]
    matched = matched[preferred + remaining + ["geometry"]]

    matched.to_file(output_gpkg, driver="GPKG")
    matched.drop(columns="geometry", errors="ignore").to_csv(output_csv, index=False)

    summary = {
        "censo_csv": str(args.censo_csv),
        "curated": str(args.curated),
        "imagery_root": str(args.imagery_root),
        "target_label": args.target_label,
        "target_crs": args.target_crs,
        "zone_value": args.zone_value,
        "limit": int(args.limit),
        "seed": int(args.seed),
        "censo_rows": int(len(censo)),
        "censo_target_rows": int((censo["__species_norm"] == target_norm).sum()),
        "censo_target_zone_rows": int(len(points)),
        "curated_target_rows": int(len(curated_target)),
        "excluded_exact_curated_key": int(exclude_key.sum()),
        "excluded_inside_curated_target": int(inside_mask.sum()),
        "excluded_near_curated_target": int(near_mask.fillna(False).sum()),
        "remaining_not_curated_candidates": int(len(not_curated)),
        "readable_tif_footprints": int(len(footprints)),
        "unreadable_tifs": int(len(unreadable)),
        "matched_imagery_candidates": int(len(assign_footprints(not_curated, footprints))),
        "written_rows": int(len(matched)),
        "output_gpkg": str(output_gpkg),
        "output_csv": str(output_csv),
        "exclude_distance_m": float(args.exclude_distance_m),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Subset GPKG: {output_gpkg}")
    print(f"Subset CSV : {output_csv}")
    print(f"Summary    : {summary_json}")


if __name__ == "__main__":
    main()

