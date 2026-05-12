"""
Compare pipeline point labels against curated/reference spatial layers.

This is intended for Stanage, where the project split shapefiles and shared
curated layers are available. It reports whether the pipeline inputs appear to
match a curated layer exactly, or whether they are merely spatially near it.
"""

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def read_vector(path):
    path = str(path)
    if "|" in path:
        base, layer = path.split("|", 1)
        if layer.startswith("layer="):
            layer = layer.split("=", 1)[1]
        return gpd.read_file(base, layer=layer)
    return gpd.read_file(path)


def normalize_text(s):
    return str(s).strip().lower()


def filter_values(gdf, field, values):
    if not field or field not in gdf.columns or not values:
        return gdf.copy()
    wanted = {normalize_text(v) for v in values}
    return gdf[gdf[field].map(normalize_text).isin(wanted)].copy()


def load_pipeline(paths, label_field, positive_values):
    frames = []
    for path in paths:
        gdf = read_vector(path)
        split = Path(path).stem.replace("valid_points_", "")
        gdf = gdf.copy()
        gdf["__source_path"] = str(path)
        gdf["__split"] = split
        frames.append(gdf)
    if not frames:
        raise ValueError("No pipeline shapefiles were provided.")
    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs)
    out["__is_positive"] = False
    if label_field in out.columns:
        wanted = {normalize_text(v) for v in positive_values}
        out["__is_positive"] = out[label_field].map(normalize_text).isin(wanted)
    return out


def coord_key(series, precision):
    x = series.geometry.x.round(precision).astype(str)
    y = series.geometry.y.round(precision).astype(str)
    return x + "," + y


def geom_summary(gdf):
    return {
        "rows": int(len(gdf)),
        "crs": str(gdf.crs),
        "geometry_types": {str(k): int(v) for k, v in gdf.geom_type.value_counts().to_dict().items()},
        "bounds": [float(x) for x in gdf.total_bounds] if len(gdf) else None,
        "columns": [str(c) for c in gdf.columns],
    }


def reproject_pair(left, right, target_crs):
    if target_crs:
        return left.to_crs(target_crs), right.to_crs(target_crs)
    if left.crs and right.crs and str(left.crs) != str(right.crs):
        return left, right.to_crs(left.crs)
    return left, right


def nearest_distances(points, target_geoms):
    try:
        joined = gpd.sjoin_nearest(
            points[["geometry"]],
            target_geoms[["geometry"]],
            how="left",
            distance_col="nearest_distance_m",
        )
        return joined.groupby(joined.index)["nearest_distance_m"].min().reindex(points.index).to_numpy()
    except Exception:
        distances = []
        target = target_geoms.geometry
        for geom in points.geometry:
            distances.append(float(target.distance(geom).min()) if len(target) else np.nan)
        return np.asarray(distances, dtype=np.float64)


def threshold_counts(distances, thresholds):
    distances = np.asarray(distances, dtype=np.float64)
    valid = np.isfinite(distances)
    out = {}
    for threshold in thresholds:
        out[f"within_{threshold:g}m"] = int((distances[valid] <= threshold).sum())
    return out


def compare_one(pipeline, curated, curated_name, target_crs, thresholds, coord_precision):
    left, right = reproject_pair(pipeline, curated, target_crs)
    positive = left[left["__is_positive"]].copy()
    selected = positive if len(positive) else left

    geom_dist = nearest_distances(selected, right) if len(right) else np.asarray([])
    centroid_right = right.copy()
    if len(centroid_right):
        centroid_right["geometry"] = centroid_right.geometry.centroid
    centroid_dist = nearest_distances(selected, centroid_right) if len(centroid_right) else np.asarray([])

    exact_coord_overlap = None
    if all(t == "Point" for t in right.geom_type.unique()) and len(selected) and len(right):
        left_keys = set(coord_key(selected, coord_precision))
        right_keys = set(coord_key(right, coord_precision))
        exact_coord_overlap = len(left_keys & right_keys)
    elif len(selected) and len(centroid_right):
        left_keys = set(coord_key(selected, coord_precision))
        right_keys = set(coord_key(centroid_right, coord_precision))
        exact_coord_overlap = len(left_keys & right_keys)

    within_polygon = None
    if len(right) and any(t in {"Polygon", "MultiPolygon"} for t in right.geom_type.unique()):
        try:
            joined = gpd.sjoin(selected[["geometry"]], right[["geometry"]], how="left", predicate="within")
            within_polygon = int(joined["index_right"].notna().groupby(joined.index).any().sum())
        except Exception:
            within_polygon = None

    common_columns = sorted(set(map(str, left.columns)) & set(map(str, right.columns)))
    return {
        "curated_name": curated_name,
        "pipeline_points_compared": int(len(selected)),
        "pipeline_positive_points": int(left["__is_positive"].sum()),
        "curated_rows": int(len(right)),
        "curated_geometry_types": {str(k): int(v) for k, v in right.geom_type.value_counts().to_dict().items()},
        "common_columns": common_columns,
        "exact_coordinate_overlap_rounded": exact_coord_overlap,
        "within_curated_polygon": within_polygon,
        "nearest_geometry_distance_m": {
            "min": float(np.nanmin(geom_dist)) if len(geom_dist) else None,
            "median": float(np.nanmedian(geom_dist)) if len(geom_dist) else None,
            "mean": float(np.nanmean(geom_dist)) if len(geom_dist) else None,
            "max": float(np.nanmax(geom_dist)) if len(geom_dist) else None,
            **threshold_counts(geom_dist, thresholds),
        },
        "nearest_centroid_distance_m": {
            "min": float(np.nanmin(centroid_dist)) if len(centroid_dist) else None,
            "median": float(np.nanmedian(centroid_dist)) if len(centroid_dist) else None,
            "mean": float(np.nanmean(centroid_dist)) if len(centroid_dist) else None,
            "max": float(np.nanmax(centroid_dist)) if len(centroid_dist) else None,
            **threshold_counts(centroid_dist, thresholds),
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare pipeline points against curated/reference layers.")
    p.add_argument(
        "--pipeline_shps",
        nargs="+",
        default=[
            "./outputs/splits_binary/valid_points_train.shp",
            "./outputs/splits_binary/valid_points_val.shp",
            "./outputs/splits_binary/valid_points_test.shp",
        ],
    )
    p.add_argument("--curated", nargs="+", required=True, help="Curated SHP/GPKG paths.")
    p.add_argument("--output_dir", default="./outputs/curated_comparison")
    p.add_argument("--target_crs", default="EPSG:32718")
    p.add_argument("--pipeline_label_field", default="BinaryTree")
    p.add_argument("--pipeline_positive_values", default="1,Shihuahuaco")
    p.add_argument("--curated_species_field", default="NOMBRE_COM")
    p.add_argument("--curated_species_value", default="Shihuahuaco")
    p.add_argument("--distance_thresholds_m", default="1,5,10,20,30")
    p.add_argument("--coord_precision", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_values = [x.strip() for x in args.pipeline_positive_values.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.distance_thresholds_m.split(",") if x.strip()]

    pipeline = load_pipeline(args.pipeline_shps, args.pipeline_label_field, positive_values)
    pipeline_summary = geom_summary(pipeline)
    pipeline_summary["positive_count"] = int(pipeline["__is_positive"].sum())
    pipeline_summary["split_counts"] = {str(k): int(v) for k, v in pipeline["__split"].value_counts().to_dict().items()}
    if args.pipeline_label_field in pipeline.columns:
        pipeline_summary["label_counts"] = {
            str(k): int(v) for k, v in pipeline[args.pipeline_label_field].astype(str).value_counts().to_dict().items()
        }

    summaries = []
    for path in args.curated:
        curated = read_vector(path)
        raw_summary = geom_summary(curated)
        if args.curated_species_field in curated.columns and args.curated_species_value:
            curated = filter_values(curated, args.curated_species_field, [args.curated_species_value])
        comparison = compare_one(
            pipeline=pipeline,
            curated=curated,
            curated_name=path,
            target_crs=args.target_crs,
            thresholds=thresholds,
            coord_precision=args.coord_precision,
        )
        comparison["raw_curated_summary"] = raw_summary
        summaries.append(comparison)

    out = {
        "pipeline_summary": pipeline_summary,
        "comparisons": summaries,
    }
    json_path = output_dir / "curated_pipeline_comparison.json"
    csv_path = output_dir / "curated_pipeline_comparison.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    flat_rows = []
    for item in summaries:
        row = {
            "curated_name": item["curated_name"],
            "pipeline_points_compared": item["pipeline_points_compared"],
            "pipeline_positive_points": item["pipeline_positive_points"],
            "curated_rows": item["curated_rows"],
            "exact_coordinate_overlap_rounded": item["exact_coordinate_overlap_rounded"],
            "within_curated_polygon": item["within_curated_polygon"],
        }
        for prefix in ["nearest_geometry_distance_m", "nearest_centroid_distance_m"]:
            for k, v in item[prefix].items():
                row[f"{prefix}.{k}"] = v
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    print(json.dumps(out, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
