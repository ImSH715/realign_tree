"""
Build evaluation input CSV files for three evaluation settings:

1. Direct GT evaluation:
   Convert valid_points.shp into a CSV where each ground-truth point is matched
   to the best TIFF tile and stored with its true world coordinates.

2. Recovery evaluation:
   Create perturbed versions of the direct GT CSV by adding random offsets to the
   original coordinates while keeping the true GT coordinates unchanged.

3. No-GT overlap evaluation:
   Convert a census CSV into an overlap-only CSV by assigning each point to the
   best-matching TIFF tile based on raster bounds and center distance.

This script does not train any model. It only prepares evaluation inputs.
"""

import os
import math
import argparse
import pandas as pd
import geopandas as gpd
import rasterio

from src.data.tif_io import recursive_find_tif_files


def load_raster_info(imagery_root: str):
    tif_paths = recursive_find_tif_files(imagery_root)
    raster_info = []

    for tif in tif_paths:
        try:
            with rasterio.open(tif) as src:
                raster_info.append(
                    {
                        "path": tif,
                        "name": os.path.basename(tif),
                        "bounds": src.bounds,
                        "crs": src.crs,
                        "width": src.width,
                        "height": src.height,
                    }
                )
        except Exception as e:
            print(f"[WARN] Could not open TIFF: {tif} | {e}")

    if len(raster_info) == 0:
        raise RuntimeError("No readable TIFFs found.")

    return raster_info


def center_distance(x: float, y: float, bounds) -> float:
    cx = (bounds.left + bounds.right) / 2.0
    cy = (bounds.bottom + bounds.top) / 2.0
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def point_matches_raster(x: float, y: float, bounds, tol: float) -> bool:
    return (
        (bounds.left - tol) <= x <= (bounds.right + tol)
        and (bounds.bottom - tol) <= y <= (bounds.top + tol)
    )


def assign_best_tif(x: float, y: float, raster_info, tol: float):
    candidates = []
    for r in raster_info:
        b = r["bounds"]
        if point_matches_raster(x, y, b, tol):
            dist = center_distance(x, y, b)
            candidates.append((dist, r))

    if len(candidates) == 0:
        return None, 0, None

    candidates.sort(key=lambda t: t[0])
    best = candidates[0][1]
    return best["path"], len(candidates), float(candidates[0][0])


def build_direct_gt_from_shp(
    shp_path: str,
    imagery_root: str,
    output_csv: str,
    label_field: str,
    point_id_field: str,
    tolerance_m: float,
    target_crs: str = "EPSG:32718",
):
    raster_info = load_raster_info(imagery_root)
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        raise RuntimeError("Input SHP has no CRS. Please define it first.")

    gdf = gdf.to_crs(target_crs)

    rows = []
    no_match = 0

    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        x = float(geom.x)
        y = float(geom.y)

        matched_tif, num_matches, center_dist = assign_best_tif(
            x=x,
            y=y,
            raster_info=raster_info,
            tol=tolerance_m,
        )

        if matched_tif is None:
            no_match += 1
            continue

        point_id = str(row[point_id_field]) if point_id_field in gdf.columns else f"gt_{i}"
        label = str(row[label_field]).strip()

        rows.append(
            {
                "point_id": point_id,
                "label": label,
                "original_east": x,
                "original_north": y,
                "gt_east": x,
                "gt_north": y,
                "matched_tif": matched_tif,
                "num_tif_matches": num_matches,
                "match_center_distance_m": center_dist,
                "source": "valid_points_direct",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("=" * 100)
    print("Direct GT input built from SHP")
    print(f"SHP path                : {shp_path}")
    print(f"Output CSV              : {output_csv}")
    print(f"Rows kept               : {len(df)}")
    print(f"Rows without TIFF match : {no_match}")
    print("=" * 100)


def build_overlap_from_censo(
    input_csv: str,
    imagery_root: str,
    output_csv: str,
    x_column: str,
    y_column: str,
    label_column: str,
    tolerance_m: float,
):
    raster_info = load_raster_info(imagery_root)
    df = pd.read_csv(input_csv)

    rows = []
    no_match = 0

    for i, row in df.iterrows():
        try:
            x = float(row[x_column])
            y = float(row[y_column])
        except Exception:
            continue

        matched_tif, num_matches, center_dist = assign_best_tif(
            x=x,
            y=y,
            raster_info=raster_info,
            tol=tolerance_m,
        )

        if matched_tif is None:
            no_match += 1
            continue

        label = str(row[label_column]).strip()

        out = dict(row)
        out["point_id"] = f"censo_{i}"
        out["label"] = label
        out["original_east"] = x
        out["original_north"] = y
        out["matched_tif"] = matched_tif
        out["num_tif_matches"] = num_matches
        out["match_center_distance_m"] = center_dist
        out["source"] = "censo_overlap"
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    print("=" * 100)
    print("Censo overlap CSV built")
    print(f"Input CSV               : {input_csv}")
    print(f"Output CSV              : {output_csv}")
    print(f"Rows kept               : {len(out_df)}")
    print(f"Rows without TIFF match : {no_match}")
    print("=" * 100)


def build_recovery_inputs(
    input_csv: str,
    output_csv: str,
    offset_m: float,
    seed: int = 42,
):
    import numpy as np

    df = pd.read_csv(input_csv)

    required = ["point_id", "label", "original_east", "original_north", "gt_east", "gt_north", "matched_tif"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in direct GT CSV.")

    np.random.seed(seed)

    angles = np.random.uniform(0, 2 * np.pi, size=len(df))
    radii = np.random.uniform(0, offset_m, size=len(df))

    dx = radii * np.cos(angles)
    dy = radii * np.sin(angles)

    out = df.copy()
    out["original_east"] = out["gt_east"] + dx
    out["original_north"] = out["gt_north"] + dy
    out["recovery_offset_m"] = radii
    out["source"] = "valid_points_recovery"

    out.to_csv(output_csv, index=False)

    print("=" * 100)
    print("Recovery input CSV built")
    print(f"Input direct GT CSV : {input_csv}")
    print(f"Output CSV          : {output_csv}")
    print(f"Offset max (m)      : {offset_m}")
    print(f"Rows                : {len(out)}")
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Build evaluation input CSVs.")

    sub = parser.add_subparsers(dest="mode", required=True)

    shp = sub.add_parser("direct_gt_from_shp")
    shp.add_argument("--shp_path", required=True)
    shp.add_argument("--imagery_root", required=True)
    shp.add_argument("--output_csv", required=True)
    shp.add_argument("--label_field", default="Tree")
    shp.add_argument("--point_id_field", default="")
    shp.add_argument("--tolerance_m", type=float, default=10.0)
    shp.add_argument("--target_crs", default="EPSG:32718")

    censo = sub.add_parser("censo_overlap")
    censo.add_argument("--input_csv", required=True)
    censo.add_argument("--imagery_root", required=True)
    censo.add_argument("--output_csv", required=True)
    censo.add_argument("--x_column", default="COORDENADA_ESTE")
    censo.add_argument("--y_column", default="COORDENADA_NORTE")
    censo.add_argument("--label_column", default="NOMBRE_COMUN")
    censo.add_argument("--tolerance_m", type=float, default=10.0)

    rec = sub.add_parser("recovery_from_direct_gt")
    rec.add_argument("--input_csv", required=True)
    rec.add_argument("--output_csv", required=True)
    rec.add_argument("--offset_m", type=float, required=True)
    rec.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "direct_gt_from_shp":
        point_id_field = args.point_id_field if args.point_id_field else args.label_field
        build_direct_gt_from_shp(
            shp_path=args.shp_path,
            imagery_root=args.imagery_root,
            output_csv=args.output_csv,
            label_field=args.label_field,
            point_id_field=point_id_field,
            tolerance_m=args.tolerance_m,
            target_crs=args.target_crs,
        )
    elif args.mode == "censo_overlap":
        build_overlap_from_censo(
            input_csv=args.input_csv,
            imagery_root=args.imagery_root,
            output_csv=args.output_csv,
            x_column=args.x_column,
            y_column=args.y_column,
            label_column=args.label_column,
            tolerance_m=args.tolerance_m,
        )
    elif args.mode == "recovery_from_direct_gt":
        build_recovery_inputs(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            offset_m=args.offset_m,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()