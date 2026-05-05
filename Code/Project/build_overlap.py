import os
import math
import argparse
import pandas as pd
import rasterio

from src.data.tif_io import recursive_find_tif_files


def center_distance(x: float, y: float, bounds) -> float:
    cx = (bounds.left + bounds.right) / 2.0
    cy = (bounds.bottom + bounds.top) / 2.0
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)


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
                        "crs": str(src.crs),
                        "width": src.width,
                        "height": src.height,
                    }
                )
        except Exception as e:
            print(f"[WARN] Could not open TIFF: {tif} | {e}")

    return raster_info


def point_matches_raster(x: float, y: float, bounds, tol: float) -> bool:
    return (
        (bounds.left - tol) <= x <= (bounds.right + tol)
        and (bounds.bottom - tol) <= y <= (bounds.top + tol)
    )


def build_overlap_csv(
    input_csv: str,
    imagery_root: str,
    output_csv: str,
    x_column: str,
    y_column: str,
    tolerance_m: float,
):
    df = pd.read_csv(input_csv)
    if x_column not in df.columns:
        raise ValueError(f"Missing x column: {x_column}")
    if y_column not in df.columns:
        raise ValueError(f"Missing y column: {y_column}")

    raster_info = load_raster_info(imagery_root)
    print(f"[INFO] Loaded TIFFs: {len(raster_info)}")

    rows = []
    no_match = 0
    multi_match = 0

    for i, row in df.iterrows():
        try:
            x = float(row[x_column])
            y = float(row[y_column])
        except Exception as e:
            print(f"[WARN] Row {i} has invalid coordinates: {e}")
            continue

        candidates = []
        for r in raster_info:
            if point_matches_raster(x, y, r["bounds"], tolerance_m):
                dist = center_distance(x, y, r["bounds"])
                candidates.append((dist, r))

        if len(candidates) == 0:
            no_match += 1
            continue

        if len(candidates) > 1:
            multi_match += 1

        candidates.sort(key=lambda t: t[0])
        best = candidates[0][1]

        out_row = row.copy()
        out_row["matched_tif"] = best["path"]
        out_row["matched_tif_name"] = best["name"]
        out_row["num_tif_matches"] = len(candidates)
        out_row["match_center_distance_m"] = float(candidates[0][0])
        rows.append(out_row)

    out_df = pd.DataFrame(rows)

    print("=" * 100)
    print("Overlap build completed")
    print(f"Input rows                 : {len(df)}")
    print(f"Output overlapping rows    : {len(out_df)}")
    print(f"No-match rows              : {no_match}")
    print(f"Rows with multiple matches : {multi_match}")
    print("=" * 100)

    if len(out_df) == 0:
        raise RuntimeError("No overlapping rows were found. Nothing to save.")

    out_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved overlap CSV to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build overlap CSV by assigning each point to the best TIFF using bounds + center distance"
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--imagery_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--x_column", type=str, default="COORDENADA_ESTE")
    parser.add_argument("--y_column", type=str, default="COORDENADA_NORTE")
    parser.add_argument("--tolerance_m", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    build_overlap_csv(
        input_csv=args.input_csv,
        imagery_root=args.imagery_root,
        output_csv=args.output_csv,
        x_column=args.x_column,
        y_column=args.y_column,
        tolerance_m=args.tolerance_m,
    )


if __name__ == "__main__":
    main()