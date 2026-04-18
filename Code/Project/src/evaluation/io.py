"""
Geospatial I/O utilities for evaluation.

This module provides reusable helper functions for:
- reading raster metadata,
- assigning points to TIFF tiles,
- converting between world and pixel coordinates,
- loading evaluation points from CSV files,
- and augmenting refinement outputs with world-coordinate columns.
"""

import math
import os
from typing import List, Dict, Tuple

import pandas as pd
import rasterio

from src.data.tif_io import recursive_find_tif_files
from src.data.points import InputPoint


def load_raster_info(imagery_root: str) -> List[Dict]:
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


def assign_best_tif(x: float, y: float, raster_info: List[Dict], tol: float):
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


def world_to_pixel(image_path: str, east: float, north: float) -> Tuple[float, float]:
    with rasterio.open(image_path) as src:
        row, col = src.index(float(east), float(north))
    return float(col), float(row)


def pixel_to_world(image_path: str, x_col: float, y_row: float) -> Tuple[float, float]:
    with rasterio.open(image_path) as src:
        east, north = src.xy(float(y_row), float(x_col))
    return float(east), float(north)


def load_eval_points(
    csv_path: str,
    point_id_column: str,
    label_column: str,
    x_column: str,
    y_column: str,
    image_column: str,
    filter_label: str = "",
):
    df = pd.read_csv(csv_path)

    required = [point_id_column, label_column, x_column, y_column, image_column]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in input CSV.")

    df[label_column] = df[label_column].astype(str).str.strip()

    if filter_label:
        df = df[df[label_column].str.lower() == filter_label.strip().lower()].copy()

    points = []
    rows = []

    for _, row in df.iterrows():
        point_id = str(row[point_id_column])
        label = str(row[label_column]).strip()
        image_path = str(row[image_column]).strip()
        east = float(row[x_column])
        north = float(row[y_column])

        try:
            x_px, y_px = world_to_pixel(image_path, east, north)
        except Exception as e:
            print(f"[WARN] Could not convert world->pixel for {point_id}: {e}")
            continue

        points.append(
            InputPoint(
                point_id=point_id,
                image_path=image_path,
                x=x_px,
                y=y_px,
                target_label=label,
            )
        )
        rows.append(row)

    meta_df = pd.DataFrame(rows)
    return points, meta_df


def augment_output_with_world_coords(output_csv: str, input_meta_df: pd.DataFrame):
    df = pd.read_csv(output_csv)

    input_meta_df = input_meta_df.copy()
    input_meta_df["point_id"] = input_meta_df["point_id"].astype(str)

    merged = df.merge(
        input_meta_df,
        on="point_id",
        how="left",
        suffixes=("", "_input"),
    )

    refined_east = []
    refined_north = []
    coarse_east = []
    coarse_north = []

    for _, row in merged.iterrows():
        try:
            re, rn = pixel_to_world(row["image_path"], row["refined_x"], row["refined_y"])
        except Exception:
            re, rn = float("nan"), float("nan")

        try:
            ce, cn = pixel_to_world(row["image_path"], row["coarse_x"], row["coarse_y"])
        except Exception:
            ce, cn = float("nan"), float("nan")

        refined_east.append(re)
        refined_north.append(rn)
        coarse_east.append(ce)
        coarse_north.append(cn)

    merged["refined_east"] = refined_east
    merged["refined_north"] = refined_north
    merged["coarse_east"] = coarse_east
    merged["coarse_north"] = coarse_north

    merged.to_csv(output_csv, index=False)
    return merged