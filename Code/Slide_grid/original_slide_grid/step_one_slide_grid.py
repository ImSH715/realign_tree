import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

"""
The script loads grid and orange tree feautres, and detects if each grids and tree clowns are overriding.
If yes, there are three conditions to make.
1. Based on the boxes (grid is in 3x3 so total 9 boxes), if 3 boxes have a likelihood higher than 0.5.
 - Create center point of those boxes with likelihood higher than 0.5.
2. If not (1-2 boxes with likelihood higher than 0.5, or 1-2 boxes with higher 0.5 and other boxes with likelihood 
(feautres are still detected into those boxes)) create a point (coordinate) on the feature and grid override spot and set their type as slide (coordinates required to slide the grid).
3. If none of the boxes or one boxes has likelihood higher than 0.5, abort them.

Generate the .csv file including or coordinates of type center and slide.
"""

# Config
GRID_SHP = "../shapefiles/3x3_grid_on_points.shp"
FEATURE_SHP = "../shapefiles/orange_trees.shp"

OUTPUT_CSV = "../coordinate/step1_points.csv"

LIKELIHOOD_THRESHOLD = 0.5

# Helper functions
def compute_likelihood(grid_geom, feature_geom):
    inter = grid_geom.intersection(feature_geom)
    if inter.is_empty:
        return 0.0
    return inter.area / grid_geom.area


def center_from_cells(cells):
    minx, miny, maxx, maxy = cells.total_bounds
    return (minx + maxx) / 2, (miny + maxy) / 2


def overlap_point(grid_geom, feature_geom):
    inter = grid_geom.intersection(feature_geom)
    return inter.centroid


# Load data
grid_gdf = gpd.read_file(GRID_SHP)
feat_gdf = gpd.read_file(FEATURE_SHP)

grid_sindex = grid_gdf.sindex

results = []

# Main loop: feature-based
for fid, feat_row in tqdm(
    feat_gdf.iterrows(),
    total=len(feat_gdf),
    desc="Processing features"
):

    feat_geom = feat_row.geometry

    # Find grid cells intersecting this feature
    cand_idx = list(grid_sindex.intersection(feat_geom.bounds))
    if not cand_idx:
        continue

    records = []

    for gi in cand_idx:
        grid_geom = grid_gdf.geometry.iloc[gi]
        if not grid_geom.intersects(feat_geom):
            continue

        likelihood = compute_likelihood(grid_geom, feat_geom)

        if likelihood >= LIKELIHOOD_THRESHOLD:
            records.append({
                "grid_id": gi,
                "likelihood": likelihood
            })

    # No valid grid cell
    if len(records) == 0:
        continue

    df = pd.DataFrame(records)

    # Case 1: 3 or more grid cells
    if len(df) >= 3:
        selected_cells = grid_gdf.loc[df["grid_id"]]
        cx, cy = center_from_cells(selected_cells)

        results.append({
            "x": cx,
            "y": cy,
            "feature_id": fid,
            "type": "center"
        })

    # Case 2: 1 or 2 grid cells
    else:
        best_gid = df.sort_values(
            "likelihood", ascending=False
        ).iloc[0]["grid_id"]

        grid_geom = grid_gdf.geometry.loc[best_gid]
        pt = overlap_point(grid_geom, feat_geom)

        results.append({
            "x": pt.x,
            "y": pt.y,
            "feature_id": fid,
            "type": "slide"
        })

# Save output
if results:
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
else:
    print("No points generated")
