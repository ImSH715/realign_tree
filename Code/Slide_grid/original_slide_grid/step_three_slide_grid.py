import geopandas as gpd
import pandas as pd
from tqdm import tqdm
"""
Do same procedure as step one.
If center points are made as the first condition of the step one,
turn type 'slide' to 'center'
"""
# Config
GRID_SHP = "../shapefiles/step2_slide_grids.shp"
FEATURE_SHP = "../shapefiles/orange_trees.shp"
STEP1_CSV = "../coordinate/step1_points.csv"

OUTPUT_CSV = "../coordinate/step3_points.csv"

LIKELIHOOD_THRESHOLD = 0.5

# Helper functions
def compute_likelihood(grid_geom, feature_geom):
    inter = grid_geom.intersection(feature_geom)
    if inter.is_empty:
        return 0.0
    return inter.area / grid_geom.area


def center_from_grid_cells(cells):
    xs = [geom.centroid.x for geom in cells.geometry]
    ys = [geom.centroid.y for geom in cells.geometry]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def overlap_point(grid_geom, feature_geom):
    return grid_geom.intersection(feature_geom).centroid

# Load data
grid_gdf = gpd.read_file(GRID_SHP)
feat_gdf = gpd.read_file(FEATURE_SHP)

step1_df = pd.read_csv(STEP1_CSV)
slide_features = step1_df[step1_df["type"] == "slide"]["feature_id"].unique()

results = []

# Main loop (slide features only)
for fid in tqdm(
    slide_features,
    total=len(slide_features),
    desc="Step 3: slide → center"
):

    feat_geom = feat_gdf.loc[fid].geometry

    # grids created from slide points
    grids = grid_gdf[grid_gdf["point_id"] == fid]

    if grids.empty:
        continue

    records = []

    for gi, grid_row in grids.iterrows():
        grid_geom = grid_row.geometry

        if not grid_geom.intersects(feat_geom):
            continue

        likelihood = compute_likelihood(grid_geom, feat_geom)

        if likelihood >= LIKELIHOOD_THRESHOLD:
            records.append({
                "grid_index": gi,
                "likelihood": likelihood
            })

    if not records:
        continue

    df = pd.DataFrame(records)

    # Case A: 3 or more boxes
    if len(df) >= 3:
        selected = grid_gdf.loc[df["grid_index"]]
        cx, cy = center_from_grid_cells(selected)

    # Case B: 1–2 boxes
    else:
        best_idx = df.sort_values(
            "likelihood", ascending=False
        ).iloc[0]["grid_index"]

        pt = overlap_point(
            grid_gdf.geometry.loc[best_idx],
            feat_geom
        )
        cx, cy = pt.x, pt.y

    results.append({
        "x": cx,
        "y": cy,
        "feature_id": fid,
        "type": "center"
    })

# Save
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Total centers (slide → center):", len(out_df))
