import pandas as pd
import geopandas as gpd
from shapely.geometry import box
"""
Scan type 'slide' from the step1_points.csv
Generate new 3x3 grid just for those coordinates with type 'slide'
generate shapefile.
"""
# Config
STEP1_CSV = "../coordinate/step1_points.csv"
OUTPUT_GRID_SHP = "../shapefiles/step2_slide_grids.shp"

CELL_SIZE = 1.2  # grid cell size

# Load step1 result
df = pd.read_csv(STEP1_CSV)

# Use only slide points
df = df[df["type"] == "slide"].copy()

if df.empty:
    raise ValueError("No slide points found in step1 result.")

df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

# Create point GeoDataFrame
points_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["x"], df["y"]),
    crs=None
)

# Create 3x3 grid centered on slide point
records = []
geometries = []

half = CELL_SIZE / 2

for pid, row in points_gdf.iterrows():
    cx = row.geometry.x
    cy = row.geometry.y

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x_min = cx + dx * CELL_SIZE - half
            x_max = cx + dx * CELL_SIZE + half
            y_min = cy + dy * CELL_SIZE - half
            y_max = cy + dy * CELL_SIZE + half

            records.append({
                "point_id": row["feature_id"],
                "dx": dx,
                "dy": dy,
                "source": "slide"
            })

            geometries.append(
                box(x_min, y_min, x_max, y_max)
            )

# Create grid GeoDataFrame
grid_gdf = gpd.GeoDataFrame(
    records,
    geometry=geometries,
    crs=None
)

# Save
grid_gdf.to_file(OUTPUT_GRID_SHP)
print(f"Saved: {OUTPUT_GRID_SHP}")
print(f"Total slide grids created: {grid_gdf['point_id'].nunique()}")
