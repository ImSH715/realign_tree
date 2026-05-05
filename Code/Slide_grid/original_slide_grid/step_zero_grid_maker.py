import geopandas as gpd
from shapely.geometry import box
import os

# 1. Configuration
shp_path = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"
cell_size = 5.2  # Grid cell size

# FIX: Added a filename (e.g., result.shp) to the end of the path
output_dir = r"grid_result"
output_shp = os.path.join(output_dir, "3x3_grid_result.shp")

# 2. Check if input file exists
if not os.path.exists(shp_path):
    raise FileNotFoundError(f"The system cannot find the input path: {shp_path}")

# 3. Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# 4. Load the Shapefile
points = gpd.read_file(shp_path)

if points.empty:
    raise ValueError("The input Shapefile is empty.")

# Ensure the CRS is set
if points.crs is None:
    points.set_crs("EPSG:32718", inplace=True)

# 5. Generate 3x3 Grid per Point
records = []
geometries = []
half = cell_size / 2

for idx, row in points.iterrows():
    cx = row.geometry.x
    cy = row.geometry.y

    # Create 9 cells (3x3) centered around each point
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x_min = cx + (dx * cell_size) - half
            x_max = cx + (dx * cell_size) + half
            y_min = cy + (dy * cell_size) - half
            y_max = cy + (dy * cell_size) + half
            
            records.append({
                "orig_idx": idx, 
                "dx": dx,
                "dy": dy
            })
            geometries.append(
                box(x_min, y_min, x_max, y_max)
            )

# 6. Create Grid GeoDataFrame
grid_gdf = gpd.GeoDataFrame(
    records,
    geometry=geometries,
    crs=points.crs
)

# 7. Export to Shapefile
grid_gdf.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")
print(f"DONE: 3x3 grid saved to {output_shp}.")