import geopandas as gpd
import numpy as np
import os

# --------------------------
# 1. Paths & Parameters
# --------------------------
INPUT_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp"
OUTPUT_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"

# Distance to shift in meters
# If a typical Amazon tree crown radius is 3-5m, setting it to 7-10m ensures it moves outside the crown.
SHIFT_DISTANCE_METERS = 8.0  

def main():
    print(f"Loading original shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    
    # Set seed for reproducibility (so we can recreate the exact same test set later)
    np.random.seed(42)
    
    # --------------------------
    # 2. Mathematical Calculation (Trigonometry)
    # --------------------------
    # Generate random angles between 0 and 2*pi (360 degrees) for each point
    angles = np.random.uniform(0, 2 * np.pi, len(gdf))
    
    # Calculate X, Y displacements (dx, dy) using the specified distance and angles
    dx = SHIFT_DISTANCE_METERS * np.cos(angles)
    dy = SHIFT_DISTANCE_METERS * np.sin(angles)
    
    # Add the displacements to the original coordinates
    new_x = gdf.geometry.x + dx
    new_y = gdf.geometry.y + dy
    
    # --------------------------
    # 3. Create and Save New Shapefile
    # --------------------------
    # Copy original attributes (ID, label, etc.) exactly as they are
    gdf_shifted = gdf.copy()
    
    # Overwrite the geometry column with the newly calculated coordinates
    gdf_shifted['geometry'] = gpd.points_from_xy(new_x, new_y, crs=gdf.crs)
    
    # Create directory if it doesn't exist, then save the file
    os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)
    gdf_shifted.to_file(OUTPUT_SHP)
    
    print(f"\nSuccessfully shifted {len(gdf)} points by exactly {SHIFT_DISTANCE_METERS}m in random directions.")
    print(f"Saved test dataset to: {OUTPUT_SHP}")

if __name__ == "__main__":
    main()