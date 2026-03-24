import geopandas as gpd
import numpy as np
import os

# --------------------------
# 1. Paths & Parameters
# --------------------------
MIN_SHIFT = 5   # Minimum shift distance in meters
MAX_SHIFT = 15  # Maximum shift distance in meters

INPUT_SHP = r"data/tree_label_rdn/valid_points.shp"
OUTPUT_SHP = f"data/tree_label_rdn/random_valid_range_{MIN_SHIFT}_{MAX_SHIFT}.shp"

def main():
    if not os.path.exists(INPUT_SHP):
        print(f"Error: File not found at {INPUT_SHP}")
        return

    print(f"Loading original shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    
    # Set seed for reproducibility
    np.random.seed(42)
    num_points = len(gdf)
    
    # --------------------------
    # 2. Random Displacement Calculation
    # --------------------------
    # Generate unique random angles (0 to 2*pi) for each point
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    
    # Generate unique random distances within the specified range for each point
    random_distances = np.random.uniform(MIN_SHIFT, MAX_SHIFT, num_points)
    
    # Calculate X, Y displacements (dx, dy) using trigonometry
    # dx = r * cos(theta), dy = r * sin(theta)
    dx = random_distances * np.cos(angles)
    dy = random_distances * np.sin(angles)
    
    # --------------------------
    # 3. Apply Displacement and Save
    # --------------------------
    # Copy original attributes
    gdf_shifted = gdf.copy()
    
    # Update coordinates by adding the random displacement
    new_x = gdf.geometry.x + dx
    new_y = gdf.geometry.y + dy
    
    # Overwrite the geometry column with new point objects
    gdf_shifted['geometry'] = gpd.points_from_xy(new_x, new_y, crs=gdf.crs)
    
    # Create output directory if it doesn't exist and save the file
    os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)
    gdf_shifted.to_file(OUTPUT_SHP)
    
    print(f"\nSuccess: Shifted {num_points} points randomly between {MIN_SHIFT}m and {MAX_SHIFT}m.")
    print(f"Saved to: {OUTPUT_SHP}")

if __name__ == "__main__":
    main()