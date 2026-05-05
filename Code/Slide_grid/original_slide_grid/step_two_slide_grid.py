import pandas as pd
import geopandas as gpd
from shapely.geometry import box

"""
Scan type 'slide' from the step1_points_lejepa.csv
Generate a new 3x3 grid just for those coordinates with type 'slide'
and generate a shapefile for the next step.
"""

# ==========================================
# 1. Configuration & Paths
# ==========================================
STEP1_CSV = "step1_points_lejepa.csv"
OUTPUT_GRID_SHP = "step2_slide_grids_lejepa.shp"

CELL_SIZE = 5.2  # Grid cell size in meters
TARGET_CRS = "EPSG:32718"  # Set the CRS to match your project

def main():
    print(f"Loading Step 1 results from: {STEP1_CSV}")
    try:
        df = pd.read_csv(STEP1_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {STEP1_CSV}. Please run Step 1 first.")

    # Use only points that require a slide
    df_slide = df[df["type"] == "slide"].copy()

    if df_slide.empty:
        print("No slide points found in Step 1 result. The algorithm can terminate early.")
        return

    print(f"Found {len(df_slide)} points requiring a slide.")

    # Ensure coordinates are numeric
    df_slide["x"] = pd.to_numeric(df_slide["x"], errors="coerce")
    df_slide["y"] = pd.to_numeric(df_slide["y"], errors="coerce")
    df_slide = df_slide.dropna(subset=["x", "y"]).reset_index(drop=True)

    # Create point GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        df_slide,
        geometry=gpd.points_from_xy(df_slide["x"], df_slide["y"]),
        crs=TARGET_CRS
    )

    # ==========================================
    # 2. Generate 3x3 Grids
    # ==========================================
    records = []
    geometries = []
    
    half = CELL_SIZE / 2.0

    print("Generating 3x3 grids for each slide point...")
    for _, row in points_gdf.iterrows():
        cx = row.geometry.x
        cy = row.geometry.y
        feature_id = row["feature_id"]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Calculate bounding box coordinates for each cell
                x_min = cx + dx * CELL_SIZE - half
                x_max = cx + dx * CELL_SIZE + half
                y_min = cy + dy * CELL_SIZE - half
                y_max = cy + dy * CELL_SIZE + half

                records.append({
                    "point_id": feature_id,
                    "dx": dx,
                    "dy": dy,
                    "source": "slide"
                })

                geometries.append(box(x_min, y_min, x_max, y_max))

    # Create grid GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs=TARGET_CRS
    )

    # ==========================================
    # 3. Save Output
    # ==========================================
    grid_gdf.to_file(OUTPUT_GRID_SHP)
    print(f"Saved generated grids to: {OUTPUT_GRID_SHP}")
    print(f"Total slide points processed: {grid_gdf['point_id'].nunique()} (Total cells generated: {len(grid_gdf)})")

if __name__ == "__main__":
    main()