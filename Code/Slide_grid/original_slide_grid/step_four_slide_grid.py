import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# ==========================================
# 1. Configuration & Paths
# ==========================================
INPUT_CSV = "step3_points_lejepa.csv"
OUTPUT_GRID_SHP = "step4_slide_grids_lejepa.shp"

CELL_SIZE = 5.2  
TARGET_CRS = "EPSG:32718"  

def main():
    print(f"Loading Step 3 results from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {INPUT_CSV}. Run Step 3 first.")

    # Use only points that STILL require a slide
    df_slide = df[df["type"] == "slide"].copy()

    if df_slide.empty:
        print("No more slide points! The algorithm can terminate.")
        return

    print(f"Found {len(df_slide)} points requiring another slide.")

    df_slide["x"] = pd.to_numeric(df_slide["x"], errors="coerce")
    df_slide["y"] = pd.to_numeric(df_slide["y"], errors="coerce")
    df_slide = df_slide.dropna(subset=["x", "y"]).reset_index(drop=True)

    points_gdf = gpd.GeoDataFrame(
        df_slide, geometry=gpd.points_from_xy(df_slide["x"], df_slide["y"]), crs=TARGET_CRS
    )

    records = []
    geometries = []
    half = CELL_SIZE / 2.0

    print("Generating new 3x3 grids...")
    for _, row in points_gdf.iterrows():
        cx, cy, feature_id = row.geometry.x, row.geometry.y, row["feature_id"]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x_min, x_max = cx + dx * CELL_SIZE - half, cx + dx * CELL_SIZE + half
                y_min, y_max = cy + dy * CELL_SIZE - half, cy + dy * CELL_SIZE + half
                records.append({"point_id": feature_id, "dx": dx, "dy": dy, "source": "slide_v2"})
                geometries.append(box(x_min, y_min, x_max, y_max))

    grid_gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=TARGET_CRS)
    grid_gdf.to_file(OUTPUT_GRID_SHP)
    print(f"Saved: {OUTPUT_GRID_SHP} (Total cells: {len(grid_gdf)})")

if __name__ == "__main__":
    main()