import os
import glob
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

# --------------------------
# Configuration & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp"

OUTPUT_DIR = "data/label"
OUTPUT_MISSING_SHP = os.path.join(OUTPUT_DIR, "missing_points.shp")
OUTPUT_VALID_POINTS_SHP = os.path.join(OUTPUT_DIR, "valid_points.shp") # New output for in-bound points
OUTPUT_FOOTPRINTS_SHP = os.path.join(OUTPUT_DIR, "valid_tif_footprints.shp")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading original shapefile: {ANNOTATED_COR}")
    gdf = gpd.read_file(ANNOTATED_COR)
    
    # Ensure a unique ID exists for tracking
    if 'temp_id' not in gdf.columns:
        gdf['temp_id'] = range(len(gdf))
        
    total_points = len(gdf)
    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    
    if not tif_files:
        print("Error: No TIF files found. Please check the BASE_DIR and glob pattern.")
        return

    extracted_ids = set()
    error_files = []
    valid_footprints = [] # List to store the bounding boxes of valid TIFs

    print("\nScanning TIF bounds to find intersecting points...")
    for tif_path in tqdm(tif_files, desc="Checking TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                bounds = src.bounds
                img_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                bbox_gdf = gpd.GeoDataFrame({'geometry': [img_box], 'filename': os.path.basename(tif_path)}, crs=src.crs)
                
                # Align CRS if necessary
                if bbox_gdf.crs != gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(gdf.crs)

                # 1. Store the bounds of successfully read TIFs
                valid_footprints.append(bbox_gdf)

                # 2. Collect IDs of points that intersect with this bounding box
                intersecting = gdf[gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
                for _, row in intersecting.iterrows():
                    extracted_ids.add(row['temp_id'])
                    
        except Exception as e:
            error_files.append((os.path.basename(tif_path), str(e)))
            continue

    # --- Filter Points ---
    # Points found inside any TIF (In-bounds)
    valid_points_gdf = gdf[gdf['temp_id'].isin(extracted_ids)]
    
    # Points not found in any TIF (Missing)
    missing_gdf = gdf[~gdf['temp_id'].isin(extracted_ids)]
    
    # --- Out of Bounds Calculation ---
    if valid_footprints:
        footprints_gdf = pd.concat(valid_footprints, ignore_index=True)
        # Create a single large polygon from all valid TIF bounds
        total_valid_area = footprints_gdf.unary_union 
        
        # Check actual out of bounds for missing points
        out_of_bounds_mask = ~missing_gdf.geometry.intersects(total_valid_area)
        out_of_bounds_count = out_of_bounds_mask.sum()
    else:
        footprints_gdf = gpd.GeoDataFrame()
        out_of_bounds_count = len(missing_gdf)

    print("\n" + "="*50)
    print("Scanning Complete")
    print("="*50)
    print(f"Total Points: {total_points}")
    print(f"Valid Points (In-bound): {len(valid_points_gdf)}")
    print(f"Missing Points: {len(missing_gdf)}")
    print(f"  -> Out of bounds count: {out_of_bounds_count}")
    
    if error_files:
        print(f"\nWarning: {len(error_files)} TIF files encountered errors:")
        for err_file, err_msg in error_files:
            print(f"  - {err_file}")
            
    # --- Save Results ---
    # Save valid points (In-bound)
    if not valid_points_gdf.empty:
        valid_points_gdf.to_file(OUTPUT_VALID_POINTS_SHP)
        
    # Save missing points
    if not missing_gdf.empty:
        missing_gdf.to_file(OUTPUT_MISSING_SHP)
        
    # Save TIF footprints
    if not footprints_gdf.empty:
        footprints_gdf.to_file(OUTPUT_FOOTPRINTS_SHP)
        
    print(f"\nFiles saved successfully!")
    print(f"  - Valid points: {OUTPUT_VALID_POINTS_SHP}")
    print(f"  - Missing points: {OUTPUT_MISSING_SHP}")
    print(f"  - Valid TIF footprints: {OUTPUT_FOOTPRINTS_SHP}")
    print("="*50)

if __name__ == "__main__":
    main()