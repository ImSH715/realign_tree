"""
    Check if there is any points out of bound from the tif file.
    Then extract the points into .shp file excluding missing points.
"""

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
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp"

OUTPUT_DIR = "data/label"
OUTPUT_MISSING_SHP = os.path.join(OUTPUT_DIR, "missing_points.shp")
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

    # --- Out of Bounds Calculation ---
    missing_gdf = gdf[~gdf['temp_id'].isin(extracted_ids)]
    
    # Merge footprints
    if valid_footprints:
        footprints_gdf = pd.concat(valid_footprints, ignore_index=True)
        # Create a single large polygon from all valid TIF bounds for out-of-bounds detection
        total_valid_area = footprints_gdf.unary_union 
        
        # Check actual out of bounds: filter points that do not intersect with the merged area
        out_of_bounds_mask = ~missing_gdf.geometry.intersects(total_valid_area)
        out_of_bounds_count = out_of_bounds_mask.sum()
    else:
        out_of_bounds_count = len(missing_gdf)

    print("\n" + "="*50)
    print("Scanning Complete")
    print("="*50)
    print(f"Total Points: {total_points}")
    print(f"Found (Inside valid TIFs): {len(extracted_ids)}")
    print(f"Missing Points: {len(missing_gdf)}")
    print(f"  -> Out of bounds count (completely outside valid TIFs): {out_of_bounds_count}")
    
    if error_files:
        print(f"\nWarning: {len(error_files)} TIF files encountered errors (cannot verify bounds):")
        for err_file, err_msg in error_files:
            print(f"  - {err_file}")
            
    # Save results
    if not missing_gdf.empty:
        missing_gdf.to_file(OUTPUT_MISSING_SHP)
    if valid_footprints:
        footprints_gdf.to_file(OUTPUT_FOOTPRINTS_SHP)
        
    print(f"\nFiles saved successfully!")
    print(f"  - Missing points: {OUTPUT_MISSING_SHP}")
    print(f"  - Valid TIF footprints: {OUTPUT_FOOTPRINTS_SHP}")
    print("="*50)

if __name__ == "__main__":
    main()