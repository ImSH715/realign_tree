import os
import glob
import rasterio
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

# --------------------------
# Configuration & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp"

# Output path for the missing points
OUTPUT_DIR = "data/label"
OUTPUT_MISSING_SHP = os.path.join(OUTPUT_DIR, "missing_points.shp")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading original shapefile: {ANNOTATED_COR}")
    gdf = gpd.read_file(ANNOTATED_COR)
    
    # Ensure a unique ID exists for tracking
    if 'temp_id' not in gdf.columns:
        gdf['temp_id'] = range(len(gdf))
        
    total_points = len(gdf)
    print(f"Total points in SHP: {total_points}")

    # Find all TIF files
    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    print(f"Found {len(tif_files)} TIF files in {BASE_DIR}")

    if not tif_files:
        print("Error: No TIF files found. Please check the BASE_DIR and glob pattern.")
        return

    extracted_ids = set()
    error_files = []

    print("\nScanning TIF bounds to find intersecting points...")
    for tif_path in tqdm(tif_files, desc="Checking TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                bounds = src.bounds
                img_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                bbox_gdf = gpd.GeoDataFrame({'geometry': [img_box]}, crs=src.crs)
                
                # Align CRS if necessary
                if bbox_gdf.crs != gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(gdf.crs)

                # Find points that intersect with this specific TIF
                intersecting = gdf[gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
                
                if intersecting.empty:
                    continue
                    
                # Add found points to our set
                for _, row in intersecting.iterrows():
                    extracted_ids.add(row['temp_id'])
                    
        except Exception as e:
            error_files.append((os.path.basename(tif_path), str(e)))
            continue

    # Identify the missing points
    missing_gdf = gdf[~gdf['temp_id'].isin(extracted_ids)]
    num_missing = len(missing_gdf)
    num_found = len(extracted_ids)

    print("\n" + "="*50)
    print("✨ Scanning Complete ✨")
    print("="*50)
    print(f"Total Points: {total_points}")
    print(f"Found (Inside TIFs): {num_found}")
    print(f"Missing (Outside/Errors): {num_missing}")
    
    if error_files:
        print(f"\n⚠️ Warning: {len(error_files)} TIF files could not be read:")
        for err_file, err_msg in error_files[:5]: # Print first 5 errors
            print(f"  - {err_file}: {err_msg}")
        if len(error_files) > 5:
            print(f"  ... and {len(error_files) - 5} more.")

    # Save the missing points to a new shapefile
    if not missing_gdf.empty:
        missing_gdf.to_file(OUTPUT_MISSING_SHP)
        print(f"\nSaved {num_missing} missing points to: {OUTPUT_MISSING_SHP}")
    else:
        print("\nPerfect! No missing points found. All points are covered by the TIF files.")
    print("="*50)

if __name__ == "__main__":
    main()