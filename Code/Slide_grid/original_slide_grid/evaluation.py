import pandas as pd
import geopandas as gpd
import numpy as np
import os

# ==========================================
# 1. Configuration & Paths
# ==========================================
FINAL_RESULTS_PATH = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/original_slide_grid/data/slided_coordinate/final_results.shp" 
ORIGINAL_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"
GROUND_TRUTH_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp" 

TARGET_CRS = "EPSG:32718"

CROWN_RADIUS = 10.0  

def main():
    print("Loading datasets for evaluation...")
    
    if not os.path.exists(FINAL_RESULTS_PATH):
        raise FileNotFoundError(f"Error: Could not find {FINAL_RESULTS_PATH}.")
    if not os.path.exists(GROUND_TRUTH_SHP):
        raise FileNotFoundError(f"Error: Ground truth file not found at {GROUND_TRUTH_SHP}.")

    # ==========================================
    # 2. Load and Prepare Data
    # ==========================================
    if FINAL_RESULTS_PATH.lower().endswith('.csv'):
        final_df = pd.read_csv(FINAL_RESULTS_PATH)
        final_df["x"] = pd.to_numeric(final_df["x"], errors="coerce")
        final_df["y"] = pd.to_numeric(final_df["y"], errors="coerce")
        final_df = final_df.dropna(subset=["x", "y"])
        
        final_gdf = gpd.GeoDataFrame(
            final_df, 
            geometry=gpd.points_from_xy(final_df["x"], final_df["y"]),
            crs=TARGET_CRS
        )
    elif FINAL_RESULTS_PATH.lower().endswith('.shp'):
        final_gdf = gpd.read_file(FINAL_RESULTS_PATH)
        if final_gdf.crs is None or final_gdf.crs != TARGET_CRS:
            final_gdf = final_gdf.to_crs(TARGET_CRS)
    else:
        raise ValueError("Error: FINAL_RESULTS_PATH must be a .csv or .shp file.")

    orig_gdf = gpd.read_file(ORIGINAL_SHP).to_crs(TARGET_CRS)
    gt_gdf = gpd.read_file(GROUND_TRUTH_SHP).to_crs(TARGET_CRS)

    # 매칭을 위해 id 통일 (temp_id가 없으면 index를 id로 사용)
    final_gdf["id"] = final_gdf.get("temp_id", final_gdf.index)
    orig_gdf["id"] = orig_gdf.get("temp_id", orig_gdf.index)
    gt_gdf["id"] = gt_gdf.get("temp_id", gt_gdf.index)

    # ==========================================
    # 3. Merge Data for Comparison
    # ==========================================
    eval_df = final_gdf.merge(gt_gdf[["id", "geometry"]], on="id", suffixes=("_final", "_gt"))
    eval_df = eval_df.merge(orig_gdf[["id", "geometry"]], on="id")
    eval_df.rename(columns={"geometry": "geometry_orig"}, inplace=True)

    if eval_df.empty:
        print("Error: Could not match any 'id' between datasets. Please check if indices or 'temp_id' align properly.")
        return

    # ==========================================
    # 4. Calculate Distances (Errors)
    # ==========================================
    eval_df["error_before"] = eval_df.apply(
        lambda row: row["geometry_orig"].distance(row["geometry_gt"]), axis=1
    )
    eval_df["error_after"] = eval_df.apply(
        lambda row: row["geometry_final"].distance(row["geometry_gt"]), axis=1
    )
    eval_df["shift_distance"] = eval_df.apply(
        lambda row: row["geometry_orig"].distance(row["geometry_final"]), axis=1
    )

    # ==========================================
    # 5. Compute "Tree Crown" Metrics
    # ==========================================
    total_points = len(eval_df)
    
    eval_df["inside_crown_before"] = eval_df["error_before"] <= CROWN_RADIUS
    eval_df["inside_crown_after"] = eval_df["error_after"] <= CROWN_RADIUS
    
    crown_hit_before = eval_df["inside_crown_before"].sum()
    crown_hit_after = eval_df["inside_crown_after"].sum()

    stable_inside = ((eval_df["inside_crown_before"] == True) & (eval_df["inside_crown_after"] == True)).sum()
    lost_outside = ((eval_df["inside_crown_before"] == True) & (eval_df["inside_crown_after"] == False)).sum()
    moved_closer = (eval_df["error_after"] < eval_df["error_before"]).sum()

    # ==========================================
    # 6. Print Report
    # ==========================================
    print("\n" + "="*55)
    print("LeJEPA Tree Crown & Alignment Evaluation")
    print("="*55)
    print(f"Total points evaluated: {total_points}")
    print(f"Assumed Tree Crown Radius: {CROWN_RADIUS} meters")
    print("-" * 55)
    
    print(f"[ 1. Canopy Hit Rate (Points inside Tree Crown) ]")
    print(f" - Before Algorithm : {crown_hit_before} points ({(crown_hit_before/total_points)*100:.1f}%)")
    print(f" - After Algorithm  : {crown_hit_after} points ({(crown_hit_after/total_points)*100:.1f}%)")
    
    print("\n[ 2. Detailed Movement Analysis ]")
    print(f" - Stable (Stayed inside crown)  : {stable_inside} points")
    print(f" - Lost (Wandered outside crown) : {lost_outside} points")
    
    print("-" * 55)
    print(f"[ 3. General Improvement ]")
    print(f" - Points that moved CLOSER to the ground truth center:")
    print(f"   {moved_closer} points ({(moved_closer/total_points)*100:.1f}%)")
    
    print("-" * 55)
    print(f"[ 4. Absolute Distance Error (For reference) ]")
    print(f" - Mean Error Before : {eval_df['error_before'].mean():.2f} m")
    print(f" - Mean Error After  : {eval_df['error_after'].mean():.2f} m")
    print(f" - Avg. Distance moved the point : {eval_df['shift_distance'].mean():.2f} m")
    print("="*55)

    # Save detailed output
    eval_output_csv = r"data/coordinate/sliding_results/crown_evaluation_metrics.csv"
    os.makedirs(os.path.dirname(eval_output_csv), exist_ok=True)
    
    clean_eval_df = eval_df.drop(columns=["geometry_final", "geometry_gt", "geometry_orig"])
    clean_eval_df.to_csv(eval_output_csv, index=False)
    
    print(f"\nSaved detailed point-by-point evaluation to: {eval_output_csv}")

if __name__ == "__main__":
    main()