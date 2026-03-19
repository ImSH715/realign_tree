import pandas as pd
import geopandas as gpd
import numpy as np
import os

# ==========================================
# 1. Configuration & Paths
# ==========================================
# 1) Final aligned coordinates from the Sliding Grid algorithm
FINAL_CSV = r"data/coordinate/sliding_results/final_results.csv"

# 2) Original (noisy/shifted) coordinates before running the algorithm
ORIGINAL_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"

# 3) Ground Truth (actual correct) coordinates
# (Note: Please update this to your actual Ground Truth shapefile path!)
GROUND_TRUTH_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/ground_truth_32718.shp" 

TARGET_CRS = "EPSG:32718"

def main():
    print("Loading datasets for evaluation...")
    
    # Check if files exist
    if not os.path.exists(FINAL_CSV):
        raise FileNotFoundError(f"Error: Could not find {FINAL_CSV}. Please run the sliding algorithm first.")
    if not os.path.exists(GROUND_TRUTH_SHP):
        raise FileNotFoundError(f"Error: Ground truth file not found at {GROUND_TRUTH_SHP}.")

    # ==========================================
    # 2. Load and Prepare Data
    # ==========================================
    # Load Final Results (CSV -> GeoDataFrame)
    final_df = pd.read_csv(FINAL_CSV)
    final_df["x"] = pd.to_numeric(final_df["x"], errors="coerce")
    final_df["y"] = pd.to_numeric(final_df["y"], errors="coerce")
    final_df = final_df.dropna(subset=["x", "y"])
    
    final_gdf = gpd.GeoDataFrame(
        final_df, 
        geometry=gpd.points_from_xy(final_df["x"], final_df["y"]),
        crs=TARGET_CRS
    )

    # Load Original and Ground Truth Shapefiles
    orig_gdf = gpd.read_file(ORIGINAL_SHP).to_crs(TARGET_CRS)
    gt_gdf = gpd.read_file(GROUND_TRUTH_SHP).to_crs(TARGET_CRS)

    # Unify ID column names to 'id' to match the final_results.csv
    # Adjust 'temp_id' if your shapefiles use a different column name for IDs
    orig_gdf["id"] = orig_gdf.get("temp_id", orig_gdf.index)
    gt_gdf["id"] = gt_gdf.get("temp_id", gt_gdf.index)

    # ==========================================
    # 3. Merge Data for Comparison
    # ==========================================
    # Merge Final with Ground Truth
    eval_df = final_gdf.merge(gt_gdf[["id", "geometry"]], on="id", suffixes=("_final", "_gt"))
    
    # Merge with Original to see the improvement
    eval_df = eval_df.merge(orig_gdf[["id", "geometry"]], on="id")
    eval_df.rename(columns={"geometry": "geometry_orig"}, inplace=True)

    if eval_df.empty:
        print("Error: Could not match any 'id' between the datasets. Please check your ID columns.")
        return

    # ==========================================
    # 4. Calculate Distances (Errors)
    # ==========================================
    # Distance between Original and Ground Truth (Error BEFORE)
    eval_df["error_before"] = eval_df.apply(
        lambda row: row["geometry_orig"].distance(row["geometry_gt"]), axis=1
    )
    
    # Distance between Final Aligned and Ground Truth (Error AFTER)
    eval_df["error_after"] = eval_df.apply(
        lambda row: row["geometry_final"].distance(row["geometry_gt"]), axis=1
    )

    # Total distance the AI shifted the point
    eval_df["shift_distance"] = eval_df.apply(
        lambda row: row["geometry_orig"].distance(row["geometry_final"]), axis=1
    )

    # ==========================================
    # 5. Compute Metrics
    # ==========================================
    mae_before = eval_df["error_before"].mean()
    mae_after = eval_df["error_after"].mean()
    
    rmse_before = np.sqrt((eval_df["error_before"] ** 2).mean())
    rmse_after = np.sqrt((eval_df["error_after"] ** 2).mean())

    # Accuracy (Percentage of points falling within specific thresholds)
    acc_5m_before = (eval_df["error_before"] <= 5.0).mean() * 100
    acc_5m_after = (eval_df["error_after"] <= 5.0).mean() * 100
    
    acc_2m_before = (eval_df["error_before"] <= 2.0).mean() * 100
    acc_2m_after = (eval_df["error_after"] <= 2.0).mean() * 100

    # ==========================================
    # 6. Print Report
    # ==========================================
    print("\n" + "="*50)
    print(" 🎯 LeJEPA Sliding Grid - Evaluation Report")
    print("="*50)
    print(f"Total points evaluated: {len(eval_df)}")
    print("-" * 50)
    print(f"[ Mean Absolute Error (MAE) ]")
    print(f" - Before Algorithm : {mae_before:.2f} meters")
    print(f" - After Algorithm  : {mae_after:.2f} meters")
    print(f"   -> Average Shift : {eval_df['shift_distance'].mean():.2f} meters")
    print("-" * 50)
    print(f"[ Root Mean Square Error (RMSE) ]")
    print(f" - Before Algorithm : {rmse_before:.2f} meters")
    print(f" - After Algorithm  : {rmse_after:.2f} meters")
    print("-" * 50)
    print(f"[ Accuracy (Points within 5 meters of Ground Truth) ]")
    print(f" - Before Algorithm : {acc_5m_before:.1f}%")
    print(f" - After Algorithm  : {acc_5m_after:.1f}%")
    print("-" * 50)
    print(f"[ Accuracy (Points within 2 meters of Ground Truth) ]")
    print(f" - Before Algorithm : {acc_2m_before:.1f}%")
    print(f" - After Algorithm  : {acc_2m_after:.1f}%")
    print("="*50)

    # ==========================================
    # 7. Save Detailed Output
    # ==========================================
    # Save the results to a CSV for point-by-point error analysis
    eval_output_csv = r"data/coordinate/sliding_results/evaluation_metrics.csv"
    
    # Drop raw geometry columns to save a clean CSV
    clean_eval_df = eval_df.drop(columns=["geometry_final", "geometry_gt", "geometry_orig"])
    clean_eval_df.to_csv(eval_output_csv, index=False)
    
    print(f"\nSaved detailed point-by-point evaluation to: {eval_output_csv}")

if __name__ == "__main__":
    main()