import pandas as pd
import geopandas as gpd
import numpy as np
import os

# 1. Configuration & Paths
# 1) Final aligned coordinates from the AI pipeline
FINAL_CSV = "data/result_coordinate/final_results.csv"

# 2) Original (noisy/shifted) coordinates before running the pipeline
ORIGINAL_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"

# 3) Ground Truth (actual correct) coordinates
# (Note: Please update this to your actual Ground Truth shapefile path!)
GROUND_TRUTH_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/ground_truth_32718.shp" 

TARGET_CRS = "EPSG:32718"

def main():
    print("Loading datasets for evaluation...")
    
    # Check if files exist
    if not os.path.exists(FINAL_CSV):
        raise FileNotFoundError(f"Error: Could not find {FINAL_CSV}. Please ensure the pipeline finished successfully.")
    if not os.path.exists(GROUND_TRUTH_SHP):
        raise FileNotFoundError(f"Error: Ground truth file not found at {GROUND_TRUTH_SHP}.")

    # Load Final Results
    final_df = pd.read_csv(FINAL_CSV)
    final_df["x"] = pd.to_numeric(final_df["x"], errors="coerce")
    final_df["y"] = pd.to_numeric(final_df["y"], errors="coerce")
    final_gdf = gpd.GeoDataFrame(
        final_df, 
        geometry=gpd.points_from_xy(final_df["x"], final_df["y"]),
        crs=TARGET_CRS
    )

    # Load Original and GT Shapefiles
    orig_gdf = gpd.read_file(ORIGINAL_SHP).to_crs(TARGET_CRS)
    gt_gdf = gpd.read_file(GROUND_TRUTH_SHP).to_crs(TARGET_CRS)

    # Unify IDs (handle cases where the ID column is named 'temp_id' or 'fid')
    # Make sure this matches the ID column name used in your previous scripts.
    orig_gdf["feature_id"] = orig_gdf.get("temp_id", orig_gdf.index)
    gt_gdf["feature_id"] = gt_gdf.get("temp_id", gt_gdf.index)

    # 2. Merge Data for Comparison
    # Merge Final with Ground Truth
    eval_df = final_gdf.merge(gt_gdf[["feature_id", "geometry"]], on="feature_id", suffixes=("_final", "_gt"))
    
    # Merge with Original to see improvement
    eval_df = eval_df.merge(orig_gdf[["feature_id", "geometry"]], on="feature_id")
    eval_df.rename(columns={"geometry": "geometry_orig"}, inplace=True)

    if eval_df.empty:
        print("Error: Could not match any feature_ids between the datasets. Check your ID columns.")
        return

    # 3. Calculate Distances (Errors)
    # Distance between Original and Ground Truth (Before)
    eval_df["error_before"] = eval_df.apply(
        lambda row: row["geometry_orig"].distance(row["geometry_gt"]), axis=1
    )
    
    # Distance between Final Aligned and Ground Truth (After)
    eval_df["error_after"] = eval_df.apply(
        lambda row: row["geometry_final"].distance(row["geometry_gt"]), axis=1
    )

    # Total distance the AI shifted the point
    eval_df["shift_distance"] = eval_df.apply(
        lambda row: row["geometry_orig"].distance(row["geometry_final"]), axis=1
    )

    # 4. Compute Metrics
    mae_before = eval_df["error_before"].mean()
    mae_after = eval_df["error_after"].mean()
    
    rmse_before = np.sqrt((eval_df["error_before"] ** 2).mean())
    rmse_after = np.sqrt((eval_df["error_after"] ** 2).mean())

    # Accuracy (Percentage of points falling within 5m and 2m thresholds)
    acc_5m_before = (eval_df["error_before"] <= 5.0).mean() * 100
    acc_5m_after = (eval_df["error_after"] <= 5.0).mean() * 100
    
    acc_2m_before = (eval_df["error_before"] <= 2.0).mean() * 100
    acc_2m_after = (eval_df["error_after"] <= 2.0).mean() * 100

    # ==========================================
    # 5. Print Report
    # ==========================================
    print("\n" + "="*50)
    print(" LeJEPA Alignment Evaluation Report")
    print("="*50)
    print(f"Total points evaluated: {len(eval_df)}")
    print("-" * 50)
    print(f"[Mean Absolute Error (MAE)]")
    print(f" - Before Alignment : {mae_before:.2f} meters")
    print(f" - After LeJEPA     : {mae_after:.2f} meters")
    print(f"   -> Improvement   : {mae_before - mae_after:.2f} meters")
    print("-" * 50)
    print(f"[Root Mean Square Error (RMSE)]")
    print(f" - Before Alignment : {rmse_before:.2f} meters")
    print(f" - After LeJEPA     : {rmse_after:.2f} meters")
    print("-" * 50)
    print(f"[Accuracy (Points within 5 meters)]")
    print(f" - Before Alignment : {acc_5m_before:.1f}%")
    print(f" - After LeJEPA     : {acc_5m_after:.1f}%")
    print("-" * 50)
    print(f"[Accuracy (Points within 2 meters)]")
    print(f" - Before Alignment : {acc_2m_before:.1f}%")
    print(f" - After LeJEPA     : {acc_2m_after:.1f}%")
    print("="*50)

    # Save the results to a CSV for point-by-point error analysis
    eval_output_csv = "evaluation_results.csv"
    eval_df.drop(columns=["geometry_final", "geometry_gt", "geometry_orig"]).to_csv(eval_output_csv, index=False)
    print(f"\nSaved detailed evaluation metrics to: {eval_output_csv}")

if __name__ == "__main__":
    main()