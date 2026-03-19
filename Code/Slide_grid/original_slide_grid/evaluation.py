import pandas as pd
import geopandas as gpd
import numpy as np
import os

# ==========================================
# 1. Configuration & Paths
# ==========================================
FINAL_CSV = r"data/coordinate/sliding_results/final_results.csv"
ORIGINAL_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"
GROUND_TRUTH_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/ground_truth_32718.shp" 

TARGET_CRS = "EPSG:32718"

# 🌳 수관 반경 (Tree Crown Radius) 설정 (단위: 미터)
# 이 반경 안에 포인트가 떨어지면 "같은 나무를 맞췄다(Hit)"고 인정합니다.
# 평균적인 나무 크기나, 원래 틀어진 거리(8m)를 고려하여 10m 정도로 설정해 봅니다.
CROWN_RADIUS = 10.0  

def main():
    print("Loading datasets for evaluation...")
    
    if not os.path.exists(FINAL_CSV):
        raise FileNotFoundError(f"Error: Could not find {FINAL_CSV}.")
    if not os.path.exists(GROUND_TRUTH_SHP):
        raise FileNotFoundError(f"Error: Ground truth file not found at {GROUND_TRUTH_SHP}.")

    # ==========================================
    # 2. Load and Prepare Data
    # ==========================================
    final_df = pd.read_csv(FINAL_CSV)
    final_df["x"] = pd.to_numeric(final_df["x"], errors="coerce")
    final_df["y"] = pd.to_numeric(final_df["y"], errors="coerce")
    final_df = final_df.dropna(subset=["x", "y"])
    
    final_gdf = gpd.GeoDataFrame(
        final_df, 
        geometry=gpd.points_from_xy(final_df["x"], final_df["y"]),
        crs=TARGET_CRS
    )

    orig_gdf = gpd.read_file(ORIGINAL_SHP).to_crs(TARGET_CRS)
    gt_gdf = gpd.read_file(GROUND_TRUTH_SHP).to_crs(TARGET_CRS)

    orig_gdf["id"] = orig_gdf.get("temp_id", orig_gdf.index)
    gt_gdf["id"] = gt_gdf.get("temp_id", gt_gdf.index)

    # ==========================================
    # 3. Merge Data for Comparison
    # ==========================================
    eval_df = final_gdf.merge(gt_gdf[["id", "geometry"]], on="id", suffixes=("_final", "_gt"))
    eval_df = eval_df.merge(orig_gdf[["id", "geometry"]], on="id")
    eval_df.rename(columns={"geometry": "geometry_orig"}, inplace=True)

    if eval_df.empty:
        print("Error: Could not match any 'id' between datasets.")
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
    # 5. Compute "Tree Crown" Metrics (수관 기준 평가)
    # ==========================================
    total_points = len(eval_df)
    
    # 1) 수관(Crown) 반경 내에 존재하는가?
    eval_df["inside_crown_before"] = eval_df["error_before"] <= CROWN_RADIUS
    eval_df["inside_crown_after"] = eval_df["error_after"] <= CROWN_RADIUS
    
    crown_hit_before = eval_df["inside_crown_before"].sum()
    crown_hit_after = eval_df["inside_crown_after"].sum()

    # 2) 상태 변화 추적
    # - 원래도 수관 안이었고, 지금도 수관 안인 경우 (잘 버텼다!)
    stable_inside = ((eval_df["inside_crown_before"] == True) & (eval_df["inside_crown_after"] == True)).sum()
    
    # - 수관 밖이었는데, 알고리즘이 수관 안으로 데리고 온 경우 (구출됨!)
    rescued_inside = ((eval_df["inside_crown_before"] == False) & (eval_df["inside_crown_after"] == True)).sum()
    
    # - 수관 안이었는데, 알고리즘이 엉뚱한 곳으로 보내버린 경우 (이탈)
    lost_outside = ((eval_df["inside_crown_before"] == True) & (eval_df["inside_crown_after"] == False)).sum()

    # 3) 조금이라도 더 정답 쪽으로 이동했는가?
    moved_closer = (eval_df["error_after"] < eval_df["error_before"]).sum()

    # ==========================================
    # 6. Print Report
    # ==========================================
    print("\n" + "="*55)
    print(" 🌳 LeJEPA Tree Crown & Alignment Evaluation")
    print("="*55)
    print(f"Total points evaluated: {total_points}")
    print(f"Assumed Tree Crown Radius: {CROWN_RADIUS} meters")
    print("-" * 55)
    
    print(f"[ 1. Canopy Hit Rate (Points inside Tree Crown) ]")
    print(f" - Before Algorithm : {crown_hit_before} points ({(crown_hit_before/total_points)*100:.1f}%)")
    print(f" - After Algorithm  : {crown_hit_after} points ({(crown_hit_after/total_points)*100:.1f}%)")
    
    print("\n[ 2. Detailed Movement Analysis ]")
    print(f" - Stable (Stayed inside crown)  : {stable_inside} points")
    print(f" - Rescued (Moved into crown)    : {rescued_inside} points")
    print(f" - Lost (Wandered outside crown) : {lost_outside} points")
    
    print("-" * 55)
    print(f"[ 3. General Improvement ]")
    print(f" - Points that moved CLOSER to the ground truth center:")
    print(f"   {moved_closer} points ({(moved_closer/total_points)*100:.1f}%)")
    
    print("-" * 55)
    print(f"[ 4. Absolute Distance Error (For reference) ]")
    print(f" - Mean Error Before : {eval_df['error_before'].mean():.2f} m")
    print(f" - Mean Error After  : {eval_df['error_after'].mean():.2f} m")
    print(f" - Avg. Distance AI moved the point : {eval_df['shift_distance'].mean():.2f} m")
    print("="*55)

    # Save detailed output
    eval_output_csv = r"data/coordinate/sliding_results/crown_evaluation_metrics.csv"
    clean_eval_df = eval_df.drop(columns=["geometry_final", "geometry_gt", "geometry_orig"])
    clean_eval_df.to_csv(eval_output_csv, index=False)
    
    print(f"\nSaved detailed point-by-point evaluation to: {eval_output_csv}")

if __name__ == "__main__":
    main()