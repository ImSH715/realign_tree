import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

SHP_PATH = "valid_points.shp"
CSV_PATH = "outputs/phase3/refined_points_world.csv"
OUTPUT_CSV = "outputs/phase3/refined_points_vs_shp_evaluated.csv"

# 1. Load ground-truth shapefile
gdf = gpd.read_file(SHP_PATH).to_crs("EPSG:32718")
gdf["gt_east"] = gdf.geometry.x
gdf["gt_north"] = gdf.geometry.y

# 2. Load refined CSV
df = pd.read_csv(CSV_PATH)

required = ["refined_east", "refined_north"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing required column '{c}' in refined CSV.")

# 3. Build nearest-neighbor matcher from SHP GT points
gt_xy = np.column_stack([gdf["gt_east"].values, gdf["gt_north"].values])
tree = cKDTree(gt_xy)

pred_xy = np.column_stack([df["refined_east"].values, df["refined_north"].values])
dist, idx = tree.query(pred_xy, k=1)

# 4. Attach nearest GT info
df["gt_index"] = idx
df["gt_east"] = gdf.iloc[idx]["gt_east"].values
df["gt_north"] = gdf.iloc[idx]["gt_north"].values
df["distance_to_gt_m"] = dist

# 5. Summary
print("=" * 80)
print("Evaluation against valid_points.shp")
print(f"N predictions: {len(df)}")
print(f"Mean distance to nearest GT: {df['distance_to_gt_m'].mean():.4f} m")
print(f"Median distance to nearest GT: {df['distance_to_gt_m'].median():.4f} m")
print("=" * 80)

for th in [0.5, 1, 2, 5, 10]:
    acc = (df["distance_to_gt_m"] <= th).mean()
    print(f"Accuracy within {th} m: {acc:.3f}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved evaluated CSV to: {OUTPUT_CSV}")