"""
    This script evaluates the sliding grid algorithm by evaluating how close the points placed from random points to the valid points.
"""

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import os

def evaluate_with_distance_error(tolerance_distance=15.0):
    # Paths (Update if needed)
    gt_path = "data/tree_label_rdn/valid_points.shp"
    pred_path = "data/tree_label_rdn/random_valid_range_20_35.shp" # Evaluation target

    if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
        print("Error: Shapefiles not found.")
        return

    gt_gdf = gpd.read_file(gt_path)
    pred_gdf = gpd.read_file(pred_path)

    gt_coords = np.array([(geom.x, geom.y) for geom in gt_gdf.geometry])
    pred_coords = np.array([(geom.x, geom.y) for geom in pred_gdf.geometry])

    gt_tree = cKDTree(gt_coords)
    
    # Query within tolerance
    distances, closest_gt_indices = gt_tree.query(pred_coords, distance_upper_bound=tolerance_distance)

    # 1. Classification Metrics (TP, FP, FN)
    matched_mask = distances <= tolerance_distance
    TP_distances = distances[matched_mask] # Distances of correctly matched points
    
    TP = np.sum(matched_mask)
    FP = len(pred_coords) - TP
    
    matched_gt_indices = set(closest_gt_indices[matched_mask])
    FN = len(gt_coords) - len(matched_gt_indices)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # ---------------------------------------------------------
    # 2. Distance Error Metrics (How close to Ground Truth?)
    # ---------------------------------------------------------
    if TP > 0:
        mean_dist_error = np.mean(TP_distances)
        rmse = np.sqrt(np.mean(TP_distances**2))
        max_dist_error = np.max(TP_distances)
        min_dist_error = np.min(TP_distances)
    else:
        mean_dist_error = rmse = max_dist_error = min_dist_error = 0

    print("\n" + "="*60)
    print(f"Spatial Accuracy Evaluation (Tolerance: {tolerance_distance}m)")
    print("="*60)
    print(f"Detection Performance:")
    print(f" - True Positives  : {TP}")
    print(f" - False Positives : {FP}")
    print(f" - False Negatives : {FN}")
    print(f" - Precision       : {precision:.4f}")
    print(f" - Recall          : {recall:.4f}")
    print(f" - F1-Score        : {f1_score:.4f}")
    print("-" * 60)
    print(f"Distance Error (Only for TPs):")
    print(f" - Mean Distance Error : {mean_dist_error:.2f} m")
    print(f" - RMSE                : {rmse:.2f} m")
    print(f" - Min/Max Error       : {min_dist_error:.2f}m / {max_dist_error:.2f}m")
    print("="*60)

if __name__ == "__main__":
    # Set tolerance slightly larger than MAX_SHIFT to capture all shifted points
    evaluate_with_distance_error(tolerance_distance=20.0)