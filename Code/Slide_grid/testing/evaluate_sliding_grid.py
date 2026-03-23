import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import os

def evaluate_sliding_grid(tolerance_distance=5.0):
    """
    Evaluates the spatial accuracy of predicted points against ground truth.
    
    Args:
        tolerance_distance (float): Maximum distance to be considered a correct match 
                                    (e.g., within 5 meters/units of the actual tree).
    """
    # Paths for Ground Truth (GT) and Predicted results
    gt_path = "data/tree_label_rdn/trees_32718.shp"
    pred_path = "data/distance/slide_grid_multi_scale_d13.shp"

    # Check if the shapefiles exist
    if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
        print("Error: Ground Truth or Prediction shapefiles not found.")
        return

    print("Loading shapefiles...")
    gt_gdf = gpd.read_file(gt_path)
    pred_gdf = gpd.read_file(pred_path)

    # Extract coordinates as (x, y) arrays
    gt_coords = np.array([(geom.x, geom.y) for geom in gt_gdf.geometry])
    pred_coords = np.array([(geom.x, geom.y) for geom in pred_gdf.geometry])

    print(f"Ground Truth points: {len(gt_coords)}")
    print(f"Predicted points:    {len(pred_coords)}")

    # Create KDTree for efficient spatial searching
    gt_tree = cKDTree(gt_coords)
    
    # Find the closest ground truth point for each predicted point
    # distance_upper_bound ensures we only look within the tolerance radius
    distances, closest_gt_indices = gt_tree.query(pred_coords, distance_upper_bound=tolerance_distance)

    # Match is successful (True Positive) if the distance is within tolerance_distance.
    # SciPy KDTree returns np.inf if no neighbor is found within the upper bound (False Positive).
    matched_preds = distances <= tolerance_distance
    
    TP = np.sum(matched_preds)  # True Positives: Correct predictions
    FP = len(pred_coords) - TP  # False Positives: Predictions that missed the target
    
    # Since multiple predicted points could theoretically match the same GT point,
    # we count the unique number of successfully matched GT points.
    matched_gt_indices = set(closest_gt_indices[matched_preds])
    FN = len(gt_coords) - len(matched_gt_indices) # False Negatives: Actual trees we missed

    # Calculate Evaluation Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print(f"Sliding Grid Performance (Tolerance: {tolerance_distance})")
    print("="*50)
    print(f"True Positives (Correct hits) : {TP}")
    print(f"False Positives (False alarms): {FP}")
    print(f"False Negatives (Missed trees): {FN}")
    print("-" * 50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    # Assuming the CRS unit is meters (e.g., EPSG:32718).
    # Adjust tolerance_distance based on your smallest grid size (10) and acceptable error margin.
    evaluate_sliding_grid(tolerance_distance=5.0)