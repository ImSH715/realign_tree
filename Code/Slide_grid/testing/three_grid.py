import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree

# --------------------------
# Configuration & Paths
# --------------------------
DISTANCE = 13
GRID_SIZES = [10] # Multi-scale grid search (Fine to Coarse)
MAX_ITERATIONS = 10        # Increased max iterations to allow convergence

INPUT_SHP = f"data/tree_label_rdn/random_trees_32718_d_{DISTANCE}.shp"
EMBEDDING_PATH = r"data/embeddings/embeddings.npy"
LABEL_PATH = r"data/label/labels.npy"
COORD_PATH = r"data/embeddings/coords.npy"
OUTPUT_DIR = r"data/distance"

# --------------------------
# Core Functions
# --------------------------
def load_data():
    """
    Load pre-extracted arrays and build a spatial KDTree.
    """
    if not (os.path.exists(LABEL_PATH) and os.path.exists(COORD_PATH)):
        raise FileNotFoundError("Required npy files not found. Run extraction script first.")

    random_trees = gpd.read_file(INPUT_SHP)
    labels = np.load(LABEL_PATH)
    coords = np.load(COORD_PATH)
    
    spatial_tree = cKDTree(coords)
    
    return random_trees, labels, spatial_tree

def create_3x3_grid(center_point, cell_size):
    """
    Generate a 3x3 grid around the center point based on the dynamic cell_size.
    """
    x, y = center_point.x, center_point.y
    boxes = []
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            minx = x + (j * cell_size) - (cell_size / 2)
            maxx = x + (j * cell_size) + (cell_size / 2)
            miny = y - (i * cell_size) - (cell_size / 2)
            maxy = y - (i * cell_size) + (cell_size / 2)
            
            box_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            boxes.append(box_poly)
            
    return boxes

def calculate_likelihood(box, target_label, labels, spatial_tree, current_radius):
    """
    Check if the nearest coordinate matches the target label.
    Only considers neighbors within the current_radius (cell_size).
    """
    box_center = [box.centroid.x, box.centroid.y]
    min_dist, closest_idx = spatial_tree.query(box_center)

    if min_dist > current_radius:
        return 0.0

    predicted_label = str(labels[closest_idx])
    
    return 1.0 if predicted_label == str(target_label) else 0.0

def process_multi_scale_slide(current_points, labels, spatial_tree):
    """
    Execute one sliding step using a cascade of grid sizes [10, 20, 30].
    Points finalize as centers if they meet high confidence at size 10,
    if they converge (stop moving), or if they are completely lost at size 30.
    """
    new_centers = []
    new_slides = []
    
    label_col = 'Tree' if 'Tree' in current_points.columns else current_points.columns[0]
    
    for _, row in current_points.iterrows():
        point = row.geometry
        target_label = row[label_col]
        
        moved = False
        
        # Cascade search: start fine (10), fallback to medium (20), then coarse (30)
        for cell_size in GRID_SIZES:
            boxes = create_3x3_grid(point, cell_size)
            
            checked_boxes = [
                box for box in boxes 
                if calculate_likelihood(box, target_label, labels, spatial_tree, cell_size) >= 0.5
            ]
            
            num_checked = len(checked_boxes)
            
            if num_checked > 0:
                avg_x = sum(b.centroid.x for b in checked_boxes) / num_checked
                avg_y = sum(b.centroid.y for b in checked_boxes) / num_checked
                new_point = Point(avg_x, avg_y)
                
                # Condition 1: High confidence at the finest resolution -> Center
                if cell_size == 10 and num_checked >= 3:
                    new_centers.append({'geometry': new_point, label_col: target_label})
                    moved = True
                    break
                    
                # Condition 2: Point converged (barely moved) -> Center
                if point.distance(new_point) < 0.01:
                    new_centers.append({'geometry': new_point, label_col: target_label})
                    moved = True
                    break
                    
                # Condition 3: Found signal but needs more refinement -> Slide
                new_slides.append({'geometry': new_point, label_col: target_label})
                moved = True
                break
                
        # Condition 4: No signal even at the largest grid (30) -> Center (Lost)
        if not moved:
            new_centers.append({'geometry': point, label_col: target_label})
            
    crs = current_points.crs
    
    gdf_centers = gpd.GeoDataFrame(new_centers, crs=crs) if new_centers else gpd.GeoDataFrame(columns=[label_col, 'geometry'], crs=crs)
    gdf_slides = gpd.GeoDataFrame(new_slides, crs=crs) if new_slides else gpd.GeoDataFrame(columns=[label_col, 'geometry'], crs=crs)
    
    return gdf_centers, gdf_slides

# --------------------------
# Main Execution
# --------------------------
def main():
    print("Loading data and initializing spatial tree...")
    random_trees, labels, spatial_tree = load_data()
    
    all_final_points = []
    current_slides = random_trees
    
    print("Starting multi-scale sliding grid algorithm.")
    print(f"Grid cascade: {GRID_SIZES}, Max iterations: {MAX_ITERATIONS}")
    
    for step in range(MAX_ITERATIONS):
        print(f"Iteration {step + 1}")
        
        if current_slides.empty:
            print("All points have converged to a center. Exiting loop.")
            break
            
        centers, current_slides = process_multi_scale_slide(
            current_slides, 
            labels, 
            spatial_tree
        )
        
        if not centers.empty:
            all_final_points.append(centers)
            
        print(f"  -> {len(centers)} finalized (Centers), {len(current_slides)} continuing (Slides)")
    
    if not current_slides.empty:
        print(f"Warning: Reached max iterations. Force finalizing remaining {len(current_slides)} points.")
        all_final_points.append(current_slides)
    
    if all_final_points:
        final_points_gdf = pd.concat(all_final_points, ignore_index=True)
        final_points_gdf = gpd.GeoDataFrame(final_points_gdf, crs=random_trees.crs)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        #output
        save_path = os.path.join(OUTPUT_DIR, f"slide_grid_multi_scale_d{DISTANCE}.shp")
        
        final_points_gdf.to_file(save_path)
        print(f"Process complete. Saved {len(final_points_gdf)} points to: {save_path}")
    else:
        print("Process complete. No points were generated.")

if __name__ == "__main__":
    main()