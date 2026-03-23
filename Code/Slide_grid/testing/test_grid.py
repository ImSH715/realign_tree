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
CELL_SIZE = 12
NUM_ITERATIONS = 3

INPUT_SHP = f"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718_{DISTANCE}.shp"
EMBEDDING_PATH = r"data/embeddings/embeddings.npy"
LABEL_PATH = r"data/label/labels.npy"
COORD_PATH = r"data/embeddings/coords.npy"
OUTPUT_DIR = r"data/distance"

# --------------------------
# Core Functions
# --------------------------
def load_data():
    """
    Load pre-extracted embeddings, labels, and coordinates.
    Builds a KDTree for fast nearest-neighbor spatial queries.
    """
    if not (os.path.exists(LABEL_PATH) and os.path.exists(COORD_PATH)):
        raise FileNotFoundError("Required npy files not found. Run the extraction script first.")

    random_trees = gpd.read_file(INPUT_SHP)
    
    # Load arrays
    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABEL_PATH)
    coords = np.load(COORD_PATH)
    
    # Build spatial index
    spatial_tree = cKDTree(coords)
    
    return random_trees, embeddings, labels, spatial_tree

def create_3x3_grid(center_point, cell_size=5.0):
    """
    Generate 9 bounding boxes forming a 3x3 grid around the center point.
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

def calculate_likelihood(box, target_label, labels, spatial_tree):
    """
    Check if the majority label in the nearest coordinate matches the target.
    Utilizes KDTree for O(log N) lookup instead of calculating all distances.
    """
    box_center = [box.centroid.x, box.centroid.y]
    min_dist, closest_idx = spatial_tree.query(box_center)

    # Return 0 probability if the nearest point is too far
    if min_dist > CELL_SIZE:
        return 0.0

    predicted_label = str(labels[closest_idx])
    
    return 1.0 if predicted_label == str(target_label) else 0.0

def process_slide_grid(current_points, labels, spatial_tree, cell_size=5.0):
    """
    Execute one step of the sliding grid logic.
    Splits points into new centers and new slide points based on likelihood.
    """
    new_centers = []
    new_slides = []
    
    label_col = 'Tree' if 'Tree' in current_points.columns else current_points.columns[0]
    
    for _, row in current_points.iterrows():
        point = row.geometry
        target_label = row[label_col]
        
        boxes = create_3x3_grid(point, cell_size)
        
        # Filter boxes that match the target label
        checked_boxes = [
            box for box in boxes 
            if calculate_likelihood(box, target_label, labels, spatial_tree) >= 0.5
        ]
        
        num_checked = len(checked_boxes)
        
        if num_checked >= 3:
            avg_x = sum(b.centroid.x for b in checked_boxes) / num_checked
            avg_y = sum(b.centroid.y for b in checked_boxes) / num_checked
            new_centers.append({'geometry': Point(avg_x, avg_y), label_col: target_label})
            
        elif 1 <= num_checked <= 2:
            avg_x = sum(b.centroid.x for b in checked_boxes) / num_checked
            avg_y = sum(b.centroid.y for b in checked_boxes) / num_checked
            new_slides.append({'geometry': Point(avg_x, avg_y), label_col: target_label})
            
        else:
            new_slides.append({'geometry': point, label_col: target_label})
            
    crs = current_points.crs
    
    gdf_centers = gpd.GeoDataFrame(new_centers, crs=crs) if new_centers else gpd.GeoDataFrame(columns=[label_col, 'geometry'], crs=crs)
    gdf_slides = gpd.GeoDataFrame(new_slides, crs=crs) if new_slides else gpd.GeoDataFrame(columns=[label_col, 'geometry'], crs=crs)
    
    return gdf_centers, gdf_slides

# --------------------------
# Main Execution
# --------------------------
def main():
    print("Loading data and initializing spatial tree...")
    random_trees, _, labels, spatial_tree = load_data()
    
    all_final_points = []
    current_slides = random_trees
    
    print(f"Starting sliding grid algorithm for {NUM_ITERATIONS} iterations.")
    
    for step in range(NUM_ITERATIONS):
        print(f"Iteration {step + 1}/{NUM_ITERATIONS}")
        
        if current_slides.empty:
            print("No more points to slide. Exiting loop.")
            break
            
        centers, current_slides = process_slide_grid(
            current_slides, 
            labels, 
            spatial_tree,
            cell_size=CELL_SIZE
        )
        
        if not centers.empty:
            all_final_points.append(centers)
            
        print(f"  -> Generated {len(centers)} center points, {len(current_slides)} slide points.")
    
    # Include remaining slide points
    if not current_slides.empty:
        all_final_points.append(current_slides)
    
    # Save results
    if all_final_points:
        final_points_gdf = pd.concat(all_final_points, ignore_index=True)
        final_points_gdf = gpd.GeoDataFrame(final_points_gdf, crs=random_trees.crs)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"slide_grid_results_d{DISTANCE}_cz{CELL_SIZE}.shp")
        
        final_points_gdf.to_file(save_path)
        print(f"Process complete. Saved {len(final_points_gdf)} points to: {save_path}")
    else:
        print("Process complete. No points were generated.")

if __name__ == "__main__":
    main()