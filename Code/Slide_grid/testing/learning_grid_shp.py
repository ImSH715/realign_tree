import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from numpy.linalg import norm

# --------------------------
# Configuration & Paths
# --------------------------
DISTANCE = 13
GRID_SIZES = [10, 20, 30]  # Multi-scale grid search (Fine to Coarse)
MAX_ITERATIONS = 10        # Maximum iterations to allow convergence

# Cosine Similarity Threshold (1.0 = identical, 0.0 = orthogonal)
# Usually, >= 0.85 means the features are visually very similar in the embedding space.
SIM_THRESHOLD = 0.85 

INPUT_SHP = f"data/tree_label_rdn/random_trees_32718_d_{DISTANCE}.shp"
EMBEDDING_PATH = r"data/embeddings/embeddings.npy"
LABEL_PATH = r"data/label/labels.npy"
COORD_PATH = r"data/embeddings/coords.npy"
OUTPUT_DIR = r"data/distance"
ITER_OUT_DIR = os.path.join(OUTPUT_DIR, "iterations_demo") # Sub-directory for demo outputs

# --------------------------
# Core Functions
# --------------------------
def load_data():
    """
    Load pre-extracted embeddings, labels, and original shapefile points.
    Builds a spatial KDTree for fast nearest-neighbor spatial queries.
    """
    if not (os.path.exists(LABEL_PATH) and os.path.exists(COORD_PATH) and os.path.exists(EMBEDDING_PATH)):
        raise FileNotFoundError("Required npy files not found. Run extraction script first.")

    print(f"Loading data from SHP: {INPUT_SHP}")
    initial_trees = gpd.read_file(INPUT_SHP)

    labels = np.load(LABEL_PATH)
    coords = np.load(COORD_PATH)
    embeddings = np.load(EMBEDDING_PATH) # Shape: (N, 128)
    
    spatial_tree = cKDTree(coords)
    
    return initial_trees, labels, coords, embeddings, spatial_tree

def assign_initial_embeddings(current_points, embeddings, spatial_tree):
    """
    Find the closest embedding for each point in the initial SHP and assign it as the 'target_embedding'.
    This gives each moving point its own unique visual identity.
    """
    target_embs = []
    for _, row in current_points.iterrows():
        pt = [row.geometry.x, row.geometry.y]
        _, closest_idx = spatial_tree.query(pt)
        target_embs.append(embeddings[closest_idx])
        
    current_points['target_embedding'] = target_embs
    return current_points

def create_3x3_grid(center_point, cell_size):
    """
    Generate 9 bounding boxes forming a 3x3 grid around the center point based on cell_size.
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

def calculate_similarity(box, target_embedding, embeddings, spatial_tree, current_radius):
    """
    Calculate Cosine Similarity between the target's embedding and the nearest neighbor's embedding within the box.
    Returns 0.0 if no neighbor is found within the current_radius.
    """
    box_center = [box.centroid.x, box.centroid.y]
    min_dist, closest_idx = spatial_tree.query(box_center)

    if min_dist > current_radius:
        return 0.0

    neighbor_embedding = embeddings[closest_idx]
    
    # Cosine Similarity Calculation
    norm_target = norm(target_embedding)
    norm_neighbor = norm(neighbor_embedding)
    
    if norm_target == 0 or norm_neighbor == 0:
        return 0.0
        
    sim = np.dot(target_embedding, neighbor_embedding) / (norm_target * norm_neighbor)
    return sim

def process_multi_scale_slide(current_points, embeddings, spatial_tree):
    """
    Execute one sliding step using a cascade of grid sizes [10, 20, 30].
    Moves points towards grids that have high embedding similarity (>= SIM_THRESHOLD).
    """
    new_centers = []
    new_slides = []
    
    grids_collected = {size: [] for size in GRID_SIZES}
    
    # Dynamically find the label column for SHP files
    label_col = 'Tree' if 'Tree' in current_points.columns else current_points.columns[0]
    
    for _, row in current_points.iterrows():
        point = row.geometry
        target_label = row[label_col]
        target_emb = row['target_embedding'] 
        
        moved = False
        
        # Cascade search: start fine (10), fallback to medium (20), then coarse (30)
        for cell_size in GRID_SIZES:
            boxes = create_3x3_grid(point, cell_size)
            
            # Store grids for visualization
            for box_poly in boxes:
                grids_collected[cell_size].append({
                    'geometry': box_poly, 
                    label_col: target_label,
                    'grid_size': cell_size
                })
            
            # Filter boxes that exceed the similarity threshold
            checked_boxes = [
                box for box in boxes 
                if calculate_similarity(box, target_emb, embeddings, spatial_tree, cell_size) >= SIM_THRESHOLD
            ]
            
            num_checked = len(checked_boxes)
            
            if num_checked > 0:
                avg_x = sum(b.centroid.x for b in checked_boxes) / num_checked
                avg_y = sum(b.centroid.y for b in checked_boxes) / num_checked
                new_point = Point(avg_x, avg_y)
                
                # Condition 1: High confidence at the finest resolution -> Center
                if cell_size == 10 and num_checked >= 3:
                    new_centers.append({'geometry': new_point, label_col: target_label, 'coord_x': avg_x, 'coord_y': avg_y, 'target_embedding': target_emb})
                    moved = True
                    break
                    
                # Condition 2: Point converged (barely moved) -> Center
                if point.distance(new_point) < 0.01:
                    new_centers.append({'geometry': new_point, label_col: target_label, 'coord_x': avg_x, 'coord_y': avg_y, 'target_embedding': target_emb})
                    moved = True
                    break
                    
                # Condition 3: Found similar cluster but needs more refinement -> Slide
                new_slides.append({'geometry': new_point, label_col: target_label, 'coord_x': avg_x, 'coord_y': avg_y, 'target_embedding': target_emb})
                moved = True
                break
                
        # Condition 4: No similar neighbors even at the largest grid (30) -> Center (Lost)
        if not moved:
            new_centers.append({'geometry': point, label_col: target_label, 'coord_x': point.x, 'coord_y': point.y, 'target_embedding': target_emb})
            
    crs = current_points.crs
    
    gdf_centers = gpd.GeoDataFrame(new_centers, crs=crs) if new_centers else gpd.GeoDataFrame(columns=[label_col, 'coord_x', 'coord_y', 'target_embedding', 'geometry'], crs=crs)
    gdf_slides = gpd.GeoDataFrame(new_slides, crs=crs) if new_slides else gpd.GeoDataFrame(columns=[label_col, 'coord_x', 'coord_y', 'target_embedding', 'geometry'], crs=crs)
    
    gdf_grids_dict = {}
    for size in GRID_SIZES:
        gdf_grids_dict[size] = gpd.GeoDataFrame(grids_collected[size], crs=crs) if grids_collected[size] else gpd.GeoDataFrame(columns=[label_col, 'grid_size', 'geometry'], crs=crs)
        
    return gdf_centers, gdf_slides, gdf_grids_dict

# --------------------------
# Main Execution
# --------------------------
def save_clean_shp(gdf, save_path):
    """
    Helper function to save GeoDataFrames to SHP files.
    Drops the 'target_embedding' column (numpy array) because SHP does not support array data types.
    """
    if not gdf.empty:
        clean_gdf = gdf.drop(columns=['target_embedding'], errors='ignore')
        clean_gdf.to_file(save_path)

def main():
    print("Loading data and initializing spatial tree...")
    initial_trees, labels, coords, embeddings, spatial_tree = load_data()
    
    print("Assigning initial embeddings to SHP points...")
    initial_trees = assign_initial_embeddings(initial_trees, embeddings, spatial_tree)
    
    # Ensure explicit coordinates exist
    if 'coord_x' not in initial_trees.columns:
        initial_trees['coord_x'] = initial_trees.geometry.x
        initial_trees['coord_y'] = initial_trees.geometry.y

    all_final_points = []
    current_slides = initial_trees
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ITER_OUT_DIR, exist_ok=True)
    
    print("Starting Multi-scale Sliding Grid (Embedding Similarity Based - DEMO)")
    print(f"Grid cascade: {GRID_SIZES}, Max iterations: {MAX_ITERATIONS}, Similarity Threshold: {SIM_THRESHOLD}")
    
    for step in range(MAX_ITERATIONS):
        iteration_num = step + 1
        print(f"Iteration {iteration_num}")
        
        if current_slides.empty:
            print("All points have converged to a center. Exiting loop.")
            break
            
        centers, current_slides, grids_dict = process_multi_scale_slide(
            current_slides, 
            embeddings, 
            spatial_tree
        )
        
        # 1. Save Evaluation Grids (10, 20, 30)
        for size in GRID_SIZES:
            if not grids_dict[size].empty:
                save_clean_shp(grids_dict[size], os.path.join(ITER_OUT_DIR, f"iter_{iteration_num}_grid_{size}.shp"))
        
        # 2. Save Finalized Centers
        if not centers.empty:
            all_final_points.append(centers)
            save_clean_shp(centers, os.path.join(ITER_OUT_DIR, f"iter_{iteration_num}_centers.shp"))
            
        # 3. Save Continuing Slides
        if not current_slides.empty:
            save_clean_shp(current_slides, os.path.join(ITER_OUT_DIR, f"iter_{iteration_num}_slides.shp"))
            
        print(f"  -> {len(centers)} finalized (Centers), {len(current_slides)} continuing (Slides)")
    
    if not current_slides.empty:
        print(f"Warning: Reached max iterations. Force finalizing remaining {len(current_slides)} points.")
        all_final_points.append(current_slides)
    
    # Combine all finalized points and save
    if all_final_points:
        final_points_gdf = pd.concat(all_final_points, ignore_index=True)
        final_points_gdf = gpd.GeoDataFrame(final_points_gdf, crs=initial_trees.crs)
        
        save_path = os.path.join(OUTPUT_DIR, f"slide_grid_embedding_sim_demo_d{DISTANCE}.shp")
        save_clean_shp(final_points_gdf, save_path)
        print(f"\nProcess complete. Saved final {len(final_points_gdf)} points to: {save_path}")
        print(f"Intermediate SHP files (Grids, Centers, Slides) saved in: {ITER_OUT_DIR}")
    else:
        print("Process complete. No points were generated.")

if __name__ == "__main__":
    main()