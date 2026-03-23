import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os

# --------------------------
# Paths
# --------------------------
GT_TREES_SHP = "trees_32718.shp"              
RANDOM_TREES_SHP = "random_trees_32718.shp"    

EMBEDDING_PATH = "data/embeddings/embeddings.npy"
COORD_PATH = "data/embeddings/coords.npy"
LABEL_PATH = "data/label/labels.npy"

OUTPUT_SHP = "slide_grid_results.shp"

# --------------------------
# Load Data
# --------------------------
def load_data():
    gt_trees = gpd.read_file(GT_TREES_SHP)
    random_trees = gpd.read_file(RANDOM_TREES_SHP)

    # Ensure coordinates match
    if gt_trees.crs != random_trees.crs:
        random_trees = random_trees.to_crs(gt_trees.crs)

    embeddings = np.load(EMBEDDING_PATH)
    coords = np.load(COORD_PATH)
    labels = np.load(LABEL_PATH)

    return gt_trees, random_trees, embeddings, coords, labels

# --------------------------
# Create 3x3 Grid
# --------------------------
def create_3x3_grid(center_point, cell_size=5.0):
    x, y = center_point.x, center_point.y

    boxes = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            minx = x + (j * cell_size) - (cell_size / 2)
            maxx = x + (j * cell_size) + (cell_size / 2)
            miny = y - (i * cell_size) - (cell_size / 2)
            maxy = y - (i * cell_size) + (cell_size / 2)

            boxes.append(Polygon([
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy)
            ]))

    return boxes

# --------------------------
# Find points inside box
# --------------------------
def get_points_in_box(box, coords):
    minx, miny, maxx, maxy = box.bounds

    mask = (
        (coords[:, 0] >= minx) &
        (coords[:, 0] <= maxx) &
        (coords[:, 1] >= miny) &
        (coords[:, 1] <= maxy)
    )

    return mask

# --------------------------
# Cosine similarity
# --------------------------
def cosine_similarity(a, b):
    # Add epsilon to prevent division by zero
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# --------------------------
# Likelihood calculation
# --------------------------
def calculate_likelihood(box, tree_label, query_embedding, embeddings, coords, labels):
    mask = get_points_in_box(box, coords)

    if np.sum(mask) == 0:
        return 0.0

    box_embeddings = embeddings[mask]
    box_labels = labels[mask]

    # Only same species
    same_species_mask = (box_labels == tree_label)

    if np.sum(same_species_mask) == 0:
        return 0.0

    candidate_embeddings = box_embeddings[same_species_mask]

    # Compute similarity
    sims = [
        cosine_similarity(query_embedding, emb)
        for emb in candidate_embeddings
    ]

    return np.max(sims)

# --------------------------
# Nearest embedding finder
# --------------------------
def get_query_embedding(point, coords, embeddings):
    dists = cdist([[point.x, point.y]], coords)
    idx = np.argmin(dists)
    return embeddings[idx]

# --------------------------
# Sliding Grid Step
# --------------------------
def process_slide_grid(current_points, embeddings, coords, labels, cell_size=5.0):
    new_points = []
    
    # Identify label column (assuming first column as per your code)
    label_col = current_points.columns[0]

    for _, row in current_points.iterrows():
        point = row.geometry
        tree_label = row[label_col]

        query_embedding = get_query_embedding(point, coords, embeddings)
        boxes = create_3x3_grid(point, cell_size)

        best_score = -1
        best_center = point

        for box in boxes:
            score = calculate_likelihood(
                box,
                tree_label,
                query_embedding,
                embeddings,
                coords,
                labels
            )

            if score > best_score:
                best_score = score
                best_center = box.centroid

        new_points.append({
            'geometry': best_center,
            label_col: tree_label
        })

    return gpd.GeoDataFrame(new_points, crs=current_points.crs)

# --------------------------
# Evaluation Metrics Calculation
# --------------------------
def evaluate_performance(initial_gdf, final_gdf, gt_gdf):
    """
    Calculate spatial distances between predicted points and ground truth points.
    Assumes all GeoDataFrames have the same length and matching indices (row 1 = row 1).
    """
    print("\n--- Evaluation Results ---")
    
    # Calculate distances (point-to-point)
    initial_distances = initial_gdf.geometry.distance(gt_gdf.geometry)
    final_distances = final_gdf.geometry.distance(gt_gdf.geometry)
    
    # 1. Mean Absolute Error (MAE)
    initial_mae = initial_distances.mean()
    final_mae = final_distances.mean()
    
    # 2. Root Mean Square Error (RMSE)
    initial_rmse = np.sqrt((initial_distances**2).mean())
    final_rmse = np.sqrt((final_distances**2).mean())
    
    # 3. Improvement Analysis
    # How many trees moved closer to the ground truth?
    improved_mask = final_distances < initial_distances
    num_improved = improved_mask.sum()
    total_trees = len(gt_gdf)
    improvement_ratio = (num_improved / total_trees) * 100
    
    print(f"Total Trees Evaluated: {total_trees}")
    print(f"Initial Mean Distance Error: {initial_mae:.2f} meters")
    print(f"Final Mean Distance Error:   {final_mae:.2f} meters")
    print("-" * 25)
    print(f"Initial RMSE: {initial_rmse:.2f} meters")
    print(f"Final RMSE:   {final_rmse:.2f} meters")
    print("-" * 25)
    print(f"Trees moved closer to GT: {num_improved} / {total_trees} ({improvement_ratio:.2f}%)")
    
    return initial_distances, final_distances

# --------------------------
# Main
# --------------------------
def main():
    print("Loading datasets and model embeddings...")
    gt_trees, random_trees, embeddings, coords, labels = load_data()

    current = random_trees.copy()

    NUM_STEPS = 3
    CELL_SIZE = 6.0

    print("Starting Sliding Grid Algorithm...")
    for step in range(NUM_STEPS):
        print(f"Step {step+1}/{NUM_STEPS} running...")
        current = process_slide_grid(
            current,
            embeddings,
            coords,
            labels,
            CELL_SIZE
        )

    # Save the output
    current.to_file(OUTPUT_SHP)
    print(f"Saved results to: {OUTPUT_SHP}")
    
    # --------------------------
    # Run Evaluation
    # --------------------------
    evaluate_performance(
        initial_gdf=random_trees, 
        final_gdf=current, 
        gt_gdf=gt_trees
    )

if __name__ == "__main__":
    main()