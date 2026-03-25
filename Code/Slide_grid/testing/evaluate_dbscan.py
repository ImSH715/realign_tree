import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import os

# --------------------------
# Config & Paths
# --------------------------
COORD_PATH = "data/embeddings/coords.npy"
LABEL_PATH = "data/label/labels.npy"
NOISY_SHP_PATH = "data/tree_label_rdn/valid_points.shp"
OUTPUT_CSV = "realignment_evaluation_100.csv"

SEARCH_RADIUS = 25.0  # Search radius in meters (max expected error)
EPS = 2.0             # DBSCAN distance threshold (approx. tree crown radius)
MIN_SAMPLES = 5       # Min points to form a valid tree cluster

# --------------------------
# 1. Data Loading & Indexing
# --------------------------
print("Loading pre-trained dense maps...")
dense_coords = np.load(COORD_PATH)     # Shape: [N, 2]
dense_labels = np.load(LABEL_PATH)     # Shape: [N] (Predicted Species)

# Build KD-Tree for efficient spatial queries
spatial_tree = cKDTree(dense_coords)

# Load the noisy shapefile
gdf_noisy = gpd.read_file(NOISY_SHP_PATH)
label_col = 'Tree' if 'Tree' in gdf_noisy.columns else gdf_noisy.columns[0]

# --------------------------
# 2. Realignment Logic (DBSCAN)
# --------------------------
realigned_results = []

print(f"Analyzing {len(gdf_noisy)} points for realignment...")

for idx, row in gdf_noisy.iterrows():
    orig_x, orig_y = row.geometry.x, row.geometry.y
    target_species = str(row[label_col])
    
    # Query all points within the 25m search radius
    indices = spatial_tree.query_ball_point([orig_x, orig_y], SEARCH_RADIUS)
    
    if not indices:
        continue
        
    # Filter: Only keep points predicted as the target species
    candidate_coords = dense_coords[indices]
    candidate_labels = dense_labels[indices]
    species_mask = (candidate_labels == target_species)
    
    target_points = candidate_coords[species_mask]
    
    if len(target_points) < MIN_SAMPLES:
        continue 
        
    # Run DBSCAN to find dense clusters of the target species
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(target_points)
    cluster_labels = db.labels_
    
    # Filter out noise (-1) and find valid clusters
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    
    if len(unique_clusters) == 0:
        continue
        
    # Find the cluster whose centroid is closest to the original (noisy) coordinate
    best_centroid = None
    min_dist = float('inf')
    
    for c_id in unique_clusters:
        cluster_points = target_points[cluster_labels == c_id]
        centroid = cluster_points.mean(axis=0)
        dist_to_orig = np.linalg.norm(centroid - [orig_x, orig_y])
        
        if dist_to_orig < min_dist:
            min_dist = dist_to_orig
            best_centroid = centroid

    if best_centroid is not None:
        realigned_results.append({
            'original_id': idx,
            'species': target_species,
            'orig_x': orig_x,
            'orig_y': orig_y,
            'new_x': best_centroid[0],
            'new_y': best_centroid[1],
            'shift_distance': min_dist
        })

# --------------------------
# 3. Export Sample for Evaluation
# --------------------------
df_final = pd.DataFrame(realigned_results)

if len(df_final) >= 100:
    evaluation_sample = df_final.sample(n=100, random_state=42)
    print(f"Success! Realigned {len(df_final)} points.")
    evaluation_sample.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved 100 test cases to: {OUTPUT_CSV}")
else:
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Only {len(df_final)} points could be realigned. All saved to {OUTPUT_CSV}")

print("\nSample Preview (Top 5):")
print(df_final[['species', 'shift_distance']].head())