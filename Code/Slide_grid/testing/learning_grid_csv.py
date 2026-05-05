"""
The sliding grid that uses learned feauters and find features in the grid and check whether they are same or not.
"""

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
GRID_SIZES = [10, 20, 30]  
MAX_ITERATIONS = 10        
TARGET_CRS = "EPSG:32718"  

# Cosine Similarity Threshold (1.0 = identical, 0.0 = orthogonal)
# 임베딩 공간에서는 보통 0.8 이상일 때 '비슷하다'고 판단합니다.
SIM_THRESHOLD = 0.85 

INPUT_CSV = r"data/tree_label_rdn/your_new_file.csv"  # <-- CSV 경로 확인
EMBEDDING_PATH = r"data/embeddings/embeddings.npy"
LABEL_PATH = r"data/label/labels.npy"
COORD_PATH = r"data/embeddings/coords.npy"
OUTPUT_DIR = r"data/distance"
ITER_OUT_DIR = os.path.join(OUTPUT_DIR, "iterations")

# --------------------------
# Core Functions
# --------------------------
def load_data():
    if not (os.path.exists(LABEL_PATH) and os.path.exists(COORD_PATH)):
        raise FileNotFoundError("Required npy files not found.")

    print(f"Loading data from CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Create GeoDataFrame
    geometry = gpd.points_from_xy(df['COORDENADA_ESTE'], df['COORDENADA_NORTE'])
    initial_trees = gpd.GeoDataFrame(df, geometry=geometry, crs=TARGET_CRS)

    labels = np.load(LABEL_PATH)
    coords = np.load(COORD_PATH)
    embeddings = np.load(EMBEDDING_PATH) # (N, 128)
    
    spatial_tree = cKDTree(coords)
    
    return initial_trees, labels, coords, embeddings, spatial_tree

def assign_initial_embeddings(current_points, embeddings, spatial_tree):
    """
    CSV 포인트의 초기 위치를 기반으로, 가장 가까운 특징(Embedding)을 찾아 타겟으로 할당합니다.
    """
    target_embs = []
    for _, row in current_points.iterrows():
        pt = [row.geometry.x, row.geometry.y]
        _, closest_idx = spatial_tree.query(pt)
        target_embs.append(embeddings[closest_idx])
        
    current_points['target_embedding'] = target_embs
    return current_points

def create_3x3_grid(center_point, cell_size):
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
    라벨 비교 대신, 타겟 임베딩과 이웃 임베딩 간의 코사인 유사도(Cosine Similarity)를 계산합니다.
    """
    box_center = [box.centroid.x, box.centroid.y]
    min_dist, closest_idx = spatial_tree.query(box_center)

    if min_dist > current_radius:
        return 0.0

    neighbor_embedding = embeddings[closest_idx]
    
    # Calculate Cosine Similarity
    norm_target = norm(target_embedding)
    norm_neighbor = norm(neighbor_embedding)
    
    if norm_target == 0 or norm_neighbor == 0:
        return 0.0
        
    sim = np.dot(target_embedding, neighbor_embedding) / (norm_target * norm_neighbor)
    return sim

def process_multi_scale_slide(current_points, embeddings, spatial_tree):
    new_centers = []
    new_slides = []
    
    grids_collected = {size: [] for size in GRID_SIZES}
    label_col = 'NOMBRE_COMUN'
    
    for _, row in current_points.iterrows():
        point = row.geometry
        target_label = row[label_col]
        target_emb = row['target_embedding']  # 사용할 타겟의 특징 벡터
        
        moved = False
        
        for cell_size in GRID_SIZES:
            boxes = create_3x3_grid(point, cell_size)
            
            for box_poly in boxes:
                grids_collected[cell_size].append({
                    'geometry': box_poly, 
                    label_col: target_label,
                    'grid_size': cell_size
                })
            
            # 유사도가 임계값(SIM_THRESHOLD) 이상인 박스만 필터링
            checked_boxes = [
                box for box in boxes 
                if calculate_similarity(box, target_emb, embeddings, spatial_tree, cell_size) >= SIM_THRESHOLD
            ]
            
            num_checked = len(checked_boxes)
            
            if num_checked > 0:
                avg_x = sum(b.centroid.x for b in checked_boxes) / num_checked
                avg_y = sum(b.centroid.y for b in checked_boxes) / num_checked
                new_point = Point(avg_x, avg_y)
                
                # Condition 1
                if cell_size == 10 and num_checked >= 3:
                    new_centers.append({'geometry': new_point, label_col: target_label, 'coord_x': avg_x, 'coord_y': avg_y, 'target_embedding': target_emb})
                    moved = True
                    break
                    
                # Condition 2
                if point.distance(new_point) < 0.01:
                    new_centers.append({'geometry': new_point, label_col: target_label, 'coord_x': avg_x, 'coord_y': avg_y, 'target_embedding': target_emb})
                    moved = True
                    break
                    
                # Condition 3
                new_slides.append({'geometry': new_point, label_col: target_label, 'coord_x': avg_x, 'coord_y': avg_y, 'target_embedding': target_emb})
                moved = True
                break
                
        # Condition 4 (Lost)
        if not moved:
            new_centers.append({'geometry': point, label_col: target_label, 'coord_x': point.x, 'coord_y': point.y, 'target_embedding': target_emb})
            
    crs = current_points.crs
    
    # DataFrame 생성
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
    """SHP 파일 저장을 위해 array 타입인 target_embedding 컬럼을 제거하고 저장하는 헬퍼 함수"""
    if not gdf.empty:
        clean_gdf = gdf.drop(columns=['target_embedding'], errors='ignore')
        clean_gdf.to_file(save_path)

def main():
    print("Loading data and initializing spatial tree...")
    initial_trees, labels, coords, embeddings, spatial_tree = load_data()
    
    # 1. 초기 임베딩 할당 (가장 중요한 부분!)
    print("Assigning initial embeddings to CSV points...")
    initial_trees = assign_initial_embeddings(initial_trees, embeddings, spatial_tree)
    
    if 'coord_x' not in initial_trees.columns:
        initial_trees['coord_x'] = initial_trees.geometry.x
        initial_trees['coord_y'] = initial_trees.geometry.y

    all_final_points = []
    current_slides = initial_trees
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ITER_OUT_DIR, exist_ok=True)
    
    print("Starting Multi-scale Sliding Grid (Embedding Similarity Based)")
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
        
        # Save Iteration Grids
        for size in GRID_SIZES:
            if not grids_dict[size].empty:
                save_clean_shp(grids_dict[size], os.path.join(ITER_OUT_DIR, f"iter_{iteration_num}_grid_{size}.shp"))
        
        # Save Iteration Centers & Slides
        if not centers.empty:
            all_final_points.append(centers)
            save_clean_shp(centers, os.path.join(ITER_OUT_DIR, f"iter_{iteration_num}_centers.shp"))
            
        if not current_slides.empty:
            save_clean_shp(current_slides, os.path.join(ITER_OUT_DIR, f"iter_{iteration_num}_slides.shp"))
            
        print(f"  -> {len(centers)} finalized (Centers), {len(current_slides)} continuing (Slides)")
    
    if not current_slides.empty:
        print(f"Warning: Reached max iterations. Force finalizing remaining {len(current_slides)} points.")
        all_final_points.append(current_slides)
    
    # Final Combine and Save
    if all_final_points:
        final_points_gdf = pd.concat(all_final_points, ignore_index=True)
        final_points_gdf = gpd.GeoDataFrame(final_points_gdf, crs=initial_trees.crs)
        
        save_path = os.path.join(OUTPUT_DIR, f"slide_grid_embedding_sim_d{DISTANCE}.shp")
        save_clean_shp(final_points_gdf, save_path)
        print(f"\nProcess complete. Saved final {len(final_points_gdf)} points to: {save_path}")
    else:
        print("Process complete. No points were generated.")

if __name__ == "__main__":
    main()