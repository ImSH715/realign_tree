import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist

DISTANCE = 13

INPUT_SHP = f"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718_{DISTANCE}.shp"

STEP3_CSV = "step3_points_lejepa.csv"
GRID_SHP = "step4_slide_grids_lejepa.shp"

MODEL_PATH = r"data/models/lejepa_encoder.pth"
EMBEDDING_PATH = r"data/embeddings/train_embeddings.npy"
LABEL_PATH = r"data/label/train_labels.npy"

CELL_SIZE = 8
def load_data():
    """
    Load shapefiles and numpy arrays.
    Ground truth is not loaded for the sliding process, 
    as we only rely on random trees, embeddings, and labels.
    """
    # Load random shifted trees
    random_trees = gpd.read_file(INPUT_SHP)
    
    # Load embeddings and labels
    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABEL_PATH)
    
    return random_trees, embeddings, labels

def create_3x3_grid(center_point, cell_size=5.0):
    """
    Create a 3x3 grid (9 boxes) around a center point.
    cell_size defines the width/height of each small box.
    """
    x, y = center_point.x, center_point.y
    
    boxes = []
    # Create 9 boxes from top-left to bottom-right
    for i in range(-1, 2):      # y offsets: 1, 0, -1
        for j in range(-1, 2):  # x offsets: -1, 0, 1
            minx = x + (j * cell_size) - (cell_size / 2)
            maxx = x + (j * cell_size) + (cell_size / 2)
            miny = y - (i * cell_size) - (cell_size / 2)
            maxy = y - (i * cell_size) + (cell_size / 2)
            
            box_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            boxes.append(box_poly)
            
    return boxes

def calculate_likelihood(box, tree_label, embeddings, labels):
    """
    Calculate the likelihood of a box containing the target tree crown.
    This purely depends on the provided embeddings and labels, without any Ground Truth.
    """
    # TODO: Implement the actual mapping from spatial coordinates (box) 
    # to the indices of 'embeddings.npy' and 'train_labels.npy'.
    # For example, if labels.npy contains spatial maps of tree species predictions,
    # you would crop the corresponding area from labels and embeddings,
    # and compute the likelihood that the box contains 'tree_label'.
    
    # For demonstration, we use a random proxy likelihood between 0 and 1.
    # In your actual implementation, this will be replaced by your logic using
    # embeddings and labels.
    likelihood = np.random.uniform(0.0, 1.0)
    
    return likelihood

def process_slide_grid(current_points, embeddings, labels, cell_size=5.0):
    """
    Execute one step of the slide_grid algorithm.
    Returns the new center points and the slide points.
    """
    new_centers = []
    new_slides = []
    
    # Identify the column containing the label/species
    label_col = current_points.columns[0]
    
    for idx, row in current_points.iterrows():
        point = row.geometry
        tree_label = row[label_col]
        
        # 1. Create 3x3 grid (9 boxes)
        boxes = create_3x3_grid(point, cell_size)
        
        # 2. Calculate likelihood for each box using ONLY embeddings/labels
        checked_boxes = []
        for box in boxes:
            likelihood = calculate_likelihood(box, tree_label, embeddings, labels)
            
            if likelihood >= 0.5:
                checked_boxes.append(box)
                
        # 3. Apply slide_grid rules based on the number of checked boxes
        num_checked = len(checked_boxes)
        
        if num_checked >= 3:
            # 3 or more checked boxes -> Center point
            avg_x = sum([b.centroid.x for b in checked_boxes]) / num_checked
            avg_y = sum([b.centroid.y for b in checked_boxes]) / num_checked
            new_centers.append({'geometry': Point(avg_x, avg_y), label_col: tree_label})
            
        elif 1 <= num_checked <= 2:
            # 1 or 2 checked boxes -> Slide point
            avg_x = sum([b.centroid.x for b in checked_boxes]) / num_checked
            avg_y = sum([b.centroid.y for b in checked_boxes]) / num_checked
            new_slides.append({'geometry': Point(avg_x, avg_y), label_col: tree_label})
            
        else:
            # 0 checked boxes
            # Rule: If likelihood is less than 0.5 for all, put a point and call it slide
            new_slides.append({'geometry': point, label_col: tree_label})
            
    # Convert lists back to GeoDataFrames
    crs = current_points.crs
    gdf_centers = gpd.GeoDataFrame(new_centers, crs=crs) if new_centers else gpd.GeoDataFrame(columns=[label_col, 'geometry'], crs=crs)
    gdf_slides = gpd.GeoDataFrame(new_slides, crs=crs) if new_slides else gpd.GeoDataFrame(columns=[label_col, 'geometry'], crs=crs)
    
    return gdf_centers, gdf_slides

def main():
    print("Loading data...")
    # Ground truth is intentionally NOT loaded here.
    random_trees, embeddings, labels = load_data()
    
    # Grid cell size configuration
    grid_cell_size = CELL_SIZE
    
    all_centers = []
    current_slides = random_trees
    
    # We loop 2 times for grid making and processing.
    num_grid_makings = 2
    
    print("Starting slide_grid algorithm...")
    for step in range(num_grid_makings):
        print(f"--- Step {step + 1} ---")
        if current_slides.empty:
            break
            
        # Perform grid making and point generation without GT
        centers, current_slides = process_slide_grid(
            current_slides, 
            embeddings, 
            labels, 
            cell_size=grid_cell_size
        )
        
        if not centers.empty:
            all_centers.append(centers)
            
        print(f"Generated {len(centers)} center points and {len(current_slides)} slide points.")
    
    # Combine remaining slides as final step coordinates
    if not current_slides.empty:
        all_centers.append(current_slides)
    
    # Combine all accumulated coordinates (centers + final slides)
    if all_centers:
        final_points_gdf = pd.concat(all_centers, ignore_index=True)
        final_points_gdf = gpd.GeoDataFrame(final_points_gdf, crs=random_trees.crs)
        
        print(f"Algorithm finished. Total final points: {len(final_points_gdf)}")
        
        # Save the result
        save_dir = f"data/distance/slide_grid_results_d:{DISTANCE}_cz:{CELL_SIZE}.shp"
        final_points_gdf.to_file(save_dir)
        print(f"Results saved to '{save_dir}'")
    else:
        print("No points generated.")

if __name__ == "__main__":
    main()