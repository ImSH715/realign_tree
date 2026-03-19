import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist

# --------------------------
# Paths
# --------------------------
RANDOM_TREES_SHP = "data/random_trees.shp"

EMBEDDING_PATH = "data/embeddings/embeddings.npy"
COORD_PATH = "data/embeddings/coords.npy"
LABEL_PATH = "data/label/labels.npy"

OUTPUT_SHP = "slide_grid_results.shp"

# --------------------------
# Load Data
# --------------------------
def load_data():
    random_trees = gpd.read_file(RANDOM_TREES_SHP)

    embeddings = np.load(EMBEDDING_PATH)
    coords = np.load(COORD_PATH)
    labels = np.load(LABEL_PATH)

    return random_trees, embeddings, coords, labels

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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
# Main
# --------------------------
def main():
    random_trees, embeddings, coords, labels = load_data()

    current = random_trees.copy()

    NUM_STEPS = 3
    CELL_SIZE = 6.0

    for step in range(NUM_STEPS):
        print(f"Step {step+1}")
        current = process_slide_grid(
            current,
            embeddings,
            coords,
            labels,
            CELL_SIZE
        )

    current.to_file(OUTPUT_SHP)
    print("Saved results:", OUTPUT_SHP)


if __name__ == "__main__":
    main()