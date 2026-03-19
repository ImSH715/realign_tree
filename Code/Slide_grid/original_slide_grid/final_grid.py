import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from sklearn.metrics.pairwise import cosine_similarity

INPUT_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718_13.shp"

EMBEDDING_PATH = r"data/embeddings/train_embeddings.npy"
LABEL_PATH = r"data/label/train_labels.npy"

OUTPUT_SHP = "slide_grid_results.shp"


def load_data():
    """
    Load shapefile, embeddings, and labels.
    Assumes that the order of shapefile rows matches embeddings and labels.
    """
    random_trees = gpd.read_file(INPUT_SHP)
    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABEL_PATH)

    return random_trees, embeddings, labels


def extract_coordinates(gdf):
    """
    Extract (x, y) coordinates from GeoDataFrame geometry.
    """
    return np.array([[geom.x, geom.y] for geom in gdf.geometry])


def create_3x3_grid(center_point, cell_size=5.0):
    """
    Create a 3x3 grid (9 boxes) around a center point.
    Each box has size defined by cell_size.
    """
    x, y = center_point.x, center_point.y
    boxes = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            minx = x + (j * cell_size) - (cell_size / 2)
            maxx = x + (j * cell_size) + (cell_size / 2)
            miny = y - (i * cell_size) - (cell_size / 2)
            maxy = y - (i * cell_size) + (cell_size / 2)

            box = Polygon([
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy)
            ])
            boxes.append(box)

    return boxes


def get_points_in_box(box, coords):
    """
    Return indices of points inside a given box.
    """
    minx, miny, maxx, maxy = box.bounds

    mask = (
        (coords[:, 0] >= minx) &
        (coords[:, 0] <= maxx) &
        (coords[:, 1] >= miny) &
        (coords[:, 1] <= maxy)
    )

    return np.where(mask)[0]


def calculate_likelihood(box, tree_label, coords, embeddings, labels):
    """
    Compute likelihood that a box contains the target tree crown.

    Steps:
    1. Find points inside the box
    2. Filter points by same species
    3. Measure embedding consistency using cosine similarity
    """
    indices = get_points_in_box(box, coords)

    if len(indices) == 0:
        return 0.0

    # Filter by same species
    same_species_indices = [i for i in indices if labels[i] == tree_label]

    if len(same_species_indices) == 0:
        return 0.0

    # Extract embeddings
    selected_embeddings = embeddings[same_species_indices]

    # Compute similarity to mean embedding
    mean_embedding = selected_embeddings.mean(axis=0, keepdims=True)
    similarity = cosine_similarity(mean_embedding, selected_embeddings).mean()

    return similarity


def process_slide_grid(current_points, coords, embeddings, labels, cell_size=5.0):
    """
    Perform one iteration of the sliding grid algorithm.

    Returns:
    - center points (confident locations)
    - slide points (uncertain, to be refined further)
    """
    new_centers = []
    new_slides = []

    label_col = current_points.columns[0]

    for _, row in current_points.iterrows():
        point = row.geometry
        tree_label = row[label_col]

        boxes = create_3x3_grid(point, cell_size)

        valid_boxes = []

        for box in boxes:
            likelihood = calculate_likelihood(
                box, tree_label, coords, embeddings, labels
            )

            if likelihood >= 0.7:
                valid_boxes.append(box)

        num_valid = len(valid_boxes)

        if num_valid >= 3:
            avg_x = np.mean([b.centroid.x for b in valid_boxes])
            avg_y = np.mean([b.centroid.y for b in valid_boxes])

            new_centers.append({
                label_col: tree_label,
                "geometry": Point(avg_x, avg_y)
            })

        elif 1 <= num_valid <= 2:
            avg_x = np.mean([b.centroid.x for b in valid_boxes])
            avg_y = np.mean([b.centroid.y for b in valid_boxes])

            new_slides.append({
                label_col: tree_label,
                "geometry": Point(avg_x, avg_y)
            })

        else:
            new_slides.append({
                label_col: tree_label,
                "geometry": point
            })

    crs = current_points.crs

    gdf_centers = (
        gpd.GeoDataFrame(new_centers, crs=crs)
        if new_centers else
        gpd.GeoDataFrame(columns=[label_col, "geometry"], crs=crs)
    )

    gdf_slides = (
        gpd.GeoDataFrame(new_slides, crs=crs)
        if new_slides else
        gpd.GeoDataFrame(columns=[label_col, "geometry"], crs=crs)
    )

    return gdf_centers, gdf_slides


def main():
    print("Loading data...")
    random_trees, embeddings, labels = load_data()

    # Validate data alignment
    assert len(random_trees) == len(embeddings) == len(labels), \
        "Mismatch in data lengths"

    coords = extract_coordinates(random_trees)

    grid_cell_size = 5.5
    num_iterations = 2

    all_centers = []
    current_slides = random_trees

    print("Starting sliding grid process...")

    for step in range(num_iterations):
        print(f"Step {step + 1}")

        if current_slides.empty:
            break

        centers, current_slides = process_slide_grid(
            current_slides,
            coords,
            embeddings,
            labels,
            cell_size=grid_cell_size
        )

        if not centers.empty:
            all_centers.append(centers)

        print(f"Centers: {len(centers)}, Slides: {len(current_slides)}")

    if not current_slides.empty:
        all_centers.append(current_slides)

    if all_centers:
        final_gdf = pd.concat(all_centers, ignore_index=True)
        final_gdf = gpd.GeoDataFrame(final_gdf, crs=random_trees.crs)

        print(f"Total output points: {len(final_gdf)}")

        final_gdf.to_file(OUTPUT_SHP)
        print(f"Saved to {OUTPUT_SHP}")
    else:
        print("No output generated")


if __name__ == "__main__":
    main()