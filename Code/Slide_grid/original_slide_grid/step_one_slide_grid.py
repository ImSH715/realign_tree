import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
import torchvision.transforms as transforms
import os
import rasterio
import glob
from shapely.geometry import box
from tqdm import tqdm

# Import the existing LeJEPA model class
from leJepa import LeJepaEncoder

# ==========================================
# 1. Hyperparameters & Paths
# ==========================================
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
# Point features containing the labels and initial locations
INPUT_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"
# Pre-generated grid shapefile
GRID_SHP = r"C:\Users\naya0\Uni\1.COM-Turing\realign_tree\Code\Slide_grid\original_slide_grid\grid_result\3x3_grid_result.shp"

MODEL_PATH = r"data/models/lejepa_encoder.pth"
EMBEDDING_PATH = r"data/embeddings/train_embeddings.npy"
LABEL_PATH = r"data/label/train_labels.npy"

OUTPUT_CSV = "step1_points_lejepa.csv"

# Thresholds and parameters
LIKELIHOOD_THRESHOLD = 0.5
SEARCH_RADIUS = 2.5  # Radius in meters to find the 9 grid cells around a point

IMG_SIZE = 448
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2

# ==========================================
# 2. Helper Functions
# ==========================================
def load_reference_embeddings(embed_path, label_path):
    """Load combined embeddings and compute the mean reference vector (fingerprint) per label."""
    embeds = np.load(embed_path)
    labels = np.load(label_path)
    
    ref_dict = {}
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        mean_embed = np.mean(embeds[idx], axis=0)
        ref_dict[str(lbl).strip().lower()] = torch.tensor(mean_embed, dtype=torch.float32)
    return ref_dict

def extract_tile(src, x, y, transform):
    """Crop the 448x448 image patch from the TIF at the given coordinates."""
    try:
        py, px = src.index(x, y)
        window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
        tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        tile = np.moveaxis(tile, 0, -1)
        
        if tile.shape[0] != CROP_SIZE or tile.shape[1] != CROP_SIZE:
            return None
            
        img_pil = Image.fromarray(tile.astype('uint8'))
        return transform(img_pil)
    except Exception:
        return None

def center_from_cells(cells):
    """Calculates the center of the total bounds of the selected grids."""
    minx, miny, maxx, maxy = cells.total_bounds
    return (minx + maxx) / 2, (miny + maxy) / 2

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading LeJEPA Model and References...")
    # Initialize the model imported from LeJepa.py
    encoder = LeJepaEncoder().to(device)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    encoder.eval()
    
    ref_embeddings = load_reference_embeddings(EMBEDDING_PATH, LABEL_PATH)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Shapefiles...")
    feat_gdf = gpd.read_file(INPUT_SHP)
    grid_gdf = gpd.read_file(GRID_SHP)
    grid_sindex = grid_gdf.sindex

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    
    results = []
    processed_fids = set()

    print("Processing Points through TIFs...")
    for tif_path in tqdm(tif_files, desc="TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*src.bounds)]}, crs=src.crs)
                
                # Check projection
                if bbox_gdf.crs != feat_gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(feat_gdf.crs)
                    
                contained = feat_gdf[feat_gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
                if contained.empty:
                    continue
                    
                for idx, feat_row in contained.iterrows():
                    fid = feat_row.get('temp_id', idx)
                    if fid in processed_fids:
                        continue
                    
                    feat_geom = feat_row.geometry
                    raw_label = feat_row.get('Tree') or feat_row.get('tree') or feat_row.get('id') or f"tree_{fid}"
                    label_val = str(raw_label).strip().lower()
                    
                    if label_val not in ref_embeddings:
                        continue
                        
                    target_ref = ref_embeddings[label_val]

                    # Find grids belonging to this point using a small buffer
                    search_area = feat_geom.buffer(SEARCH_RADIUS)
                    cand_idx = list(grid_sindex.intersection(search_area.bounds))
                    
                    if not cand_idx:
                        continue
                        
                    cell_records = []
                    tensors = []
                    valid_gids = []

                    # Extract image and evaluate for each nearby grid cell
                    for gi in cand_idx:
                        grid_geom = grid_gdf.geometry.iloc[gi]
                        if not grid_geom.intersects(search_area):
                            continue
                            
                        # Use the centroid of the grid cell for the camera location
                        cx, cy = grid_geom.centroid.x, grid_geom.centroid.y
                        tensor = extract_tile(src, cx, cy, transform)
                        
                        if tensor is not None:
                            tensors.append(tensor)
                            valid_gids.append(gi)

                    if not tensors:
                        continue

                    # Batch process through model
                    batch_tensor = torch.stack(tensors).to(device)
                    with torch.no_grad():
                        embeds = encoder(batch_tensor).mean(dim=1).cpu() 
                        
                    # Compute Cosine Similarity (New Likelihood)
                    target_ref_expanded = target_ref.unsqueeze(0).expand(len(embeds), -1)
                    similarities = F.cosine_similarity(embeds, target_ref_expanded).numpy()

                    # Filter based on your LIKELIHOOD_THRESHOLD
                    df_records = []
                    for i, sim in enumerate(similarities):
                        if sim >= LIKELIHOOD_THRESHOLD:
                            df_records.append({"grid_id": valid_gids[i], "likelihood": sim})
                            
                    if not df_records:
                        processed_fids.add(fid)
                        continue
                        
                    df = pd.DataFrame(df_records)

                    # ------------------------------------------------
                    # Apply the rule set
                    # ------------------------------------------------
                    # Case 1: 3 or more grid cells
                    if len(df) >= 3:
                        selected_cells = grid_gdf.loc[df["grid_id"]]
                        cx, cy = center_from_cells(selected_cells)
                        
                        results.append({
                            "x": cx,
                            "y": cy,
                            "feature_id": fid,
                            "label": label_val,
                            "type": "center"
                        })
                        
                    # Case 2: 1 or 2 grid cells
                    else:
                        best_gid = df.sort_values("likelihood", ascending=False).iloc[0]["grid_id"]
                        best_geom = grid_gdf.geometry.loc[best_gid]
                        
                        # In the semantic space, the "overlap point" is simply the center of the best matching grid
                        pt_x, pt_y = best_geom.centroid.x, best_geom.centroid.y
                        
                        results.append({
                            "x": pt_x,
                            "y": pt_y,
                            "feature_id": fid,
                            "label": label_val,
                            "type": "slide"
                        })
                    
                    processed_fids.add(fid)

        except Exception as e:
            pass

    # Save output
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(df_out)} points to: {OUTPUT_CSV}")
        print(df_out["type"].value_counts().to_string())
    else:
        print("\nNo points generated.")

if __name__ == "__main__":
    main()