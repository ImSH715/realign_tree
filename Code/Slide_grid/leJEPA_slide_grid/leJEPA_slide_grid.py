import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import rasterio
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# Import the model class defined in 1409.py (modify the filename as needed)
from leJEPA_default import LeJepaEncoder 

# -----------------------------------------
# Configuration
# -----------------------------------------
ORIGINAL_COORDS_CSV = "original_coordinates.csv" # Original coordinates (including x, y, label)
TIF_DIR = "/mnt/parscratch/.../2023"
MODEL_WEIGHTS = "data/models/lejepa_encoder.pth"
EMBEDDINGS_NPY = "data/embeddings/embeddings.npy"
LABELS_NPY = "data/label/labels.npy"

OUTPUT_GRID_SHP = "output/final_grids.shp"
OUTPUT_CENTER_CSV = "output/final_centered_points.csv"

CELL_SIZE = 1.2 # Grid size
IMG_SIZE = 448
HALF_CROP = IMG_SIZE * 4 // 2
SIMILARITY_THRESHOLD = 0.8 # Use feature similarity threshold instead of intersection area

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------
# 1. Model and Reference Feature Preparation
# -----------------------------------------
encoder = LeJepaEncoder(img_size=IMG_SIZE).to(device)
encoder.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
encoder.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Calculate average embeddings (features) per label
all_embeddings = np.load(EMBEDDINGS_NPY)
all_labels = np.load(LABELS_NPY)

label_to_feature = {}
unique_labels = np.unique(all_labels)
for lbl in unique_labels:
    idx = np.where(all_labels == lbl)[0]
    mean_feat = np.mean(all_embeddings[idx], axis=0)
    label_to_feature[lbl] = torch.tensor(mean_feat).to(device)

def get_patch_embedding(src, px, py):
    """Extract patches from specific pixel coordinates and extract model features"""
    window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, HALF_CROP*2, HALF_CROP*2)
    tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
    tile = np.moveaxis(tile, 0, -1)
    
    # Padding logic (refer to outOfBoundChecker)
    if tile.shape[:2] != (HALF_CROP*2, HALF_CROP*2):
        h, w, c = tile.shape
        tile = np.pad(tile, ((0, max(0, HALF_CROP*2 - h)), (0, max(0, HALF_CROP*2 - w)), (0, 0)), mode='constant')
        tile = tile[:HALF_CROP*2, :HALF_CROP*2, :]
        
    img_pil = Image.fromarray(tile.astype('uint8'))
    tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat = encoder(tensor).mean(dim=1).squeeze(0) # [128]
    return feat

def compute_similarity(feat1, feat2):
    """Calculate cosine similarity between two features"""
    return torch.nn.functional.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

# -----------------------------------------
# 2. Integrated Main Logic (Grid -> Slide -> Center)
# -----------------------------------------
def process_points():
    df = pd.read_csv(ORIGINAL_COORDS_CSV)
    
    final_results = []
    grid_records = []
    
    # Assuming an arbitrary TIF file is opened (actually requires logic to find the TIF containing the coordinates)
    # Utilize the spatial join logic from 1409.py to match src
    # Here, for simplicity, we assume src is already matched.
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Points"):
        fid = row['feature_id']
        cx, cy = row['x'], row['y']
        target_label = str(row['label'])
        
        if target_label not in label_to_feature:
            continue
            
        target_feat = label_to_feature[target_label]
        
        # --- 1st Grid Generation and Evaluation ---
        def evaluate_grid(center_x, center_y):
            eval_records = []
            half = CELL_SIZE / 2
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    gx = center_x + dx * CELL_SIZE
                    gy = center_y + dy * CELL_SIZE
                    
                    # Calculate px, py of the coordinates in TIF and pass through the model
                    # py, px = src.index(gx, gy) 
                    # patch_feat = get_patch_embedding(src, px, py)
                    # sim = compute_similarity(patch_feat, target_feat)
                    
                    # Return dummy similarity (uncomment the above in actual implementation)
                    sim = 0.0 
                    
                    grid_geom = box(gx - half, gy - half, gx + half, gy + half)
                    
                    eval_records.append({
                        "grid_geom": grid_geom,
                        "likelihood": sim,
                        "cx": gx, "cy": gy
                    })
                    
                    grid_records.append({
                        "point_id": fid, "geometry": grid_geom, "likelihood": sim
                    })
                    
            return pd.DataFrame(eval_records)
        
        # 1st evaluation
        grid_df = evaluate_grid(cx, cy)
        high_prob = grid_df[grid_df['likelihood'] >= SIMILARITY_THRESHOLD]
        
        if len(high_prob) >= 3:
            # Step 1 condition met: Assign Center
            new_cx = high_prob['cx'].mean()
            new_cy = high_prob['cy'].mean()
            final_results.append({"feature_id": fid, "x": new_cx, "y": new_cy, "type": "center"})
            
        elif len(high_prob) in [1, 2]:
            # Step 1 condition unmet: Assign Slide point and generate 2nd Grid (Step 2 & 3 integrated)
            best_grid = high_prob.sort_values("likelihood", ascending=False).iloc[0]
            slide_cx, slide_cy = best_grid['cx'], best_grid['cy']
            
            # 2nd evaluation at the Slide location
            slide_grid_df = evaluate_grid(slide_cx, slide_cy)
            slide_high_prob = slide_grid_df[slide_grid_df['likelihood'] >= SIMILARITY_THRESHOLD]
            
            if len(slide_high_prob) >= 3:
                new_cx = slide_high_prob['cx'].mean()
                new_cy = slide_high_prob['cy'].mean()
            else:
                # If 1~2 boxes in the 2nd evaluation, set the one with the highest similarity as the final Center (Step 3 logic)
                best_final = slide_grid_df.sort_values("likelihood", ascending=False).iloc[0]
                new_cx, new_cy = best_final['cx'], best_final['cy']
                
            final_results.append({"feature_id": fid, "x": new_cx, "y": new_cy, "type": "center"})
            
        else:
            # If not found, keep original coordinates (or discard)
            final_results.append({"feature_id": fid, "x": cx, "y": cy, "type": "unresolved"})

    # -----------------------------------------
    # 3. Save Results (CSV and Shapefile)
    # -----------------------------------------
    # Final adjusted CSV
    out_csv = pd.DataFrame(final_results)
    out_csv.to_csv(OUTPUT_CENTER_CSV, index=False)
    print(f"Saved centered coordinates to: {OUTPUT_CENTER_CSV}")
    
    # Save all evaluated Grids as Shapefile
    grid_gdf = gpd.GeoDataFrame(grid_records, geometry="geometry", crs="EPSG:32718") # Use appropriate CRS
    grid_gdf.to_file(OUTPUT_GRID_SHP)
    print(f"Saved grid shapefile to: {OUTPUT_GRID_SHP}")

if __name__ == "__main__":
    # process_points()
    pass