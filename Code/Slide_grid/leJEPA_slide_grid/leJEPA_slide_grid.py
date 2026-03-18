import torch
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
import torch.nn.functional as F

# --------------------------
# 1. Import Model
# --------------------------
from currated_test_train_SI import LeJepaEncoder

# --------------------------
# 2. Config & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ORIGINAL_POINTS_EXCEL = r"data/coordinate/Censo Forestal.xlsx" 
MODEL_PATH = r"data/models/lejepa_encoder.pth"

# Use TRAIN embeddings as the reference "fingerprints" to prevent data leakage
EMBEDDING_PATH = r"data/embeddings/combined_embeddings.npy"
LABEL_PATH = r"data/label/combined_labels.npy"

OUTPUT_CSV = r"../coordinate/final_lejepa_centered_points.csv"

# Excel Column Config 
X_COL = "COORDENADA_ESTE"           
Y_COL = "COORDENADA_NORTE"          
LABEL_COL = "NOMBRE_COMUN" 
ID_COL = "ID"        

EXCEL_CRS = "EPSG:32718" 

# Image Parameters
IMG_SIZE = 448
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2

# Grid Parameters
CELL_SIZE = 1.2  
LIKELIHOOD_THRESHOLD = 0.75 

# --------------------------
# 3. Helper Functions 
# --------------------------
def load_reference_embeddings(embed_path, label_path):
    """Load embeddings and labels, compute mean reference embedding per label."""
    embeds = np.load(embed_path)
    labels = np.load(label_path)
    
    ref_dict = {}
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        mean_embed = np.mean(embeds[idx], axis=0)
        ref_dict[lbl] = torch.tensor(mean_embed, dtype=torch.float32)
    return ref_dict

def extract_tile(src, x, y, transform):
    """Crop the image and convert to Tensor."""
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

def scan_3x3_grid_batched(cx, cy, src, encoder, transform, device, target_ref_embed):
    """Batch process 9 grid images at once for massive speedup."""
    coords = []
    tensors = []
    
    # 1. Collect 9 grid images
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            gx = cx + dx * CELL_SIZE
            gy = cy + dy * CELL_SIZE
            
            tensor = extract_tile(src, gx, gy, transform)
            if tensor is not None:
                coords.append({"x": gx, "y": gy})
                tensors.append(tensor)
                
    if not tensors:
        return []

    # 2. Stack 9 images into a single batch and pass through GPU
    batch_tensor = torch.stack(tensors).to(device)
    
    with torch.no_grad():
        embeds = encoder(batch_tensor).mean(dim=1).cpu() 
        
    # 3. Calculate cosine similarity for the batch
    target_ref_expanded = target_ref_embed.unsqueeze(0).expand(len(embeds), -1)
    similarities = F.cosine_similarity(embeds, target_ref_expanded).numpy()
    
    # 4. Format results
    results = []
    for i, sim in enumerate(similarities):
        if sim >= LIKELIHOOD_THRESHOLD:
            results.append({"x": coords[i]["x"], "y": coords[i]["y"], "likelihood": sim})
            
    return results

def calculate_center(valid_cells):
    """Compute the geometric center of the valid grid cells."""
    xs = [cell["x"] for cell in valid_cells]
    ys = [cell["y"] for cell in valid_cells]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# --------------------------
# 4. Main Inference Pipeline
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading LeJEPA Model and Reference Embeddings...")
    
    encoder = LeJepaEncoder().to(device)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    encoder.eval()
    
    raw_ref_embeddings = load_reference_embeddings(EMBEDDING_PATH, LABEL_PATH)
    ref_embeddings = {str(k).strip().lower(): v for k, v in raw_ref_embeddings.items()}
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Original Coordinates from Excel...")
    df = pd.read_excel(ORIGINAL_POINTS_EXCEL)
    df[X_COL] = pd.to_numeric(df[X_COL], errors='coerce')
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors='coerce')
    df = df.dropna(subset=[X_COL, Y_COL]).reset_index(drop=True)
    
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[X_COL], df[Y_COL]),
        crs=EXCEL_CRS
    )
    
    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    final_points = []
    
    points_inside_tifs = 0
    points_label_matched = 0

    print(f"Starting Sliding Grid alignment over {len(tif_files)} TIFs...")

    for tif_path in tqdm(tif_files, desc="Processing TIF Files", position=0):
        try:
            with rasterio.open(tif_path) as src:
                b = src.bounds
                img_box = box(b.left, b.bottom, b.right, b.top)
                
                # ⚡ Speed Optimization: Fast extraction using Bounding Box ⚡
                bbox_gdf = gpd.GeoDataFrame({'geometry': [img_box]}, crs=src.crs)
                if bbox_gdf.crs != gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(gdf.crs)
                img_box_transformed = bbox_gdf.geometry.iloc[0]

                contained = gdf[gdf.geometry.intersects(img_box_transformed)]
                if contained.empty:
                    continue 
                
                contained = contained.copy()
                if contained.crs != src.crs:
                    contained = contained.to_crs(src.crs)
                
                points_inside_tifs += len(contained)
                
                inner_pbar = tqdm(contained.iterrows(), total=len(contained), 
                                  desc=f"Trees in {os.path.basename(tif_path)[:15]}...",
                                  leave=False, position=1)
                
                for idx, row in inner_pbar:
                    orig_x, orig_y = row.geometry.x, row.geometry.y
                    label_val = str(row.get(LABEL_COL)).strip().lower() 
                    f_id = row.get(ID_COL, idx)
                    
                    if label_val not in ref_embeddings:
                        continue 
                        
                    points_label_matched += 1
                    target_ref = ref_embeddings[label_val]
                    
                    # -----------------------------------------------------
                    # SLIDING GRID ALGORITHM
                    # -----------------------------------------------------
                    # STEP 1: Scan the first 3x3 grid
                    step1_valid = scan_3x3_grid_batched(orig_x, orig_y, src, encoder, transform, device, target_ref)
                    
                    if len(step1_valid) >= 3:
                        nx, ny = calculate_center(step1_valid)
                        final_points.append({"feature_id": f_id, "label": label_val, "orig_x": orig_x, "orig_y": orig_y, "new_x": nx, "new_y": ny, "status": "Centered_Step1"})
                    
                    elif len(step1_valid) > 0:
                        best_cell = max(step1_valid, key=lambda c: c["likelihood"])
                        slide_x, slide_y = best_cell["x"], best_cell["y"]
                        
                        # STEP 2 & 3: Slide grid to the best cell and scan again
                        step3_valid = scan_3x3_grid_batched(slide_x, slide_y, src, encoder, transform, device, target_ref)
                        
                        if len(step3_valid) >= 3:
                            nx, ny = calculate_center(step3_valid)
                            final_points.append({"feature_id": f_id, "label": label_val, "orig_x": orig_x, "orig_y": orig_y, "new_x": nx, "new_y": ny, "status": "Centered_Step2"})
                        elif len(step3_valid) > 0:
                            best_cell_2 = max(step3_valid, key=lambda c: c["likelihood"])
                            final_points.append({"feature_id": f_id, "label": label_val, "orig_x": orig_x, "orig_y": orig_y, "new_x": best_cell_2["x"], "new_y": best_cell_2["y"], "status": "Shifted_BestCell"})
                        else:
                            final_points.append({"feature_id": f_id, "label": label_val, "orig_x": orig_x, "orig_y": orig_y, "new_x": slide_x, "new_y": slide_y, "status": "Shifted_Unresolved"})
                    
                    else:
                        # No matches found, keep original
                        final_points.append({"feature_id": f_id, "label": label_val, "orig_x": orig_x, "orig_y": orig_y, "new_x": orig_x, "new_y": orig_y, "status": "Aborted_NoMatch"})
                        
        except Exception as e:
            pass  

    print("\n\n--- Diagnostic Summary ---")
    print(f"Total points overlapping with TIFs: {points_inside_tifs}")
    print(f"Total points with matching labels: {points_label_matched}")

    if final_points:
        final_df = pd.DataFrame(final_points)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully saved centered points to {OUTPUT_CSV}")
        print("\n=== Adjustment Status Breakdown ===")
        print(final_df["status"].value_counts())
    else:
        print("\nNo valid points found or processed.")

if __name__ == "__main__":
    main()