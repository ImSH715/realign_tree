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

from leJepa import LeJepaEncoder

"""
Do the same procedure as step one using LeJEPA embeddings.
Evaluate the 3x3 grids generated in Step 2.
Turn type 'slide' to 'center' based on the newly calculated coordinates.
(Fallback logic included to prevent any dropped points)
"""

# ==========================================
# 1. Configuration & Paths
# ==========================================
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
INPUT_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"

# Files generated from previous steps
STEP1_CSV = "step1_points_lejepa.csv"
GRID_SHP = "step2_slide_grids_lejepa.shp"

# LeJEPA model files
MODEL_PATH = r"data/models/lejepa_encoder.pth"
EMBEDDING_PATH = r"data/embeddings/train_embeddings.npy"
LABEL_PATH = r"data/label/train_labels.npy"

OUTPUT_CSV = "step3_points_lejepa.csv"

# Thresholds and parameters
LIKELIHOOD_THRESHOLD = 0.75
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
    """Crop the image patch from the TIF at the given coordinates."""
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

def center_from_grid_cells(cells):
    """Calculates the center point based on the centroids of selected grid geometries."""
    xs = [geom.centroid.x for geom in cells.geometry]
    ys = [geom.centroid.y for geom in cells.geometry]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading LeJEPA Model and References...")
    encoder = LeJepaEncoder().to(device)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    encoder.eval()
    
    ref_embeddings = load_reference_embeddings(EMBEDDING_PATH, LABEL_PATH)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Data...")
    feat_gdf = gpd.read_file(INPUT_SHP)
    
    try:
        grid_gdf = gpd.read_file(GRID_SHP)
        step1_df = pd.read_csv(STEP1_CSV)
    except Exception as e:
        print(f"Error loading Step 1/2 data: {e}. Please run Step 1 and 2 first.")
        return

    # Identify features that need processing in Step 3
    slide_features = step1_df[step1_df["type"] == "slide"]["feature_id"].unique()
    if len(slide_features) == 0:
        print("No slide features found in Step 1. Exiting Step 3.")
        return

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    
    results = []
    processed_fids = set()

    print(f"Processing {len(slide_features)} slide points through TIFs...")
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
                    
                    # Only process points that were marked as 'slide' and not yet processed
                    if fid not in slide_features or fid in processed_fids:
                        continue
                    
                    raw_label = feat_row.get('Tree') or feat_row.get('tree') or feat_row.get('id') or f"tree_{fid}"
                    label_val = str(raw_label).strip().lower()
                    
                    if label_val not in ref_embeddings:
                        continue
                        
                    target_ref = ref_embeddings[label_val]

                    # Get the 3x3 grids specifically created for this point in Step 2
                    grids = grid_gdf[grid_gdf["point_id"] == fid]
                    if grids.empty:
                        continue
                        
                    tensors = []
                    valid_gids = []

                    # Extract image and evaluate for each grid cell
                    for gi, grid_row in grids.iterrows():
                        grid_geom = grid_row.geometry
                        cx, cy = grid_geom.centroid.x, grid_geom.centroid.y
                        
                        tensor = extract_tile(src, cx, cy, transform)
                        if tensor is not None:
                            tensors.append(tensor)
                            valid_gids.append(gi)

                    # [누락 방지] 이미지를 아예 추출하지 못했을 경우 Step 1의 원본 좌표 유지
                    if not tensors:
                        orig_row = step1_df[step1_df["feature_id"] == fid].iloc[0]
                        results.append({
                            "x": orig_row["x"],
                            "y": orig_row["y"],
                            "feature_id": fid,
                            "label": label_val,
                            "type": "slide" # 최종 실패로 기록
                        })
                        processed_fids.add(fid)
                        continue

                    # Batch process through model
                    batch_tensor = torch.stack(tensors).to(device)
                    with torch.no_grad():
                        embeds = encoder(batch_tensor).mean(dim=1).cpu() 
                        
                    # Compute Cosine Similarity
                    target_ref_expanded = target_ref.unsqueeze(0).expand(len(embeds), -1)
                    similarities = F.cosine_similarity(embeds, target_ref_expanded).numpy()

                    # Filter based on your LIKELIHOOD_THRESHOLD
                    df_records = []
                    for i, sim in enumerate(similarities):
                        if sim >= LIKELIHOOD_THRESHOLD:
                            df_records.append({"grid_index": valid_gids[i], "likelihood": sim})
                            
                    # ================================================
                    # [누락 방지] 0.5를 넘는 박스가 하나도 없을 때
                    # ================================================
                    if not df_records:
                        best_idx = int(np.argmax(similarities))
                        best_gid = valid_gids[best_idx]
                        best_geom = grid_gdf.geometry.loc[best_gid]
                        
                        results.append({
                            "x": best_geom.centroid.x,
                            "y": best_geom.centroid.y,
                            "feature_id": fid,
                            "label": label_val,
                            "type": "slide" # 최종 실패 (0.5 못 넘음)
                        })
                        processed_fids.add(fid)
                        continue
                        
                    df = pd.DataFrame(df_records)

                    # ------------------------------------------------
                    # Apply the rule set (Success output as 'center')
                    # ------------------------------------------------
                    # Case A: 3 or more boxes
                    if len(df) >= 3:
                        selected = grid_gdf.loc[df["grid_index"]]
                        cx, cy = center_from_grid_cells(selected)
                        
                    # Case B: 1-2 boxes
                    else:
                        best_idx = df.sort_values("likelihood", ascending=False).iloc[0]["grid_index"]
                        best_geom = grid_gdf.geometry.loc[best_idx]
                        cx, cy = best_geom.centroid.x, best_geom.centroid.y
                        
                    results.append({
                        "x": cx,
                        "y": cy,
                        "feature_id": fid,
                        "label": label_val,
                        "type": "center" # 성공
                    })
                    
                    processed_fids.add(fid)

        except Exception as e:
            pass

    # Save output
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(out_df)} updated points to: {OUTPUT_CSV}")
        print(out_df["type"].value_counts().to_string())
    else:
        print("\nNo points were processed.")

if __name__ == "__main__":
    main()