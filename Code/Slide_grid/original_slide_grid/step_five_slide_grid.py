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

# ==========================================
# 1. Configuration & Paths
# ==========================================
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
INPUT_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"

# 입력 파일 (Step 3의 실패자들 & Step 4의 새 그리드)
STEP3_CSV = "step3_points_lejepa.csv"
GRID_SHP = "step4_slide_grids_lejepa.shp"

MODEL_PATH = r"data/models/lejepa_encoder.pth"
EMBEDDING_PATH = r"data/embeddings/train_embeddings.npy"
LABEL_PATH = r"data/label/train_labels.npy"

# 출력 파일
OUTPUT_CSV = "step5_points_lejepa.csv"

LIKELIHOOD_THRESHOLD = 0.75
IMG_SIZE = 448
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2

def load_reference_embeddings(embed_path, label_path):
    embeds, labels = np.load(embed_path), np.load(label_path)
    ref_dict = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        ref_dict[str(lbl).strip().lower()] = torch.tensor(np.mean(embeds[idx], axis=0), dtype=torch.float32)
    return ref_dict

def extract_tile(src, x, y, transform):
    try:
        py, px = src.index(x, y)
        window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
        tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        tile = np.moveaxis(tile, 0, -1)
        if tile.shape[0] != CROP_SIZE or tile.shape[1] != CROP_SIZE: return None
        return transform(Image.fromarray(tile.astype('uint8')))
    except: return None

def center_from_grid_cells(cells):
    xs, ys = [g.centroid.x for g in cells.geometry], [g.centroid.y for g in cells.geometry]
    return sum(xs) / len(xs), sum(ys) / len(ys)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LeJepaEncoder().to(device)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    encoder.eval()
    
    ref_embeddings = load_reference_embeddings(EMBEDDING_PATH, LABEL_PATH)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feat_gdf = gpd.read_file(INPUT_SHP)
    try:
        grid_gdf = gpd.read_file(GRID_SHP)
        prev_df = pd.read_csv(STEP3_CSV)
    except Exception as e:
        print(f"Error loading Step 3/4 data. Run them first."); return

    slide_features = prev_df[prev_df["type"] == "slide"]["feature_id"].unique()
    if len(slide_features) == 0: return

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    results, processed_fids = [], set()

    for tif_path in tqdm(tif_files, desc="TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*src.bounds)]}, crs=src.crs)
                if bbox_gdf.crs != feat_gdf.crs: bbox_gdf = bbox_gdf.to_crs(feat_gdf.crs)
                contained = feat_gdf[feat_gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
                if contained.empty: continue
                    
                for idx, feat_row in contained.iterrows():
                    fid = feat_row.get('temp_id', idx)
                    if fid not in slide_features or fid in processed_fids: continue
                    
                    label_val = str(feat_row.get('Tree', f"tree_{fid}")).strip().lower()
                    if label_val not in ref_embeddings: continue
                    
                    grids = grid_gdf[grid_gdf["point_id"] == fid]
                    if grids.empty: continue
                        
                    tensors, valid_gids = [], []
                    for gi, grid_row in grids.iterrows():
                        cx, cy = grid_row.geometry.centroid.x, grid_row.geometry.centroid.y
                        tensor = extract_tile(src, cx, cy, transform)
                        if tensor is not None:
                            tensors.append(tensor); valid_gids.append(gi)

                    if not tensors:
                        orig_row = prev_df[prev_df["feature_id"] == fid].iloc[0]
                        results.append({"x": orig_row["x"], "y": orig_row["y"], "feature_id": fid, "label": label_val, "type": "slide"})
                        processed_fids.add(fid); continue

                    batch_tensor = torch.stack(tensors).to(device)
                    with torch.no_grad():
                        embeds = encoder(batch_tensor).mean(dim=1).cpu() 
                    
                    similarities = F.cosine_similarity(embeds, ref_embeddings[label_val].unsqueeze(0).expand(len(embeds), -1)).numpy()
                    df_records = [{"grid_index": valid_gids[i], "likelihood": sim} for i, sim in enumerate(similarities) if sim >= LIKELIHOOD_THRESHOLD]

                    if not df_records:
                        best_geom = grid_gdf.geometry.loc[valid_gids[int(np.argmax(similarities))]]
                        results.append({"x": best_geom.centroid.x, "y": best_geom.centroid.y, "feature_id": fid, "label": label_val, "type": "slide"})
                        processed_fids.add(fid); continue
                        
                    df = pd.DataFrame(df_records)
                    if len(df) >= 3:
                        cx, cy = center_from_grid_cells(grid_gdf.loc[df["grid_index"]])
                    else:
                        best_geom = grid_gdf.geometry.loc[df.sort_values("likelihood", ascending=False).iloc[0]["grid_index"]]
                        cx, cy = best_geom.centroid.x, best_geom.centroid.y
                        
                    results.append({"x": cx, "y": cy, "feature_id": fid, "label": label_val, "type": "center"})
                    processed_fids.add(fid)
        except Exception: pass

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(out_df)} updated points to: {OUTPUT_CSV}")
        print(out_df["type"].value_counts().to_string())

if __name__ == "__main__":
    main()