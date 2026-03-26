import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from lejepa_fixed import LeJepaEncoder

# --- [PATH CONFIGURATION] ---
CURATED_TIF_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
CURATED_SHP = r"/mnt/parscratch/users/acb20si/realign_tree/Code/Slide_grid/testing/data/tree_label_rdn/valid_points.shp"

LARGE_TIF_DIR = r"/mnt/parscratch/users/acb20si/2025_Turing_L/datasets/Osinfor/Ortomosaicos"
LARGE_CENSUS_CSV = "data/tree_label_rdn/Censo Forestal.xlsx - Datos.csv"
LABEL_COL_SHP = "Tree"
LABEL_COL_CSV = "NOMBRE_CIENTIFICO"
FINAL_OUTPUT = "OSINFOR_2023_Realigned_Final.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- [1. ERROR-RESISTANT TIF SPATIAL INDEXING] ---
def index_tifs(directory):
    index = []
    # Find all TIFs recursively
    files = glob.glob(os.path.join(directory, "**", "*.tif"), recursive=True)
    print(f"Indexing {len(files)} files in {directory}...")
    
    for f in tqdm(files, desc="Checking TIF Integrity"):
        try:
            with rasterio.open(f) as src:
                # If we can read the bounds, the file header is likely okay
                index.append({'path': f, 'bounds': src.bounds})
        except Exception as e:
            # This catches the TIFFReadDirectory error and skips the file
            print(f"\n[CORRUPT FILE SKIPPED]: {f}")
            print(f"Error details: {e}")
            continue
    return index

def find_tif(x, y, index):
    for item in index:
        b = item['bounds']
        if (b.left <= x <= b.right and b.bottom <= y <= b.top):
            return item['path']
    return None

# --- [2. PRODUCTION EXECUTION] ---
def run_realignment():
    # A. Index TIFs (Skips broken ones)
    curated_index = index_tifs(CURATED_TIF_DIR)
    large_index = index_tifs(LARGE_TIF_DIR)

    # B. Train Classifier on Small (Curated) Dataset
    print("\n[Step 1] Building Reference Classifier...")
    encoder = LeJepaEncoder().to(device) 
    encoder.eval()

    curated_gdf = gpd.read_file(CURATED_SHP)
    X_ref, y_ref = [], []
    
    for _, row in tqdm(curated_gdf.iterrows(), total=len(curated_gdf), desc="Extracting Training Feats"):
        path = find_tif(row.geometry.x, row.geometry.y, curated_index)
        if path:
            try:
                with rasterio.open(path) as src:
                    py, px = src.index(row.geometry.x, row.geometry.y)
                    win = rasterio.windows.Window(px-224, py-224, 448, 448)
                    img = src.read([1,2,3], window=win, boundless=True, fill_value=0)
                    img_t = torch.from_numpy(img).float().unsqueeze(0).to(device) / 255.0
                    with torch.no_grad():
                        feat = encoder(img_t).mean(dim=1).cpu().numpy().squeeze()
                        X_ref.append(feat)
                        y_ref.append(row[LABEL_COL_SHP])
            except Exception:
                continue # Skip individual point if read fails
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_ref, y_ref)
    known_species = set(clf.classes_)

    # C. Realign Large Census
    print(f"\n[Step 2] Processing {LARGE_CENSUS_CSV}...")
    df = pd.read_csv(LARGE_CENSUS_CSV)
    df[LABEL_COL_CSV] = df[LABEL_COL_CSV].str.strip()
    
    realigned_results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Realigning Census"):
        orig_x, orig_y = row['COORDENADA_ESTE'], row['COORDENADA_NORTE']
        target_sp = row[LABEL_COL_CSV]
        
        row['ORIGINAL_X'] = orig_x
        row['ORIGINAL_Y'] = orig_y
        row['REALIGNED_X'] = orig_x  
        row['REALIGNED_Y'] = orig_y  
        row['SHIFT_DISTANCE'] = 0.0
        row['STATUS'] = "UNPROCESSED"

        if target_sp not in known_species:
            row['STATUS'] = "KEEP_ORIGINAL_UNKNOWN_SPECIES"
            realigned_results.append(row)
            continue

        tif_path = find_tif(orig_x, orig_y, large_index)
        if not tif_path:
            row['STATUS'] = "KEEP_ORIGINAL_NO_IMAGERY_OR_CORRUPT"
            realigned_results.append(row)
            continue

        try:
            with rasterio.open(tif_path) as src:
                py, px = src.index(orig_x, orig_y)
                coords, preds = [], []
                for oy in range(-40, 40, 8):
                    for ox in range(-40, 40, 8):
                        win = rasterio.windows.Window(px+ox-224, py+oy-224, 448, 448)
                        img = src.read([1,2,3], window=win, boundless=True, fill_value=0)
                        img_t = torch.from_numpy(img).float().unsqueeze(0).to(device) / 255.0
                        with torch.no_grad():
                            tokens = encoder(img_t).mean(dim=1).cpu().numpy()
                            preds.append(clf.predict(tokens)[0])
                            mx, my = src.xy(py+oy, px+ox)
                            coords.append([mx, my])

                c_arr, p_arr = np.array(coords), np.array(preds)
                mask = (p_arr == target_sp)
                
                if mask.sum() >= 3:
                    db = DBSCAN(eps=5.0, min_samples=3).fit(c_arr[mask])
                    if len(set(db.labels_)) > (1 if -1 in db.labels_ else 0):
                        u, counts = np.unique(db.labels_[db.labels_!=-1], return_counts=True)
                        best_c = u[np.argmax(counts)]
                        new_pos = c_arr[mask][db.labels_ == best_c].mean(axis=0)
                        
                        row['REALIGNED_X'] = new_pos[0]
                        row['REALIGNED_Y'] = new_pos[1]
                        row['SHIFT_DISTANCE'] = np.linalg.norm(new_pos - [orig_x, orig_y])
                        row['STATUS'] = "SUCCESSFULLY_MOVED"
                    else:
                        row['STATUS'] = "KEEP_ORIGINAL_NO_CLUSTER"
                else:
                    row['STATUS'] = "KEEP_ORIGINAL_SPECIES_NOT_DETECTED"
        except Exception as e:
            row['STATUS'] = f"ERROR_READING_TIF_DURING_SCAN"

        realigned_results.append(row)
        
        if idx % 1000 == 0:
            pd.DataFrame(realigned_results).to_csv("census_realign_checkpoint.csv", index=False)

    final_df = pd.DataFrame(realigned_results)
    final_df.to_csv(FINAL_OUTPUT, index=False)
    print(f"\nDone! Results saved to {FINAL_OUTPUT}")

if __name__ == "__main__":
    run_realignment()