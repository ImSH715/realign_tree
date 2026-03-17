import torch
import torch.nn as nn
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
# 1. Config & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ORIGINAL_POINTS_EXCEL = r"data/coordinate/Censo Forestal.xlsx" # Updated to Excel file
MODEL_PATH = r"data/models/lejepa_encoder.pth"
EMBEDDING_PATH = r"data/embeddings/embeddings.npy"
LABEL_PATH = r"data/label/labels.npy"

OUTPUT_CSV = r"../coordinate/final_lejepa_centered_points.csv"

# Excel Column Config (Change these to match your Excel header names)
X_COL = "COORDENADA_ESTE"           # Column name for X coordinate / Longitude / Easting
Y_COL = "COORDENADA_NORTE"           # Column name for Y coordinate / Latitude / Northing
LABEL_COL = "NOMBRE_COMUN" # Column name for Species / Label (e.g., 'Tree', 'Especie')
ID_COL = "ID"         # Column name for Unique ID (Optional)

# Coordinate Reference System of the Excel points (e.g., "EPSG:4326" for lat/lon, "EPSG:32718" for UTM 18S)
EXCEL_CRS = "EPSG:32718" 

# Model & Image Parameters
IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2

# Grid Parameters
CELL_SIZE = 1.2  # Size of each grid cell (adjust according to your CRS)
LIKELIHOOD_THRESHOLD = 0.75 # Cosine similarity threshold (0.0 ~ 1.0)

# --------------------------
# 2. LeJEPA Encoder Model Definition
# --------------------------
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, 
            activation='gelu', batch_first=True
        )
        self.blocks = nn.ModuleList([encoder_layer for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        if ids_keep is not None:
            D = x.shape[-1]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# --------------------------
# 3. Helper Functions
# --------------------------
def load_reference_embeddings(embed_path, label_path):
    """
    Load extracted embeddings and labels from 1409.py to compute 
    the mean reference embedding for each species (label).
    """
    embeds = np.load(embed_path)
    labels = np.load(label_path)
    
    ref_dict = {}
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        mean_embed = np.mean(embeds[idx], axis=0)
        ref_dict[lbl] = torch.tensor(mean_embed, dtype=torch.float32)
    return ref_dict

def extract_and_embed(src, x, y, encoder, transform, device):
    """
    Crop the image at the given coordinate (x, y) from the raster,
    pass it through the LeJEPA encoder, and return the feature embedding.
    """
    py, px = src.index(x, y)
    window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
    tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
    tile = np.moveaxis(tile, 0, -1)
    
    # Check if the cropped tile matches the expected dimensions
    if tile.shape[0] != CROP_SIZE or tile.shape[1] != CROP_SIZE:
        return None
        
    img_pil = Image.fromarray(tile.astype('uint8'))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Extract features and apply mean pooling
        embed = encoder(img_tensor).mean(dim=1).squeeze(0).cpu() 
    return embed

def scan_3x3_grid(cx, cy, src, encoder, transform, device, target_ref_embed):
    """
    Scan a 3x3 grid around the center coordinate (cx, cy).
    Returns a list of cells with cosine similarity >= LIKELIHOOD_THRESHOLD.
    """
    results = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            gx = cx + dx * CELL_SIZE
            gy = cy + dy * CELL_SIZE
            
            embed = extract_and_embed(src, gx, gy, encoder, transform, device)
            if embed is not None:
                # Calculate Cosine Similarity to represent "Likelihood"
                similarity = F.cosine_similarity(embed.unsqueeze(0), target_ref_embed.unsqueeze(0)).item()
                if similarity >= LIKELIHOOD_THRESHOLD:
                    results.append({"x": gx, "y": gy, "likelihood": similarity})
    return results

def calculate_center(valid_cells):
    """
    Compute the geometric center of the valid grid cells.
    """
    xs = [cell["x"] for cell in valid_cells]
    ys = [cell["y"] for cell in valid_cells]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# --------------------------
# 4. Main Inference Pipeline (Debug Version)
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading LeJEPA Model and Reference Embeddings...")
    
    encoder = LeJepaEncoder().to(device)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    encoder.eval()
    
    ref_embeddings = load_reference_embeddings(EMBEDDING_PATH, LABEL_PATH)
    print(f"[*] Model recognizes {len(ref_embeddings)} labels. Examples: {list(ref_embeddings.keys())[:5]}")
    
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
    
    print(f"[*] Successfully loaded {len(gdf)} coordinates from Excel.")
    print(f"[*] Sample Excel Label: {gdf[LABEL_COL].iloc[0]}")
    print(f"[*] Sample Excel Coordinate: X={gdf.geometry.x.iloc[0]}, Y={gdf.geometry.y.iloc[0]}")

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    final_points = []
    
    points_inside_tifs = 0
    points_label_matched = 0

    for tif_path in tqdm(tif_files, desc="Scanning TIFs for Grid Sliding"):
        try:
            with rasterio.open(tif_path) as src:
                current_gdf = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf.copy()
                b = src.bounds
                img_box = box(b.left, b.bottom, b.right, b.top)
                
                contained = current_gdf[current_gdf.geometry.intersects(img_box)]
                points_inside_tifs += len(contained)
                
                for idx, row in contained.iterrows():
                    orig_x, orig_y = row.geometry.x, row.geometry.y
                    label_val = str(row.get(LABEL_COL)).strip() # 공백 제거
                    
                    if label_val not in ref_embeddings:
                        continue 
                        
                    points_label_matched += 1
                    target_ref = ref_embeddings[label_val]
                    
                    # === STEP 1: Scan the first 3x3 grid ===
                    step1_valid = scan_3x3_grid(orig_x, orig_y, src, encoder, transform, device, target_ref)
                    
                    if len(step1_valid) >= 3:
                        nx, ny = calculate_center(step1_valid)
                        final_points.append({"feature_id": row.get(ID_COL, idx), "x": nx, "y": ny, "type": "center"})
                    elif len(step1_valid) > 0:
                        best_cell = max(step1_valid, key=lambda c: c["likelihood"])
                        slide_x, slide_y = best_cell["x"], best_cell["y"]
                        
                        # === STEP 2 & 3: New 3x3 grid on slid coordinate ===
                        step3_valid = scan_3x3_grid(slide_x, slide_y, src, encoder, transform, device, target_ref)
                        
                        if len(step3_valid) >= 3:
                            nx, ny = calculate_center(step3_valid)
                            final_points.append({"feature_id": row.get(ID_COL, idx), "x": nx, "y": ny, "type": "center"})
                        elif len(step3_valid) > 0:
                            best_cell_2 = max(step3_valid, key=lambda c: c["likelihood"])
                            final_points.append({"feature_id": row.get(ID_COL, idx), "x": best_cell_2["x"], "y": best_cell_2["y"], "type": "center"})
                        else:
                            final_points.append({"feature_id": row.get(ID_COL, idx), "x": slide_x, "y": slide_y, "type": "slide_unresolved"})
                    else:
                        final_points.append({"feature_id": row.get(ID_COL, idx), "x": orig_x, "y": orig_y, "type": "abort"})
                        
        except Exception as e:
            print(f"\n[!] Error processing {os.path.basename(tif_path)}: {e}")

    print("\n--- Diagnostic Summary ---")
    print(f"Total points overlapping with TIFs: {points_inside_tifs}")
    print(f"Total points with matching labels: {points_label_matched}")

    # === STEP 4: Merge and save ===
    if final_points:
        final_df = pd.DataFrame(final_points)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully saved centered points to {OUTPUT_CSV}")
        print("Summary of coordinate types:")
        print(final_df["type"].value_counts())
    else:
        print("\nNo valid points found or processed. Please check the diagnostics above.")

if __name__ == "__main__":
    main()