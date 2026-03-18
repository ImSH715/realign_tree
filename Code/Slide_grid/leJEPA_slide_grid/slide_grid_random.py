import torch
import torch.nn as nn
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

# ==========================================
# 1. Hyperparameters & Paths
# ==========================================
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"

MODEL_PATH = r"data/models/lejepa_encoder.pth"
EMBEDDING_PATH = r"data/embeddings/combined_embeddings.npy"
LABEL_PATH = r"data/label/combined_labels.npy"

OUTPUT_DIR = r"data/coordinate/sliding_results"

# Grid Parameters
IMG_SIZE = 448
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2

CELL_SIZE = 1.2  # Distance between grid centers in meters
LIKELIHOOD_THRESHOLD = 0.75  # Cosine similarity threshold to be "checked"

MAX_STEPS = 3  # Step 1, Step 2, and Final Step 3

# ==========================================
# 2. Model Definition
# ==========================================
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=448, patch_size=16, in_chans=3, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                activation='gelu', batch_first=True
            ) for _ in range(depth)
        ])
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

# ==========================================
# 3. Helper Functions
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
        # Store label in lowercase string format for robust matching
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

def get_checked_boxes(cx, cy, src, encoder, transform, device, target_ref_embed):
    """Generate 3x3 grid, evaluate features, and return boxes that pass the likelihood threshold."""
    coords = []
    tensors = []
    
    # Generate 9 boxes
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

    # Batch process all 9 images through the encoder
    batch_tensor = torch.stack(tensors).to(device)
    with torch.no_grad():
        embeds = encoder(batch_tensor).mean(dim=1).cpu() 
        
    # Calculate cosine similarity against the reference tree crown
    target_ref_expanded = target_ref_embed.unsqueeze(0).expand(len(embeds), -1)
    similarities = F.cosine_similarity(embeds, target_ref_expanded).numpy()
    
    # Filter boxes based on likelihood rule
    checked = []
    for i, sim in enumerate(similarities):
        if sim >= LIKELIHOOD_THRESHOLD:
            checked.append(coords[i])
            
    return checked

def calculate_center(checked_cells):
    """Compute the geometric center of the checked boxes."""
    if not checked_cells:
        return None, None
    xs = [cell["x"] for cell in checked_cells]
    ys = [cell["y"] for cell in checked_cells]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# ==========================================
# 4. Main Algorithm Pipeline
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading LeJEPA Model...")
    encoder = LeJepaEncoder().to(device)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    encoder.eval()
    
    print("Loading Reference Embeddings (Ground Truth Features)...")
    ref_embeddings = load_reference_embeddings(EMBEDDING_PATH, LABEL_PATH)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading Shifted Shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    
    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    if not tif_files:
        print("No TIF files found.")
        return

    # Dictionary to store the trajectory of each point across 3 steps
    # Format: { point_id: { 1: {x, y, status}, 2: {x, y, status}, 3: {x, y, status}, 'label': label_name } }
    point_trajectories = {}

    print(f"\nStarting Feature-based Sliding Grid over {len(tif_files)} TIFs...")

    for tif_path in tqdm(tif_files, desc="Processing TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                # Fast Bounding Box filter
                bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*src.bounds)]}, crs=src.crs)
                if bbox_gdf.crs != gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(gdf.crs)
                
                contained = gdf[gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
                if contained.empty:
                    continue 
                
                contained = contained.copy()
                if contained.crs != src.crs:
                    contained = contained.to_crs(src.crs)
                
                inner_pbar = tqdm(contained.iterrows(), total=len(contained), leave=False, desc="Trees")
                
                for idx, row in inner_pbar:
                    fid = row.get('temp_id', idx)
                    if fid in point_trajectories:
                        continue  # Already processed in another overlapping TIF
                        
                    orig_x, orig_y = row.geometry.x, row.geometry.y
                    
                    # Resolve label safely
                    raw_label = row.get('Tree') or row.get('tree') or row.get('id') or f"tree_{fid}"
                    label_val = str(raw_label).strip().lower()
                    
                    point_trajectories[fid] = {"label": label_val}
                    
                    if label_val not in ref_embeddings:
                        # Label not recognized in embeddings, mark as failed for all steps
                        for step in range(1, MAX_STEPS + 1):
                            point_trajectories[fid][step] = {"x": orig_x, "y": orig_y, "status": "failed"}
                        continue
                        
                    target_ref = ref_embeddings[label_val]
                    current_x, current_y = orig_x, orig_y
                    
                    # -----------------------------------------------------
                    # ITERATIVE SLIDING ALGORITHM (Step 1 -> 2 -> 3)
                    # -----------------------------------------------------
                    for step in range(1, MAX_STEPS + 1):
                        # If previously failed or centered, just carry over the state
                        if step > 1 and point_trajectories[fid][step - 1]["status"] in ["center", "failed"]:
                            point_trajectories[fid][step] = point_trajectories[fid][step - 1].copy()
                            continue
                            
                        # Generate Grid & Evaluate Likelihood
                        checked_boxes = get_checked_boxes(current_x, current_y, src, encoder, transform, device, target_ref)
                        num_checked = len(checked_boxes)
                        
                        # Rule 1: 3 or more boxes checked -> Center
                        if num_checked >= 3:
                            cx, cy = calculate_center(checked_boxes)
                            current_x, current_y = cx, cy
                            point_trajectories[fid][step] = {"x": cx, "y": cy, "status": "center"}
                            
                        # Rule 2: 1 or 2 boxes checked -> Slide
                        elif 1 <= num_checked <= 2:
                            cx, cy = calculate_center(checked_boxes)
                            current_x, current_y = cx, cy
                            
                            if step == MAX_STEPS:
                                # Rule 3a: Last step, but still requires slide -> Mark as Failed
                                point_trajectories[fid][step] = {"x": cx, "y": cy, "status": "failed"}
                            else:
                                point_trajectories[fid][step] = {"x": cx, "y": cy, "status": "slide"}
                                
                        # Rule 3b: 0 boxes checked -> Immediate Failure
                        else:
                            point_trajectories[fid][step] = {"x": current_x, "y": current_y, "status": "failed"}

        except Exception as e:
            pass # Skip corrupted TIFs

    # ==========================================
    # 5. Export Results to CSV per Step
    # ==========================================
    print("\n\n--- Generating Output CSVs ---")
    if not point_trajectories:
        print("No points were processed. Check your shapefile projection or TIF bounds.")
        return

    # Create CSVs for Step 1, 2, and 3
    for step in range(1, MAX_STEPS + 1):
        step_records = []
        for fid, data in point_trajectories.items():
            if step in data:
                step_records.append({
                    "id": fid,
                    "label": data["label"],
                    "x": data[step]["x"],
                    "y": data[step]["y"],
                    "status": data[step]["status"]
                })
        
        step_df = pd.DataFrame(step_records)
        out_csv = os.path.join(OUTPUT_DIR, f"step{step}_results.csv")
        step_df.to_csv(out_csv, index=False)
        
        print(f"\nSaved: {out_csv}")
        print("Status Breakdown:")
        print(step_df["status"].value_counts().to_string())

    print("\nSliding Grid Algorithm completed successfully!")

if __name__ == "__main__":
    main()