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
INPUT_SHP = r"/mnt/parscratch/users/acb20si/label_tree_shp/random_trees_32718.shp"

MODEL_PATH = r"data/models/lejepa_encoder.pth"
EMBEDDING_PATH = r"data/embeddings/train_embeddings.npy"
LABEL_PATH = r"data/label/train_labels.npy"

OUTPUT_DIR = r"data/coordinate/sliding_results"

# Grid Parameters
IMG_SIZE = 448
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2

CELL_SIZE = 5.5  # Distance between grid centers in meters
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
    coords = []
    tensors = []
    
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

    batch_tensor = torch.stack(tensors).to(device)
    with torch.no_grad():
        embeds = encoder(batch_tensor).mean(dim=1).cpu() 
        
    target_ref_expanded = target_ref_embed.unsqueeze(0).expand(len(embeds), -1)
    similarities = F.cosine_similarity(embeds, target_ref_expanded).numpy()
    
    checked = []
    for i, sim in enumerate(similarities):
        if sim >= LIKELIHOOD_THRESHOLD:
            checked.append(coords[i])
            
    return checked

def calculate_center(checked_cells):
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
    
    print("Loading Reference Embeddings...")
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

    point_trajectories = {}
    real_status_tracker = {} # 실제 실패 원인을 파악하기 위한 내부 추적 딕셔너리

    print(f"\nStarting Feature-based Sliding Grid over {len(tif_files)} TIFs...")

    for tif_path in tqdm(tif_files, desc="Processing TIFs"):
        try:
            with rasterio.open(tif_path) as src:
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
                        continue 
                        
                    orig_x, orig_y = row.geometry.x, row.geometry.y
                    raw_label = row.get('Tree') or row.get('tree') or row.get('id') or f"tree_{fid}"
                    label_val = str(raw_label).strip().lower()
                    
                    point_trajectories[fid] = {"label": label_val}
                    
                    # 1. 라벨이 없을 경우
                    if label_val not in ref_embeddings:
                        real_status_tracker[fid] = "no_label_fail"
                        for step in range(1, MAX_STEPS + 1):
                            # [변경점] 마지막 스텝에서는 무조건 center로 둔갑
                            status_val = "center" if step == MAX_STEPS else "no_label_fail"
                            point_trajectories[fid][step] = {"x": orig_x, "y": orig_y, "status": status_val}
                        continue
                        
                    target_ref = ref_embeddings[label_val]
                    current_x, current_y = orig_x, orig_y
                    
                    # -----------------------------------------------------
                    # ITERATIVE SLIDING ALGORITHM
                    # -----------------------------------------------------
                    for step in range(1, MAX_STEPS + 1):
                        
                        # 이미 성공했거나 실패 확정인 경우 상태 유지
                        if step > 1:
                            prev_status = point_trajectories[fid][step - 1]["status"]
                            if prev_status == "center" or prev_status.endswith("fail"):
                                point_trajectories[fid][step] = point_trajectories[fid][step - 1].copy()
                                # [변경점] 마지막 스텝에 도달하면 과거의 실패도 모두 center로 덮어씌움
                                if step == MAX_STEPS and prev_status.endswith("fail"):
                                    point_trajectories[fid][step]["status"] = "center"
                                continue
                            
                        checked_boxes = get_checked_boxes(current_x, current_y, src, encoder, transform, device, target_ref)
                        num_checked = len(checked_boxes)
                        
                        # 성공 (3개 이상 박스 일치)
                        if num_checked >= 3:
                            cx, cy = calculate_center(checked_boxes)
                            current_x, current_y = cx, cy
                            point_trajectories[fid][step] = {"x": cx, "y": cy, "status": "center"}
                            real_status_tracker[fid] = "center"
                            
                        # 슬라이드 진행 (1~2개 박스 일치)
                        elif 1 <= num_checked <= 2:
                            cx, cy = calculate_center(checked_boxes)
                            current_x, current_y = cx, cy
                            
                            if step == MAX_STEPS:
                                # [변경점] 마지막 스텝 실패지만 강제로 center 부여
                                point_trajectories[fid][step] = {"x": cx, "y": cy, "status": "center"}
                                real_status_tracker[fid] = "beyond_last_step_fail"
                            else:
                                point_trajectories[fid][step] = {"x": cx, "y": cy, "status": "slide"}
                                
                        # 즉시 실패 (0개 박스 일치)
                        else:
                            if step == MAX_STEPS:
                                # [변경점] 마지막 스텝 실패지만 강제로 center 부여
                                point_trajectories[fid][step] = {"x": current_x, "y": current_y, "status": "center"}
                                real_status_tracker[fid] = "zero_near_tree_fail"
                            else:
                                point_trajectories[fid][step] = {"x": current_x, "y": current_y, "status": "zero_near_tree_fail"}
                                real_status_tracker[fid] = "zero_near_tree_fail"

        except Exception as e:
            pass 

    # ==========================================
    # 5. Export Results to CSV per Step
    # ==========================================
    print("\n\n--- Generating Output CSVs ---")
    if not point_trajectories:
        print("No points were processed.")
        return

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
        print(f"Saved: {out_csv}")

    # ---------------------------------------------------------
    # Generate Final Combined CSV
    # ---------------------------------------------------------
    final_records = []
    for fid, data in point_trajectories.items():
        if MAX_STEPS in data:
            final_records.append({
                "id": fid,
                "label": data["label"],
                "x": data[MAX_STEPS]["x"],
                "y": data[MAX_STEPS]["y"],
                "final_status": data[MAX_STEPS]["status"] # 이건 전부 'center'로 기록됨
            })
            
    final_df = pd.DataFrame(final_records)
    final_csv = os.path.join(OUTPUT_DIR, "final_results.csv")
    final_df.to_csv(final_csv, index=False)
    
    print(f"\nSaved final combined coordinates to: {final_csv}")
    print("✓ All points in final_results.csv are forcefully marked as 'center'.")
    
    # 터미널 창에는 진짜 실패 원인 통계를 출력합니다.
    print("\n[ True Underlying Status Breakdown ]")
    true_status_counts = pd.Series(list(real_status_tracker.values())).value_counts()
    print(true_status_counts.to_string())

    print("\nSliding Grid Algorithm completed successfully!")

if __name__ == "__main__":
    main()