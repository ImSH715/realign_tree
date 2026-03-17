import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
import rasterio
import geopandas as gpd
import glob                                 
from shapely.geometry import box          
from torch.utils.data import TensorDataset, DataLoader 
from tqdm import tqdm
import time

# --------------------------
# Hyperparameters & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp"

IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2
BATCH_SIZE = 16 
EPOCHS = 300

# --------------------------
# Model Definitions
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

class LeJepaPredictor(nn.Module):
    def __init__(self, embed_dim=128, predictor_depth=2, num_heads=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, 
            activation='gelu', batch_first=True
        )
        self.blocks = nn.ModuleList([predictor_layer for _ in range(predictor_depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_embeds, mask_pos_embeds):
        B = context_embeds.shape[0]
        N_mask = mask_pos_embeds.shape[1]
        mask_tokens = self.mask_token.repeat(B, N_mask, 1) + mask_pos_embeds 
        x = torch.cat([context_embeds, mask_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, -N_mask:, :]

# --------------------------
# Main Execution
# --------------------------
def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 0. Setup output directories
    model_dir, embedding_dir, label_dir = "data/models", "data/embeddings", "data/label"
    for d in [model_dir, embedding_dir, label_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Initialize Model & Tools
    encoder = LeJepaEncoder(img_size=IMG_SIZE).to(device)
    predictor = LeJepaPredictor().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Shapefile & Prep IDs
    print("Loading Original Shapefile...")
    gdf = gpd.read_file(ANNOTATED_COR)
    original_crs = gdf.crs
    if 'temp_id' not in gdf.columns:
        gdf['temp_id'] = range(len(gdf))

    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    if not tif_files:
        print("No .tif files found.")
        return

    patches = []
    patch_labels = []
    successful_rows = [] 
    extracted_temp_ids = set() 

    # 3. Optimized Extraction Mechanism (1409개 추출 로직)
    print(f"\nStarting extraction for {len(tif_files)} TIF files...")
    
    for tif_path in tqdm(tif_files, desc="Processing TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                # 개별 TIF의 CRS에 맞게 변환하여 오차 최소화
                current_gdf = gdf.copy()
                if current_gdf.crs != src.crs:
                    current_gdf = current_gdf.to_crs(src.crs)
                
                # 공간 인덱스를 활용한 필터링
                b = src.bounds
                img_box = box(b.left, b.bottom, b.right, b.top)
                contained = current_gdf[current_gdf.geometry.intersects(img_box)]
                
                for idx, row in contained.iterrows():
                    temp_id = row['temp_id']
                    
                    # 이미 다른 TIF에서 추출된 나무는 건너뜀 (중복 방지)
                    if temp_id in extracted_temp_ids:
                        continue
                        
                    x, y = row.geometry.x, row.geometry.y
                    py, px = src.index(x, y)
                    
                    # Boundless=True를 사용하여 가장자리 나무들도 유실 없이 추출
                    window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
                    tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                    tile = np.moveaxis(tile, 0, -1)
                    
                    if tile.shape[0] == CROP_SIZE and tile.shape[1] == CROP_SIZE:
                        img_pil = Image.fromarray(tile.astype('uint8'))
                        patches.append(transform(img_pil))
                        
                        label_val = row.get('Tree') or row.get('tree') or row.get('id') or f"tree_{temp_id}"
                        patch_labels.append(str(label_val))
                        
                        # 원본 CRS 속성을 유지하며 결과 리스트에 추가
                        successful_rows.append(gdf.loc[idx])
                        extracted_temp_ids.add(temp_id)
                            
        except Exception as e:
            tqdm.write(f"Error reading {tif_path}: {e}")

    if not patches:
        print("No patches extracted. Check file paths and CRS."); return
        
    print(f"\nTotal Successfully Extracted Patches: {len(patches)}")

    # 4. Save validated subset to Shapefile
    valid_points_gdf = gpd.GeoDataFrame(successful_rows, crs=original_crs)
    labels_shp_path = os.path.join(label_dir, "labels.shp")
    valid_points_gdf.drop(columns=['temp_id'], errors='ignore').to_file(labels_shp_path)
    print(f"Saved filtered attributes to: {labels_shp_path}")

    # 5. Dataloader Setup
    all_patches_tensor = torch.stack(patches)
    dataset = TensorDataset(all_patches_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 6. LeJEPA Training Loop
    encoder.train()
    predictor.train()
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2 
    keep_num = int(num_patches * 0.25)
    
    print("\nStarting LeJEPA Training...")
    epoch_pbar = tqdm(range(EPOCHS), desc="Training Model")
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        for batch in dataloader:
            batch_imgs = batch[0].to(device)
            curr_batch_size = batch_imgs.shape[0]
            
            optimizer.zero_grad()
            with torch.no_grad():
                target = encoder(batch_imgs).detach()

            ids_shuffle = torch.argsort(torch.rand(curr_batch_size, num_patches, device=device), dim=1)
            ids_keep = ids_shuffle[:, :keep_num] 
            ids_mask = ids_shuffle[:, keep_num:] 

            context_embeds = encoder(batch_imgs, ids_keep=ids_keep)
            mask_pos_embeds = encoder.pos_embed.expand(curr_batch_size, -1, -1)
            mask_pos_embeds = torch.gather(mask_pos_embeds, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, 128))
            
            pred_features = predictor(context_embeds, mask_pos_embeds)
            actual_target_features = torch.gather(target, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, 128))
            
            loss = criterion(pred_features, actual_target_features)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        epoch_pbar.set_postfix(Loss=f"{avg_loss:.6f}")

    torch.save(encoder.state_dict(), f"{model_dir}/lejepa_encoder.pth")
    print("\nTraining Complete. Extracting Final Embeddings...")
    
    # 7. Embedding Extraction
    encoder.eval()
    all_embeddings = []
    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Extracting Embeddings"):
            batch_imgs = batch[0].to(device)
            embeds = encoder(batch_imgs).mean(dim=1).cpu().numpy()
            all_embeddings.append(embeds)
            
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.array(patch_labels)
    
    # Final Validation
    assert len(final_embeddings) == len(final_labels) == len(valid_points_gdf), "Count mismatch!"

    np.save(f"{embedding_dir}/embeddings.npy", final_embeddings)
    np.save(f"{label_dir}/labels.npy", final_labels)
    
    total_time = (time.time() - start_time) / 60
    print(f"\nSuccess! Extracted {len(final_embeddings)} samples.")
    print(f"Total Execution Time: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()