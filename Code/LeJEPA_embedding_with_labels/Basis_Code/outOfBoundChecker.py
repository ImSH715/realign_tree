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

# Directory paths
BASE_DIR = r"Z:\ai4eo\Shared\2025_Forge\OSINFOR_data\01. Ortomosaicos\2023"
ANNOTATED_COR = r"Z:\ai4eo\Shared\2025_Turing_L\Project\Annotated tree centroids\trees_32718.shp"

# Hyperparameters
IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2
BATCH_SIZE = 16 

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
    epochs = 300
    
    model_dir = "data/models"
    embedding_dir = "data/embeddings"
    label_dir = "data/label"
    for d in [model_dir, embedding_dir, label_dir]:
        os.makedirs(d, exist_ok=True)

    encoder = LeJepaEncoder(img_size=IMG_SIZE).to(device)
    predictor = LeJepaPredictor().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    print(f"\nStarting Bulletproof Patch Extraction from {len(tif_files)} TIF files...")
    
    # 1. Image extraction
    for tif_path in tqdm(tif_files, desc="Processing TIFs", unit="file"):
        try:
            with rasterio.open(tif_path) as src:
                current_gdf = gdf.copy()
                if current_gdf.crs != src.crs:
                    current_gdf = current_gdf.to_crs(src.crs)
                
                b = src.bounds
                img_box = box(b.left, b.bottom, b.right, b.top)
                contained = current_gdf[current_gdf.geometry.intersects(img_box)]
                
                # 채널 개수 확인 (이미지가 1채널 흑백일 경우를 대비해 예외 방지)
                bands_to_read = [1, 2, 3] if src.count >= 3 else [1] * 3 
                
                # 각 점(Point)마다 개별적으로 Try-Except 적용 (하나 에러나도 다른 점은 무사히 통과)
                for idx, row in contained.iterrows():
                    temp_id = row['temp_id']
                    
                    if temp_id in extracted_temp_ids:
                        continue
                        
                    try:
                        x, y = row.geometry.x, row.geometry.y
                        py, px = src.index(x, y)
                        
                        window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
                        tile = src.read(bands_to_read, window=window, boundless=True, fill_value=0)
                        tile = np.moveaxis(tile, 0, -1)
                        
                        # 타일 크기가 예상과 다를 경우 강제로 패딩하여 살림 (가장자리 점 방어)
                        if tile.shape[:2] != (CROP_SIZE, CROP_SIZE):
                            h, w, c = tile.shape
                            pad_h = max(0, CROP_SIZE - h)
                            pad_w = max(0, CROP_SIZE - w)
                            tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                            tile = tile[:CROP_SIZE, :CROP_SIZE, :] # 혹시라도 더 커졌을 경우 잘라냄
                            
                        img_pil = Image.fromarray(tile.astype('uint8'))
                        patches.append(transform(img_pil))
                        
                        label_val = row.get('Tree') or row.get('tree') or row.get('id') or f"tree_{temp_id}"
                        patch_labels.append(str(label_val))
                        
                        successful_rows.append(gdf.loc[idx])
                        extracted_temp_ids.add(temp_id)
                        
                    except Exception as e_point:
                        # 점 단위 에러 출력 (전체 TIF를 중단시키지 않음)
                        # tqdm.write(f"Point {temp_id} skipped: {e_point}")
                        pass
                        
        except Exception as e_file:
            tqdm.write(f"Error opening/processing {os.path.basename(tif_path)}: {e_file}")

    if not patches:
        print("Cannot find any matching patches. Exiting.")
        return
        
    print(f"\n==============================================")
    print(f"✅ Total Successfully Extracted Patches: {len(patches)}")
    print(f"==============================================\n")

    # 2. Save validated subset
    valid_points_gdf = gpd.GeoDataFrame(successful_rows, crs=original_crs)
    cols_to_drop = ['temp_id']
    valid_points_gdf = valid_points_gdf.drop(columns=[c for c in cols_to_drop if c in valid_points_gdf.columns], errors='ignore')
    
    labels_shp_path = os.path.join(label_dir, "labels.shp")
    valid_points_gdf.to_file(labels_shp_path)
    print(f"Saved filtered attributes to: {labels_shp_path}")

    # 3. Dataloader Setup
    all_patches_tensor = torch.stack(patches)
    dataset = TensorDataset(all_patches_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder.train()
    predictor.train()

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2 
    keep_num = int(num_patches * 0.25)
    
    # 4. Training Loop
    print("\nStarting LeJEPA Training...")
    epoch_pbar = tqdm(range(epochs), desc="Training Model", unit="epoch")
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        for batch in dataloader:
            batch_imgs = batch[0].to(device)
            current_batch_size = batch_imgs.shape[0]
            
            optimizer.zero_grad()
            with torch.no_grad():
                target = encoder(batch_imgs).detach()

            ids_shuffle = torch.argsort(torch.rand(current_batch_size, num_patches, device=device), dim=1)
            ids_keep = ids_shuffle[:, :keep_num] 
            ids_mask = ids_shuffle[:, keep_num:] 

            context_embeds = encoder(batch_imgs, ids_keep=ids_keep)
            mask_pos_embeds = encoder.pos_embed.expand(current_batch_size, -1, -1)
            mask_pos_embeds = torch.gather(mask_pos_embeds, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, 128))
            
            pred_features = predictor(context_embeds, mask_pos_embeds)
            actual_target_features = torch.gather(target, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, 128))
            
            loss = criterion(pred_features, actual_target_features)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        epoch_pbar.set_postfix(Loss=f"{avg_loss:.4f}")

    torch.save(encoder.state_dict(), f"{model_dir}/lejepa_encoder.pth")
    print("\nTraining Complete.")
    
    # 5. Embedding Extraction
    encoder.eval()
    all_embeddings = []
    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Extracting Embeddings", unit="batch"):
            batch_imgs = batch[0].to(device)
            embeds = encoder(batch_imgs).mean(dim=1).cpu().numpy()
            all_embeddings.append(embeds)
            
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.array(patch_labels)
    
    assert len(final_embeddings) == len(final_labels) == len(valid_points_gdf), "Mismatch in extracted counts!"

    np.save(f"{embedding_dir}/embeddings.npy", final_embeddings)
    np.save(f"{label_dir}/labels.npy", final_labels)
    
    print(f"\nSuccessfully saved {len(final_embeddings)} embeddings and labels.")
    
    total_time = time.time() - start_time
    print(f"Total Execution Time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()