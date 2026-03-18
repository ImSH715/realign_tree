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

# Directory paths
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp"

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 300
    
    # Setup output directories
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
    
    # [핵심] 원본 SHP의 모든 행에 고유 ID 부여. 같은 좌표라도 ID가 다르므로 개별 데이터로 인식됨.
    gdf['temp_id'] = range(len(gdf))

    print("Scanning TIF files and building bounding box indices...")
    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    if not tif_files:
        print("No .tif files found.")
        return

    # 1. Create GeoDataFrame for TIF bounds
    tif_records = []
    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as src:
                b = src.bounds
                tif_records.append({
                    'tif_path': tif_path,
                    'geometry': box(b.left, b.bottom, b.right, b.top),
                    'crs': src.crs
                })
        except Exception:
            continue
            
    tif_gdf = gpd.GeoDataFrame(tif_records, crs=tif_records[0]['crs'])

    if gdf.crs != tif_gdf.crs:
        gdf = gdf.to_crs(tif_gdf.crs)

    # 2. Spatial Join
    joined = gpd.sjoin(gdf, tif_gdf, how='inner', predicate='intersects')
    
    # [핵심] temp_id를 기준으로 중복 제거를 다시 활성화. 
    # 이는 '하나의 포인트가 두 개의 TIF에 속해서 두 번 뽑히는 것'을 막아줍니다.
    # 하지만 '원래 SHP 파일에서 동일한 좌표로 찍힌 중복 포인트들'은 temp_id가 서로 다르기 때문에 삭제되지 않고 모두 보존됩니다!
    joined = joined.drop_duplicates(subset='temp_id')

    patches = []
    patch_labels = []
    successful_rows = [] 

    print(f"Start optimized patch extraction for {len(joined)} potential points...")
    
    # 3. Group by TIF file
    grouped = joined.groupby('tif_path')
    
    for tif_path, group in grouped:
        try:
            with rasterio.open(tif_path) as src:
                for idx, row in group.iterrows():
                    x, y = row.geometry.x, row.geometry.y
                    py, px = src.index(x, y)
                    
                    if (px - HALF_CROP >= 0 and py - HALF_CROP >= 0 and 
                        px + HALF_CROP <= src.width and py + HALF_CROP <= src.height):
                        
                        window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
                        tile = src.read([1, 2, 3], window=window)
                        tile = np.moveaxis(tile, 0, -1)
                        
                        if tile.shape[0] == CROP_SIZE and tile.shape[1] == CROP_SIZE:
                            img_pil = Image.fromarray(tile.astype('uint8'))
                            patches.append(transform(img_pil))
                            
                            label_val = row.get('Tree') or row.get('tree') or row.get('id') or f"tree_{row['temp_id']}"
                            patch_labels.append(str(label_val))
                            
                            successful_rows.append(row)
                            
        except Exception as e:
            print(f"Error reading {tif_path}: {e}")

    if not patches:
        print("Cannot find any matching patches. Exiting.")
        return
        
    print(f"Total Successfully Extracted Patches: {len(patches)}")

    # 4. Save Validated subset to new Shapefile (labels.shp)
    valid_points_gdf = gpd.GeoDataFrame(successful_rows, crs=joined.crs)
    
    cols_to_drop = ['temp_id', 'index_right']
    valid_points_gdf = valid_points_gdf.drop(columns=[c for c in cols_to_drop if c in valid_points_gdf.columns], errors='ignore')
    
    labels_shp_path = os.path.join(label_dir, "labels.shp")
    valid_points_gdf.to_file(labels_shp_path)
    print(f"Saved filtered Original attributes to: {labels_shp_path} ({len(valid_points_gdf)} points)")

    # 5. Dataloader Setup
    all_patches_tensor = torch.stack(patches)
    dataset = TensorDataset(all_patches_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder.train()
    predictor.train()

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2 
    keep_num = int(num_patches * 0.25)
    
    # 6. Training Loop
    print("\nStarting LeJEPA Training...")
    for epoch in range(epochs):
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
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")

    torch.save(encoder.state_dict(), f"{model_dir}/lejepa_encoder.pth")
    print("\nTraining Complete. Extracting Embeddings...")
    
    # 7. Embedding Extraction
    encoder.eval()
    all_embeddings = []
    
    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch_imgs = batch[0].to(device)
            embeds = encoder(batch_imgs).mean(dim=1).cpu().numpy()
            all_embeddings.append(embeds)
            
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.array(patch_labels)
    
    # 3가지 데이터가 완벽하게 숫자가 일치하는지 검증
    assert len(final_embeddings) == len(final_labels) == len(valid_points_gdf), "Mismatch in extracted counts!"

    np.save(f"{embedding_dir}/embeddings.npy", final_embeddings)
    np.save(f"{label_dir}/labels.npy", final_labels)
    
    print(f"Successfully saved {len(final_embeddings)} embeddings to {embedding_dir}/embeddings.npy")
    print(f"Successfully saved {len(final_labels)} labels to {label_dir}/labels.npy")

if __name__ == "__main__":
    main()