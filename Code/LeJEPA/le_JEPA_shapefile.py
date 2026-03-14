"""
Required packages: pip install torch torchvision numpy matplotlib scikit-learn rasterio geopandas shapely scikit-image
Run: python lejepa_tsne_visualization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image
import torchvision.transforms as transforms
import os
import rasterio
import geopandas as gpd
from shapely.geometry import box
import skimage.segmentation as seg

# Settings for loading large satellite images (TIFF)
Image.MAX_IMAGE_PIXELS = None
path_data = "data/shapefile"
# ==========================================
# 1. Encoder and Predictor Definition (Le-JEPA Architecture)
# ==========================================
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=448, patch_size=16, in_chans=3, embed_dim=128, depth=4, num_heads=4):
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
            B, L, D = x.shape
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
        return x[:, -N_mask:, :]

# ==========================================
# 2. Training Utilities
# ==========================================
def apply_masking(img_tensor, patch_size=16, mask_ratio=0.70):
    B, C, H, W = img_tensor.shape
    num_patches = (H // patch_size) * (W // patch_size)
    num_keep = int(num_patches * (1 - mask_ratio))
    
    noise = torch.rand(B, num_patches, device=img_tensor.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    
    ids_keep = ids_shuffle[:, :num_keep]
    ids_mask = ids_shuffle[:, num_keep:]
    return ids_keep, ids_mask

def update_target_encoder_ema(context_encoder, target_encoder, momentum=0.99):
    with torch.no_grad():
        for param_c, param_t in zip(context_encoder.parameters(), target_encoder.parameters()):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_c.detach().data)

def load_and_crop_image(image_path, img_size=448, crop_x=0, crop_y=0, crop_size=1024):
    if os.path.exists(image_path):
        print(f"Loading image: {image_path}")
        img = Image.open(image_path).convert('RGB')
        right, bottom = crop_x + crop_size, crop_y + crop_size
        img = img.crop((crop_x, crop_y, right, bottom))
    else:
        print(f"Warning: File not found. Generating random noise pattern.")
        img_array = np.random.rand(crop_size, crop_size, 3) * 255
        img = Image.fromarray(img_array.astype(np.uint8))
        
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0), img

# ==========================================
# 3. Unified Geospatial Processing (Tiles -> Polygons -> Points)
# ==========================================
def process_geospatial_data(image_path, crop_x, crop_y, crop_size, rgb_map_small, output_dir, scale=1.0, min_size=2):
    if not os.path.exists(image_path):
        return

    print("\n--- Processing Geospatial Data ---")
    try:
        with rasterio.open(image_path) as src:
            transform = src.transform # Check: No Korean characters here!
            crs = src.crs

        h_p, w_p, _ = rgb_map_small.shape
        patch_pixel_w = crop_size / w_p
        patch_pixel_h = crop_size / h_p

        # 1. Image Segmentation (Groups adjacent similar colors)
        print("1. Grouping patches based on color similarity...")
        labels = seg.felzenszwalb(rgb_map_small, scale=scale, sigma=0.2, min_size=min_size)

        polygons = []
        records = []
        
        for row in range(h_p):
            for col in range(w_p):
                px_min = crop_x + col * patch_pixel_w
                py_min = crop_y + row * patch_pixel_h
                px_max = px_min + patch_pixel_w
                py_max = py_min + patch_pixel_h
                
                geo_x_min, geo_y_max = transform * (px_min, py_min)
                geo_x_max, geo_y_min = transform * (px_max, py_max)
                
                geom = box(geo_x_min, geo_y_min, geo_x_max, geo_y_max)
                polygons.append(geom)
                
                r, g, b = rgb_map_small[row, col]
                hex_color = '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
                
                records.append({
                    'tree_id': int(labels[row, col]), # Assign the grouped ID
                    'color': hex_color
                })
                
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        gdf_grid = gpd.GeoDataFrame(records, geometry=polygons, crs=crs)
        
        # [Step A] Save Individual Tiles (With Group ID)
        tiles_path = os.path.join(output_dir, "lejepa_tiles.shp")
        gdf_grid.to_file(tiles_path)
        print(f"-> Saved [1/3]: Individual Grid Tiles to '{tiles_path}'")
        
        # [Step B] Dissolve Tiles into Grouped Polygons
        print("2. Merging tiles into large tree polygons...")
        # This physically merges small boxes sharing the same 'tree_id' into larger polygons
        gdf_trees = gdf_grid.dissolve(by='tree_id', aggfunc='first').reset_index()
        trees_path = os.path.join(output_dir, "lejepa_tree_polygons.shp")
        gdf_trees.to_file(trees_path)
        print(f"-> Saved [2/3]: Merged Tree Polygons to '{trees_path}'")
        
        # [Step C] Extract Centroids from the merged Polygons
        print("3. Extracting centroids from merged polygons...")
        gdf_centers = gdf_trees.copy()
        gdf_centers['geometry'] = gdf_centers['geometry'].centroid
        centers_path = os.path.join(output_dir, "lejepa_tree_centers.shp")
        gdf_centers.to_file(centers_path)
        print(f"-> Saved [3/3]: Tree Centroids to '{centers_path}'")
        print(f"   (Total merged tree areas detected: {len(gdf_trees)})")

        # Generate QGIS styles for easy viewing
        _generate_qml_styles(output_dir)

    except Exception as e:
        print(f"Geospatial processing failed. Error: {e}")

def _generate_qml_styles(output_dir):
    # Style for Tree Polygons (Yellow thick outline, transparent inside)
    poly_qml = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0.0-Pi" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" forceraster="0" enableorderby="0" symbollevels="0">
    <symbols><symbol type="fill" clip_to_extent="1" force_rhr="0" name="0" alpha="1">
      <layer pass="0" locked="0" enabled="1" class="SimpleLine">
        <prop k="line_color" v="255,255,0,255"/><prop k="line_width" v="0.6"/>
      </layer>
    </symbol></symbols>
  </renderer-v2>
</qgis>"""
    with open(os.path.join(output_dir, 'lejepa_tree_polygons.qml'), 'w', encoding='utf-8') as f: f.write(poly_qml)

    # Style for Tree Centers (Red dot)
    point_qml = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0.0-Pi" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" forceraster="0" enableorderby="0" symbollevels="0">
    <symbols><symbol type="marker" clip_to_extent="1" force_rhr="0" name="0" alpha="1">
      <layer pass="0" locked="0" enabled="1" class="SimpleMarker">
        <prop k="color" v="255,0,0,255"/><prop k="outline_color" v="255,255,255,255"/><prop k="outline_width" v="0.4"/><prop k="size" v="3.5"/>
      </layer>
    </symbol></symbols>
  </renderer-v2>
</qgis>"""
    with open(os.path.join(output_dir, 'lejepa_tree_centers.qml'), 'w', encoding='utf-8') as f: f.write(point_qml)


def save_rgb_map_to_geotiff(image_path, crop_x, crop_y, crop_size, rgb_map_tensor, output_filename):
    if not os.path.exists(image_path): return
    try:
        out_dir = os.path.dirname(output_filename)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
            
        with rasterio.open(image_path) as src:
            crs = src.crs
            window = rasterio.windows.Window(crop_x, crop_y, crop_size, crop_size)
            win_transform = src.window_transform(window)
            rgb_map_full_res = F.interpolate(rgb_map_tensor, size=(crop_size, crop_size), mode='nearest')
            out_img = np.clip(rgb_map_full_res[0].cpu().numpy(), 0, 1)
            out_img = (out_img * 255).astype(np.uint8)
            
            profile = src.profile
            profile.update({
                'driver': 'GTiff', 'height': crop_size, 'width': crop_size,
                'count': 3, 'dtype': 'uint8', 'transform': win_transform,
                'crs': crs, 'nodata': None, 'compress': 'deflate'
            })
            with rasterio.open(output_filename, 'w', **profile) as dst:
                dst.write(out_img)
    except Exception as e:
        print(f"Failed to generate GeoTIFF. Error: {e}")

# ==========================================
# 4. Main Analysis
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========================================================
    IMAGE_PATH = r"G:\.shortcut-targets-by-id\1IWblie-cf89tMuc4dQ8umWTHw7XOdl0p\PROYECTO FORGE\01. Ortomosaicos\2023\2023-01\25-PUC-C-DE-CPC-002-12_18032023_001_idw_transparent_mosaic_group1.tif"
    
    CROP_X = 2000       
    CROP_Y = 2000       
    CROP_SIZE = 2048    
    
    DO_TRAINING = True
    TRAIN_EPOCHS = 6000
    
    IMG_SIZE = 448      
    PATCH_SIZE = 16
    EMBED_DIM = 128
    
    # [NEW] Segmentation Settings (Adjust to merge more or fewer boxes)
    SEG_SCALE = 1.0  
    SEG_MIN_SIZE = 2 
    # ========================================================
    
    img_tensor, _ = load_and_crop_image(IMAGE_PATH, img_size=IMG_SIZE, crop_x=CROP_X, crop_y=CROP_Y, crop_size=CROP_SIZE)
    img_tensor = img_tensor.to(device)
    
    target_encoder = LeJepaEncoder(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM).to(device)
    
    if DO_TRAINING:
        print(f"\n--- Starting Le-JEPA training ({TRAIN_EPOCHS} Epochs) ---")
        context_encoder = LeJepaEncoder(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM).to(device)
        predictor = LeJepaPredictor(embed_dim=EMBED_DIM).to(device)
        
        target_encoder.load_state_dict(context_encoder.state_dict())
        optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()), lr=1e-3)
        
        context_encoder.train()
        predictor.train()
        
        for epoch in range(TRAIN_EPOCHS):
            ids_keep, ids_mask = apply_masking(img_tensor, patch_size=PATCH_SIZE, mask_ratio=0.7)
            B, N_mask = img_tensor.shape[0], ids_mask.shape[1]
            
            with torch.no_grad():
                full_targets = target_encoder(img_tensor)
                D = full_targets.shape[2]
                targets = torch.gather(full_targets, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
                
            contexts = context_encoder(img_tensor, ids_keep=ids_keep)
            pos_emb = context_encoder.pos_embed.expand(B, -1, -1)
            mask_pos = torch.gather(pos_emb, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
            
            preds = predictor(contexts, mask_pos)
            loss = F.mse_loss(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            update_target_encoder_ema(context_encoder, target_encoder, momentum=0.99)
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{TRAIN_EPOCHS}] Loss: {loss.item():.4f}")
    
    target_encoder.eval()
    with torch.no_grad():
        features = target_encoder(img_tensor)
        
    B, L, D = features.shape
    H_p = W_p = int(np.sqrt(L))
    
    features_spatial = features.transpose(1, 2).reshape(1, D, H_p, W_p)
    features_padded = F.pad(features_spatial, pad=(1, 1, 1, 1), mode='replicate')
    pooled_features = F.avg_pool2d(features_padded, kernel_size=3, stride=1, padding=0)
    
    flat_features = pooled_features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    flat_features_2d = flat_features.reshape(-1, D)
    
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(flat_features_2d)
    
    rgb_features = np.zeros_like(pca_features)
    for i in range(3):
        pc = pca_features[:, i]
        min_val, max_val = np.percentile(pc, 2), np.percentile(pc, 98)
        rgb_features[:, i] = np.clip((pc - min_val) / (max_val - min_val + 1e-8), 0, 1)
        
    rgb_map_small = rgb_features.reshape(H_p, W_p, 3) 
    rgb_tensor = torch.tensor(rgb_map_small).permute(2, 0, 1).unsqueeze(0).float()
    
    # ---------------------------------------------------------
    # Generate all geospatial outputs here
    # ---------------------------------------------------------
    process_geospatial_data(
        IMAGE_PATH, CROP_X, CROP_Y, CROP_SIZE, rgb_map_small,
        output_dir=path_data,
        scale=SEG_SCALE, min_size=SEG_MIN_SIZE
    )
    
    save_rgb_map_to_geotiff(
        IMAGE_PATH, CROP_X, CROP_Y, CROP_SIZE, rgb_tensor,
        output_filename=f"{path_data}/lejepa_rgb_map.tif"
    )
    print("\n[Process Complete!]")

if __name__ == "__main__":
    main()