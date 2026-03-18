import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import torchvision.transforms as transforms
import os
import rasterio
import geopandas as gpd
import glob                                 
from shapely.geometry import box          
from torch.utils.data import TensorDataset, DataLoader 

# To access with the dataset folder
BASE_DIR = r"\mnt\parscratch\users\aca21jo\2025_Forge\OSINFOR_data\01. Ortomosaicos\2023"
ANNOTATED_COR = r"\shared\ai4eo\Shared\2025_Turing_L\Project\Annotated tree centroids\trees_32718.shp"

IMG_SIZE = 448
PATCH_SIZE = 16

CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2
BATCH_SIZE = 16 # Control based on the VRAM

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
        x = self.norm(x)
        return x[:, -N_mask:, :]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    epochs = 300
    model_dir = "data/models"
    embedding_dir = "data/embeddings"
    label_dir = "data/label"
    plot_dir = "data/plots"
    
    for d in [model_dir, embedding_dir, label_dir, plot_dir]:
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

    print("Loading shapefile...")
    gdf = gpd.read_file(ANNOTATED_COR)
    
    patches = []
    patch_labels = []

    # 1. Get the folder 2023 and get tif file

    search_pattern = os.path.join(BASE_DIR, "2023-*", "*.tif") # for the under branch specification.
    #search_pattern = os.path.join(BASE_DIR, "*.tif")
    tif_files = glob.glob(search_pattern)
    
    if not tif_files:
        print(f"Cannot find any .tif files in {BASE_DIR}")
        return
        
    print(f"Found {len(tif_files)} .tif files. Starting extraction...")

    # 2. find tif files and patch them
    for tif_path in tif_files:
        print(f"Processing: {os.path.basename(tif_path)}")
        try:
            with rasterio.open(tif_path) as src:
                # find boundary
                bounds = src.bounds
                tif_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                
                # Filter the coordinates
                points_in_tif = gdf[gdf.geometry.intersects(tif_bbox)]
                
                if points_in_tif.empty:
                    continue
                    
                for idx, row in points_in_tif.iterrows():
                    geom = row.geometry
                    x, y = geom.x, geom.y
                    
                    py, px = src.index(x, y)
                    
                    # handle coordinate over the image
                    if (px - HALF_CROP >= 0 and py - HALF_CROP >= 0 and 
                        px + HALF_CROP <= src.width and py + HALF_CROP <= src.height):
                        
                        window = rasterio.windows.Window(
                            px - HALF_CROP, 
                            py - HALF_CROP, 
                            CROP_SIZE, CROP_SIZE
                        )
                        
                        tile = src.read([1, 2, 3], window=window)
                        tile = np.moveaxis(tile, 0, -1)
                        
                        if tile.shape[0] == CROP_SIZE and tile.shape[1] == CROP_SIZE:
                            img_pil = Image.fromarray(tile.astype('uint8'))
                            patches.append(transform(img_pil)) 
                            
                            if 'Tree' in row:
                                label_val = row['Tree']
                            elif 'tree' in row:
                                label_val = row['tree']
                            elif 'id' in row:
                                label_val = row['id']
                            else:
                                label_val = f"tree_{idx}"
                                
                            patch_labels.append(str(label_val))
        except Exception as e:
            print(f"Error reading {tif_path}: {e}")

    if not patches:
        print("Cannot find any matching patches across all images.")
        return
        
    print(f"Total Collected Patches: {len(patch_labels)}")

    # 3. Apply dataloader
    all_patches_tensor = torch.stack(patches)
    dataset = TensorDataset(all_patches_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10,        
        pin_memory=True       
    )

    encoder.train()
    predictor.train()

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2  # 784 patches
    keep_num = int(num_patches * 0.25)
    
    # 4. Change training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in dataloader:
            batch_imgs = batch[0].to(device)
            current_batch_size = batch_imgs.shape[0]
            
            optimizer.zero_grad()

            with torch.no_grad():
                all_features = encoder(batch_imgs)
                target = all_features.detach()

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
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")

    print("\n[Fast t-SNE Visualisation]")
    encoder.eval()
    with torch.no_grad():
        # Get first patch and apply visulaisation
        test_img = dataset[0][0].unsqueeze(0).to(device)
        features = encoder(test_img) 
        
        L = features.shape[1]
        H_p = W_p = int(np.sqrt(L))
        
        flat_features = features.squeeze(0).cpu().numpy()
        
        tsne = TSNE(
            n_components=3, 
            perplexity=min(30, L-1), 
            n_iter=500, 
            init='pca', 
            learning_rate='auto', 
            random_state=42
        )
        tsne_results = tsne.fit_transform(flat_features)
        
        t_min, t_max = tsne_results.min(axis=0), tsne_results.max(axis=0)
        rgb_values = (tsne_results - t_min) / (t_max - t_min + 1e-8)
        
        rgb_map_small = rgb_values.reshape(H_p, W_p, 3)
        rgb_tensor = torch.tensor(rgb_map_small).permute(2, 0, 1).unsqueeze(0).float()
        rgb_map_large = F.interpolate(rgb_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bicubic', align_corners=True)
        rgb_map_final = np.clip(rgb_map_large[0].permute(1, 2, 0).numpy(), 0, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(f"Wide Context t-SNE RGB Map (Epoch {epochs})", fontsize=14)
        
        ax1.imshow(test_img[0].cpu().permute(1, 2, 0).numpy() * 0.2 + 0.5)
        ax1.set_title("Input Image (Wide View)"); ax1.axis('off')
        
        ax2.imshow(rgb_map_final)
        ax2.set_title("t-SNE RGB Map"); ax2.axis('off')
        
        plt.tight_layout()
        save_path = f"{plot_dir}/tsne_rgb_map_wide.png"
        plt.savefig(save_path, dpi=150)
        print(f"Visualisation saved: {save_path}")

    torch.save(encoder.state_dict(), f"{model_dir}/lejepa_encoder.pth")
    
    # Use dataloader while embedding extraction.
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch_imgs = batch[0].to(device)
            embeds = encoder(batch_imgs).mean(dim=1).cpu().numpy()
            all_embeddings.append(embeds)
            
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(f"{embedding_dir}/embeddings.npy", final_embeddings)
    
    with open(f"{label_dir}/labels.txt", "w") as f:
        for lbl in patch_labels:
            f.write(f"{lbl}\n")

    print(f"Saved {len(final_embeddings)} embeddings and labels.")

if __name__ == "__main__":
    main()