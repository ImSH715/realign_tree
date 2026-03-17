import torch
import torch.nn as nn
import numpy as np
import os
import rasterio
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from shapely.geometry import box
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

# Configuration and paths reflecting 2023/2023-N directory structure
BASE_DIR = r"Z:\ai4eo\Shared\2025_Forge\OSINFOR_data\01. Ortomosaicos\2023"
ANNOTATED_COR = r"Z:\ai4eo\Shared\2025_Turing_L\Project\Annotated tree centroids\trees_32718.shp"

IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2
BATCH_SIZE = 16 

# LeJEPA model definition
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True)
        self.blocks = nn.ModuleList([encoder_layer for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        if ids_keep is not None:
            B, L, D = x.shape
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        for block in self.blocks: x = block(x)
        return self.norm(x)

class LeJepaPredictor(nn.Module):
    def __init__(self, embed_dim=128, predictor_depth=2, num_heads=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        predictor_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True)
        self.blocks = nn.ModuleList([predictor_layer for _ in range(predictor_depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_embeds, mask_pos_embeds):
        B = context_embeds.shape[0]
        N_mask = mask_pos_embeds.shape[1]
        mask_tokens = self.mask_token.repeat(B, N_mask, 1) + mask_pos_embeds 
        x = torch.cat([context_embeds, mask_tokens], dim=1)
        for block in self.blocks: x = block(x)
        x = self.norm(x)
        return x[:, -N_mask:, :]

# Extract patches and map them to their exact Shapefile index
def extract_patches_from_single_tif(tif_path, gdf_indexed):
    results_dict = {}
    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            
            # Core speed optimization: Filter points within the image using R-tree Spatial Index
            possible_matches_index = gdf_indexed.sindex.query(box(*bounds), predicate="intersects")
            points_in_tif = gdf_indexed.iloc[possible_matches_index]
            
            if points_in_tif.empty:
                return {}
            
            for idx, row in points_in_tif.iterrows():
                py, px = src.index(row.geometry.x, row.geometry.y)
                
                # Check if coordinates are inside the image safely
                if (px - HALF_CROP >= 0 and py - HALF_CROP >= 0 and px + HALF_CROP <= src.width and py + HALF_CROP <= src.height):
                    window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
                    tile = src.read([1, 2, 3], window=window)
                    tile = np.moveaxis(tile, 0, -1)
                    
                    if tile.shape[0] == CROP_SIZE and tile.shape[1] == CROP_SIZE:
                        # Extract label safely
                        label = row.get('Tree', row.get('tree', row.get('id', f"tree_{idx}")))
                        if label is None or str(label).lower() == 'nan':
                            label = f"tree_{idx}"
                            
                        # Save mapping: Shapefile Index -> (Image Array, Label Name)
                        results_dict[idx] = (tile.astype('uint8'), str(label))
    except Exception as e:
        pass # Skip on file error
    return results_dict

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir, embedding_dir, label_dir, plot_dir = "data/models", "data/embeddings", "data/label", "data/plots"
    for d in [model_dir, embedding_dir, label_dir, plot_dir]: os.makedirs(d, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Shapefile and building Spatial Index...")
    if not os.path.exists(ANNOTATED_COR):
        print(f"Error: Shapefile not found at {ANNOTATED_COR}")
        return
    gdf = gpd.read_file(ANNOTATED_COR)
    
    # Initialize all entries with blank patches and "out-of-bound" labels
    # This guarantees the output length matches the Shapefile perfectly.
    black_patch = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    all_patches_dict = {idx: (black_patch, "out-of-bound") for idx in gdf.index}
    
    # Search for .tif files
    search_pattern = os.path.join(BASE_DIR, "2023-*", "*.tif")
    tif_files = glob.glob(search_pattern)
    
    if not tif_files:
        print(f"Error: Could not find files with pattern: {search_pattern}")
        return

    print(f"Found {len(tif_files)} .tif files. Extracting patches & resolving duplicates...")
    
    # Parallel extraction
    with ProcessPoolExecutor() as executor:
        func = partial(extract_patches_from_single_tif, gdf_indexed=gdf)
        futures = {executor.submit(func, tif): tif for tif in tif_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tif_files), desc="Extracting", unit="file"):
            res_dict = future.result()
            # Update the global dictionary. This automatically overwrites duplicates with the latest valid patch.
            for idx, val in res_dict.items():
                all_patches_dict[idx] = val

    # Split into Valid (for Training) and All (for Evaluation/Embedding extraction)
    eval_raw_patches = [all_patches_dict[idx][0] for idx in gdf.index]
    eval_patch_labels = [all_patches_dict[idx][1] for idx in gdf.index]
    
    # Filter out out-of-bound patches for model training so the model doesn't train on black images
    train_raw_patches = []
    for patch, label in zip(eval_raw_patches, eval_patch_labels):
        if label != "out-of-bound":
            train_raw_patches.append(patch)
            
    num_valid = len(train_raw_patches)
    num_oob = len(eval_patch_labels) - num_valid
    print(f"\nExtraction Summary:")
    print(f"- Total coordinates in Shapefile: {len(gdf)}")
    print(f"- Valid patches extracted successfully: {num_valid}")
    print(f"- Out-of-bound or missing patches: {num_oob}")

    if num_valid == 0:
        print("Error: No valid patches extracted. Cannot train the model.")
        return

    print("Converting to tensors and creating dataloaders...")
    
    # Dataloader for Training (Only Valid patches)
    processed_train_patches = torch.stack([transform(p) for p in train_raw_patches])
    train_dataset = TensorDataset(processed_train_patches)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Dataloader for Extraction (All patches, including Out-of-bounds)
    processed_eval_patches = torch.stack([transform(p) for p in eval_raw_patches])
    eval_dataset = TensorDataset(processed_eval_patches)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model training
    encoder = LeJepaEncoder().to(device)
    predictor = LeJepaPredictor().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 300
    print(f"\nStarting LeJEPA model training on {num_valid} valid patches. (Epochs: {epochs})")
    
    for epoch in range(epochs):
        encoder.train(); predictor.train()
        epoch_loss = 0.0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        
        for batch in batch_iterator:
            imgs = batch[0].to(device)
            B = imgs.shape[0]
            optimizer.zero_grad()
            with torch.no_grad(): target = encoder(imgs).detach()
            
            ids_shuffle = torch.argsort(torch.rand(B, (IMG_SIZE//PATCH_SIZE)**2, device=device), dim=1)
            ids_keep, ids_mask = ids_shuffle[:, :196], ids_shuffle[:, 196:]
            
            context = encoder(imgs, ids_keep=ids_keep)
            m_pos = torch.gather(encoder.pos_embed.expand(B, -1, -1), dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, 128))
            pred = predictor(context, m_pos)
            actual_target = torch.gather(target, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, 128))
            
            loss = criterion(pred, actual_target)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Completed - Average Loss: {epoch_loss/len(train_loader):.6f}")

    print("\nTraining complete. Extracting final embeddings for all 1,481 items...")
    encoder.eval()
    
    # Extract embeddings for ALL data (including out-of-bounds)
    all_embeds = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Generating Embeddings", unit="batch"):
            e = encoder(batch[0].to(device)).mean(dim=1).cpu().numpy()
            all_embeds.append(e)
    final_embeds = np.concatenate(all_embeds, axis=0)
    
    # Visualization: Global t-SNE (Filter out 'out-of-bound' to prevent squishing the valid points)
    valid_indices = [i for i, lbl in enumerate(eval_patch_labels) if lbl != "out-of-bound"]
    if len(valid_indices) > 1:
        print("Generating Global t-SNE Plot (Valid trees only)...")
        valid_embeds = final_embeds[valid_indices]
        tsne_2d = TSNE(n_components=2, perplexity=min(30, len(valid_embeds)-1), random_state=42).fit_transform(valid_embeds)
        plt.figure(figsize=(8,8))
        plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], alpha=0.5, s=10)
        plt.title("Global Embedding Space (t-SNE)")
        plt.savefig(f"{plot_dir}/global_tsne.png"); plt.close()

    # Save exactly 1481 embeddings and labels
    np.save(f"{embedding_dir}/embeddings.npy", final_embeds)
    np.save(f"{label_dir}/labels.npy", np.array(eval_patch_labels))
    torch.save(encoder.state_dict(), f"{model_dir}/lejepa_encoder.pth")
    
    print("\nAll tasks completed successfully!")
    print(f"Total Embeddings Saved: {len(final_embeds)}")
    print(f"Total Labels Saved: {len(eval_patch_labels)}")
    print(f"Embedding file: {embedding_dir}/embeddings.npy")
    print(f"Label file: {label_dir}/labels.npy")

if __name__ == "__main__":
    main()