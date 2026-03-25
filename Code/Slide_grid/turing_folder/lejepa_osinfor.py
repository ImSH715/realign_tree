"""
    This script is using the original data of the Osinfor. This is to realign Central Forestal.csv file
    It uses the trained lejepa model's embeddings and labels (geographic dataset) to predict if there are similar species like the trained model.
    The output shows the labels, coordinate of the predicted trees.
    The output will be used to perform sliding grid to realign Centro_Forest.csv file.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import rasterio
import geopandas as gpd
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from shapely.geometry import Point
from tqdm import tqdm

# --------------------------
# 1. Configurations & Paths
# --------------------------
# Paths to your previously trained model and extracted knowledge
MODEL_PATH = "data/models/lejepa_encoder_best.pth"
OLD_EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"
OLD_LABELS_PATH = "data/label/labels.npy"

# Path to the new, unlabeled TIF files (Make sure this points to your parscratch area)
NEW_TIF_DIR = r"/mnt/parscratch/users/acb20si/2025_Turing_L/datasets/Osinfor/Ortomosaicos"
OUTPUT_SHP = "data/predictions/predicted_map.shp"

# Hyperparameters (Must match the ones used during training)
IMG_SIZE = 448
PATCH_SIZE = 16
CROP_MULTIPLIER = 4
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER
HALF_CROP = CROP_SIZE // 2

# Sliding window stride. 
# If STRIDE == CROP_SIZE, patches will not overlap (faster). 
# If STRIDE < CROP_SIZE, patches will overlap (denser predictions but takes longer).
STRIDE = CROP_SIZE  
BATCH_SIZE = 32

# --------------------------
# 2. Pre-trained Model Definition
# --------------------------
# Note: We only need the Encoder for inference, as the Predictor is only used during SSL training.
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=128, depth=4, num_heads=4):
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

# --------------------------
# 3. Grid-based Dataset for Unlabeled Data
# --------------------------
class GridPatchDataset(Dataset):
    """
    Scans a TIF image and generates grid coordinates for patch extraction.
    Does not load the entire image into RAM at once to prevent OOM errors.
    """
    def __init__(self, tif_path, transform=None):
        self.tif_path = tif_path
        self.transform = transform
        self.points = []
        
        # Open TIF to get dimensions and spatial reference
        with rasterio.open(tif_path) as src:
            width, height = src.width, src.height
            self.crs = src.crs
            self.transform_matrix = src.transform
            
            # Generate grid coordinates, ensuring we stay within safe bounds (ignoring edges)
            for py in range(HALF_CROP, height - HALF_CROP, STRIDE):
                for px in range(HALF_CROP, width - HALF_CROP, STRIDE):
                    # Convert pixel coordinates to actual geographical coordinates (Longitude, Latitude)
                    lon, lat = rasterio.transform.xy(self.transform_matrix, py, px)
                    self.points.append({'px': px, 'py': py, 'x': lon, 'y': lat})

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pt = self.points[idx]
        
        # Read only the specific patch from the disk
        with rasterio.open(self.tif_path) as src:
            window = rasterio.windows.Window(pt['px'] - HALF_CROP, pt['py'] - HALF_CROP, CROP_SIZE, CROP_SIZE)
            tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            tile = np.moveaxis(tile, 0, -1) # Convert (C, H, W) to (H, W, C)

        img = Image.fromarray(tile.astype('uint8'))
        if self.transform:
            img = self.transform(img)
            
        return img, pt['x'], pt['y']

# --------------------------
# 4. Main Execution
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)

    # Step 1: Load prior knowledge (Embeddings & Labels from previous training)
    print("1. Loading prior knowledge (Embeddings & Labels)...")
    old_embeddings = np.load(OLD_EMBEDDINGS_PATH)
    old_labels = np.load(OLD_LABELS_PATH)
    
    # Step 2: Initialize and train the KNN Classifier
    # metric='cosine' is generally better for high-dimensional feature vectors than Euclidean distance.
    print("2. Fitting KNN Classifier with prior data...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(old_embeddings, old_labels)
    print(" -> KNN Ready!")

    # Step 3: Load the pre-trained LeJEPA Encoder
    print("3. Loading pre-trained LeJEPA Encoder...")
    encoder = LeJepaEncoder().to(device)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tif_files = glob.glob(os.path.join(NEW_TIF_DIR, "*.tif"))
    if not tif_files:
        print("Error: No TIF files found in the new directory. Please check NEW_TIF_DIR.")
        return

    all_predicted_records = []
    target_crs = None

    # Step 4: Extract features and predict labels for the new data
    print(f"4. Starting Inference on {len(tif_files)} TIF files...")
    with torch.no_grad():
        for tif in tif_files:
            dataset = GridPatchDataset(tif, transform=transform)
            
            if len(dataset) == 0:
                print(f"Skipping {os.path.basename(tif)} (No valid grid points found).")
                continue
                
            if target_crs is None:
                target_crs = dataset.crs

            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
            for imgs, xs, ys in tqdm(dataloader, desc=f"Processing {os.path.basename(tif)}"):
                imgs = imgs.to(device)
                
                # Forward pass: Extract features (embeddings) for the new images
                new_embeds = encoder(imgs).mean(dim=1).cpu().numpy()
                
                # KNN Prediction: Assign a label based on the closest old embeddings
                predicted_labels = knn.predict(new_embeds)
                
                # Store the results with coordinates for GIS export
                for i in range(len(predicted_labels)):
                    all_predicted_records.append({
                        'geometry': Point(xs[i].item(), ys[i].item()),
                        'pred_label': str(predicted_labels[i]),
                        'source_tif': os.path.basename(tif)
                    })

    # Step 5: Save the predictions to a Shapefile
    if all_predicted_records:
        print("\n5. Saving predictions to Shapefile...")
        gdf_preds = gpd.GeoDataFrame(all_predicted_records, crs=target_crs)
        gdf_preds.to_file(OUTPUT_SHP)
        print(f"Successfully saved {len(gdf_preds)} predicted points to: {OUTPUT_SHP}")
    else:
        print("Warning: No patches were processed or predicted.")

if __name__ == "__main__":
    main()