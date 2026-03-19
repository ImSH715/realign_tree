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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from leJepa import LeJepaEncoder

# ==========================================
# 1. Configuration & Paths
# ==========================================
FINAL_CSV = r"data/coordinate/sliding_results/final_centered_points_lejepa.csv"
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
MODEL_PATH = r"data/models/lejepa_encoder.pth"

# 새로 추출된 임베딩을 저장할 경로 (나중에 재사용하기 위함)
NEW_EMBED_DIR = r"data/embeddings"
os.makedirs(NEW_EMBED_DIR, exist_ok=True)
NEW_X_PATH = os.path.join(NEW_EMBED_DIR, "new_centered_embeddings.npy")
NEW_Y_PATH = os.path.join(NEW_EMBED_DIR, "new_centered_labels.npy")

IMG_SIZE = 448
CROP_MULTIPLIER = 4 
CROP_SIZE = IMG_SIZE * CROP_MULTIPLIER 
HALF_CROP = CROP_SIZE // 2

def extract_new_embeddings():
    """새로운 좌표(final_results.csv)를 바탕으로 TIF에서 이미지를 잘라 임베딩을 추출합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading LeJEPA Model for extraction...")
    encoder = LeJepaEncoder().to(device)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading final coordinates from: {FINAL_CSV}")
    df = pd.read_csv(FINAL_CSV)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    
    # 빠른 공간 검색을 위해 GeoDataFrame으로 변환
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:32718")
    
    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    
    X_list = []
    y_list = []

    print("Extracting fresh embeddings from centered coordinates...")
    for tif_path in tqdm(tif_files, desc="TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*src.bounds)]}, crs=src.crs)
                if bbox_gdf.crs != gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(gdf.crs)
                
                contained = gdf[gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
                if contained.empty:
                    continue
                
                # 좌표계 일치
                contained = contained.to_crs(src.crs)
                
                for _, row in contained.iterrows():
                    x, y = row.geometry.x, row.geometry.y
                    try:
                        py, px = src.index(x, y)
                        window = rasterio.windows.Window(px - HALF_CROP, py - HALF_CROP, CROP_SIZE, CROP_SIZE)
                        tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                        tile = np.moveaxis(tile, 0, -1)
                        
                        if tile.shape[0] != CROP_SIZE or tile.shape[1] != CROP_SIZE:
                            continue
                            
                        img_pil = Image.fromarray(tile.astype('uint8'))
                        tensor_img = transform(img_pil).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            embed = encoder(tensor_img).mean(dim=1).cpu().numpy()[0]
                            
                        X_list.append(embed)
                        y_list.append(row["label"])
                    except:
                        continue
        except:
            continue

    X = np.array(X_list)
    y = np.array(y_list)
    
    np.save(NEW_X_PATH, X)
    np.save(NEW_Y_PATH, y)
    print(f"Extraction complete! Saved {len(X)} embeddings.")
    return X, y

def run_linear_probe(X, y):
    """추출된 임베딩(X)과 라벨(y)로 Linear Probe 및 평가를 수행합니다."""
    print("\n" + "="*50)
    print(" 🚀 Starting Linear Probe Evaluation")
    print("="*50)
    
    # 클래스(수종)가 1개인 데이터 등 예외 처리
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("Error: Need at least 2 different classes to run classification and silhouette score.")
        return

    # 1. Silhouette Score
    sil_score = silhouette_score(X, y)
    print(f"🌟 New Silhouette Score : {sil_score:.4f}")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Logistic Regression (Linear Probe)
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)

    # 4. Accuracy
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Linear Probe Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("="*50)

    # 5. t-SNE Visualization
    print("\nGenerating t-SNE plot...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1],
        hue=y,
        palette="tab10",
        alpha=0.7
    )
    plt.title("t-SNE of Centered Tree Embeddings")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(NEW_EMBED_DIR, "tsne_centered_plot.png")
    plt.savefig(plot_path)
    print(f"Saved t-SNE plot to: {plot_path}")

def main():
    # 이미 추출해둔 파일이 있다면 불러오고, 없다면 새로 추출
    if os.path.exists(NEW_X_PATH) and os.path.exists(NEW_Y_PATH):
        print("Found existing new embeddings. Loading them...")
        X = np.load(NEW_X_PATH)
        y = np.load(NEW_Y_PATH, allow_pickle=True)
    else:
        print("No new embeddings found. Extracting them now...")
        X, y = extract_new_embeddings()

    if len(X) == 0:
        print("No embeddings to process.")
        return

    run_linear_probe(X, y)

if __name__ == "__main__":
    main()