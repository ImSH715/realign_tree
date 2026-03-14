"""
필요 패키지: pip install torch torchvision numpy matplotlib scikit-learn
실행: python lejepa_tsne_visualization.py
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

# 대용량 위성 이미지(TIFF) 로드를 위한 설정
Image.MAX_IMAGE_PIXELS = None

# ==========================================
# 1. 인코더 및 Predictor 정의 (Le-JEPA 구조)
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
        
        # 학습 시 마스킹된 패치를 제외하고 연산하기 위함
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

# ==========================================
# 2. 학습 유틸리티 (마스킹, EMA 업데이트, 크롭)
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
        print(f"이미지 로드 중: {image_path}")
        img = Image.open(image_path).convert('RGB')
        right, bottom = crop_x + crop_size, crop_y + crop_size
        img = img.crop((crop_x, crop_y, right, bottom))
    else:
        print(f"경고: 파일을 찾을 수 없어 임의의 노이즈 패턴을 생성합니다.")
        img_array = np.random.rand(crop_size, crop_size, 3) * 255
        img = Image.fromarray(img_array.astype(np.uint8))
        
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0), img

# ==========================================
# 3. 메인 분석 (학습 + 시각화)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 기기: {device}")
    
    # ========================================================
    # [사용자 설정 영역] 이미지 경로, 크롭, 학습 설정
    # ========================================================
    IMAGE_PATH = r"G:\.shortcut-targets-by-id\1IWblie-cf89tMuc4dQ8umWTHw7XOdl0p\PROYECTO FORGE\01. Ortomosaicos\2023\2023-01\25-PUC-C-DE-CPC-002-12_18032023_001_idw_transparent_mosaic_group1.tif"
    
    # 크롭 설정
    CROP_X = 2000       
    CROP_Y = 2000       
    CROP_SIZE = 2048    
    
    # 학습(Epoch) 설정
    DO_TRAINING = True
    TRAIN_EPOCHS = 500
    
    IMG_SIZE = 448      
    PATCH_SIZE = 16
    EMBED_DIM = 128
    # ========================================================
    
    # 1. 이미지 로드 및 크롭
    img_tensor, _ = load_and_crop_image(IMAGE_PATH, img_size=IMG_SIZE, crop_x=CROP_X, crop_y=CROP_Y, crop_size=CROP_SIZE)
    img_tensor = img_tensor.to(device)
    
    # 2. 모델 초기화
    target_encoder = LeJepaEncoder(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM).to(device)
    
    if DO_TRAINING:
        print(f"\n--- 크롭 영역에 대한 Le-JEPA 자체 학습 시작 ({TRAIN_EPOCHS} Epochs) ---")
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
                
        print("--- 학습 완료! ---")
    else:
        print("\n--- 학습 스킵 ---")
    
    # 3. 마스킹 없이 전체 이미지의 잠재 벡터 추출 (시각화용)
    target_encoder.eval()
    print("\nTarget Encoder로 특징 추출 중...")
    with torch.no_grad():
        features = target_encoder(img_tensor) # (1, 784, 128)
        
    B, L, D = features.shape
    H_p = W_p = int(np.sqrt(L)) # 28
    
    # 공간 차원으로 복구: (1, 128, 28, 28)
    features_spatial = features.transpose(1, 2).reshape(1, D, H_p, W_p)
    
    # ==========================================================
    # [수정된 부분] 테두리 문제(Edge Artifact) 해결
    # - Zero-padding 대신 Replication padding(가장자리 복제)을 사용하여
    #   테두리에 검은색(0)이 섞여 들어가는 현상을 방지합니다.
    # ==========================================================
    features_padded = F.pad(features_spatial, pad=(1, 1, 1, 1), mode='replicate')
    pooled_features = F.avg_pool2d(features_padded, kernel_size=3, stride=1, padding=0)
    
    flat_features = pooled_features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    flat_features_2d = flat_features.reshape(-1, D)
    
    # 4. PCA 및 t-SNE 변환
    print("- PCA 변환 (RGB 맵 생성용)...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(flat_features_2d)
    
    rgb_features = np.zeros_like(pca_features)
    for i in range(3):
        pc = pca_features[:, i]
        min_val, max_val = np.percentile(pc, 2), np.percentile(pc, 98)
        rgb_features[:, i] = np.clip((pc - min_val) / (max_val - min_val + 1e-8), 0, 1)
        
    rgb_map_small = rgb_features.reshape(H_p, W_p, 3)
    rgb_tensor = torch.tensor(rgb_map_small).permute(2, 0, 1).unsqueeze(0).float()
    
    # 보간(Interpolation)시에도 테두리가 부드럽게 이어지도록 처리
    rgb_map_large = F.interpolate(rgb_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bicubic', align_corners=True)
    rgb_map_final = np.clip(rgb_map_large[0].permute(1, 2, 0).numpy(), 0, 1)
    
    print("- t-SNE 변환 (산점도 시각화용)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    tsne_features = tsne.fit_transform(flat_features_2d)
    
    # 5. 결과 시각화
    fig = plt.figure(figsize=(20, 6))
    plt.suptitle(f"Target Encoder Analysis (Cropped Region: {CROP_SIZE}x{CROP_SIZE} at X:{CROP_X}, Y:{CROP_Y})", fontsize=16, fontweight='bold')
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_tensor[0].cpu().permute(1, 2, 0).numpy())
    ax1.set_title("1. Cropped Input Image", fontsize=14)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(rgb_map_final)
    ax2.set_title("2. PCA-based RGB Feature Map", fontsize=14)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(tsne_features[:, 0], tsne_features[:, 1], c=rgb_features, s=20, alpha=0.8)
    ax3.set_title("3. t-SNE Scatter Plot of Feature Space", fontsize=14)
    ax3.set_xlabel("t-SNE Dimension 1")
    ax3.set_ylabel("t-SNE Dimension 2")
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    save_filename = f"target_tsne_crop_{TRAIN_EPOCHS}.png"
    plt.savefig(save_filename, dpi=200)
    print(f"\n[성공] '{save_filename}' 파일로 저장되었습니다. 이제 테두리가 자연스러울 것입니다!")

if __name__ == "__main__":
    main()