"""
Le-JEPA Advanced Trainer (Data Augmentation & PCA Visualization)
질문자님의 아이디어인 "부분 확대 및 미세한 변화(Augmentation)"를 적용하여 
단일 고해상도 이미지로부터 의미론적 특징(Semantic Features)을 학습합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import torchvision.transforms as transforms
import os
from checkpoints.pca import save_latent_scatter_plots
from visualise_masking import save_prediction_visualization

IMAGE_PATH = r"G:\.shortcut-targets-by-id\1IWblie-cf89tMuc4dQ8umWTHw7XOdl0p\PROYECTO FORGE\01. Ortomosaicos\2023\2023-01\25-PUC-C-DE-CPC-002-12_18032023_001_idw_transparent_mosaic_group1.tif"
    
# ==========================================
# 1. 모델 정의 (Encoder & Predictor)
# ==========================================
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128, depth=4, num_heads=4):
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
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class LeJepaPredictor(nn.Module):
    def __init__(self, embed_dim=128, predictor_depth=2, num_heads=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        predictor_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True)
        self.blocks = nn.ModuleList([predictor_layer for _ in range(predictor_depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_embeds, mask_pos_embeds):
        B, N_mask, _ = mask_pos_embeds.shape
        mask_tokens = self.mask_token.repeat(B, N_mask, 1) + mask_pos_embeds
        x = torch.cat([context_embeds, mask_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, -N_mask:, :]

def update_target_encoder_ema(context_encoder, target_encoder, momentum=0.996):
    with torch.no_grad():
        for param_c, param_t in zip(context_encoder.parameters(), target_encoder.parameters()):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_c.detach().data)

def apply_masking(img_tensor, patch_size=16, mask_ratio=0.7):
    B, C, H, W = img_tensor.shape
    num_patches = (H // patch_size) * (W // patch_size)
    num_keep = int(num_patches * (1 - mask_ratio))
    noise = torch.rand(B, num_patches, device=img_tensor.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep, ids_mask = ids_shuffle[:, :num_keep], ids_shuffle[:, num_keep:]
    return ids_keep, ids_mask

# ==========================================
# 2. 질문자님의 아이디어: 데이터 증강 (Augmentation) 파이프라인
# ==========================================
def get_train_transforms(img_size=400):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)), # 무작위로 20%~100% 면적을 잘라내어 확대
        transforms.RandomHorizontalFlip(),                        # 50% 확률로 좌우 반전
        transforms.RandomVerticalFlip(),                          # 50% 확률로 상하 반전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), # 색상 미세 변화
        transforms.ToTensor(),
    ])

def get_test_transforms(img_size=400):
    """
    평가용: 시각화를 위해 정중앙을 반듯하게 자릅니다.
    """
    return transforms.Compose([
        transforms.CenterCrop(img_size), # 중앙만 깔끔하게 크롭
        transforms.ToTensor(),
    ])

# ==========================================
# 3. PCA 시각화 함수 (논문 재현)
# ==========================================
def save_spatial_pca_visualization(original_img, full_latents, patch_size=16, filename="improved_pca_rgb.png"):
    B, C, H, W = original_img.shape
    H_patches, W_patches = H // patch_size, W // patch_size
    
    features = full_latents[0].cpu().detach().numpy()
    
    # PCA 적용 (128D -> 3D)
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    
    # 0~1 사이로 정규화하여 RGB 색상으로 변환
    for i in range(3):
        min_val, max_val = pca_features[:, i].min(), pca_features[:, i].max()
        # 분모가 0이 되는 것을 방지
        if max_val > min_val:
            pca_features[:, i] = (pca_features[:, i] - min_val) / (max_val - min_val)
        else:
            pca_features[:, i] = 0.5 
            
    pca_grid = pca_features.reshape(H_patches, W_patches, 3)
    pca_tensor = torch.tensor(pca_grid).permute(2, 0, 1).unsqueeze(0).float()
    
    # 원본 해상도로 보간(Interpolation)
    pca_resized = F.interpolate(pca_tensor, size=(H, W), mode='bicubic', align_corners=False)
    pca_rgb_image = np.clip(pca_resized[0].permute(1, 2, 0).numpy(), 0, 1)
    
    # 그리기
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    img_display = original_img[0].permute(1, 2, 0).cpu().numpy()
    img_display = np.clip(img_display, 0, 1)
    
    axes[0].imshow(img_display)
    axes[0].axis('off')
    axes[0].set_title("Original Crop", fontsize=14)
    
    axes[1].imshow(pca_rgb_image)
    axes[1].axis('off')
    axes[1].set_title("Advanced Le-JEPA Features (PCA to RGB)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[성공] 향상된 PCA 시각화가 '{filename}'에 저장되었습니다!")

# ==========================================
# 4. 향상된 학습 루프
# ==========================================
def main():
    # ---------------------------------------------------------
    # [사용자 설정] 넓은 영토를 담고 있는 해상도가 큰 이미지 경로를 넣으세요!
    # (이미 잘게 쪼갠 224x224 이미지가 아니라, 더 넓은 영역이 담긴 원본에 가까운 이미지일수록 좋습니다)
    # ---------------------------------------------------------
    IMAGE_PATH
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 1. 모델 초기화
    context_encoder = LeJepaEncoder(embed_dim=128).to(device)
    target_encoder = LeJepaEncoder(embed_dim=128).to(device)
    target_encoder.load_state_dict(context_encoder.state_dict())
    predictor = LeJepaPredictor(embed_dim=128).to(device)
    
    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()), lr=5e-4)
    
    # 2. 이미지 불러오기 (메모리에 로드)
    try:
        Image.MAX_IMAGE_PIXELS = None
        raw_pil_image = Image.open(IMAGE_PATH).convert('RGB')
        print(f"원본 이미지 로드 완료 (해상도: {raw_pil_image.size})")
    except Exception as e:
        print(f"이미지 로드 실패: {e}")
        print("가상의 이미지로 테스트를 진행합니다.")
        raw_pil_image = Image.new('RGB', (1000, 1000), color='green')

    train_transform = get_train_transforms(img_size=224)
    test_transform = get_test_transforms(img_size=224)

    # 3. 강화된 학습 루프 시작
    epochs = 2000
    print("\n--- 본격적인 Augmented 학습(Training) 시작 ---")
    
    context_encoder.train()
    predictor.train()
    
    for epoch in range(epochs):
        # 핵심!! 매 에폭마다 원본 이미지의 "다른 부분"을 "다른 크기"로 잘라오고 "색상"을 살짝 바꿉니다!
        img_tensor = train_transform(raw_pil_image).unsqueeze(0).to(device)
        
        ids_keep, ids_mask = apply_masking(img_tensor, mask_ratio=0.7)
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
        
        # 초반에는 타겟 앙상블을 천천히 업데이트하다가 나중에 빨라지게 하는 것이 안정적입니다.
        momentum = 0.996 + (1.0 - 0.996) * (epoch / epochs)
        update_target_encoder_ema(context_encoder, target_encoder, momentum=momentum)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1:4d}/{epochs}] Loss: {loss.item():.4f}")

    print("\n--- 학습 완료! 시각화 추출 시작 ---")
    # 4. 결과 시각화
    context_encoder.eval()
    
    with torch.no_grad():
        # 평가할 때는 무작위 자르기(Augmentation) 없이 정중앙을 예쁘게 자릅니다.
        test_img = test_transform(raw_pil_image).unsqueeze(0).to(device)

        full_targets = target_encoder(test_img)
        D = full_targets.shape[2]
        test_targets = torch.gather(full_targets, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        
        contexts = context_encoder(test_img, ids_keep=ids_keep)
        pos_emb = context_encoder.pos_embed.expand(B, -1, -1)
        mask_pos = torch.gather(pos_emb, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        
        test_preds = predictor(contexts, mask_pos)

        full_latents = context_encoder(test_img)
        save_latent_scatter_plots(test_targets, test_preds, filename="lejepa_pca_result.png")
        save_spatial_pca_visualization(test_img, full_latents, patch_size=16, filename="improved_pca_rgb.png")
        save_prediction_visualization(test_img, ids_mask, test_targets, test_preds, filename="lejepa_prediction_result.png")

if __name__ == "__main__":
    main()