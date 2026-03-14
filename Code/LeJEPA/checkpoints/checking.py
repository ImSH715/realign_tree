"""
Le-JEPA의 두 번째 핵심 단계: Predictor와 Target Encoder 구현
실행: python predictor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms
from pca import save_latent_scatter_plots
from pca_visualisation import save_spatial_pca_visualization

img_path = "../cropped_tif/cropped_target_area.tif"
def update_target_encoder_ema(context_encoder, target_encoder, momentum=0.99):
    """
    Context Encoder의 가중치를 Target Encoder로 서서히 복사합니다.
    JEPA에서 모델 붕괴(Collapse)를 막는 핵심 매커니즘입니다.
    """
    with torch.no_grad():
        for param_c, param_t in zip(context_encoder.parameters(), target_encoder.parameters()):
            param_t.data.mul_(momentum).add_((1.0 - momentum) * param_c.detach().data)

# ==========================================
# 1. Le-JEPA 인코더 (Context & Target 인코더로 사용됨)
# ==========================================
class LeJepaEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128, depth=4, num_heads=4):
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
        
        x = x + self.pos_embed # 모든 패치에 위치 정보 추가
        
        if ids_keep is not None:
            B, L, D = x.shape
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
            
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# ==========================================
# 2. 새로 추가된 Predictor (예측기)
# ==========================================
class LeJepaPredictor(nn.Module):
    def __init__(self, embed_dim=128, predictor_depth=2, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 가려진 부분을 대신할 '학습 가능한 마스크 토큰'
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 예측용 트랜스포머 블록 (보통 인코더보다 얕게 설정)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, 
            activation='gelu', batch_first=True
        )
        self.blocks = nn.ModuleList([predictor_layer for _ in range(predictor_depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_embeds, mask_pos_embeds):
        """
        context_embeds: Context Encoder를 통과한 '보이는 패치'들의 잠재 벡터 (B, N_keep, D)
        mask_pos_embeds: 가려진 위치의 '위치 임베딩' 정보 (B, N_mask, D)
        """
        B = context_embeds.shape[0]
        N_mask = mask_pos_embeds.shape[1]

        # 1. 마스크 토큰 생성 및 가려진 곳의 위치 정보 추가
        mask_tokens = self.mask_token.repeat(B, N_mask, 1) 
        mask_tokens = mask_tokens + mask_pos_embeds 

        # 2. Context 벡터와 마스크 토큰 결합
        x = torch.cat([context_embeds, mask_tokens], dim=1)

        # 3. 트랜스포머를 통한 잠재 공간 예측
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 4. Context 부분은 버리고, '마스크 토큰' 위치의 예측된 결과만 반환
        predicted_latents = x[:, -N_mask:, :] 
        return predicted_latents

# ==========================================
# 유틸리티 함수
# ==========================================
def create_synthetic_image():
    """가상의 노이즈 이미지를 생성합니다 (대체용)."""
    img = np.random.rand(224, 224, 3).astype(np.float32)
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

def load_custom_image(image_path, img_size=224):
    """사용자의 실제 이미지를 불러와 Le-JEPA 입력 형태(1, 3, 224, 224)로 변환합니다."""
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        return transform(img).unsqueeze(0)
    else:
        print(f"경고: '{image_path}' 파일을 찾을 수 없어 임의의 가상 이미지를 생성하여 진행합니다.")
        return create_synthetic_image()

def apply_masking(img_tensor, patch_size=16, mask_ratio=0.75):
    B, C, H, W = img_tensor.shape
    num_patches = (H // patch_size) * (W // patch_size)
    num_keep = int(num_patches * (1 - mask_ratio))
    
    noise = torch.rand(B, num_patches)
    ids_shuffle = torch.argsort(noise, dim=1)
    
    ids_keep = ids_shuffle[:, :num_keep]
    ids_mask = ids_shuffle[:, num_keep:] # 가려질 패치의 인덱스
    
    return ids_keep, ids_mask

# ==========================================
# 결과 시각화 및 이미지 저장 함수
# ==========================================
def save_prediction_visualization(original_img, ids_mask, target_latents, predicted_latents, patch_size=16, filename="lejepa_prediction_result.png"):
    """학습 완료 후, 마스킹 전후 이미지와 잠재 공간 예측 결과를 이미지 파일로 저장합니다."""
    B, C, H, W = original_img.shape
    masked_img = original_img.clone()
    num_patches_w = W // patch_size
    
    for idx in ids_mask[0]: 
        row = (idx // num_patches_w) * patch_size
        col = (idx % num_patches_w) * patch_size
        masked_img[0, :, row:row+patch_size, col:col+patch_size] = 0.0 
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Le-JEPA Prediction Results (Before & After)", fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(original_img[0].permute(1, 2, 0).numpy())
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(masked_img[0].permute(1, 2, 0).numpy())
    axes[0, 1].set_title("2. Masked Image (Input to Encoder)")
    axes[0, 1].axis('off')
    
    im1 = axes[1, 0].imshow(target_latents[0, :10, :].cpu().numpy(), aspect='auto', cmap='viridis')
    axes[1, 0].set_title("3. Target Latent Vectors (Ground Truth)\n[First 10 Patches]")
    axes[1, 0].set_ylabel("Patch Index")
    axes[1, 0].set_xlabel("Latent Dimension")
    fig.colorbar(im1, ax=axes[1, 0])
    
    im2 = axes[1, 1].imshow(predicted_latents[0, :10, :].cpu().detach().numpy(), aspect='auto', cmap='viridis')
    axes[1, 1].set_title("4. Predicted Latent Vectors (Model Output)\n[First 10 Patches]")
    axes[1, 1].set_ylabel("Patch Index")
    axes[1, 1].set_xlabel("Latent Dimension")
    fig.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n[성공] 예측 결과가 '{filename}' 파일로 저장되었습니다! 직접 열어서 확인해 보세요.")

# ==========================================
# 3. 전체 흐름 테스트 (Loss 계산 포함)
# ==========================================
def main():
    torch.manual_seed(42) # 재현성을 위한 시드 고정
    
    # ========================================================
    # ★ 사용자 설정 부분 ★
    # 아래에 분석하고 싶은 본인의 이미지 경로를 입력하세요.
    # ========================================================
    USER_IMAGE_PATH = img_path
    # 2. 데이터 준비 및 마스킹
    
    # 1. 모델 인스턴스화
    print("--- 1. 모델 초기화 ---")
    context_encoder = LeJepaEncoder(embed_dim=128)
    
    target_encoder = LeJepaEncoder(embed_dim=128) 
    target_encoder.load_state_dict(context_encoder.state_dict()) 
    
    predictor = LeJepaPredictor(embed_dim=128)
    
    # 2. 데이터 준비 및 마스킹
    print("\n--- 2. 이미지 불러오기 및 마스킹 적용 ---")
    original_img = load_custom_image(USER_IMAGE_PATH) # 사용자 이미지 불러오기
    ids_keep, ids_mask = apply_masking(original_img, mask_ratio=0.7)
    
    print(f"전체 패치 수: 196")
    print(f"보이는 패치(Context) 수: {ids_keep.shape[1]}")
    print(f"가려진 패치(Target) 수: {ids_mask.shape[1]}")

    # 3. Target (정답) 생성
    print("\n--- 3. Target Encoder 연산 (정답 생성) ---")
    with torch.no_grad():
        full_target_latents = target_encoder(original_img)
        B, _, D = full_target_latents.shape
        target_latents = torch.gather(
            full_target_latents, dim=1, 
            index=ids_mask.unsqueeze(-1).expand(-1, -1, D)
        )
    print(f"추출된 정답 벡터 형태: {target_latents.shape}")

    # 4. Context 추출
    print("\n--- 4. Context Encoder 연산 (힌트 인코딩) ---")
    context_latents = context_encoder(original_img, ids_keep=ids_keep)
    print(f"추출된 힌트 벡터 형태: {context_latents.shape}")

    # 5. Predictor 연산
    print("\n--- 5. Predictor 연산 (가려진 부분 예측) ---")
    pos_embed_expanded = context_encoder.pos_embed.expand(B, -1, -1)
    mask_pos_embeds = torch.gather(
        pos_embed_expanded, dim=1, 
        index=ids_mask.unsqueeze(-1).expand(-1, -1, D)
    )
    
    predicted_latents = predictor(context_latents, mask_pos_embeds)
    print(f"예측된 벡터 형태: {predicted_latents.shape}")

    # 6. Loss 계산
    print("\n--- 6. 최종 Loss 계산 (Euclidean / L2 Loss) ---")
    loss = F.mse_loss(predicted_latents, target_latents)
    print(f"계산된 초기 Latent MSE Loss: {loss.item():.4f}")

    # 7. 실제 학습 루프
    print("\n--- 7. 본격적인 학습(Training) 시작 ---")
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()), 
        lr=1e-3
    )
    
    epochs = 200
    
    for epoch in range(epochs):
        # 이번 데모에서는 사용자가 입력한 단일 이미지 구조를 완벽히 외우도록(Overfit) 학습합니다.
        img = load_custom_image(USER_IMAGE_PATH)
        ids_keep, ids_mask = apply_masking(img, mask_ratio=0.7)
        B, N_mask = img.shape[0], ids_mask.shape[1]
        
        with torch.no_grad():
            full_targets = target_encoder(img)
            D = full_targets.shape[2]
            targets = torch.gather(full_targets, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
            
        contexts = context_encoder(img, ids_keep=ids_keep)
        
        pos_emb = context_encoder.pos_embed.expand(B, -1, -1)
        mask_pos = torch.gather(pos_emb, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        preds = predictor(contexts, mask_pos)
        
        train_loss = F.mse_loss(preds, targets)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        update_target_encoder_ema(context_encoder, target_encoder, momentum=0.99)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {train_loss.item():.4f}")

    print("\n--- 학습 완료! ---")

    # 8. 테스트 예측 및 결과 이미지 저장
    print("\n--- 8. 학습된 이미지 구조로 예측 및 결과 저장 ---")
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()
    
    with torch.no_grad():
        test_img = load_custom_image(USER_IMAGE_PATH) # 학습했던 이미지로 다시 테스트하여 결과 확인
        ids_keep, ids_mask = apply_masking(test_img, mask_ratio=0.7)
        B, N_mask = test_img.shape[0], ids_mask.shape[1]
        
        full_targets = target_encoder(test_img)
        D = full_targets.shape[2]
        test_targets = torch.gather(full_targets, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        
        contexts = context_encoder(test_img, ids_keep=ids_keep)
        pos_emb = context_encoder.pos_embed.expand(B, -1, -1)
        mask_pos = torch.gather(pos_emb, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        
        test_preds = predictor(contexts, mask_pos)
        
        test_loss = F.mse_loss(test_preds, test_targets)
        print(f"테스트 이미지 예측 Loss: {test_loss.item():.4f}")
        
        save_prediction_visualization(test_img, ids_mask, test_targets, test_preds, filename="lejepa_prediction_result.png")
        save_latent_scatter_plots(test_targets, test_preds, filename="lejepa_pca_result.png")
        # 8. 테스트 단계에서 예측을 마친 직후, 아래 코드 추가!
        # (주의: 마스킹 되지 않은 '전체 이미지'의 텐서와 인코더 출력을 넣어야 형태가 잡힙니다)
        with torch.no_grad():
            full_targets = target_encoder(test_img) # 전체 이미지에 대한 특징 추출
            save_spatial_pca_visualization(test_img, full_targets, patch_size=16, filename="lejepa_paper_fig14.png")

if __name__ == "__main__":
    main()
    