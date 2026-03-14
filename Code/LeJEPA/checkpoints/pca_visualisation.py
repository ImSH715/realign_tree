"""
Le-JEPA Spatial PCA to RGB 시각화 도구
실행: python latent_pca_visualizer.py
설명: 인코더의 마지막 레이어에서 추출한 고차원 잠재 벡터를 
      PCA를 통해 3차원(RGB)으로 압축하여 의미론적 지도(Semantic Map)를 만듭니다.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def save_spatial_pca_visualization(original_img, full_latents, patch_size=16, filename="lejepa_spatial_pca.png"):
    """
    논문의 Figure 14 와 동일한 방식으로 잠재 공간을 RGB 이미지로 시각화합니다.
    
    Args:
        original_img: 원본 이미지 텐서 [1, 3, H, W]
        full_latents: 마스킹 없이 전체 이미지를 인코딩한 잠재 벡터 [1, N, D]
        patch_size: 인코더가 사용한 패치 크기 (기본값: 16)
        filename: 저장할 파일 이름
    """
    # 1. 원본 이미지 크기 및 패치 그리드 계산
    B, C, H, W = original_img.shape
    H_patches = H // patch_size
    W_patches = W // patch_size
    
    # 2. 잠재 벡터 평탄화 ([1, N, 128] -> [N, 128])
    features = full_latents[0].cpu().detach().numpy()
    
    # 3. PCA 적용: 128차원 -> 3차원 (RGB로 사용하기 위함)
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features) # [N, 3]
    
    # 4. RGB 색상 범위(0~1)로 정규화 (Min-Max Scaling)
    # 각 주성분(PC1, PC2, PC3)을 R, G, B 채널로 매핑합니다.
    for i in range(3):
        min_val = pca_features[:, i].min()
        max_val = pca_features[:, i].max()
        pca_features[:, i] = (pca_features[:, i] - min_val) / (max_val - min_val)
        
    # 5. 공간 그리드 형태로 재배열 ([N, 3] -> [H_patches, W_patches, 3])
    pca_grid = pca_features.reshape(H_patches, W_patches, 3)
    
    # 6. 원본 이미지 해상도로 부드럽게 확대 (Bicubic Interpolation)
    pca_tensor = torch.tensor(pca_grid).permute(2, 0, 1).unsqueeze(0).float() # [1, 3, H_p, W_p]
    pca_resized = F.interpolate(pca_tensor, size=(H, W), mode='bicubic', align_corners=False)
    
    # 최종 이미지 배열로 변환
    pca_rgb_image = pca_resized[0].permute(1, 2, 0).numpy()
    pca_rgb_image = np.clip(pca_rgb_image, 0, 1) # 범위를 벗어나는 값 잘라내기
    
    # ==========================================
    # 결과 그리기 (원본 vs PCA 시각화)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 좌측: 원본 이미지
    img_display = original_img[0].permute(1, 2, 0).cpu().numpy()
    # 원본 이미지가 정규화 되어있을 경우를 대비해 0~1로 맞춤
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    axes[0].imshow(img_display)
    axes[0].axis('off')
    axes[0].set_title("Original Image")
    
    # 우측: PCA RGB Feature Map
    axes[1].imshow(pca_rgb_image)
    axes[1].axis('off')
    axes[1].set_title("Le-JEPA Features (PCA to RGB)")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[성공] 논문 스타일의 Spatial PCA 시각화가 '{filename}'에 저장되었습니다!")

if __name__ == "__main__":
    print("--- Spatial PCA 단독 테스트 시작 ---")
    
    # 224x224 크기의 임의의 가짜 이미지 생성
    H, W = 224, 224
    patch_size = 16
    H_p, W_p = H // patch_size, W // patch_size
    dummy_img = torch.rand(1, 3, H, W)
    
    # 가상의 인코더 출력 (잠재 벡터) 생성
    # 논문의 결과처럼 전경과 배경이 분리되는 것을 모사하기 위해
    # 이미지 중앙(전경)과 테두리(배경)의 벡터 값을 다르게 설정합니다.
    dummy_latents = torch.zeros(1, H_p * W_p, 128)
    for i in range(H_p):
        for j in range(W_p):
            idx = i * W_p + j
            # 중앙 원형 영역을 '전경(Foreground)'으로 가정
            if (i - H_p//2)**2 + (j - W_p//2)**2 < (H_p//4)**2:
                dummy_latents[0, idx, :] = torch.randn(128) * 0.1 + 1.0 
            else:
                dummy_latents[0, idx, :] = torch.randn(128) * 0.1 - 1.0 
                
    # 함수 실행
    save_spatial_pca_visualization(dummy_img, dummy_latents, patch_size=16, filename="spatial_pca_test.png")
    print("스크립트가 완료되었습니다. 'spatial_pca_test.png'를 확인해보세요!")