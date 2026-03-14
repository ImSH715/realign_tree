"""
Le-JEPA 잠재 공간(Latent Space) 산점도 및 PCA 시각화 도구
실행: python latent_pca_visualizer.py
필요 라이브러리: pip install torch numpy matplotlib scikit-learn
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def save_latent_scatter_plots(target_latents, predicted_latents, filename="latent_pca_scatter.png"):
    """
    정답(Target) 잠재 벡터와 예측(Predicted) 잠재 벡터를 산점도(Scatter Plot)로 시각화합니다.
    
    Args:
        target_latents: 정답 텐서 (Batch, N_mask, 128)
        predicted_latents: 예측 텐서 (Batch, N_mask, 128)
        filename: 저장할 이미지 파일 이름
    """
    # 텐서를 CPU로 옮기고, 넘파이 배열로 변환한 뒤 2차원 [N, 128]으로 평탄화
    D = target_latents.shape[-1]
    targets = target_latents.view(-1, D).cpu().detach().numpy()
    preds = predicted_latents.view(-1, D).cpu().detach().numpy()

    # 그래프 캔버스 설정 (1행 2열)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Latent Space Visualization (Target vs Predicted)", fontsize=16, fontweight='bold')

    # ==========================================
    # 1. 원본 차원 산점도 (Dim 1 vs Dim 2)
    # 128개 차원 중 가장 앞의 2개 차원만 단순히 플롯합니다.
    # ==========================================
    axes[0].scatter(targets[:, 0], targets[:, 1], s=15, alpha=0.5, label='Target Latents', c='#1f77b4') # 파란색
    axes[0].scatter(preds[:, 0], preds[:, 1], s=15, alpha=0.5, label='Predicted Latents', c='#ff7f0e')   # 주황색
    axes[0].set_title("Raw Latent Dimensions (Dim 1 vs Dim 2)")
    axes[0].set_xlabel("Latent Dimension 1")
    axes[0].set_ylabel("Latent Dimension 2")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # ==========================================
    # 2. PCA 적용 산점도 (128D -> 2D) - 요청하신 형태
    # 전체 128차원의 분산을 가장 잘 설명하는 2개의 축으로 압축합니다.
    # ==========================================
    # PCA 학습을 위해 Target과 Predict 데이터를 합칩니다
    combined_data = np.vstack((targets, preds))
    
    pca = PCA(n_components=2)
    pca.fit(combined_data) 
    
    # 128차원을 2차원으로 변환(Transform)
    targets_pca = pca.transform(targets)
    preds_pca = pca.transform(preds)

    axes[1].scatter(targets_pca[:, 0], targets_pca[:, 1], s=15, alpha=0.5, label='Target Latents', c='#1f77b4')
    axes[1].scatter(preds_pca[:, 0], preds_pca[:, 1], s=15, alpha=0.5, label='Predicted Latents', c='#ff7f0e')
    axes[1].set_title("Latent Space using PCA (128D -> 2D)")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n[성공] 잠재 공간 산점도 결과가 '{filename}' 파일로 저장되었습니다!")


if __name__ == "__main__":
    print("--- PCA 시각화 단독 테스트 시작 ---")
    print("가상의 Target 및 Predicted Latent 벡터를 생성하여 시각화를 테스트합니다...")
    
    # 가상의 128차원 벡터 5000개 생성 (예: 여러 패치가 뭉친 결과)
    torch.manual_seed(42)
    # Target은 특정 형태의 분포를 가지도록 임의 생성
    dummy_targets = torch.randn(1, 5000, 128) * 10.0
    
    # Predict는 Target에 약간의 노이즈(오차)가 섞인 형태로 모사 (학습이 꽤 진행된 상태)
    dummy_preds = dummy_targets + (torch.randn(1, 5000, 128) * 3.0) 