import matplotlib.pyplot as plt
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
