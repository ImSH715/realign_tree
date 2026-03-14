import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import torchvision.transforms as transforms
import os
import random
import math

# ==========================================
# 0. 초기화 함수 (ViT 표준)
# ==========================================
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

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
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True)
        self.blocks = nn.ModuleList([encoder_layer for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
    def __init__(self, embed_dim=128, predictor_depth=3, num_heads=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=.02)
        
        predictor_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True)
        self.blocks = nn.ModuleList([predictor_layer for _ in range(predictor_depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

# ==========================================
# 2. 데이터 증강 및 마스킹
# ==========================================
def apply_block_masking(img_tensor, patch_size=16, mask_ratio=0.6):
    B, C, H, W = img_tensor.shape
    H_p, W_p = H // patch_size, W // patch_size
    num_patches = H_p * W_p
    num_mask = int(num_patches * mask_ratio)
    
    device = img_tensor.device
    mask = torch.zeros(B, H_p, W_p, device=device)
    
    for b in range(B):
        mask_count = 0
        while mask_count < num_mask:
            block_h = random.randint(2, max(3, H_p // 2))
            block_w = random.randint(2, max(3, W_p // 2))
            top = random.randint(0, max(0, H_p - block_h))
            left = random.randint(0, max(0, W_p - block_w))
            
            current_block = mask[b, top:top+block_h, left:left+block_w]
            new_mask_pixels = (current_block == 0).sum().item()
            
            if mask_count + new_mask_pixels <= num_mask + (num_patches * 0.05):
                mask[b, top:top+block_h, left:left+block_w] = 1
                mask_count += new_mask_pixels
            else:
                break
                
    for b in range(B):
        flat_mask = mask[b].flatten()
        remaining = num_mask - int(flat_mask.sum().item())
        if remaining > 0:
            zero_indices = (flat_mask == 0).nonzero(as_tuple=True)[0]
            if len(zero_indices) > 0:
                random_indices = zero_indices[torch.randperm(len(zero_indices))[:remaining]]
                flat_mask[random_indices] = 1
        mask[b] = flat_mask.view(H_p, W_p)

    mask = mask.flatten(1) 
    ids_shuffle = torch.argsort(mask, dim=1) 
    num_keep = num_patches - num_mask
    ids_keep = ids_shuffle[:, :num_keep]
    ids_mask = ids_shuffle[:, num_keep:]
    
    return ids_keep, ids_mask

def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
    ])

# ==========================================
# 3. [흑백 전환] 흑백 PCA 시각화 함수
# ==========================================
def save_spatial_pca_grayscale(original_img, full_latents, patch_size=16, filename="pca_grayscale_no_border.png"):
    B, C, H, W = original_img.shape
    H_patches, W_patches = H // patch_size, W // patch_size
    
    features_normalized = F.normalize(full_latents, dim=-1)
    features_tensor = features_normalized.transpose(1, 2).view(1, -1, H_patches, W_patches)
    
    # 테두리 버그 수정 로직 (Replicate 패딩)
    padded_features = F.pad(features_tensor, pad=(1, 1, 1, 1), mode='replicate')
    smoothed_features = F.avg_pool2d(padded_features, kernel_size=3, stride=1, padding=0)
    
    features_flat = smoothed_features.view(1, -1, H_patches * W_patches).transpose(1, 2)[0].cpu().numpy()
    
    # -------------------------------------------------------------
    # [수정된 부분] 3개가 아닌, 가장 중요한 첫 번째 성분(PC1) 1개만 추출!
    # -------------------------------------------------------------
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(features_flat)[:, 0] # 첫 번째 채널만 가져옴
    
    # 정규화 (아웃라이어 제거하여 대비를 선명하게)
    min_val, max_val = np.percentile(pc1, 2), np.percentile(pc1, 98)
    pc1_normalized = np.clip((pc1 - min_val) / (max_val - min_val), 0, 1)
            
    # H_patches x W_patches 형태로 되돌리기
    pca_grid = pc1_normalized.reshape(H_patches, W_patches)
    
    # 이미지 사이즈 복원을 위해 텐서로 변환 [1, 1, H, W]
    pca_tensor = torch.tensor(pca_grid).unsqueeze(0).unsqueeze(0).float()
    
    # 원본 해상도로 부드럽게 크기 확대 (Bicubic)
    pca_resized = F.interpolate(pca_tensor, size=(H, W), mode='bicubic', align_corners=False)
    
    # 최종 2D 배열로 변환
    pca_grayscale_image = np.clip(pca_resized[0, 0].numpy(), 0, 1)
    
    # -------------------------------------------------------------
    # 그리기 영역 (흑백 컬러맵 적용)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    img_display = original_img[0].permute(1, 2, 0).cpu().numpy()
    img_display = np.clip(img_display, 0, 1)
    
    axes[0].imshow(img_display)
    axes[0].axis('off')
    axes[0].set_title("Original Crop", fontsize=14)
    
    # cmap='gray'를 설정하여 흑백으로 출력합니다!
    axes[1].imshow(pca_grayscale_image, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Features (Grayscale - PC1)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[성공] 흑백(Grayscale) 테두리 제거 PCA 시각화 저장됨: '{filename}'")

# 가상의 시각화 함수 
def save_latent_scatter_plots(*args, **kwargs): pass
def save_prediction_visualization(*args, **kwargs): pass


def main():
    # 파일명과 경로 설정 유지
    IMAGE_PATH = r"G:\.shortcut-targets-by-id\1IWblie-cf89tMuc4dQ8umWTHw7XOdl0p\PROYECTO FORGE\01. Ortomosaicos\2023\2023-01\25-PUC-C-DE-CPC-002-12_18032023_001_idw_transparent_mosaic_group1.tif"

    
    TARGET_SIZE = 384  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device} | 목표 이미지 사이즈: {TARGET_SIZE}x{TARGET_SIZE}")
    
    context_encoder = LeJepaEncoder(img_size=TARGET_SIZE, embed_dim=128).to(device)
    target_encoder = LeJepaEncoder(img_size=TARGET_SIZE, embed_dim=128).to(device)
    target_encoder.load_state_dict(context_encoder.state_dict())
    
    for param in target_encoder.parameters():
        param.requires_grad = False
        
    predictor = LeJepaPredictor(embed_dim=128).to(device)
    
    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()), lr=3e-4, weight_decay=0.04)
    
    try:
        Image.MAX_IMAGE_PIXELS = None
        raw_pil_image = Image.open(IMAGE_PATH).convert('RGB')
        print(f"원본 이미지 로드 완료 (해상도: {raw_pil_image.size})")
        
        USE_CUSTOM_AREA = True 
        
        if USE_CUSTOM_AREA:
            crop_box = (500, 500, 1500, 1500) 
            left, upper, right, lower = crop_box
            if right <= raw_pil_image.size[0] and lower <= raw_pil_image.size[1]:
                raw_pil_image = raw_pil_image.crop(crop_box)
            else:
                print("경고: 지정한 좌표가 이미지 원본 크기를 벗어났습니다. 전체 이미지를 사용합니다.")
                
    except Exception as e:
        print(f"이미지 로드 실패: {e}")
        raw_pil_image = Image.new('RGB', (1000, 1000), color='green')

    train_transform = get_train_transforms(img_size=TARGET_SIZE)

    epochs = 1000 
    print("\n--- Anti-Collapse 학습 시작 ---")
    
    context_encoder.train()
    predictor.train()
    
    for epoch in range(epochs):
        img_tensor = train_transform(raw_pil_image).unsqueeze(0).to(device)
        
        ids_keep, ids_mask = apply_block_masking(img_tensor, mask_ratio=0.6)
        B, N_mask = img_tensor.shape[0], ids_mask.shape[1]
        
        with torch.no_grad():
            full_targets = target_encoder(img_tensor)
            D = full_targets.shape[2]
            targets = torch.gather(full_targets, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
            
        contexts = context_encoder(img_tensor, ids_keep=ids_keep)
        pos_emb = context_encoder.pos_embed.expand(B, -1, -1)
        mask_pos = torch.gather(pos_emb, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        
        preds = predictor(contexts, mask_pos)
        
        preds_norm = F.normalize(preds, dim=-1)
        targets_norm = F.normalize(targets, dim=-1)
        
        loss = (1 - (preds_norm * targets_norm).sum(dim=-1)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        momentum = 1.0 - (1.0 - 0.996) * (math.cos(math.pi * epoch / epochs) + 1) / 2
        update_target_encoder_ema(context_encoder, target_encoder, momentum=momentum)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1:4d}/{epochs}] Cosine Loss: {loss.item():.4f}")

    print("\n--- 학습 완료! 시각화 추출 시작 ---")
    
    context_encoder.eval()
    with torch.no_grad():
        test_transform = transforms.Compose([
            transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
            transforms.ToTensor(),
        ])
        test_img = test_transform(raw_pil_image).unsqueeze(0).to(device)
        
        test_ids_keep, test_ids_mask = apply_block_masking(test_img, mask_ratio=0.6)
        
        full_targets = target_encoder(test_img)
        D = full_targets.shape[2]
        test_targets = torch.gather(full_targets, dim=1, index=test_ids_mask.unsqueeze(-1).expand(-1, -1, D))
        
        contexts = context_encoder(test_img, ids_keep=test_ids_keep)
        pos_emb = context_encoder.pos_embed.expand(1, -1, -1)
        mask_pos = torch.gather(pos_emb, dim=1, index=test_ids_mask.unsqueeze(-1).expand(-1, -1, D))
        
        test_preds = predictor(contexts, mask_pos)
        full_latents = context_encoder(test_img)
        
        os.makedirs("leJEPA_data", exist_ok=True) 
        
        try:
            # .cpu() 에러 해결 적용
            save_latent_scatter_plots(test_targets.cpu(), test_preds.cpu(), filename="leJEPA_data/lejepa_pca_result.png")
            save_prediction_visualization(test_img.cpu(), test_ids_mask.cpu(), test_targets.cpu(), test_preds.cpu(), filename="leJEPA_data/lejepa_prediction_result.png")
        except Exception as e:
            pass
            
        # 흑백(Grayscale) 시각화 함수 호출 (.cpu() 에러 해결 적용)
        save_spatial_pca_grayscale(test_img.cpu(), full_latents.cpu(), patch_size=16, filename="leJEPA_data/improved_pca_grayscale.png")

if __name__ == "__main__":
    main()