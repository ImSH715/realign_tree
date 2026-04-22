"""
Unified SSL encoder module for LeJEPA-style training and downstream use.

This file provides:
- a backbone-agnostic encoder that supports ViT, ResNet50, and DINOv2 backbones,
- projector and predictor heads for self-supervised training,
- LeJEPA-like regularization and loss,
- and backward-compatible class names so older imports do not break.

Phase 2 and Phase 3 only depend on model.encode(...), so the downstream pipeline
can remain unchanged as long as this interface is preserved.
"""

from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import create_backbone


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("Input must be a square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class MLPProjector(nn.Module):
    """
    MLP projector head used for self-supervised learning.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPPredictor(nn.Module):
    """
    Predictor head applied after the projector.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SliceRegularization(nn.Module):
    """
    Slice-based regularization term to discourage collapsed solutions.
    """

    def __init__(self, num_slices: int = 256):
        super().__init__()
        self.num_slices = num_slices

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        d = z.size(1)
        directions = torch.randn(self.num_slices, d, device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=1)
        projections = z @ directions.t()
        mean_term = projections.mean(dim=0).pow(2).mean()
        std_term = F.relu(1.0 - projections.std(dim=0, unbiased=False)).pow(2).mean()
        return mean_term + std_term


class LeJEPALikeLoss(nn.Module):
    """
    LeJEPA-like SSL loss composed of:
    - alignment loss between predicted and target projected views,
    - variance regularization,
    - covariance regularization,
    - slice regularization.
    """

    def __init__(
        self,
        align_weight: float = 1.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        slice_weight: float = 1.0,
        num_slices: int = 256,
    ) -> None:
        super().__init__()
        self.align_weight = align_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.slice_weight = slice_weight
        self.slice_reg = SliceRegularization(num_slices=num_slices)

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        var = z.var(dim=0, unbiased=False)
        std = torch.sqrt(var + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / max(1, (z.size(0) - 1))
        return off_diagonal(cov).pow(2).sum() / z.size(1)

    def forward(
        self,
        predicted_views: List[torch.Tensor],
        target_views: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        align_losses = []
        var_losses = []
        cov_losses = []
        slice_losses = []

        anchor_target = target_views[0].detach()

        for pv in predicted_views[1:]:
            align_losses.append(
                F.mse_loss(F.normalize(pv, dim=-1), F.normalize(anchor_target, dim=-1))
            )

        for zv in predicted_views:
            var_losses.append(self.variance_loss(zv))
            cov_losses.append(self.covariance_loss(zv))
            slice_losses.append(self.slice_reg(zv))

        for zv in target_views:
            var_losses.append(self.variance_loss(zv))
            cov_losses.append(self.covariance_loss(zv))
            slice_losses.append(self.slice_reg(zv))

        align_loss = torch.stack(align_losses).mean()
        var_loss = torch.stack(var_losses).mean()
        cov_loss = torch.stack(cov_losses).mean()
        slice_loss = torch.stack(slice_losses).mean()

        total = (
            self.align_weight * align_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
            + self.slice_weight * slice_loss
        )

        metrics = {
            "ssl_total": float(total.item()),
            "align_loss": float(align_loss.item()),
            "var_loss": float(var_loss.item()),
            "cov_loss": float(cov_loss.item()),
            "slice_loss": float(slice_loss.item()),
        }
        return total, metrics


class SSLImageEncoder(nn.Module):
    """
    Unified image encoder for LeJEPA-style training and downstream use.

    Supports:
    - timm ViTs
    - ResNet50
    - DINOv2 timm backbones

    Downstream compatibility:
    - Phase 2 / Phase 3 only need model.encode(x)
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained_backbone: bool = False,
        image_size: int = 224,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 512,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained_backbone = pretrained_backbone
        self.image_size = image_size
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_out_dim = projector_out_dim

        self.backbone, self.feature_dim = create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            img_size=image_size,
            in_chans=3,
        )

        self.projector = MLPProjector(
            in_dim=self.feature_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
        )
        self.predictor = MLPPredictor(dim=projector_out_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def project(self, feat: torch.Tensor) -> torch.Tensor:
        return self.projector(feat)

    def predict(self, proj: torch.Tensor) -> torch.Tensor:
        return self.predictor(proj)


class LeJEPALikeEncoder(SSLImageEncoder):
    """
    Backward-compatible alias for older code paths that still import
    LeJEPALikeEncoder.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 512,
        backbone_pretrained: bool = False,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained_backbone=backbone_pretrained,
            image_size=224,
            projector_hidden_dim=projector_hidden_dim,
            projector_out_dim=projector_out_dim,
        )