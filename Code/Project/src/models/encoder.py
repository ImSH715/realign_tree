from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm with: pip install timm") from e


class ViTBackbone(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        if hasattr(self.model, "num_features"):
            self.embed_dim = self.model.num_features
        elif hasattr(self.model, "embed_dim"):
            self.embed_dim = self.model.embed_dim
        else:
            raise ValueError("Could not infer embedding dimension from the backbone.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.forward_features(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if feat.ndim == 3:
            if hasattr(self.model, "num_prefix_tokens") and self.model.num_prefix_tokens > 0:
                return feat[:, 0]
            return feat.mean(dim=1)

        return feat


class MLPProjector(nn.Module):
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


class LeJEPALikeEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 512,
        backbone_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ViTBackbone(backbone_name, pretrained=backbone_pretrained)
        self.projector = MLPProjector(
            in_dim=self.backbone.embed_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
        )
        self.predictor = nn.Sequential(
            nn.Linear(projector_out_dim, projector_out_dim),
            nn.GELU(),
            nn.Linear(projector_out_dim, projector_out_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project(self, feat: torch.Tensor) -> torch.Tensor:
        return self.projector(feat)

    def predict(self, proj: torch.Tensor) -> torch.Tensor:
        return self.predictor(proj)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("Input must be a square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SliceRegularization(nn.Module):
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

    def forward(self, predicted_views: List[torch.Tensor], target_views: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
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