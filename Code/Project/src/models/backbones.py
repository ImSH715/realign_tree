"""
Backbone factory for LeJEPA-style training and downstream evaluation.

This module creates a unified feature extractor interface for multiple backbones:
- standard ViT backbones from timm,
- ResNet50 CNN baseline,
- DINOv2 ViT backbones from timm (when available in the environment).

The goal is to keep Phase 2 and Phase 3 unchanged by exposing a consistent
forward() output: a [B, D] feature vector for any backbone.
"""

from typing import Tuple

import torch
import torch.nn as nn
import timm


def _pool_token_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Convert token or feature-map outputs into a [B, D] tensor.

    Cases:
    - [B, D] -> return directly
    - [B, N, D] -> use CLS token if present, otherwise mean over tokens
    - [B, C, H, W] -> global average pool
    """
    if x.ndim == 2:
        return x

    if x.ndim == 3:
        # Vision Transformer token output
        # Usually first token is CLS token
        return x[:, 0]

    if x.ndim == 4:
        # CNN feature map output
        return x.mean(dim=(2, 3))

    raise ValueError(f"Unsupported backbone output shape: {tuple(x.shape)}")


class TimmBackboneWrapper(nn.Module):
    """
    Wrap a timm model so that forward(x) always returns [B, D].
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.num_features = getattr(model, "num_features", None)
        if self.num_features is None:
            raise ValueError("Backbone model does not expose num_features.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            feat = self.model.forward_features(x)
        else:
            feat = self.model(x)

        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            elif "x_prenorm" in feat:
                feat = feat["x_prenorm"]
            else:
                for v in feat.values():
                    if torch.is_tensor(v):
                        feat = v
                        break

        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        feat = _pool_token_tensor(feat)
        return feat


def create_backbone(
    backbone_name: str,
    pretrained: bool = False,
    img_size: int = 224,
    in_chans: int = 3,
) -> Tuple[nn.Module, int]:
    """
    Create a backbone and return (wrapped_model, feature_dim).

    Recommended names:
    - "vit_base_patch16_224"              -> LeJEPA baseline backbone
    - "resnet50"                          -> CNN baseline
    - "vit_small_patch14_dinov2.lvd142m" -> DINOv2 small
    - "vit_base_patch14_dinov2.lvd142m"  -> DINOv2 base

    Notes:
    - pretrained=True for DINOv2 only works if weights are available in cache
      or the runtime can download them.
    - For large orthomosaic patch datasets, resnet50 and dino vit-small are
      practical choices.
    """
    backbone_name = backbone_name.strip()

    if backbone_name == "resnet50":
        model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_chans,
        )
        wrapped = TimmBackboneWrapper(model)
        return wrapped, wrapped.num_features

    model = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,
        img_size=img_size,
        in_chans=in_chans,
    )
    wrapped = TimmBackboneWrapper(model)
    return wrapped, wrapped.num_features