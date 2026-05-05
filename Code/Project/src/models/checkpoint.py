"""
Checkpoint utilities for saving and loading backbone-agnostic encoder models.

This module keeps Phase 2 and Phase 3 stable by storing enough metadata to
recreate the encoder architecture during checkpoint loading.

It supports:
- new checkpoints with explicit backbone configuration,
- older checkpoints that may contain model_state_dict,
- and safe fallback defaults for backward compatibility.
"""

import os
from typing import Any, Dict, Tuple, Optional

import torch

from src.models.encoder import SSLImageEncoder


def save_encoder_checkpoint(
    ckpt_path: str,
    model: SSLImageEncoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    payload = {
        "model_state": model.state_dict(),
        "backbone_name": model.backbone_name,
        "pretrained_backbone": model.pretrained_backbone,
        "image_size": model.image_size,
        "feature_dim": model.feature_dim,
        "projector_hidden_dim": model.projector_hidden_dim,
        "projector_out_dim": model.projector_out_dim,
    }

    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if config is not None:
        payload["config"] = config
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if extra is not None:
        payload["extra"] = extra

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(payload, ckpt_path)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    path: str,
    config_dict: dict,
) -> None:
    """
    Backward-compatible wrapper so older training code can still call save_checkpoint.
    """
    save_encoder_checkpoint(
        ckpt_path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        config=config_dict,
        best_metric=best_metric,
    )


def _infer_backbone_config(state: Dict[str, Any]) -> Dict[str, Any]:
    config = state.get("config", {}) if isinstance(state, dict) else {}

    backbone_name = (
        state.get("backbone_name")
        or config.get("backbone_name")
        or "vit_base_patch16_224"
    )

    pretrained_backbone = (
        state.get("pretrained_backbone")
        if "pretrained_backbone" in state
        else config.get("pretrained_backbone", config.get("backbone_pretrained", False))
    )

    image_size = (
        state.get("image_size")
        or config.get("image_size")
        or config.get("image_size_global")
        or 224
    )

    projector_hidden_dim = (
        state.get("projector_hidden_dim")
        or config.get("projector_hidden_dim")
        or 2048
    )

    projector_out_dim = (
        state.get("projector_out_dim")
        or config.get("projector_out_dim")
        or 512
    )

    return {
        "backbone_name": backbone_name,
        "pretrained_backbone": bool(pretrained_backbone),
        "image_size": int(image_size),
        "projector_hidden_dim": int(projector_hidden_dim),
        "projector_out_dim": int(projector_out_dim),
    }


def load_encoder_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[SSLImageEncoder, Dict[str, Any]]:
    state = torch.load(ckpt_path, map_location=device)

    if not isinstance(state, dict):
        raise ValueError("Checkpoint format is invalid.")

    cfg = _infer_backbone_config(state)

    model = SSLImageEncoder(
        backbone_name=cfg["backbone_name"],
        pretrained_backbone=False,  # always False when loading weights
        image_size=cfg["image_size"],
        projector_hidden_dim=cfg["projector_hidden_dim"],
        projector_out_dim=cfg["projector_out_dim"],
    )

    model_state = (
        state.get("model_state")
        or state.get("model_state_dict")
        or state.get("state_dict")
    )
    if model_state is None:
        raise ValueError("Checkpoint does not contain model_state/model_state_dict/state_dict.")

    missing, unexpected = model.load_state_dict(model_state, strict=False)

    if len(missing) > 0:
        print(f"[WARN] Missing keys while loading checkpoint: {missing}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys while loading checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model, state