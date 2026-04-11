import os
import torch

from src.models.encoder import LeJEPALikeEncoder


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    path: str,
    config_dict: dict,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config_dict,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_encoder_from_checkpoint(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("config", {})

    model = LeJEPALikeEncoder(
        backbone_name=cfg.get("backbone_name", "vit_base_patch16_224"),
        projector_hidden_dim=cfg.get("projector_hidden_dim", 2048),
        projector_out_dim=cfg.get("projector_out_dim", 512),
        backbone_pretrained=cfg.get("backbone_pretrained", False),
    )
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, state