"""Classification-only Stage 1 loader for feature-space Stage 2 experiments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .models import SleepStageClassifier, VAEEncoder


class ClassificationStage1(nn.Module):
    """Stage 1 classifier with the same encoder/head keys as classification checkpoints."""

    def __init__(
        self,
        input_len: int,
        latent_dim: int = 128,
        base_ch: int = 64,
        num_classes: int = 5,
        logvar_min: float = -8.0,
        logvar_max: float = 8.0,
        subwindow_len: int | None = None,
        use_transformer_encoder: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        transformer_cls_mean_pool: bool = True,
    ):
        super().__init__()
        self.encoder = VAEEncoder(
            input_len,
            latent_dim,
            base_ch,
            subwindow_len=subwindow_len,
            use_transformer_encoder=use_transformer_encoder,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            transformer_cls_mean_pool=transformer_cls_mean_pool,
        )
        self.stage_classifier = SleepStageClassifier(latent_dim, num_classes)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, _ = self.encoder(x)
        logvar = torch.zeros_like(mu)
        return mu, mu, logvar

    def encode_with_subwindows(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, _, mu_w, logvar_w, attn = self.encoder(x, return_subwindows=True)
        logvar = torch.zeros_like(mu)
        if logvar_w.numel() > 0:
            logvar_w = torch.zeros_like(logvar_w)
        return {
            "z": mu,
            "mu": mu,
            "logvar": logvar,
            "mu_w": mu_w,
            "logvar_w": logvar_w,
            "subwindow_attn": attn,
        }

    def reconstruct(self, x: torch.Tensor, use_mu: bool = True) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode_with_subwindows(x)
        z = encoded["z"]
        logits = self.stage_classifier(z)
        out = {
            "x": x,
            "x_hat": x,
            "logits": logits,
            "z": z,
            "mu": encoded["mu"],
            "logvar": encoded["logvar"],
        }
        if encoded["mu_w"].numel() > 0:
            subwindow_logits = self.stage_classifier(encoded["mu_w"].reshape(-1, encoded["mu_w"].size(-1)))
            out["subwindow_logits"] = subwindow_logits.view(encoded["mu_w"].size(0), encoded["mu_w"].size(1), -1)
            out["subwindow_attn"] = encoded["subwindow_attn"]
        return out


def load_stage1_classifier(ckpt_path: Path, device: torch.device) -> ClassificationStage1:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except (EOFError, RuntimeError, OSError) as exc:
        size = ckpt_path.stat().st_size if ckpt_path.exists() else 0
        raise RuntimeError(
            "Could not load Stage 1 classification checkpoint. "
            f"Path: {ckpt_path} (size={size} bytes). "
            "The checkpoint is likely truncated/corrupt or still syncing."
        ) from exc
    cfg = ckpt["cfg"]
    model = ClassificationStage1(
        input_len=ckpt["input_len"],
        latent_dim=cfg.latent_dim,
        base_ch=cfg.base_channels,
        num_classes=cfg.num_stages,
        logvar_min=getattr(cfg, "logvar_min", -8.0),
        logvar_max=getattr(cfg, "logvar_max", 8.0),
        subwindow_len=(
            int(round(ckpt["input_len"] * getattr(cfg, "subwindow_sec", 6.0) / 30.0))
            if getattr(cfg, "use_subwindow_encoder", False)
            else None
        ),
        use_transformer_encoder=getattr(cfg, "use_transformer_encoder", False),
        transformer_layers=getattr(cfg, "transformer_layers", 2),
        transformer_heads=getattr(cfg, "transformer_heads", 4),
        transformer_dropout=getattr(cfg, "transformer_dropout", 0.1),
        transformer_cls_mean_pool=getattr(cfg, "transformer_cls_mean_pool", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def load_frozen_classifier(ckpt_path: Path, device: torch.device) -> ClassificationStage1:
    model = load_stage1_classifier(ckpt_path, device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
