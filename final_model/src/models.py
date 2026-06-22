"""Multi-head VAE (Stage 1) and conditional latent diffusion (Stage 2)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VAEEncoder(nn.Module):
    """Raw EEG x_t -> mu, logvar for z_t."""

    def __init__(
        self,
        input_len: int,
        latent_dim: int = 256,
        base_ch: int = 64,
        subwindow_len: int | None = None,
        use_transformer_encoder: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        transformer_cls_mean_pool: bool = True,
    ):
        super().__init__()
        self.input_len = input_len
        self.subwindow_len = subwindow_len if subwindow_len and subwindow_len < input_len else None
        self.use_transformer_encoder = bool(use_transformer_encoder and self.subwindow_len is not None)
        self.transformer_cls_mean_pool = bool(transformer_cls_mean_pool)
        encoder_len = self.subwindow_len or input_len
        self.backbone = nn.Sequential(
            ConvBlock1d(1, base_ch, stride=2),
            ConvBlock1d(base_ch, base_ch * 2, stride=2),
            ConvBlock1d(base_ch * 2, base_ch * 4, stride=2),
            ConvBlock1d(base_ch * 4, base_ch * 8, stride=2),
            ConvBlock1d(base_ch * 8, base_ch * 16, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, encoder_len)
            flat_dim = self.backbone(dummy).view(1, -1).shape[1]
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.subwindow_attn = (
            nn.Linear(latent_dim, 1)
            if self.subwindow_len is not None and not self.use_transformer_encoder
            else None
        )
        if self.use_transformer_encoder:
            n_windows = max(1, input_len // self.subwindow_len)
            n_heads = max(1, min(int(transformer_heads), latent_dim))
            while latent_dim % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            enc_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=n_heads,
                dim_feedforward=latent_dim * 4,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_windows + 1, latent_dim))
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
            self.transformer_norm = nn.LayerNorm(latent_dim)
            pooled_dim = latent_dim * 2 if self.transformer_cls_mean_pool else latent_dim
            self.transformer_mu = nn.Linear(pooled_dim, latent_dim)
            self.transformer_logvar = nn.Linear(pooled_dim, latent_dim)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_subwindows: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.subwindow_len is None:
            h = self.backbone(x).flatten(1)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            if return_subwindows:
                empty = mu.new_empty(mu.size(0), 0, mu.size(1))
                return mu, logvar, empty, empty, mu.new_empty(mu.size(0), 0)
            return mu, logvar

        usable_len = (x.size(-1) // self.subwindow_len) * self.subwindow_len
        if usable_len < self.subwindow_len:
            raise ValueError(f"input length {x.size(-1)} is shorter than subwindow_len={self.subwindow_len}")
        xw = x[..., :usable_len].unfold(dimension=-1, size=self.subwindow_len, step=self.subwindow_len)
        b, c, n_win, win_len = xw.shape
        xw = xw.permute(0, 2, 1, 3).reshape(b * n_win, c, win_len)
        h = self.backbone(xw).flatten(1)
        mu_w = self.fc_mu(h).view(b, n_win, -1)
        logvar_w = self.fc_logvar(h).view(b, n_win, -1)
        if self.use_transformer_encoder:
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, mu_w], dim=1)
            tokens = tokens + self.pos_embed[:, : tokens.size(1)]
            encoded = self.transformer_norm(self.transformer(tokens))
            encoded_w = encoded[:, 1:]
            if self.transformer_cls_mean_pool:
                pooled = torch.cat([encoded[:, 0], encoded_w.mean(dim=1)], dim=1)
            else:
                pooled = encoded[:, 0]
            mu = self.transformer_mu(pooled)
            logvar = self.transformer_logvar(pooled)
            if return_subwindows:
                attn = mu.new_full((b, n_win), 1.0 / max(n_win, 1))
                return mu, logvar, encoded_w, logvar_w, attn
            return mu, logvar
        attn = torch.softmax(self.subwindow_attn(torch.tanh(mu_w)).squeeze(-1), dim=1)
        mu = (mu_w * attn.unsqueeze(-1)).sum(dim=1)
        logvar = (logvar_w * attn.unsqueeze(-1)).sum(dim=1)
        if return_subwindows:
            return mu, logvar, mu_w, logvar_w, attn
        return mu, logvar


class ReconstructionDecoder(nn.Module):
    """z_t -> reconstructed EEG."""

    def __init__(self, output_len: int, latent_dim: int = 256, base_ch: int = 64):
        super().__init__()
        self.output_len = output_len
        init_len = output_len // 32
        self.init_len = max(1, init_len)
        self.fc = nn.Linear(latent_dim, base_ch * 16 * self.init_len)
        self.base_ch = base_ch
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_ch * 16, base_ch * 8, 4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch * 8),
            nn.GELU(),
            nn.ConvTranspose1d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch * 4),
            nn.GELU(),
            nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch * 2),
            nn.GELU(),
            nn.ConvTranspose1d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.ConvTranspose1d(base_ch, 1, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), self.base_ch * 16, self.init_len)
        out = self.decoder(h)
        if out.size(-1) != self.output_len:
            out = F.interpolate(out, size=self.output_len, mode="linear", align_corners=False)
        return out


class SleepStageClassifier(nn.Module):
    """z_t -> 5-class sleep stage logits (Linear head + CE / weighted CE)."""

    def __init__(self, latent_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultiHeadVAE(nn.Module):
    """Stage 1: shared encoder, reconstruction decoder, stage classifier."""

    def __init__(
        self,
        input_len: int,
        latent_dim: int = 256,
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
        self.recon_decoder = ReconstructionDecoder(input_len, latent_dim, base_ch)
        self.stage_classifier = SleepStageClassifier(latent_dim, num_classes)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode_with_subwindows(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar, mu_w, logvar_w, attn = self.encoder(x, return_subwindows=True)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        if logvar_w.numel() > 0:
            logvar_w = torch.clamp(logvar_w, self.logvar_min, self.logvar_max)
        z = self.reparameterize(mu, logvar)
        return {"z": z, "mu": mu, "logvar": logvar, "mu_w": mu_w, "logvar_w": logvar_w, "subwindow_attn": attn}

    def reconstruct(self, x: torch.Tensor, use_mu: bool = True) -> torch.Tensor:
        """Deterministic reconstruction for inspection when use_mu=True."""
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        z = self.reparameterize(mu, logvar)
        return self.recon_decoder(mu if use_mu else z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode_with_subwindows(x)
        z = encoded["z"]
        mu = encoded["mu"]
        logvar = encoded["logvar"]
        x_hat = self.recon_decoder(z)
        logits = self.stage_classifier(z)
        out = {"x_hat": x_hat, "logits": logits, "z": z, "mu": mu, "logvar": logvar}
        if encoded["mu_w"].numel() > 0:
            subwindow_logits = self.stage_classifier(encoded["mu_w"].reshape(-1, encoded["mu_w"].size(-1)))
            out["subwindow_logits"] = subwindow_logits.view(
                encoded["mu_w"].size(0),
                encoded["mu_w"].size(1),
                -1,
            )
            out["subwindow_attn"] = encoded["subwindow_attn"]
        return out


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def kl_loss_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def spectral_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 64,
    win_length: int = 256,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    L_spec is the per-sample mean squared error between STFT magnitudes.
    Phase is ignored; only magnitude (band power) is matched.  Using a mean
    keeps this term on a scale comparable to the waveform MSE.
    """
    if x.dim() == 3:
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    spec_x = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    spec_hat = torch.stft(
        x_hat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    diff = spec_x.abs() - spec_hat.abs()
    per_sample = diff.square().reshape(diff.size(0), -1).mean(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def bandpower_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    sfreq: float = 100.0,
    bands: tuple[tuple[float, float], ...] = (
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 30.0),
    ),
) -> torch.Tensor:
    """MSE between log band powers: delta/theta/alpha/sigma/beta."""
    if x.dim() == 2:
        x = x.unsqueeze(1)
        x_hat = x_hat.unsqueeze(1)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / sfreq).to(x.device)
    px = torch.fft.rfft(x, dim=-1).abs().square()
    ph = torch.fft.rfft(x_hat, dim=-1).abs().square()

    losses = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if not bool(mask.any()):
            continue
        bx = torch.log1p(px[..., mask].mean(dim=-1))
        bh = torch.log1p(ph[..., mask].mean(dim=-1))
        losses.append((bx - bh).square())
    if not losses:
        return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    return torch.stack(losses, dim=-1).mean(dim=(1, 2))


def _fft_bandpass(x: torch.Tensor, sfreq: float, lo: float, hi: float) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(1)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / sfreq).to(x.device)
    mask = ((freqs >= lo) & (freqs < hi)).to(x.dtype)
    spec = torch.fft.rfft(x, dim=-1)
    return torch.fft.irfft(spec * mask.view(1, 1, -1), n=x.size(-1), dim=-1)


def _rms_envelope(
    x: torch.Tensor,
    sfreq: float,
    window_sec: float,
) -> torch.Tensor:
    kernel = max(1, int(round(window_sec * sfreq)))
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    return F.avg_pool1d(x.square(), kernel_size=kernel, stride=1, padding=pad).clamp_min(1e-12).sqrt()


def sigma_envelope_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    sfreq: float = 100.0,
    sigma_band: tuple[float, float] = (12.0, 16.0),
    envelope_window_sec: float = 0.5,
) -> torch.Tensor:
    """
    Per-sample MSE between smoothed 12-16 Hz RMS envelopes.

    This is a spindle proxy, not a supervised spindle-event loss: it encourages
    reconstructed epochs to preserve sigma bursts without requiring spindle labels.
    """
    xb = _fft_bandpass(x, sfreq, sigma_band[0], sigma_band[1])
    hb = _fft_bandpass(x_hat, sfreq, sigma_band[0], sigma_band[1])
    env_x = _rms_envelope(xb, sfreq, envelope_window_sec)
    env_h = _rms_envelope(hb, sfreq, envelope_window_sec)
    return (env_x - env_h).square().flatten(1).mean(dim=1)


def stage1_losses(
    batch: dict[str, torch.Tensor],
    y: torch.Tensor,
    lambda_rec: float,
    lambda_stage: float,
    lambda_kl: float,
    lambda_spec: float = 0.0,
    lambda_band: float = 0.0,
    lambda_sigma: float = 0.0,
    wake_loss_weight: float = 0.25,
    class_weight: torch.Tensor | None = None,
    sigma_stage_weights: tuple[float, ...] | torch.Tensor | None = None,
    stft_n_fft: int = 256,
    stft_hop_length: int = 64,
    stft_win_length: int = 256,
    y_subwindows: torch.Tensor | None = None,
    subwindow_stage_loss_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_weight = torch.ones_like(y, dtype=batch["x"].dtype, device=batch["x"].device)
    sample_weight = torch.where(y == 0, sample_weight * wake_loss_weight, sample_weight)
    weight_norm = sample_weight.mean().clamp_min(1e-6)

    def weighted_mean(
        loss_per_sample: torch.Tensor,
        extra_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if extra_weight is None:
            return (loss_per_sample * sample_weight).mean() / weight_norm
        total_weight = sample_weight * extra_weight
        return (loss_per_sample * total_weight).mean() / total_weight.mean().clamp_min(1e-6)

    def stage_weight(weights: tuple[float, ...] | torch.Tensor | None) -> torch.Tensor | None:
        if weights is None:
            return None
        weights_t = torch.as_tensor(weights, dtype=batch["x"].dtype, device=batch["x"].device)
        if weights_t.numel() != batch["logits"].size(1):
            raise ValueError(
                f"stage weight length must be {batch['logits'].size(1)}, got {weights_t.numel()}"
            )
        return weights_t[y]

    rec_per_sample = F.mse_loss(batch["x_hat"], batch["x"], reduction="none").flatten(1).mean(1)
    l_rec = weighted_mean(rec_per_sample)
    l_spec = (
        weighted_mean(spectral_loss(
            batch["x"],
            batch["x_hat"],
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            reduction="none",
        ))
        if lambda_spec > 0
        else torch.tensor(0.0, device=batch["x"].device)
    )
    l_band = (
        weighted_mean(bandpower_loss(batch["x"], batch["x_hat"]))
        if lambda_band > 0
        else torch.tensor(0.0, device=batch["x"].device)
    )
    l_sigma = (
        weighted_mean(
            sigma_envelope_loss(batch["x"], batch["x_hat"]),
            extra_weight=stage_weight(sigma_stage_weights),
        )
        if lambda_sigma > 0
        else torch.tensor(0.0, device=batch["x"].device)
    )
    pooled_stage_per_sample = F.cross_entropy(
        batch["logits"],
        y,
        weight=class_weight,
        reduction="none",
    )
    subwindow_stage_weight = float(max(0.0, min(1.0, subwindow_stage_loss_weight)))
    if (
        subwindow_stage_weight > 0
        and y_subwindows is not None
        and "subwindow_logits" in batch
        and batch["subwindow_logits"].numel() > 0
    ):
        sub_logits = batch["subwindow_logits"]
        if y_subwindows.shape != sub_logits.shape[:2]:
            raise ValueError(
                f"y_subwindows shape {tuple(y_subwindows.shape)} does not match "
                f"subwindow logits shape {tuple(sub_logits.shape[:2])}"
            )
        valid_subwindows = y_subwindows >= 0
        stage_per_window = F.cross_entropy(
            sub_logits.reshape(-1, sub_logits.size(-1)),
            y_subwindows.reshape(-1),
            weight=class_weight,
            ignore_index=-1,
            reduction="none",
        ).view_as(y_subwindows)
        valid_count = valid_subwindows.sum(dim=1).clamp_min(1)
        subwindow_stage_per_sample = (
            stage_per_window * valid_subwindows.to(stage_per_window.dtype)
        ).sum(dim=1) / valid_count
        stage_per_sample = (
            (1.0 - subwindow_stage_weight) * pooled_stage_per_sample
            + subwindow_stage_weight * subwindow_stage_per_sample
        )
    else:
        stage_per_sample = pooled_stage_per_sample
    l_stage = weighted_mean(stage_per_sample)
    l_kl = weighted_mean(kl_loss_per_sample(batch["mu"], batch["logvar"]))
    total = (
        lambda_rec * l_rec
        + lambda_spec * l_spec
        + lambda_band * l_band
        + lambda_sigma * l_sigma
        + lambda_stage * l_stage
        + lambda_kl * l_kl
    )
    return total, {
        "rec": l_rec.item(),
        "spec": l_spec.item() if lambda_spec > 0 else 0.0,
        "band": l_band.item() if lambda_band > 0 else 0.0,
        "sigma": l_sigma.item() if lambda_sigma > 0 else 0.0,
        "stage": l_stage.item(),
        "kl": l_kl.item(),
        "total": total.item(),
    }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalLatentDiffusion(nn.Module):
    """Stage 2: predict noise eps given latent context, noisy next latent, and diffusion step."""

    def __init__(
        self,
        latent_dim: int = 128,
        time_dim: int = 128,
        hidden: int = 512,
        context_len: int = 1,
    ):
        super().__init__()
        self.context_len = max(1, int(context_len))
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim * (self.context_len + 1) + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_cond: torch.Tensor, z_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z_cond.dim() == 3:
            z_cond = z_cond.flatten(1)
        t_emb = self.time_mlp(t)
        h = torch.cat([z_cond, z_noisy, t_emb], dim=-1)
        return self.net(h)


class MultiHorizonConditionalLatentDiffusion(nn.Module):
    """Predict noise for multiple future latents in one DDPM reverse process."""

    def __init__(
        self,
        latent_dim: int = 128,
        horizons: int = 3,
        time_dim: int = 128,
        hidden: int = 512,
        context_len: int = 1,
    ):
        super().__init__()
        self.context_len = max(1, int(context_len))
        self.horizons = max(1, int(horizons))
        self.latent_dim = int(latent_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim * (self.context_len + self.horizons) + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim * self.horizons),
        )

    def forward(self, z_cond: torch.Tensor, z_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z_cond.dim() == 3:
            z_cond = z_cond.flatten(1)
        if z_noisy.dim() == 3:
            bsz = z_noisy.size(0)
            z_noisy = z_noisy.flatten(1)
        else:
            bsz = z_noisy.size(0)
        t_emb = self.time_mlp(t)
        h = torch.cat([z_cond, z_noisy, t_emb], dim=-1)
        return self.net(h).view(bsz, self.horizons, self.latent_dim)


class MultiHorizonConditionalFlow(nn.Module):
    """Rectified-flow velocity model for multiple future latents."""

    def __init__(
        self,
        latent_dim: int = 128,
        horizons: int = 3,
        time_dim: int = 128,
        hidden: int = 512,
        context_len: int = 1,
    ):
        super().__init__()
        self.context_len = max(1, int(context_len))
        self.horizons = max(1, int(horizons))
        self.latent_dim = int(latent_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim * (self.context_len + self.horizons) + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim * self.horizons),
        )

    def forward(self, z_cond: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z_cond.dim() == 3:
            z_cond = z_cond.flatten(1)
        if z_t.dim() == 3:
            bsz = z_t.size(0)
            z_t = z_t.flatten(1)
        else:
            bsz = z_t.size(0)
        t_emb = self.time_mlp(t)
        h = torch.cat([z_cond, z_t, t_emb], dim=-1)
        return self.net(h).view(bsz, self.horizons, self.latent_dim)


class DiffusionSchedule:
    def __init__(self, steps: int, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].unsqueeze(-1)
        return torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * noise


@torch.no_grad()
def sample_future_latent(
    diffusion: ConditionalLatentDiffusion,
    schedule: DiffusionSchedule,
    z_cond: torch.Tensor,
) -> torch.Tensor:
    """DDPM reverse: z_t (cond) -> z_hat_{t+1}. Single step, not autoregressive."""
    device = z_cond.device
    b = z_cond.size(0)
    dim = z_cond.size(-1)
    z = torch.randn(b, dim, device=device)
    for step in reversed(range(schedule.steps)):
        t = torch.full((b,), step, device=device, dtype=torch.long)
        eps = diffusion(z_cond, z, t.float())
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bar[step]
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = schedule.betas[step] / torch.sqrt(1.0 - alpha_bar)
        z = coef1 * (z - coef2 * eps)
        if step > 0:
            z = z + torch.sqrt(schedule.betas[step]) * torch.randn_like(z)
    return z


@torch.no_grad()
def sample_future_latents_multi(
    diffusion: MultiHorizonConditionalLatentDiffusion,
    schedule: DiffusionSchedule,
    z_cond: torch.Tensor,
    horizons: int | None = None,
) -> torch.Tensor:
    """DDPM reverse: context -> z_hat_{t+1:t+h} generated jointly."""
    device = z_cond.device
    bsz = z_cond.size(0)
    h = int(horizons or diffusion.horizons)
    dim = diffusion.latent_dim
    z = torch.randn(bsz, h, dim, device=device)
    for step in reversed(range(schedule.steps)):
        t = torch.full((bsz,), step, device=device, dtype=torch.long)
        eps = diffusion(z_cond, z, t.float())
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bar[step]
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = schedule.betas[step] / torch.sqrt(1.0 - alpha_bar)
        z = coef1 * (z - coef2 * eps)
        if step > 0:
            z = z + torch.sqrt(schedule.betas[step]) * torch.randn_like(z)
    return z


@torch.no_grad()
def sample_future_latents_flow(
    flow: MultiHorizonConditionalFlow,
    z_cond: torch.Tensor,
    steps: int = 20,
) -> torch.Tensor:
    """Euler integration for rectified flow: noise -> future latents."""
    device = z_cond.device
    bsz = z_cond.size(0)
    steps = max(1, int(steps))
    z = torch.randn(bsz, flow.horizons, flow.latent_dim, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_value = (i + 0.5) / steps
        t = torch.full((bsz,), t_value, device=device, dtype=z.dtype)
        z = z + dt * flow(z_cond, z, t)
    return z
