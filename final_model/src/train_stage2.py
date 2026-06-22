"""Stage 2: EMA-teacher VAE encoder + conditional latent diffusion."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import Stage2Config
from .classification_stage1 import ClassificationStage1, load_stage1_classifier
from .models import ConditionalLatentDiffusion, DiffusionSchedule, MultiHeadVAE
from .paths import default_checkpoint_dir


class EpochPairDataset(torch.utils.data.Dataset):
    """Consecutive epochs: context x_{t-k+1:t} predicts x_{t+1}."""

    def __init__(
        self,
        X: np.ndarray,
        subject_ids: list[str],
        y: np.ndarray | None = None,
        epoch_onsets: np.ndarray | None = None,
        segment_sec: float | tuple[float, ...] = 30.0,
        onset_tolerance_sec: float = 1e-3,
        transition_pair_weight: float = 5.0,
        transition_context: int = 1,
        transition_context_weight: float = 2.0,
        context_len: int = 5,
    ):
        self.pairs = []
        self.transition_flags = []
        self.context_flags = []
        self.label_pairs = []
        self.sample_weights = []
        if X.ndim == 3 and X.shape[1] != 1:
            X = X.reshape(X.shape[0], -1)
        X_t = torch.from_numpy(X[:, None, :]).float()
        subj = np.array(subject_ids)
        labels = np.asarray(y) if y is not None else None
        onsets = np.asarray(epoch_onsets, dtype=np.float64) if epoch_onsets is not None else None
        allowed_gaps = np.atleast_1d(np.asarray(segment_sec, dtype=np.float64))
        context_len = max(1, int(context_len))
        for sid in np.unique(subj):
            idx = np.where(subj == sid)[0]
            if len(idx) <= context_len:
                continue
            idx = np.sort(idx)
            local_pairs = []
            for pos in range(context_len - 1, len(idx) - 1):
                ctx = idx[pos - context_len + 1 : pos + 1]
                b = idx[pos + 1]
                if onsets is not None:
                    gaps = np.diff(onsets[np.concatenate([ctx, [b]])])
                    gap_ok = np.any(np.abs(gaps[:, None] - allowed_gaps[None, :]) <= onset_tolerance_sec, axis=1)
                    if not bool(np.all(gap_ok)):
                        continue
                local_pairs.append((pos, ctx, b))
            transition_pair_positions: set[int] = set()
            if labels is not None:
                for pos, ctx, b in local_pairs:
                    if int(labels[ctx[-1]]) != int(labels[b]):
                        transition_pair_positions.add(pos)
            for pos, ctx, b in local_pairs:
                is_transition = labels is not None and int(labels[ctx[-1]]) != int(labels[b])
                near_transition = False
                if labels is not None and transition_context >= 0:
                    near_transition = any(
                        abs(pos - tpos) <= transition_context
                        for tpos in transition_pair_positions
                    )
                self.pairs.append((X_t[ctx], X_t[b]))
                self.transition_flags.append(bool(is_transition))
                if is_transition:
                    self.context_flags.append(False)
                    self.sample_weights.append(float(transition_pair_weight))
                elif near_transition:
                    self.context_flags.append(True)
                    self.sample_weights.append(float(transition_context_weight))
                else:
                    self.context_flags.append(False)
                    self.sample_weights.append(1.0)
                if labels is None:
                    self.label_pairs.append((-1, -1))
                else:
                    self.label_pairs.append((int(labels[ctx[-1]]), int(labels[b])))
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.double)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        x_ctx, x_tp1 = self.pairs[i]
        y_t, y_tp1 = self.label_pairs[i]
        update_encoder = self.transition_flags[i] or self.context_flags[i]
        return (
            x_ctx,
            x_tp1,
            torch.tensor(y_t, dtype=torch.long),
            torch.tensor(y_tp1, dtype=torch.long),
            torch.tensor(update_encoder, dtype=torch.bool),
        )


def _stage_metrics_from_arrays(true: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:
    if len(true) == 0:
        return {"acc": float("nan"), "balanced_acc": float("nan"), "macro_f1": float("nan")}
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    recalls = []
    f1s = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        if support > 0:
            recalls.append(tp / support)
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return {
        "acc": float((true == pred).mean()),
        "balanced_acc": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _target_stage_weights(
    y_target: np.ndarray | None,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    if y_target is None:
        return None
    counts = np.bincount(y_target, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _stage_balanced_pair_weights(ds: EpochPairDataset, num_classes: int) -> torch.Tensor:
    targets = np.asarray([pair[1] for pair in ds.label_pairs], dtype=np.int64)
    valid = (targets >= 0) & (targets < num_classes)
    if not np.any(valid):
        return torch.ones(len(ds), dtype=torch.double)
    counts = np.bincount(targets[valid], minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.zeros(len(ds), dtype=np.float64)
    weights[valid] = 1.0 / counts[targets[valid]]
    weights[~valid] = 1.0
    return torch.tensor(weights, dtype=torch.double)


@torch.no_grad()
def _update_ema_module(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)


def load_stage1_vae(ckpt_path: Path, device: torch.device) -> MultiHeadVAE | ClassificationStage1:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except (EOFError, RuntimeError, OSError) as exc:
        size = ckpt_path.stat().st_size if ckpt_path.exists() else 0
        raise RuntimeError(
            "Could not load Stage 1 VAE checkpoint. "
            f"Path: {ckpt_path} (size={size} bytes). "
            "If PyTorch reports a missing central directory, the .pt file is "
            "usually truncated/corrupt or still syncing. Re-copy or re-train "
            "that Stage 1 fold, or pass a different intact checkpoint with "
            "--vae-ckpt."
        ) from exc
    cfg = ckpt["cfg"]
    model = MultiHeadVAE(
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
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        msg = str(exc)
        if "recon_decoder" not in msg and "Missing key(s)" not in msg:
            raise
        return load_stage1_classifier(ckpt_path, device)
    return model


def load_frozen_vae(ckpt_path: Path, device: torch.device) -> MultiHeadVAE | ClassificationStage1:
    model = load_stage1_vae(ckpt_path, device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def _train_encoder_only(model: MultiHeadVAE | ClassificationStage1) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = True
    model.train()


@torch.no_grad()
def encode_all(vae: MultiHeadVAE | ClassificationStage1, X: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    if X.ndim == 3 and X.shape[1] != 1:
        X = X.reshape(X.shape[0], -1)
    X = X[:, None, :]
    zs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        mu, _ = vae.encoder(xb)
        zs.append(mu.cpu().numpy())
    return np.concatenate(zs, axis=0)


def normalize_latents(
    Z: np.ndarray,
    train_idx: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize VAE latents using train split statistics for stable diffusion training."""
    z_mean = Z[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    z_std = np.maximum(Z[train_idx].std(axis=0, keepdims=True), eps).astype(np.float32)
    return ((Z - z_mean) / z_std).astype(np.float32), z_mean.squeeze(0), z_std.squeeze(0)


def train_stage2(
    X: np.ndarray,
    subject_ids: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    vae_ckpt: Path,
    y: np.ndarray | None = None,
    epoch_onsets: np.ndarray | None = None,
    cfg: Stage2Config | None = None,
    save_dir: Path | None = None,
    device: str | None = None,
) -> tuple[ConditionalLatentDiffusion, dict]:
    cfg = cfg or Stage2Config()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(save_dir or default_checkpoint_dir("stage2"))
    save_dir.mkdir(parents=True, exist_ok=True)

    vae = load_stage1_vae(vae_ckpt, device_t)
    _train_encoder_only(vae)
    ema_vae = copy.deepcopy(vae).eval()
    for p in ema_vae.parameters():
        p.requires_grad = False

    Z = encode_all(ema_vae, X, device_t)
    Z, z_mean, z_std = normalize_latents(Z, train_idx)
    z_mean_t = torch.as_tensor(z_mean, dtype=torch.float32, device=device_t)
    z_std_t = torch.as_tensor(z_std, dtype=torch.float32, device=device_t)

    subj = np.array(subject_ids)
    y_train = y[train_idx] if y is not None else None
    y_val = y[val_idx] if y is not None else None
    onsets_train = epoch_onsets[train_idx] if epoch_onsets is not None else None
    onsets_val = epoch_onsets[val_idx] if epoch_onsets is not None else None
    train_ds = EpochPairDataset(
        X[train_idx],
        subj[train_idx].tolist(),
        y=y_train,
        epoch_onsets=onsets_train,
        segment_sec=getattr(cfg, "pair_stride_sec", 30.0),
        transition_pair_weight=cfg.transition_pair_weight,
        transition_context=cfg.transition_context,
        transition_context_weight=cfg.transition_context_weight,
        context_len=cfg.context_len,
    )
    val_ds = EpochPairDataset(
        X[val_idx],
        subj[val_idx].tolist(),
        y=y_val,
        epoch_onsets=onsets_val,
        segment_sec=getattr(cfg, "pair_stride_sec", 30.0),
        context_len=cfg.context_len,
    )
    sampler = None
    shuffle = True
    num_classes = ema_vae.stage_classifier.net[-1].out_features
    sampling = getattr(cfg, "sampling", "transition")
    if sampling == "stage_balanced" and y is not None:
        sampler = WeightedRandomSampler(
            weights=_stage_balanced_pair_weights(train_ds, num_classes),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False
    elif cfg.transition_weighted_sampling and y is not None:
        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Stage2 needs consecutive latent pairs. train_pairs={len(train_ds)} val_pairs={len(val_ds)}"
        )

    model = ConditionalLatentDiffusion(
        latent_dim=vae.encoder.fc_mu.out_features,
        time_dim=cfg.time_dim,
        hidden=cfg.hidden_dim,
        context_len=cfg.context_len,
    ).to(device_t)
    ema_model = copy.deepcopy(model).eval() if cfg.use_ema else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad = False
    schedule = DiffusionSchedule(cfg.diffusion_steps, device=str(device_t))
    opt_params: list[dict] = [{"params": model.parameters(), "lr": cfg.lr}]
    if cfg.train_encoder_near_transition:
        opt_params.append({"params": vae.encoder.parameters(), "lr": cfg.encoder_lr})
    opt = torch.optim.AdamW(opt_params)
    target_stage_weight = (
        _target_stage_weights(y_train, num_classes, device_t)
        if cfg.use_target_stage_loss_weights
        else None
    )
    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")

    print(
        f"Stage2 training start: device={device_t}, epochs={cfg.epochs}, "
        f"train_pairs={len(train_ds)}, val_pairs={len(val_ds)}, "
        f"context_len={cfg.context_len}, "
        f"latent_dim={Z.shape[1]}, z_std_mean={float(z_std.mean()):.4f}",
        flush=True,
    )
    print(
        f"Stage2 loss weights: lambda_diff={cfg.lambda_diff} "
        f"lambda_next_stage={cfg.lambda_next_stage} "
        f"transition_wake_target_weight={cfg.transition_wake_target_weight}"
    )
    print(
        f"Stage2 sampling: {sampling}; "
        f"target_stage_loss_weights={cfg.use_target_stage_loss_weights}",
        flush=True,
    )
    print(
        f"Stage2 encoder tuning: enabled={cfg.train_encoder_near_transition} "
        f"encoder_lr={cfg.encoder_lr} lambda_transition_ema={cfg.lambda_transition_ema}",
        flush=True,
    )
    if y is not None:
        n_transition = int(sum(train_ds.transition_flags))
        n_context = int(sum(train_ds.context_flags))
        n_normal = len(train_ds) - n_transition - n_context
        weighted_transition = n_transition * cfg.transition_pair_weight
        weighted_context = n_context * cfg.transition_context_weight
        weighted_normal = n_normal
        weighted_total = max(weighted_transition + weighted_context + weighted_normal, 1e-12)
        print(
            f"Stage2 transition sampling: enabled={cfg.transition_weighted_sampling}, "
            f"transition_pairs={n_transition}/{len(train_ds)} "
            f"({n_transition / max(len(train_ds), 1):.3f}), "
            f"transition_weight={cfg.transition_pair_weight}, "
            f"context={cfg.transition_context}, context_weight={cfg.transition_context_weight}",
            flush=True,
        )
        print(
            "Stage2 pair mix: "
            f"exact={n_transition} ({n_transition / max(len(train_ds), 1):.3f}, "
            f"weighted≈{weighted_transition / weighted_total:.3f}) | "
            f"context={n_context} ({n_context / max(len(train_ds), 1):.3f}, "
            f"weighted≈{weighted_context / weighted_total:.3f}) | "
            f"normal={n_normal} ({n_normal / max(len(train_ds), 1):.3f}, "
            f"weighted≈{weighted_normal / weighted_total:.3f})",
            flush=True,
        )
    if target_stage_weight is not None:
        print(
            "Stage2 target-stage loss weights:",
            {
                name: round(float(weight), 3)
                for name, weight in zip(("W", "N1", "N2", "N3", "REM"), target_stage_weight.cpu())
            },
            flush=True,
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        vae.train()
        ema_vae.eval()
        train_loss = 0.0
        train_diff_loss = 0.0
        train_next_stage_loss = 0.0
        train_transition_ema_loss = 0.0
        train_iter = tqdm(
            train_loader,
            desc=f"Stage2 epoch {epoch:03d}/{cfg.epochs} train",
            leave=False,
        )
        for x_ctx, x_tp1, y_t, y_tp1, update_encoder in train_iter:
            x_ctx = x_ctx.to(device_t)
            x_tp1 = x_tp1.to(device_t)
            y_t = y_t.to(device_t)
            y_tp1 = y_tp1.to(device_t)
            update_encoder = update_encoder.to(device_t) & bool(cfg.train_encoder_near_transition)
            bsz, context_len = x_ctx.shape[:2]
            x_ctx_flat = x_ctx.reshape(bsz * context_len, *x_ctx.shape[2:])
            z_ctx_student = (vae.encoder(x_ctx_flat)[0].view(bsz, context_len, -1) - z_mean_t) / z_std_t
            z_tp1_student = (vae.encoder(x_tp1)[0] - z_mean_t) / z_std_t
            update_mask = update_encoder.unsqueeze(-1)
            z_ctx = torch.where(update_mask.unsqueeze(-1), z_ctx_student, z_ctx_student.detach())
            with torch.no_grad():
                z_ctx_teacher = (ema_vae.encoder(x_ctx_flat)[0].view(bsz, context_len, -1) - z_mean_t) / z_std_t
                z_tp1_teacher = (ema_vae.encoder(x_tp1)[0] - z_mean_t) / z_std_t
            z_tp1 = z_tp1_student.detach()
            t = torch.randint(0, cfg.diffusion_steps, (z_tp1.size(0),), device=device_t)
            noise = torch.randn_like(z_tp1)
            z_noisy = schedule.q_sample(z_tp1, t, noise)
            pred = model(z_ctx, z_noisy, t.float())
            loss_per_sample = F.mse_loss(pred, noise, reduction="none").mean(dim=1)
            if target_stage_weight is not None:
                sample_weight = target_stage_weight[y_tp1]
            else:
                sample_weight = torch.ones_like(loss_per_sample)
            if cfg.transition_wake_target_weight != 1.0:
                transition_to_wake = (y_t != y_tp1) & (y_tp1 == 0)
                wake_boost = torch.where(
                    transition_to_wake,
                    torch.full_like(sample_weight, float(cfg.transition_wake_target_weight)),
                    torch.ones_like(sample_weight),
                )
                sample_weight = sample_weight * wake_boost
            diff_loss = (loss_per_sample * sample_weight).mean() / sample_weight.mean().clamp_min(1e-6)
            if cfg.lambda_next_stage > 0:
                ab = schedule.alpha_bar[t].unsqueeze(-1)
                z0_hat = (z_noisy - torch.sqrt(1 - ab) * pred) / torch.sqrt(ab).clamp_min(1e-6)
                logits = ema_vae.stage_classifier(z0_hat * z_std_t + z_mean_t)
                ce_per_sample = F.cross_entropy(logits, y_tp1, reduction="none")
                next_stage_loss = (ce_per_sample * sample_weight).mean() / sample_weight.mean().clamp_min(1e-6)
            else:
                next_stage_loss = torch.tensor(0.0, device=device_t)
            if cfg.lambda_transition_ema > 0:
                transition_mask = update_encoder
                if transition_mask.any():
                    transition_ema_loss = 0.5 * (
                        F.mse_loss(z_ctx_student[transition_mask], z_ctx_teacher[transition_mask])
                        + F.mse_loss(z_tp1_student[transition_mask], z_tp1_teacher[transition_mask])
                    )
                else:
                    transition_ema_loss = torch.tensor(0.0, device=device_t)
            else:
                transition_ema_loss = torch.tensor(0.0, device=device_t)
            loss = (
                cfg.lambda_diff * diff_loss
                + cfg.lambda_next_stage * next_stage_loss
                + cfg.lambda_transition_ema * transition_ema_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema_model is not None:
                _update_ema_module(ema_model, model, cfg.ema_decay)
            if cfg.vae_ema_decay > 0:
                _update_ema_module(ema_vae, vae, cfg.vae_ema_decay)
            train_loss += loss.item()
            train_diff_loss += diff_loss.item()
            train_next_stage_loss += next_stage_loss.item()
            train_transition_ema_loss += transition_ema_loss.item()
            train_iter.set_postfix(
                total=f"{loss.item():.3f}",
                diff=f"{diff_loss.item():.3f}",
                next=f"{next_stage_loss.item():.3f}",
                ema=f"{transition_ema_loss.item():.3f}",
            )
        train_loss /= max(len(train_loader), 1)
        train_diff_loss /= max(len(train_loader), 1)
        train_next_stage_loss /= max(len(train_loader), 1)
        train_transition_ema_loss /= max(len(train_loader), 1)

        model.eval()
        vae.eval()
        ema_vae.eval()
        eval_model = ema_model if ema_model is not None else model
        val_loss = 0.0
        val_diff_loss = 0.0
        val_next_stage_loss = 0.0
        with torch.no_grad():
            stage_true = []
            stage_pred = []
            transition_mask = []
            val_iter = tqdm(
                val_loader,
                desc=f"Stage2 epoch {epoch:03d}/{cfg.epochs} val",
                leave=False,
            )
            for x_ctx, x_tp1, y_t, y_tp1, _ in val_iter:
                x_ctx = x_ctx.to(device_t)
                x_tp1 = x_tp1.to(device_t)
                bsz, context_len = x_ctx.shape[:2]
                x_ctx_flat = x_ctx.reshape(bsz * context_len, *x_ctx.shape[2:])
                z_ctx = (vae.encoder(x_ctx_flat)[0].view(bsz, context_len, -1) - z_mean_t) / z_std_t
                z_tp1 = (ema_vae.encoder(x_tp1)[0] - z_mean_t) / z_std_t
                t = torch.randint(0, cfg.diffusion_steps, (z_tp1.size(0),), device=device_t)
                noise = torch.randn_like(z_tp1)
                z_noisy = schedule.q_sample(z_tp1, t, noise)
                pred = eval_model(z_ctx, z_noisy, t.float())
                diff_loss_val = F.mse_loss(pred, noise)
                val_diff_loss += diff_loss_val.item()
                if y is not None:
                    z_hat = (z_noisy - torch.sqrt(1 - schedule.alpha_bar[t]).unsqueeze(-1) * pred) / (
                        torch.sqrt(schedule.alpha_bar[t]).unsqueeze(-1).clamp_min(1e-6)
                    )
                    logits = ema_vae.stage_classifier(z_hat * z_std_t + z_mean_t)
                    next_stage_loss_val = F.cross_entropy(logits, y_tp1.to(device_t))
                    val_next_stage_loss += next_stage_loss_val.item()
                    val_loss += (cfg.lambda_diff * diff_loss_val + cfg.lambda_next_stage * next_stage_loss_val).item()
                    stage_true.append(y_tp1.numpy())
                    stage_pred.append(logits.argmax(dim=1).cpu().numpy())
                    transition_mask.append((y_t.numpy() != y_tp1.numpy()))
                    val_iter.set_postfix(
                        loss=f"{(cfg.lambda_diff * diff_loss_val + cfg.lambda_next_stage * next_stage_loss_val).item():.3f}",
                        diff=f"{diff_loss_val.item():.3f}",
                        next=f"{next_stage_loss_val.item():.3f}",
                    )
                else:
                    val_loss += (cfg.lambda_diff * diff_loss_val).item()
                    val_iter.set_postfix(loss=f"{(cfg.lambda_diff * diff_loss_val).item():.3f}")
        val_loss /= max(len(val_loader), 1)
        val_diff_loss /= max(len(val_loader), 1)
        val_next_stage_loss /= max(len(val_loader), 1)
        val_stage = None
        if y is not None and stage_true:
            true_arr = np.concatenate(stage_true)
            pred_arr = np.concatenate(stage_pred)
            trans_arr = np.concatenate(transition_mask).astype(bool)
            val_stage = _stage_metrics_from_arrays(true_arr, pred_arr, ema_vae.stage_classifier.net[-1].out_features)
            stable_stage = _stage_metrics_from_arrays(true_arr[~trans_arr], pred_arr[~trans_arr], ema_vae.stage_classifier.net[-1].out_features)
            transition_stage = _stage_metrics_from_arrays(true_arr[trans_arr], pred_arr[trans_arr], ema_vae.stage_classifier.net[-1].out_features)
            val_stage.update(
                {
                    "stable_acc": stable_stage["acc"],
                    "transition_acc": transition_stage["acc"],
                    "n_stable": int((~trans_arr).sum()),
                    "n_transition": int(trans_arr.sum()),
                }
            )
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        msg = (
            f"Stage2 epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} "
            f"(diff train={train_diff_loss:.4f} val={val_diff_loss:.4f} "
            f"nextCE train={train_next_stage_loss:.4f} val={val_next_stage_loss:.4f} "
            f"transEMA train={train_transition_ema_loss:.4f}) "
            f"(zero-noise baseline≈1.0000)"
        )
        if ema_model is not None:
            msg += f" ema_decay={cfg.ema_decay}"
        msg += f" vae_ema_decay={cfg.vae_ema_decay}"
        if val_stage is not None:
            msg += (
                f" | next-stage acc={val_stage['acc']:.3f} "
                f"bal={val_stage['balanced_acc']:.3f} mf1={val_stage['macro_f1']:.3f} "
                f"stable_acc={val_stage['stable_acc']:.3f} "
                f"transition_acc={val_stage['transition_acc']:.3f} "
                f"(n_trans={val_stage['n_transition']})"
            )
        print(msg, flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict() if ema_model is not None else None,
                    "used_ema_for_val": ema_model is not None,
                    "vae_model": vae.state_dict(),
                    "ema_vae_model": ema_vae.state_dict(),
                    "used_vae_ema_teacher": True,
                    "cfg": cfg,
                    "latent_mean": z_mean,
                    "latent_std": z_std,
                    "latent_normalized": True,
                    "context_len": cfg.context_len,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_dir / "diffusion.pt",
            )
    return model, history
