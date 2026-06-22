"""Stage 2 multi-horizon rectified flow matching.

This is separate from the DDPM Stage 2 code. It learns a velocity field from
Gaussian noise to the future latent sequence z_{t+1:t+h}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import Stage2Config
from .models import MultiHorizonConditionalFlow, sample_future_latents_flow
from .paths import default_checkpoint_dir
from .train_stage2 import _stage_metrics_from_arrays, _target_stage_weights, encode_all, load_stage1_vae, normalize_latents
from .train_stage2_multi import MultiHorizonEpochDataset


def train_stage2_flow(
    X: np.ndarray,
    subject_ids: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    vae_ckpt: Path,
    y: np.ndarray,
    epoch_onsets: np.ndarray | None = None,
    cfg: Stage2Config | None = None,
    save_dir: Path | None = None,
    device: str | None = None,
    horizons: int = 3,
    inference_steps: int = 20,
) -> tuple[MultiHorizonConditionalFlow, dict]:
    cfg = cfg or Stage2Config()
    horizons = max(1, int(horizons))
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(save_dir or default_checkpoint_dir("stage2_flow"))
    save_dir.mkdir(parents=True, exist_ok=True)

    vae = load_stage1_vae(vae_ckpt, device_t).eval()
    for p in vae.parameters():
        p.requires_grad = False

    Z = encode_all(vae, X, device_t)
    _, z_mean, z_std = normalize_latents(Z, train_idx)
    z_mean_t = torch.as_tensor(z_mean, dtype=torch.float32, device=device_t)
    z_std_t = torch.as_tensor(z_std, dtype=torch.float32, device=device_t).clamp_min(1e-6)

    subj = np.asarray(subject_ids)
    train_ds = MultiHorizonEpochDataset(
        X[train_idx],
        subj[train_idx].tolist(),
        y[train_idx],
        epoch_onsets=epoch_onsets[train_idx] if epoch_onsets is not None else None,
        segment_sec=getattr(cfg, "pair_stride_sec", 30.0),
        context_len=cfg.context_len,
        horizons=horizons,
        transition_pair_weight=cfg.transition_pair_weight,
        transition_context=cfg.transition_context,
        transition_context_weight=cfg.transition_context_weight,
    )
    val_ds = MultiHorizonEpochDataset(
        X[val_idx],
        subj[val_idx].tolist(),
        y[val_idx],
        epoch_onsets=epoch_onsets[val_idx] if epoch_onsets is not None else None,
        segment_sec=getattr(cfg, "pair_stride_sec", 30.0),
        context_len=cfg.context_len,
        horizons=horizons,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Stage2 flow needs consecutive windows. train={len(train_ds)} val={len(val_ds)}")

    sampler = None
    shuffle = True
    if cfg.transition_weighted_sampling and y is not None:
        sampler = WeightedRandomSampler(train_ds.sample_weights, num_samples=len(train_ds), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    latent_dim = vae.encoder.fc_mu.out_features
    num_classes = vae.stage_classifier.net[-1].out_features
    model = MultiHorizonConditionalFlow(
        latent_dim=latent_dim,
        horizons=horizons,
        time_dim=cfg.time_dim,
        hidden=cfg.hidden_dim,
        context_len=cfg.context_len,
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    target_stage_weight = _target_stage_weights(y[train_idx], num_classes, device_t) if cfg.use_target_stage_loss_weights else None
    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")

    print(
        f"Stage2 FLOW training start: device={device_t}, epochs={cfg.epochs}, "
        f"horizons={horizons}, context_len={cfg.context_len}, "
        f"train_windows={len(train_ds)}, val_windows={len(val_ds)}",
        flush=True,
    )
    print(
        "Stage2 FLOW method: rectified_flow linear path "
        f"loss=lambda_diff*velocity_mse + lambda_next_stage*CE, "
        f"lambda_diff={cfg.lambda_diff}, lambda_next_stage={cfg.lambda_next_stage}",
        flush=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_flow = 0.0
        train_ce = 0.0
        train_iter = tqdm(train_loader, desc=f"Stage2 flow epoch {epoch:03d}/{cfg.epochs} train", leave=False)
        for x_ctx, x_future, _, y_future in train_iter:
            x_ctx = x_ctx.to(device_t)
            x_future = x_future.to(device_t)
            y_future = y_future.to(device_t)
            bsz, context_len = x_ctx.shape[:2]
            x_ctx_flat = x_ctx.reshape(bsz * context_len, *x_ctx.shape[2:])
            x_fut_flat = x_future.reshape(bsz * horizons, *x_future.shape[2:])
            with torch.no_grad():
                z_ctx = (vae.encoder(x_ctx_flat)[0].view(bsz, context_len, -1) - z_mean_t) / z_std_t
                z1 = (vae.encoder(x_fut_flat)[0].view(bsz, horizons, -1) - z_mean_t) / z_std_t
            z0 = torch.randn_like(z1)
            t = torch.rand(bsz, device=device_t)
            t_view = t.view(bsz, 1, 1)
            z_t = (1.0 - t_view) * z0 + t_view * z1
            target_v = z1 - z0
            pred_v = model(z_ctx, z_t, t)
            loss_per = F.mse_loss(pred_v, target_v, reduction="none").mean(dim=(1, 2))
            if target_stage_weight is not None:
                sample_weight = target_stage_weight[y_future].mean(dim=1)
            else:
                sample_weight = torch.ones_like(loss_per)
            flow_loss = (loss_per * sample_weight).mean() / sample_weight.mean().clamp_min(1e-6)
            if cfg.lambda_next_stage > 0:
                z1_hat = z_t + (1.0 - t_view) * pred_v
                logits = vae.stage_classifier((z1_hat * z_std_t + z_mean_t).reshape(bsz * horizons, -1))
                ce_per = F.cross_entropy(logits, y_future.reshape(-1), reduction="none").view(bsz, horizons).mean(dim=1)
                next_stage_loss = (ce_per * sample_weight).mean() / sample_weight.mean().clamp_min(1e-6)
            else:
                next_stage_loss = torch.tensor(0.0, device=device_t)
            loss = cfg.lambda_diff * flow_loss + cfg.lambda_next_stage * next_stage_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_flow += flow_loss.item()
            train_ce += next_stage_loss.item()
            train_iter.set_postfix(total=f"{loss.item():.3f}", flow=f"{flow_loss.item():.3f}", next=f"{next_stage_loss.item():.3f}")
        train_loss /= max(len(train_loader), 1)
        train_flow /= max(len(train_loader), 1)
        train_ce /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_flow = 0.0
        val_ce = 0.0
        true_by_h = [[] for _ in range(horizons)]
        pred_by_h = [[] for _ in range(horizons)]
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Stage2 flow epoch {epoch:03d}/{cfg.epochs} val", leave=False)
            for x_ctx, x_future, _, y_future in val_iter:
                x_ctx = x_ctx.to(device_t)
                x_future = x_future.to(device_t)
                y_future = y_future.to(device_t)
                bsz, context_len = x_ctx.shape[:2]
                x_ctx_flat = x_ctx.reshape(bsz * context_len, *x_ctx.shape[2:])
                x_fut_flat = x_future.reshape(bsz * horizons, *x_future.shape[2:])
                z_ctx = (vae.encoder(x_ctx_flat)[0].view(bsz, context_len, -1) - z_mean_t) / z_std_t
                z1 = (vae.encoder(x_fut_flat)[0].view(bsz, horizons, -1) - z_mean_t) / z_std_t
                z0 = torch.randn_like(z1)
                t = torch.rand(bsz, device=device_t)
                t_view = t.view(bsz, 1, 1)
                z_t = (1.0 - t_view) * z0 + t_view * z1
                target_v = z1 - z0
                pred_v = model(z_ctx, z_t, t)
                flow_loss = F.mse_loss(pred_v, target_v)
                z1_hat = z_t + (1.0 - t_view) * pred_v
                logits = vae.stage_classifier((z1_hat * z_std_t + z_mean_t).reshape(bsz * horizons, -1))
                ce_loss = F.cross_entropy(logits, y_future.reshape(-1))
                loss = cfg.lambda_diff * flow_loss + cfg.lambda_next_stage * ce_loss
                val_loss += loss.item()
                val_flow += flow_loss.item()
                val_ce += ce_loss.item()

                z_sample = sample_future_latents_flow(model, z_ctx, steps=inference_steps)
                sample_logits = vae.stage_classifier((z_sample * z_std_t + z_mean_t).reshape(bsz * horizons, -1))
                pred_stage = sample_logits.argmax(dim=1).view(bsz, horizons).cpu().numpy()
                true_stage = y_future.cpu().numpy()
                for h in range(horizons):
                    true_by_h[h].append(true_stage[:, h])
                    pred_by_h[h].append(pred_stage[:, h])
        val_loss /= max(len(val_loader), 1)
        val_flow /= max(len(val_loader), 1)
        val_ce /= max(len(val_loader), 1)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        per_h_msg = []
        for h in range(horizons):
            t_arr = np.concatenate(true_by_h[h]) if true_by_h[h] else np.array([], dtype=np.int64)
            p_arr = np.concatenate(pred_by_h[h]) if pred_by_h[h] else np.array([], dtype=np.int64)
            m = _stage_metrics_from_arrays(t_arr, p_arr, num_classes)
            per_h_msg.append(f"h{h + 1}_acc={m['acc']:.3f} h{h + 1}_mf1={m['macro_f1']:.3f}")
        print(
            f"Stage2 flow epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} "
            f"(flow train={train_flow:.4f} val={val_flow:.4f} nextCE train={train_ce:.4f} val={val_ce:.4f}) "
            + " ".join(per_h_msg),
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "flow_method": "rectified_flow_linear",
                    "horizons": horizons,
                    "inference_steps": int(inference_steps),
                    "latent_mean": z_mean,
                    "latent_std": z_std,
                    "latent_normalized": True,
                    "context_len": cfg.context_len,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_dir / "flow_multi.pt",
            )
    return model, history
