"""
Stage 3: Inference (non-autoregressive).

Present EEG x_t -> Stage 2 EMA-teacher encoder -> z_t
z_t + diffusion (single step) -> z_hat_{t+1}
z_hat_{t+1} -> frozen stage classifier (Stage 1) -> s_hat_{t+1}

Autoregressive rollout (z_hat chain) is NOT used.
Multi-horizon Acc_n: each step i uses real EEG at t+i-1 as present, predicts s_{t+i}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import InferenceConfig, Stage2Config
from .models import (
    ConditionalLatentDiffusion,
    DiffusionSchedule,
    MultiHeadVAE,
    MultiHorizonConditionalLatentDiffusion,
    MultiHorizonConditionalFlow,
    sample_future_latent,
    sample_future_latents_multi,
    sample_future_latents_flow,
)
from .paths import default_checkpoint_dir
from .preprocess import STAGE_NAMES
from .train_stage2 import load_frozen_vae


def _metrics_from_confusion(cm: np.ndarray) -> dict:
    correct = int(np.trace(cm))
    total = int(cm.sum())
    recalls = []
    f1s = []
    per_class_acc = {}
    false_alarm_rates = []
    per_class_false_alarm = {}
    for i, name in enumerate(STAGE_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        support = cm[i, :].sum()
        recall = float(tp / support) if support > 0 else float("nan")
        per_class_acc[name] = recall
        false_alarm = float(fp / (fp + tn)) if fp + tn > 0 else float("nan")
        per_class_false_alarm[name] = false_alarm
        if not np.isnan(false_alarm):
            false_alarm_rates.append(false_alarm)
        if support > 0:
            recalls.append(recall)
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(2 * precision * rec / (precision + rec) if precision + rec > 0 else 0.0)
    return {
        "accuracy": correct / max(total, 1),
        "balanced_acc": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "macro_false_alarm": float(np.mean(false_alarm_rates)) if false_alarm_rates else 0.0,
        "per_class_acc": per_class_acc,
        "per_class_false_alarm": per_class_false_alarm,
        "confusion_matrix": cm,
        "n_pairs": total,
    }


def _binary_metrics_from_confusion(cm: np.ndarray) -> dict:
    correct = int(np.trace(cm))
    total = int(cm.sum())
    f1s = []
    per_class_acc = {}
    for i, name in enumerate(("Wake", "Sleep")):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        per_class_acc[name] = float(tp / support) if support > 0 else float("nan")
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "n_pairs": total,
    }


def _direction_metrics(
    w2s_cms: dict[int, np.ndarray],
    s2w_cms: dict[int, np.ndarray],
    w2s_binary_cms: dict[int, np.ndarray],
    s2w_binary_cms: dict[int, np.ndarray],
    n: int,
) -> dict:
    return {
        "wake_to_sleep": {
            i: _metrics_from_confusion(w2s_cms[i]) for i in range(1, n + 1)
        },
        "sleep_to_wake": {
            i: _metrics_from_confusion(s2w_cms[i]) for i in range(1, n + 1)
        },
        "binary_wake_to_sleep": {
            i: _binary_metrics_from_confusion(w2s_binary_cms[i]) for i in range(1, n + 1)
        },
        "binary_sleep_to_wake": {
            i: _binary_metrics_from_confusion(s2w_binary_cms[i]) for i in range(1, n + 1)
        },
    }


def load_diffusion(
    ckpt_path: Path,
    latent_dim: int,
    device: torch.device,
) -> tuple[ConditionalLatentDiffusion, Stage2Config, torch.Tensor, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = ConditionalLatentDiffusion(
        latent_dim=latent_dim,
        time_dim=getattr(cfg, "time_dim", 128),
        hidden=getattr(cfg, "hidden_dim", 512),
        context_len=getattr(cfg, "context_len", ckpt.get("context_len", 1)),
    ).to(device)
    state_key = "ema_model" if ckpt.get("ema_model") is not None else "model"
    model.load_state_dict(ckpt[state_key])
    if state_key == "ema_model":
        print("Loaded Stage2 diffusion EMA weights for inference.", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    latent_mean = torch.as_tensor(
        ckpt.get("latent_mean", np.zeros(latent_dim, dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    latent_std = torch.as_tensor(
        ckpt.get("latent_std", np.ones(latent_dim, dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-6)
    return model, cfg, latent_mean, latent_std


def load_diffusion_multi(
    ckpt_path: Path,
    latent_dim: int,
    device: torch.device,
) -> tuple[MultiHorizonConditionalLatentDiffusion, Stage2Config, torch.Tensor, torch.Tensor, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    horizons = int(ckpt.get("horizons", 3))
    model = MultiHorizonConditionalLatentDiffusion(
        latent_dim=latent_dim,
        horizons=horizons,
        time_dim=getattr(cfg, "time_dim", 128),
        hidden=getattr(cfg, "hidden_dim", 512),
        context_len=getattr(cfg, "context_len", ckpt.get("context_len", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    latent_mean = torch.as_tensor(
        ckpt.get("latent_mean", np.zeros(latent_dim, dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    latent_std = torch.as_tensor(
        ckpt.get("latent_std", np.ones(latent_dim, dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-6)
    return model, cfg, latent_mean, latent_std, horizons


def load_flow_multi(
    ckpt_path: Path,
    latent_dim: int,
    device: torch.device,
) -> tuple[MultiHorizonConditionalFlow, Stage2Config, torch.Tensor, torch.Tensor, int, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    horizons = int(ckpt.get("horizons", 3))
    model = MultiHorizonConditionalFlow(
        latent_dim=latent_dim,
        horizons=horizons,
        time_dim=getattr(cfg, "time_dim", 128),
        hidden=getattr(cfg, "hidden_dim", 512),
        context_len=getattr(cfg, "context_len", ckpt.get("context_len", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    latent_mean = torch.as_tensor(
        ckpt.get("latent_mean", np.zeros(latent_dim, dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    latent_std = torch.as_tensor(
        ckpt.get("latent_std", np.ones(latent_dim, dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-6)
    inference_steps = int(ckpt.get("inference_steps", 20))
    return model, cfg, latent_mean, latent_std, horizons, inference_steps


class SleepInferencePipeline:
    """Frozen VAE encoder + stage head; trainable diffusion for z_{t+1}."""

    def __init__(
        self,
        vae: MultiHeadVAE,
        diffusion: ConditionalLatentDiffusion,
        schedule: DiffusionSchedule,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        self.vae = vae
        self.diffusion = diffusion
        self.schedule = schedule
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent_dim = self.vae.encoder.fc_mu.out_features
        self.latent_mean = latent_mean if latent_mean is not None else torch.zeros(latent_dim)
        self.latent_std = latent_std if latent_std is not None else torch.ones(latent_dim)
        self.latent_mean = self.latent_mean.to(self.device)
        self.latent_std = self.latent_std.to(self.device).clamp_min(1e-6)
        self.vae.to(self.device).eval()
        self.diffusion.to(self.device).eval()
        self.context_len = getattr(self.diffusion, "context_len", 1)

    @classmethod
    def from_checkpoints(
        cls,
        vae_ckpt: Path | str,
        diffusion_ckpt: Path | str | None = None,
        device: str | None = None,
    ) -> SleepInferencePipeline:
        device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        diff_path = Path(diffusion_ckpt or default_checkpoint_dir("stage2") / "diffusion.pt")
        vae = load_frozen_vae(Path(vae_ckpt), device_t)
        stage2_ckpt = torch.load(diff_path, map_location=device_t, weights_only=False)
        if stage2_ckpt.get("ema_vae_model") is not None:
            vae.load_state_dict(stage2_ckpt["ema_vae_model"])
            print("Loaded Stage2 EMA-teacher VAE weights for inference.", flush=True)
        diffusion, s2_cfg, latent_mean, latent_std = load_diffusion(
            diff_path,
            vae.encoder.fc_mu.out_features,
            device_t,
        )
        schedule = DiffusionSchedule(s2_cfg.diffusion_steps, device=str(device_t))
        return cls(vae, diffusion, schedule, latent_mean, latent_std, device_t)

    @torch.no_grad()
    def encode_present(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """x: (T,) or (1,1,T) -> z_t (mu)."""
        if isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[0] != 1:
                x = x.reshape(-1)
            x = torch.from_numpy(x).float()
        elif x.ndim == 3 and x.shape[1] != 1:
            x = x.reshape(x.shape[0], -1)
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        x = x.to(self.device)
        mu, _ = self.vae.encoder(x)
        return mu

    @torch.no_grad()
    def predict_latent_next(self, z_context: torch.Tensor) -> torch.Tensor:
        """z_context -> z_hat_{t+1}; accepts (B,D) or (B,K,D)."""
        z_context_norm = (z_context - self.latent_mean) / self.latent_std
        if z_context_norm.dim() == 2 and self.context_len > 1:
            z_context_norm = z_context_norm.unsqueeze(1).repeat(1, self.context_len, 1)
        z_hat_norm = sample_future_latent(self.diffusion, self.schedule, z_context_norm)
        return z_hat_norm * self.latent_std + self.latent_mean

    @torch.no_grad()
    def latent_to_stage(self, z: torch.Tensor) -> torch.Tensor:
        """Stage 1 frozen SleepStageClassifier."""
        return self.vae.stage_classifier(z).argmax(dim=-1)

    @torch.no_grad()
    def predict_next_stage(self, x_present: np.ndarray | torch.Tensor) -> dict:
        """
        Full single-step inference from present EEG.
        Returns logits, predicted stage id/name, z_t, z_hat_{t+1}.
        """
        z_t = self.encode_present(x_present)
        z_hat = self.predict_latent_next(z_t)
        logits = self.vae.stage_classifier(z_hat)
        pred = logits.argmax(dim=-1)
        return {
            "z_t": z_t.cpu(),
            "z_hat_tp1": z_hat.cpu(),
            "logits": logits.cpu(),
            "pred_id": int(pred.item()) if pred.numel() == 1 else pred.cpu().numpy(),
            "pred_name": STAGE_NAMES[int(pred.item())] if pred.numel() == 1 else [STAGE_NAMES[i] for i in pred.tolist()],
        }

    @torch.no_grad()
    def predict_batch(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """X (N, T) -> predicted stage ids (N,) for each present epoch (one-step ahead)."""
        if X.ndim == 3 and X.shape[1] != 1:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 3:
            X = X.squeeze(1)
        preds = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(self.device)
            xb = xb.unsqueeze(1)
            z_t = self.encode_present(xb)
            z_hat = self.predict_latent_next(z_t)
            preds.append(self.latent_to_stage(z_hat).cpu().numpy())
        return np.concatenate(preds)

    @torch.no_grad()
    def predict_next_stage_from_context(self, x_context: np.ndarray | torch.Tensor) -> dict:
        """Predict next stage from K consecutive present EEG epochs."""
        if isinstance(x_context, np.ndarray):
            if x_context.ndim == 3 and x_context.shape[1] != 1:
                x_context = x_context.reshape(x_context.shape[0], -1)
            x_context = torch.from_numpy(x_context).float()
        if x_context.ndim == 2:
            x_context = x_context.unsqueeze(1)
        x_context = x_context.to(self.device)
        z_ctx, _ = self.vae.encoder(x_context)
        z_ctx = z_ctx.unsqueeze(0)
        z_hat = self.predict_latent_next(z_ctx)
        logits = self.vae.stage_classifier(z_hat)
        pred = logits.argmax(dim=-1)
        return {
            "z_context": z_ctx.cpu(),
            "z_hat_tp1": z_hat.cpu(),
            "logits": logits.cpu(),
            "pred_id": int(pred.item()),
            "pred_name": STAGE_NAMES[int(pred.item())],
        }

    @torch.no_grad()
    def predict_next_stage_from_context_batch(
        self,
        x_context: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Predict next stage for a batch of K-epoch contexts: (B,K,...) -> pred ids."""
        if isinstance(x_context, np.ndarray):
            if x_context.ndim == 4 and x_context.shape[2] != 1:
                bsz, ctx_len = x_context.shape[:2]
                x_context = x_context.reshape(bsz, ctx_len, -1)
            x_context = torch.from_numpy(x_context).float()
        if x_context.ndim == 3:
            x_context = x_context.unsqueeze(2)
        bsz, ctx_len = x_context.shape[:2]
        x_context = x_context.to(self.device)
        x_flat = x_context.reshape(bsz * ctx_len, *x_context.shape[2:])
        z_ctx, _ = self.vae.encoder(x_flat)
        z_ctx = z_ctx.view(bsz, ctx_len, -1)
        z_hat = self.predict_latent_next(z_ctx)
        pred = self.latent_to_stage(z_hat)
        return pred.cpu().numpy()


def _subject_sorted_indices(subject_ids: list[str], indices: np.ndarray) -> dict[str, np.ndarray]:
    subj = np.array(subject_ids)
    out = {}
    for sid in np.unique(subj[indices]):
        out[sid] = np.sort(indices[subj[indices] == sid])
    return out


def _safe_accuracy(correct: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones_like(correct, dtype=bool)
    denom = int(mask.sum())
    return float(correct[mask].mean()) if denom > 0 else float("nan")


def _subject_indices_for_rollout(
    subject_ids: list[str],
    eval_indices: np.ndarray,
    subject_id: str | None = None,
) -> tuple[str, np.ndarray]:
    by_subj = _subject_sorted_indices(subject_ids, eval_indices)
    if not by_subj:
        raise ValueError("No subject indices available for rollout.")
    if subject_id is None:
        subject_id = max(by_subj, key=lambda sid: len(by_subj[sid]))
    if subject_id not in by_subj:
        raise ValueError(f"subject_id={subject_id!r} not found in eval_indices.")
    idxs = by_subj[subject_id]
    if len(idxs) < 2:
        raise ValueError(f"subject_id={subject_id!r} needs at least 2 epochs, got {len(idxs)}.")
    return subject_id, idxs


def _valid_next_pair_positions(
    idxs: np.ndarray,
    epoch_onsets: np.ndarray | None = None,
    segment_sec: float = 30.0,
    onset_tolerance_sec: float = 1e-3,
) -> np.ndarray:
    if len(idxs) < 2:
        return np.asarray([], dtype=np.int64)
    if epoch_onsets is None:
        return np.arange(len(idxs) - 1, dtype=np.int64)
    gaps = np.asarray(epoch_onsets[idxs[1:]] - epoch_onsets[idxs[:-1]], dtype=np.float64)
    return np.where(np.abs(gaps - segment_sec) <= onset_tolerance_sec)[0].astype(np.int64)


def _valid_rollout_starts(
    idxs: np.ndarray,
    horizon: int,
    epoch_onsets: np.ndarray | None = None,
    segment_sec: float = 30.0,
    onset_tolerance_sec: float = 1e-3,
) -> np.ndarray:
    if len(idxs) <= horizon:
        return np.asarray([], dtype=np.int64)
    if epoch_onsets is None:
        return np.arange(len(idxs) - horizon, dtype=np.int64)
    pair_pos = set(_valid_next_pair_positions(idxs, epoch_onsets, segment_sec, onset_tolerance_sec).tolist())
    starts = [
        pos
        for pos in range(len(idxs) - horizon)
        if all((pos + offset) in pair_pos for offset in range(horizon))
    ]
    return np.asarray(starts, dtype=np.int64)


def _near_transition_mask(transition: np.ndarray, window: int = 1) -> np.ndarray:
    """Mark exact transitions and their neighboring prediction targets."""
    transition = np.asarray(transition, dtype=bool)
    near = transition.copy()
    for offset in range(1, max(0, window) + 1):
        near[offset:] |= transition[:-offset]
        near[:-offset] |= transition[offset:]
    return near


def _rollout_stats(
    true: np.ndarray,
    pred: np.ndarray,
    transition: np.ndarray,
    near_transition: np.ndarray | None = None,
) -> dict:
    correct = pred == true
    stable = ~transition
    if near_transition is None:
        near_transition = _near_transition_mask(transition, window=1)
    cm = np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    metrics = _metrics_from_confusion(cm)
    metrics.update(
        {
            "acc": _safe_accuracy(correct),
            "stable_acc": _safe_accuracy(correct, stable),
            "transition_acc": _safe_accuracy(correct, transition),
            "near_transition_acc": _safe_accuracy(correct, near_transition),
            "n": int(len(true)),
            "n_stable": int(stable.sum()),
            "n_transition": int(transition.sum()),
            "n_near_transition": int(near_transition.sum()),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_step1_accuracy(
    pipeline: SleepInferencePipeline,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    eval_indices: np.ndarray,
    epoch_onsets: np.ndarray | None = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> dict:
    """
    Non-AR step-1: present x_t -> predict s_hat_{t+1}, compare to true y at t+1.
    Consecutive pairs within each subject on eval_indices.
    """
    by_subj = _subject_sorted_indices(subject_ids, eval_indices)
    cm = np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64)
    correct, total = 0, 0
    per_h = {1: {"correct": 0, "total": 0}}
    true_all = []
    pred_all = []
    present_all = []
    transition_all = []

    pairs = []
    for idxs in by_subj.values():
        if len(idxs) <= pipeline.context_len:
            continue
        starts = _valid_rollout_starts(idxs, pipeline.context_len, epoch_onsets)
        for start in starts:
            context = idxs[start : start + pipeline.context_len]
            target = int(idxs[start + pipeline.context_len])
            pairs.append((context, target))

    pair_iter = tqdm(range(0, len(pairs), batch_size), desc="Inference step-1", disable=not show_progress)
    for start in pair_iter:
        batch_pairs = pairs[start : start + batch_size]
        contexts = np.stack([X[context] for context, _ in batch_pairs])
        targets = np.asarray([b for _, b in batch_pairs], dtype=np.int64)
        preds = pipeline.predict_next_stage_from_context_batch(contexts).astype(np.int64)
        for (context, _), b, pred in zip(batch_pairs, targets, preds):
            true = int(y[b])
            present = int(y[int(context[-1])])
            is_transition = bool(present != true)
            cm[true, int(pred)] += 1
            true_all.append(true)
            pred_all.append(int(pred))
            present_all.append(present)
            transition_all.append(is_transition)
            if int(pred) == true:
                correct += 1
            total += 1
            per_h[1]["total"] += 1
            if int(pred) == true:
                per_h[1]["correct"] += 1
        if show_progress and total > 0:
            pair_iter.set_postfix(acc=f"{correct / max(total, 1):.3f}")

    metrics = _metrics_from_confusion(cm)
    metrics["per_horizon"] = {k: v["correct"] / max(v["total"], 1) for k, v in per_h.items()}
    true_arr = np.asarray(true_all, dtype=np.int64)
    pred_arr = np.asarray(pred_all, dtype=np.int64)
    present_arr = np.asarray(present_all, dtype=np.int64)
    transition_arr = np.asarray(transition_all, dtype=bool)
    near_transition_arr = _near_transition_mask(transition_arr, window=1)
    split_stats = _rollout_stats(
        true_arr,
        pred_arr,
        transition_arr,
        near_transition=near_transition_arr,
    )
    metrics.update(
        {
            "acc": split_stats["acc"],
            "stable_acc": split_stats["stable_acc"],
            "transition_acc": split_stats["transition_acc"],
            "near_transition_acc": split_stats["near_transition_acc"],
            "n": split_stats["n"],
            "n_stable": split_stats["n_stable"],
            "n_transition": split_stats["n_transition"],
            "n_near_transition": split_stats["n_near_transition"],
            "transition_confusion_matrix": split_stats["confusion_matrix"]
            if split_stats["n_transition"] == split_stats["n"]
            else _rollout_stats(
                true_arr[transition_arr],
                pred_arr[transition_arr],
                np.ones(int(transition_arr.sum()), dtype=bool),
                near_transition=np.ones(int(transition_arr.sum()), dtype=bool),
            )["confusion_matrix"],
            "true": true_arr,
            "pred": pred_arr,
            "present": present_arr,
            "transition": transition_arr,
            "near_transition": near_transition_arr,
            "wake_to_sleep": (present_arr == 0) & (true_arr != 0),
            "sleep_to_wake": (present_arr != 0) & (true_arr == 0),
            "binary_boundary": ((present_arr == 0) != (true_arr == 0)),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_acc_n_non_autoregressive(
    pipeline: SleepInferencePipeline,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    eval_indices: np.ndarray,
    n: int = 1,
    epoch_onsets: np.ndarray | None = None,
    show_progress: bool = True,
) -> dict:
    """
    (?щ씪?대뱶??autoregressive Acc_n 怨?**?ㅻ쫫**)

    **non-AR ?ㅼ쨷 horizon**: 媛?i留덈떎 **?ㅼ젣 EEG** present留??곌퀬, ?덉륫 latent瑜?    ?ㅼ쓬 step 議곌굔?쇰줈 ?섍린吏 ?딆쓬.

    - horizon 1: x_t ??힆_{t+1} vs 吏꾩쭨 s_{t+1}
    - horizon 2: x_{t+1} ??힆_{t+2} vs 吏꾩쭨 s_{t+2}
    - horizon 3: x_{t+2} ??힆_{t+3} vs 吏꾩쭨 s_{t+3}

    Acc_n = ??n媛?horizon ?뺥솗?꾩쓽 **?됯퇏**. (媛숈? t?먯꽌 ?곗뇙 ?덉륫 ?꾨떂)
    """
    by_subj = _subject_sorted_indices(subject_ids, eval_indices)
    per_h = {i: {"correct": 0, "total": 0} for i in range(1, n + 1)}

    tasks = []
    for idxs in by_subj.values():
        if len(idxs) <= n:
            continue
        for start in _valid_rollout_starts(idxs, n, epoch_onsets):
            for i in range(1, n + 1):
                present_idx = idxs[start + i - 1]
                target_idx = idxs[start + i]
                tasks.append((i, int(present_idx), int(target_idx)))

    task_iter = tqdm(tasks, desc=f"Inference Acc_{n}", disable=not show_progress)
    for seen, (i, present_idx, target_idx) in enumerate(task_iter, start=1):
        out = pipeline.predict_next_stage(X[present_idx])
        pred = int(out["pred_id"])
        per_h[i]["total"] += 1
        if pred == int(y[target_idx]):
            per_h[i]["correct"] += 1
        if show_progress and seen % 25 == 0:
            running = {
                h: per_h[h]["correct"] / max(per_h[h]["total"], 1)
                for h in range(1, n + 1)
            }
            task_iter.set_postfix({f"h{h}": f"{v:.3f}" for h, v in running.items()})

    horizon_acc = {i: per_h[i]["correct"] / max(per_h[i]["total"], 1) for i in range(1, n + 1)}
    acc_n = float(np.mean(list(horizon_acc.values()))) if horizon_acc else 0.0
    return {
        "acc_n": acc_n,
        "n_horizons": n,
        "per_horizon_acc": horizon_acc,
        "per_horizon_n": {i: per_h[i]["total"] for i in range(1, n + 1)},
    }


@torch.no_grad()
def evaluate_acc_n_autoregressive(
    pipeline: SleepInferencePipeline,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    eval_indices: np.ndarray,
    n: int = 1,
    epoch_onsets: np.ndarray | None = None,
    show_progress: bool = True,
) -> dict:
    """
    Autoregressive Acc_n.

    Start with the real EEG context used by Stage 2, then feed each predicted
    latent back into the context for the next horizon.
    """
    by_subj = _subject_sorted_indices(subject_ids, eval_indices)
    per_h = {i: {"correct": 0, "total": 0} for i in range(1, n + 1)}
    cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}

    tasks = []
    needed = pipeline.context_len + n
    for idxs in by_subj.values():
        if len(idxs) <= needed:
            continue
        for start in _valid_rollout_starts(idxs, needed - 1, epoch_onsets):
            context = idxs[start : start + pipeline.context_len]
            targets = idxs[start + pipeline.context_len : start + pipeline.context_len + n]
            tasks.append((context, targets))

    task_iter = tqdm(tasks, desc=f"Inference AR Acc_{n}", disable=not show_progress)
    for seen, (context, targets) in enumerate(task_iter, start=1):
        x_context = X[context]
        if x_context.ndim == 3 and x_context.shape[1] != 1:
            x_context = x_context.reshape(x_context.shape[0], -1)
        xb = torch.from_numpy(x_context).float()
        if xb.ndim == 2:
            xb = xb.unsqueeze(1)
        xb = xb.to(pipeline.device)
        z_ctx, _ = pipeline.vae.encoder(xb)
        z_ctx = z_ctx.unsqueeze(0)

        for h, target_idx in enumerate(targets, start=1):
            z_next = pipeline.predict_latent_next(z_ctx)
            pred = int(pipeline.latent_to_stage(z_next).item())
            true = int(y[int(target_idx)])
            cms[h][true, pred] += 1
            per_h[h]["total"] += 1
            if pred == true:
                per_h[h]["correct"] += 1
            if pipeline.context_len > 1:
                z_ctx = torch.cat([z_ctx[:, 1:, :], z_next.unsqueeze(1)], dim=1)
            else:
                z_ctx = z_next

        if show_progress and seen % 25 == 0:
            running = {
                h: per_h[h]["correct"] / max(per_h[h]["total"], 1)
                for h in range(1, n + 1)
            }
            task_iter.set_postfix({f"h{h}": f"{v:.3f}" for h, v in running.items()})

    horizon_acc = {i: per_h[i]["correct"] / max(per_h[i]["total"], 1) for i in range(1, n + 1)}
    acc_n = float(np.mean(list(horizon_acc.values()))) if horizon_acc else 0.0
    combined_cm = sum(cms.values())
    metrics = _metrics_from_confusion(combined_cm)
    metrics.update(
        {
            "acc_n": acc_n,
            "n_horizons": n,
            "per_horizon_acc": horizon_acc,
            "per_horizon_n": {i: per_h[i]["total"] for i in range(1, n + 1)},
            "per_horizon_confusion": cms,
        }
    )
    return metrics


@torch.no_grad()
def evaluate_acc_n_direct_multi(
    vae_ckpt: Path | str,
    diffusion_multi_ckpt: Path | str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    eval_indices: np.ndarray,
    n: int = 3,
    epoch_onsets: np.ndarray | None = None,
    device: str | None = None,
    show_progress: bool = True,
) -> dict:
    """One-shot multi-horizon diffusion evaluation.

    A single reverse diffusion process generates z_{t+1:t+n} jointly from the
    real EEG context. It does not feed predicted latents back into context.
    """
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae = load_frozen_vae(Path(vae_ckpt), device_t)
    diffusion, cfg, latent_mean, latent_std, ckpt_horizons = load_diffusion_multi(
        Path(diffusion_multi_ckpt),
        vae.encoder.fc_mu.out_features,
        device_t,
    )
    n = min(max(1, int(n)), ckpt_horizons)
    context_len = getattr(diffusion, "context_len", getattr(cfg, "context_len", 1))
    schedule = DiffusionSchedule(cfg.diffusion_steps, device=str(device_t))

    by_subj = _subject_sorted_indices(subject_ids, eval_indices)
    per_h = {i: {"correct": 0, "total": 0} for i in range(1, n + 1)}
    cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    trans_cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    stable_cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    w2s_cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    s2w_cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    w2s_binary_cms = {i: np.zeros((2, 2), dtype=np.int64) for i in range(1, n + 1)}
    s2w_binary_cms = {i: np.zeros((2, 2), dtype=np.int64) for i in range(1, n + 1)}
    tasks = []
    needed = context_len + n
    for idxs in by_subj.values():
        if len(idxs) < needed:
            continue
        for start in _valid_rollout_starts(idxs, needed - 1, epoch_onsets):
            context = idxs[start : start + context_len]
            targets = idxs[start + context_len : start + context_len + n]
            prev_targets = idxs[start + context_len - 1 : start + context_len + n - 1]
            tasks.append((context, targets, prev_targets))

    task_iter = tqdm(tasks, desc=f"Inference direct multi Acc_{n}", disable=not show_progress)
    for seen, (context, targets, prev_targets) in enumerate(task_iter, start=1):
        x_context = X[context]
        if x_context.ndim == 3 and x_context.shape[1] != 1:
            x_context = x_context.reshape(x_context.shape[0], -1)
        xb = torch.from_numpy(x_context).float()
        if xb.ndim == 2:
            xb = xb.unsqueeze(1)
        xb = xb.to(device_t)
        z_ctx = vae.encoder(xb)[0].unsqueeze(0)
        z_ctx_norm = (z_ctx - latent_mean) / latent_std
        z_future_norm = sample_future_latents_multi(diffusion, schedule, z_ctx_norm, horizons=ckpt_horizons)
        z_future = z_future_norm[:, :n, :] * latent_std + latent_mean
        logits = vae.stage_classifier(z_future.reshape(n, -1))
        preds = logits.argmax(dim=1).cpu().numpy()
        for h, (target_idx, prev_idx) in enumerate(zip(targets, prev_targets), start=1):
            prev = int(y[int(prev_idx)])
            true = int(y[int(target_idx)])
            pred = int(preds[h - 1])
            cms[h][true, pred] += 1
            if prev != true:
                trans_cms[h][true, pred] += 1
            else:
                stable_cms[h][true, pred] += 1
            true_bin = 0 if true == 0 else 1
            pred_bin = 0 if pred == 0 else 1
            if prev == 0 and true != 0:
                w2s_cms[h][true, pred] += 1
                w2s_binary_cms[h][true_bin, pred_bin] += 1
            elif prev != 0 and true == 0:
                s2w_cms[h][true, pred] += 1
                s2w_binary_cms[h][true_bin, pred_bin] += 1
            per_h[h]["total"] += 1
            if pred == true:
                per_h[h]["correct"] += 1
        if show_progress and seen % 25 == 0:
            running = {h: per_h[h]["correct"] / max(per_h[h]["total"], 1) for h in range(1, n + 1)}
            task_iter.set_postfix({f"h{h}": f"{v:.3f}" for h, v in running.items()})

    horizon_acc = {i: per_h[i]["correct"] / max(per_h[i]["total"], 1) for i in range(1, n + 1)}
    acc_n = float(np.mean(list(horizon_acc.values()))) if horizon_acc else 0.0
    combined_cm = sum(cms.values())
    metrics = _metrics_from_confusion(combined_cm)
    metrics.update(
        {
            "acc_n": acc_n,
            "n_horizons": n,
            "checkpoint_horizons": ckpt_horizons,
            "context_len": context_len,
            "per_horizon_acc": horizon_acc,
            "per_horizon_n": {i: per_h[i]["total"] for i in range(1, n + 1)},
            "per_horizon_confusion": cms,
            "per_horizon_transition_metrics": {
                i: _metrics_from_confusion(trans_cms[i]) for i in range(1, n + 1)
            },
            "per_horizon_stable_metrics": {
                i: _metrics_from_confusion(stable_cms[i]) for i in range(1, n + 1)
            },
            "per_horizon_direction_metrics": _direction_metrics(
                w2s_cms,
                s2w_cms,
                w2s_binary_cms,
                s2w_binary_cms,
                n,
            ),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_acc_n_direct_flow(
    vae_ckpt: Path | str,
    flow_multi_ckpt: Path | str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    eval_indices: np.ndarray,
    n: int = 3,
    epoch_onsets: np.ndarray | None = None,
    device: str | None = None,
    flow_steps: int | None = None,
    show_progress: bool = True,
) -> dict:
    """One-shot multi-horizon rectified-flow evaluation."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae = load_frozen_vae(Path(vae_ckpt), device_t)
    flow, cfg, latent_mean, latent_std, ckpt_horizons, ckpt_steps = load_flow_multi(
        Path(flow_multi_ckpt),
        vae.encoder.fc_mu.out_features,
        device_t,
    )
    n = min(max(1, int(n)), ckpt_horizons)
    context_len = getattr(flow, "context_len", getattr(cfg, "context_len", 1))
    steps = max(1, int(flow_steps or ckpt_steps))

    by_subj = _subject_sorted_indices(subject_ids, eval_indices)
    per_h = {i: {"correct": 0, "total": 0} for i in range(1, n + 1)}
    cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    trans_cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    stable_cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    w2s_cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    s2w_cms = {i: np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64) for i in range(1, n + 1)}
    w2s_binary_cms = {i: np.zeros((2, 2), dtype=np.int64) for i in range(1, n + 1)}
    s2w_binary_cms = {i: np.zeros((2, 2), dtype=np.int64) for i in range(1, n + 1)}
    tasks = []
    needed = context_len + n
    for idxs in by_subj.values():
        if len(idxs) < needed:
            continue
        for start in _valid_rollout_starts(idxs, needed - 1, epoch_onsets):
            context = idxs[start : start + context_len]
            targets = idxs[start + context_len : start + context_len + n]
            prev_targets = idxs[start + context_len - 1 : start + context_len + n - 1]
            tasks.append((context, targets, prev_targets))

    task_iter = tqdm(tasks, desc=f"Inference direct flow Acc_{n}", disable=not show_progress)
    for seen, (context, targets, prev_targets) in enumerate(task_iter, start=1):
        x_context = X[context]
        if x_context.ndim == 3 and x_context.shape[1] != 1:
            x_context = x_context.reshape(x_context.shape[0], -1)
        xb = torch.from_numpy(x_context).float()
        if xb.ndim == 2:
            xb = xb.unsqueeze(1)
        xb = xb.to(device_t)
        z_ctx = vae.encoder(xb)[0].unsqueeze(0)
        z_ctx_norm = (z_ctx - latent_mean) / latent_std
        z_future_norm = sample_future_latents_flow(flow, z_ctx_norm, steps=steps)
        z_future = z_future_norm[:, :n, :] * latent_std + latent_mean
        logits = vae.stage_classifier(z_future.reshape(n, -1))
        preds = logits.argmax(dim=1).cpu().numpy()
        for h, (target_idx, prev_idx) in enumerate(zip(targets, prev_targets), start=1):
            prev = int(y[int(prev_idx)])
            true = int(y[int(target_idx)])
            pred = int(preds[h - 1])
            cms[h][true, pred] += 1
            if prev != true:
                trans_cms[h][true, pred] += 1
            else:
                stable_cms[h][true, pred] += 1
            true_bin = 0 if true == 0 else 1
            pred_bin = 0 if pred == 0 else 1
            if prev == 0 and true != 0:
                w2s_cms[h][true, pred] += 1
                w2s_binary_cms[h][true_bin, pred_bin] += 1
            elif prev != 0 and true == 0:
                s2w_cms[h][true, pred] += 1
                s2w_binary_cms[h][true_bin, pred_bin] += 1
            per_h[h]["total"] += 1
            if pred == true:
                per_h[h]["correct"] += 1
        if show_progress and seen % 25 == 0:
            running = {h: per_h[h]["correct"] / max(per_h[h]["total"], 1) for h in range(1, n + 1)}
            task_iter.set_postfix({f"h{h}": f"{v:.3f}" for h, v in running.items()})

    horizon_acc = {i: per_h[i]["correct"] / max(per_h[i]["total"], 1) for i in range(1, n + 1)}
    acc_n = float(np.mean(list(horizon_acc.values()))) if horizon_acc else 0.0
    combined_cm = sum(cms.values())
    metrics = _metrics_from_confusion(combined_cm)
    metrics.update(
        {
            "acc_n": acc_n,
            "n_horizons": n,
            "checkpoint_horizons": ckpt_horizons,
            "context_len": context_len,
            "flow_steps": steps,
            "per_horizon_acc": horizon_acc,
            "per_horizon_n": {i: per_h[i]["total"] for i in range(1, n + 1)},
            "per_horizon_confusion": cms,
            "per_horizon_transition_metrics": {
                i: _metrics_from_confusion(trans_cms[i]) for i in range(1, n + 1)
            },
            "per_horizon_stable_metrics": {
                i: _metrics_from_confusion(stable_cms[i]) for i in range(1, n + 1)
            },
            "per_horizon_direction_metrics": _direction_metrics(
                w2s_cms,
                s2w_cms,
                w2s_binary_cms,
                s2w_binary_cms,
                n,
            ),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_subject_rollout(
    pipeline: SleepInferencePipeline,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    eval_indices: np.ndarray,
    subject_id: str | None = None,
    n_autoregressive: int = 3,
    epoch_onsets: np.ndarray | None = None,
    show_progress: bool = True,
) -> dict:
    """
    Evaluate one subject from first to last epoch.

    step1:
        x_t -> diffusion -> predicted s_{t+1}, repeated with the real EEG at
        every t.
    autoregressive:
        x_t -> z_t, then repeatedly z_hat_{t+h} -> z_hat_{t+h+1} without using
        the real intermediate EEG.  Metrics are reported for horizons 1..n.
    """
    sid, idxs = _subject_indices_for_rollout(subject_ids, eval_indices, subject_id)
    y_subject = np.asarray(y[idxs], dtype=np.int64)

    step_positions = _valid_next_pair_positions(idxs, epoch_onsets)
    step_true = y_subject[step_positions + 1]
    step_transition = y_subject[step_positions] != y_subject[step_positions + 1]
    step_pred = []
    step_iter = step_positions
    if show_progress:
        step_iter = tqdm(step_iter, desc=f"{sid} step1 rollout")
    for pos in step_iter:
        out = pipeline.predict_next_stage(X[int(idxs[pos])])
        step_pred.append(int(out["pred_id"]))
    step_pred_arr = np.asarray(step_pred, dtype=np.int64)
    step_near_transition = _near_transition_mask(step_transition, window=1)
    step_stats = _rollout_stats(
        step_true,
        step_pred_arr,
        step_transition,
        near_transition=step_near_transition,
    )

    ar: dict[int, dict] = {}
    max_h = max(1, int(n_autoregressive))
    starts = _valid_rollout_starts(idxs, max_h, epoch_onsets)
    if show_progress:
        starts = tqdm(starts, desc=f"{sid} AR rollout h=1..{max_h}")

    ar_true = {h: [] for h in range(1, max_h + 1)}
    ar_pred = {h: [] for h in range(1, max_h + 1)}
    ar_transition = {h: [] for h in range(1, max_h + 1)}
    ar_near_transition = {h: [] for h in range(1, max_h + 1)}
    for pos in starts:
        z = pipeline.encode_present(X[int(idxs[pos])])
        for h in range(1, max_h + 1):
            z = pipeline.predict_latent_next(z)
            pred = int(pipeline.latent_to_stage(z).item())
            ar_pred[h].append(pred)
            ar_true[h].append(int(y_subject[pos + h]))
            exact_transition = bool(y_subject[pos + h - 1] != y_subject[pos + h])
            near_transition = False
            for k in range(max(0, pos + h - 2), min(len(y_subject) - 1, pos + h + 1)):
                if y_subject[k] != y_subject[k + 1]:
                    near_transition = True
                    break
            ar_transition[h].append(exact_transition)
            ar_near_transition[h].append(near_transition)

    for h in range(1, max_h + 1):
        ar[h] = {
            "true": np.asarray(ar_true[h], dtype=np.int64),
            "pred": np.asarray(ar_pred[h], dtype=np.int64),
            "transition": np.asarray(ar_transition[h], dtype=bool),
            "near_transition": np.asarray(ar_near_transition[h], dtype=bool),
        }
        ar[h]["stats"] = _rollout_stats(
            ar[h]["true"],
            ar[h]["pred"],
            ar[h]["transition"],
            near_transition=ar[h]["near_transition"],
        )

    return {
        "subject_id": sid,
        "indices": idxs,
        "step_positions": step_positions,
        "ar_start_positions": starts,
        "true_stages": y_subject,
        "step1": {
            "true": step_true,
            "pred": step_pred_arr,
            "transition": step_transition,
            "near_transition": step_near_transition,
            "stats": step_stats,
        },
        "autoregressive": ar,
    }


def plot_subject_rollout_report(
    result: dict,
    max_points: int = 300,
    show: bool = True,
    include_sequence: bool = False,
):
    """Matplotlib summary for evaluate_subject_rollout()."""
    import matplotlib.pyplot as plt

    sid = result["subject_id"]
    true_stages = result["true_stages"]
    n_show = min(max_points, len(true_stages))

    panels = 3 if include_sequence else 2
    fig, axes = plt.subplots(panels, 1, figsize=(14, 9 if include_sequence else 6))
    if panels == 1:
        axes = [axes]

    axis_offset = 0
    if include_sequence:
        x_full = np.arange(n_show)
        ax = axes[0]
        ax.step(x_full, true_stages[:n_show], where="post", label="true", lw=1.5, color="black")
        step_pred = result["step1"]["pred"]
        if len(step_pred) > 0:
            x_step = np.arange(1, min(n_show, len(step_pred) + 1))
            ax.step(x_step, step_pred[: len(x_step)], where="post", label="real x step1", alpha=0.8)
        for h, payload in result["autoregressive"].items():
            pred = payload["pred"]
            if len(pred) == 0:
                continue
            x_ar = np.arange(h, min(n_show, len(pred) + h))
            ax.step(x_ar, pred[: len(x_ar)], where="post", label=f"AR h{h}", alpha=0.55)
        ax.set_yticks(range(len(STAGE_NAMES)))
        ax.set_yticklabels(STAGE_NAMES)
        ax.set_title(f"Subject {sid}: true stages vs predictions")
        ax.set_xlabel("epoch within subject")
        ax.set_ylabel("stage")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
        axis_offset = 1

    labels = ["real x"] + [f"AR h{h}" for h in result["autoregressive"]]
    overall = [result["step1"]["stats"]["acc"]] + [
        result["autoregressive"][h]["stats"]["acc"] for h in result["autoregressive"]
    ]
    stable = [result["step1"]["stats"]["stable_acc"]] + [
        result["autoregressive"][h]["stats"]["stable_acc"] for h in result["autoregressive"]
    ]
    trans = [result["step1"]["stats"]["transition_acc"]] + [
        result["autoregressive"][h]["stats"]["transition_acc"] for h in result["autoregressive"]
    ]
    near_trans = [result["step1"]["stats"]["near_transition_acc"]] + [
        result["autoregressive"][h]["stats"]["near_transition_acc"] for h in result["autoregressive"]
    ]

    ax = axes[axis_offset]
    x = np.arange(len(labels))
    width = 0.2
    ax.bar(x - 1.5 * width, overall, width, label="overall")
    ax.bar(x - 0.5 * width, stable, width, label="stable")
    ax.bar(x + 0.5 * width, trans, width, label="transition")
    ax.bar(x + 1.5 * width, near_trans, width, label="near transition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy split by all / stable / transition / near-transition targets")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    ax = axes[axis_offset + 1]
    cm = result["step1"]["stats"]["confusion_matrix"].astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(STAGE_NAMES)))
    ax.set_yticks(range(len(STAGE_NAMES)))
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Step1 confusion matrix (row-normalized)")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def print_subject_rollout_summary(result: dict) -> None:
    """Compact console summary for evaluate_subject_rollout()."""

    def fmt(value: float) -> str:
        return "nan" if np.isnan(value) else f"{value:.3f}"

    def row(name: str, stats: dict) -> str:
        return (
            f"{name:<8} "
            f"acc={fmt(stats['acc'])} "
            f"bal={fmt(stats['balanced_acc'])} "
            f"mf1={fmt(stats['macro_f1'])} "
            f"stable={fmt(stats['stable_acc'])} "
            f"trans={fmt(stats['transition_acc'])} "
            f"near_trans={fmt(stats['near_transition_acc'])} "
            f"n={stats['n']} "
            f"n_trans={stats['n_transition']} "
            f"n_near={stats['n_near_transition']}"
        )

    print(f"[Subject rollout] subject={result['subject_id']} epochs={len(result['true_stages'])}")
    print(row("real x", result["step1"]["stats"]))
    for h, payload in result["autoregressive"].items():
        print(row(f"AR h{h}", payload["stats"]))

    step_recall = result["step1"]["stats"]["per_class_acc"]
    print(
        "step1 recall: "
        + ", ".join(f"{name}={fmt(step_recall[name])}" for name in STAGE_NAMES)
    )


def subject_rollout_table(result: dict, max_rows: int | None = 120):
    """
    Return a pandas table for one-subject rollout predictions.

    real_x_step1 is not part of the autoregressive chain: it is the baseline
    that uses the real EEG at each t to predict t+1.
    """
    import pandas as pd

    true_stages = result["true_stages"]
    n_step = len(result["step1"]["pred"])
    n = n_step if max_rows is None else min(n_step, max_rows)
    step_positions = result.get("step_positions", np.arange(n_step))
    ar_start_positions = result.get("ar_start_positions", np.asarray([], dtype=np.int64))
    ar_pos_to_idx = {int(pos): i for i, pos in enumerate(ar_start_positions)}
    rows = []
    for row_idx in range(n):
        pos = int(step_positions[row_idx])
        row = {
            "t": pos,
            "stage_t": STAGE_NAMES[int(true_stages[pos])],
            "true_t+1": STAGE_NAMES[int(result["step1"]["true"][row_idx])],
            "transition_t->t+1": bool(result["step1"]["transition"][row_idx]),
            "near_transition_t+1": bool(result["step1"]["near_transition"][row_idx]),
            "real_x_step1_pred": STAGE_NAMES[int(result["step1"]["pred"][row_idx])],
            "real_x_step1_ok": bool(
                result["step1"]["pred"][row_idx] == result["step1"]["true"][row_idx]
            ),
        }
        ar_idx = ar_pos_to_idx.get(pos)
        for h, payload in result["autoregressive"].items():
            # AR rows are indexed by valid rollout start order, not necessarily
            # by the raw subject epoch position when gaps were removed.
            if ar_idx is not None and ar_idx < len(payload["pred"]):
                row[f"true_t+{h}"] = STAGE_NAMES[int(payload["true"][ar_idx])]
                row[f"AR_h{h}_pred"] = STAGE_NAMES[int(payload["pred"][ar_idx])]
                row[f"AR_h{h}_ok"] = bool(payload["pred"][ar_idx] == payload["true"][ar_idx])
                row[f"transition_t+{h-1}->t+{h}"] = bool(payload["transition"][ar_idx])
                row[f"near_transition_t+{h}"] = bool(payload["near_transition"][ar_idx])
            else:
                row[f"true_t+{h}"] = None
                row[f"AR_h{h}_pred"] = None
                row[f"AR_h{h}_ok"] = None
                row[f"transition_t+{h-1}->t+{h}"] = None
                row[f"near_transition_t+{h}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def run_inference_demo(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    test_idx: np.ndarray,
    vae_ckpt: Path | str,
    diffusion_ckpt: Path | str | None = None,
    n_horizons: int | None = None,
    device: str | None = None,
    show_progress: bool = True,
) -> dict:
    """Load checkpoints, run step-1 and autoregressive Acc_n on test split."""
    inf_cfg = InferenceConfig()
    n_h = n_horizons if n_horizons is not None else inf_cfg.n_horizons
    pipe = SleepInferencePipeline.from_checkpoints(vae_ckpt, diffusion_ckpt, device=device)
    step1 = evaluate_step1_accuracy(
        pipe,
        X,
        y,
        subject_ids,
        test_idx,
        batch_size=inf_cfg.batch_size,
        show_progress=show_progress,
    )
    acc_n = evaluate_acc_n_autoregressive(
        pipe,
        X,
        y,
        subject_ids,
        test_idx,
        n=max(1, n_h),
        show_progress=show_progress,
    )
    print(
        f"[Inference step-1] acc={step1['accuracy']:.4f} "
        f"bal={step1['balanced_acc']:.4f} mf1={step1['macro_f1']:.4f} "
        f"false_alarm={step1['macro_false_alarm']:.4f} "
        f"pairs={step1['n_pairs']}"
    )
    print("[Inference per-stage recall] " + ", ".join(
        f"{k}={v:.3f}" for k, v in step1["per_class_acc"].items()
    ))
    print("[Inference per-stage false alarm] " + ", ".join(
        f"{k}={v:.3f}" for k, v in step1["per_class_false_alarm"].items()
    ))
    print("[Inference confusion matrix rows=true cols=pred]")
    print(step1["confusion_matrix"])
    print(
        f"[Inference Acc_{n_h} AR] acc_n={acc_n['acc_n']:.4f} "
        f"mf1={acc_n['macro_f1']:.4f} false_alarm={acc_n['macro_false_alarm']:.4f} "
        f"per_h={acc_n['per_horizon_acc']}\n"
        f"  AR Acc_{n_h}: real EEG context {pipe.context_len}개로 시작한 뒤, "
        f"predicted latent를 다음 horizon context에 넘겨서 평가."
    )
    return {"pipeline": pipe, "step1": step1, "acc_n": acc_n}
