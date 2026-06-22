"""Frozen Stage 1 latent -> MLP next-step sleep-stage prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from .preprocess import STAGE_NAMES
from .train_stage2 import load_frozen_vae


@dataclass
class LatentMLPConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    dropout: float = 0.2
    use_class_weights: bool = True
    use_weighted_sampler: bool = True
    seed: int = 42
    segment_sec: float = 30.0
    onset_tolerance_sec: float = 1e-3


class LatentNextStepMLP(nn.Module):
    """Predict y_{t+1} from frozen Stage 1 mu_t."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def make_next_step_pairs(
    subject_ids: list[str],
    split_idx: np.ndarray,
    y: np.ndarray,
    epoch_onsets: np.ndarray | None = None,
    *,
    segment_sec: float = 30.0,
    onset_tolerance_sec: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices, target labels, source labels, and exact transition flags."""
    split_idx = np.asarray(split_idx, dtype=np.int64)
    subj = np.asarray(subject_ids)
    y = np.asarray(y, dtype=np.int64)
    onsets = np.asarray(epoch_onsets, dtype=np.float64) if epoch_onsets is not None else None

    src_indices: list[int] = []
    y_src: list[int] = []
    y_next: list[int] = []
    transition: list[bool] = []
    for sid in np.unique(subj[split_idx]):
        idxs = np.sort(split_idx[subj[split_idx] == sid])
        if len(idxs) < 2:
            continue
        for a, b in zip(idxs[:-1], idxs[1:]):
            if onsets is not None and abs(float(onsets[b] - onsets[a]) - segment_sec) > onset_tolerance_sec:
                continue
            src_indices.append(int(a))
            y_src.append(int(y[a]))
            y_next.append(int(y[b]))
            transition.append(bool(y[a] != y[b]))

    return (
        np.asarray(src_indices, dtype=np.int64),
        np.asarray(y_next, dtype=np.int64),
        np.asarray(y_src, dtype=np.int64),
        np.asarray(transition, dtype=bool),
    )


@torch.no_grad()
def encode_indices(
    vae: nn.Module,
    X: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if X.ndim == 3 and X.shape[1] != 1:
        X_arr = X.reshape(X.shape[0], -1)[:, None, :]
    else:
        X_arr = X[:, None, :] if X.ndim == 2 else X
    zs = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        xb = torch.from_numpy(X_arr[idx]).float().to(device)
        mu, _ = vae.encoder(xb)
        zs.append(mu.cpu().numpy())
    if not zs:
        return np.empty((0, vae.encoder.fc_mu.out_features), dtype=np.float32)
    return np.concatenate(zs, axis=0).astype(np.float32)


def _metrics_from_arrays(true: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1

    per_class_acc: dict[str, float] = {}
    recalls = []
    f1s = []
    for i, name in enumerate(STAGE_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        if support > 0:
            recall = tp / support
            recalls.append(recall)
            per_class_acc[name] = float(recall)
        else:
            per_class_acc[name] = float("nan")
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)

    return {
        "accuracy": float((true == pred).mean()) if len(true) else float("nan"),
        "balanced_acc": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "n": int(len(true)),
    }


@torch.no_grad()
def evaluate_latent_mlp(
    model: LatentNextStepMLP,
    Z: np.ndarray,
    y_next: np.ndarray,
    transition: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    preds = []
    for start in range(0, len(Z), batch_size):
        xb = torch.from_numpy(Z[start : start + batch_size]).float().to(device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)
    all_metrics = _metrics_from_arrays(y_next, pred, len(STAGE_NAMES))
    transition_metrics = _metrics_from_arrays(y_next[transition], pred[transition], len(STAGE_NAMES))
    stable_metrics = _metrics_from_arrays(y_next[~transition], pred[~transition], len(STAGE_NAMES))
    return {
        **all_metrics,
        "pred": pred,
        "true": y_next,
        "transition": transition,
        "transition_acc": transition_metrics["accuracy"],
        "stable_acc": stable_metrics["accuracy"],
        "n_transition": transition_metrics["n"],
        "n_stable": stable_metrics["n"],
        "transition_confusion_matrix": transition_metrics["confusion_matrix"],
        "transition_metrics": transition_metrics,
        "stable_metrics": stable_metrics,
    }


def _class_weight(y_target: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_target, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_latent_mlp_next_step(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    vae_ckpt: str | Path,
    epoch_onsets: np.ndarray | None = None,
    cfg: LatentMLPConfig | None = None,
    save_dir: str | Path | None = None,
    device: str | None = None,
    show_progress: bool = True,
) -> tuple[LatentNextStepMLP, dict]:
    cfg = cfg or LatentMLPConfig()
    save_dir = Path(save_dir or "checkpoints/latent_mlp_next_step")
    save_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    vae = load_frozen_vae(Path(vae_ckpt), device_t)
    latent_dim = int(vae.encoder.fc_mu.out_features)
    num_classes = len(STAGE_NAMES)

    pair_data = {}
    for split_name, split_idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        src_idx, y_next, y_src, transition = make_next_step_pairs(
            subject_ids,
            split_idx,
            y,
            epoch_onsets,
            segment_sec=cfg.segment_sec,
            onset_tolerance_sec=cfg.onset_tolerance_sec,
        )
        if split_name in {"train", "val"} and len(src_idx) == 0:
            raise RuntimeError(f"No consecutive {split_name} pairs for latent MLP training.")
        Z = encode_indices(vae, X, src_idx, device_t, cfg.batch_size)
        pair_data[split_name] = {
            "src_idx": src_idx,
            "Z": Z,
            "y_next": y_next,
            "y_src": y_src,
            "transition": transition,
        }

    z_mean = pair_data["train"]["Z"].mean(axis=0, keepdims=True).astype(np.float32)
    z_std = np.maximum(pair_data["train"]["Z"].std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    for split in pair_data.values():
        split["Z"] = ((split["Z"] - z_mean) / z_std).astype(np.float32)

    train_y = pair_data["train"]["y_next"]
    train_ds = TensorDataset(
        torch.from_numpy(pair_data["train"]["Z"]).float(),
        torch.from_numpy(train_y).long(),
    )
    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        counts = np.bincount(train_y, minlength=num_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        sample_weights = 1.0 / counts[train_y]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler)

    model = LatentNextStepMLP(
        latent_dim=latent_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=num_classes,
        dropout=cfg.dropout,
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce_weight = _class_weight(train_y, num_classes, device_t) if cfg.use_class_weights else None
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    best_val = float("-inf")

    print(
        f"Latent MLP next-step training start: device={device_t}, epochs={cfg.epochs}, "
        f"latent_dim={latent_dim}, train_pairs={len(pair_data['train']['Z'])}, "
        f"val_pairs={len(pair_data['val']['Z'])}, test_pairs={len(pair_data['test']['Z'])}",
        flush=True,
    )
    print(
        f"Train transition pairs: {int(pair_data['train']['transition'].sum())}/"
        f"{len(pair_data['train']['transition'])}",
        flush=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        iterator = tqdm(
            train_loader,
            desc=f"Latent MLP epoch {epoch:03d}/{cfg.epochs}",
            leave=False,
            disable=not show_progress,
        )
        for xb, yb in iterator:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            loss = F.cross_entropy(model(xb), yb, weight=ce_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            if show_progress:
                iterator.set_postfix(loss=f"{loss.item():.3f}")
        running_loss /= max(len(train_loader), 1)

        train_eval = evaluate_latent_mlp(
            model,
            pair_data["train"]["Z"],
            pair_data["train"]["y_next"],
            pair_data["train"]["transition"],
            device_t,
            cfg.batch_size,
        )
        val_eval = evaluate_latent_mlp(
            model,
            pair_data["val"]["Z"],
            pair_data["val"]["y_next"],
            pair_data["val"]["transition"],
            device_t,
            cfg.batch_size,
        )
        train_log = {
            "loss": running_loss,
            "acc": train_eval["accuracy"],
            "balanced_acc": train_eval["balanced_acc"],
            "macro_f1": train_eval["macro_f1"],
            "transition_acc": train_eval["transition_acc"],
        }
        val_log = {
            "acc": val_eval["accuracy"],
            "balanced_acc": val_eval["balanced_acc"],
            "macro_f1": val_eval["macro_f1"],
            "transition_acc": val_eval["transition_acc"],
        }
        history["train"].append(train_log)
        history["val"].append(val_log)
        print(
            f"Latent MLP epoch {epoch:03d} | train loss={running_loss:.4f} "
            f"acc={train_eval['accuracy']:.3f} mf1={train_eval['macro_f1']:.3f} "
            f"trans={train_eval['transition_acc']:.3f} | "
            f"val acc={val_eval['accuracy']:.3f} mf1={val_eval['macro_f1']:.3f} "
            f"trans={val_eval['transition_acc']:.3f}",
            flush=True,
        )

        if float(val_eval["macro_f1"]) > best_val:
            best_val = float(val_eval["macro_f1"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "latent_dim": latent_dim,
                    "num_classes": num_classes,
                    "latent_mean": z_mean.squeeze(0),
                    "latent_std": z_std.squeeze(0),
                    "vae_ckpt": str(vae_ckpt),
                    "epoch": epoch,
                    "val_macro_f1": best_val,
                    "history": history,
                },
                save_dir / "best_latent_mlp.pt",
            )

    best = torch.load(save_dir / "best_latent_mlp.pt", map_location=device_t, weights_only=False)
    model.load_state_dict(best["model"])
    test_eval = evaluate_latent_mlp(
        model,
        pair_data["test"]["Z"],
        pair_data["test"]["y_next"],
        pair_data["test"]["transition"],
        device_t,
        cfg.batch_size,
    )
    history["test_metrics"] = test_eval
    history["pair_counts"] = {
        name: {
            "pairs": int(len(split["Z"])),
            "transition_pairs": int(split["transition"].sum()),
        }
        for name, split in pair_data.items()
    }
    history["best_checkpoint"] = str(save_dir / "best_latent_mlp.pt")
    return model, history
