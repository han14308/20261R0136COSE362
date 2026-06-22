"""Matplotlib plots for Stage 1/2 training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import MultiHeadVAE
from .preprocess import STAGE_NAMES
from .train_stage1 import evaluate_stage_test


def plot_stage1_history(
    history: dict[str, list[dict[str, float]]],
    save_path: str | Path | None = None,
    show: bool = True,
    kl_plot_cap: float | None = 50.0,
) -> plt.Figure:
    """Train/val: total/rec/stage (left), KL only (middle, capped for display), acc (right)."""
    train = history["train"]
    val = history["val"]
    if not train:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "empty history", ha="center")
        return fig
    epochs = range(1, len(train) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    loss_keys = [("total", "total"), ("rec", "rec"), ("stage", "stage")]
    if train and "spec" in train[0]:
        loss_keys.insert(2, ("spec", "spec"))
    if train and "band" in train[0]:
        loss_keys.insert(-1, ("band", "band"))
    if train and "sigma" in train[0] and any(m.get("sigma", 0.0) != 0.0 for m in train + val):
        loss_keys.insert(-1, ("sigma", "sigma"))
    for key, label in loss_keys:
        if not all(key in m for m in train) or not all(key in m for m in val):
            continue
        ax.plot(epochs, [m[key] for m in train], label=f"train {label}")
        ax.plot(epochs, [m[key] for m in val], "--", label=f"val {label}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Stage 1: total / rec / spec / stage")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    tr_kl = [m["kl"] for m in train]
    va_kl = [m["kl"] for m in val]
    if kl_plot_cap is not None:
        tr_kl = [min(v, kl_plot_cap) for v in tr_kl]
        va_kl = [min(v, kl_plot_cap) for v in va_kl]
        cap_note = f" (y capped at {kl_plot_cap} for plot)"
    else:
        cap_note = ""
    ax.plot(epochs, tr_kl, label="train kl")
    ax.plot(epochs, va_kl, "--", label="val kl")
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL loss")
    ax.set_title(f"Stage 1: KL only{cap_note}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if "acc" in train[0]:
        ax.plot(epochs, [m["acc"] for m in train], "o-", color="tab:blue", label="train acc", ms=3)
    ax.plot(epochs, [m["acc"] for m in val], "s-", color="tab:green", label="val acc", ms=3)
    ax.axhline(0.2, color="gray", ls="--", lw=1, label="random (5-class)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Stage classification accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: tuple[str, ...] = STAGE_NAMES,
    save_path: str | Path | None = None,
    show: bool = True,
    normalize: bool = True,
) -> plt.Figure:
    """Confusion matrix heatmap (rows=true, cols=pred)."""
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, row_sum, where=row_sum > 0, out=np.zeros_like(cm, dtype=float))
        fmt = ".2f"
        title = "Confusion matrix (row-normalized)"
    else:
        display = cm.astype(float)
        fmt = "d"
        title = "Confusion matrix (counts)"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    thresh = display.max() / 2 if display.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = format(display[i, j], fmt) if normalize else str(int(cm[i, j]))
            ax.text(
                j, i, text, ha="center", va="center",
                color="white" if display[i, j] > thresh else "black", fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_class_distribution(
    y: np.ndarray,
    title: str = "Label distribution",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    counts = [(y == i).sum() for i in range(len(STAGE_NAMES))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(STAGE_NAMES, counts, color="steelblue")
    ax.set_ylabel("epochs")
    ax.set_title(title)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


@torch.no_grad()
def plot_reconstruction_samples(
    model: MultiHeadVAE,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray | None = None,
    epoch_mean: np.ndarray | None = None,
    epoch_std: np.ndarray | None = None,
    n_show: int = 4,
    device: torch.device | str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    VAE는 **per-epoch z-score** 입력만 사용 (raw µV 아님).
    epoch_mean/std가 있으면 raw µV 복원해 함께 표시.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if X.ndim == 2:
        X = X[:, None, :]
    if indices is None:
        rng = np.random.default_rng(0)
        z_std_row = np.std(X.reshape(X.shape[0], -1), axis=1)
        pool = np.where(z_std_row > 0.05)[0]
        if len(pool) == 0:
            pool = np.arange(len(y))
        indices = rng.choice(pool, size=min(n_show, len(pool)), replace=False)
    else:
        indices = np.asarray(indices)[:n_show]

    has_denorm = epoch_mean is not None and epoch_std is not None
    rows_per = 3 if has_denorm else 2
    n_rows = rows_per * len(indices)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 1.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0
    for idx in indices:
        z_in = np.asarray(X[idx]).squeeze().astype(np.float32)
        if z_in.ndim != 1:
            raise ValueError(f"Expected 1D epoch at index {idx}, got shape {z_in.shape}")
        x = torch.from_numpy(z_in).view(1, 1, -1).to(device)
        pred_z = model.reconstruct(x, use_mu=True)[0, 0].detach().cpu().numpy()
        t = np.arange(z_in.shape[0]) / 100.0
        lab = STAGE_NAMES[int(y[idx])]

        if has_denorm:
            mu = float(epoch_mean[idx])
            sig = float(epoch_std[idx])
            raw_in = z_in * sig + mu
            pred_raw = pred_z * sig + mu
            ax0 = axes[row]
            ax0.plot(t, raw_in, color="C0", lw=0.7, label="raw EEG (µV, filtered)")
            ax0.set_ylabel(f"{lab}\nraw", fontsize=8)
            ax0.legend(loc="upper right", fontsize=6)
            ax0.grid(True, alpha=0.3)
            row += 1

            ax1 = axes[row]
            ax1.plot(t, z_in, color="C2", lw=0.7, label="VAE input (z-score)")
            ax1.set_ylabel("z in", fontsize=8)
            if float(np.std(z_in)) < 1e-5:
                ax1.text(
                    0.5, 0.5, "거의 상수 구간 → z≈0",
                    transform=ax1.transAxes, ha="center", fontsize=8, color="crimson",
                )
            ax1.legend(loc="upper right", fontsize=6)
            ax1.grid(True, alpha=0.3)
            row += 1

            ax2 = axes[row]
            ax2.plot(t, raw_in, color="C0", lw=0.5, alpha=0.5, label="raw")
            ax2.plot(t, pred_raw, color="C1", lw=0.8, label="recon → denorm µV")
            ax2.set_ylabel("recon", fontsize=8)
            ax2.legend(loc="upper right", fontsize=6)
            ax2.grid(True, alpha=0.3)
            row += 1
        else:
            ax_in = axes[row]
            ax_in.plot(t, z_in, color="C0", lw=0.8, label="VAE input (z-score only)")
            ax_in.set_ylabel(f"{lab}\ninput", fontsize=8)
            ax_in.legend(loc="upper right", fontsize=7)
            ax_in.grid(True, alpha=0.3)
            row += 1

            ax_r = axes[row]
            ax_r.plot(t, z_in, color="C0", lw=0.5, alpha=0.5, label="z input")
            ax_r.plot(t, pred_z, color="C1", lw=0.8, label="recon (z-space)")
            ax_r.set_ylabel("recon", fontsize=8)
            ax_r.legend(loc="upper right", fontsize=7)
            ax_r.grid(True, alpha=0.3)
            row += 1

    axes[-1].set_xlabel("time (s)")
    title = "Reconstruction: raw µV + z-score (VAE trains on z only)" if has_denorm else "Reconstruction (z-score space)"
    fig.suptitle(title, y=1.002, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_stage2_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    epochs = range(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train"], label="train noise MSE")
    ax.plot(epochs, history["val"], "--", label="val noise MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Stage 2 diffusion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_stage1_report(
    model: MultiHeadVAE,
    X: np.ndarray,
    y: np.ndarray,
    history: dict[str, list[dict[str, float]]],
    test_idx: np.ndarray | None = None,
    epoch_mean: np.ndarray | None = None,
    epoch_std: np.ndarray | None = None,
    out_dir: str | Path | None = None,
    show: bool = True,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """One-shot: loss curves, label dist, confusion matrix, recon samples."""
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"figures": []}
    results["figures"].append(
        plot_stage1_history(
            history,
            save_path=out_dir / "stage1_losses.png" if out_dir else None,
            show=show,
        )
    )
    results["figures"].append(
        plot_class_distribution(
            y,
            title="All loaded epochs",
            save_path=out_dir / "label_dist.png" if out_dir else None,
            show=show,
        )
    )

    if test_idx is not None and len(test_idx) > 0:
        test_m = evaluate_stage_test(model, X, y, test_idx, device=device)
        results["test_metrics"] = test_m
        results["figures"].append(
            plot_confusion_matrix(
                test_m["confusion_matrix"],
                save_path=out_dir / "confusion_matrix.png" if out_dir else None,
                show=show,
            )
        )

    rng = np.random.default_rng(0)
    sample_idx = test_idx if test_idx is not None and len(test_idx) else np.arange(len(y))
    pick = rng.choice(sample_idx, size=min(4, len(sample_idx)), replace=False)
    results["figures"].append(
        plot_reconstruction_samples(
            model, X, y, indices=pick, epoch_mean=epoch_mean, epoch_std=epoch_std,
            device=device,
            save_path=out_dir / "reconstruction.png" if out_dir else None,
            show=show,
        )
    )
    return results
