"""
전처리 품질 점검: raw µV / z-score / 역정규화 일치 여부, 상수 구간·std≈0 탐지.

사용 예:
    from src.preprocess import build_dataset
    from src.preprocess_inspect import summarize_preprocessing, plot_preprocess_audit

    X, y, subs, epoch_mean, epoch_std = build_dataset(DATA_ROOT)
    summarize_preprocessing(X, y, epoch_mean, epoch_std)
    plot_preprocess_audit(X, y, epoch_mean, epoch_std, show=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import PreprocessConfig
from .preprocess import STAGE_NAMES, denorm_microvolts, load_psg_epochs


def summarize_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    epoch_mean: np.ndarray,
    epoch_std: np.ndarray,
    *,
    z_flat_thresh: float = 1e-5,
    std_tiny_thresh: float = 1e-6,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    데이터셋 전체 통계. raw가 0으로 보이는 경우(상수 구간, std≈0) 개수를 집계.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X[:, 0, :]
    y = np.asarray(y)
    mu = np.asarray(epoch_mean, dtype=np.float64)
    sig = np.asarray(epoch_std, dtype=np.float64)

    raw = denorm_microvolts(X, mu[:, None], sig[:, None])
    z_std = X.std(axis=1)
    raw_std = raw.std(axis=1)
    raw_ptp = np.ptp(raw, axis=1)

    flat_z = z_std < z_flat_thresh
    tiny_sig = sig < std_tiny_thresh
    flat_raw = raw_std < std_tiny_thresh

    per_stage: dict[str, dict[str, float]] = {}
    for i, name in enumerate(STAGE_NAMES):
        mask = y == i
        if not mask.any():
            continue
        per_stage[name] = {
            "n": int(mask.sum()),
            "raw_std_median_uV": float(np.median(raw_std[mask])),
            "raw_ptp_median_uV": float(np.median(raw_ptp[mask])),
            "z_std_median": float(np.median(z_std[mask])),
            "frac_flat_z": float(flat_z[mask].mean()),
        }

    summary = {
        "n_epochs": len(y),
        "z_std_median": float(np.median(z_std)),
        "z_std_min": float(z_std.min()),
        "raw_std_median_uV": float(np.median(raw_std)),
        "raw_ptp_median_uV": float(np.median(raw_ptp)),
        "raw_abs_max_uV": float(np.max(np.abs(raw))),
        "n_flat_z": int(flat_z.sum()),
        "n_tiny_epoch_std": int(tiny_sig.sum()),
        "n_flat_raw_uV": int(flat_raw.sum()),
        "per_stage": per_stage,
    }

    if verbose:
        print("=== Preprocessing audit ===")
        print(f"epochs: {summary['n_epochs']}")
        print(
            f"z-score per epoch: std median={summary['z_std_median']:.4f} "
            f"min={summary['z_std_min']:.4e}"
        )
        print(
            f"raw (denorm µV): std median={summary['raw_std_median_uV']:.4f} "
            f"ptp median={summary['raw_ptp_median_uV']:.4f} "
            f"|max|={summary['raw_abs_max_uV']:.4f}"
        )
        print(
            f"suspicious: flat z ({summary['n_flat_z']}), "
            f"tiny epoch_std ({summary['n_tiny_epoch_std']}), "
            f"flat raw ({summary['n_flat_raw_uV']})"
        )
        if summary["raw_abs_max_uV"] < 5.0:
            print(
                "  [WARN] 전체 raw |max| < 5 µV → hypnogram 정렬/EDF 로드 문제. "
                "preprocess.py 수정 후 build_dataset()을 다시 실행하세요."
            )
        if summary["n_flat_z"] or summary["n_flat_raw_uV"]:
            print(
                "  → VAE 입력·raw 플롯이 0 직선이면 상수 구간이거나 "
                "epoch_mean/std 미전달·구버전 전처리 캐시 가능성을 확인하세요."
            )
        print("--- per stage ---")
        for name, st in per_stage.items():
            print(
                f"  {name:3s} n={st['n']:5d}  raw_std_med={st['raw_std_median_uV']:.3f} µV  "
                f"raw_ptp_med={st['raw_ptp_median_uV']:.3f} µV  "
                f"z_std_med={st['z_std_median']:.3f}  flat_z={st['frac_flat_z']:.1%}"
            )

    return summary


def _pick_example_indices(
    y: np.ndarray,
    epoch_std: np.ndarray,
    *,
    n_per_stage: int = 1,
    strategy: str = "median_raw_std",
    seed: int = 0,
) -> list[int]:
    """단계별 예시: median_raw_std(대표) | spread(min·med·max raw σ)."""
    rng = np.random.default_rng(seed)
    sig = np.asarray(epoch_std)
    indices: list[int] = []

    for stage in range(len(STAGE_NAMES)):
        pool = np.where(y == stage)[0]
        if len(pool) == 0:
            continue
        order = pool[np.argsort(sig[pool])]
        if strategy == "median_raw_std":
            picks = [int(order[len(order) // 2])]
        elif strategy == "spread":
            picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if n_per_stage > len(picks):
            extra = rng.choice(pool, size=min(n_per_stage - len(picks), len(pool)), replace=False)
            picks.extend(int(i) for i in extra)

        seen: set[int] = set()
        for idx in picks:
            if idx in seen:
                continue
            indices.append(idx)
            seen.add(idx)
            if sum(1 for i in indices if y[i] == stage) >= n_per_stage:
                break

    return indices


def plot_preprocess_audit(
    X: np.ndarray,
    y: np.ndarray,
    epoch_mean: np.ndarray,
    epoch_std: np.ndarray,
    indices: np.ndarray | list[int] | None = None,
    *,
    n_per_stage: int = 1,
    pick_strategy: str = "median_raw_std",
    sfreq: float = 100.0,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    에포크별 4행: raw µV | z-score | raw vs 역정규화 일치 | 통계 텍스트.
    indices 미지정 시 단계별 median·min·max raw_std 예시를 고름.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X[:, 0, :]
    y = np.asarray(y)
    mu = np.asarray(epoch_mean, dtype=np.float64)
    sig = np.asarray(epoch_std, dtype=np.float64)

    if indices is None:
        indices = _pick_example_indices(
            y, sig, n_per_stage=n_per_stage, strategy=pick_strategy
        )
    indices = list(np.asarray(indices).astype(int))

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(16, 2.2 * n), squeeze=False)
    t = np.arange(X.shape[1]) / sfreq

    for row, idx in enumerate(indices):
        z = X[idx]
        m, s = float(mu[idx]), float(sig[idx])
        raw = denorm_microvolts(z, m, s)
        raw_from_store = raw
        z_check = (raw_from_store - m) / s
        z_err = float(np.max(np.abs(z_check - z)))

        ax0, ax1, ax2, ax3 = axes[row]
        lab = STAGE_NAMES[int(y[idx])]

        ax0.plot(t, raw, color="C0", lw=0.7)
        ax0.set_ylabel(f"{lab}\nµV", fontsize=8)
        ax0.set_title(f"idx={idx} raw (denorm)", fontsize=9)
        ax0.grid(True, alpha=0.3)
        if raw.std() < 1e-6:
            ax0.text(
                0.5, 0.5, "flat raw",
                transform=ax0.transAxes, ha="center", color="crimson", fontsize=9,
            )

        ax1.plot(t, z, color="C2", lw=0.7)
        ax1.set_ylabel("z-score", fontsize=8)
        ax1.set_title("VAE input (stored X)", fontsize=9)
        ax1.grid(True, alpha=0.3)
        if z.std() < 1e-5:
            ax1.text(
                0.5, 0.5, "z ≈ 0 (const epoch?)",
                transform=ax1.transAxes, ha="center", color="crimson", fontsize=9,
            )

        ax2.plot(t, raw, color="C0", lw=0.6, alpha=0.6, label="denorm")
        ax2.plot(t, denorm_microvolts(z, m, s), color="C1", lw=0.8, ls="--", label="z*σ+μ")
        ax2.set_ylabel("match", fontsize=8)
        ax2.legend(fontsize=6, loc="upper right")
        ax2.grid(True, alpha=0.3)

        stats = (
            f"idx {idx}  {lab}\n"
            f"μ={m:.4g} µV  σ={s:.4g} µV\n"
            f"raw: min={raw.min():.4g} max={raw.max():.4g} "
            f"std={raw.std():.4g} ptp={np.ptp(raw):.4g}\n"
            f"z:   min={z.min():.4g} max={z.max():.4g} std={z.std():.4f}\n"
            f"|re-z - z|_max = {z_err:.2e}"
        )
        ax3.axis("off")
        ax3.text(0.02, 0.95, stats, va="top", fontsize=8, family="monospace")
        if z_err > 1e-3:
            ax3.text(0.02, 0.15, "WARN: denorm mismatch", color="crimson", fontsize=9)

    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[-1, 2].set_xlabel("time (s)")
    fig.suptitle("Preprocessing audit (raw µV vs z-score vs denorm check)", fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_edf_pipeline_compare(
    psg_path: str | Path,
    hyp_path: str | Path,
    cfg: PreprocessConfig | None = None,
    *,
    stage: int | None = None,
    example_rank: str = "median",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    EDF 한 파일을 파이프라인과 동일하게 다시 로드해, build_dataset 결과와 비교.
    stage 지정 시 해당 라벨 에포크 중 raw_std median/min/max 중 하나 선택.
    """
    cfg = cfg or PreprocessConfig()
    X_norm, y, mu, sig = load_psg_epochs(Path(psg_path), Path(hyp_path), cfg)
    if len(y) == 0:
        raise RuntimeError(f"No epochs in {psg_path}")

    raw = denorm_microvolts(X_norm, mu[:, None], sig[:, None])
    raw_std = raw.std(axis=1)

    if stage is not None:
        pool = np.where(y == stage)[0]
        if len(pool) == 0:
            raise ValueError(f"No epochs for stage {STAGE_NAMES[stage]} in {psg_path.name}")
    else:
        pool = np.arange(len(y))

    order = pool[np.argsort(raw_std[pool])]
    if example_rank == "min":
        idx = int(order[0])
    elif example_rank == "max":
        idx = int(order[-1])
    else:
        idx = int(order[len(order) // 2])

    z = X_norm[idx]
    r = raw[idx]
    t = np.arange(z.shape[0]) / cfg.target_sfreq
    lab = STAGE_NAMES[int(y[idx])]

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    axes[0].plot(t, r, lw=0.7, color="C0")
    axes[0].set_ylabel("µV")
    axes[0].set_title(f"{Path(psg_path).name}  epoch={idx}  {lab}  (reload from EDF)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, z, lw=0.7, color="C2")
    axes[1].set_ylabel("z-score")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title(
        f"μ={mu[idx]:.4g} σ={sig[idx]:.4g}  raw_std={raw_std[idx]:.4g}  z_std={z.std():.4f}"
    )
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def run_preprocess_audit(
    data_root: str | Path | None = None,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    epoch_mean: np.ndarray | None = None,
    epoch_std: np.ndarray | None = None,
    *,
    cfg: PreprocessConfig | None = None,
    subset: str | None = "cassette",
    max_subjects: int = 3,
    n_per_stage: int = 1,
    out_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """
    한 번에: (선택) 데이터 로드 → summarize → audit 플롯 → EDF 1건 직접 비교.
    """
    from .preprocess import build_dataset, iter_recordings

    cfg = cfg or PreprocessConfig()
    if X is None:
        if data_root is None:
            raise ValueError("Provide data_root or pre-loaded X, y, epoch_mean, epoch_std")
        X, y, _, epoch_mean, epoch_std = build_dataset(
            data_root, cfg=cfg, max_subjects=max_subjects, subset=subset
        )

    summary = summarize_preprocessing(X, y, epoch_mean, epoch_std)
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    fig_audit = plot_preprocess_audit(
        X, y, epoch_mean, epoch_std,
        n_per_stage=n_per_stage,
        save_path=out_dir / "preprocess_audit.png" if out_dir else None,
        show=show,
    )

    fig_edf = None
    if data_root is not None:
        data_root = Path(data_root)
        for psg, hyp in iter_recordings(data_root, subset=subset):
            try:
                fig_edf = plot_edf_pipeline_compare(
                    psg, hyp, cfg,
                    save_path=out_dir / "preprocess_edf_reload.png" if out_dir else None,
                    show=show,
                )
                break
            except Exception as exc:
                print(f"EDF compare skip {psg.name}: {exc}")

    return {"summary": summary, "figures": [fig_audit, fig_edf]}
