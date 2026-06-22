"""Stage 1 training loop for MultiHeadVAE."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import Stage1Config
from .models import MultiHeadVAE, stage1_losses
from .paths import default_checkpoint_dir
from .preprocess import STAGE_NAMES, validate_loaded_dataset
from .sampling import make_train_sampler


def _loss_kwargs(
    cfg: Stage1Config,
    class_weight: torch.Tensor | None,
    lambda_kl: float | None = None,
) -> dict:
    return {
        "lambda_rec": cfg.lambda_rec,
        "lambda_stage": cfg.lambda_stage,
        "lambda_kl": cfg.lambda_kl if lambda_kl is None else lambda_kl,
        "lambda_spec": cfg.lambda_spec,
        "lambda_band": cfg.lambda_band,
        "lambda_sigma": cfg.lambda_sigma,
        "wake_loss_weight": cfg.wake_loss_weight,
        "class_weight": class_weight,
        "sigma_stage_weights": cfg.sigma_stage_weights,
        "stft_n_fft": cfg.stft_n_fft,
        "stft_hop_length": cfg.stft_hop_length,
        "stft_win_length": cfg.stft_win_length,
        "subwindow_stage_loss_weight": cfg.subwindow_stage_loss_weight,
    }


def _make_loader(X: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int) -> DataLoader:
    if X.ndim == 3 and X.shape[1] != 1:
        X = X.reshape(X.shape[0], -1)
    X = X[:, None, :] if X.ndim == 2 else X
    ds = TensorDataset(
        torch.from_numpy(X[idx]),
        torch.from_numpy(y[idx]).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_stage1_dataset(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    subwindow_y: np.ndarray | None = None,
) -> TensorDataset:
    tensors = [
        torch.from_numpy(X[idx]),
        torch.from_numpy(y[idx]).long(),
    ]
    if subwindow_y is not None and subwindow_y.ndim == 2 and subwindow_y.shape[1] > 0:
        tensors.append(torch.from_numpy(subwindow_y[idx]).long())
    return TensorDataset(*tensors)


def _unpack_stage1_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = batch[0].to(device)
    y = batch[1].to(device)
    y_subwindows = batch[2].to(device) if len(batch) > 2 else None
    return x, y, y_subwindows


@torch.no_grad()
def evaluate_stage_test(
    model: MultiHeadVAE,
    X: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> dict:
    """Test set에서 수면 단계 분류만 평가 (encoder z -> stage head)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = _make_loader(X, y, test_idx, batch_size)

    num_classes = len(STAGE_NAMES)
    correct = 0
    n = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, yb in loader:
        x = x.to(device)
        yb = yb.to(device)
        mu, _ = model.encoder(x)
        logits = model.stage_classifier(mu)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += yb.size(0)
        for t, p in zip(yb.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1

    per_class_acc = {}
    for i, name in enumerate(STAGE_NAMES):
        support = cm[i].sum()
        per_class_acc[name] = float(cm[i, i] / support) if support > 0 else float("nan")

    f1s = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)

    return {
        "accuracy": correct / max(n, 1),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "n_test": n,
    }


@torch.no_grad()
def evaluate(
    model: MultiHeadVAE,
    loader: DataLoader,
    device: torch.device,
    cfg: Stage1Config,
    class_weight: torch.Tensor | None = None,
) -> dict:
    model.eval()
    totals = {
        "rec": 0.0,
        "spec": 0.0,
        "band": 0.0,
        "sigma": 0.0,
        "stage": 0.0,
        "kl": 0.0,
        "total": 0.0,
    }
    correct, n = 0, 0
    cm = np.zeros((cfg.num_stages, cfg.num_stages), dtype=np.int64)
    for batch in loader:
        x, y, y_subwindows = _unpack_stage1_batch(batch, device)
        out = model(x)
        out["x"] = x
        _, metrics = stage1_losses(
            out,
            y,
            **_loss_kwargs(cfg, class_weight),
            y_subwindows=y_subwindows,
        )
        for k in totals:
            totals[k] += metrics[k]
        logits_eval = model.stage_classifier(out["mu"])
        pred = logits_eval.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += y.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[int(t), int(p)] += 1
    for k in totals:
        totals[k] /= max(len(loader), 1)
    totals["acc"] = correct / max(n, 1)
    recalls = []
    f1s = []
    per_class_acc = {}
    for i in range(cfg.num_stages):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        if support > 0:
            recall_i = tp / support
            recalls.append(recall_i)
            per_class_acc[STAGE_NAMES[i]] = float(recall_i)
        else:
            per_class_acc[STAGE_NAMES[i]] = float("nan")
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    totals["balanced_acc"] = float(np.mean(recalls)) if recalls else 0.0
    totals["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    totals["per_class_acc"] = per_class_acc
    return totals


def train_stage1(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    subject_ids: list[str] | None = None,
    test_idx: np.ndarray | None = None,
    epoch_mean: np.ndarray | None = None,
    epoch_std: np.ndarray | None = None,
    cfg: Stage1Config | None = None,
    save_dir: str | Path | None = None,
    device: str | None = None,
    eval_test_stage: bool = True,
    validate_data: bool = True,
    show_progress: bool = True,
    resume_ckpt: str | Path | None = None,
    subwindow_y: np.ndarray | None = None,
) -> tuple[MultiHeadVAE, dict]:
    cfg = cfg or Stage1Config()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(save_dir or default_checkpoint_dir("stage1"))
    save_dir.mkdir(parents=True, exist_ok=True)

    if validate_data and epoch_mean is not None and epoch_std is not None:
        validate_loaded_dataset(X, y, epoch_mean, epoch_std)
        print("train_stage1: input X passed dataset validation", flush=True)
    elif validate_data:
        z_std_med = float(np.median(np.std(X.reshape(X.shape[0], -1), axis=1)))
        if z_std_med < 0.1:
            raise RuntimeError(
                f"X looks degenerate (z_std median={z_std_med:.4f}). "
                "Pass epoch_mean/epoch_std from load_sleep_edf_dataset() and "
                "re-run data loading — train_stage1 does not preprocess EDF."
            )

    if X.ndim == 3 and X.shape[1] != 1:
        print(f"train_stage1: flattening 6x5 window input {X.shape} -> (N, {X.shape[1] * X.shape[2]})")
        X = X.reshape(X.shape[0], -1)
    if X.ndim == 2:
        X = X[:, None, :]
    if subwindow_y is not None:
        subwindow_y = np.asarray(subwindow_y, dtype=np.int64)
        if subwindow_y.shape[0] != y.shape[0]:
            raise ValueError(f"subwindow_y first dim must match y ({y.shape[0]}), got {subwindow_y.shape}")
        if subwindow_y.ndim != 2:
            raise ValueError(f"subwindow_y must be 2D (N, n_subwindows), got {subwindow_y.shape}")
    train_ds = _make_stage1_dataset(X, y, train_idx, subwindow_y=subwindow_y)
    sampler = make_train_sampler(
        cfg.train_sampling,
        train_idx,
        y,
        subject_ids=subject_ids,
        num_stages=cfg.num_stages,
        seed=cfg.seed,
    )
    if sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=False, drop_last=True
        )
        if cfg.train_sampling == "stage_balanced":
            print("Train sampling: stage_balanced (W,N1,N2,N3,REM each 1/5)")
            for name, cnt in sampler.class_counts.items():
                print(f"  {name}: {cnt} train epochs")
            missing = [n for n, c in sampler.class_counts.items() if c == 0]
            if missing:
                print(f"  [warn] train에 없는 단계 → 실제 샘플은 나머지 단계만 균등: {missing}")
        elif cfg.train_sampling == "subject_balanced":
            print(f"Train sampling: subject_balanced ({len(sampler.subjects)} subjects)")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        print("Train sampling: shuffle (구간 수·클래스 빈도 비례, N2/N3 편향 가능)")
    train_eval_ds = _make_stage1_dataset(X, y, train_idx, subwindow_y=subwindow_y)
    val_ds = _make_stage1_dataset(X, y, val_idx, subwindow_y=subwindow_y)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MultiHeadVAE(
        input_len=X.shape[-1],
        latent_dim=cfg.latent_dim,
        base_ch=cfg.base_channels,
        num_classes=cfg.num_stages,
        logvar_min=cfg.logvar_min,
        logvar_max=cfg.logvar_max,
        subwindow_len=(int(round(X.shape[-1] * cfg.subwindow_sec / 30.0)) if cfg.use_subwindow_encoder else None),
        use_transformer_encoder=getattr(cfg, "use_transformer_encoder", False),
        transformer_layers=getattr(cfg, "transformer_layers", 2),
        transformer_heads=getattr(cfg, "transformer_heads", 4),
        transformer_dropout=getattr(cfg, "transformer_dropout", 0.1),
        transformer_cls_mean_pool=getattr(cfg, "transformer_cls_mean_pool", True),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ce_weight: torch.Tensor | None = None
    if cfg.use_class_weights:
        y_tr = y[train_idx]
        counts = np.bincount(y_tr, minlength=cfg.num_stages).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = 1.0 / counts
        w = w / w.mean()
        multiplier = np.asarray(cfg.stage_class_weight_multiplier, dtype=np.float64)
        if multiplier.shape != (cfg.num_stages,):
            raise ValueError(
                f"stage_class_weight_multiplier must have length {cfg.num_stages}, "
                f"got {multiplier.shape}"
            )
        w = w * multiplier
        w = w / w.mean()
        ce_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("Class CE weights:", dict(zip(STAGE_NAMES, w.round(3).tolist())), flush=True)
        print(
            "Class CE multipliers:",
            dict(zip(STAGE_NAMES, multiplier.round(3).tolist())),
            flush=True,
        )

    history: dict[str, list] = {"train": [], "val": []}
    test_metrics: dict | None = None
    maximize_metric = cfg.checkpoint_metric in {"val_acc", "val_balanced_acc", "val_macro_f1"}
    best_score = float("-inf") if maximize_metric else float("inf")
    start_epoch = 1
    end_epoch = cfg.epochs
    if resume_ckpt is not None:
        resume_path = Path(resume_ckpt)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            print(f"Resumed optimizer state from {resume_path}", flush=True)
        else:
            print(f"Resumed model weights from {resume_path} with a fresh optimizer", flush=True)
        history = ckpt.get("history", history)
        resumed_epoch = int(ckpt.get("epoch", 0))
        start_epoch = resumed_epoch + 1
        end_epoch = resumed_epoch + cfg.epochs
        metric_name = cfg.checkpoint_metric.removeprefix("val_")
        score_key = cfg.checkpoint_metric if cfg.checkpoint_metric in ckpt else metric_name
        if score_key in ckpt:
            best_score = float(ckpt[score_key])
        print(
            f"Stage1 resume: start_epoch={start_epoch}, "
            f"additional_epochs={cfg.epochs}, end_epoch={end_epoch}",
            flush=True,
        )
    print(
        f"Stage1 training start: device={device}, epochs={cfg.epochs}, "
        f"batch_size={cfg.batch_size}, train_batches={len(train_loader)}, "
        f"train_n={len(train_idx)}, val_n={len(val_idx)}",
        flush=True,
    )
    print(f"Checkpoint metric: {cfg.checkpoint_metric}", flush=True)
    print(
        f"Loss weights: λ_rec={cfg.lambda_rec} λ_spec={cfg.lambda_spec} "
        f"λ_band={cfg.lambda_band} λ_sigma={cfg.lambda_sigma} "
        f"λ_stage={cfg.lambda_stage} "
        f"λ_kl={cfg.lambda_kl}"
    )
    if cfg.kl_warmup_epochs > 0:
        print(f"KL warmup: {cfg.kl_warmup_epochs} epochs", flush=True)
    print(
        f"STFT (L_spec): n_fft={cfg.stft_n_fft} hop={cfg.stft_hop_length} "
        f"win={cfg.stft_win_length}"
    )
    print(f"Wake loss weight: W epochs x{cfg.wake_loss_weight}", flush=True)
    print(
        "Stage CE target: "
        + ("subwindow labels" if subwindow_y is not None and subwindow_y.shape[1] > 0 else "epoch center labels"),
        flush=True,
    )

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.perf_counter()
        model.train()
        running = {
            "rec": 0.0,
            "spec": 0.0,
            "stage": 0.0,
            "sigma": 0.0,
            "kl": 0.0,
            "total": 0.0,
        }
        if cfg.lambda_band > 0:
            running["band"] = 0.0
        batch_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{cfg.epochs}",
            leave=False,
            disable=not show_progress,
        )
        for batch in batch_iter:
            x, yb, yb_sub = _unpack_stage1_batch(batch, device)
            out = model(x)
            out["x"] = x
            kl_scale = min(1.0, epoch / max(cfg.kl_warmup_epochs, 1)) if cfg.kl_warmup_epochs > 0 else 1.0
            effective_lambda_kl = cfg.lambda_kl * kl_scale
            loss, metrics = stage1_losses(
                out,
                yb,
                **_loss_kwargs(cfg, ce_weight, lambda_kl=effective_lambda_kl),
                y_subwindows=yb_sub,
            )
            opt.zero_grad()
            loss.backward()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            opt.step()
            for k in running:
                running[k] += metrics[k]
            if show_progress:
                batch_iter.set_postfix(
                    total=f"{metrics['total']:.3f}",
                    rec=f"{metrics['rec']:.3f}",
                    band=f"{metrics.get('band', 0.0):.3f}",
                    sigma=f"{metrics.get('sigma', 0.0):.3f}",
                    stage=f"{metrics['stage']:.3f}",
                )

        for k in running:
            running[k] /= max(len(train_loader), 1)

        print(f"Epoch {epoch:03d}: evaluating train/val...", flush=True)
        train_m = evaluate(model, train_eval_loader, device, cfg, class_weight=ce_weight)
        val_m = evaluate(model, val_loader, device, cfg, class_weight=ce_weight)
        epoch_elapsed = time.perf_counter() - epoch_start
        sec_per_batch = epoch_elapsed / max(len(train_loader), 1)
        train_log = {
            **running,
            "acc": train_m["acc"],
            "balanced_acc": train_m["balanced_acc"],
            "macro_f1": train_m["macro_f1"],
        }
        history["train"].append(train_log)
        history["val"].append(dict(val_m))

        print(
            f"Epoch {epoch:03d} | train total={running['total']:.4f} "
            f"acc={train_m['acc']:.3f} bal={train_m['balanced_acc']:.3f} "
            f"mf1={train_m['macro_f1']:.3f} "
            f"(rec={running['rec']:.4f} spec={running['spec']:.4f} "
            f"band={running.get('band', 0.0):.4f} sigma={running['sigma']:.4f} "
            f"stage={running['stage']:.4f} "
            f"kl={running['kl']:.4f}) | "
            f"val total={val_m['total']:.4f} acc={val_m['acc']:.3f} "
            f"bal={val_m['balanced_acc']:.3f} mf1={val_m['macro_f1']:.3f} | "
            f"time={epoch_elapsed:.1f}s ({sec_per_batch:.3f}s/batch)",
            flush=True,
        )
        print(
            "  val recall: "
            + ", ".join(
                f"{name}={val_m['per_class_acc'][name]:.3f}"
                for name in STAGE_NAMES
            ),
            flush=True,
        )

        metric_name = cfg.checkpoint_metric.removeprefix("val_")
        if metric_name not in val_m:
            raise ValueError(f"Unknown checkpoint_metric: {cfg.checkpoint_metric}")
        current_score = val_m[metric_name]
        best_path = save_dir / "best_vae.pt"
        improved = current_score > best_score if maximize_metric else current_score < best_score
        if not best_path.exists():
            improved = True
        if improved:
            best_score = current_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": cfg,
                    "input_len": X.shape[-1],
                    "epoch": epoch,
                    "val_acc": val_m["acc"],
                    "val_balanced_acc": val_m["balanced_acc"],
                    "val_macro_f1": val_m["macro_f1"],
                    "val_total": val_m["total"],
                    "history": history,
                },
                best_path,
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "cfg": cfg,
                "input_len": X.shape[-1],
                "epoch": epoch,
                "val_acc": val_m["acc"],
                "val_balanced_acc": val_m["balanced_acc"],
                "val_macro_f1": val_m["macro_f1"],
                "val_total": val_m["total"],
                "best_score": best_score,
                "history": history,
            },
            save_dir / "last_vae.pt",
        )

    if test_idx is not None and eval_test_stage and len(test_idx) > 0:
        ckpt = torch.load(save_dir / "best_vae.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate_stage_test(model, X, y, test_idx, device, cfg.batch_size)
        print(
            f"[TEST stage-only] acc={test_metrics['accuracy']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f} n={test_metrics['n_test']}"
        )
        for name, acc in test_metrics["per_class_acc"].items():
            print(f"  {name}: {acc:.3f}")
    history["test_metrics"] = test_metrics
    return model, history
