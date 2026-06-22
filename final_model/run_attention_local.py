"""Local runner for the Sleep-EDF subwindow-attention Stage 1 model.

Run from the code-attention directory:

    python run_attention_local.py --epochs 2 --max-subjects 3

The attention path is enabled by default through
Stage1Config(use_subwindow_encoder=True). It splits each 30-second epoch into
short windows, encodes each window, and attention-pools the window latents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")
BINARY_STAGE_NAMES = ("Wake", "Sleep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Sleep-EDF Stage 1 VAE with subwindow attention locally."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing sleep-cassette/ and sleep-telemetry/.",
    )
    parser.add_argument(
        "--subset",
        choices=("cassette", "telemetry", "all"),
        default="cassette",
        help="Dataset subset to load. Use 'all' for cassette + telemetry.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=3,
        help="Number of PSG recordings to load for a local smoke run. Use 0 for all.",
    )
    parser.add_argument(
        "--load-subject",
        default=None,
        help="Load only one recording subject such as SC4272 instead of scanning/loading many subjects.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument(
        "--resume-stage1",
        type=Path,
        default=None,
        help="Resume/fine-tune Stage 1 from a checkpoint such as checkpoints/stage1_attention/best_vae.pt.",
    )
    parser.add_argument(
        "--eval-only-stage1",
        action="store_true",
        help="Skip training and regenerate Stage 1 test/transition confusion matrices from a checkpoint.",
    )
    parser.add_argument(
        "--train-stage2-only",
        action="store_true",
        help="Skip Stage 1 train/eval and train Stage 2 directly from a Stage 1 checkpoint.",
    )
    parser.add_argument(
        "--train-stage2-multi-only",
        action="store_true",
        help="Skip Stage 1 train/eval and train one-shot multi-horizon Stage 2 diffusion.",
    )
    parser.add_argument(
        "--eval-ckpt",
        type=Path,
        default=None,
        help="Checkpoint for --eval-only-stage1. Defaults to SAVE_DIR/best_vae.pt.",
    )
    parser.add_argument(
        "--infer-step1-only",
        action="store_true",
        help=(
            "Load Stage 1 + Stage 2 checkpoints, run only non-AR step1 inference "
            "(x/context -> t+1), print statistics, save confusion matrices, and exit."
        ),
    )
    parser.add_argument(
        "--infer-multistep-only",
        action="store_true",
        help=(
            "Load Stage 1 + Stage 2 checkpoints and run autoregressive diffusion "
            "multi-step inference up to --infer-horizons."
        ),
    )
    parser.add_argument(
        "--infer-multistep-direct-only",
        action="store_true",
        help=(
            "Load Stage 1 + multi-horizon Stage 2 checkpoints and generate "
            "t+1..t+n in one diffusion process."
        ),
    )
    parser.add_argument(
        "--infer-horizons",
        type=int,
        default=3,
        help="Number of autoregressive Stage 2 diffusion horizons for --infer-multistep-only.",
    )
    parser.add_argument(
        "--train-latent-mlp-only",
        action="store_true",
        help=(
            "Skip Stage 1/2 training and train a frozen Stage 1 latent mu_t -> MLP -> y_{t+1} "
            "next-step predictor."
        ),
    )
    parser.add_argument(
        "--vae-ckpt",
        type=Path,
        default=None,
        help="Stage 1 VAE checkpoint for --infer-step1-only or --train-stage2-only. Defaults to SAVE_DIR/best_vae.pt.",
    )
    parser.add_argument(
        "--diffusion-ckpt",
        type=Path,
        default=None,
        help="Stage 2 diffusion checkpoint for --infer-step1-only. Defaults to STAGE2_SAVE_DIR/diffusion.pt.",
    )
    parser.add_argument(
        "--inference-save-dir",
        type=Path,
        default=None,
        help="Output directory for --infer-step1-only confusion matrices and summary.",
    )
    parser.add_argument(
        "--latent-mlp-save-dir",
        type=Path,
        default=CODE_ROOT / "checkpoints" / "latent_mlp_next_step",
        help="Output directory for --train-latent-mlp-only checkpoint, confusion matrices, and summary.",
    )
    parser.add_argument(
        "--latent-mlp-epochs",
        type=int,
        default=20,
        help="Latent MLP training epochs.",
    )
    parser.add_argument(
        "--latent-mlp-batch-size",
        type=int,
        default=128,
        help="Latent MLP mini-batch size.",
    )
    parser.add_argument(
        "--latent-mlp-lr",
        type=float,
        default=1e-3,
        help="Latent MLP learning rate.",
    )
    parser.add_argument(
        "--latent-mlp-hidden-dim",
        type=int,
        default=256,
        help="Latent MLP hidden dimension.",
    )
    parser.add_argument(
        "--latent-mlp-dropout",
        type=float,
        default=0.2,
        help="Latent MLP dropout probability.",
    )
    parser.add_argument(
        "--infer-subject",
        default=None,
        help=(
            "For --infer-step1-only, load the full dataset/split but evaluate only this "
            "test subject, e.g. SC4272. If omitted, the largest test subject is used."
        ),
    )
    parser.add_argument(
        "--infer-all-test",
        action="store_true",
        help="For --infer-step1-only, evaluate the full subject-wise test split instead of one subject.",
    )
    parser.add_argument(
        "--infer-split",
        choices=("train", "val", "test"),
        default="test",
        help="For --infer-step1-only, evaluate a full split. Use 'val' for the full validation split.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent vector size.")
    parser.add_argument("--base-channels", type=int, default=32, help="CNN base channel count.")
    parser.add_argument(
        "--subwindow-sec",
        type=float,
        default=6.0,
        help="Seconds per attention subwindow inside each 30-second epoch.",
    )
    parser.add_argument(
        "--sliding-epoch-stride-sec",
        type=float,
        default=None,
        help=(
            "If set, build overlapping 30-second input windows with this stride. "
            "Example: 5 means 0-30, 5-35, 10-40. Labels use the window center time."
        ),
    )
    parser.add_argument(
        "--transition-sliding-only",
        action="store_true",
        help="Keep normal 30-second epochs and add sliding windows only near stage transitions.",
    )
    parser.add_argument(
        "--transition-sliding-context-sec",
        type=float,
        default=60.0,
        help="Seconds around a stage boundary where sliding windows are added.",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Disable subwindow attention and run the plain full-epoch encoder.",
    )
    parser.add_argument("--lambda-rec", type=float, default=0.3, help="Stage 1 reconstruction MSE weight.")
    parser.add_argument("--lambda-spec", type=float, default=0.2, help="Stage 1 STFT magnitude loss weight.")
    parser.add_argument("--lambda-band", type=float, default=0.1, help="Stage 1 bandpower loss weight.")
    parser.add_argument("--lambda-sigma", type=float, default=0.05, help="Stage 1 sigma envelope loss weight.")
    parser.add_argument("--lambda-stage", type=float, default=3.0, help="Stage 1 sleep-stage CE loss weight.")
    parser.add_argument("--lambda-kl", type=float, default=3e-4, help="Stage 1 VAE KL loss weight.")
    parser.add_argument("--kl-warmup-epochs", type=int, default=10, help="Stage 1 KL warmup epochs.")
    parser.add_argument("--wake-loss-weight", type=float, default=1.0, help="Stage 1 W-epoch sample loss multiplier.")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights for Stage 1 CE.",
    )
    parser.add_argument(
        "--stage-class-weight-multiplier",
        nargs=5,
        type=float,
        metavar=("W", "N1", "N2", "N3", "REM"),
        default=(0.25, 0.75, 1.0, 1.0, 1.0),
        help="Five Stage 1 class-weight multipliers in W N1 N2 N3 REM order.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda or cpu. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=CODE_ROOT / "checkpoints" / "stage1_attention",
        help="Directory for checkpoints and run summary.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--first-subjects",
        action="store_true",
        help="Use the first N recordings instead of a seeded random sample.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--transition-window",
        type=int,
        default=1,
        help=(
            "Epochs before/after a stage-change boundary to include in the "
            "transition-only confusion matrix. Use 0 for exact boundary targets only."
        ),
    )
    parser.add_argument(
        "--train-stage2",
        action="store_true",
        help="Train Stage 2 diffusion immediately after Stage 1 finishes.",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=20,
        help="Stage 2 diffusion training epochs.",
    )
    parser.add_argument(
        "--stage2-batch-size",
        type=int,
        default=128,
        help="Stage 2 mini-batch size.",
    )
    parser.add_argument(
        "--stage2-context-len",
        type=int,
        default=5,
        help="Number of previous latents used as Stage 2 condition.",
    )
    parser.add_argument(
        "--stage2-horizons",
        type=int,
        default=3,
        help="Number of future latents generated jointly by multi-horizon Stage 2.",
    )
    parser.add_argument(
        "--stage2-lambda-next-stage",
        type=float,
        default=0.1,
        help="Weight for Stage 2 next-stage CE loss.",
    )
    parser.add_argument(
        "--stage2-sampling",
        choices=("transition", "stage_balanced", "shuffle"),
        default="transition",
        help=(
            "Stage 2 pair sampling. 'stage_balanced' samples pairs by target y_{t+1} "
            "with approximately equal W/N1/N2/N3/REM probability."
        ),
    )
    parser.add_argument(
        "--stage2-no-target-stage-weights",
        action="store_true",
        help="Disable Stage 2 inverse-frequency loss weights by target y_{t+1}.",
    )
    parser.add_argument(
        "--stage2-transition-wake-target-weight",
        type=float,
        default=1.0,
        help="Extra Stage 2 loss weight for exact transition pairs with target y_{t+1}=Wake.",
    )
    parser.add_argument(
        "--stage2-vae-ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for the Stage 2 VAE teacher encoder. Use 0 to disable EMA updates.",
    )
    parser.add_argument(
        "--stage2-save-dir",
        type=Path,
        default=CODE_ROOT / "checkpoints" / "stage2_attention_ctx5",
        help="Directory for Stage 2 diffusion checkpoint.",
    )
    parser.add_argument(
        "--stage2-multi-save-dir",
        type=Path,
        default=CODE_ROOT / "checkpoints" / "stage2_attention_multi_h3_ctx5",
        help="Directory for multi-horizon Stage 2 diffusion checkpoint.",
    )
    return parser.parse_args()


def load_stage1_model_from_checkpoint(ckpt_path: Path, device: str | None = None) -> Any:
    import torch

    from src.config import Stage1Config
    from src.models import MultiHeadVAE

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=device_t, weights_only=False)
    cfg = ckpt.get("cfg", Stage1Config())
    input_len = int(ckpt.get("input_len", 3000))
    subwindow_len = (
        int(round(input_len * cfg.subwindow_sec / 30.0))
        if getattr(cfg, "use_subwindow_encoder", False)
        else None
    )
    model = MultiHeadVAE(
        input_len=input_len,
        latent_dim=cfg.latent_dim,
        base_ch=cfg.base_channels,
        num_classes=cfg.num_stages,
        logvar_min=cfg.logvar_min,
        logvar_max=cfg.logvar_max,
        subwindow_len=subwindow_len,
        use_transformer_encoder=getattr(cfg, "use_transformer_encoder", False),
        transformer_layers=getattr(cfg, "transformer_layers", 2),
        transformer_heads=getattr(cfg, "transformer_heads", 4),
        transformer_dropout=getattr(cfg, "transformer_dropout", 0.1),
        transformer_cls_mean_pool=getattr(cfg, "transformer_cls_mean_pool", True),
    ).to(device_t)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def json_safe(value: Any) -> Any:
    module = value.__class__.__module__
    if module.startswith("numpy") and hasattr(value, "tolist"):
        return value.tolist()
    if module.startswith("numpy") and hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def transition_indices(y: Any, subject_ids: list[str], eval_idx: Any, window: int = 1) -> Any:
    """Return eval indices near within-subject stage-change boundaries."""
    import numpy as np

    y = np.asarray(y)
    eval_idx = np.asarray(eval_idx, dtype=np.int64)
    eval_set = set(int(i) for i in eval_idx)
    subj = np.asarray(subject_ids)
    selected: set[int] = set()
    window = max(0, int(window))

    for sid in np.unique(subj[eval_idx]):
        idxs = np.sort(eval_idx[subj[eval_idx] == sid])
        if len(idxs) < 2:
            continue
        for pos in range(len(idxs) - 1):
            left = int(idxs[pos])
            right = int(idxs[pos + 1])
            if int(y[left]) == int(y[right]):
                continue
            for offset in range(-window, window + 1):
                for center_pos in (pos, pos + 1):
                    mark_pos = center_pos + offset
                    if 0 <= mark_pos < len(idxs):
                        mark_idx = int(idxs[mark_pos])
                        if mark_idx in eval_set:
                            selected.add(mark_idx)

    return np.asarray(sorted(selected), dtype=np.int64)


def confusion_from_indices(model: Any, X: Any, y: Any, indices: Any, batch_size: int, device: str | None) -> dict[str, Any]:
    import numpy as np
    import torch

    y = np.asarray(y)
    indices = np.asarray(indices, dtype=np.int64)
    cm = np.zeros((len(STAGE_NAMES), len(STAGE_NAMES)), dtype=np.int64)
    if len(indices) == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "per_class_acc": {name: float("nan") for name in STAGE_NAMES},
            "confusion_matrix": cm,
            "n": 0,
        }

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_t).eval()
    if getattr(X, "ndim", None) == 3 and X.shape[1] != 1:
        X_arr = X.reshape(X.shape[0], -1)[:, None, :]
    else:
        X_arr = X[:, None, :] if getattr(X, "ndim", None) == 2 else X

    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = torch.from_numpy(X_arr[batch_idx]).float().to(device_t)
            mu, _ = model.encoder(xb)
            logits = model.stage_classifier(mu)
            pred = logits.argmax(dim=1).cpu().numpy()
            true = y[batch_idx].astype(np.int64)
            correct += int((pred == true).sum())
            total += len(true)
            for t, p in zip(true, pred):
                cm[int(t), int(p)] += 1

    per_class_acc = {}
    f1s = []
    for i, name in enumerate(STAGE_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
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
        "n": total,
    }


def save_confusion_outputs(
    metrics: dict[str, Any],
    save_dir: Path,
    stem: str,
    title: str,
    class_names: tuple[str, ...] = STAGE_NAMES,
) -> None:
    import numpy as np

    cm = metrics["confusion_matrix"]
    csv_path = save_dir / f"{stem}.csv"
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d", header=",".join(class_names), comments="")

    try:
        from src.visualize import plot_confusion_matrix

        plot_confusion_matrix(
            cm,
            class_names=class_names,
            save_path=save_dir / f"{stem}.png",
            show=False,
            normalize=True,
        )
        print(f"{title} plot:    {save_dir / f'{stem}.png'}")
    except Exception as exc:
        print(f"{title} plot skipped: {exc}")

    print(f"{title} counts:  {csv_path}")


def binary_wake_sleep_metrics(true: Any, pred: Any, mask: Any | None = None) -> dict[str, Any]:
    import numpy as np

    true = np.asarray(true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        true = true[mask]
        pred = pred[mask]

    true_bin = (true != 0).astype(np.int64)
    pred_bin = (pred != 0).astype(np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(true_bin, pred_bin):
        cm[int(t), int(p)] += 1

    correct = int(np.trace(cm))
    total = int(cm.sum())
    per_class_acc = {}
    f1s = []
    for i, name in enumerate(BINARY_STAGE_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
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
        "n": total,
    }


def select_test_subject_indices(
    subject_ids: list[str],
    test_idx: Any,
    subject_id: str | None = None,
) -> tuple[str, Any]:
    import numpy as np

    subj = np.asarray(subject_ids)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if len(test_idx) == 0:
        raise SystemExit("Test split is empty; cannot run one-subject inference.")

    test_subjects = sorted(str(sid) for sid in np.unique(subj[test_idx]))
    if subject_id is None:
        subject_id = max(test_subjects, key=lambda sid: int((subj[test_idx] == sid).sum()))
    if subject_id not in test_subjects:
        available = ", ".join(test_subjects)
        raise SystemExit(
            f"--infer-subject {subject_id!r} is not in the test split. "
            f"Available test subjects: {available}"
        )

    selected = np.sort(test_idx[subj[test_idx] == subject_id])
    if len(selected) < 2:
        raise SystemExit(
            f"Test subject {subject_id!r} has fewer than 2 epochs after preprocessing."
        )
    return subject_id, selected


def test_split_label(subject_ids: list[str], test_idx: Any) -> str:
    import numpy as np

    subj = np.asarray(subject_ids)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    test_subjects = sorted(str(sid) for sid in np.unique(subj[test_idx]))
    return f"all_test_{len(test_subjects)}subjects"


def split_label(split_name: str, subject_ids: list[str], idx: Any) -> str:
    import numpy as np

    subj = np.asarray(subject_ids)
    idx = np.asarray(idx, dtype=np.int64)
    split_subjects = sorted(str(sid) for sid in np.unique(subj[idx]))
    return f"all_{split_name}_{len(split_subjects)}subjects"


def main() -> None:
    args = parse_args()
    try:
        from src.config import PreprocessConfig, Stage1Config, Stage2Config
        from src.preprocess import load_sleep_edf_dataset, subject_wise_split
        from src.train_stage1 import train_stage1
        from src.train_stage2 import train_stage2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}\n"
            "Install local requirements first:\n"
            "    python -m pip install -r requirements.txt"
        ) from exc

    max_subjects = None if args.max_subjects == 0 else args.max_subjects
    subset = None if args.subset == "all" else args.subset

    pp_cfg = PreprocessConfig(
        max_subjects=max_subjects,
        random_subjects=not args.first_subjects,
        seed=args.seed,
        use_6x5_windows=not args.no_attention,
        window_sec=args.subwindow_sec,
        windows_per_epoch=5,
        sliding_epoch_stride_sec=args.sliding_epoch_stride_sec,
        transition_sliding_only=args.transition_sliding_only,
        transition_sliding_context_sec=args.transition_sliding_context_sec,
    )
    st_cfg = Stage1Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        seed=args.seed,
        use_subwindow_encoder=not args.no_attention,
        subwindow_sec=args.subwindow_sec,
        lambda_rec=args.lambda_rec,
        lambda_spec=args.lambda_spec,
        lambda_band=args.lambda_band,
        lambda_sigma=args.lambda_sigma,
        lambda_stage=args.lambda_stage,
        lambda_kl=args.lambda_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
        wake_loss_weight=args.wake_loss_weight,
        use_class_weights=not args.no_class_weights,
        stage_class_weight_multiplier=tuple(args.stage_class_weight_multiplier),
    )
    s2_cfg = Stage2Config(
        epochs=args.stage2_epochs,
        batch_size=args.stage2_batch_size,
        context_len=args.stage2_context_len,
        pair_stride_sec=(
            (args.sliding_epoch_stride_sec, 30.0)
            if args.sliding_epoch_stride_sec and args.transition_sliding_only
            else (args.sliding_epoch_stride_sec or 30.0)
        ),
        lambda_next_stage=args.stage2_lambda_next_stage,
        sampling=args.stage2_sampling,
        transition_weighted_sampling=args.stage2_sampling == "transition",
        use_target_stage_loss_weights=not args.stage2_no_target_stage_weights,
        transition_wake_target_weight=args.stage2_transition_wake_target_weight,
        vae_ema_decay=args.stage2_vae_ema_decay,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(f"CODE_ROOT: {CODE_ROOT}")
    print(f"DATA_ROOT: {args.data_root}")
    print(f"SAVE_DIR:  {args.save_dir}")
    print(f"Subset:    {args.subset}")
    print(f"Attention: {'on' if st_cfg.use_subwindow_encoder else 'off'}")
    print(f"Sliding epoch stride: {args.sliding_epoch_stride_sec or 'off'}")
    print(f"Transition-only sliding: {'on' if args.transition_sliding_only else 'off'}")

    X, y, subject_ids, epoch_onsets, epoch_mean, epoch_std, subwindow_y = load_sleep_edf_dataset(
        args.data_root,
        cfg=pp_cfg,
        max_subjects=max_subjects,
        subset=subset,
        subject_filter=args.load_subject,
        return_epoch_onsets=True,
        return_subwindow_labels=True,
    )
    train_idx, val_idx, test_idx = subject_wise_split(
        subject_ids,
        val_ratio=st_cfg.val_ratio,
        test_ratio=st_cfg.test_ratio,
        seed=args.seed,
    )
    print(
        "Split sizes: "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"subjects={len(set(subject_ids))}"
    )

    if args.train_stage2_only:
        vae_ckpt = args.vae_ckpt or args.eval_ckpt or (args.save_dir / "best_vae.pt")
        if not vae_ckpt.exists():
            raise SystemExit(f"Stage 1 VAE checkpoint not found: {vae_ckpt}")
        args.stage2_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Stage2-only VAE checkpoint: {vae_ckpt}")
        print(f"Stage 2 SAVE_DIR: {args.stage2_save_dir}")
        diffusion, stage2_history = train_stage2(
            X,
            subject_ids,
            train_idx,
            val_idx,
            vae_ckpt=vae_ckpt,
            y=y,
            epoch_onsets=epoch_onsets,
            cfg=s2_cfg,
            save_dir=args.stage2_save_dir,
            device=args.device,
        )
        stage2_path = args.stage2_save_dir / "diffusion.pt"
        print(f"Stage 2 model class: {diffusion.__class__.__name__}")
        print(f"Stage 2 checkpoint:  {stage2_path}")
        return

    if args.train_stage2_multi_only:
        from src.inference import evaluate_acc_n_direct_multi
        from src.train_stage2_multi import train_stage2_multi

        vae_ckpt = args.vae_ckpt or args.eval_ckpt or (args.save_dir / "best_vae.pt")
        if not vae_ckpt.exists():
            raise SystemExit(f"Stage 1 VAE checkpoint not found: {vae_ckpt}")
        args.stage2_multi_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Stage2-multi VAE checkpoint: {vae_ckpt}")
        print(f"Stage 2 MULTI SAVE_DIR:     {args.stage2_multi_save_dir}")
        diffusion, stage2_multi_history = train_stage2_multi(
            X,
            subject_ids,
            train_idx,
            val_idx,
            vae_ckpt=vae_ckpt,
            y=y,
            epoch_onsets=epoch_onsets,
            cfg=s2_cfg,
            save_dir=args.stage2_multi_save_dir,
            device=args.device,
            horizons=args.stage2_horizons,
        )
        diffusion_ckpt = args.stage2_multi_save_dir / "diffusion_multi.pt"
        print(f"Stage 2 MULTI model class: {diffusion.__class__.__name__}")
        print(f"Stage 2 MULTI checkpoint:  {diffusion_ckpt}")

        direct_metrics = evaluate_acc_n_direct_multi(
            vae_ckpt,
            diffusion_ckpt,
            X,
            y,
            subject_ids,
            test_idx,
            n=args.stage2_horizons,
            epoch_onsets=epoch_onsets,
            device=args.device,
            show_progress=not args.no_progress,
        )
        print(
            f"[Stage2 MULTI direct test Acc_{args.stage2_horizons}] "
            f"acc_n={direct_metrics['acc_n']:.4f} "
            f"acc={direct_metrics['accuracy']:.4f} "
            f"macro_f1={direct_metrics['macro_f1']:.4f}"
        )
        for h, acc in direct_metrics["per_horizon_acc"].items():
            print(f"  h{h}: acc={acc:.4f} n={direct_metrics['per_horizon_n'][h]}")
        save_confusion_outputs(
            direct_metrics,
            args.stage2_multi_save_dir,
            stem=f"stage2_multi_direct_h{args.stage2_horizons}_combined_confusion_matrix",
            title=f"Stage2 multi direct h1..h{args.stage2_horizons} combined confusion matrix",
        )
        summary = {
            "mode": "train_stage2_multi_only",
            "vae_ckpt": vae_ckpt,
            "diffusion_ckpt": diffusion_ckpt,
            "stage2_config": asdict(s2_cfg),
            "stage2_horizons": int(args.stage2_horizons),
            "stage2_multi_save_dir": args.stage2_multi_save_dir,
            "stage2_multi_history": stage2_multi_history,
            "direct_test_metrics": direct_metrics,
        }
        summary_path = args.stage2_multi_save_dir / "stage2_multi_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print(f"Stage 2 MULTI summary:     {summary_path}")
        return

    if args.eval_only_stage1:
        eval_ckpt = args.eval_ckpt or (args.save_dir / "best_vae.pt")
        if not eval_ckpt.exists():
            raise SystemExit(f"Stage 1 checkpoint not found: {eval_ckpt}")
        print(f"Eval-only Stage 1 checkpoint: {eval_ckpt}")
        model = load_stage1_model_from_checkpoint(eval_ckpt, device=args.device)

        test_metrics = confusion_from_indices(
            model,
            X,
            y,
            test_idx,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(
            "[TEST stage-only] "
            f"acc={test_metrics['accuracy']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f} "
            f"n={test_metrics['n']}"
        )
        for name, acc in test_metrics["per_class_acc"].items():
            print(f"  {name}: {acc:.3f}")
        save_confusion_outputs(
            test_metrics,
            args.save_dir,
            stem="confusion_matrix",
            title="Test confusion matrix",
        )

        trans_idx = transition_indices(y, subject_ids, test_idx, window=args.transition_window)
        trans_metrics = confusion_from_indices(
            model,
            X,
            y,
            trans_idx,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(
            "[TEST transition-only] "
            f"acc={trans_metrics['accuracy']:.4f} "
            f"macro_f1={trans_metrics['macro_f1']:.4f} "
            f"n={trans_metrics['n']} "
            f"window=+/-{args.transition_window}"
        )
        save_confusion_outputs(
            trans_metrics,
            args.save_dir,
            stem="transition_confusion_matrix",
            title="Transition confusion matrix",
        )
        if args.train_stage2:
            args.stage2_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Stage 2 SAVE_DIR: {args.stage2_save_dir}")
            diffusion, stage2_history = train_stage2(
                X,
                subject_ids,
                train_idx,
                val_idx,
                vae_ckpt=eval_ckpt,
                y=y,
                epoch_onsets=epoch_onsets,
                cfg=s2_cfg,
                save_dir=args.stage2_save_dir,
                device=args.device,
            )
            stage2_path = args.stage2_save_dir / "diffusion.pt"
            print(f"Stage 2 model class: {diffusion.__class__.__name__}")
            print(f"Stage 2 checkpoint:  {stage2_path}")
        return

    if args.train_latent_mlp_only:
        from src.train_latent_mlp import LatentMLPConfig, train_latent_mlp_next_step

        vae_ckpt = args.vae_ckpt or args.eval_ckpt or (args.save_dir / "best_vae.pt")
        if not vae_ckpt.exists():
            raise SystemExit(f"Stage 1 VAE checkpoint not found: {vae_ckpt}")
        if args.infer_step1_only or args.infer_multistep_only or args.infer_multistep_direct_only:
            raise SystemExit("Use latent MLP mode separately from diffusion inference modes.")

        latent_mlp_cfg = LatentMLPConfig(
            epochs=args.latent_mlp_epochs,
            batch_size=args.latent_mlp_batch_size,
            lr=args.latent_mlp_lr,
            hidden_dim=args.latent_mlp_hidden_dim,
            dropout=args.latent_mlp_dropout,
            seed=args.seed,
            segment_sec=(args.sliding_epoch_stride_sec or pp_cfg.segment_sec),
        )
        args.latent_mlp_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Latent MLP Stage 1 checkpoint: {vae_ckpt}")
        print(f"Latent MLP SAVE_DIR:          {args.latent_mlp_save_dir}")
        _, latent_mlp_history = train_latent_mlp_next_step(
            X,
            y,
            subject_ids,
            train_idx,
            val_idx,
            test_idx,
            vae_ckpt=vae_ckpt,
            epoch_onsets=epoch_onsets,
            cfg=latent_mlp_cfg,
            save_dir=args.latent_mlp_save_dir,
            device=args.device,
            show_progress=not args.no_progress,
        )
        latent_mlp_test = latent_mlp_history["test_metrics"]
        print(
            "[Latent MLP next-step TEST] "
            f"acc={latent_mlp_test['accuracy']:.4f} "
            f"balanced_acc={latent_mlp_test['balanced_acc']:.4f} "
            f"macro_f1={latent_mlp_test['macro_f1']:.4f} "
            f"n={latent_mlp_test['n']}"
        )
        print(
            "[Latent MLP next-step transition TEST] "
            f"acc={latent_mlp_test['transition_acc']:.4f} "
            f"stable_acc={latent_mlp_test['stable_acc']:.4f} "
            f"n_transition={latent_mlp_test['n_transition']} "
            f"n_stable={latent_mlp_test['n_stable']}"
        )
        print("[Latent MLP confusion matrix rows=true cols=pred]")
        print(latent_mlp_test["confusion_matrix"])
        print("[Latent MLP transition confusion matrix rows=true cols=pred]")
        print(latent_mlp_test["transition_confusion_matrix"])

        save_confusion_outputs(
            latent_mlp_test,
            args.latent_mlp_save_dir,
            stem="latent_mlp_confusion_matrix",
            title="Latent MLP next-step confusion matrix",
        )
        latent_mlp_transition = {
            **latent_mlp_test["transition_metrics"],
            "accuracy": latent_mlp_test["transition_acc"],
            "confusion_matrix": latent_mlp_test["transition_confusion_matrix"],
            "n": latent_mlp_test["n_transition"],
        }
        save_confusion_outputs(
            latent_mlp_transition,
            args.latent_mlp_save_dir,
            stem="latent_mlp_transition_confusion_matrix",
            title="Latent MLP next-step transition confusion matrix",
        )
        summary = {
            "mode": "train_latent_mlp_only",
            "vae_ckpt": vae_ckpt,
            "data_root": args.data_root,
            "subset": args.subset,
            "max_subjects": max_subjects,
            "latent_mlp_config": asdict(latent_mlp_cfg),
            "latent_mlp_save_dir": args.latent_mlp_save_dir,
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "history": latent_mlp_history,
        }
        summary_path = args.latent_mlp_save_dir / "latent_mlp_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print(f"Latent MLP checkpoint: {latent_mlp_history['best_checkpoint']}")
        print(f"Latent MLP summary:    {summary_path}")
        return

    if args.infer_multistep_direct_only:
        from src.inference import evaluate_acc_n_direct_multi

        vae_ckpt = args.vae_ckpt or (args.save_dir / "best_vae.pt")
        diffusion_ckpt = args.diffusion_ckpt or (args.stage2_multi_save_dir / "diffusion_multi.pt")
        if (args.infer_all_test or args.infer_split != "test") and args.infer_subject is not None:
            raise SystemExit("Use either a full split (--infer-all-test/--infer-split) or --infer-subject, not both.")
        if args.infer_all_test:
            selected_subject = test_split_label(subject_ids, test_idx)
            infer_idx = test_idx
        elif args.infer_split != "test":
            split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
            infer_idx = split_indices[args.infer_split]
            selected_subject = split_label(args.infer_split, subject_ids, infer_idx)
        else:
            selected_subject, infer_idx = select_test_subject_indices(
                subject_ids,
                test_idx,
                subject_id=args.infer_subject,
            )
        n_horizons = max(1, int(args.infer_horizons))
        inference_save_dir = args.inference_save_dir or (
            args.stage2_multi_save_dir / f"direct_multistep_h{n_horizons}_inference" / selected_subject
        )
        if not vae_ckpt.exists():
            raise SystemExit(f"Stage 1 VAE checkpoint not found: {vae_ckpt}")
        if not diffusion_ckpt.exists():
            raise SystemExit(f"Stage 2 multi diffusion checkpoint not found: {diffusion_ckpt}")
        inference_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Direct multi inference VAE checkpoint:       {vae_ckpt}")
        print(f"Direct multi inference diffusion checkpoint: {diffusion_ckpt}")
        print(f"Direct multi inference SAVE_DIR:             {inference_save_dir}")
        print(
            f"Direct multi inference target:               {selected_subject} "
            f"({len(infer_idx)} epochs), horizons=1..{n_horizons}"
        )
        direct_metrics = evaluate_acc_n_direct_multi(
            vae_ckpt,
            diffusion_ckpt,
            X,
            y,
            subject_ids,
            infer_idx,
            n=n_horizons,
            epoch_onsets=epoch_onsets,
            device=args.device,
            show_progress=not args.no_progress,
        )
        print(
            f"[Inference diffusion DIRECT-MULTI Acc_{n_horizons}] "
            f"acc_n={direct_metrics['acc_n']:.4f} "
            f"acc={direct_metrics['accuracy']:.4f} "
            f"balanced_acc={direct_metrics['balanced_acc']:.4f} "
            f"macro_f1={direct_metrics['macro_f1']:.4f}"
        )
        print("[Inference diffusion DIRECT-MULTI per-horizon acc]")
        for h, acc in direct_metrics["per_horizon_acc"].items():
            print(f"  h{h}: acc={acc:.4f} n={direct_metrics['per_horizon_n'][h]}")
        print("[Inference diffusion DIRECT-MULTI combined confusion matrix rows=true cols=pred]")
        print(direct_metrics["confusion_matrix"])
        save_confusion_outputs(
            direct_metrics,
            inference_save_dir,
            stem=f"inference_direct_multistep_h{n_horizons}_combined_confusion_matrix",
            title=f"Inference direct multi h1..h{n_horizons} combined confusion matrix",
        )
        summary = {
            "mode": "infer_multistep_direct_only",
            "vae_ckpt": vae_ckpt,
            "diffusion_ckpt": diffusion_ckpt,
            "data_root": args.data_root,
            "subset": args.subset,
            "max_subjects": max_subjects,
            "infer_all_test": args.infer_all_test,
            "infer_target": selected_subject,
            "infer_subject_requested": args.infer_subject,
            "infer_epochs": int(len(infer_idx)),
            "n_horizons": n_horizons,
            "metrics": direct_metrics,
        }
        summary_path = inference_save_dir / f"direct_multistep_h{n_horizons}_inference_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print(f"Direct multi inference summary:              {summary_path}")
        return

    if args.infer_multistep_only:
        from src.inference import SleepInferencePipeline, evaluate_acc_n_autoregressive

        vae_ckpt = args.vae_ckpt or (args.save_dir / "best_vae.pt")
        diffusion_ckpt = args.diffusion_ckpt or (args.stage2_save_dir / "diffusion.pt")
        if (args.infer_all_test or args.infer_split != "test") and args.infer_subject is not None:
            raise SystemExit("Use either a full split (--infer-all-test/--infer-split) or --infer-subject, not both.")
        if args.infer_all_test:
            selected_subject = test_split_label(subject_ids, test_idx)
            infer_idx = test_idx
        elif args.infer_split != "test":
            split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
            infer_idx = split_indices[args.infer_split]
            selected_subject = split_label(args.infer_split, subject_ids, infer_idx)
        else:
            selected_subject, infer_idx = select_test_subject_indices(
                subject_ids,
                test_idx,
                subject_id=args.infer_subject,
            )
        n_horizons = max(1, int(args.infer_horizons))
        inference_save_dir = args.inference_save_dir or (
            args.stage2_save_dir / f"multistep_h{n_horizons}_inference" / selected_subject
        )
        if not vae_ckpt.exists():
            raise SystemExit(f"Stage 1 VAE checkpoint not found: {vae_ckpt}")
        if not diffusion_ckpt.exists():
            raise SystemExit(f"Stage 2 diffusion checkpoint not found: {diffusion_ckpt}")
        inference_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Multistep inference VAE checkpoint:       {vae_ckpt}")
        print(f"Multistep inference diffusion checkpoint: {diffusion_ckpt}")
        print(f"Multistep inference SAVE_DIR:             {inference_save_dir}")
        print(
            f"Multistep inference target:               {selected_subject} "
            f"({len(infer_idx)} epochs), horizons=1..{n_horizons}"
        )
        pipe = SleepInferencePipeline.from_checkpoints(
            vae_ckpt,
            diffusion_ckpt,
            device=args.device,
        )
        multistep_metrics = evaluate_acc_n_autoregressive(
            pipe,
            X,
            y,
            subject_ids,
            infer_idx,
            n=n_horizons,
            epoch_onsets=epoch_onsets,
            show_progress=not args.no_progress,
        )
        print(
            f"[Inference diffusion AR Acc_{n_horizons}] "
            f"acc_n={multistep_metrics['acc_n']:.4f} "
            f"acc={multistep_metrics['accuracy']:.4f} "
            f"balanced_acc={multistep_metrics['balanced_acc']:.4f} "
            f"macro_f1={multistep_metrics['macro_f1']:.4f}"
        )
        print("[Inference diffusion AR per-horizon acc]")
        for h, acc in multistep_metrics["per_horizon_acc"].items():
            n_eval = multistep_metrics["per_horizon_n"][h]
            print(f"  h{h}: acc={acc:.4f} n={n_eval}")
        print("[Inference diffusion AR combined confusion matrix rows=true cols=pred]")
        print(multistep_metrics["confusion_matrix"])
        save_confusion_outputs(
            multistep_metrics,
            inference_save_dir,
            stem=f"inference_multistep_h{n_horizons}_combined_confusion_matrix",
            title=f"Inference diffusion AR h1..h{n_horizons} combined confusion matrix",
        )
        summary = {
            "mode": "infer_multistep_only",
            "vae_ckpt": vae_ckpt,
            "diffusion_ckpt": diffusion_ckpt,
            "data_root": args.data_root,
            "subset": args.subset,
            "max_subjects": max_subjects,
            "infer_all_test": args.infer_all_test,
            "infer_target": selected_subject,
            "infer_subject_requested": args.infer_subject,
            "infer_epochs": int(len(infer_idx)),
            "n_horizons": n_horizons,
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "metrics": multistep_metrics,
        }
        summary_path = inference_save_dir / f"multistep_h{n_horizons}_inference_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print(f"Multistep inference summary:              {summary_path}")
        return

    if args.infer_step1_only:
        from src.inference import SleepInferencePipeline, evaluate_step1_accuracy

        vae_ckpt = args.vae_ckpt or (args.save_dir / "best_vae.pt")
        diffusion_ckpt = args.diffusion_ckpt or (args.stage2_save_dir / "diffusion.pt")
        if (args.infer_all_test or args.infer_split != "test") and args.infer_subject is not None:
            raise SystemExit("Use either a full split (--infer-all-test/--infer-split) or --infer-subject, not both.")
        if args.infer_all_test:
            selected_subject = test_split_label(subject_ids, test_idx)
            infer_idx = test_idx
        elif args.infer_split != "test":
            split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
            infer_idx = split_indices[args.infer_split]
            selected_subject = split_label(args.infer_split, subject_ids, infer_idx)
        else:
            selected_subject, infer_idx = select_test_subject_indices(
                subject_ids,
                test_idx,
                subject_id=args.infer_subject,
            )
        inference_save_dir = args.inference_save_dir or (
            args.stage2_save_dir / "step1_inference" / selected_subject
        )
        if not vae_ckpt.exists():
            raise SystemExit(f"Stage 1 VAE checkpoint not found: {vae_ckpt}")
        if not diffusion_ckpt.exists():
            raise SystemExit(f"Stage 2 diffusion checkpoint not found: {diffusion_ckpt}")
        inference_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Step1 inference VAE checkpoint:       {vae_ckpt}")
        print(f"Step1 inference diffusion checkpoint: {diffusion_ckpt}")
        print(f"Step1 inference SAVE_DIR:             {inference_save_dir}")
        print(
            f"Step1 inference target:               {selected_subject} "
            f"({len(infer_idx)} test epochs)"
        )
        pipe = SleepInferencePipeline.from_checkpoints(
            vae_ckpt,
            diffusion_ckpt,
            device=args.device,
        )
        step1_metrics = evaluate_step1_accuracy(
            pipe,
            X,
            y,
            subject_ids,
            infer_idx,
            epoch_onsets=epoch_onsets,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
        )
        print(
            "[Inference step1 non-AR] "
            f"acc={step1_metrics['accuracy']:.4f} "
            f"balanced_acc={step1_metrics['balanced_acc']:.4f} "
            f"macro_f1={step1_metrics['macro_f1']:.4f} "
            f"n={step1_metrics['n']}"
        )
        print(
            "[Inference step1 transition] "
            f"transition_acc={step1_metrics['transition_acc']:.4f} "
            f"stable_acc={step1_metrics['stable_acc']:.4f} "
            f"near_transition_acc={step1_metrics['near_transition_acc']:.4f} "
            f"n_transition={step1_metrics['n_transition']} "
            f"n_stable={step1_metrics['n_stable']}"
        )
        print("[Inference step1 confusion matrix rows=true cols=pred]")
        print(step1_metrics["confusion_matrix"])
        print("[Inference step1 transition confusion matrix rows=true cols=pred]")
        print(step1_metrics["transition_confusion_matrix"])

        save_confusion_outputs(
            step1_metrics,
            inference_save_dir,
            stem="inference_step1_confusion_matrix",
            title="Inference step1 confusion matrix",
        )
        transition_metrics = {
            **step1_metrics,
            "accuracy": step1_metrics["transition_acc"],
            "confusion_matrix": step1_metrics["transition_confusion_matrix"],
            "n": step1_metrics["n_transition"],
        }
        save_confusion_outputs(
            transition_metrics,
            inference_save_dir,
            stem="inference_step1_transition_confusion_matrix",
            title="Inference step1 transition confusion matrix",
        )

        binary_metrics = binary_wake_sleep_metrics(
            step1_metrics["true"],
            step1_metrics["pred"],
        )
        binary_transition_metrics = binary_wake_sleep_metrics(
            step1_metrics["true"],
            step1_metrics["pred"],
            mask=step1_metrics["transition"],
        )
        binary_near_transition_metrics = binary_wake_sleep_metrics(
            step1_metrics["true"],
            step1_metrics["pred"],
            mask=step1_metrics["near_transition"],
        )
        binary_boundary_metrics = binary_wake_sleep_metrics(
            step1_metrics["true"],
            step1_metrics["pred"],
            mask=step1_metrics["binary_boundary"],
        )
        print(
            "[Inference step1 binary Wake/Sleep] "
            f"acc={binary_metrics['accuracy']:.4f} "
            f"macro_f1={binary_metrics['macro_f1']:.4f} "
            f"n={binary_metrics['n']}"
        )
        for name, acc in binary_metrics["per_class_acc"].items():
            print(f"  {name}: {acc:.3f}")
        print("[Inference step1 binary Wake/Sleep confusion matrix rows=true cols=pred]")
        print(binary_metrics["confusion_matrix"])
        print(
            "[Inference step1 binary Wake/Sleep transition] "
            f"acc={binary_transition_metrics['accuracy']:.4f} "
            f"macro_f1={binary_transition_metrics['macro_f1']:.4f} "
            f"n={binary_transition_metrics['n']}"
        )
        print("[Inference step1 binary Wake/Sleep transition confusion matrix rows=true cols=pred]")
        print(binary_transition_metrics["confusion_matrix"])
        print(
            "[Inference step1 binary Wake/Sleep near-transition] "
            f"acc={binary_near_transition_metrics['accuracy']:.4f} "
            f"macro_f1={binary_near_transition_metrics['macro_f1']:.4f} "
            f"n={binary_near_transition_metrics['n']}"
        )
        print("[Inference step1 binary Wake/Sleep near-transition confusion matrix rows=true cols=pred]")
        print(binary_near_transition_metrics["confusion_matrix"])
        print(
            "[Inference step1 binary Wake/Sleep W<->Sleep boundary only] "
            f"acc={binary_boundary_metrics['accuracy']:.4f} "
            f"macro_f1={binary_boundary_metrics['macro_f1']:.4f} "
            f"n={binary_boundary_metrics['n']}"
        )
        print("[Inference step1 binary Wake/Sleep W<->Sleep boundary confusion matrix rows=true cols=pred]")
        print(binary_boundary_metrics["confusion_matrix"])
        save_confusion_outputs(
            binary_metrics,
            inference_save_dir,
            stem="inference_step1_binary_wake_sleep_confusion_matrix",
            title="Inference step1 binary Wake/Sleep confusion matrix",
            class_names=BINARY_STAGE_NAMES,
        )
        save_confusion_outputs(
            binary_transition_metrics,
            inference_save_dir,
            stem="inference_step1_binary_wake_sleep_transition_confusion_matrix",
            title="Inference step1 binary Wake/Sleep transition confusion matrix",
            class_names=BINARY_STAGE_NAMES,
        )
        save_confusion_outputs(
            binary_near_transition_metrics,
            inference_save_dir,
            stem="inference_step1_binary_wake_sleep_near_transition_confusion_matrix",
            title="Inference step1 binary Wake/Sleep near-transition confusion matrix",
            class_names=BINARY_STAGE_NAMES,
        )
        save_confusion_outputs(
            binary_boundary_metrics,
            inference_save_dir,
            stem="inference_step1_binary_wake_sleep_boundary_only_confusion_matrix",
            title="Inference step1 binary Wake/Sleep W<->Sleep boundary-only confusion matrix",
            class_names=BINARY_STAGE_NAMES,
        )

        summary = {
            "mode": "infer_step1_only",
            "vae_ckpt": vae_ckpt,
            "diffusion_ckpt": diffusion_ckpt,
            "data_root": args.data_root,
            "subset": args.subset,
            "max_subjects": max_subjects,
            "infer_all_test": args.infer_all_test,
            "infer_target": selected_subject,
            "infer_subject_requested": args.infer_subject,
            "infer_test_epochs": int(len(infer_idx)),
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "metrics": step1_metrics,
            "binary_wake_sleep_metrics": binary_metrics,
            "binary_wake_sleep_transition_metrics": binary_transition_metrics,
            "binary_wake_sleep_near_transition_metrics": binary_near_transition_metrics,
            "binary_wake_sleep_boundary_only_metrics": binary_boundary_metrics,
        }
        summary_path = inference_save_dir / "step1_inference_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print(f"Step1 inference summary:              {summary_path}")
        return

    model, history = train_stage1(
        X,
        y,
        train_idx,
        val_idx,
        subject_ids=subject_ids,
        test_idx=test_idx,
        epoch_mean=epoch_mean,
        epoch_std=epoch_std,
        cfg=st_cfg,
        save_dir=args.save_dir,
        device=args.device,
        show_progress=not args.no_progress,
        resume_ckpt=args.resume_stage1,
        subwindow_y=subwindow_y,
    )

    test_metrics = confusion_from_indices(
        model,
        X,
        y,
        test_idx,
        batch_size=st_cfg.batch_size,
        device=args.device,
    )
    print(
        "[TEST stage-only confusion] "
        f"acc={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"n={test_metrics['n']}"
    )
    save_confusion_outputs(
        test_metrics,
        args.save_dir,
        stem="confusion_matrix",
        title="Test confusion matrix",
    )

    trans_idx = transition_indices(y, subject_ids, test_idx, window=args.transition_window)
    trans_metrics = confusion_from_indices(
        model,
        X,
        y,
        trans_idx,
        batch_size=st_cfg.batch_size,
        device=args.device,
    )
    print(
        "[TEST transition-only] "
        f"acc={trans_metrics['accuracy']:.4f} "
        f"macro_f1={trans_metrics['macro_f1']:.4f} "
        f"n={trans_metrics['n']} "
        f"window=+/-{args.transition_window}"
    )
    save_confusion_outputs(
        trans_metrics,
        args.save_dir,
        stem="transition_confusion_matrix",
        title="Transition confusion matrix",
    )

    stage2_history = None
    stage2_path = None
    if args.train_stage2:
        best_path = args.save_dir / "best_vae.pt"
        args.stage2_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Stage 2 SAVE_DIR: {args.stage2_save_dir}")
        diffusion, stage2_history = train_stage2(
            X,
            subject_ids,
            train_idx,
            val_idx,
            vae_ckpt=best_path,
            y=y,
            epoch_onsets=epoch_onsets,
            cfg=s2_cfg,
            save_dir=args.stage2_save_dir,
            device=args.device,
        )
        stage2_path = args.stage2_save_dir / "diffusion.pt"
        print(f"Stage 2 model class: {diffusion.__class__.__name__}")
        print(f"Stage 2 checkpoint:  {stage2_path}")

    summary = {
        "preprocess_config": asdict(pp_cfg),
        "stage1_config": asdict(st_cfg),
        "stage2_config": asdict(s2_cfg) if args.train_stage2 else None,
        "resume_stage1": args.resume_stage1,
        "data_root": args.data_root,
        "save_dir": args.save_dir,
        "stage2_save_dir": args.stage2_save_dir if args.train_stage2 else None,
        "n_epochs_total": int(len(y)),
        "n_subjects": int(len(set(subject_ids))),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "history": history,
        "test_confusion_metrics": test_metrics,
        "transition_test_metrics": trans_metrics,
        "transition_test_indices": trans_idx,
        "transition_window": args.transition_window,
        "stage2_history": stage2_history,
        "stage2_checkpoint": stage2_path,
    }
    summary_path = args.save_dir / "run_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    best_path = args.save_dir / "best_vae.pt"
    print(f"Done. Best checkpoint: {best_path}")
    if stage2_path is not None:
        print(f"Done. Stage 2 checkpoint: {stage2_path}")
    print(f"Run summary:          {summary_path}")
    print(f"Model class:          {model.__class__.__name__}")


if __name__ == "__main__":
    main()
