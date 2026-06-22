"""Subject-wise Stage 1 K-fold runner for code-attention.

Run from the final_model directory:

    python run_kfold10_stage1.py --folds 20 --max-subjects 20 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from run_attention_local import (
    CODE_ROOT,
    REPO_ROOT,
    confusion_from_indices,
    json_safe,
    load_stage1_model_from_checkpoint,
    save_confusion_outputs,
    transition_indices,
)
from src.config import PreprocessConfig, Stage1Config
from src.preprocess import load_sleep_edf_dataset
from src.train_stage1 import train_stage1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-wise Stage 1 K-fold CV.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--subset", choices=("cassette", "telemetry", "all"), default="cassette")
    parser.add_argument("--max-subjects", type=int, default=20, help="0 means all recordings.")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--fold-start", type=int, default=0, help="Inclusive, 0-based.")
    parser.add_argument("--fold-end", type=int, default=None, help="Exclusive, 0-based.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--subwindow-sec", type=float, default=6.0)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-transformer-encoder", action="store_true")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--sliding-epoch-stride-sec", type=float, default=None)
    parser.add_argument("--transition-sliding-only", action="store_true")
    parser.add_argument("--transition-sliding-context-sec", type=float, default=60.0)
    parser.add_argument("--transition-window", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", type=Path, default=CODE_ROOT / "checkpoints" / "stage1_kfold20")
    parser.add_argument(
        "--resume-from-save-dir",
        type=Path,
        default=None,
        help="Resume each fold from RESUME_DIR/fold_XX/checkpoint-name when present.",
    )
    parser.add_argument("--resume-checkpoint-name", default="last_vae.pt")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--lambda-rec", type=float, default=0.3)
    parser.add_argument("--lambda-spec", type=float, default=0.2)
    parser.add_argument("--lambda-band", type=float, default=0.1)
    parser.add_argument("--lambda-sigma", type=float, default=0.05)
    parser.add_argument("--lambda-stage", type=float, default=3.0)
    parser.add_argument("--subwindow-stage-loss-weight", type=float, default=0.5)
    parser.add_argument("--lambda-kl", type=float, default=3e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--wake-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--train-sampling",
        choices=("shuffle", "stage_balanced", "subject_balanced"),
        default="shuffle",
        help="Stage 1 train sampler.",
    )
    parser.add_argument(
        "--stage-class-weight-multiplier",
        nargs=5,
        type=float,
        metavar=("W", "N1", "N2", "N3", "REM"),
        default=(0.25, 0.55, 1.0, 1.0, 1.0),
    )
    return parser.parse_args()


def make_subject_folds(subject_ids: list[str], n_folds: int, seed: int) -> list[np.ndarray]:
    unique = np.array(sorted(set(subject_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return [fold for fold in np.array_split(unique, n_folds) if len(fold) > 0]


def indices_for_subject_sets(
    subject_ids: list[str],
    train_subjects: np.ndarray,
    val_subjects: np.ndarray,
    test_subjects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subj = np.asarray(subject_ids, dtype=object)
    train_idx = np.where(np.isin(subj, train_subjects))[0]
    val_idx = np.where(np.isin(subj, val_subjects))[0]
    test_idx = np.where(np.isin(subj, test_subjects))[0]
    return train_idx, val_idx, test_idx


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    per_class_acc: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    f1s: list[float] = []
    weighted_f1_sum = 0.0
    weighted_f1_support = 0
    for i, name in enumerate(("W", "N1", "N2", "N3", "REM")):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        per_class_acc[name] = float(tp / support) if support > 0 else float("nan")
        f1 = float("nan")
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            f1s.append(f1)
        per_class_f1[name] = f1
        if support > 0 and not np.isnan(f1):
            weighted_f1_sum += f1 * support
            weighted_f1_support += support
    return {
        "accuracy": float(correct / total) if total > 0 else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "weighted_f1": float(weighted_f1_sum / weighted_f1_support) if weighted_f1_support > 0 else float("nan"),
        "per_class_acc": per_class_acc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "n": total,
    }


def load_existing_fold_summaries(save_dir: Path, n_folds: int) -> tuple[list[dict[str, Any]], list[int]]:
    summaries: list[dict[str, Any]] = []
    missing: list[int] = []
    for fold_i in range(n_folds):
        summary_path = save_dir / f"fold_{fold_i:02d}" / "fold_summary.json"
        if not summary_path.exists():
            missing.append(fold_i)
            continue
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    summaries.sort(key=lambda s: int(s["fold"]))
    return summaries, missing


def sum_confusion_matrices(summaries: list[dict[str, Any]], key: str, num_stages: int) -> np.ndarray:
    cm = np.zeros((num_stages, num_stages), dtype=np.int64)
    for summary in summaries:
        cm += np.asarray(summary[key]["confusion_matrix"], dtype=np.int64)
    return cm


def binary_wake_sleep_confusion_from_5class(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=np.int64)
    return np.asarray(
        [
            [cm[0, 0], cm[0, 1:].sum()],
            [cm[1:, 0].sum(), cm[1:, 1:].sum()],
        ],
        dtype=np.int64,
    )


def binary_metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    per_class_acc: dict[str, float] = {}
    f1s: list[float] = []
    for i, name in enumerate(("Wake", "Sleep")):
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
        "accuracy": float(correct / total) if total > 0 else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "n": total,
    }


def save_per_class_table_outputs(
    metrics: dict[str, Any],
    save_dir: Path,
    stem: str,
    title: str,
    class_names: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM"),
) -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(class_names):
        tp = float(cm[i, i])
        row_sum = float(cm[i, :].sum())
        col_sum = float(cm[:, i].sum())
        recall = 100.0 * tp / row_sum if row_sum > 0 else float("nan")
        precision = 100.0 * tp / col_sum if col_sum > 0 else float("nan")
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
        rows.append(
            {
                "Stage": name,
                **{cls: int(cm[i, j]) for j, cls in enumerate(class_names)},
                "RE": recall,
                "PR": precision,
                "F1": f1,
            }
        )

    csv_path = save_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        header = ["Stage", *class_names, "RE", "PR", "F1"]
        f.write(",".join(header) + "\n")
        for row in rows:
            values = [str(row["Stage"])]
            values.extend(str(row[cls]) for cls in class_names)
            values.extend(f"{row[key]:.2f}" for key in ("RE", "PR", "F1"))
            f.write(",".join(values) + "\n")

    try:
        import matplotlib.pyplot as plt

        headers = ["Stage", *class_names, "RE", "PR", "F1"]
        cell_text = []
        for row in rows:
            cell_text.append(
                [row["Stage"]]
                + [str(row[cls]) for cls in class_names]
                + [f"{row[key]:.2f}" for key in ("RE", "PR", "F1")]
            )
        fig_w = max(9.0, 1.05 * len(headers))
        fig_h = 1.2 + 0.45 * len(cell_text)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        table = ax.table(cellText=cell_text, colLabels=headers, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.35)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#555555")
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f2f2f2")
            if c == 0 and r > 0:
                cell.set_text_props(weight="bold")
        png_path = save_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"{title} per-class table: {png_path}")
    except Exception as exc:
        print(f"{title} per-class table plot skipped: {exc}")
    print(f"{title} per-class table counts: {csv_path}")


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    max_subjects = None if args.max_subjects == 0 else args.max_subjects
    subset = None if args.subset == "all" else args.subset
    args.save_dir.mkdir(parents=True, exist_ok=True)

    pp_cfg = PreprocessConfig(
        max_subjects=max_subjects,
        random_subjects=True,
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
        use_transformer_encoder=(not args.no_attention and not args.no_transformer_encoder),
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.transformer_dropout,
        lambda_rec=args.lambda_rec,
        lambda_spec=args.lambda_spec,
        lambda_band=args.lambda_band,
        lambda_sigma=args.lambda_sigma,
        lambda_stage=args.lambda_stage,
        subwindow_stage_loss_weight=args.subwindow_stage_loss_weight,
        lambda_kl=args.lambda_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
        wake_loss_weight=args.wake_loss_weight,
        use_class_weights=not args.no_class_weights,
        stage_class_weight_multiplier=tuple(args.stage_class_weight_multiplier),
        train_sampling=args.train_sampling,
    )

    print(f"CODE_ROOT: {CODE_ROOT}")
    print(f"DATA_ROOT: {args.data_root}")
    print(f"SAVE_DIR:  {args.save_dir}")
    if args.resume_from_save_dir is not None:
        print(f"RESUME_DIR: {args.resume_from_save_dir} ({args.resume_checkpoint_name})")
    print(f"K-fold:    folds={args.folds}, seed={args.seed}")
    print(f"Subset:    {args.subset}")
    print(f"Attention: {'on' if st_cfg.use_subwindow_encoder else 'off'}")
    print(f"Transformer encoder: {'on' if st_cfg.use_transformer_encoder else 'off'}")
    print(f"Sliding epoch stride: {args.sliding_epoch_stride_sec or 'off'}")
    print(f"Transition-only sliding: {'on' if args.transition_sliding_only else 'off'}")

    requested_fold_end = args.fold_end if args.fold_end is not None else args.folds
    requested_selected = range(max(0, args.fold_start), min(requested_fold_end, args.folds))
    if not requested_selected:
        print("No folds selected; rebuilding aggregate summary from existing fold_summary.json files.", flush=True)
        summaries, missing_folds = load_existing_fold_summaries(args.save_dir, args.folds)
        if missing_folds:
            print(
                "Warning: aggregate summary excludes missing fold summaries: "
                + ", ".join(f"{i:02d}" for i in missing_folds),
                flush=True,
            )
        if not summaries:
            raise SystemExit(f"No fold_summary.json files found under {args.save_dir}")

        aggregate_test_cm = sum_confusion_matrices(summaries, "test_metrics", st_cfg.num_stages)
        aggregate_transition_cm = sum_confusion_matrices(summaries, "transition_test_metrics", st_cfg.num_stages)
        aggregate_test_metrics = metrics_from_confusion(aggregate_test_cm)
        aggregate_transition_metrics = metrics_from_confusion(aggregate_transition_cm)
        aggregate_binary_test_metrics = binary_metrics_from_confusion(
            binary_wake_sleep_confusion_from_5class(aggregate_test_cm)
        )
        total_training_seconds = time.perf_counter() - run_started
        save_confusion_outputs(
            aggregate_test_metrics,
            args.save_dir,
            stem="all_folds_confusion_matrix",
            title="All folds test",
        )
        save_confusion_outputs(
            aggregate_transition_metrics,
            args.save_dir,
            stem="all_folds_transition_confusion_matrix",
            title="All folds transition",
        )
        save_confusion_outputs(
            aggregate_binary_test_metrics,
            args.save_dir,
            stem="all_folds_binary_wake_sleep_confusion_matrix",
            title="All folds binary Wake/Sleep test",
            class_names=("Wake", "Sleep"),
        )
        save_per_class_table_outputs(
            aggregate_test_metrics,
            args.save_dir,
            stem="all_folds_per_class_metrics_table",
            title="All folds test",
        )
        save_per_class_table_outputs(
            aggregate_transition_metrics,
            args.save_dir,
            stem="all_folds_transition_per_class_metrics_table",
            title="All folds transition",
        )

        aggregate = {
            "mode": f"stage1_subject_kfold{args.folds}",
            "data_root": args.data_root,
            "subset": args.subset,
            "max_subjects": max_subjects,
            "preprocess_config": asdict(pp_cfg),
            "stage1_config": asdict(st_cfg),
            "folds_requested": args.folds,
            "folds_run": [s["fold"] for s in summaries],
            "folds_trained_this_run": [],
            "missing_fold_summaries": missing_folds,
            "resume_from_save_dir": args.resume_from_save_dir,
            "resume_checkpoint_name": args.resume_checkpoint_name,
            "n_subjects": None,
            "n_epochs_total": None,
            "test_accuracy": mean_std([s["test_metrics"]["accuracy"] for s in summaries]),
            "test_macro_f1": mean_std([s["test_metrics"]["macro_f1"] for s in summaries]),
            "transition_accuracy": mean_std([s["transition_test_metrics"]["accuracy"] for s in summaries]),
            "transition_macro_f1": mean_std([s["transition_test_metrics"]["macro_f1"] for s in summaries]),
            "all_folds_test_metrics": aggregate_test_metrics,
            "all_folds_binary_wake_sleep_metrics": aggregate_binary_test_metrics,
            "all_folds_transition_metrics": aggregate_transition_metrics,
            "total_training_seconds": float(total_training_seconds),
            "fold_summaries": summaries,
        }
        summary_path = args.save_dir / f"kfold{args.folds}_summary.json"
        summary_path.write_text(json.dumps(json_safe(aggregate), indent=2), encoding="utf-8")
        print(f"\nK-fold summary: {summary_path}")
        print(
            "Aggregate | "
            f"test acc={aggregate['test_accuracy']['mean']:.4f}+/-{aggregate['test_accuracy']['std']:.4f} "
            f"mf1={aggregate['test_macro_f1']['mean']:.4f}+/-{aggregate['test_macro_f1']['std']:.4f} | "
            f"transition acc={aggregate['transition_accuracy']['mean']:.4f}+/-{aggregate['transition_accuracy']['std']:.4f} "
            f"mf1={aggregate['transition_macro_f1']['mean']:.4f}+/-{aggregate['transition_macro_f1']['std']:.4f}"
        )
        print(
            "Pooled weighted F1 | "
            f"test={aggregate_test_metrics['weighted_f1']:.4f} "
            f"transition={aggregate_transition_metrics['weighted_f1']:.4f}"
        )
        print(f"Summary rebuild time: {total_training_seconds:.1f} sec")
        return

    X, y, subject_ids, epoch_onsets, epoch_mean, epoch_std, subwindow_y = load_sleep_edf_dataset(
        args.data_root,
        cfg=pp_cfg,
        max_subjects=max_subjects,
        subset=subset,
        return_epoch_onsets=True,
        return_subwindow_labels=True,
    )

    folds = make_subject_folds(subject_ids, args.folds, args.seed)
    fold_end = args.fold_end if args.fold_end is not None else len(folds)
    selected = range(max(0, args.fold_start), min(fold_end, len(folds)))
    summaries: list[dict[str, Any]] = []
    folds_trained_this_run: list[int] = []

    for fold_i in selected:
        test_subjects = folds[fold_i]
        val_subjects = folds[(fold_i + 1) % len(folds)]
        train_subjects = np.concatenate(
            [folds[j] for j in range(len(folds)) if j not in {fold_i, (fold_i + 1) % len(folds)}]
        )
        train_idx, val_idx, test_idx = indices_for_subject_sets(
            subject_ids,
            train_subjects,
            val_subjects,
            test_subjects,
        )
        fold_dir = args.save_dir / f"fold_{fold_i:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        resume_ckpt = None
        if args.resume_from_save_dir is not None:
            candidate = args.resume_from_save_dir / f"fold_{fold_i:02d}" / args.resume_checkpoint_name
            if candidate.exists():
                resume_ckpt = candidate
                print(f"Resume fold {fold_i:02d} from {resume_ckpt}", flush=True)
            else:
                print(f"Resume checkpoint not found for fold {fold_i:02d}: {candidate}; training from scratch", flush=True)
        print(
            f"\nFold {fold_i:02d}/{len(folds) - 1:02d}: "
            f"train_n={len(train_idx)} val_n={len(val_idx)} test_n={len(test_idx)} "
            f"train_subjects={len(train_subjects)} val_subjects={len(val_subjects)} "
            f"test_subjects={len(test_subjects)}",
            flush=True,
        )

        _, history = train_stage1(
            X,
            y,
            train_idx,
            val_idx,
            subject_ids=subject_ids,
            test_idx=test_idx,
            epoch_mean=epoch_mean,
            epoch_std=epoch_std,
            cfg=st_cfg,
            save_dir=fold_dir,
            device=args.device,
            eval_test_stage=False,
            validate_data=False,
            show_progress=not args.no_progress,
            subwindow_y=subwindow_y,
            resume_ckpt=resume_ckpt,
        )
        eval_ckpt = fold_dir / "best_vae.pt"
        if not eval_ckpt.exists():
            eval_ckpt = fold_dir / "last_vae.pt"
            print(f"best_vae.pt not found for fold {fold_i:02d}; using {eval_ckpt}", flush=True)
        model = load_stage1_model_from_checkpoint(eval_ckpt, device=args.device)

        test_metrics = confusion_from_indices(
            model,
            X,
            y,
            test_idx,
            batch_size=st_cfg.batch_size,
            device=args.device,
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
        save_confusion_outputs(test_metrics, fold_dir, stem="confusion_matrix", title=f"Fold {fold_i:02d} test")
        save_confusion_outputs(
            trans_metrics,
            fold_dir,
            stem="transition_confusion_matrix",
            title=f"Fold {fold_i:02d} transition",
        )
        fold_summary = {
            "fold": fold_i,
            "train_subjects": train_subjects.tolist(),
            "val_subjects": val_subjects.tolist(),
            "test_subjects": test_subjects.tolist(),
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
                "transition_test": int(len(trans_idx)),
            },
            "test_metrics": test_metrics,
            "transition_test_metrics": trans_metrics,
            "history": history,
        }
        (fold_dir / "fold_summary.json").write_text(
            json.dumps(json_safe(fold_summary), indent=2),
            encoding="utf-8",
        )
        summaries.append(fold_summary)
        folds_trained_this_run.append(fold_i)
        print(
            f"Fold {fold_i:02d} done | "
            f"test acc={test_metrics['accuracy']:.4f} mf1={test_metrics['macro_f1']:.4f} | "
            f"transition acc={trans_metrics['accuracy']:.4f} mf1={trans_metrics['macro_f1']:.4f}",
            flush=True,
        )

    summaries, missing_folds = load_existing_fold_summaries(args.save_dir, len(folds))
    if missing_folds:
        print(
            "Warning: aggregate summary excludes missing fold summaries: "
            + ", ".join(f"{i:02d}" for i in missing_folds),
            flush=True,
        )
    if not summaries:
        raise SystemExit(f"No fold_summary.json files found under {args.save_dir}")

    aggregate_test_cm = sum_confusion_matrices(summaries, "test_metrics", st_cfg.num_stages)
    aggregate_transition_cm = sum_confusion_matrices(summaries, "transition_test_metrics", st_cfg.num_stages)
    aggregate_test_metrics = metrics_from_confusion(aggregate_test_cm)
    aggregate_transition_metrics = metrics_from_confusion(aggregate_transition_cm)
    aggregate_binary_test_metrics = binary_metrics_from_confusion(
        binary_wake_sleep_confusion_from_5class(aggregate_test_cm)
    )
    total_training_seconds = time.perf_counter() - run_started
    save_confusion_outputs(
        aggregate_test_metrics,
        args.save_dir,
        stem="all_folds_confusion_matrix",
        title="All folds test",
    )
    save_confusion_outputs(
        aggregate_transition_metrics,
        args.save_dir,
        stem="all_folds_transition_confusion_matrix",
        title="All folds transition",
    )
    save_confusion_outputs(
        aggregate_binary_test_metrics,
        args.save_dir,
        stem="all_folds_binary_wake_sleep_confusion_matrix",
        title="All folds binary Wake/Sleep test",
        class_names=("Wake", "Sleep"),
    )
    save_per_class_table_outputs(
        aggregate_test_metrics,
        args.save_dir,
        stem="all_folds_per_class_metrics_table",
        title="All folds test",
    )
    save_per_class_table_outputs(
        aggregate_transition_metrics,
        args.save_dir,
        stem="all_folds_transition_per_class_metrics_table",
        title="All folds transition",
    )

    aggregate = {
        "mode": f"stage1_subject_kfold{args.folds}",
        "data_root": args.data_root,
        "subset": args.subset,
        "max_subjects": max_subjects,
        "preprocess_config": asdict(pp_cfg),
        "stage1_config": asdict(st_cfg),
        "folds_requested": args.folds,
        "folds_run": [s["fold"] for s in summaries],
        "folds_trained_this_run": folds_trained_this_run,
        "missing_fold_summaries": missing_folds,
        "resume_from_save_dir": args.resume_from_save_dir,
        "resume_checkpoint_name": args.resume_checkpoint_name,
        "n_subjects": len(set(subject_ids)),
        "n_epochs_total": int(len(y)),
        "test_accuracy": mean_std([s["test_metrics"]["accuracy"] for s in summaries]),
        "test_macro_f1": mean_std([s["test_metrics"]["macro_f1"] for s in summaries]),
        "transition_accuracy": mean_std([s["transition_test_metrics"]["accuracy"] for s in summaries]),
        "transition_macro_f1": mean_std([s["transition_test_metrics"]["macro_f1"] for s in summaries]),
        "all_folds_test_metrics": aggregate_test_metrics,
        "all_folds_binary_wake_sleep_metrics": aggregate_binary_test_metrics,
        "all_folds_transition_metrics": aggregate_transition_metrics,
        "total_training_seconds": float(total_training_seconds),
        "fold_summaries": summaries,
    }
    summary_path = args.save_dir / f"kfold{args.folds}_summary.json"
    summary_path.write_text(json.dumps(json_safe(aggregate), indent=2), encoding="utf-8")
    print(f"\nK-fold summary: {summary_path}")
    print(
        "Aggregate | "
        f"test acc={aggregate['test_accuracy']['mean']:.4f}+/-{aggregate['test_accuracy']['std']:.4f} "
        f"mf1={aggregate['test_macro_f1']['mean']:.4f}+/-{aggregate['test_macro_f1']['std']:.4f} | "
        f"transition acc={aggregate['transition_accuracy']['mean']:.4f}+/-{aggregate['transition_accuracy']['std']:.4f} "
        f"mf1={aggregate['transition_macro_f1']['mean']:.4f}+/-{aggregate['transition_macro_f1']['std']:.4f}"
    )
    print(
        "Pooled weighted F1 | "
        f"test={aggregate_test_metrics['weighted_f1']:.4f} "
        f"transition={aggregate_transition_metrics['weighted_f1']:.4f}"
    )
    print(f"Total training time: {total_training_seconds / 60:.2f} min ({total_training_seconds:.1f} sec)")


if __name__ == "__main__":
    main()
