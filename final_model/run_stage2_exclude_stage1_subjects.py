"""Train Stage 2 after excluding subjects used by a Stage 1 fold run.

This loads the full dataset, removes every subject listed in the Stage 1
fold_summary train/val/test lists, then creates a fresh subject-wise
train/val/test split from the remaining subjects for Stage 2.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from run_attention_local import (
    BINARY_STAGE_NAMES,
    CODE_ROOT,
    REPO_ROOT,
    STAGE_NAMES,
    binary_wake_sleep_metrics,
    json_safe,
    save_confusion_outputs,
)
from src.config import PreprocessConfig, Stage2Config
from src.inference import (
    SleepInferencePipeline,
    evaluate_acc_n_direct_flow,
    evaluate_acc_n_direct_multi,
    evaluate_step1_accuracy,
)
from src.preprocess import load_sleep_edf_dataset
from src.train_stage2 import train_stage2
from src.train_stage2_flow import train_stage2_flow
from src.train_stage2_multi import train_stage2_multi


def multiclass_metrics(true: Any, pred: Any, mask: Any | None = None) -> dict[str, Any]:
    true_arr = np.asarray(true, dtype=np.int64)
    pred_arr = np.asarray(pred, dtype=np.int64)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        true_arr = true_arr[mask_arr]
        pred_arr = pred_arr[mask_arr]

    n_classes = len(STAGE_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true_arr, pred_arr):
        cm[int(t), int(p)] += 1

    total = int(cm.sum())
    correct = int(np.trace(cm))
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
        "accuracy": float(correct / total) if total else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "n": total,
    }


def binary_metrics_from_confusion(cm: Any) -> dict[str, Any]:
    cm_arr = np.asarray(cm, dtype=np.int64)
    total = int(cm_arr.sum())
    correct = int(np.trace(cm_arr))
    f1s = []
    per_class_acc = {}
    for i, name in enumerate(BINARY_STAGE_NAMES):
        tp = int(cm_arr[i, i])
        fp = int(cm_arr[:, i].sum() - tp)
        fn = int(cm_arr[i, :].sum() - tp)
        support = int(cm_arr[i, :].sum())
        per_class_acc[name] = float(tp / support) if support > 0 else float("nan")
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return {
        "accuracy": float(correct / total) if total else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm_arr,
        "n": total,
        "n_pairs": total,
    }


def binary_all_from_multiclass_confusion(cm: Any) -> dict[str, Any]:
    cm_arr = np.asarray(cm, dtype=np.int64)
    binary_cm = np.asarray(
        [
            [cm_arr[0, 0], cm_arr[0, 1:].sum()],
            [cm_arr[1:, 0].sum(), cm_arr[1:, 1:].sum()],
        ],
        dtype=np.int64,
    )
    return binary_metrics_from_confusion(binary_cm)


def get_horizon_item(items: dict[Any, Any], h: int) -> Any:
    return items.get(h, items.get(str(h)))


def save_per_horizon_direct_outputs(metrics: dict[str, Any], save_dir: Path, stem_prefix: str, title_prefix: str) -> None:
    per_horizon_confusion = metrics.get("per_horizon_confusion", {})
    transition_metrics = metrics.get("per_horizon_transition_metrics", {})
    direction_metrics = metrics.get("per_horizon_direction_metrics", {})
    binary_w2s = direction_metrics.get("binary_wake_to_sleep", {})
    binary_s2w = direction_metrics.get("binary_sleep_to_wake", {})

    for raw_h in sorted(per_horizon_confusion, key=lambda x: int(x)):
        h = int(raw_h)
        cm = np.asarray(per_horizon_confusion[raw_h], dtype=np.int64)
        multiclass_all = {
            **multiclass_metrics([], []),
            "confusion_matrix": cm,
        }
        multiclass_all.update(multiclass_metrics_from_confusion(cm))
        binary_all = binary_all_from_multiclass_confusion(cm)

        trans = get_horizon_item(transition_metrics, h)
        if trans is not None:
            trans_cm = np.asarray(trans["confusion_matrix"], dtype=np.int64)
            multiclass_transition = {
                **trans,
                "confusion_matrix": trans_cm,
                "n": int(trans.get("n_pairs", trans_cm.sum())),
            }
        else:
            trans_cm = np.zeros_like(cm)
            multiclass_transition = {
                **multiclass_metrics_from_confusion(trans_cm),
                "n": 0,
            }

        w2s = get_horizon_item(binary_w2s, h)
        s2w = get_horizon_item(binary_s2w, h)
        w2s_cm = np.asarray(w2s["confusion_matrix"], dtype=np.int64) if w2s is not None else np.zeros((2, 2), dtype=np.int64)
        s2w_cm = np.asarray(s2w["confusion_matrix"], dtype=np.int64) if s2w is not None else np.zeros((2, 2), dtype=np.int64)
        binary_transition = binary_metrics_from_confusion(w2s_cm + s2w_cm)

        print(
            f"[{title_prefix} h{h} 5-class all] "
            f"acc={multiclass_all['accuracy']:.4f} bal={multiclass_all['balanced_acc']:.4f} "
            f"mf1={multiclass_all['macro_f1']:.4f} n={multiclass_all['n_pairs']}",
            flush=True,
        )
        print(
            f"[{title_prefix} h{h} binary Wake/Sleep all] "
            f"acc={binary_all['accuracy']:.4f} mf1={binary_all['macro_f1']:.4f} n={binary_all['n']}",
            flush=True,
        )
        print(
            f"[{title_prefix} h{h} binary Wake/Sleep transition only] "
            f"acc={binary_transition['accuracy']:.4f} mf1={binary_transition['macro_f1']:.4f} "
            f"n={binary_transition['n']}",
            flush=True,
        )
        save_confusion_outputs(
            multiclass_all,
            save_dir,
            stem=f"{stem_prefix}_h{h}_5class_all_confusion_matrix",
            title=f"{title_prefix} h{h} 5-class all confusion matrix",
        )
        save_confusion_outputs(
            multiclass_transition,
            save_dir,
            stem=f"{stem_prefix}_h{h}_5class_transition_confusion_matrix",
            title=f"{title_prefix} h{h} 5-class transition confusion matrix",
        )
        save_confusion_outputs(
            binary_all,
            save_dir,
            stem=f"{stem_prefix}_h{h}_binary_wake_sleep_all_confusion_matrix",
            title=f"{title_prefix} h{h} binary Wake/Sleep all confusion matrix",
            class_names=BINARY_STAGE_NAMES,
        )
        save_confusion_outputs(
            binary_transition,
            save_dir,
            stem=f"{stem_prefix}_h{h}_binary_wake_sleep_transition_only_confusion_matrix",
            title=f"{title_prefix} h{h} binary Wake/Sleep transition-only confusion matrix",
            class_names=BINARY_STAGE_NAMES,
        )


def multiclass_metrics_from_confusion(cm: Any) -> dict[str, Any]:
    cm_arr = np.asarray(cm, dtype=np.int64)
    total = int(cm_arr.sum())
    correct = int(np.trace(cm_arr))
    per_class_acc = {}
    recalls = []
    f1s = []
    for i, name in enumerate(STAGE_NAMES):
        tp = int(cm_arr[i, i])
        fp = int(cm_arr[:, i].sum() - tp)
        fn = int(cm_arr[i, :].sum() - tp)
        support = int(cm_arr[i, :].sum())
        per_class_acc[name] = float(tp / support) if support > 0 else float("nan")
        if support > 0:
            recalls.append(tp / support)
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return {
        "accuracy": float(correct / total) if total else float("nan"),
        "balanced_acc": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm_arr,
        "n": total,
        "n_pairs": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 2 on full data after excluding Stage 1 fold subjects."
    )
    parser.add_argument("--stage1-fold-dir", type=Path, required=True)
    parser.add_argument("--vae-ckpt", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--subset", choices=("cassette", "telemetry", "all"), default="cassette")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--subwindow-sec", type=float, default=6.0)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--sliding-epoch-stride-sec", type=float, default=None)
    parser.add_argument("--transition-sliding-only", action="store_true")
    parser.add_argument("--transition-sliding-context-sec", type=float, default=60.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage2-batch-size", type=int, default=128)
    parser.add_argument("--stage2-context-len", type=int, default=5)
    parser.add_argument("--stage2-multi", action="store_true", help="Train/evaluate direct multi-horizon diffusion.")
    parser.add_argument("--stage2-flow", action="store_true", help="Train/evaluate direct multi-horizon rectified flow.")
    parser.add_argument("--stage2-horizons", type=int, default=3)
    parser.add_argument("--stage2-flow-steps", type=int, default=20, help="Euler steps for flow inference.")
    parser.add_argument("--stage2-lambda-next-stage", type=float, default=0.1)
    parser.add_argument("--stage2-sampling", choices=("transition", "stage_balanced", "shuffle"), default="transition")
    parser.add_argument("--stage2-no-target-stage-weights", action="store_true")
    parser.add_argument("--stage2-transition-wake-target-weight", type=float, default=1.0)
    parser.add_argument("--stage2-vae-ema-decay", type=float, default=0.0)
    parser.add_argument("--stage2-save-dir", type=Path, default=CODE_ROOT / "checkpoints" / "stage2_exclude_stage1")
    parser.add_argument("--diffusion-ckpt", type=Path, default=None)
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--inference-save-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_stage1_run(stage1_path: Path, vae_ckpt: Path | None) -> tuple[set[str], Path, Path]:
    """Accept either a fold directory or a parent k-fold run directory."""
    fold_summary_path = stage1_path / "fold_summary.json"
    if fold_summary_path.exists():
        fold_summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
        excluded_subjects = set(
            str(s)
            for key in ("train_subjects", "val_subjects", "test_subjects")
            for s in fold_summary.get(key, [])
        )
        resolved_vae = vae_ckpt or (stage1_path / "best_vae.pt")
        return excluded_subjects, resolved_vae, stage1_path

    run_summary_path = stage1_path / "kfold20_summary.json"
    if not run_summary_path.exists():
        run_summary_path = stage1_path / "kfold10_summary.json"
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        excluded_subjects: set[str] = set()
        for fold_summary in run_summary.get("fold_summaries", []):
            for key in ("train_subjects", "val_subjects", "test_subjects"):
                excluded_subjects.update(str(s) for s in fold_summary.get(key, []))

        fold_dirs = sorted(p for p in stage1_path.glob("fold_*") if p.is_dir())
        resolved_fold_dir = next((p for p in fold_dirs if (p / "fold_summary.json").exists()), stage1_path)
        resolved_vae = vae_ckpt or (
            stage1_path / "best_vae.pt"
            if (stage1_path / "best_vae.pt").exists()
            else next(
                (p / "best_vae.pt" for p in fold_dirs if (p / "best_vae.pt").exists()),
                stage1_path / "best_vae.pt",
            )
        )
        return excluded_subjects, resolved_vae, resolved_fold_dir

    raise SystemExit(
        "Could not find fold_summary.json, kfold20_summary.json, or kfold10_summary.json under "
        f"{stage1_path}"
    )


def split_remaining_subjects(
    subject_ids: list[str],
    excluded_subjects: set[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    subj = np.asarray(subject_ids, dtype=object)
    remaining_subjects = sorted(str(s) for s in set(subject_ids) if str(s) not in excluded_subjects)
    if len(remaining_subjects) < 3:
        raise ValueError(
            f"Need at least 3 remaining subjects after exclusion, got {len(remaining_subjects)}"
        )
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(remaining_subjects, dtype=object)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_s = set(shuffled[:n_test].tolist())
    val_s = set(shuffled[n_test : n_test + n_val].tolist())
    train_s = set(shuffled[n_test + n_val :].tolist())
    train_idx = np.where(np.isin(subj, list(train_s)))[0]
    val_idx = np.where(np.isin(subj, list(val_s)))[0]
    test_idx = np.where(np.isin(subj, list(test_s)))[0]
    return train_idx, val_idx, test_idx, remaining_subjects


def main() -> None:
    args = parse_args()
    if args.stage2_flow and args.stage2_multi:
        raise SystemExit("Use only one of --stage2-flow or --stage2-multi.")
    excluded_subjects, vae_ckpt, resolved_stage1_fold_dir = resolve_stage1_run(
        args.stage1_fold_dir,
        args.vae_ckpt,
    )
    if not excluded_subjects:
        raise SystemExit(f"No excluded subjects found under {args.stage1_fold_dir}")
    if not vae_ckpt.exists():
        raise SystemExit(f"Stage 1 checkpoint not found: {vae_ckpt}")
    subset = None if args.subset == "all" else args.subset
    pp_cfg = PreprocessConfig(
        max_subjects=None,
        random_subjects=True,
        seed=args.seed,
        use_6x5_windows=not args.no_attention,
        window_sec=args.subwindow_sec,
        windows_per_epoch=5,
        sliding_epoch_stride_sec=args.sliding_epoch_stride_sec,
        transition_sliding_only=args.transition_sliding_only,
        transition_sliding_context_sec=args.transition_sliding_context_sec,
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

    print(f"CODE_ROOT:       {CODE_ROOT}")
    print(f"DATA_ROOT:       {args.data_root}")
    print(f"STAGE1_SOURCE:   {args.stage1_fold_dir}")
    print(f"STAGE1_FOLD_DIR: {resolved_stage1_fold_dir}")
    print(f"VAE_CKPT:        {vae_ckpt}")
    print(f"Excluded Stage1 subjects: {len(excluded_subjects)}")

    X, y, subject_ids, epoch_onsets, epoch_mean, epoch_std = load_sleep_edf_dataset(
        args.data_root,
        cfg=pp_cfg,
        max_subjects=None,
        subset=subset,
        return_epoch_onsets=True,
    )
    train_idx, val_idx, test_idx, remaining_subjects = split_remaining_subjects(
        subject_ids,
        excluded_subjects,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        "Stage2 split after exclusion: "
        f"remaining_subjects={len(remaining_subjects)} "
        f"train_n={len(train_idx)} val_n={len(val_idx)} test_n={len(test_idx)}",
        flush=True,
    )

    args.stage2_save_dir.mkdir(parents=True, exist_ok=True)
    if args.stage2_flow:
        ckpt_name = "flow_multi.pt"
    elif args.stage2_multi:
        ckpt_name = "diffusion_multi.pt"
    else:
        ckpt_name = "diffusion.pt"
    diffusion_ckpt = args.diffusion_ckpt or (args.stage2_save_dir / ckpt_name)
    if args.infer_only:
        stage2_history = []
        if not diffusion_ckpt.exists():
            raise SystemExit(f"Stage 2 checkpoint not found: {diffusion_ckpt}")
    elif args.stage2_flow:
        flow, stage2_history = train_stage2_flow(
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
            horizons=args.stage2_horizons,
            inference_steps=args.stage2_flow_steps,
        )
        diffusion_ckpt = args.stage2_save_dir / "flow_multi.pt"
    elif args.stage2_multi:
        diffusion, stage2_history = train_stage2_multi(
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
            horizons=args.stage2_horizons,
        )
        diffusion_ckpt = args.stage2_save_dir / "diffusion_multi.pt"
    else:
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
        diffusion_ckpt = args.stage2_save_dir / "diffusion.pt"
    print(f"Stage 2 checkpoint: {diffusion_ckpt}")

    inference_save_dir = args.inference_save_dir or (
        args.stage2_save_dir
        / (
            "direct_flow_inference"
            if args.stage2_flow
            else "direct_multi_inference"
            if args.stage2_multi
            else "step1_inference"
        )
        / "excluded_stage1_test_split"
    )
    inference_save_dir.mkdir(parents=True, exist_ok=True)
    if args.stage2_flow:
        flow_metrics = evaluate_acc_n_direct_flow(
            vae_ckpt,
            diffusion_ckpt,
            X,
            y,
            subject_ids,
            test_idx,
            n=args.stage2_horizons,
            epoch_onsets=epoch_onsets,
            device=args.device,
            flow_steps=args.stage2_flow_steps,
            show_progress=not args.no_progress,
        )
        print(
            f"[Excluded Stage1 subjects DIRECT-FLOW Acc_{args.stage2_horizons}] "
            f"acc_n={flow_metrics['acc_n']:.4f} "
            f"acc={flow_metrics['accuracy']:.4f} "
            f"balanced_acc={flow_metrics['balanced_acc']:.4f} "
            f"macro_f1={flow_metrics['macro_f1']:.4f} "
            f"flow_steps={flow_metrics['flow_steps']}",
            flush=True,
        )
        for h, acc in flow_metrics["per_horizon_acc"].items():
            print(f"  h{h}: acc={acc:.4f} n={flow_metrics['per_horizon_n'][h]}", flush=True)
        print("[DIRECT-FLOW per-horizon transition]")
        for h, metrics in flow_metrics["per_horizon_transition_metrics"].items():
            print(
                f"  h{h}: transition acc={metrics['accuracy']:.4f} "
                f"mf1={metrics['macro_f1']:.4f} n={metrics['n_pairs']}",
                flush=True,
            )
        print("[DIRECT-FLOW per-horizon W<->Sleep direction, binary]")
        flow_dirs = flow_metrics["per_horizon_direction_metrics"]
        for h in flow_metrics["per_horizon_acc"]:
            w2s = flow_dirs["binary_wake_to_sleep"][h]
            s2w = flow_dirs["binary_sleep_to_wake"][h]
            print(
                f"  h{h}: W->Sleep acc={w2s['accuracy']:.4f} mf1={w2s['macro_f1']:.4f} n={w2s['n_pairs']} | "
                f"Sleep->W acc={s2w['accuracy']:.4f} mf1={s2w['macro_f1']:.4f} n={s2w['n_pairs']}",
                flush=True,
            )
        save_confusion_outputs(
            flow_metrics,
            inference_save_dir,
            stem=f"inference_direct_flow_h{args.stage2_horizons}_combined_confusion_matrix",
            title=f"Excluded Stage1 subjects direct flow h1..h{args.stage2_horizons}",
        )
        save_per_horizon_direct_outputs(
            flow_metrics,
            inference_save_dir,
            stem_prefix="inference_direct_flow",
            title_prefix="Excluded Stage1 subjects direct flow",
        )
        summary = {
            "mode": "stage2_flow_exclude_stage1_subjects",
            "flow_method": "rectified_flow_linear",
            "stage1_fold_dir": args.stage1_fold_dir,
            "vae_ckpt": vae_ckpt,
            "flow_ckpt": diffusion_ckpt,
            "data_root": args.data_root,
            "subset": args.subset,
            "excluded_subjects": sorted(excluded_subjects),
            "remaining_subjects": remaining_subjects,
            "preprocess_config": asdict(pp_cfg),
            "stage2_config": asdict(s2_cfg),
            "stage2_horizons": int(args.stage2_horizons),
            "stage2_flow_steps": int(args.stage2_flow_steps),
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "stage2_history": stage2_history,
            "direct_flow_metrics": flow_metrics,
        }
        summary_path = inference_save_dir / f"stage2_flow_exclude_stage1_h{args.stage2_horizons}_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print(f"Stage2 flow exclude-subject summary: {summary_path}")
        return

    if args.stage2_multi:
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
            f"[Excluded Stage1 subjects DIRECT-MULTI Acc_{args.stage2_horizons}] "
            f"acc_n={direct_metrics['acc_n']:.4f} "
            f"acc={direct_metrics['accuracy']:.4f} "
            f"balanced_acc={direct_metrics['balanced_acc']:.4f} "
            f"macro_f1={direct_metrics['macro_f1']:.4f}",
            flush=True,
        )
        for h, acc in direct_metrics["per_horizon_acc"].items():
            print(f"  h{h}: acc={acc:.4f} n={direct_metrics['per_horizon_n'][h]}", flush=True)
        print("[DIRECT-MULTI per-horizon transition]")
        for h, metrics in direct_metrics["per_horizon_transition_metrics"].items():
            print(
                f"  h{h}: transition acc={metrics['accuracy']:.4f} "
                f"mf1={metrics['macro_f1']:.4f} n={metrics['n_pairs']}",
                flush=True,
            )
        print("[DIRECT-MULTI per-horizon W<->Sleep direction, binary]")
        direct_dirs = direct_metrics["per_horizon_direction_metrics"]
        for h in direct_metrics["per_horizon_acc"]:
            w2s = direct_dirs["binary_wake_to_sleep"][h]
            s2w = direct_dirs["binary_sleep_to_wake"][h]
            print(
                f"  h{h}: W->Sleep acc={w2s['accuracy']:.4f} mf1={w2s['macro_f1']:.4f} n={w2s['n_pairs']} | "
                f"Sleep->W acc={s2w['accuracy']:.4f} mf1={s2w['macro_f1']:.4f} n={s2w['n_pairs']}",
                flush=True,
            )
        save_confusion_outputs(
            direct_metrics,
            inference_save_dir,
            stem=f"inference_direct_multi_h{args.stage2_horizons}_combined_confusion_matrix",
            title=f"Excluded Stage1 subjects direct multi h1..h{args.stage2_horizons}",
        )
        save_per_horizon_direct_outputs(
            direct_metrics,
            inference_save_dir,
            stem_prefix="inference_direct_multi",
            title_prefix="Excluded Stage1 subjects direct multi diffusion",
        )
        summary = {
            "mode": "stage2_multi_exclude_stage1_subjects",
            "stage1_fold_dir": args.stage1_fold_dir,
            "vae_ckpt": vae_ckpt,
            "diffusion_ckpt": diffusion_ckpt,
            "data_root": args.data_root,
            "subset": args.subset,
            "excluded_subjects": sorted(excluded_subjects),
            "remaining_subjects": remaining_subjects,
            "preprocess_config": asdict(pp_cfg),
            "stage2_config": asdict(s2_cfg),
            "stage2_horizons": int(args.stage2_horizons),
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "stage2_history": stage2_history,
            "direct_multi_metrics": direct_metrics,
        }
        summary_path = inference_save_dir / f"stage2_multi_exclude_stage1_h{args.stage2_horizons}_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print(f"Stage2 multi exclude-subject summary: {summary_path}")
        return

    pipe = SleepInferencePipeline.from_checkpoints(vae_ckpt, diffusion_ckpt, device=args.device)
    step1_metrics = evaluate_step1_accuracy(
        pipe,
        X,
        y,
        subject_ids,
        test_idx,
        epoch_onsets=epoch_onsets,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    print(
        "[Excluded Stage1 subjects step1 test] "
        f"acc={step1_metrics['accuracy']:.4f} "
        f"balanced_acc={step1_metrics['balanced_acc']:.4f} "
        f"macro_f1={step1_metrics['macro_f1']:.4f} "
        f"n={step1_metrics['n']}",
        flush=True,
    )
    save_confusion_outputs(
        step1_metrics,
        inference_save_dir,
        stem="inference_step1_confusion_matrix",
        title="Excluded Stage1 subjects step1 confusion matrix",
    )
    transition_metrics = {
        **step1_metrics,
        "accuracy": step1_metrics["transition_acc"],
        "confusion_matrix": step1_metrics["transition_confusion_matrix"],
        "n": step1_metrics["n_transition"],
    }
    multiclass_all_metrics = multiclass_metrics(step1_metrics["true"], step1_metrics["pred"])
    multiclass_transition_metrics = multiclass_metrics(
        step1_metrics["true"],
        step1_metrics["pred"],
        mask=step1_metrics["transition"],
    )
    multiclass_near_transition_metrics = multiclass_metrics(
        step1_metrics["true"],
        step1_metrics["pred"],
        mask=step1_metrics["near_transition"],
    )
    multiclass_wake_to_sleep_metrics = multiclass_metrics(
        step1_metrics["true"],
        step1_metrics["pred"],
        mask=step1_metrics["wake_to_sleep"],
    )
    multiclass_sleep_to_wake_metrics = multiclass_metrics(
        step1_metrics["true"],
        step1_metrics["pred"],
        mask=step1_metrics["sleep_to_wake"],
    )
    print(
        "[Step1 5-class metrics] "
        f"all acc={multiclass_all_metrics['accuracy']:.4f} mf1={multiclass_all_metrics['macro_f1']:.4f} n={multiclass_all_metrics['n']} | "
        f"transition acc={multiclass_transition_metrics['accuracy']:.4f} mf1={multiclass_transition_metrics['macro_f1']:.4f} n={multiclass_transition_metrics['n']} | "
        f"near_transition acc={multiclass_near_transition_metrics['accuracy']:.4f} mf1={multiclass_near_transition_metrics['macro_f1']:.4f} n={multiclass_near_transition_metrics['n']} | "
        f"wake_to_sleep acc={multiclass_wake_to_sleep_metrics['accuracy']:.4f} mf1={multiclass_wake_to_sleep_metrics['macro_f1']:.4f} n={multiclass_wake_to_sleep_metrics['n']} | "
        f"sleep_to_wake acc={multiclass_sleep_to_wake_metrics['accuracy']:.4f} mf1={multiclass_sleep_to_wake_metrics['macro_f1']:.4f} n={multiclass_sleep_to_wake_metrics['n']}",
        flush=True,
    )
    save_confusion_outputs(
        transition_metrics,
        inference_save_dir,
        stem="inference_step1_transition_confusion_matrix",
        title="Excluded Stage1 subjects step1 transition confusion matrix",
    )
    binary_metrics = binary_wake_sleep_metrics(step1_metrics["true"], step1_metrics["pred"])
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
    binary_wake_to_sleep_metrics = binary_wake_sleep_metrics(
        step1_metrics["true"],
        step1_metrics["pred"],
        mask=step1_metrics["wake_to_sleep"],
    )
    binary_sleep_to_wake_metrics = binary_wake_sleep_metrics(
        step1_metrics["true"],
        step1_metrics["pred"],
        mask=step1_metrics["sleep_to_wake"],
    )
    print(
        "[Step1 binary Wake/Sleep metrics] "
        f"all acc={binary_metrics['accuracy']:.4f} mf1={binary_metrics['macro_f1']:.4f} n={binary_metrics['n']} | "
        f"transition acc={binary_transition_metrics['accuracy']:.4f} mf1={binary_transition_metrics['macro_f1']:.4f} n={binary_transition_metrics['n']} | "
        f"near_transition acc={binary_near_transition_metrics['accuracy']:.4f} mf1={binary_near_transition_metrics['macro_f1']:.4f} n={binary_near_transition_metrics['n']} | "
        f"wake_to_sleep acc={binary_wake_to_sleep_metrics['accuracy']:.4f} mf1={binary_wake_to_sleep_metrics['macro_f1']:.4f} n={binary_wake_to_sleep_metrics['n']} | "
        f"sleep_to_wake acc={binary_sleep_to_wake_metrics['accuracy']:.4f} mf1={binary_sleep_to_wake_metrics['macro_f1']:.4f} n={binary_sleep_to_wake_metrics['n']}",
        flush=True,
    )
    save_confusion_outputs(
        binary_metrics,
        inference_save_dir,
        stem="inference_step1_binary_wake_sleep_confusion_matrix",
        title="Excluded Stage1 subjects step1 binary Wake/Sleep confusion matrix",
        class_names=BINARY_STAGE_NAMES,
    )
    save_confusion_outputs(
        binary_transition_metrics,
        inference_save_dir,
        stem="inference_step1_binary_wake_sleep_transition_confusion_matrix",
        title="Excluded Stage1 subjects step1 binary Wake/Sleep transition confusion matrix",
        class_names=BINARY_STAGE_NAMES,
    )
    save_confusion_outputs(
        binary_near_transition_metrics,
        inference_save_dir,
        stem="inference_step1_binary_wake_sleep_near_transition_confusion_matrix",
        title="Excluded Stage1 subjects step1 binary Wake/Sleep near-transition confusion matrix",
        class_names=BINARY_STAGE_NAMES,
    )
    save_confusion_outputs(
        binary_boundary_metrics,
        inference_save_dir,
        stem="inference_step1_binary_wake_sleep_boundary_only_confusion_matrix",
        title="Excluded Stage1 subjects step1 binary Wake/Sleep boundary-only confusion matrix",
        class_names=BINARY_STAGE_NAMES,
    )
    save_confusion_outputs(
        binary_wake_to_sleep_metrics,
        inference_save_dir,
        stem="inference_step1_binary_wake_to_sleep_only_confusion_matrix",
        title="Excluded Stage1 subjects step1 binary Wake-to-Sleep only confusion matrix",
        class_names=BINARY_STAGE_NAMES,
    )
    save_confusion_outputs(
        binary_sleep_to_wake_metrics,
        inference_save_dir,
        stem="inference_step1_binary_sleep_to_wake_only_confusion_matrix",
        title="Excluded Stage1 subjects step1 binary Sleep-to-Wake only confusion matrix",
        class_names=BINARY_STAGE_NAMES,
    )

    summary = {
        "mode": "stage2_exclude_stage1_subjects",
        "stage1_fold_dir": args.stage1_fold_dir,
        "vae_ckpt": vae_ckpt,
        "diffusion_ckpt": diffusion_ckpt,
        "data_root": args.data_root,
        "subset": args.subset,
        "excluded_subjects": sorted(excluded_subjects),
        "remaining_subjects": remaining_subjects,
        "preprocess_config": asdict(pp_cfg),
        "stage2_config": asdict(s2_cfg),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "stage2_history": stage2_history,
        "step1_metrics": step1_metrics,
        "step1_multiclass_all_metrics": multiclass_all_metrics,
        "step1_multiclass_transition_metrics": multiclass_transition_metrics,
        "step1_multiclass_near_transition_metrics": multiclass_near_transition_metrics,
        "step1_multiclass_wake_to_sleep_metrics": multiclass_wake_to_sleep_metrics,
        "step1_multiclass_sleep_to_wake_metrics": multiclass_sleep_to_wake_metrics,
        "binary_wake_sleep_metrics": binary_metrics,
        "binary_wake_sleep_transition_metrics": binary_transition_metrics,
        "binary_wake_sleep_near_transition_metrics": binary_near_transition_metrics,
        "binary_wake_sleep_boundary_only_metrics": binary_boundary_metrics,
        "binary_wake_to_sleep_only_metrics": binary_wake_to_sleep_metrics,
        "binary_sleep_to_wake_only_metrics": binary_sleep_to_wake_metrics,
    }
    summary_path = inference_save_dir / "stage2_exclude_stage1_step1_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Stage2 exclude-subject summary: {summary_path}")


if __name__ == "__main__":
    main()
