"""Train/evaluate improved Stage 2 delta-flow on subjects excluded from Stage 1.

This is an experimental runner that leaves the original flow/diffusion code
untouched. It uses:

* larger context via --stage2-context-len
* stronger classification auxiliary loss via --stage2-lambda-next-stage
* delta-flow generation: z_{t+h} = z_t + generated_delta_h
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_attention_local import CODE_ROOT, REPO_ROOT, json_safe
from run_stage2_exclude_stage1_subjects import (
    resolve_stage1_run,
    save_per_horizon_direct_outputs,
    split_remaining_subjects,
)
from src.config import PreprocessConfig, Stage2Config
from src.inference import (
    _direction_metrics,
    _metrics_from_confusion,
    _subject_sorted_indices,
    _valid_rollout_starts,
    load_flow_multi,
)
from src.preprocess import STAGE_NAMES, iter_recordings, load_sleep_edf_dataset, subject_id_from_psg
from src.train_stage2 import load_frozen_vae
from src.train_stage2_flow_delta import sample_future_latents_delta_flow, train_stage2_flow_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved delta-flow Stage 2 after excluding Stage 1 subjects.")
    parser.add_argument("--stage1-fold-dir", type=Path, required=True)
    parser.add_argument("--vae-ckpt", "--stage1-ckpt", dest="vae_ckpt", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--subset", choices=("cassette", "telemetry", "all"), default="cassette")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--stage2-max-subjects",
        type=int,
        default=0,
        help="Limit Stage 2 to this many subjects after excluding Stage 1 subjects. 0 means all remaining subjects.",
    )
    parser.add_argument("--subwindow-sec", type=float, default=6.0)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--sliding-epoch-stride-sec", type=float, default=None)
    parser.add_argument("--transition-sliding-only", action="store_true")
    parser.add_argument("--transition-sliding-context-sec", type=float, default=60.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage2-batch-size", type=int, default=128)
    parser.add_argument("--stage2-context-len", type=int, default=10)
    parser.add_argument("--stage2-horizons", type=int, default=3)
    parser.add_argument("--stage2-flow-steps", type=int, default=20)
    parser.add_argument("--stage2-lambda-next-stage", type=float, default=0.5)
    parser.add_argument("--stage2-use-ema", action="store_true")
    parser.add_argument("--stage2-ema-decay", type=float, default=0.995)
    parser.add_argument("--stage2-train-encoder-near-transition", action="store_true")
    parser.add_argument("--stage2-encoder-lr", type=float, default=1e-5)
    parser.add_argument("--stage2-vae-ema-decay", type=float, default=0.995)
    parser.add_argument("--stage2-lambda-transition-ema", type=float, default=0.1)
    parser.add_argument("--stage2-sampling", choices=("transition", "stage_balanced", "shuffle"), default="transition")
    parser.add_argument("--stage2-no-target-stage-weights", action="store_true")
    parser.add_argument(
        "--stage2-target-stage-weight-multiplier",
        type=float,
        nargs=5,
        default=(1.0, 1.0, 1.0, 1.0, 1.0),
        metavar=("W", "N1", "N2", "N3", "REM"),
        help="Extra multiplier for Stage 2 target-stage weights. Order: W N1 N2 N3 REM.",
    )
    parser.add_argument("--stage2-transition-wake-target-weight", type=float, default=1.0)
    parser.add_argument("--stage2-save-dir", type=Path, default=CODE_ROOT / "checkpoints" / "stage2_delta_flow_exclude_stage1")
    parser.add_argument("--flow-ckpt", type=Path, default=None)
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--inference-save-dir", type=Path, default=None)
    return parser.parse_args()


def choose_stage2_subjects(
    data_root: Path,
    subset: str | None,
    excluded_subjects: set[str],
    max_subjects: int,
    seed: int,
) -> list[str] | None:
    if max_subjects <= 0:
        return None
    subjects = sorted(
        {
            subject_id_from_psg(psg)
            for psg, _ in iter_recordings(data_root, subset=subset)
            if subject_id_from_psg(psg) not in excluded_subjects
        }
    )
    if len(subjects) < 3:
        raise SystemExit(f"Need at least 3 Stage 2 subjects after exclusion, got {len(subjects)}")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(subjects, dtype=object)
    rng.shuffle(shuffled)
    limit = min(max_subjects, len(shuffled))
    if limit < 3:
        raise SystemExit("--stage2-max-subjects must leave at least 3 subjects for train/val/test")
    selected = sorted(str(s) for s in shuffled[:limit].tolist())
    print(
        f"Stage 2 subject subset: using {len(selected)}/{len(subjects)} remaining subjects "
        f"(seed={seed})",
        flush=True,
    )
    print("Stage 2 subjects:", ", ".join(selected), flush=True)
    return selected


@torch.no_grad()
def evaluate_acc_n_direct_delta_flow(
    vae_ckpt: Path | str,
    flow_ckpt: Path | str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    eval_indices: np.ndarray,
    n: int = 3,
    epoch_onsets: np.ndarray | None = None,
    device: str | None = None,
    flow_steps: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    from tqdm.auto import tqdm

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae = load_frozen_vae(Path(vae_ckpt), device_t)
    flow, cfg, latent_mean, latent_std, ckpt_horizons, ckpt_steps = load_flow_multi(
        Path(flow_ckpt),
        vae.encoder.fc_mu.out_features,
        device_t,
    )
    ckpt = torch.load(Path(flow_ckpt), map_location=device_t, weights_only=False)
    if ckpt.get("ema_vae_model") is not None:
        vae.load_state_dict(ckpt["ema_vae_model"])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
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

    if device_t.type == "cuda":
        torch.cuda.synchronize(device_t)
    inference_started = time.perf_counter()
    task_iter = tqdm(tasks, desc=f"Inference direct delta-flow Acc_{n}", disable=not show_progress)
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
        z_future_norm = sample_future_latents_delta_flow(flow, z_ctx_norm, steps=steps)
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
    if device_t.type == "cuda":
        torch.cuda.synchronize(device_t)
    inference_seconds = time.perf_counter() - inference_started

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
            "delta_flow": True,
            "inference_seconds": float(inference_seconds),
            "inference_tasks": int(len(tasks)),
            "inference_predictions": int(sum(per_h[i]["total"] for i in range(1, n + 1))),
            "inference_tasks_per_sec": float(len(tasks) / inference_seconds) if inference_seconds > 0 else 0.0,
            "inference_predictions_per_sec": (
                float(sum(per_h[i]["total"] for i in range(1, n + 1)) / inference_seconds)
                if inference_seconds > 0
                else 0.0
            ),
            "per_horizon_acc": horizon_acc,
            "per_horizon_n": {i: per_h[i]["total"] for i in range(1, n + 1)},
            "per_horizon_confusion": cms,
            "per_horizon_transition_metrics": {i: _metrics_from_confusion(trans_cms[i]) for i in range(1, n + 1)},
            "per_horizon_stable_metrics": {i: _metrics_from_confusion(stable_cms[i]) for i in range(1, n + 1)},
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


def main() -> None:
    args = parse_args()
    excluded_subjects, vae_ckpt, resolved_stage1_fold_dir = resolve_stage1_run(args.stage1_fold_dir, args.vae_ckpt)
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
        target_stage_weight_multiplier=tuple(args.stage2_target_stage_weight_multiplier),
        transition_wake_target_weight=args.stage2_transition_wake_target_weight,
        use_ema=args.stage2_use_ema,
        ema_decay=args.stage2_ema_decay,
        train_encoder_near_transition=args.stage2_train_encoder_near_transition,
        encoder_lr=args.stage2_encoder_lr,
        vae_ema_decay=args.stage2_vae_ema_decay if args.stage2_train_encoder_near_transition else 0.0,
        lambda_transition_ema=args.stage2_lambda_transition_ema if args.stage2_train_encoder_near_transition else 0.0,
    )

    print(f"CODE_ROOT:       {CODE_ROOT}")
    print(f"DATA_ROOT:       {args.data_root}")
    print(f"STAGE1_SOURCE:   {args.stage1_fold_dir}")
    print(f"STAGE1_FOLD_DIR: {resolved_stage1_fold_dir}")
    print(f"VAE_CKPT:        {vae_ckpt}")
    print(f"Excluded Stage1 subjects: {len(excluded_subjects)}")
    stage2_subject_filter = choose_stage2_subjects(
        args.data_root,
        subset,
        excluded_subjects,
        args.stage2_max_subjects,
        args.seed,
    )

    X, y, subject_ids, epoch_onsets, epoch_mean, epoch_std = load_sleep_edf_dataset(
        args.data_root,
        cfg=pp_cfg,
        max_subjects=None,
        subset=subset,
        subject_filter=stage2_subject_filter,
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
        "Delta-flow split after exclusion: "
        f"remaining_subjects={len(remaining_subjects)} "
        f"train_n={len(train_idx)} val_n={len(val_idx)} test_n={len(test_idx)}",
        flush=True,
    )

    args.stage2_save_dir.mkdir(parents=True, exist_ok=True)
    flow_ckpt = args.flow_ckpt or (args.stage2_save_dir / "flow_delta_multi.pt")
    if args.infer_only:
        stage2_history: dict[str, Any] | list[Any] = []
        if not flow_ckpt.exists():
            raise SystemExit(f"Delta-flow checkpoint not found: {flow_ckpt}")
    else:
        _, stage2_history = train_stage2_flow_delta(
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
        flow_ckpt = args.stage2_save_dir / "flow_delta_multi.pt"

    inference_save_dir = args.inference_save_dir or (
        args.stage2_save_dir / "direct_delta_flow_inference" / "excluded_stage1_test_split"
    )
    inference_save_dir.mkdir(parents=True, exist_ok=True)
    flow_metrics = evaluate_acc_n_direct_delta_flow(
        vae_ckpt,
        flow_ckpt,
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
        f"[Excluded Stage1 subjects DIRECT-DELTA-FLOW Acc_{args.stage2_horizons}] "
        f"acc_n={flow_metrics['acc_n']:.4f} "
        f"acc={flow_metrics['accuracy']:.4f} "
        f"balanced_acc={flow_metrics['balanced_acc']:.4f} "
        f"macro_f1={flow_metrics['macro_f1']:.4f} "
        f"flow_steps={flow_metrics['flow_steps']}",
        flush=True,
    )
    for h, acc in flow_metrics["per_horizon_acc"].items():
        print(f"  h{h}: acc={acc:.4f} n={flow_metrics['per_horizon_n'][h]}", flush=True)
    save_per_horizon_direct_outputs(
        flow_metrics,
        inference_save_dir,
        stem_prefix="inference_direct_delta_flow",
        title_prefix="Excluded Stage1 subjects direct delta-flow",
    )

    summary = {
        "mode": "stage2_delta_flow_exclude_stage1_subjects",
        "flow_method": "rectified_flow_delta",
        "stage1_fold_dir": args.stage1_fold_dir,
        "resolved_stage1_fold_dir": resolved_stage1_fold_dir,
        "vae_ckpt": vae_ckpt,
        "flow_ckpt": flow_ckpt,
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
        "direct_delta_flow_metrics": flow_metrics,
    }
    summary_path = inference_save_dir / f"stage2_delta_flow_exclude_stage1_h{args.stage2_horizons}_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Stage2 delta-flow exclude-subject summary: {summary_path}")


if __name__ == "__main__":
    main()
