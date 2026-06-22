from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def display_image(path: Path) -> None:
    try:
        from IPython.display import Image, display

        display(Image(filename=str(path)))
    except Exception as exc:
        print(f"Image display skipped for {path}: {exc}")


def save_confusion_heatmap(
    cm: Any,
    labels: list[str],
    title: str,
    save_path: Path,
    normalize: bool = True,
) -> None:
    if cm is None:
        return
    import numpy as np

    import matplotlib.pyplot as plt

    arr = np.asarray(cm, dtype=np.int64)
    if normalize:
        row_sum = arr.sum(axis=1, keepdims=True)
        shown = np.divide(arr, row_sum, where=row_sum > 0, out=np.zeros_like(arr, dtype=float))
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        shown = arr.astype(float)
        fmt = "d"
        vmin, vmax = None, None

    fig_w = max(4.0, 0.85 * len(labels) + 2.0)
    fig_h = max(3.5, 0.75 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(shown, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    threshold = shown.max() / 2 if shown.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            text = format(shown[i, j], fmt) if normalize else str(int(arr[i, j]))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if shown[i, j] > threshold else "black",
                fontsize=10,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    actual_path = save_path
    try:
        fig.savefig(actual_path, dpi=160)
    except OSError as exc:
        fallback_root = Path("/content")
        if not fallback_root.exists():
            plt.close(fig)
            print(f"visualization save skipped for {save_path}: {exc}")
            return
        fallback_dir = fallback_root / "confusion_visualizations"
        actual_path = fallback_dir / save_path.name
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(actual_path, dpi=160)
        print(f"visualization save fallback: {actual_path} ({exc})")
    plt.close(fig)
    print(f"visualization: {actual_path}")
    display_image(actual_path)


def fmt_pct(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "NA"
    if x != x:
        return "NA"
    return f"{100.0 * x:.2f}%"


def binary_cm_from_5class(cm: Any) -> list[list[int]] | None:
    if cm is None:
        return None
    wake_wake = int(cm[0][0])
    wake_sleep = int(sum(cm[0][1:]))
    sleep_wake = int(sum(row[0] for row in cm[1:]))
    sleep_sleep = int(sum(sum(row[1:]) for row in cm[1:]))
    return [[wake_wake, wake_sleep], [sleep_wake, sleep_sleep]]


def sum_binary_cms(*cms: Any) -> list[list[int]] | None:
    valid = [cm for cm in cms if cm is not None]
    if not valid:
        return None
    return [
        [sum(int(cm[0][0]) for cm in valid), sum(int(cm[0][1]) for cm in valid)],
        [sum(int(cm[1][0]) for cm in valid), sum(int(cm[1][1]) for cm in valid)],
    ]


def binary_metrics(cm: list[list[int]] | None) -> dict[str, Any]:
    if cm is None:
        return {}
    total = sum(sum(row) for row in cm)
    correct = cm[0][0] + cm[1][1]
    wake_recall = cm[0][0] / sum(cm[0]) if sum(cm[0]) else None
    sleep_recall = cm[1][1] / sum(cm[1]) if sum(cm[1]) else None
    f1s: list[float] = []
    for i in range(2):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(2)) - tp
        fn = sum(cm[i]) - tp
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return {
        "accuracy": correct / total if total else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "wake_recall": wake_recall,
        "sleep_recall": sleep_recall,
        "n": total,
    }


def print_cm(title: str, labels: list[str], cm: Any, image_path: Path | None = None) -> None:
    print(f"\n=== {title} ===")
    if cm is None:
        print("not available")
        return
    print("rows=true cols=pred")
    print("labels:", labels)
    for label, row in zip(labels, cm):
        print(f"{label}: {row}")
    if image_path is not None:
        save_confusion_heatmap(cm, labels, title, image_path)


def print_binary_cm(title: str, cm: list[list[int]] | None, image_path: Path | None = None) -> dict[str, Any]:
    bm = binary_metrics(cm)
    print(f"\n=== {title} ===")
    if cm is None:
        print("not available")
        return bm
    print("rows=true cols=pred; labels=[Wake, Sleep]")
    print("Wake :", cm[0])
    print("Sleep:", cm[1])
    print(
        f"ACC={fmt_pct(bm.get('accuracy'))} MF1={fmt_pct(bm.get('macro_f1'))} "
        f"WakeRecall={fmt_pct(bm.get('wake_recall'))} "
        f"SleepRecall={fmt_pct(bm.get('sleep_recall'))} n={bm.get('n')}"
    )
    if image_path is not None:
        save_confusion_heatmap(cm, ["Wake", "Sleep"], title, image_path)
    return bm


def horizon_items(d: dict[str, Any]) -> list[tuple[int, Any]]:
    return [(int(k), v) for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))]


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def binary_class_acc(cm: list[list[int]] | None, class_idx: int) -> float | None:
    if cm is None:
        return None
    total = sum(cm[class_idx])
    return cm[class_idx][class_idx] / total if total else None


def print_stage2_acc_table(rows: list[dict[str, Any]]) -> None:
    print("\n=== Stage 2 binary ACC summary ===")
    print("| Step | W ACC | S ACC | W->S ACC | S->W ACC | All ACC | W/S Transition ACC |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| step {row['step']} | {fmt_pct(row.get('w_acc'))} | {fmt_pct(row.get('s_acc'))} | "
            f"{fmt_pct(row.get('w2s_acc'))} | {fmt_pct(row.get('s2w_acc'))} | "
            f"{fmt_pct(row.get('all_acc'))} | {fmt_pct(row.get('boundary_acc'))} |"
        )
    mean_row = {
        "w_acc": mean_metric(rows, "w_acc"),
        "s_acc": mean_metric(rows, "s_acc"),
        "w2s_acc": mean_metric(rows, "w2s_acc"),
        "s2w_acc": mean_metric(rows, "s2w_acc"),
        "all_acc": mean_metric(rows, "all_acc"),
        "boundary_acc": mean_metric(rows, "boundary_acc"),
    }
    print(
        f"| Mean | {fmt_pct(mean_row.get('w_acc'))} | {fmt_pct(mean_row.get('s_acc'))} | "
        f"{fmt_pct(mean_row.get('w2s_acc'))} | {fmt_pct(mean_row.get('s2w_acc'))} | "
        f"{fmt_pct(mean_row.get('all_acc'))} | {fmt_pct(mean_row.get('boundary_acc'))} |"
    )


def print_stage1_summary(summary_path: Path, elapsed_seconds: float | None = None) -> None:
    print("summary:", summary_path)
    if not summary_path.exists():
        print("Stage 1 summary not found; skipping summary display")
        return

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\n=== Fold mean +/- std ===")
    for key in ["test_accuracy", "test_macro_f1"]:
        v = summary.get(key, {})
        print(f"{key}: {fmt_pct(v.get('mean'))} +/- {fmt_pct(v.get('std'))}")

    print("\n=== Pooled all-fold metrics ===")
    m = summary.get("all_folds_test_metrics", {})
    print(
        f"ACC={fmt_pct(m.get('accuracy'))} MF1={fmt_pct(m.get('macro_f1'))} "
        f"WF1={fmt_pct(m.get('weighted_f1'))} n={m.get('n')}"
    )
    print("per-class recall:", m.get("per_class_acc"))

    output_dir = summary_path.parent
    all_cm = m.get("confusion_matrix")
    print_cm(
        "Stage 1 all-fold 5-class confusion matrix",
        ["W", "N1", "N2", "N3", "REM"],
        all_cm,
        output_dir / "all_folds_confusion_matrix_colab.png",
    )
    binary_cm = binary_cm_from_5class(all_cm)
    print_binary_cm(
        "Stage 1 all-fold binary Wake/Sleep confusion matrix",
        binary_cm,
        output_dir / "all_folds_binary_wake_sleep_confusion_matrix_colab.png",
    )

    elapsed = summary.get("total_training_seconds", elapsed_seconds)
    if elapsed is not None:
        print(f"\nTotal training time: {float(elapsed) / 60:.2f} min ({float(elapsed):.1f} sec)")


def print_stage2_summary(stage2_save_dir: str | Path, horizons: int) -> None:
    stage2_save_dir = Path(stage2_save_dir)
    stage2_summary_path = (
        stage2_save_dir
        / "direct_delta_flow_inference"
        / "excluded_stage1_test_split"
        / f"stage2_delta_flow_exclude_stage1_h{horizons}_summary.json"
    )
    if not stage2_summary_path.exists():
        candidates = sorted(
            stage2_save_dir.glob(f"**/stage2_delta_flow_exclude_stage1_h{horizons}_summary.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            stage2_summary_path = candidates[0]

    print("stage2 summary:", stage2_summary_path)
    if not stage2_summary_path.exists():
        print("Stage 2 summary not found yet")
        return

    with stage2_summary_path.open("r", encoding="utf-8") as f:
        s2 = json.load(f)
    metrics = s2.get("direct_delta_flow_metrics", {})

    inf_sec = metrics.get("inference_seconds")
    if inf_sec is not None:
        print("\n=== Stage 2 inference speed ===")
        print(
            f"time={float(inf_sec):.2f} sec "
            f"tasks={metrics.get('inference_tasks')} "
            f"predictions={metrics.get('inference_predictions')}"
        )
        print(
            f"tasks/sec={float(metrics.get('inference_tasks_per_sec', 0.0)):.2f} "
            f"predictions/sec={float(metrics.get('inference_predictions_per_sec', 0.0)):.2f}"
        )

    output_dir = stage2_summary_path.parent
    all_rows: list[dict[str, Any]] = []
    trans_rows: list[dict[str, Any]] = []
    acc_rows: list[dict[str, Any]] = []
    direction = metrics.get("per_horizon_direction_metrics", {})
    for h, cm5 in horizon_items(metrics.get("per_horizon_confusion", {})):
        all_binary_cm = binary_cm_from_5class(cm5)
        all_rows.append(
            print_binary_cm(
                f"Stage 2 h{h} ALL binary Wake/Sleep confusion",
                all_binary_cm,
                output_dir / f"stage2_h{h}_all_binary_wake_sleep_confusion.png",
            )
        )
        w2s_metrics = direction.get("binary_wake_to_sleep", {}).get(str(h), {})
        s2w_metrics = direction.get("binary_sleep_to_wake", {}).get(str(h), {})
        trans_binary_cm = sum_binary_cms(
            w2s_metrics.get("confusion_matrix"),
            s2w_metrics.get("confusion_matrix"),
        )
        trans_rows.append(
            print_binary_cm(
                f"Stage 2 h{h} W/S TRANSITION-ONLY binary confusion",
                trans_binary_cm,
                output_dir / f"stage2_h{h}_wake_sleep_boundary_binary_confusion.png",
            )
        )
        acc_rows.append(
            {
                "step": h,
                "w_acc": binary_class_acc(all_binary_cm, 0),
                "s_acc": binary_class_acc(all_binary_cm, 1),
                "w2s_acc": w2s_metrics.get("accuracy"),
                "s2w_acc": s2w_metrics.get("accuracy"),
                "all_acc": binary_metrics(all_binary_cm).get("accuracy"),
                "boundary_acc": binary_metrics(trans_binary_cm).get("accuracy"),
            }
        )

    print("\n=== Stage 2 3-horizon binary mean ===")
    print(f"ALL: ACC={fmt_pct(mean_metric(all_rows, 'accuracy'))} MF1={fmt_pct(mean_metric(all_rows, 'macro_f1'))}")
    print(
        f"W/S TRANSITION: ACC={fmt_pct(mean_metric(trans_rows, 'accuracy'))} "
        f"MF1={fmt_pct(mean_metric(trans_rows, 'macro_f1'))}"
    )
    print_stage2_acc_table(acc_rows)
