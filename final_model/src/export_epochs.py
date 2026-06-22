"""Export Sleep-EDF EEG 30-second epochs with sleep-stage labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import PreprocessConfig
from .paths import REPO_ROOT
from .preprocess import STAGE_NAMES, build_dataset


def _write_labels_csv(path: Path, y: np.ndarray, subject_ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_index", "subject_id", "label", "stage"])
        for idx, (label, subject_id) in enumerate(zip(y.tolist(), subject_ids)):
            writer.writerow([idx, subject_id, label, STAGE_NAMES[int(label)]])


def export_sleep_edf_epochs(
    data_root: str | Path = REPO_ROOT,
    out_dir: str | Path | None = None,
    *,
    subset: str | None = "cassette",
    max_subjects: int | None = None,
    target_sfreq: float = 100.0,
    eeg_channel: str = "Fpz-Cz",
) -> tuple[Path, Path]:
    """Create an NPZ file with X/y and a CSV file with epoch labels."""
    data_root = Path(data_root)
    out_dir = Path(out_dir) if out_dir is not None else data_root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessConfig(
        target_sfreq=target_sfreq,
        eeg_channel=eeg_channel,
        max_subjects=max_subjects,
    )
    X, y, subject_ids, epoch_mean, epoch_std = build_dataset(
        data_root,
        cfg=cfg,
        max_subjects=max_subjects,
        subset=subset,
    )

    subset_name = subset or "all"
    stem = f"sleep_edf_{subset_name}_30s_eeg"
    npz_path = out_dir / f"{stem}.npz"
    csv_path = out_dir / f"{stem}_labels.csv"

    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        subject_ids=np.asarray(subject_ids),
        epoch_mean=epoch_mean.astype(np.float32),
        epoch_std=epoch_std.astype(np.float32),
        stage_names=np.asarray(STAGE_NAMES),
        target_sfreq=np.asarray(target_sfreq, dtype=np.float32),
        segment_sec=np.asarray(30.0, dtype=np.float32),
        eeg_channel=np.asarray(eeg_channel),
    )
    _write_labels_csv(csv_path, y, subject_ids)
    return npz_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Sleep-EDF EEG as 30-second epochs with sleep-stage labels."
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--subset", choices=["cassette", "telemetry", "all"], default="cassette")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--target-sfreq", type=float, default=100.0)
    parser.add_argument("--eeg-channel", default="Fpz-Cz")
    args = parser.parse_args()

    subset = None if args.subset == "all" else args.subset
    npz_path, csv_path = export_sleep_edf_epochs(
        args.data_root,
        args.out_dir,
        subset=subset,
        max_subjects=args.max_subjects,
        target_sfreq=args.target_sfreq,
        eeg_channel=args.eeg_channel,
    )
    print(f"saved: {npz_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
