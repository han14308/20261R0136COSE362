"""Sleep-EDF: 30-second EEG epochs with hypnogram labels (W, N1, N2, N3, REM)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import mne
import numpy as np

from .config import PreprocessConfig

# AASM 5-class mapping (Sleep stage 4 merged into N3)
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")
EXCLUDE_STAGES = {"Movement time", "Sleep stage ?"}


def subject_id_from_psg(psg_path: Path) -> str:
    """e.g. SC4001E0-PSG.edf -> SC4001"""
    stem = psg_path.stem.replace("-PSG", "")
    return re.match(r"(SC\d+|ST\d+)", stem).group(1)


def find_hypnogram(psg_path: Path, data_root: Path) -> Path | None:
    sid = subject_id_from_psg(psg_path)
    parent = psg_path.parent
    candidates = sorted(parent.glob(f"{sid}*-Hypnogram.edf"))
    if not candidates:
        candidates = sorted(data_root.rglob(f"{sid}*-Hypnogram.edf"))
    return candidates[0] if candidates else None


def read_psg_edf(psg_path: Path, *, retries: int = 2) -> mne.io.BaseRaw:
    """Sleep-EDF PSG 로드 (cassette: Event marker, telemetry: fallback)."""
    common = dict(preload=True, infer_types=True, verbose="error")
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            try:
                return mne.io.read_raw_edf(psg_path, stim_channel="Event marker", **common)
            except (ValueError, RuntimeError):
                return mne.io.read_raw_edf(psg_path, **common)
        except OSError as exc:
            last_err = exc
            if getattr(exc, "errno", None) == 107 and attempt < retries:
                continue
            raise
    raise last_err  # type: ignore[misc]


def rebuild_hypnogram_annotations(
    hyp_path: Path,
    segment_sec: float = 30.0,
    exclude_context_epochs: int = 0,
) -> mne.Annotations:
    """
    Sleep-EDF hypnogram: EDF 헤더 시각과 PSG가 어긋날 수 있어,
    라벨 순서대로 0, 30, 60, … s 에 30초 에포크를 재구성합니다.
    (MNE 'outside data range' / peak=0 문제 방지)
    """
    raw_annot = mne.read_annotations(hyp_path)
    expanded: list[tuple[float, float, str]] = []
    excluded_onsets: list[float] = []
    for onset, desc, dur in zip(raw_annot.onset, raw_annot.description, raw_annot.duration):
        desc = str(desc)
        duration = float(dur)
        n_segments = 1 if duration <= 0 else int(np.floor(duration / segment_sec))
        for idx in range(n_segments):
            epoch_onset = float(onset) + idx * segment_sec
            expanded.append((epoch_onset, segment_sec, desc))
            if desc in EXCLUDE_STAGES or desc not in STAGE_MAP:
                excluded_onsets.append(epoch_onset)

    blocked_onsets: set[float] = set()
    for onset in excluded_onsets:
        for offset in range(-exclude_context_epochs, exclude_context_epochs + 1):
            blocked_onsets.add(round(onset + offset * segment_sec, 6))

    onsets: list[float] = []
    durs: list[float] = []
    descs: list[str] = []
    for onset, dur, desc in expanded:
        if desc in EXCLUDE_STAGES or desc not in STAGE_MAP:
            continue
        if round(float(onset), 6) in blocked_onsets:
            continue
        onsets.append(float(onset))
        durs.append(float(dur))
        descs.append(desc)
    if not onsets:
        return mne.Annotations([], [], [], orig_time=None)
    return mne.Annotations(
        onset=onsets,
        duration=durs,
        description=descs,
        orig_time=None,
    )


def _hypnogram_end_sec(annot: mne.Annotations) -> float:
    if len(annot) == 0:
        return 0.0
    return float(annot.onset[-1] + annot.duration[-1])


def _events_from_hyp(
    raw: mne.io.BaseRaw,
    event_id: dict[str, int],
    segment_sec: float,
) -> np.ndarray:
    """chunk_duration=30 우선, 실패 시 기본 annotation 이벤트."""
    for chunk in (segment_sec, None):
        try:
            events, _ = mne.events_from_annotations(
                raw,
                event_id=event_id,
                chunk_duration=chunk,
                verbose=False,
            )
            if len(events) > 0:
                return events
        except Exception:
            continue
    return np.empty((0, 3), dtype=int)


def _time_to_stop_sample(raw: mne.io.BaseRaw, tmax_sec: float) -> int:
    """MNE get_data(start, stop)는 stop이 샘플 인덱스(int)여야 함."""
    return min(int(tmax_sec * float(raw.info["sfreq"])), int(raw.n_times))


def _get_raw_window(
    raw: mne.io.BaseRaw,
    ch: str,
    tmax_sec: float,
) -> np.ndarray:
    stop = _time_to_stop_sample(raw, tmax_sec)
    return raw.get_data(picks=ch, start=0, stop=stop)


def _ensure_microvolts(X: np.ndarray) -> np.ndarray:
    """EDF가 V 단위일 때 µV로 변환."""
    X = np.asarray(X, dtype=np.float32)
    peak = float(np.max(np.abs(X))) if X.size else 0.0
    if peak < 0.05:
        X = (X * 1e6).astype(np.float32)
    return X


# Sleep-EDF cassette: EEG = Fpz-Cz / Pz-Oz (이름에 'EEG' 없음)
_EEG_MONTAGE_KEYS = frozenset({"FPZ-CZ", "PZ-OZ", "EEGFPZ-CZ", "EEGPZ-OZ"})
_NON_EEG_NAME_MARKERS = (
    "EOG",
    "EMG",
    "ECG",
    "HORIZONTAL",
    "VERTICAL",
    "ORO-NASAL",
    "ORONASAL",
    "SUBMENTAL",
    "RECTAL",
    "RESP",
    "SAO2",
    "TEMP",
    "EVENT",
    "MARKER",
)


def _channel_name_key(ch: str) -> str:
    return ch.upper().replace(" ", "")


def _is_non_eeg_name(ch: str) -> bool:
    u = ch.upper()
    return any(m in u for m in _NON_EEG_NAME_MARKERS)


def _is_eeg_montage_name(ch: str) -> bool:
    key = _channel_name_key(ch)
    if key in _EEG_MONTAGE_KEYS:
        return True
    if "EEG" in ch.upper() and not _is_non_eeg_name(ch):
        return True
    return False


def list_eeg_only_channels(raw: mne.io.BaseRaw, *, eeg_only: bool = True) -> list[str]:
    """
    EEG만: Fpz-Cz / Pz-Oz (cassette) 또는 'EEG …' 접두 채널.
    EOG(horizontal)·EMG(submental)·호흡·Event marker 제외.
    """
    if not eeg_only:
        return list(raw.ch_names)

    from_mne = [
        raw.ch_names[i]
        for i in mne.pick_types(raw.info, eeg=True, exclude="bads")
        if not _is_non_eeg_name(raw.ch_names[i])
    ]
    from_name = [ch for ch in raw.ch_names if _is_eeg_montage_name(ch) and not _is_non_eeg_name(ch)]

    out: list[str] = []
    seen: set[str] = set()
    for ch in from_mne + from_name:
        if ch not in seen:
            out.append(ch)
            seen.add(ch)
    return out


def _resolve_channel_name(names: list[str], target: str) -> str | None:
    """'EEG Fpz-Cz' 설정 → 파일의 'Fpz-Cz' 매칭."""
    key = _channel_name_key(target).replace("EEG", "")
    for ch in names:
        if ch == target or ch.upper() == target.upper():
            return ch
        if _channel_name_key(ch) == key or _channel_name_key(ch) == _channel_name_key(target):
            return ch
    return None


def pick_eeg_channel(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> str:
    """EEG 1채널 (기본 Fpz-Cz). EOG/EMG/호흡 채널 제외."""
    eeg_chs = list_eeg_only_channels(raw, eeg_only=cfg.eeg_only)
    if not eeg_chs:
        raise RuntimeError(
            f"No EEG montage channel. all_ch_names={raw.ch_names} file={raw.filenames}"
        )

    for cand in (cfg.eeg_channel, *cfg.eeg_channel_fallbacks):
        hit = _resolve_channel_name(eeg_chs, cand)
        if hit is not None:
            return hit

    fpz = [ch for ch in eeg_chs if "FPZ" in ch.upper() and "CZ" in ch.upper()]
    if fpz:
        return fpz[0]

    if len(eeg_chs) == 1:
        return eeg_chs[0]

    raise RuntimeError(
        f"Ambiguous EEG channels {eeg_chs}. Set PreprocessConfig.eeg_channel (e.g. 'Fpz-Cz')."
    )


def load_psg_epochs(
    psg_path: Path,
    hyp_path: Path,
    cfg: PreprocessConfig,
    *,
    min_peak_uV: float = 1.0,
    verbose_load: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_norm, y, epoch_onsets, per-epoch mean/std, and subwindow labels."""
    raw = read_psg_edf(psg_path)
    eeg_available = list_eeg_only_channels(raw, eeg_only=cfg.eeg_only)
    ch = pick_eeg_channel(raw, cfg)
    if verbose_load:
        dropped = [c for c in raw.ch_names if c not in eeg_available]
        print(f"    EEG only: use [{ch}] | eeg_candidates={eeg_available}")
        if dropped:
            print(f"    dropped (non-EEG): {dropped}")
    raw.pick([ch])
    raw.filter(cfg.l_freq, cfg.h_freq, verbose=False)
    raw.resample(cfg.target_sfreq, verbose=False)

    probe = _ensure_microvolts(_get_raw_window(raw, ch, min(120.0, float(raw.times[-1]))))
    probe_peak = float(np.max(np.abs(probe)))
    probe_std = float(np.std(probe))
    if probe_peak < 1.0:
        raise RuntimeError(
            f"EEG '{ch}' flat after filter (peak {probe_peak:.4g} µV, std {probe_std:.4g}). "
            f"ch_names={raw.ch_names}"
        )

    annot = rebuild_hypnogram_annotations(
        hyp_path,
        cfg.segment_sec,
        exclude_context_epochs=cfg.exclude_unknown_context_epochs,
    )
    if len(annot) == 0:
        raise RuntimeError(f"No scored stages in {hyp_path.name}")

    hyp_end = _hypnogram_end_sec(annot)
    if hyp_end > raw.times[-1] + 1.0:
        raise RuntimeError(
            f"Hypnogram ({hyp_end:.0f}s) longer than PSG ({raw.times[-1]:.0f}s) for {psg_path.name}"
        )
    raw.crop(tmin=0.0, tmax=min(hyp_end, raw.times[-1]))

    sfreq = float(raw.info["sfreq"])
    if cfg.use_6x5_windows:
        window_len = int(round(cfg.window_sec * sfreq))
        epoch_len = window_len * int(cfg.windows_per_epoch)
        expected_sec = cfg.window_sec * int(cfg.windows_per_epoch)
        if abs(expected_sec - cfg.segment_sec) > 1e-6:
            raise RuntimeError(
                f"use_6x5_windows expects segment_sec={expected_sec:g}, got {cfg.segment_sec:g}"
            )
    else:
        window_len = 0
        epoch_len = int(round(cfg.segment_sec * sfreq))
    data_uV = _ensure_microvolts(raw.get_data(picks=ch)).squeeze(0)
    xs: list[np.ndarray] = []
    labels: list[int] = []
    subwindow_labels: list[np.ndarray] = []
    epoch_onsets: list[float] = []
    sliding_stride_sec = getattr(cfg, "sliding_epoch_stride_sec", None)

    scored = [
        (
            float(onset),
            float(onset) + float(dur),
            STAGE_MAP[str(desc)],
        )
        for onset, dur, desc in zip(annot.onset, annot.duration, annot.description)
        if str(desc) in STAGE_MAP
    ]

    def stage_at_time(t_sec: float, start_i: int = 0) -> tuple[int | None, int]:
        i = start_i
        while i + 1 < len(scored) and t_sec >= scored[i][1]:
            i += 1
        lo, hi, lab = scored[i]
        if lo <= t_sec < hi:
            return int(lab), i
        return None, i

    def subwindow_labels_for_onset(onset_sec: float, start_i: int = 0) -> np.ndarray:
        labs: list[int] = []
        score_i_local = start_i
        for win_i in range(int(cfg.windows_per_epoch)):
            center = float(onset_sec) + (win_i + 0.5) * float(cfg.window_sec)
            lab, score_i_local = stage_at_time(center, score_i_local)
            labs.append(-1 if lab is None else int(lab))
        return np.asarray(labs, dtype=np.int64)

    if sliding_stride_sec is not None and float(sliding_stride_sec) > 0:
        if not scored:
            raise RuntimeError(f"No scored stages for sliding windows in {hyp_path.name}")
        stride = float(sliding_stride_sec)
        first_start = scored[0][0]
        last_start = min(scored[-1][1], raw.times[-1]) - cfg.segment_sec
        candidate_onsets: set[float] = set()
        base_onsets = [lo for lo, _, _ in scored if lo <= last_start + 1e-9]
        if getattr(cfg, "transition_sliding_only", False):
            candidate_onsets.update(round(float(o), 6) for o in base_onsets)
            context_sec = float(getattr(cfg, "transition_sliding_context_sec", 60.0))
            boundaries = [
                scored[i][1]
                for i in range(len(scored) - 1)
                if scored[i][2] != scored[i + 1][2]
            ]
            for boundary in boundaries:
                start_onset = max(first_start, boundary - context_sec - 0.5 * cfg.segment_sec)
                end_onset = min(last_start, boundary + context_sec - 0.5 * cfg.segment_sec)
                onset = round(start_onset / stride) * stride
                while onset < start_onset - 1e-9:
                    onset += stride
                while onset <= end_onset + 1e-9:
                    candidate_onsets.add(round(float(onset), 6))
                    onset += stride
        else:
            onset = first_start
            while onset <= last_start + 1e-9:
                candidate_onsets.add(round(float(onset), 6))
                onset += stride

        score_i = 0
        for onset in sorted(candidate_onsets):
            center = float(onset) + 0.5 * cfg.segment_sec
            while score_i + 1 < len(scored) and center >= scored[score_i][1]:
                score_i += 1
            lo, hi, lab = scored[score_i]
            if not (lo <= center < hi):
                continue
            start = int(round(float(onset) * sfreq))
            stop = start + epoch_len
            if not (0 <= start and stop <= data_uV.shape[0]):
                continue
            segment = data_uV[start:stop]
            if cfg.use_6x5_windows:
                if len(segment) != epoch_len:
                    continue
                segment = segment.reshape(int(cfg.windows_per_epoch), window_len)
                subwindow_labels.append(subwindow_labels_for_onset(float(onset), score_i))
            xs.append(segment)
            labels.append(int(lab))
            epoch_onsets.append(float(onset))
    else:
        for onset, desc in zip(annot.onset, annot.description):
            start = int(round(float(onset) * sfreq))
            stop = start + epoch_len
            if start < 0 or stop > data_uV.shape[0]:
                continue
            segment = data_uV[start:stop]
            if cfg.use_6x5_windows:
                if len(segment) != epoch_len:
                    continue
                segment = segment.reshape(int(cfg.windows_per_epoch), window_len)
                subwindow_labels.append(
                    np.full(int(cfg.windows_per_epoch), STAGE_MAP[str(desc)], dtype=np.int64)
                )
            xs.append(segment)
            labels.append(STAGE_MAP[str(desc)])
            epoch_onsets.append(float(onset))

    if not xs:
        raise RuntimeError(
            f"No complete {cfg.segment_sec:g}s EEG epochs from {hyp_path.name} "
            f"(check hypnogram labels / pairing with {psg_path.name})"
        )

    X_raw = np.stack(xs).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    y_subwindows = (
        np.stack(subwindow_labels).astype(np.int64)
        if cfg.use_6x5_windows and subwindow_labels
        else np.empty((len(y), 0), dtype=np.int64)
    )
    epoch_onsets_arr = np.asarray(epoch_onsets, dtype=np.float64)

    peak = float(np.max(np.abs(X_raw))) if len(X_raw) else 0.0
    flat_frac = float((X_raw.std(axis=1) < 1e-6).mean()) if len(X_raw) else 1.0
    if verbose_load:
        print(
            f"  [{psg_path.name}] ch={ch} epochs={len(y)} "
            f"peak={peak:.2f} µV flat_frac={flat_frac:.1%}"
        )
    if peak < min_peak_uV:
        raise RuntimeError(
            f"EEG amplitude too small (peak {peak:.4g} µV, ch={ch}). "
            f"Wrong channel or hypnogram misaligned: {psg_path.name}"
        )

    # VAE 입력: 구간(30s)마다 mean/std — 전역 통계 아님
    if cfg.use_6x5_windows:
        epoch_mean = X_raw.mean(axis=2, keepdims=True)
        epoch_std = np.maximum(X_raw.std(axis=2, keepdims=True), 1e-6)
        X_norm = ((X_raw - epoch_mean) / epoch_std).astype(np.float32)
        return X_norm, y, epoch_onsets_arr, epoch_mean.squeeze(2), epoch_std.squeeze(2), y_subwindows
    epoch_mean = X_raw.mean(axis=1, keepdims=True)
    epoch_std = np.maximum(X_raw.std(axis=1, keepdims=True), 1e-6)
    X_norm = ((X_raw - epoch_mean) / epoch_std).astype(np.float32)
    return X_norm, y, epoch_onsets_arr, epoch_mean.squeeze(1), epoch_std.squeeze(1), y_subwindows


def iter_recordings(
    data_root: Path,
    subset: str | None = None,
) -> Iterator[tuple[Path, Path]]:
    """Yield (psg_path, hypnogram_path) pairs."""
    patterns = ["sleep-cassette/*-PSG.edf", "sleep-telemetry/*-PSG.edf"]
    if subset == "cassette":
        patterns = ["sleep-cassette/*-PSG.edf"]
    elif subset == "telemetry":
        patterns = ["sleep-telemetry/*-PSG.edf"]

    for pat in patterns:
        for psg in sorted(data_root.glob(pat)):
            hyp = find_hypnogram(psg, data_root)
            if hyp is not None:
                yield psg, hyp


def count_recordings(data_root: str | Path, subset: str | None = None) -> int:
    return sum(1 for _ in iter_recordings(Path(data_root), subset=subset))


def validate_loaded_dataset(
    X: np.ndarray,
    y: np.ndarray,
    epoch_mean: np.ndarray,
    epoch_std: np.ndarray,
    *,
    min_peak_uV: float = 5.0,
    min_z_std_median: float = 0.5,
    max_flat_z_frac: float = 0.05,
) -> dict[str, float]:
    """
    build_dataset 결과가 학습에 쓸 만한지 검사.
    inspection 노트북과 동일 기준 — 실패 시 RuntimeError.
    """
    X = np.asarray(X)
    epoch_mean = np.asarray(epoch_mean)
    epoch_std = np.asarray(epoch_std)
    if X.ndim == 3 and X.shape[1] != 1:
        z_std = X.reshape(X.shape[0], -1).std(axis=1)
        raw_peak = 0.0
        for start in range(0, X.shape[0], 4096):
            stop = min(start + 4096, X.shape[0])
            raw_chunk = (
                X[start:stop].astype(np.float32, copy=False)
                * epoch_std[start:stop, :, None].astype(np.float32, copy=False)
                + epoch_mean[start:stop, :, None].astype(np.float32, copy=False)
            )
            raw_peak = max(raw_peak, float(np.max(np.abs(raw_chunk))))
    else:
        if X.ndim == 3:
            X = X[:, 0, :]
        z_std = X.std(axis=1)
        raw_peak = 0.0
        for start in range(0, X.shape[0], 4096):
            stop = min(start + 4096, X.shape[0])
            raw_chunk = (
                X[start:stop].astype(np.float32, copy=False)
                * epoch_std[start:stop, None].astype(np.float32, copy=False)
                + epoch_mean[start:stop, None].astype(np.float32, copy=False)
            )
            raw_peak = max(raw_peak, float(np.max(np.abs(raw_chunk))))
    flat_z = float((z_std < 1e-5).mean())

    stats = {
        "raw_peak_uV": raw_peak,
        "z_std_median": float(np.median(z_std)),
        "flat_z_frac": flat_z,
        "n_epochs": float(len(y)),
    }
    problems = []
    if raw_peak < min_peak_uV:
        problems.append(f"raw peak {raw_peak:.4g} µV < {min_peak_uV}")
    if stats["z_std_median"] < min_z_std_median:
        problems.append(f"z_std median {stats['z_std_median']:.4f} < {min_z_std_median}")
    if flat_z > max_flat_z_frac:
        problems.append(f"flat z epochs {flat_z:.1%} > {max_flat_z_frac:.0%}")
    if problems:
        raise RuntimeError(
            "Dataset failed validation (preprocess likely stale or wrong). "
            + "; ".join(problems)
            + ". Re-run load_sleep_edf_dataset() after reload(src.preprocess)."
        )
    return stats


def denorm_microvolts(
    x_norm: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
) -> np.ndarray:
    """Per-epoch z-score → µV."""
    return np.asarray(x_norm, dtype=np.float64) * np.asarray(sig) + np.asarray(mu)


def load_sleep_edf_dataset(
    data_root: str | Path,
    cfg: PreprocessConfig | None = None,
    max_subjects: int | None = None,
    subset: str | None = None,
    *,
    subject_filter: str | list[str] | tuple[str, ...] | None = None,
    validate: bool = True,
    return_epoch_onsets: bool = False,
    return_subwindow_labels: bool = False,
) -> tuple:
    """
    학습·inspection 공통 진입점: build_dataset + 검증.
    train_stage1은 전처리를 다시 하지 않으므로 반드시 이 함수로 X를 만드세요.
    """
    result = build_dataset(
        data_root,
        cfg=cfg,
        max_subjects=max_subjects,
        subset=subset,
        subject_filter=subject_filter,
        return_epoch_onsets=return_epoch_onsets,
        return_subwindow_labels=return_subwindow_labels,
    )
    if return_epoch_onsets and return_subwindow_labels:
        X, y, subs, epoch_onsets, mu, sig, y_subwindows = result
    elif return_epoch_onsets:
        X, y, subs, epoch_onsets, mu, sig = result
    elif return_subwindow_labels:
        X, y, subs, mu, sig, y_subwindows = result
    else:
        X, y, subs, mu, sig = result
    if validate:
        st = validate_loaded_dataset(X, y, mu, sig)
        print(
            f"Dataset OK: n={int(st['n_epochs'])} "
            f"raw_peak={st['raw_peak_uV']:.1f} µV z_std_med={st['z_std_median']:.3f}"
        )
    if return_epoch_onsets and return_subwindow_labels:
        return X, y, subs, epoch_onsets, mu, sig, y_subwindows
    if return_epoch_onsets:
        return X, y, subs, epoch_onsets, mu, sig
    if return_subwindow_labels:
        return X, y, subs, mu, sig, y_subwindows
    return X, y, subs, mu, sig


def build_dataset(
    data_root: str | Path,
    cfg: PreprocessConfig | None = None,
    max_subjects: int | None = None,
    subset: str | None = None,
    subject_filter: str | list[str] | tuple[str, ...] | None = None,
    return_epoch_onsets: bool = False,
    return_subwindow_labels: bool = False,
) -> tuple:
    """Load all subjects. Optionally returns epoch_onsets for gap-aware sequence pairs."""
    cfg = cfg or PreprocessConfig()
    data_root = Path(data_root)
    pairs = list(iter_recordings(data_root, subset=subset))
    if subject_filter is not None:
        wanted = {subject_filter} if isinstance(subject_filter, str) else set(subject_filter)
        pairs = [(psg, hyp) for psg, hyp in pairs if subject_id_from_psg(psg) in wanted]
    if not pairs:
        raise RuntimeError(
            f"No PSG+Hypnogram pairs under {data_root}. "
            f"Expected e.g. {data_root}/sleep-cassette/*-PSG.edf"
        )

    xs, ys, subs, onsets_all, mus, sigs, sub_ys = [], [], [], [], [], [], []
    limit = max_subjects if max_subjects is not None else cfg.max_subjects
    selected_pairs = pairs
    if limit is not None and cfg.random_subjects:
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(pairs))[:limit]
        selected_pairs = [pairs[int(i)] for i in order]
    elif limit is not None:
        selected_pairs = pairs[:limit]
    n_skip = 0

    sample_msg = "all" if limit is None else str(len(selected_pairs))
    random_msg = f", random seed={cfg.seed}" if limit is not None and cfg.random_subjects else ""
    print(f"Found {len(pairs)} PSG recordings (loading {sample_msg}{random_msg})")
    for psg, hyp in selected_pairs:
        try:
            X, y, onsets, m, s, y_subwindows = load_psg_epochs(psg, hyp, cfg, verbose_load=True)
        except OSError as exc:
            n_skip += 1
            hint = ""
            if getattr(exc, "errno", None) == 107:
                hint = " (Google Drive 끊김 → 런타임 재시작 후 drive.mount 다시)"
            print(f"Skip {psg.name}: {exc}{hint}")
            continue
        except Exception as exc:
            n_skip += 1
            print(f"Skip {psg.name}: {exc}")
            continue
        sid = subject_id_from_psg(psg)
        xs.append(X)
        ys.append(y)
        onsets_all.append(onsets)
        mus.append(m)
        sigs.append(s)
        sub_ys.append(y_subwindows)
        subs.extend([sid] * len(y))
        print(f"{sid}: {len(y)} epochs")

    if not xs:
        raise RuntimeError(
            f"No epochs loaded from {data_root} ({n_skip}/{len(selected_pairs)} "
            f"recordings skipped). Check 'Skip ...' messages above and EDF paths."
        )

    if return_epoch_onsets and return_subwindow_labels:
        return (
            np.concatenate(xs),
            np.concatenate(ys),
            subs,
            np.concatenate(onsets_all),
            np.concatenate(mus),
            np.concatenate(sigs),
            np.concatenate(sub_ys),
        )
    if return_epoch_onsets:
        return (
            np.concatenate(xs),
            np.concatenate(ys),
            subs,
            np.concatenate(onsets_all),
            np.concatenate(mus),
            np.concatenate(sigs),
        )
    if return_subwindow_labels:
        return (
            np.concatenate(xs),
            np.concatenate(ys),
            subs,
            np.concatenate(mus),
            np.concatenate(sigs),
            np.concatenate(sub_ys),
        )
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        subs,
        np.concatenate(mus),
        np.concatenate(sigs),
    )


def subject_wise_split(
    subject_ids: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/val/test indices without subject leakage."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(subject_ids))
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_s = set(unique[:n_test])
    val_s = set(unique[n_test : n_test + n_val])
    train_s = set(unique[n_test + n_val :])

    subj = np.array(subject_ids)
    train_idx = np.where(np.isin(subj, list(train_s)))[0]
    val_idx = np.where(np.isin(subj, list(val_s)))[0]
    test_idx = np.where(np.isin(subj, list(test_s)))[0]
    return train_idx, val_idx, test_idx
