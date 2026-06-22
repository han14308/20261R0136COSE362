"""Training samplers: stage-balanced, subject-balanced, shuffle."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler

from .preprocess import STAGE_NAMES


class StageBalancedSampler(Sampler[int]):
    """
    W, N1, N2, N3, REM을 **동일 확률(각 1/5)** 로 고른 뒤,
    그 단계에 속한 train 구간 중 하나를 uniform 샘플.

    (데이터에 없는 단계는 제외하고, 남은 단계끼리 균등 확률.)
    """

    def __init__(
        self,
        labels: np.ndarray,
        num_stages: int = 5,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        labels = np.asarray(labels).astype(np.int64)
        by_stage: dict[int, list[int]] = defaultdict(list)
        for dataset_index, lab in enumerate(labels):
            by_stage[int(lab)].append(dataset_index)

        self.by_stage = dict(by_stage)
        self.stages_uniform = list(range(num_stages))
        self.stages_present = [c for c in self.stages_uniform if c in by_stage]
        self.num_samples = num_samples or len(labels)
        self.rng = np.random.default_rng(seed)
        self.class_counts = {STAGE_NAMES[i]: len(by_stage.get(i, [])) for i in range(num_stages)}

    def __iter__(self):
        for _ in range(self.num_samples):
            c = int(self.rng.choice(self.stages_uniform))
            if c not in self.by_stage:
                c = int(self.rng.choice(self.stages_present))
            yield int(self.rng.choice(self.by_stage[c]))

    def __len__(self) -> int:
        return self.num_samples


class SubjectBalancedSampler(Sampler[int]):
    """피험자 동일 확률 → 그 피험자 구간 중 uniform."""

    def __init__(
        self,
        subject_ids: list[str],
        train_idx: np.ndarray,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        subj = np.array(subject_ids)[train_idx]
        by_subject: dict[str, list[int]] = defaultdict(list)
        for dataset_index, sid in enumerate(subj):
            by_subject[sid].append(dataset_index)

        self.by_subject = dict(by_subject)
        self.subjects = sorted(by_subject.keys())
        self.num_samples = num_samples or len(subj)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.num_samples):
            sid = self.rng.choice(self.subjects)
            yield int(self.rng.choice(self.by_subject[sid]))

    def __len__(self) -> int:
        return self.num_samples


def make_train_sampler(
    mode: str,
    train_idx: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str] | None = None,
    num_stages: int = 5,
    seed: int = 42,
) -> StageBalancedSampler | SubjectBalancedSampler | None:
    """
    mode:
      - 'stage_balanced': W,N1,N2,N3,REM 각 1/5 확률 → 해당 단계 구간 랜덤 (기본)
      - 'subject_balanced': 피험자 균등 → 구간 랜덤
      - 'shuffle': DataLoader shuffle (구간 수 비례, N2/N3 편향)
    """
    y_train = np.asarray(y)[train_idx]
    n_train = len(y_train)

    if mode == "stage_balanced":
        return StageBalancedSampler(y_train, num_stages=num_stages, num_samples=n_train, seed=seed)
    if mode == "subject_balanced":
        if subject_ids is None:
            raise ValueError("subject_balanced requires subject_ids")
        return SubjectBalancedSampler(subject_ids, train_idx, num_samples=n_train, seed=seed)
    if mode == "shuffle":
        return None
    raise ValueError(f"Unknown train_sampling mode: {mode}")
