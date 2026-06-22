"""Shared hyperparameters for Sleep-EDF VAE + diffusion pipeline."""

from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    # 한 epoch 길이입니다. Sleep staging 표준에 맞춰 30초 단위로 자릅니다.
    segment_sec: float = 30.0
    # 모든 EDF 신호를 이 sampling rate로 맞춥니다.
    target_sfreq: float = 100.0
    # 전처리 band-pass filter 범위입니다.
    l_freq: float = 0.5
    h_freq: float = 35.0
    # 기본 EEG 채널입니다. cassette는 보통 Fpz-Cz, telemetry는 EEG Fpz-Cz 이름을 씁니다.
    eeg_channel: str = "Fpz-Cz"
    # 기본 채널명이 없을 때 순서대로 시도할 대체 EEG 채널명입니다.
    eeg_channel_fallbacks: tuple = (
        "Fpz-Cz",
        "EEG Fpz-Cz",
        "EEG FPZ-CZ",
        "Pz-Oz",
        "EEG Pz-Oz",
    )
    # True면 EEG 채널만 사용하고 EOG/EMG/Event marker 등은 제외합니다.
    eeg_only: bool = True
    # 사용할 subject 수 제한입니다. None이면 가능한 전체 subject를 사용합니다.
    max_subjects: int | None = 15
    # max_subjects가 지정되었을 때 subject를 seed 기반으로 랜덤 선택합니다.
    random_subjects: bool = True
    # Sleep stage ? / Movement time 좌우 몇 epoch까지 제거할지입니다. 1이면 양옆 30초도 제거합니다.
    exclude_unknown_context_epochs: int = 1
    use_6x5_windows: bool = False
    window_sec: float = 6.0
    windows_per_epoch: int = 5
    sliding_epoch_stride_sec: float | None = None
    transition_sliding_only: bool = False
    transition_sliding_context_sec: float = 60.0
    seed: int = 42


@dataclass
class Stage1Config:
    # VAE encoder가 만드는 latent vector 차원입니다.
    latent_dim: int = 128
    # 1D CNN 채널 폭입니다. 클수록 모델 용량과 GPU 메모리 사용량이 커집니다.
    base_channels: int = 64
    # Sleep stage 개수: W, N1, N2, N3, REM.
    num_stages: int = 5
    # 한 번에 학습하는 epoch 샘플 수입니다.
    batch_size: int = 64
    # AdamW learning rate입니다. VAE는 여러 loss가 섞여서 너무 크면 흔들릴 수 있습니다.
    lr: float = 3e-4
    # 전체 train set을 몇 번 반복해서 학습할지입니다.
    epochs: int = 20
    # 파형 reconstruction MSE 가중치: x와 x_hat을 sample 단위로 직접 맞춥니다.
    lambda_rec: float = 0.3
    # STFT magnitude loss 가중치: 시간-주파수 에너지 패턴을 맞춥니다.
    lambda_spec: float = 0.2
    # delta/theta/alpha/sigma/beta log band-power loss 가중치입니다.
    lambda_band: float = 0.1
    # 12-16 Hz sigma envelope loss 가중치입니다. spindle event label 없는 proxy loss입니다.
    lambda_sigma: float = 0.05
    # sigma loss를 stage별로 얼마나 강하게 줄지입니다. 순서: W, N1, N2, N3, REM.
    sigma_stage_weights: tuple[float, ...] = (0.2, 0.5, 1.0, 0.2, 0.5)
    # N1의 alpha 감소/theta 증가 패턴을 보존하는 loss 가중치입니다. 0이면 꺼집니다.
    # alpha-theta transition loss의 stage별 가중치입니다. 순서: W, N1, N2, N3, REM.
    # N2 K-complex-ish 저주파 transient envelope 보존 loss 가중치입니다. 0이면 꺼집니다.
    # K-complex proxy loss의 stage별 가중치입니다. 순서: W, N1, N2, N3, REM.
    # 현재 epoch의 sleep stage를 맞히는 cross-entropy loss 가중치입니다.
    lambda_stage: float = 3.0
    # If subwindow labels are available, mix pooled center-label CE with subwindow CE.
    # 0.0 = pooled CE only, 1.0 = subwindow CE only.
    subwindow_stage_loss_weight: float = 0.5
    # stage 간 거리를 반영한 penalty 가중치입니다. 멀리 틀릴수록 더 벌점을 줍니다.
    # VAE KL loss 가중치입니다. 크면 latent가 정규분포에 가까워지지만 recon이 뭉개질 수 있습니다.
    lambda_kl: float = 3e-4
    # 초반에는 KL을 약하게 넣고 이 epoch 수 동안 선형으로 키웁니다.
    kl_warmup_epochs: int = 10
    # Wake 구간이 길어서 전체 loss를 지배하지 않도록 W epoch의 loss를 낮춥니다.
    wake_loss_weight: float = 1.0
    # L_spec 계산용 STFT 설정입니다. 100 Hz 기준 256 samples는 약 2.56초입니다.
    stft_n_fft: int = 256
    stft_hop_length: int = 64
    stft_win_length: int = 256
    # logvar 폭주를 막기 위한 clamp 범위입니다.
    logvar_min: float = -8.0
    logvar_max: float = 8.0
    # gradient explosion 방지용 clipping입니다. None이면 끕니다.
    gradient_clip_norm: float | None = 1.0
    # class 빈도의 역수를 CE weight로 써서 희귀 stage를 보정합니다.
    use_class_weights: bool = True
    # CE class weight에 추가로 곱할 stage별 multiplier입니다. 순서: W, N1, N2, N3, REM.
    stage_class_weight_multiplier: tuple[float, ...] = (0.25, 0.75, 1.0, 1.0, 1.0)
    # subject-wise split에서 validation/test 비율입니다.
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    # train sampler: 'shuffle' | 'stage_balanced' | 'subject_balanced'
    train_sampling: str = "stage_balanced"
    # best checkpoint를 고를 validation metric입니다.
    checkpoint_metric: str = "val_macro_f1"
    # True면 30초 epoch를 짧은 subwindow로 나눠 shared encoder + attention pooling으로 latent를 만듭니다.
    use_subwindow_encoder: bool = False
    # subwindow 길이(초)입니다. Sleep onset transition 실험은 5~6초 권장입니다.
    subwindow_sec: float = 6.0
    use_transformer_encoder: bool = True
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    transformer_cls_mean_pool: bool = True


@dataclass
class InferenceConfig:
    """Inference 평가 기본값입니다."""

    # Acc_n에서 사용할 horizon 수입니다. 현재 구현은 non-autoregressive 평가입니다.
    n_horizons: int = 3
    batch_size: int = 64
    context_len: int = 5


@dataclass
class Stage2Config:
    # DDPM diffusion step 수입니다.
    diffusion_steps: int = 200
    # Stage 2 diffusion learning rate입니다.
    lr: float = 2e-4
    epochs: int = 20
    batch_size: int = 128
    context_len: int = 5
    pair_stride_sec: float | tuple[float, ...] = 30.0
    # diffusion noise prediction MSE 가중치입니다.
    lambda_diff: float = 1.0
    # diffusion timestep embedding 차원입니다.
    time_dim: int = 128
    # diffusion MLP hidden 차원입니다.
    hidden_dim: int = 512
    # Stage 2 pair sampling: 'transition' | 'stage_balanced' | 'shuffle'.
    sampling: str = "transition"
    # True면 Stage 2에서 stage transition 근처 pair를 더 자주 샘플링합니다.
    transition_weighted_sampling: bool = True
    # y_t != y_{t+1}인 transition pair에 곱할 sampling weight입니다.
    transition_pair_weight: float = 5.0
    # transition 앞뒤 몇 pair까지 추가로 강조할지입니다. 0이면 exact transition만 강조합니다.
    transition_context: int = 1
    # transition 주변이지만 exact transition은 아닌 pair에 곱할 sampling weight입니다.
    transition_context_weight: float = 2.0
    # True면 Stage 2 diffusion loss에 target stage(y_{t+1}) inverse-frequency weight를 곱합니다.
    use_target_stage_loss_weights: bool = True
    target_stage_weight_multiplier: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    # Stage 2가 복원한 z_{t+1}이 frozen stage classifier에서 y_{t+1}로 분류되도록 주는 CE loss 가중치입니다.
    lambda_next_stage: float = 0.1
    # Extra loss weight for exact transition pairs whose target stage is Wake.
    transition_wake_target_weight: float = 1.0
    # True면 diffusion model의 EMA copy를 유지하고 checkpoint/inference에 사용합니다.
    use_ema: bool = False
    # EMA decay입니다. 0.995면 teacher가 student를 천천히 따라갑니다.
    ema_decay: float = 0.995
    # Stage 2 VAE encoder student를 EMA teacher로 따라가게 하는 decay입니다.
    vae_ema_decay: float = 0.0
    # y_t != y_{t+1} transition pair에서만 student encoder를 EMA teacher에 붙잡는 loss 가중치입니다.
    train_encoder_near_transition: bool = False
    encoder_lr: float = 1e-5
    lambda_transition_ema: float = 0.0
