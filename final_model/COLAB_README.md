# Final Model Colab Package

This package is minimized for:

- Stage 1: Transformer model with VAE architecture, trained with stage classification loss only
- Stage 2: VAE latent delta-flow multi-horizon prediction

## Colab setup

Upload/extract this folder as `final_model`, then run:

```python
%cd final_model
!pip install -q -r requirements.txt
```

Sleep-EDF data folders (`sleep-cassette`, `sleep-telemetry`, `RECORDS`, etc.) must be one level above this folder, or pass `--data-root`.

## Rebuild Stage 1 aggregate summary only

Use this after all `fold_00` to `fold_19` folders already contain `fold_summary.json`.

```bash
python run_kfold10_stage1.py \
  --folds 20 \
  --max-subjects 20 \
  --seed 42 \
  --epochs 10 \
  --batch-size 64 \
  --train-sampling shuffle \
  --kl-warmup-epochs 5 \
  --wake-loss-weight 1.0 \
  --lambda-rec 0.0 \
  --lambda-spec 0.0 \
  --lambda-band 0.0 \
  --lambda-sigma 0.0 \
  --lambda-stage 3.0 \
  --subwindow-stage-loss-weight 0.5 \
  --lambda-kl 0.0 \
  --stage-class-weight-multiplier 0.25 0.55 1.0 1.0 1.0 \
  --save-dir checkpoints/final_model_stage1_20fold_transformer_stageonly_w025_n1x055 \
  --fold-start 20 \
  --fold-end 20 \
  --no-progress
```

Output:

```text
checkpoints/final_model_stage1_20fold_transformer_stageonly_w025_n1x055/kfold20_summary.json
```

## Train Stage 2 delta flow

Run after Stage 1 checkpoints exist under `checkpoints/final_model_stage1_20fold_transformer_stageonly_w025_n1x055/fold_XX`.
The example below uses `fold_13`; change both paths if you choose another fold.

```bash
python run_stage2_delta_flow_exclude_stage1_subjects.py \
  --stage1-fold-dir checkpoints/final_model_stage1_20fold_transformer_stageonly_w025_n1x055/fold_13 \
  --stage1-ckpt checkpoints/final_model_stage1_20fold_transformer_stageonly_w025_n1x055/fold_13/best_vae.pt \
  --seed 42 \
  --stage2-epochs 20 \
  --stage2-batch-size 128 \
  --stage2-context-len 10 \
  --stage2-horizons 3 \
  --stage2-flow-steps 20 \
  --stage2-lambda-next-stage 0.5 \
  --stage2-sampling transition \
  --stage2-save-dir checkpoints/final_model_stage2_delta_flow_h3_ctx10_transition \
  --no-progress
```

Output summary:

```text
checkpoints/final_model_stage2_delta_flow_h3_ctx10_transition/direct_delta_flow_inference/excluded_stage1_test_split/stage2_delta_flow_exclude_stage1_h3_summary.json
```

## Skip Stage 1 and train Stage 2 from an uploaded checkpoint

Use this when Stage 1 is already trained and uploaded to Drive. `--stage1-fold-dir`
must point to the fold directory containing `fold_summary.json`; `--stage1-ckpt`
points to the matching `best_vae.pt`.

```bash
python run_stage2_delta_flow_exclude_stage1_subjects.py \
  --stage1-fold-dir checkpoints/final_model_stage1_20fold_transformer_stageonly_w025_n1x055/fold_13 \
  --stage1-ckpt checkpoints/final_model_stage1_20fold_transformer_stageonly_w025_n1x055/fold_13/best_vae.pt \
  --seed 42 \
  --stage2-epochs 10 \
  --stage2-batch-size 128 \
  --stage2-context-len 10 \
  --stage2-horizons 3 \
  --stage2-flow-steps 20 \
  --stage2-lambda-next-stage 0.5 \
  --stage2-sampling transition \
  --stage2-save-dir checkpoints/final_model_stage2_delta_flow_fold13_h3_ctx10_transition_ep10 \
  --no-progress
```

## Run Stage 2 from an existing Stage 1 checkpoint in Colab

In `final_model_colab.ipynb`, use:

```python
RUN_STAGE1 = False
RUN_STAGE2 = True
FOLD_START = 0
FOLD_END = 20
STAGE2_FOLD = 13
SAVE_NAME = 'final_model_stage1_20fold_transformer_vae_w025_n1x055'
SAVE_DIR = f'checkpoints/{SAVE_NAME}'
STAGE2_EXISTING_FOLD_DIR = f"{SAVE_DIR}/fold_{STAGE2_FOLD:02d}"
STAGE2_EXISTING_CKPT = f"{STAGE2_EXISTING_FOLD_DIR}/best_vae.pt"
STAGE2_BATCH_SIZE = 16
STAGE2_CONTEXT_LEN = 5
STAGE2_FLOW_STEPS = 10
STAGE2_MAX_SUBJECTS = 30
STAGE2_SAMPLING = 'transition'
STAGE2_TARGET_STAGE_WEIGHT_MULTIPLIER = [1.0, 0.55, 1.0, 1.0, 1.0]  # W N1 N2 N3 REM
STAGE2_SAVE_NAME = 'final_model_stage2_delta_flow_transformer_vae_fold13_h3_ctx5_subj30_n1x055_next05'
STAGE2_SAVE_DIR = f'checkpoints/{STAGE2_SAVE_NAME}'
STAGE2_INFER_ONLY = False
STAGE2_FLOW_CKPT = f'{STAGE2_SAVE_DIR}/flow_delta_multi.pt'
```

This uses:

```python
checkpoints/final_model_stage1_20fold_transformer_vae_w025_n1x055/fold_13/fold_summary.json
checkpoints/final_model_stage1_20fold_transformer_vae_w025_n1x055/fold_13/best_vae.pt
```

The Stage 2 command must include `--stage2-max-subjects 30`; otherwise Colab may try to load too much data and get killed by RAM limits. The N1 target-stage loss weight is lowered by passing `--stage2-target-stage-weight-multiplier 1.0 0.55 1.0 1.0 1.0`.

To run Stage 2 inference only from existing checkpoints, set:

```python
RUN_STAGE1 = False
RUN_STAGE2 = True
STAGE2_INFER_ONLY = True
STAGE2_EXISTING_FOLD_DIR = 'checkpoints/final_model_stage1_20fold_transformer_vae_w025_n1x055/fold_13'
STAGE2_EXISTING_CKPT = f'{STAGE2_EXISTING_FOLD_DIR}/best_vae.pt'
STAGE2_SAVE_DIR = 'checkpoints/final_model_stage2_delta_flow_transformer_vae_fold13_h3_ctx5_subj30_n1x055_next05'
STAGE2_FLOW_CKPT = f'{STAGE2_SAVE_DIR}/flow_delta_multi.pt'
```

This skips Stage 2 training and writes a fresh inference summary under `STAGE2_SAVE_DIR/direct_delta_flow_inference/excluded_stage1_test_split`.

Stage 2 writes its final summary under `STAGE2_SAVE_DIR`, for example:

```text
checkpoints/final_model_stage2_delta_flow_transformer_vae_fold13_h3_ctx5_subj30_n1x055_next05/direct_delta_flow_inference/excluded_stage1_test_split/stage2_delta_flow_exclude_stage1_h3_summary.json
```

The Colab summary cells print the confusion matrices and display heatmap PNGs for Stage 1 and Stage 2 inference. Stage 2 W/S transition-only heatmaps use only Wake->Sleep and Sleep->Wake boundary cases, excluding Sleep->Sleep segments. Stage 2 summaries also report inference time, tasks/sec, and predictions/sec after a new Stage 2 run. If Drive image saving fails, the heatmaps fall back to `/content/confusion_visualizations`.
