# COSE362 Machine Learning Project  
## AI x Medical: EEG-Based Prediction Model for Sleep Transitions

**Final code:** `final_model/final_model_colab.ipynb`

Final submission notebook: `final_model/final_model_colab.ipynb`

The `final_model/` folder contains the final implementation used for training, inference, and evaluation. Earlier files and intermediate results are kept for reference, but the final submitted pipeline is provided in the Colab notebook above.

## Project Overview

We built a two-stage framework for future sleep-stage prediction using single-channel EEG signals from the Sleep-EDF dataset.

Stage 1 learns sleep-stage-discriminative EEG latent representations using a Patch-Transformer VAE encoder. Each 30-second EEG epoch is divided into five 6-second patches, encoded into latent representations, and trained with reconstruction, spectral, KL, and sleep stage classification losses.

Stage 2 predicts future sleep-stage transitions in the learned latent space. A conditional latent delta flow matching model uses the previous five EEG epochs as context and jointly predicts future latent transitions across three horizons. The predicted future latents are then classified into Wake, N1, N2, N3, and REM using the frozen Stage 1 classifier.

## How to Run

1. Upload the entire `20261R0136COSE362` folder to Google Drive.

2. Rename the uploaded folder:
   `20261R0136COSE362` -> `sleep-edf`

3. Open the Colab notebook:
   `sleep-edf/final_model/final_model_colab.ipynb`
