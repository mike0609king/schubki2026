# Schubki 2026

This document gives an explanation of the repository structure as well as the
machine learning pipeline.

## Structure
- `src`. Contains source code to quickly access implementations, which are shared across notebooks.
- `notebooks`. Contains notebooks. Most notebooks in this directory is for testing the machine learning pipeline. The final machine learning pipeline can be found in
    - TODO: Generating the matrices for subjects `...`
    - TODO: Test
- `data`. This directory is used for preprocessed Data. The final predictions
for each subject are available in this directory. Some directories such as the
given data and intermediate results have not been pushed because of their size.

## ML pipeline

The machine learning pipeline can be divided into
- Preprocessing
- Feature extraction and
- Classification

those will be explained in the following sections.

(( TODO give an overview of the machine learning model and the ensemble learning 
approach. ))

## Preprocessing
The given data was already preprocessed by applying a band pass filter and
removal of blink components. We applied the following preprocessing steps
in addition
- bandpass filter 0.5 - 75.0 Hz
- notch filter with 50 Hz to remove powerline interference

((TODO remove averaging over channels.))

Because of our feature extraction process the data has not been heavily
processed.

## Feature extraction

The machine learning model uses both extracted EEG features as well as features
for the video.

### EEG features

**EEG foundation models.** The following three EEG foundation models have been used
- LaBraM ((TODO cite))
- modified LaBraM. LaBraM is modified to use 3 second long EEG signals instead of
adding zero padding.
- LUNA ((TODO cite))
- CBraMod

**Engineered features.** ((TODO wait for tests))

### Video features

The following video foundation models have been used
- OpenCLIP ((TODO cite))
- Dino ((TODO cite))
- VideoMAE ((TODO cite))

## Classification

For each of 6 subjects we learn a linear mapping from video features to the EEG response (17 channels x 300 time samples after decimation). At test time we predict the EEG for each of the 102 candidate videos and compare it with the recorded EEG of each trial. For every trial, the candidates are ranked by Pearson correlation, giving a ranked candidate list per subject.

### Model

- One RidgeCV per subject (5-fold CV, alphas = logspace(-3, 6, 10)), mapping PCA features to the full EEG response.

- Selected alphas were 1e4 or 1e5 depending on the subject.

### Results

Top-1 / Top-5 retrieval accuracy.

**Train (in-sample, not a generalization estimate)**

| Subject | Top-1 | Top-5 |
|---|---|---|
| 01 | 0.8977 | 0.9298 |
| 02 | 0.7253 | 0.8282 |
| 03 | 0.9063 | 0.9365 |
| 04 | 0.7476 | 0.8448 |
| 05 | 0.8929 | 0.9254 |
| 06 | 0.8059 | 0.8659 |

**Test (102 candidates; chance: top-1 = 0.0098, top-5 = 0.0490)**

| Subject | Top-1 | Top-5 | Median rank |
|---|---|---|---|
| 01 | 0.0588 | 0.2451 | 15.0 |
| 02 | 0.0392 | 0.2451 | 19.5 |
| 03 | 0.0490 | 0.1765 | 20.5 |
| 04 | 0.0490 | 0.1471 | 22.5 |
| 05 | n/a (no ground truth available) | | |
| 06 | n/a (no ground truth available) | | |

### Feature ablation (Subject 1)

| Features | Train top-1 | Test top-1 | Test top-5 | Test median rank |
|---|---|---|---|---|
| AlexNet `features.3` | 0.781 | 0.039 | 0.118 | 25.5 |
| DINO `blocks.6` | 0.930 | 0.049 | 0.304 | 13.0 |
| CLIP `resblocks.6` | 0.942 | 0.039 | 0.167 | 17.0 |
| CLIP `resblocks.7` | 0.900 | 0.049 | 0.216 | 14.0 |
| **Ensemble (submitted)** | 0.926 | 0.059 | 0.265 | 14.0 |

The ensemble gave the best test top-1 and a strong top-5, so it was chosen for the submission. 

## Other approaches

We also attempted other approaches:

Besides the ridge regression on raw (decimated) EEG, we tried three learned models that operate on pretrained EEG foundation-model embeddings instead of the raw signal. They are in cca_classifier.ipynb, clip_classifier.ipynb and mlp_classifier.ipynb.

EEG input: per-trial embeddings from four sources (schubki_cbramod, schubki_labram, schubki_luna, schubki_labram_special)

### 1. CCA classifier

Canonical correlation analysis between the EEG embeddings and the video features (CCA_Classifier in src/classifiers/cca.py). EEG trials and candidate videos are projected into the shared canonical space and scored by similarity there.

### 2. CLIP-style contrastive model

Two projection towers (EEG and video), each Linear(256) -> LayerNorm -> GELU -> Dropout -> ResidualBlock -> Linear(128), with L2-normalized outputs and a learnable temperature. It is trained with a symmetric InfoNCE loss with multiple positives (trials of the same video count as matches).

### 3. MLP regression to video features

An MLP (input -> 128 -> 64 -> 825, ReLU, dropout 0.3) that regresses the video feature vector from the EEG embedding with an MSE loss (full-batch, 20 epochs, AdamW lr 1e-3). Candidates are ranked by cosine similarity between the predicted and candidate video features.

### Results

Train accuracy is measured on the training trials against all 1000 training videos (chance top-1 = 0.001). Test accuracy is against the 102 test videos (chance top-1 = 0.0098, top-5 = 0.049, median rank about 51.5). All numbers are for the 4-source ensemble.

| Model | Subject | Train top-1 | Train top-5 | Test top-1 | Test top-5 | Test median rank |
|---|---|---|---|---|---|---|
| CCA | 01 | 1.0000 | 1.0000 | 0.0098 | 0.0784 | 50.0 |
| CCA | 02 | 0.9969 | 1.0000 | 0.0098 | 0.0686 | 57.0 |
| CCA | 03 | 1.0000 | 1.0000 | 0.0196 | 0.0588 | 48.5 |
| CCA | 04 | 1.0000 | 1.0000 | 0.0000 | 0.0686 | 51.5 |
| CLIP | 01 | 1.0000 | 1.0000 | 0.0294 | 0.0784 | 51.5 |
| CLIP | 02 | 1.0000 | 1.0000 | 0.0000 | 0.0490 | 52.0 |
| CLIP | 03 | 1.0000 | 1.0000 | 0.0098 | 0.0490 | 51.0 |
| CLIP | 04 | 1.0000 | 1.0000 | 0.0000 | 0.0294 | 46.0 |
| MLP | 01 | 0.0021 | 0.0062 | 0.0098 | 0.0588 | 50.0 |
| MLP | 02 | 0.0010 | 0.0051 | 0.0196 | 0.0490 | 52.0 |
| MLP | 03 | 0.0010 | 0.0083 | 0.0098 | 0.0392 | 51.5 |
| MLP | 04 | 0.0011 | 0.0095 | 0.0098 | 0.0588 | 54.5 |

- **CCA and CLIP** overfit heavily. They reach 100% train top-1 for every subject (individual CLIP sources already hit 97 to 99% for LaBraM), but on unseen videos they fall to chance: top-1 between 0.00 and 0.03 and median rank about 50, which is what random ranking of 102 candidates gives. With about 1000 training stimuli and 200 to 256 embedding dimensions mapped to an 825-dimensional video space, these models memorize the training pairs instead of learning a transferable EEG-to-video mapping.

- The **MLP** underfits instead. Its train accuracy is already at chance (about 0.002 top-1 on 1000 candidates), so it did not learn the mapping at all. With full-batch training for only 20 epochs, this is likely an optimization or capacity issue and was not tuned further. Its test results are at chance too.

- **Ridge** on raw EEG was the best. It is the only approach with clearly above-chance test retrieval (about 4 to 6% top-1 and 15 to 25% top-5), so it is our submission. Strong regularization (alpha 1e4 to 1e5) seems to matter for generalization here.

## References
[1] Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., & Schmidt, L. (2021). OpenCLIP (Version v0.1) [Computer software]. https://doi.org/10.5281/zenodo.5143773
