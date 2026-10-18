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

## Other approaches

We also attempted other approaches ((TODO explain CCA and CLIP approach and offer short analysis))

## References
[1] Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., & Schmidt, L. (2021). OpenCLIP (Version v0.1) [Computer software]. https://doi.org/10.5281/zenodo.5143773
