# Schubki 2026

## Workflow
In the following we will describe our rough competition workflow
1. **Preprocessing and feature extraction.** Template implementation in `emd_dataset_prerpcessing_and_feature_extraction.ipynb`-file. We do preprocessing and extract
features, which we store in `h5` files.
2. **Feature selection.** Select features we want to test first. With visualization notebook.
3. **Classification.** 
    - Run classification algorithms with the extracted feature
    - Run CCA as baseline implementation for CLIP algorithm
    - We can start of by running the CLIP algorithm, while we wait for preprocessing
    and feature extraction
4. **Getting ready for handin.** If we finished training a model, we store their parametersand implement the inference script according to the submission guidelines. Starting at the end of the first day we need to have a tested version with parameters in hand to avoid not handing in anything in the last moment.

## Quick links
- [Preprocessing](./docs/preprocessing.md)
- [Feature extraction](./docs/feature_extraction.md)
- [Classification](./docs/classification.md)

## Questions to answer
- What are the most important channels, during classification?
- What are the most important frequencies during classification? We can try to find out via classical features or adjusting the bandpass filter.

## Structure
- `src`. Contains source code to quickly access implementations, which are shared accross notebooks.
- `notebooks`. Contains notebooks; make sure to use a single notebook for each pipeline. 
