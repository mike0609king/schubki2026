# Feature extraction

**Output.** We want a machine learning algorithm to work normally with the extracted data
- format `(n_trials, feature)`
- the data is standardized

**Notes.**

**Dimensionalilty reduction.** PCA (+LDA) is used for feature extraction
- Do not forget to standardize the data beforehand!

**Extraction algorithms.**
- *PSD.* Make sure to use the same frequency for `fmin` and `fmax` as for the bandpass filter. I expect this not to perform as well, because of the non-stationarity of EEGs. We might eed to reduce the dimension of this slightly and pair with another classical feature.
- *Wavelet transform.* Not implemented yet
- *HHT.* Not implemented yet
- *STFT.* Not implemented yet. We need to test around with the window size here. There is a tradeoff between short and long windows sizes.
- *BIOT.*
- *LUNA.*
- *LaBraM.*
- *CBraMod.*

**Mixing feature extraction algorithms.** 
- TODO define a few combinations; after those combinations, we can apply PCA (potentially with LDA).
