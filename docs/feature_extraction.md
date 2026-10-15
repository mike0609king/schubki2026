# Feature extraction

**Output.** Data is given in the format `(n_trials, feature)`, meaning we can work normally with the data like with any other machine learning setup

**Notes.**

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
