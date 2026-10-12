"""
Classifcal feature extraction for EEG. This does not include DL methods
for embedding. They are kept in separate files, because it makes this easier
to read.
"""

import numpy as np
from scipy.signal import welch


def _freq_mask(freqs: np.ndarray, fmin: float, fmax: float | None) -> np.ndarray:
    return (freqs >= fmin) & (freqs <= (np.inf if fmax is None else fmax))


# TODO We might want further extractions of features, such as averaging certain
# bands. Not sure whether 
def psd_welch(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
):
    """
    Power spectral density with Welch's method: the signal is cut into
    overlapping Hann-windowed segments and their periodograms are averaged.

    :param data: (n_trials, n_channels, n_times) EEG
    :param sfreq: Sampling frequency in Hz
    :param fmin, fmax: Frequency range that is kept; set to the same as the
    bandpass filter you used in preprocessing. Makes sure to remove dimensions
    that are not needed.
    :param window_sec: Window segment length in seconds
    :param overlap: Fraction of overlap between consecutive segments
    :param average: Try out "mean" or "median"; median is more robust to artefacts

    :returns: Returns (psd, freqs), psd of shape (n_trials, n_channels, n_freqs) and
    freqs of shape (n_freqs,)
    """
    if data.ndim != 3:
        raise ValueError(
            f"Expected (n_trials, n_channels, n_times), got shape {data.shape}."
        )

    nperseg = min(int(round(window_sec * sfreq)), data.shape[-1])
    freqs, psd = welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=int(nperseg * overlap),
        axis=-1,
    )
    mask = _freq_mask(freqs, fmin, fmax)
    return psd[:, :, mask], freqs[mask]
