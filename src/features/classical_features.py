"""
Classifcal feature extraction for EEG. This does not include DL methods
for embedding. They are kept in separate files, because it makes this easier
to read.
"""

import warnings

import numpy as np
from scipy.signal import welch
from mne.time_frequency import tfr_array_morlet
import pywt


def _freq_mask(freqs: np.ndarray, fmin: float, fmax: float | None) -> np.ndarray:
    return (freqs >= fmin) & (freqs <= (np.inf if fmax is None else fmax))


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
    psd, freqs = psd[:, :, mask], freqs[mask]
    np.maximum(psd, 1e-20, out=psd)
    np.log10(psd, out=psd)
    return psd, freqs


def cwt_morlet(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    freq_step: float = 1.0,
    n_cycles: float = 5.0,
    decim: int = 1,
    chunk_size: int = 256,
    dtype: type = np.float32,
):
    """
    Continuous wavelet transform power with Morlet wavelets.

    :param fmin, fmax: Frequency range
    :param freq_step: Spacing of the analysed frequencies in Hz
    :param n_cycles: Number of cycles per wavelet.
    :param decim: Keep only every decim-th time sample of the output to make
    it smaller.
    :param chunk_size: Number of trials transformed at once. MNE works in
    float64/complex128, so this bounds the extra memory on top of the output.
    :param dtype: dtype of the returned power.
    """
    if data.ndim != 3:
        raise ValueError(
            f"Expected (n_trials, n_channels, n_times), got shape {data.shape}."
        )
    if fmin <= 0:
        raise ValueError(f"fmin must be > 0 Hz for Morlet wavelets, got {fmin}.")

    freqs = np.arange(fmin, fmax + freq_step / 2, freq_step)

    half_len = (data.shape[-1] + 1) // 2
    max_cycles = 0.99 * 2 * np.pi * freqs * half_len / (5 * sfreq)
    cycles = np.minimum(n_cycles, max_cycles)
    if np.any(cycles < n_cycles):
        capped = freqs[cycles < n_cycles]
        warnings.warn(
            f"n_cycles reduced for {capped[0]:.2f}-{capped[-1]:.2f} Hz so the "
            f"wavelets fit into {data.shape[-1]} samples (down to "
            f"{cycles.min():.2f} cycles at {freqs[0]:.2f} Hz)."
        )

    times = np.arange(data.shape[-1])[::decim] / sfreq
    power = np.empty((*data.shape[:2], len(freqs), len(times)), dtype=dtype)
    for start in range(0, len(data), chunk_size):
        stop = start + chunk_size
        power[start:stop] = tfr_array_morlet(
            data[start:stop],
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=cycles,
            decim=decim,
            output="power",
            verbose=False,
        )

    power = np.maximum(power, 1e-20, out=power)
    np.log10(power, out=power)
    return power, freqs, times


def dwt(data, wavelet="db4", level=7, drop_finest=2):
    """
    Decomposes the EEG epochs using Discrete Wavelet Transform (DWT)
    and extracts the relative energy of each coefficient band.
    """
    coeffs = pywt.wavedec(data, wavelet, level=level, axis=-1)
    coeffs = coeffs[: len(coeffs) - drop_finest]

    # Energy in each band: (n_epochs, n_channels, n_bands)
    energies = np.stack([np.sum(np.square(c), axis=-1) for c in coeffs], axis=-1)
    total_energy = energies.sum(axis=-1, keepdims=True)
    total_energy[total_energy == 0] = 1

    # Relative energy profile per channel
    relative_energies = energies / total_energy
    relative_energies = relative_energies.reshape(len(data), -1)

    power = np.maximum(relative_energies, 1e-20, out=relative_energies)
    np.log10(relative_energies, out=relative_energies)
    return relative_energies
